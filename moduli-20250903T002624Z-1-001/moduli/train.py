import os, csv, math, statistics, argparse
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, get_scheduler, set_seed
import evaluate

from classe import NERDataset
from funzioni import concat_domains, tokenize_and_align, train_one_epoch, evaluate_model, compute_metrics

os.environ["TOKENIZERS_PARALLELISM"] = "false"

#Configurazione parametri linea di comando 

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning BERT NER ottimizzato")
    parser.add_argument("--base_dir", type=str, default="/content/drive/MyDrive/Progetto/KIND-main/evalita-2023")
    parser.add_argument("--model_name", type=str, default="dbmdz/bert-base-italian-cased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="runs/bert_manual_opt")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    CSV_PATH = os.path.join(args.output_dir, "results.csv")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    #Caricamento dati

    train_sents, train_labels = concat_domains("train", base_dir=args.base_dir)
    dev_sents, dev_labels     = concat_domains("dev", base_dir=args.base_dir)
    test_sents, test_labels   = concat_domains("test", base_dir=args.base_dir)

    label_list = sorted(set(l for seq in (train_labels + dev_labels + test_labels) for l in seq))
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}
    label_list_metrics = sorted({l for seq in (train_labels + dev_labels + test_labels) for l in seq if l != "O"})

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    metric = evaluate.load("seqeval")

    train_tok = tokenize_and_align(train_sents, train_labels, tokenizer, label2id)
    dev_tok   = tokenize_and_align(dev_sents, dev_labels, tokenizer, label2id)
    test_tok  = tokenize_and_align(test_sents, test_labels, tokenizer, label2id)

    train_dataset = NERDataset(train_tok)
    dev_dataset   = NERDataset(dev_tok)
    test_dataset  = NERDataset(test_tok)

    num_workers = min(2, os.cpu_count())
    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=data_collator, num_workers=num_workers, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=data_collator, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=data_collator, num_workers=num_workers, pin_memory=True)

    TYPES = ["LOC", "PER", "ORG"]
    all_results = defaultdict(list)

    #Ciclo delle run

    for run_id in range(1, args.num_runs + 1):
        print(f"\n=== RUN {run_id}/{args.num_runs} ===")
        set_seed(42 * run_id)

        model = AutoModelForTokenClassification.from_pretrained(
            args.model_name, num_labels=len(label_list),
            id2label=id2label, label2id=label2id
        ).to(DEVICE)

        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        num_training_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
        scheduler = get_scheduler("linear", optimizer=optimizer,
                                  num_warmup_steps=int(0.1*num_training_steps),
                                  num_training_steps=num_training_steps)
        scaler = torch.amp.GradScaler(enabled=USE_AMP)

        best_f1, best_state, bad_epochs = -1.0, None, 0

        for epoch in range(1, args.epochs+1):
            train_one_epoch(model, train_loader, optimizer, scheduler, scaler,
                            grad_accum=args.grad_accum, device=DEVICE, device_type=DEVICE.type,
                            use_amp=USE_AMP, run_id=run_id, epoch=epoch, epochs=args.epochs)

            val_metrics, avg_val_loss = evaluate_model(
                model, dev_loader, compute_metrics, id2label,
                label_list=label_list_metrics, metric=metric, device=DEVICE
            )
            val_f1 = val_metrics.get("f1_macro_by_type", 0.0) #f1_micro
            print(f"[Run {run_id}] Epoch {epoch}: val_loss={val_metrics.get('val_loss', avg_val_loss):.4f} | "
                  f"f1_micro={val_f1:.4f} | f1_macro_by_type={val_metrics.get('f1_macro_by_type',0.0):.4f}")
            for t in TYPES:
                print(f"  {t} F1: {val_metrics.get(f'f1_{t}',0.0):.4f}")

            #Early stopping

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    print("Early stopping.")
                    break

        
        #Salvataggio del modello migliore
  
        if best_state is not None:
            save_path = os.path.join(args.output_dir, f"run{run_id}_best.pt")
            torch.save({
                "state_dict": best_state,
                "label2id": label2id,
                "id2label": id2label,
                "model_name": args.model_name,
                "seed": 42*run_id,
                "run_id": run_id
            }, save_path)
            print(f"Best model for run {run_id} saved to: {save_path}")

        if best_state is None:
            continue

        
        #Test finale
  
        model.load_state_dict({k:v.to(DEVICE) for k,v in best_state.items()})
        test_metrics, avg_test_loss = evaluate_model(
            model, test_loader, compute_metrics, id2label,
            label_list=label_list_metrics, metric=metric, device=DEVICE
        )
        print(f"[Run {run_id}] TEST test_loss={test_metrics.get('val_loss', avg_test_loss):.4f} | "
              f"f1_micro={test_metrics.get('f1_micro',0.0):.4f} | "
              f"f1_macro_by_type={test_metrics.get('f1_macro_by_type',0.0):.4f}")
        for t in TYPES:
            print(f"  {t} F1: {test_metrics.get(f'f1_{t}',0.0):.4f}")

        for k,v in test_metrics.items():
            all_results[k].append(v)

    
    #Aggregazione risultati per il csv
    
    print("\n=== Risultati medi sui run ===")
    metriche_chiave = [
        "val_loss",
        "precision_micro", "recall_micro", "f1_micro",
        "precision_macro", "recall_macro", "f1_macro_by_type",
        "precision_weighted", "recall_weighted", "f1_weighted",
        "support_PER", "support_LOC", "support_ORG",
        "precision_PER", "recall_PER", "f1_PER",
        "precision_LOC", "recall_LOC", "f1_LOC",
        "precision_ORG", "recall_ORG", "f1_ORG"
    ]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["metric"] + [f"run_{i}" for i in range(1, args.num_runs+1)] + ["mean","std"]
        writer.writerow(header)
        for m in metriche_chiave:
            vals = all_results.get(m, [])
            vals = [float(v) if v is not None else 0.0 for v in vals]
            if any(x in m.lower() for x in ["precision","recall","f1"]):
                vals = [v*100 for v in vals]
            mean_val = statistics.mean(vals) if vals else 0.0
            std_val  = statistics.stdev(vals) if len(vals)>1 else 0.0
            row = [m] + [f"{v:.2f}" for v in vals] + [f"{mean_val:.2f}", f"{std_val:.2f}"]
            writer.writerow(row)

    print(f"\nRisultati salvati in percentuale in: {CSV_PATH}")

if __name__=="__main__":
    main()
