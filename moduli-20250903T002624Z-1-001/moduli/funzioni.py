import os
import torch
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
from collections import defaultdict
from copy import deepcopy

#Funzione per leggere file CoNLL/TSV

def read_conll(path):
    sents, labels = [], []                    
    cur_toks, cur_labs = [], []               
    with open(path, encoding="utf-8") as f:
        for line in f:                        
            line = line.strip()               
            if not line:                      
                if cur_toks:                  
                    sents.append(cur_toks)    
                    labels.append(cur_labs)
                    cur_toks, cur_labs = [], [] 
            else:
                tok, lab = line.split()[:2]     
                cur_toks.append(tok)
                cur_labs.append(lab)
    if cur_toks:                                 
        sents.append(cur_toks)
        labels.append(cur_labs)
    return sents, labels

#==============================================================================

#Funzione per la concatenazione dei domini

def concat_domains(split, domains=["WN", "FIC", "ADG"], base_dir=''):
    all_sents, all_labels = [], []                            
    for dom in domains:                                       
        path = os.path.join(base_dir, f"{dom}_{split}.tsv")   
        s, l = read_conll(path)                               
        all_sents.extend(s)                                   
        all_labels.extend(l)
    return all_sents, all_labels

#==============================================================================

#Funzione per la tokenizzazione e l'allineamento

def tokenize_and_align(sentences, labels, tokenizer, label2id, max_length=256):
    encodings = tokenizer(
        sentences,
        is_split_into_words=True,                         
        truncation=True,                                  
        padding=False,                                     
        max_length=max_length,                            
        return_tensors=None                               
    )

    all_lab_ids = []                                      
    for i, labs in enumerate(labels):
        word_ids = encodings.word_ids(i)                  
        lab_ids = []
        prev_word_idx = None                              
        for word_idx in word_ids:                         
            if word_idx is None:                          
                lab_ids.append(-100)
            elif word_idx != prev_word_idx:               
                lab_ids.append(label2id[labs[word_idx]])  
            else:                                         
                lab_ids.append(-100)
            prev_word_idx = word_idx                      
        all_lab_ids.append(lab_ids)                       

    encodings["labels"] = all_lab_ids                     
    return encodings                                      

#===============================================================================

#Calcolo delle metriche

def compute_metrics(preds, labels, id2label, label_list=None, metric=None, val_loss=None):
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_preds  = [[id2label[p] for (p,l) in zip(pred, label) if l != -100] for pred, label in zip(preds, labels)]

    if metric is not None:
        seqeval_res = metric.compute(predictions=true_preds, references=true_labels)
        overall_precision = seqeval_res.get("overall_precision", 0.0)
        overall_recall    = seqeval_res.get("overall_recall", 0.0)
        overall_f1        = seqeval_res.get("overall_f1", 0.0)
    else:
        overall_precision = overall_recall = overall_f1 = 0.0

    TYPES = ["PER", "LOC", "ORG"]
    precision_type = {}
    recall_type = {}
    f1_type = {}
    support_type = {}

    for t in TYPES:
        cls_labels = [[l if l.endswith(t) else "O" for l in seq] for seq in true_labels]
        cls_preds  = [[p if p.endswith(t) else "O" for p in seq] for seq in true_preds]
        if metric is not None:
            res = metric.compute(predictions=cls_preds, references=cls_labels)
            precision_type[t] = res.get("overall_precision", 0.0)
            recall_type[t] = res.get("overall_recall", 0.0)
            f1_type[t] = res.get("overall_f1", 0.0)
        else:
            precision_type[t] = recall_type[t] = f1_type[t] = 0.0
        support_type[t] = sum([sum(1 for l in seq if l.endswith(t)) for seq in cls_labels])

    precision_macro = np.mean(list(precision_type.values()))
    recall_macro    = np.mean(list(recall_type.values()))
    f1_macro_by_type = np.mean(list(f1_type.values()))
    total_support = sum(support_type.values())
    precision_weighted = sum(precision_type[t]*support_type[t] for t in TYPES)/total_support if total_support>0 else 0.0
    recall_weighted    = sum(recall_type[t]*support_type[t] for t in TYPES)/total_support if total_support>0 else 0.0
    f1_weighted        = sum(f1_type[t]*support_type[t] for t in TYPES)/total_support if total_support>0 else 0.0

  
    if label_list is None:
        label_list = sorted(set([l for seq in true_labels for l in seq if l != "O"]))

    f1_per_class = {}
    for cls in label_list:
        cls_labels = [[l if l == cls else "O" for l in seq] for seq in true_labels]
        cls_preds  = [[p if p == cls else "O" for p in seq] for seq in true_preds]
        if metric is not None:
            f1_per_class[cls] = metric.compute(predictions=cls_preds, references=cls_labels).get("overall_f1", 0.0)
        else:
            f1_per_class[cls] = 0.0

    f1_macro_total = np.mean(list(f1_per_class.values())) if f1_per_class else 0.0

    results = {
        "precision_micro": overall_precision,
        "recall_micro": overall_recall,
        "f1_micro": overall_f1,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro_by_type": f1_macro_by_type,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "f1_macro_total": f1_macro_total,
        "f1_per_class": deepcopy(f1_per_class),
    }

    for t in TYPES:
        results.update({
            f"precision_{t}": precision_type[t],
            f"recall_{t}": recall_type[t],
            f"f1_{t}": f1_type[t],
            f"support_{t}": support_type[t]
        })

    if val_loss is not None:
        results["val_loss"] = val_loss

    return results

#===============================================================================

#Invia al device

def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}

#===============================================================================

#Training di un'epoca

def train_one_epoch(model, loader, optimizer, scheduler, scaler, grad_accum,
                    device, device_type, use_amp=True, run_id=1, epoch=1,
                    epochs=1):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loop = tqdm(enumerate(loader), total=len(loader), ncols=100,
                desc=f"[Run {run_id}] Epoch {epoch}/{epochs}")
    for step, batch in loop:
        batch = to_device(batch, device)
        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            outputs = model(**batch)
            loss = outputs.loss / grad_accum

        scaler.scale(loss).backward()

        is_update_step = ((step + 1) % grad_accum == 0) or ((step + 1) == len(loader))
        if is_update_step:
          scaler.unscale_(optimizer)
          torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
          scaler.step(optimizer)
          scaler.update()
          optimizer.zero_grad(set_to_none=True)
          scheduler.step()

        loop.set_postfix(loss=loss.item() * grad_accum)

#==============================================================================

#Valutazione

def evaluate_model(model, dataloader, compute_metrics, id2label,
                   label_list=None, metric=None, device="cuda"):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
            labels = batch["labels"].detach().cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    avg_loss = total_loss / len(dataloader)

    preds_metrics = compute_metrics(
        preds=all_preds,
        labels=all_labels,
        id2label=id2label,
        label_list=label_list,
        metric=metric,
        val_loss=avg_loss
    )

    return preds_metrics, avg_loss
