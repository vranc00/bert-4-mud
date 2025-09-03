!git clone https://github.com/vranc00/bert-4-mud.git

%cd bert-4-mud

!pip install transformers==4.55.0
!pip install datasets
!pip install evaluate
!pip install transformers datasets seqeval
!pip install seqeval transformers evaluate



#Esegui lo script train.py con i parametri desiderati
!python moduli/train.py \
--base_dir "data-evalita-2023" \
--epochs 3 \
--batch_size 16 \
--lr 2e-5 \
--num_runs 3 \
--output_dir "runs/bert_manual_opt"
