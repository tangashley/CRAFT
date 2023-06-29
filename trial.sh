#!/usr/bin/env bash

for MONITOR in  'accuracy' 'f1_score' ; do
  for LR in 3e-3 1e-3 7e-4 5e-4 3e-4 1e-4 7e-5 5e-5 3e-5; do
    python tabtransformer_with_fraud_age_gender.py --lr $LR --monitor $MONITOR | tee -a tab_trans_attn_log_file.txt
  done
done