#!/bin/sh


for num_feature_to_remove in 0 1 2 3 4 5 6; do
    echo "****************************** REMOVE $num_feature_to_remove FEATURES ************************************"
    for seed in 1 2 3; do
      python less_feature_fraud_detect_age_gender_bi_lstm_attn.py \
      --num_feature_to_remove=$num_feature_to_remove \
      --attn_data_path=results/Fraud/fraud_age_gender_non_smote_weighted_class_correct_attn/attn_scores.npy \
      --save_path=checkpoint/Fraud/greedy/2_${num_feature_to_remove}_less_seed_${seed} \
      --result_dir=results/Fraud/greedy/2_${num_feature_to_remove}_less \
      --seed=$seed
    done
done
	
