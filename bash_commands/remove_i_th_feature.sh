#!/bin/sh


#for idx_feature_to_remove in 0 1 2 3 4 5 6; do
#    echo "************************************************************************************************************"
#    echo "****************************** REMOVE $idx_feature_to_remove-th FEATURE ************************************"
#    echo "************************************************************************************************************"
#    for seed in 1 2 3; do
#      python less_feature_fraud_detect_age_gender_bi_lstm_attn.py \
#      --idx_feature_to_remove=$idx_feature_to_remove \
#      --attn_data_path=results/Fraud/fraud_age_gender_non_smote_weighted_class_correct_attn/attn_scores.npy \
#      --save_path=checkpoint/Fraud/greedy/remove_idx_${idx_feature_to_remove}_seed_${seed} \
#      --result_dir=results/Fraud/greedy/remove_idx_${idx_feature_to_remove} \
#      --seed=$seed
#    done
#done
	
for num_feature_to_remove in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22; do
    echo "************************************************************************************************************"
    echo "****************************** REMOVE $num_feature_to_remove FEATURE ************************************"
    echo "************************************************************************************************************"
    for seed in 1 2 3; do
      python less_feature_UCI_bi_lstm_attn.py \
      --num_feature_to_remove=$num_feature_to_remove \
      --input_data_path=data/UCI/feature_prefix_bin_stride_2000 \
      --attn_data_path=results/UCI/old/feature_prefix_bin_stride_20000_new/attn_scores.npy \
      --hidden_units=64 \
      --num_layers=1 \
      --save_path=checkpoint/UCI/greedy/${num_feature_to_remove}_less_features_seed_${seed} \
      --epochs=100 \
      --result_dir=results/UCI/greedy/${num_feature_to_remove}_less_features \
      --seed=${seed}
    done
done