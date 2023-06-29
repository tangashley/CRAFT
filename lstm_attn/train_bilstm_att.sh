#!/bin/sh

GPU=0

BASE_PATH=

DATA=data/all_nums/
CPKT_DIR=checkpoint/all_nums/
LOG_DIR=log/all_nums/
RESULT_DIR=results/all_nums/

mkdir -p $CPKT_DIR $LOG_DIR $RESULT_DIR

CUDA_VISIBLE_DEVICES=$GPU python get_attn.py --input_data_path=$DATA --model_type=LSTM --is_attention=True --is_finetune=False --hidden_units=64 --num_layers=1 --is_bidirectional=True --max_sequence_length=100 --validation_split=0.2 --save_path=$CPKT_DIR --epochs 100 --result_dir=$RESULT_DIR | tee $LOG_DIR/train.out

#for TRG in level_any level_1 level_2 level_3
# do
#	DATA=$BASE_PATH/data/set3/$TRG/
#	CPKT_DIR=$BASE_PATH/checkpoint/set3/bilstm/${TRG}_tmp/
#	LOG_DIR=$BASE_PATH/log/set3/bilstm/${TRG}_tmp/
#	RESULT_DIR=$BASE_PATH/results/set3/bilstm/${TRG}_tmp/

#	mkdir -p $DATA_BIN $CPKT_DIR $LOG_DIR $RESULT_DIR

#	CUDA_VISIBLE_DEVICES=$GPU python get_attn.py --input_data_path=$DATA --model_type=LSTM --is_attention=True --is_finetune=False --hidden_units=64 --num_layers=1 --is_bidirectional=True --max_sequence_length=100 --validation_split=0.2 --save_path=$CPKT_DIR --epochs 100 --result_dir=$RESULT_DIR | tee $LOG_DIR/train.out
#done

