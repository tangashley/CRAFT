"""
Program for
ATTENTION-BASED LSTM FOR PSYCHOLOGICAL STRESS DETECTION FROM SPOKEN LANGUAGE USING DISTANT SUPERVISION
IEEE ICASSP 2018 (Genta Indra Winata, Onno Pepijn Kampman, Pascale Fung)

To run the model
# e.g. CUDA_VISIBLE_DEVICES=0 python main.py --input_data_path="../../dataset" --model_type=LSTM --is_attention=True --is_finetune=True --hidden_units=64 --num_layers=1 --is_bidirectional=True --max_sequence_length=35 --validation_split=0.2
with custom embedding
# e.g. CUDA_VISIBLE_DEVICES=0 python main.py --input_data_path="../../dataset" --model_type=LSTM --is_attention=True --is_finetune=False --hidden_units=64 --num_layers=1 --is_bidirectional=True --max_sequence_length=35 --word_embedding_path="../embedding/emotion_embedding_size_50/emotion_embedding.pkl" --vocab_path="../embedding/emotion_embedding_size_50/vocab.pkl" --skip_preprocess=False --validation_split=0.2

Tips:
1. Skip the preprocess, if you are using the same embedding vectors
--skip-preprocess=True
"""

##############################################
# LOAD LIBRARIES
##############################################
# PYTHON LIBS
import sys
# import matplotlib
# import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np
import pickle
import datetime
from copy import deepcopy
# from scipy import spatial
from tqdm import tqdm
import os

# TRAINING MODULES
import tensorflow as tf
import numpy
from sklearn import metrics
import lstm_models as lstm_models
import util as util
# import evaluation as eval
# from tensorflow.keras.models import Model
from numpy.random import seed
import pandas as pd

seed(1)
tf.random.set_seed(1)


if __name__ == '__main__':

    ##############################################
    # PARSE COMMAND OPTIONS
    ##############################################

    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("--input_data_path", dest="input_data_path", default="../data/JP/JP_feature_prefix_bin_stride_1",
                      help="input datapath",
                      metavar="FILE")
    parser.add_option("--max_sequence_length", dest="max_sequence_length", default=12, help="max sequence length",
                      metavar="FILE")

    (options, args) = parser.parse_args()
    print("options", options)
    print("args", args)

    input_data_path = options.input_data_path
    max_sequence_length = int(options.max_sequence_length)

    ##############################################
    # READ DATA
    ##############################################

    ##############################################
    # PREPROCESS DATA
    ##############################################
    all_data = []
    train_df = pd.read_csv(input_data_path + '/train_data.csv', dtype=str)
    val_df = pd.read_csv(input_data_path + '/validation_data.csv', dtype=str)
    test_df = pd.read_csv(input_data_path + '/test_data.csv', dtype=str)

    # train_data_df = train_df.loc[:, 'LIMIT_BAL':'PAY_AMT6']
    # train_data = train_data_df.apply(lambda x: ' '.join(x), axis=1)
    # train_labels = train_df.pop('def_pay')
    #
    # val_data_df = val_df.loc[:, 'LIMIT_BAL':'PAY_AMT6']
    # val_data = val_data_df.apply(lambda x: ' '.join(x), axis=1)
    # val_labels = val_df.pop('def_pay')
    #
    # test_data_df = test_df.loc[:, 'LIMIT_BAL':'PAY_AMT6']
    # test_data = test_data_df.apply(lambda x: ' '.join(x), axis=1)
    # test_labels = test_df.pop('def_pay')

    # train_df.fillna('N', inplace=True)
    # val_df.fillna('N', inplace=True)
    # test_df.fillna('N', inplace=True)

    train_labels = train_df.pop('Label')
    train_data = train_df.apply(lambda x: ' '.join(x), axis=1)

    val_labels = val_df.pop('Label')
    val_data = val_df.apply(lambda x: ' '.join(x), axis=1)

    test_labels = test_df.pop('Label')
    test_data = test_df.apply(lambda x: ' '.join(x), axis=1)



    print(">>> generate vocabulary from all data")

    word_to_idx, idx_to_word, vocab = util.generate_vocab(train_data.values)

    train_data, train_labels = util.generate_word_index(train_data, train_labels, word_to_idx,
                                                        idx_to_word, vocab, max_sequence_length)
    val_data, val_labels = util.generate_word_index(val_data, val_labels, word_to_idx,
                                                    idx_to_word, vocab, max_sequence_length)
    test_data, test_labels = util.generate_word_index(test_data, test_labels, word_to_idx,
                                                      idx_to_word, vocab, max_sequence_length)

    val_sum = 0
    val_word_sum = 0
    for sample in val_data:
        for i in range(len(sample)):
            val_word_sum += 1
            # not include unseen transaction id, transaction ids are unique
            if sample[i] == 0 and not i == 1:
                val_sum += 1
    print("Validation vocabulary has " + str(val_sum / val_word_sum * 100) + "% unknown words")


    test_sum = 0
    test_word_sum = 0
    for sample in test_data:
        for i in range(len(sample)):
            test_word_sum += 1
            # not include unseen transaction id, transaction ids are unique
            if sample[i] == 0 and not i == 1:
                test_sum += 1
    print("Test vocabulary has " + str(test_sum / test_word_sum * 100) + "% unknown words")

