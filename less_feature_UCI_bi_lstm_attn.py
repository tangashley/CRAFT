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
import argparse
import csv
import random
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
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Reshape, Flatten, LSTM, Dense, Dropout, Embedding, Bidirectional
# from keras.optimizers import Adam
from keras.models import load_model
# from keras.utils import plot_model
import tensorflow as tf
# from IPython.display import SVG
import numpy
import warnings
from sklearn import metrics
from copy import deepcopy
# import matplotlib.pyplot as plt
# from keras.backend.tensorflow_backend import set_session
import lstm_attn.lstm_models as lstm_models
import lstm_attn.util as util
# import evaluation as eval
from tensorflow.keras.models import Model
from numpy.random import seed
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

seed(1)
tf.random.set_seed(1)


def get_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)

    print('True Negatives: ', cm[0][0])
    print('False Positives: ', cm[0][1])
    print('False Negatives: ', cm[1][0])
    print('True Positives: ', cm[1][1])
    print('Total Positives: ', np.sum(cm[1]))


def test(test_model, test_data, test_labels, result_dir, show_mistake=False):
    print("Testing started!")
    test_predictions = test_model.predict(test_data, verbose=0)
    if not Path(result_dir).exists():
        os.makedirs(result_dir)
        print('Create directory: ' + result_dir)
    filepath1 = os.path.join(result_dir, 'test_predictions')
    filepath2 = os.path.join(result_dir, 'test_labels')
    np.save(filepath1, test_predictions)
    np.save(filepath2, test_labels)
    print("SAVED!")

    # x = np.argmax(test_labels, axis=1)
    # y = np.argmax(test_predictions, axis=1)

    x = test_labels
    y = [1 if pred > 0.5 else 0 for pred in test_predictions]

    print("Classification Report for LSTM-attn: \n", classification_report(test_labels, y))
    print("Confusion Matrix of LSTM-attn: \n", confusion_matrix(test_labels, y))

    res_accu = metrics.accuracy_score(x, y)
    roc_auc = metrics.roc_auc_score(test_labels, test_predictions)
    res_f1 = metrics.f1_score(x, y, average='weighted')
    res_recall = metrics.recall_score(x, y, average='weighted')
    res_precision = metrics.precision_score(x, y, average='weighted')

    print('Test Accuracy: %.3f' % res_accu)
    print('Test ROC AUC %.3f' % roc_auc)
    print("Weighted!")
    print('Test F1-score: %.3f' % res_f1)
    print('Test Recall: %.3f' % res_recall)
    print('Test Precision: %.3f' % res_precision)

    header = ["Acc", "ROC AUC", "Weighted F1", "Weighted Recall", "Weighted Precision"]

    if not Path(result_dir).exists():
        print("Create folder: " + result_dir)
        os.makedirs(result_dir)

    # do not write header again if the file already exists
    if Path(result_dir + '/metrics.csv').exists():
        with open(os.path.join(result_dir, 'metrics.csv'), 'a') as f:
            w = csv.writer(f)
            w.writerow([res_accu, roc_auc, res_f1, res_recall, res_precision])
    else:
        with open(os.path.join(result_dir, 'metrics.csv'), 'w') as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerow([res_accu, roc_auc, res_f1, res_recall, res_precision])

    # intermediate_layer_model2 = Model(inputs=test_model.input, outputs=test_model.layers[2].output)
    #
    # intermediate_layer_model1 = Model(inputs=test_model.input, outputs=test_model.layers[1].output)
    #
    # # for i, ax in enumerate(axn.flat):
    # # line2,line3,line7,line33
    # # sentences = ["15195 b731304416ba107c83262ce2e875c4df B 32 1 1 1 1 0 31 1 220 0 0 0 0 0 0 31 31 0 0 2 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0","46358 da87cbc1b5b8501acf3b49eec1cc52c3 W 31 1 0 1 1 0 30 2 250 0 0 0 454 0 0 30 30 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0","38403 77b830096c1888016b4d7a730bbe9731 B 32 1 4 0 2 0 31 1 169 0 0 0 186 0 0 31 31 0 0 3 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0","26257 16f88bcb1b253282c0414e4539984174 W 39 1 0 0 1 0 38 3 32 0 0 0 608.8 39385.7 198.458307964167 38 38 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"]
    # total_val = []
    # for i in tqdm(range(test_data.shape[0])):
    #     # for i in range(20):
    #     # seq = sentences[i]
    #     # words = seq.split(" ")
    #     # print(words)
    #     # arr = numpy.zeros(len(words))
    #     # for j in range(len(words)):
    #     #	if words[j] in word_to_idx:
    #     #		arr[j] = word_to_idx[words[j].lower()]
    #     #	#else:
    #     #	#	arr[j] = word_to_idx[""]
    #     arr = test_data[i]
    #     arr = numpy.reshape(arr, (1, arr.shape[0]))
    #     intermediate_output2 = intermediate_layer_model2.predict(arr, verbose=0)
    #     intermediate_output1 = intermediate_layer_model1.predict(arr, verbose=0)
    #     # print(arr, test_model.predict(arr))
    #
    #     weights = intermediate_output2 / intermediate_output1
    #     val = []
    #     total = 0
    #     for j in range(test_data.shape[1]):
    #         val.append(weights[0][j][0])
    #         total += weights[0][j][0]
    #     # print(val)
    #     total_val.append(val)
    # np.save(os.path.join(result_dir, 'attn_scores'), total_val)


if __name__ == '__main__':

    # LIMIT TENSORFLOW MEMORY USAGE
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.7
    # tf.compat.v1.keras.backend.set_session(tf.Session(config=config))

    ##############################################
    # PARSE COMMAND OPTIONS
    ##############################################

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_path", type=str, default="../data", help="input datapath")
    parser.add_argument('--num_feature_to_remove', type=int, default=None,
                        help='number of unimportant features to remove')
    parser.add_argument('--idx_feature_to_remove', type=int, default=None,
                        help='according to the importance order, remove the i-th less important feature, '
                             'where i is the idx_feature_to_remove')
    parser.add_argument("--hidden_units", type=int, default=100, help="default=100, number of hidden units")
    parser.add_argument("--epochs", type=int, default=100, help="default=100, number of epochs")
    parser.add_argument("--num_layers", type=int, default=1,
                        help="default=1, number of layers (only for LSTM/BLSTM models)")
    parser.add_argument("--attn_data_path", type=str, default="../data", help="attn datapath")
    parser.add_argument("--save_path", type=str, default="../saved_ckpt", help="save checkpoint path")
    parser.add_argument("--result_dir", type=str, default="../results/", help="save result path")
    parser.add_argument("--seed", type=int, default=1)

    config = parser.parse_args()
    print("config", config)

    input_data_path = config.input_data_path
    num_feature_to_remove = config.num_feature_to_remove
    idx_feature_to_remove = config.idx_feature_to_remove
    hidden_units = int(config.hidden_units)
    epochs = int(config.epochs)
    num_layers = int(config.num_layers)
    attn_data_path = config.attn_data_path
    save_path = config.save_path
    result_dir = config.result_dir
    seed = config.seed

    print("\n###################################")
    print("			  SUMMARY		  ")
    print("###################################")
    print("input_data_path\t\t:", input_data_path)
    print("num_feature_to_remove\t\t:", num_feature_to_remove)
    print("idx_feature_to_remove\t\t:", idx_feature_to_remove)
    print("hidden_units\t\t:", hidden_units)
    print("num_layers\t\t:", num_layers)
    print("epochs\t:", epochs)
    print("save_path\t:", save_path)
    print("result_dir\t:", result_dir)
    print("seed\t:", seed)
    print("")
    print("###################################\n")

    # set seed for randomness
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

    ##############################################
    # READ DATA
    ##############################################


    train_df = pd.read_csv(input_data_path + '/train_data.csv', dtype=str)
    val_df = pd.read_csv(input_data_path + '/validation_data.csv', dtype=str)
    test_df = pd.read_csv(input_data_path + '/test_data.csv', dtype=str)

    ##############################################
    # PREPROCESS DATA
    ##############################################

    train_data_df = train_df.loc[:, 'LIMIT_BAL':'PAY_AMT6']
    # train_data = train_data_df.apply(lambda x: ' '.join(x), axis=1)
    train_labels = train_df.pop('def_pay').astype(int)

    val_data_df = val_df.loc[:, 'LIMIT_BAL':'PAY_AMT6']
    # val_data = val_data_df.apply(lambda x: ' '.join(x), axis=1)
    val_labels = val_df.pop('def_pay').astype(int)

    test_data_df = test_df.loc[:, 'LIMIT_BAL':'PAY_AMT6']
    # test_data = test_data_df.apply(lambda x: ' '.join(x), axis=1)
    test_labels = test_df.pop('def_pay').astype(int)

    num_default = train_labels.loc[train_labels == 1]
    num_non_default = train_labels.loc[train_labels == 0]
    pos_weight = len(num_non_default) / len(num_default)
    pos_weight = np.sqrt(pos_weight)

    # load attention scores and sort the scores
    attn_scores = np.load(attn_data_path)
    column_names = train_data_df.columns
    attn_avg = np.average(attn_scores, axis=0)
    column_attn = list(zip(column_names, attn_avg))
    column_attn.sort(key=lambda x: x[1])

    print("**** Importance from low to high: ****")
    for feature_name, score in column_attn:
        print(feature_name + ": " + str(score))

    # remove features with less attention
    print("**** Remove features ****")
    if num_feature_to_remove is not None:
        for i in range(num_feature_to_remove):
            feature_name = column_attn[i][0]
            train_data_df = train_data_df.drop(columns=[feature_name])
            val_data_df = val_data_df.drop(columns=[feature_name])
            test_data_df = test_data_df.drop(columns=[feature_name])
            print("Remove " + str(feature_name))
    elif idx_feature_to_remove is not None:
        feature_name = column_attn[idx_feature_to_remove][0]
        train_data_df = train_data_df.drop(columns=[feature_name])
        val_data_df = val_data_df.drop(columns=[feature_name])
        test_data_df = test_data_df.drop(columns=[feature_name])
        print("Remove the " + str(idx_feature_to_remove) + "-th feature: " + str(feature_name))

    max_sequence_length = len(train_data_df.columns)

    train_data = train_data_df.apply(lambda x: ' '.join(x), axis=1)
    val_data = val_data_df.apply(lambda x: ' '.join(x), axis=1)
    test_data = test_data_df.apply(lambda x: ' '.join(x), axis=1)


    word_to_idx, idx_to_word, vocab = util.generate_vocab(train_data.values)

    train_data, _ = util.generate_word_index(train_data, None, word_to_idx,
                                             idx_to_word, vocab, max_sequence_length)
    val_data, _ = util.generate_word_index(val_data, None, word_to_idx,
                                           idx_to_word, vocab, max_sequence_length)
    test_data, _ = util.generate_word_index(test_data, None, word_to_idx,
                                            idx_to_word, vocab, max_sequence_length)

    ##############################################
    # TRAIN AND TEST MODEL
    ##############################################

    batch_size = 2048

    num_initialization_epochs = 2
    num_epochs = epochs

    print("> Train LSTM with trainable word embedding matrix")
    model = lstm_models.lstm(vocab, hidden_units, num_layers, max_sequence_length)

    checkpoint_dir = os.path.dirname(save_path)
    checkpoint_path = save_path

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     monitor="val_auc",
                                                     verbose=1,
                                                     save_best_only=True,
                                                     mode="max")
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_auc',
                                                           patience=10,
                                                           mode="max",
                                                           restore_best_weights=True)
    print('\n>>> Starting training...')
    train_labels = train_labels.to_frame()
    val_labels = val_labels.to_frame()

    class_weight = {0: 1.0, 1: pos_weight}
    print(class_weight)

    history = model.fit(train_data, train_labels, validation_data=(val_data, val_labels),
                        batch_size=batch_size, epochs=num_epochs, shuffle=True, verbose=1,
                        callbacks=[cp_callback, early_stop_callback],
                        class_weight=class_weight)

    print('Training finished ', str(datetime.datetime.now()))

    model = tf.keras.models.load_model(checkpoint_path)

    # TESTING
    print('\n>>> Testing...')
    test(model, test_data, test_labels, result_dir, False)
