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
import argparse
import datetime
# from scipy import spatial
import os
import pickle
##############################################
# LOAD LIBRARIES
##############################################
# PYTHON LIBS
import random
# import matplotlib
# import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np
import pandas as pd
# TRAINING MODULES
# from keras.optimizers import Adam
# from keras.utils import plot_model
import tensorflow as tf
# import evaluation as eval
from numpy.random import seed
# from IPython.display import SVG
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report

# import matplotlib.pyplot as plt
# from keras.backend.tensorflow_backend import set_session
import lstm_models as lstm_models
import util as util

np.random.seed(1)
tf.random.set_seed(1)
random.seed(1)


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

    res_accu = metrics.accuracy_score(x, y)
    roc_auc = metrics.roc_auc_score(test_labels, test_predictions)
    res_precision = metrics.precision_score(x, y, average='weighted')
    res_recall = metrics.recall_score(x, y, average='weighted')
    res_f1 = metrics.f1_score(x, y, average='weighted')

    print('Test Accuracy: %.3f' % res_accu)
    print('Test ROC AUC %.3f' % roc_auc)
    print("Weighted!")
    print('Test F1-score: %.3f' % res_f1)
    print('Test Recall: %.3f' % res_recall)
    print('Test Precision: %.3f' % res_precision)

    print("Classification Report for LSTM-attn: \n", classification_report(x, y))
    print("Confusion Matrix of LSTM-attn: \n", confusion_matrix(x, y))

    # get_cm(test_labels, test_predictions)

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
    # parser.add_argument('--num_feature_to_remove', default=0, type=int, help='number of unimportant features to remove')
    parser.add_argument("--hidden_units", type=int, default=64, help="default=64, number of hidden units")
    parser.add_argument("--epochs", type=int, default=100, help="default=100, number of epochs")
    parser.add_argument("--num_layers", type=int, default=1,
                        help="default=1, number of layers (only for LSTM/BLSTM models)")
    # parser.add_argument("--is_bidirectional", default=True, help="default=True, use BLSTM")
    parser.add_argument("--attn_data_path", type=str, default="../data", help="attn datapath")
    # parser.add_argument("--max_sequence_length", type=int, default=35, help="max sequence length")
    parser.add_argument("--save_path", type=str, default="../saved_ckpt", help="save checkpoint path")
    parser.add_argument("--result_dir", type=str, default="../results/", help="save result path")

    config = parser.parse_args()
    print("config", config)

    input_data_path = config.input_data_path
    # num_feature_to_remove = config.num_feature_to_remove
    hidden_units = int(config.hidden_units)
    epochs = int(config.epochs)
    num_layers = int(config.num_layers)
    # is_bidirectional = config.is_bidirectional
    attn_data_path = config.attn_data_path
    # max_sequence_length = int(config.max_sequence_length)
    save_path = config.save_path
    result_dir = config.result_dir

    print("\n###################################")
    print("			  SUMMARY		  ")
    print("###################################")
    print("input_data_path\t\t:", input_data_path)
    # print("num_feature_to_remove\t\t:", num_feature_to_remove)
    print("hidden_units\t\t:", hidden_units)
    print("num_layers\t\t:", num_layers)
    # print("is_bidirectional\t:", is_bidirectional)
    print("input_data_path\t\t:", input_data_path)
    print("epochs\t:", epochs)
    print("save_path\t:", save_path)
    print("result_dir\t:", result_dir)
    print("")
    print("###################################\n")

    ##############################################
    # READ DATA
    ##############################################

    ##############################################
    # PREPROCESS DATA
    ##############################################
    train_df = pd.read_csv(input_data_path + '/train_data.csv', dtype=str)
    val_df = pd.read_csv(input_data_path + '/validation_data.csv', dtype=str)
    test_df = pd.read_csv(input_data_path + '/test_data.csv', dtype=str)

    train_default = train_df.loc[train_df.def_pay == '1']
    train_non_default = train_df.loc[train_df.def_pay == '0']
    pos_weight = len(train_non_default) / len(train_default)
    pos_weight = np.sqrt(pos_weight)

    train_data_df = train_df.loc[:, 'LIMIT_BAL':'PAY_AMT6']
    # train_data = train_data_df.apply(lambda x: ' '.join(x), axis=1)
    train_labels = train_df.pop('def_pay').astype(float)

    val_data_df = val_df.loc[:, 'LIMIT_BAL':'PAY_AMT6']
    # val_data = val_data_df.apply(lambda x: ' '.join(x), axis=1)
    val_labels = val_df.pop('def_pay').astype(float)

    test_data_df = test_df.loc[:, 'LIMIT_BAL':'PAY_AMT6']
    # test_data = test_data_df.apply(lambda x: ' '.join(x), axis=1)
    test_labels = test_df.pop('def_pay').astype(float)

    # # remove features with less attention
    # attn_scores = np.load(attn_data_path)
    # column_names = train_data_df.columns
    # attn_avg = np.average(attn_scores, axis=0)
    # column_attn = list(zip(column_names, attn_avg))
    # column_attn.sort(key=lambda x: x[1], reverse=True)
    # print("**** Importance from high to low: ****")
    # for feature_name, score in column_attn:
    #     print(feature_name + ": " + str(score))

    print("number of features before removing is: " + str(len(train_data_df.columns)))

    print("**** Remove personal features ****")
    personal_features = ["SEX", "EDUCATION", "MARRIAGE", "AGE"]
    for feature_name in personal_features:
        train_data_df = train_data_df.drop(columns=[feature_name])
        val_data_df = val_data_df.drop(columns=[feature_name])
        test_data_df = test_data_df.drop(columns=[feature_name])
        print("Remove " + str(feature_name))

    print("number of features after removing is: " + str(len(train_data_df.columns)))

    train_data = train_data_df.apply(lambda x: ' '.join(x), axis=1)
    val_data = val_data_df.apply(lambda x: ' '.join(x), axis=1)
    test_data = test_data_df.apply(lambda x: ' '.join(x), axis=1)

    print(">>> generate vocabulary from all data")

    word_to_idx, idx_to_word, vocab = util.generate_vocab(train_data.values)

    max_sequence_length = len(train_data_df.columns)
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

    print("Default class weight: " + str(pos_weight))

    print('\n>>> Starting training...')
    class_weight = {0: 1.0, 1: pos_weight}
    history = model.fit(train_data, train_labels, validation_data=(val_data, val_labels),
                        batch_size=batch_size, epochs=num_epochs, shuffle=True, verbose=1,
                        callbacks=[cp_callback, early_stop_callback],
                        class_weight=class_weight)

    print('Training finished ', str(datetime.datetime.now()))

    model = tf.keras.models.load_model(checkpoint_path)

    # TESTING
    print('\n>>> Testing...')
    test(model, test_data, test_labels, result_dir, False)
