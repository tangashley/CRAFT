"""
Program for
ATTENTION-BASED LSTM FOR PSYCHOCAL STRESS DETECTION FROM SPOKEN LANGUAGE USING DISTANT SUPERVISION
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
import sklearn
from sklearn.utils import resample
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
import lstm_models as lstm_models
import util as util
# import evaluation as eval
from tensorflow.keras.models import Model
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import confusion_matrix

np.random.seed(1)
tf.random.set_seed(1)


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

    plot_cm(test_labels, test_predictions)

    intermediate_layer_model2 = Model(inputs=test_model.input, outputs=test_model.layers[2].output)

    intermediate_layer_model1 = Model(inputs=test_model.input, outputs=test_model.layers[1].output)

    total_val = []
    for i in tqdm(range(test_data.shape[0])):
        arr = test_data[i]
        arr = numpy.reshape(arr, (1, arr.shape[0]))
        intermediate_output2 = intermediate_layer_model2.predict(arr, verbose=0)
        intermediate_output1 = intermediate_layer_model1.predict(arr, verbose=0)

        weights = intermediate_output2 / intermediate_output1
        val = []
        total = 0
        for j in range(test_data.shape[1]):
            val.append(weights[0][j][0])
            total += weights[0][j][0]
        # print(val)
        total_val.append(val)
    np.save(os.path.join(result_dir, 'attn_scores'), total_val)


def plot_metrics(history):
    metrics = ['loss', 'prc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_' + metric],
                 color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            # plt.ylim([0.8, 1])
            plt.ylim([0, 1])
        else:
            plt.ylim([0, 1])

        plt.legend()


def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

    plt.plot(100 * fp, 100 * tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    # plt.xlim([-0.5, 20])
    # plt.ylim([80, 100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')


def plot_prc(name, labels, predictions, **kwargs):
    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions)

    plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')


def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    # plt.figure(figsize=(5,5))
    # sns.heatmap(cm, annot=True, fmt="d")
    # plt.title('Confusion matrix @{:.2f}'.format(p))
    # plt.ylabel('Actual label')
    # plt.xlabel('Predicted label')

    print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
    print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
    print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
    print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
    print('Total Fraudulent Transactions: ', np.sum(cm[1]))


if __name__ == '__main__':

    # LIMIT TENSORFLOW MEMORY USAGE
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.7
    # tf.compat.v1.keras.backend.set_session(tf.Session(config=config))

    ##############################################
    # PARSE COMMAND OPTIONS
    ##############################################

    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("--model_type", dest="model_type", default="LSTM", help="default=LSTM, LSTM/SVM", metavar="FILE")
    parser.add_option("--is_attention", dest="is_attention", default=True, help="default=True, use attention mechanism",
                      metavar="FILE")
    parser.add_option("--is_finetune", dest="is_finetune", default=True, help="default=True, use fine tuning",
                      metavar="FILE")
    parser.add_option("--hidden_units", dest="hidden_units", default=64, help="default=64, number of hidden units",
                      metavar="FILE")
    parser.add_option("--epochs", dest="epochs", default=100, help="default=100, number of epochs", metavar="FILE")
    parser.add_option("--num_layers", dest="num_layers", default=1,
                      help="default=1, number of layers (only for LSTM/BLSTM models)", metavar="FILE")
    parser.add_option("--is_bidirectional", dest="is_bidirectional", default=True, help="default=True, use BLSTM",
                      metavar="FILE")
    parser.add_option("--input_data_path", dest="input_data_path", default="../data", help="input datapath",
                      metavar="FILE")
    parser.add_option("--attn_data_path", dest="attn_data_path", default="../data", help="attn datapath",
                      metavar="FILE")
    parser.add_option("--max_sequence_length", dest="max_sequence_length", default=35, help="max sequence length",
                      metavar="FILE")
    parser.add_option("--word_embedding_path", dest="word_embedding_path", default=None,
                      help="word embedding in numpy array format path", metavar="FILE")
    parser.add_option("--vocab_path", dest="vocab_path", default=None, help="vocab in numpy array path", metavar="FILE")
    parser.add_option("--skip_preprocess", dest="skip_preprocess", default=True,
                      help="load preprocess files, skip preprocessing part", metavar="FILE")
    parser.add_option("--validation_split", dest="validation_split", default=0.2, help="validation split",
                      metavar="FILE")
    parser.add_option("--save_path", dest="save_path", default="../saved_ckpt", help="save checkpoint path",
                      metavar="FILE")
    parser.add_option("--result_dir", dest="result_dir", default="../results/", help="save result path", metavar="FILE")

    (options, args) = parser.parse_args()
    print("options", options)
    print("args", args)

    model_type = options.model_type
    is_attention = options.is_attention == "True"
    is_finetune = options.is_finetune == "True"
    hidden_units = int(options.hidden_units)
    epochs = int(options.epochs)
    num_layers = int(options.num_layers)
    is_bidirectional = options.is_bidirectional == "True"
    input_data_path = options.input_data_path
    attn_data_path = options.attn_data_path
    max_sequence_length = int(options.max_sequence_length)
    skip_preprocess = options.skip_preprocess == "True"
    validation_split = float(options.validation_split)
    save_path = options.save_path
    result_dir = options.result_dir

    word_embedding = None
    word_embedding_path = None
    vocab_path = None
    is_custom_embedding = False

    if not options.word_embedding_path == None:
        word_embedding_path = options.word_embedding_path

        if not options.vocab_path == None:
            vocab_path = options.vocab_path
            is_custom_embedding = True

    print("\n###################################")
    print("			  SUMMARY		  ")
    print("###################################")
    print("model_type\t\t:", model_type)
    print("is_attention\t\t:", is_attention)
    print("is_finetune\t\t:", is_finetune)
    print("hidden_units\t\t:", hidden_units)
    print("num_layers\t\t:", num_layers)
    print("is_bidirectional\t:", is_bidirectional)
    print("input_data_path\t\t:", input_data_path)
    print("is_custom_embedding\t:", is_custom_embedding)
    print("word_embedding_path\t:", word_embedding_path)
    print("vocab_path\t\t:", vocab_path)
    print("skip_preprocess\t\t:", skip_preprocess)
    print("validation_split\t:", validation_split)
    print("epochs\t:", epochs)
    print("save_path\t:", save_path)
    print("result_dir\t:", result_dir)
    print("")
    print("###################################\n")

    mpl.rcParams['figure.figsize'] = (12, 10)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    ##############################################
    # READ DATA
    ##############################################

    ##############################################
    # PREPROCESS DATA
    ##############################################

    train_df = pd.read_csv(input_data_path + '/train_data.csv')
    val_df = pd.read_csv(input_data_path + '/validation_data.csv')
    test_df = pd.read_csv(input_data_path + '/test_data.csv')

    train_labels = train_df.pop('Label')
    # train_data = train_data_df.apply(lambda x: ' '.join(x), axis=1)

    val_labels = val_df.pop('Label')
    # val_data = val_data_df.apply(lambda x: ' '.join(x), axis=1)

    test_labels = test_df.pop('Label')
    # test_data = test_data_df.apply(lambda x: ' '.join(x), axis=1)

    # remove features with less attention
    attn_scores = np.load(attn_data_path)
    column_names = train_df.columns
    attn_avg = np.average(attn_scores, axis=0)
    column_attn = list(zip(column_names, attn_avg))
    column_attn.sort(key=lambda x: x[1], reverse=True)
    print("**** Importance from high to low: ****")
    for feature_name, score in column_attn:
        print(feature_name + ": " + str(score))

    sender_col = train_df.filter(regex='^Sender', axis=1).columns

    for col in sender_col:
        train_df.pop(col)
        val_df.pop(col)
        test_df.pop(col)

    # upsampling the minority class

    not_fraud = train_df[train_df.Label == 0]
    fraud = train_df[train_df.Label == 1]

    # upsample minority
    fraud_upsampled = resample(fraud,
                               replace=True,  # sample with replacement
                               n_samples=len(not_fraud),  # match number in majority class
                               random_state=27)  # reproducible results

    # combine majority and upsampled minority
    upsampled = pd.concat([not_fraud, fraud_upsampled])
    # upsampled.pop('Transaction_Id')
    upsampled_train_label = np.array(upsampled.pop('Label'))
    upsampled_train_data = upsampled.apply(lambda x: ' '.join(x), axis=1)

    train_labels = train_df.pop('Label')
    # train_df.pop('Transaction_Id')
    train_data = train_df.apply(lambda x: ' '.join(x), axis=1)

    val_labels = np.array(val_df.pop('Label'))
    # val_df.pop('Transaction_Id')
    val_data = val_df.apply(lambda x: ' '.join(x), axis=1)

    test_labels = np.array(test_df.pop('Label'))
    # test_df.pop('Transaction_Id')
    test_data = test_df.apply(lambda x: ' '.join(x), axis=1)

    print(">>> generate vocabulary from all data")

    # word_to_idx, idx_to_word, vocab = util.generate_vocab(train_data.values)
    #
    # train_data, _ = util.generate_word_index(train_data, None, word_to_idx,
    #                                          idx_to_word, vocab, max_sequence_length)
    # val_data, _ = util.generate_word_index(val_data, None, word_to_idx,
    #                                        idx_to_word, vocab, max_sequence_length)
    # test_data, _ = util.generate_word_index(test_data, None, word_to_idx,
    #                                         idx_to_word, vocab, max_sequence_length)

    word_to_idx, idx_to_word, vocab = util.generate_vocab(train_df.values)

    upsampled_train_data, _ = util.generate_word_index(upsampled_train_data.values, None, word_to_idx,
                                                       idx_to_word, vocab, max_sequence_length)
    train_data, _ = util.generate_word_index(train_data, None, word_to_idx,
                                             idx_to_word, vocab, max_sequence_length)
    val_data, _ = util.generate_word_index(val_data, None, word_to_idx,
                                           idx_to_word, vocab, max_sequence_length)
    test_data, _ = util.generate_word_index(test_data, None, word_to_idx,
                                            idx_to_word, vocab, max_sequence_length)

    batch_size = 2048

    num_initialization_epochs = 2
    num_epochs = epochs

    # gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    # for device in gpu_devices:
    #     tf.config.experimental.set_memory_growth(device, True)

    print("> Train LSTM with pretrained word embedding matrix")
    model = lstm_models.lstm(vocab, hidden_units, num_layers, max_sequence_length, is_attention, is_bidirectional)

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
    baseline_history = model.fit(upsampled_train_data, upsampled_train_label, validation_data=(val_data, val_labels),
                                 batch_size=batch_size, epochs=num_epochs, shuffle=True, verbose=1,
                                 callbacks=[cp_callback, early_stop_callback])

    print('Training finished ', str(datetime.datetime.now()))

    train_predictions_baseline = model.predict(train_data, batch_size=batch_size)
    test_predictions_baseline = model.predict(test_data, batch_size=batch_size)

    plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])
    plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')
    plt.legend(loc='lower right')
    plt.show()

    plot_prc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])
    plot_prc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')
    plt.legend(loc='lower right')
    plt.show()

    plot_metrics(baseline_history)
    plt.show()

    # TESTING
    print('\n>>> Testing...')
    test(model, test_data, test_labels, result_dir, False)
