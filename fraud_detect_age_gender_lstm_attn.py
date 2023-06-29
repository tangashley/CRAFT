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

import datetime
import os
##############################################
# LOAD LIBRARIES
##############################################
# PYTHON LIBS
# import matplotlib
# import matplotlib.pyplot as plt
import random
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
# from IPython.display import SVG
import numpy
import numpy as np
import pandas as pd
# from scipy import spatial
import sklearn
# TRAINING MODULES
# from keras.optimizers import Adam
# from keras.utils import plot_model
import tensorflow as tf
from imblearn.over_sampling import SMOTENC
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
# import evaluation as eval
from tensorflow.keras.models import Model
from tqdm import tqdm

# import matplotlib.pyplot as plt
# from keras.backend.tensorflow_backend import set_session
import basic_lstm
import lstm_models as lstm_models
import util as util
from random import sample

np.random.seed(1)
tf.random.set_seed(1)
random.seed(1)


def test(test_model, test_data, y_test, result_dir):
    print("Testing started!")
    test_predictions = test_model.predict(test_data, verbose=0)
    y_pred = [1 if pred > 0.5 else 0 for pred in test_predictions]

    print("Classification Report for LSTM-attn: \n", classification_report(y_test, y_pred))
    print("Confusion Matrix of LSTM-attn: \n", confusion_matrix(y_test, y_pred))

    intermediate_layer_model2 = Model(inputs=test_model.input, outputs=test_model.layers[2].output)

    intermediate_layer_model1 = Model(inputs=test_model.input, outputs=test_model.layers[1].output)

    total_val = []

    sample_data = sample(range(test_data.shape[0]), 1000)

    for i in tqdm(sample_data):
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

    if not Path(result_dir).exists():
        print("Create folder: " + result_dir)
        os.makedirs(result_dir)

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
    parser.add_option("--result_dir", dest="result_dir", default="results/", help="save result path", metavar="FILE")

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

    df = pd.read_csv('data/kaggle_fraud_detection_age_gender/bs140513_032310.csv')

    df_fraud = df.loc[df.fraud == 1]
    df_non_fraud = df.loc[df.fraud == 0]
    pos_weight = len(df_non_fraud) / len(df_fraud)
    pos_weight = np.sqrt(pos_weight)

    # dropping zipcodeori and zipMerchant since they have only one unique value
    data_reduced = df.drop(['zipcodeOri', 'zipMerchant'], axis=1)

    col_categorical = data_reduced.select_dtypes(include=['object']).columns
    # col_num = data_reduced.select_dtypes(exclude=['object']).columns

    data_reduced[col_categorical] = data_reduced[col_categorical].applymap(lambda x: x.replace("'", ""))

    y = df[['fraud']]
    data_reduced = data_reduced.drop(['fraud'], axis=1)

    X_tmp, X_test, y_tmp, y_test = train_test_split(data_reduced, y, random_state=1, shuffle=True, test_size=0.2,
                                                    stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, random_state=1, shuffle=True, test_size=0.2,
                                                      stratify=y_tmp)

    # # SMOTE for balancing training dataset
    # idx = []
    # for v in col_categorical.values:
    #     idx.append(X_train.columns.get_loc(v))
    # sm = SMOTENC(categorical_features=idx, random_state=42)
    # X_train, y_train = sm.fit_resample(X_train, y_train)
    # y_train = pd.DataFrame(y_train)

    # create bins for money values
    X_train['amount'] = X_train['amount'].apply(lambda x: round(x / 10))
    # add prefix to number values
    X_train['amount'] = 'USD_' + X_train['amount'].astype(str)
    X_train['step'] = 'step_' + X_train['step'].astype(str)
    X_train['age'] = 'age_' + X_train['age'].astype(str)

    X_val['amount'] = X_val['amount'].apply(lambda x: round(x / 10))
    # add prefix to number values
    X_val['amount'] = 'USD_' + X_val['amount'].astype(str)
    X_val['step'] = 'step_' + X_val['step'].astype(str)
    X_val['age'] = 'age_' + X_val['age'].astype(str)

    X_test['amount'] = X_test['amount'].apply(lambda x: round(x / 10))
    # add prefix to number values
    X_test['amount'] = 'USD_' + X_test['amount'].astype(str)
    X_test['step'] = 'step_' + X_test['step'].astype(str)
    X_test['age'] = 'age_' + X_test['age'].astype(str)

    # df['step'] = df['step'].astype('category')
    # df['age'] = df['age'].astype('category')
    # df['amount'] = df['amount'].astype('category')
    #

    train_data = X_train.apply(lambda x: ' '.join(x), axis=1)

    val_data = X_val.apply(lambda x: ' '.join(x), axis=1)

    test_data = X_test.apply(lambda x: ' '.join(x), axis=1)

    print(">>> generate vocabulary from all data")

    # word_to_idx, idx_to_word, vocab = util.generate_vocab(train_data.values)
    #
    # train_data, _ = util.generate_word_index(train_data, None, word_to_idx,
    #                                          idx_to_word, vocab, max_sequence_length)
    # val_data, _ = util.generate_word_index(val_data, None, word_to_idx,
    #                                        idx_to_word, vocab, max_sequence_length)
    # test_data, _ = util.generate_word_index(test_data, None, word_to_idx,
    #                                         idx_to_word, vocab, max_sequence_length)

    word_to_idx, idx_to_word, vocab = util.generate_vocab(train_data.values)

    train_data, _ = util.generate_word_index(train_data.values, None, word_to_idx,
                                             idx_to_word, vocab, max_sequence_length)
    val_data, _ = util.generate_word_index(val_data.values, None, word_to_idx,
                                           idx_to_word, vocab, max_sequence_length)
    test_data, _ = util.generate_word_index(test_data.values, None, word_to_idx,
                                            idx_to_word, vocab, max_sequence_length)

    batch_size = 2048

    num_initialization_epochs = 2
    num_epochs = epochs

    print("> Train LSTM with pretrained word embedding matrix")
    model = lstm_models.lstm(vocab, hidden_units, num_layers, max_sequence_length, is_attention)

    print(model.summary())

    checkpoint_dir = os.path.dirname(save_path)
    checkpoint_path = save_path

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     monitor="val_loss",
                                                     verbose=1,
                                                     save_best_only=True,
                                                     mode="min")
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                           patience=10,
                                                           mode="min",
                                                           restore_best_weights=True)
    print('\n>>> Starting training...')
    class_weight = {0: 1.0, 1: pos_weight}

    baseline_history = model.fit(train_data, y_train, validation_data=(val_data, y_val),
                                 batch_size=batch_size, epochs=num_epochs, shuffle=True, verbose=1,
                                 callbacks=[cp_callback, early_stop_callback],
                                 class_weight=class_weight)

    # model = tf.keras.models.load_model(checkpoint_path)

    print('Training finished ', str(datetime.datetime.now()))

    # train_predictions_baseline = model.predict(train_data, batch_size=batch_size)
    # test_predictions_baseline = model.predict(test_data, batch_size=batch_size)

    # TESTING
    print('\n>>> Testing...')
    test(model, test_data, y_test, result_dir)
