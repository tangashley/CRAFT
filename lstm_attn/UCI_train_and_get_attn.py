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

import datetime
import os
import pickle
##############################################
# LOAD LIBRARIES
##############################################
# PYTHON LIBS
# import matplotlib
# import matplotlib.pyplot as plt
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
# from IPython.display import SVG
import numpy
import numpy as np
import pandas as pd
# TRAINING MODULES
# from keras.optimizers import Adam
# from keras.utils import plot_model
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import confusion_matrix
# import evaluation as eval
from tensorflow.keras.models import Model
# from scipy import spatial
from tqdm import tqdm

# import matplotlib.pyplot as plt
# from keras.backend.tensorflow_backend import set_session
import lstm_models as lstm_models
import util as util

numpy.random.seed(2)
tf.random.set_seed(2)


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

    get_cm(test_labels, test_predictions)

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
    mpl.rcParams['figure.figsize'] = (12, 10)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
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

def get_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)

    print('True Negatives: ', cm[0][0])
    print('False Positives: ', cm[0][1])
    print('False Negatives: ', cm[1][0])
    print('True Positives: ', cm[1][1])
    print('Total Positives: ', np.sum(cm[1]))

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

    ##############################################
    # READ DATA
    ##############################################

    ##############################################
    # PREPROCESS DATA
    ##############################################
    train_df = pd.read_csv(input_data_path + '/train_data.csv', dtype=str)
    val_df = pd.read_csv(input_data_path + '/validation_data.csv', dtype=str)
    test_df = pd.read_csv(input_data_path + '/test_data.csv', dtype=str)

    # # upsampling the minority class
    # not_def_pay = train_df[train_df.def_pay == '0']
    # def_pay = train_df[train_df.def_pay == '1']
    # # upsample minority
    # def_pay_upsampled = resample(def_pay,
    #                              replace=True,  # sample with replacement
    #                              n_samples=len(not_def_pay),  # match number in majority class
    #                              random_state=27)  # reproducible results
    # # combine majority and upsampled minority
    # upsampled = pd.concat([not_def_pay, def_pay_upsampled])
    # # upsampled.pop('Transaction_Id')
    # upsampled_train_label = np.array(upsampled.pop('def_pay')).astype(float)
    # upsampled_train_df = upsampled.loc[:, 'LIMIT_BAL':'PAY_AMT6']
    # upsampled_train_data = upsampled_train_df.apply(lambda x: ' '.join(x), axis=1)

    train_data_df = train_df.loc[:, 'LIMIT_BAL':'PAY_AMT6']
    train_data = train_data_df.apply(lambda x: ' '.join(x), axis=1)
    train_labels = train_df.pop('def_pay').astype(float)

    val_data_df = val_df.loc[:, 'LIMIT_BAL':'PAY_AMT6']
    val_data = val_data_df.apply(lambda x: ' '.join(x), axis=1)
    val_labels = val_df.pop('def_pay').astype(float)

    test_data_df = test_df.loc[:, 'LIMIT_BAL':'PAY_AMT6']
    test_data = test_data_df.apply(lambda x: ' '.join(x), axis=1)
    test_labels = test_df.pop('def_pay').astype(float)

    print(">>> generate vocabulary")
    if is_custom_embedding:
        print(">>> load vocab and embedding pickle data")
        vocab = list(pickle.load(open(vocab_path, "rb"), encoding='latin1').values())
        word_embedding = pickle.load(open(word_embedding_path, "rb"), encoding='latin1')

        word_to_idx, idx_to_word, vocab, word_embedding = util.generate_vocab_with_custom_embedding(vocab,
                                                                                                    word_embedding)
    else:
        print(">>> generate vocabulary from all data")

        word_to_idx, idx_to_word, vocab = util.generate_vocab(train_data_df.values)

    train_data, _ = util.generate_word_index(train_data, None, word_to_idx,
                                             idx_to_word, vocab, max_sequence_length)
    val_data, _ = util.generate_word_index(val_data, None, word_to_idx,
                                           idx_to_word, vocab, max_sequence_length)
    test_data, _ = util.generate_word_index(test_data, None, word_to_idx,
                                            idx_to_word, vocab, max_sequence_length)

    if is_custom_embedding:
        np.save(input_data_path + "/preprocessed/word_embedding", word_embedding)

    batch_size = 2048

    num_initialization_epochs = 2
    num_epochs = epochs

    print("> Train LSTM with trainable word embedding matrix")
    model = lstm_models.lstm(vocab, hidden_units, num_layers, max_sequence_length, is_attention, is_bidirectional)

    checkpoint_dir = os.path.dirname(save_path)
    checkpoint_path = save_path

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     monitor="val_recall",
                                                     verbose=1,
                                                     save_best_only=True,
                                                     mode="max")
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_recall',
                                                           patience=10,
                                                           mode="max",
                                                           restore_best_weights=True)
    print('\n>>> Starting training...')
    history = model.fit(train_data, train_labels, validation_data=(val_data, val_labels),
                        batch_size=batch_size, epochs=num_epochs, shuffle=True, verbose=1,
                        callbacks=[cp_callback, early_stop_callback])

    print('Training finished ', str(datetime.datetime.now()))

    plot_metrics(history)
    plt.show()

    # TESTING
    print('\n>>> Testing...')
    test(model, test_data, test_labels, result_dir, False)
