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
import argparse
import csv
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

# from IPython.display import SVG
import numpy as np
import pandas as pd
# from scipy import spatial
# TRAINING MODULES
# from keras.optimizers import Adam
# from keras.utils import plot_model
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# import matplotlib.pyplot as plt
# from keras.backend.tensorflow_backend import set_session

import lstm_attn.lstm_models as lstm_models
import lstm_attn.util as util

# import evaluation as eval




def test(test_model, test_data, y_test, result_dir):
    print("Testing started!")
    test_predictions = test_model.predict(test_data, verbose=0)
    y_pred = [1 if pred > 0.5 else 0 for pred in test_predictions]

    print("Classification Report for LSTM-attn: \n", classification_report(y_test, y_pred))
    print("Confusion Matrix of LSTM-attn: \n", confusion_matrix(y_test, y_pred))

    res_accu = metrics.accuracy_score(y_test, y_pred)
    roc_auc = metrics.roc_auc_score(y_test, test_predictions)
    res_f1 = metrics.f1_score(y_test, y_pred, average='weighted')
    res_recall = metrics.recall_score(y_test, y_pred, average='weighted')
    res_precision = metrics.precision_score(y_test, y_pred, average='weighted')

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
    # total_val = []
    #
    # sample_data = sample(range(test_data.shape[0]), 1000)
    #
    # for i in tqdm(sample_data):
    #     arr = test_data[i]
    #     arr = numpy.reshape(arr, (1, arr.shape[0]))
    #     intermediate_output2 = intermediate_layer_model2.predict(arr, verbose=0)
    #     intermediate_output1 = intermediate_layer_model1.predict(arr, verbose=0)
    #
    #     weights = intermediate_output2 / intermediate_output1
    #     val = []
    #     total = 0
    #     for j in range(test_data.shape[1]):
    #         val.append(weights[0][j][0])
    #         total += weights[0][j][0]
    #     # print(val)
    #     total_val.append(val)
    #
    #
    #
    # np.save(os.path.join(result_dir, 'attn_scores'), total_val)


# def plot_metrics(history):
#     metrics = ['loss', 'prc', 'precision', 'recall']
#     for n, metric in enumerate(metrics):
#         name = metric.replace("_", " ").capitalize()
#         plt.subplot(2, 2, n + 1)
#         plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
#         plt.plot(history.epoch, history.history['val_' + metric],
#                  color=colors[0], linestyle="--", label='Val')
#         plt.xlabel('Epoch')
#         plt.ylabel(name)
#         if metric == 'loss':
#             plt.ylim([0, plt.ylim()[1]])
#         elif metric == 'auc':
#             # plt.ylim([0.8, 1])
#             plt.ylim([0, 1])
#         else:
#             plt.ylim([0, 1])
#
#         plt.legend()
#
#
# def plot_roc(name, labels, predictions, **kwargs):
#     fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)
#
#     plt.plot(100 * fp, 100 * tp, label=name, linewidth=2, **kwargs)
#     plt.xlabel('False positives [%]')
#     plt.ylabel('True positives [%]')
#     # plt.xlim([-0.5, 20])
#     # plt.ylim([80, 100.5])
#     plt.grid(True)
#     ax = plt.gca()
#     ax.set_aspect('equal')
#
#
# def plot_prc(name, labels, predictions, **kwargs):
#     precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions)
#
#     plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.grid(True)
#     ax = plt.gca()
#     ax.set_aspect('equal')
#
#
# def plot_cm(labels, predictions, p=0.5):
#     cm = confusion_matrix(labels, predictions > p)
#     # plt.figure(figsize=(5,5))
#     # sns.heatmap(cm, annot=True, fmt="d")
#     # plt.title('Confusion matrix @{:.2f}'.format(p))
#     # plt.ylabel('Actual label')
#     # plt.xlabel('Predicted label')
#
#     print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
#     print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
#     print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
#     print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
#     print('Total Fraudulent Transactions: ', np.sum(cm[1]))


if __name__ == '__main__':

    # LIMIT TENSORFLOW MEMORY USAGE
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.7
    # tf.compat.v1.keras.backend.set_session(tf.Session(config=config))

    ##############################################
    # PARSE COMMAND OPTIONS
    ##############################################

    parser = argparse.ArgumentParser()
    # parser.add_argument("--input_data_path", type=str, default="../data", help="input datapath")
    parser.add_argument('--num_feature_to_remove', type=int, default=None,
                        help='number of unimportant features to remove')
    parser.add_argument('--idx_feature_to_remove', type=int, default=None,
                        help='according to the importance order, remove the i-th less important feature, '
                             'where i is the idx_feature_to_remove')
    parser.add_argument("--hidden_units", type=int, default=100, help="default=100, number of hidden units")
    parser.add_argument("--epochs", type=int, default=100, help="default=100, number of epochs")
    parser.add_argument("--num_layers", type=int, default=1, help="default=1, number of layers (only for LSTM/BLSTM models)")
    parser.add_argument("--attn_data_path", type=str, default="../data", help="attn datapath")
    parser.add_argument("--save_path", type=str, default="../saved_ckpt", help="save checkpoint path")
    parser.add_argument("--result_dir", type=str, default="../results/", help="save result path")
    parser.add_argument("--seed", type=int, default=1)

    config = parser.parse_args()
    print("config", config)

    # input_data_path = config.input_data_path
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
    # print("input_data_path\t\t:", input_data_path)
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

    df = pd.read_csv('data/kaggle_fraud_detection_age_gender/bs140513_032310.csv')

    ##############################################
    # PREPROCESS DATA
    ##############################################

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

    # create bins for money values
    data_reduced['amount'] = data_reduced['amount'].apply(lambda x: round(x / 10))
    # add prefix to number values
    data_reduced['amount'] = 'USD_' + data_reduced['amount'].astype(str)
    data_reduced['step'] = 'step_' + data_reduced['step'].astype(str)
    data_reduced['age'] = 'age_' + data_reduced['age'].astype(str)

    # load attention scores and sort the scores
    attn_scores = np.load(attn_data_path)
    column_names = data_reduced.columns
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
            data_reduced = data_reduced.drop(columns=[feature_name])
            print("Remove " + str(feature_name))
    elif idx_feature_to_remove is not None:
        feature_name = column_attn[idx_feature_to_remove][0]
        data_reduced = data_reduced.drop(columns=[feature_name])
        print("Remove the " + str(idx_feature_to_remove) + "-th feature: " + str(feature_name))


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

    # df['step'] = df['step'].astype('category')
    # df['age'] = df['age'].astype('category')
    # df['amount'] = df['amount'].astype('category')

    max_sequence_length = len(X_train.columns)

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
    model = lstm_models.lstm(vocab, hidden_units, num_layers, max_sequence_length)

    # print(model.summary())

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
