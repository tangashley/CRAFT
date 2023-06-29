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
# import lstm_models as lstm_models
# import util as util
# import evaluation as eval
from tensorflow.keras.models import Model
from numpy.random import seed
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


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

    train_data_df = train_df.loc[:, 'LIMIT_BAL':'PAY_AMT6']
    # train_data = train_data_df.apply(lambda x: ' '.join(x), axis=1)
    train_labels = train_df.pop('def_pay')

    val_data_df = val_df.loc[:, 'LIMIT_BAL':'PAY_AMT6']
    # val_data = val_data_df.apply(lambda x: ' '.join(x), axis=1)
    val_labels = val_df.pop('def_pay')

    test_data_df = test_df.loc[:, 'LIMIT_BAL':'PAY_AMT6']
    # test_data = test_data_df.apply(lambda x: ' '.join(x), axis=1)
    test_labels = test_df.pop('def_pay')

    # remove features with less attention
    attn_scores = np.load(attn_data_path)

    train_data_df = train_data_df.rename(columns={'SEX': 'GENDER',
                            'PAY_1': 'PAY_0','LIMIT_BAL': 'LIMIT BALANCE'})

    column_names = train_data_df.columns

    # for i in range(column_names):
    #     if column_names[i] == '':
    #         column_names[i] = ''
    #     if column_names[i] == 'PAY_1':
    #         column_names[i] = 'PAY_0'

    personal_features = ["SEX", "EDUCATION", "MARRIAGE", "AGE"]
    attn_avg = np.average(attn_scores, axis=0)
    column_attn = list(zip(column_names, attn_avg))
    column_attn.sort(key=lambda x: x[1])
    sorted_names = []
    print("**** Importance from low to high: ****")
    for feature_name, score in column_attn:
        sorted_names.append(feature_name)
        print(feature_name + ": " + str(score))

    # x1 = np.asarray([x for x, _ in column_attn])
    x2 = np.asarray([[x*100 for _, x in column_attn]])

    # fig, ax = plt.subplots()
    # im = ax.imshow(x2, cmap='autumn_r')
    #
    # # Show all ticks and label them with the respective list entries
    # ax.set_xticks(np.arange(len(x1)), labels=x1)
    # ax.set_yticks([])
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #          rotation_mode="anchor")
    #
    # # plt.imshow(x2, cmap='hot', interpolation='nearest',)
    # # fig.tight_layout()
    # # plt.savefig('heat_map.png', dpi=600)
    # plt.show()





    def heatmap(data, row_labels, col_labels, ax=None,
                cbar_kw={}, cbarlabel="", **kwargs):
        """
        Create a heatmap from a numpy array and two lists of labels.

        Parameters
        ----------
        data
            A 2D numpy array of shape (M, N).
        row_labels
            A list or array of length M with the labels for the rows.
        col_labels
            A list or array of length N with the labels for the columns.
        ax
            A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
            not provided, use current axes or create a new one.  Optional.
        cbar_kw
            A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
        cbarlabel
            The label for the colorbar.  Optional.
        **kwargs
            All other arguments are forwarded to `imshow`.
        """

        if not ax:
            ax = plt.gca()

        # Plot the heatmap
        im = ax.imshow(data, **kwargs)

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        # Show all ticks and label them with the respective list entries.
        ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
        ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False,
                       labeltop=True, labelbottom=False)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                 rotation_mode="anchor")

        # Turn spines off and create white grid.
        ax.spines[:].set_visible(False)

        ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
        ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        return im, cbar


    def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                         textcolors=("black", "white"),
                         threshold=None, **textkw):
        """
        A function to annotate a heatmap.

        Parameters
        ----------
        im
            The AxesImage to be labeled.
        data
            Data used to annotate.  If None, the image's data is used.  Optional.
        valfmt
            The format of the annotations inside the heatmap.  This should either
            use the string format method, e.g. "$ {x:.2f}", or be a
            `matplotlib.ticker.Formatter`.  Optional.
        textcolors
            A pair of colors.  The first is used for values below a threshold,
            the second for those above.  Optional.
        threshold
            Value in data units according to which the colors from textcolors are
            applied.  If None (the default) uses the middle of the colormap as
            separation.  Optional.
        **kwargs
            All other arguments are forwarded to each call to `text` used to create
            the text labels.
        """

        if not isinstance(data, (list, np.ndarray)):
            data = im.get_array()

        # Normalize the threshold to the images color range.
        if threshold is not None:
            threshold = im.norm(threshold)
        else:
            threshold = im.norm(data.max()) / 2.

        # Set default alignment to center, but allow it to be
        # overwritten by textkw.
        kw = dict(horizontalalignment="center",
                  verticalalignment="center")
        kw.update(textkw)

        # Get the formatter in case a string is supplied
        if isinstance(valfmt, str):
            valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)

        return texts


    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(20, 3))



    im, cbar = heatmap(x2, None, sorted_names, ax=ax,
                       cmap='autumn_r', cbarlabel=r"weight *1e-2")
    texts = annotate_heatmap(im, valfmt="{x:.1f}")

    fig.tight_layout()
    plt.savefig('../plot/UCI_heat_map2.png', dpi=600)
    plt.show()
    #
