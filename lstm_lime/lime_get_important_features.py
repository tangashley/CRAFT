import argparse
import os
import re
from pathlib import Path

import tensorflow as tf
from lime.lime_text import LimeTextExplainer
import numpy as np
from tqdm import tqdm
import time
import csv
from lstm_attn import util, lstm_models
import pandas as pd


# def predict_label(sent):
#     inputs = self.tokenizer.encode(sent, return_tensors='pt').to('cuda')
#     with torch.no_grad():
#         outputs = self.model(inputs)
#     logits = outputs.logits
#     logits = torch.softmax(logits, dim=1)
#     pred = torch.argmax(logits, dim=1)
#     return pred


if __name__ == '__main__':

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
    checkpoint_path = save_path
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

    train_df = pd.read_csv(input_data_path + '/train_data.csv', dtype=str)
    val_df = pd.read_csv(input_data_path + '/validation_data.csv', dtype=str)
    test_df = pd.read_csv(input_data_path + '/test_data.csv', dtype=str)

    train_data_df = train_df.loc[:, 'LIMIT_BAL':'PAY_AMT6']
    train_data = train_data_df.apply(lambda x: ' '.join(x), axis=1)
    train_labels = train_df.pop('def_pay')

    val_data_df = val_df.loc[:, 'LIMIT_BAL':'PAY_AMT6']
    val_data = val_data_df.apply(lambda x: ' '.join(x), axis=1)
    val_labels = val_df.pop('def_pay')

    test_data_df = test_df.loc[:, 'LIMIT_BAL':'PAY_AMT6']
    test_data = test_data_df.apply(lambda x: ' '.join(x), axis=1)
    test_labels = test_df.pop('def_pay')

    print(">>> generate vocabulary from all data")

    word_to_idx, idx_to_word, vocab = util.generate_vocab(train_data_df.values)

    # train_data, train_labels = util.generate_word_index(train_data, train_labels, word_to_idx,
    #                                                     idx_to_word, vocab, max_sequence_length)
    # val_data, val_labels = util.generate_word_index(val_data, val_labels, word_to_idx,
    #                                                 idx_to_word, vocab, max_sequence_length)
    test_data_to_ids, test_labels = util.generate_word_index(test_data, test_labels, word_to_idx,
                                                             idx_to_word, vocab, max_sequence_length)
    model = lstm_models.lstm(vocab, hidden_units, num_layers, max_sequence_length, is_attention, is_bidirectional)
    model = tf.keras.models.load_model(checkpoint_path)

    class_names = [0, 1]
    explainer = LimeTextExplainer(class_names=class_names, mask_string='UNK', bow=False,
                                  split_expression=r'[^a-zA-Z0-9-_]+')

    data_len = len(test_data)

    # get all predictions
    test_predictions = model.predict(test_data_to_ids, verbose=0)
    test_predictions = np.argmax(test_predictions, axis=1)
    print("Processing test data: ")

    feature_scores = []


    def predictor(examples):
        data, _ = util.generate_word_index(examples, None, word_to_idx,
                                           idx_to_word, vocab, max_sequence_length)
        return model.predict(data)



    personal_features = ["SEX", "EDUCATION", "MARRIAGE", "AGE"]
    column_names = train_data_df.columns
    feature_scores = {}
    for column in column_names:
        feature_scores[column] = 0

    # data_len = 10
    for i in tqdm(range(data_len)):
        raw_sent = test_data[i]
        label = test_labels[i]
        num_samples = 500
        value_feature_map = {}
        # splitter = re.compile(r'(%s)|$' % r'[^a-zA-Z0-9_]+')
        # values = [s for s in splitter.split(raw_sent) if len(s) > 0]
        values = raw_sent.split()
        for column, val in zip(column_names, values):
            value_feature_map[val] = column
        predicted_label = [test_predictions[i]]
        exp = explainer.explain_instance(raw_sent,
                                         predictor,
                                         num_features=max_sequence_length,
                                         labels=predicted_label,
                                         num_samples=num_samples)
        if exp is not None:
            exp_list = exp.as_list(label=predicted_label[0])
            # feature_scores.append([score for _, score in exp_list])
            for val, score in exp_list:
                feature_scores[value_feature_map[val]] = score

    # score_avg = np.average(feature_scores, axis=0)
    # column_importance = list(zip(column_names, score_avg))
    for feature in feature_scores.keys():
        feature_scores[feature] /= data_len

    sorted_features = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)
    print("**** Importance from high to low: ****")
    for feature_name, score in sorted_features:
        print(feature_name + ": " + str(score))
