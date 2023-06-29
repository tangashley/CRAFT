from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import os

import tensorflow as tf
from sklearn import metrics
from numpy.random import seed
import pandas as pd
from sklearn.linear_model import LogisticRegression

seed(1)
tf.random.set_seed(1)


def test(test_model, test_data, test_labels, result_dir, show_mistake=False):
    print("Testing started!")
    test_predictions = test_model.predict(test_data)
    if not Path(result_dir).exists():
        os.makedirs(result_dir)
        print('Create directory: ' + result_dir)
    filepath1 = os.path.join(result_dir, 'test_predictions')
    filepath2 = os.path.join(result_dir, 'test_labels')
    np.save(filepath1, test_predictions)
    np.save(filepath2, test_labels)
    print("SAVED!")

    x = test_labels
    y = test_predictions

    res_accu = metrics.accuracy_score(x, y)
    res_precision = metrics.precision_score(x, y, average='weighted')
    res_recall = metrics.recall_score(x, y, average='weighted')
    res_f1 = metrics.f1_score(x, y, average='weighted')

    print('Test Accuracy: %.3f' % res_accu)
    print("Weighted!")
    print('Test F1-score: %.3f' % res_f1)
    print('Test Recall: %.3f' % res_recall)
    print('Test Precision: %.3f' % res_precision)


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
    parser.add_option("--input_data_path", dest="input_data_path", default="../data", help="input datapath",
                      metavar="FILE")
    parser.add_option("--result_dir", dest="result_dir", default="../results/", help="save result path", metavar="FILE")

    (options, args) = parser.parse_args()
    print("options", options)
    print("args", args)

    input_data_path = options.input_data_path
    result_dir = options.result_dir

    train_df = pd.read_csv(input_data_path + '/train_data.csv')
    val_df = pd.read_csv(input_data_path + '/validation_data.csv')
    test_df = pd.read_csv(input_data_path + '/test_data.csv')

    train_data_df = train_df.loc[:, 'LIMIT_BAL':'PAY_AMT6']
    train_labels = train_df.pop('def_pay')

    val_data_df = val_df.loc[:, 'LIMIT_BAL':'PAY_AMT6']
    val_labels = val_df.pop('def_pay')

    test_data_df = test_df.loc[:, 'LIMIT_BAL':'PAY_AMT6']
    test_labels = test_df.pop('def_pay')

    categorical_features = ['MARRIAGE', 'EDUCATION', 'SEX']
    numerical_features = [x for x in train_data_df.columns if x not in categorical_features]
    transformer = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('num', MinMaxScaler(), numerical_features)])

    model = LogisticRegression(max_iter=1000)
    # fit on the training set
    clf = Pipeline(
        steps=[("preprocessor", transformer), ("classifier", model)]
    )
    clf.fit(train_data_df, train_labels)
    # TESTING
    print('\n>>> Testing...')
    test(clf, test_data_df, test_labels, result_dir, False)
