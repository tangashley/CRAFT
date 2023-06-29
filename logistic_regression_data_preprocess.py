import os
from pathlib import Path

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


# import visualization libraries
# import seaborn as sns
# import matplotlib.pyplot as plt
# from ggplot import *
# %matplotlib inline

def proprocess(save_folder):
    df = pd.read_csv('data/UCI_Credit_Card.csv')

    # clean data, some data fields have undocumented labels and values
    fil = (df.EDUCATION == 6) | (df.EDUCATION == 0)
    df.loc[fil, 'EDUCATION'] = 5

    df['EDUCATION'] = df['EDUCATION'].replace(1, 'grad')
    df['EDUCATION'] = df['EDUCATION'].replace(2, 'univ')
    df['EDUCATION'] = df['EDUCATION'].replace(3, 'high')
    df['EDUCATION'] = df['EDUCATION'].replace(4, 'others')
    df['EDUCATION'] = df['EDUCATION'].replace(4, 'unknown')

    df.loc[df.MARRIAGE == 0, 'MARRIAGE'] = 3
    df['MARRIAGE'] = df['MARRIAGE'].replace(1, 'married')
    df['MARRIAGE'] = df['MARRIAGE'].replace(2, 'single')
    df['MARRIAGE'] = df['MARRIAGE'].replace(3, 'others')

    df['SEX'] = df['SEX'].replace(1, 'male')
    df['SEX'] = df['SEX'].replace(2, 'female')


    df = df.rename(columns={'default.payment.next.month': 'def_pay',
                            'PAY_0': 'PAY_1'})
    fil = (df.PAY_1 == -2) | (df.PAY_1 == -1) | (df.PAY_1 == 0)
    df.loc[fil, 'PAY_1'] = 0
    fil = (df.PAY_2 == -2) | (df.PAY_2 == -1) | (df.PAY_2 == 0)
    df.loc[fil, 'PAY_2'] = 0
    fil = (df.PAY_3 == -2) | (df.PAY_3 == -1) | (df.PAY_3 == 0)
    df.loc[fil, 'PAY_3'] = 0
    fil = (df.PAY_4 == -2) | (df.PAY_4 == -1) | (df.PAY_4 == 0)
    df.loc[fil, 'PAY_4'] = 0
    fil = (df.PAY_5 == -2) | (df.PAY_5 == -1) | (df.PAY_5 == 0)
    df.loc[fil, 'PAY_5'] = 0
    fil = (df.PAY_6 == -2) | (df.PAY_6 == -1) | (df.PAY_6 == 0)
    df.loc[fil, 'PAY_6'] = 0

    # group numerical values to different bins
    # feature_names = ['LIMIT_BAL', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4',
    #                  'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2',
    #                  'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    #
    # agg = df[[*feature_names]].agg(['min', 'max'])
    # for name in feature_names:
    #     bins = []
    #     bin_names = []
    #     count = 1
    #     for i in range(int(agg.min().min()) - 1000, int(agg.max().max()) + 1000, 1000):
    #         bins.append(i)
    #         bin_names.append(count)
    #
    #     df[name] = pd.cut(df[name], bins, labels=False)

    # pays = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    # scale_features = ['LIMIT_BAL', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4',
    #                 'BILL_AMT5', 'BILL_AMT6','PAY_AMT1', 'PAY_AMT2',
    #                 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    #
    # for pay in pays:
    #     df[pay] = pay + '_' + df[pay].astype(str)
    #
    # for feature in scale_features:
    #     temp = df[feature].apply(lambda x: round(x / 1000))
    #     df[feature] = feature + '_' + temp.astype(str)



    # randomly select train(60%), validation(20%) and test(20%)
    train, validation, test = np.split(df.sample(frac=1, random_state=42),
                                       [int(.6 * len(df)), int(.8 * len(df))])

    train.to_csv(os.path.join(save_folder, "train_data.csv"),
                 encoding='utf-8', index=False)
    validation.to_csv(os.path.join(save_folder, "validation_data.csv"),
                      encoding='utf-8', index=False)
    test.to_csv(os.path.join(save_folder, "test_data.csv"),
                encoding='utf-8', index=False)


if __name__ == '__main__':
    save_folder = "data/logistic_regression"
    if not Path(save_folder).exists():
        os.makedirs(save_folder)
        print('Create directory: ' + save_folder)
    proprocess(save_folder)
