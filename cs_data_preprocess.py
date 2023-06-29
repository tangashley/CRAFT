import os
from pathlib import Path

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


# import visualization libraries
# import seaborn as sns
# import matplotlib.pyplot as plt
# from ggplot import *
# %matplotlib inline
from sklearn.ensemble import RandomForestRegressor


def preprocess(save_folder):
    train_df = pd.read_csv('data/give_me_some_credit/cs-training.csv')
    test_df = pd.read_csv('data/give_me_some_credit/cs-test.csv')

    # clean data, some data fields have undocumented labels and values
    combine = [train_df, test_df]
    for df in combine:
        df.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)
        df['NumberOfDependents'].fillna(0, inplace=True)

    def randomforest_filled_func(df):
        # get data with non-missing value and missing value
        train_tmp = df[df.MonthlyIncome.notnull()]
        test_tmp = df[df.MonthlyIncome.isnull()]

        train_x_tmp = train_tmp.iloc[:, 2:].drop('MonthlyIncome', axis=1)
        train_y_tmp = train_tmp['MonthlyIncome']

        test_x = test_tmp.iloc[:, 2:].drop('MonthlyIncome', axis=1)

        rfr = RandomForestRegressor(random_state=2021, n_estimators=200, max_depth=3, n_jobs=-1)
        rfr.fit(train_x_tmp, train_y_tmp)

        # fill missing value with predicted value
        predicted = rfr.predict(test_x).round(0)
        df.loc[(df.MonthlyIncome.isnull()), 'MonthlyIncome'] = predicted
        return df

    for df in combine:
        df = randomforest_filled_func(df)

    train_df = train_df[train_df['age'] > 0]
    train_df = train_df[train_df['RevolvingUtilizationOfUnsecuredLines'] < 13]
    train_df = train_df[train_df['NumberOfTimes90DaysLate'] <= 17]
    train_df = train_df.loc[train_df["DebtRatio"] <= train_df["DebtRatio"].quantile(0.975)]
    train_df = train_df.drop_duplicates()

    x100_columns = ['RevolvingUtilizationOfUnsecuredLines',]

    # add column prefix
    for df in combine:
        for column in df.columns:
            prefix = ''.join([char for char in column if char.isupper()])
            df[column] =


if __name__ == '__main__':
    save_folder = "data/feature_prefix_bin_stride_20000"
    if not Path(save_folder).exists():
        os.makedirs(save_folder)
        print('Create directory: ' + save_folder)
    preprocess(save_folder)
