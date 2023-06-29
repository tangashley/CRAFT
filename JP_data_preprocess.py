import os
from pathlib import Path

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


# import visualization libraries
# import seaborn as sns
# import matplotlib.pyplot as plt
# from ggplot import *
# %matplotlib inline

def preprocess(save_folder):
    df = pd.read_csv('data/JP/Payment-Fraud.csv')

    # create bins for money values
    scaled_usd = df['USD_amount'].apply(lambda x: round(x/10))
    # add prefix to number values
    df['USD_amount'] = 'USD_' + scaled_usd.astype(str)
    df['Sender_Sector'] = 'Sec_' + df['Sender_Sector'].astype(str)
    # concat date and time into a single string
    df['Time'] = df['Time_step'].str.split().str[1].str.split(":").str[0]
    df['Time_step'] = df['Time_step'].str.split().str[0]
    df['Time_step'] = df[['Time_step', 'Time']].agg('_'.join, axis=1)
    df.pop('Time')

    for column in df.columns:
        df[column].fillna(column + '_N', inplace=True)

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
    save_folder = "data/JP/JP_feature_prefix_bin_stride_10_date_hour"
    if not Path(save_folder).exists():
        os.makedirs(save_folder)
        print('Create directory: ' + save_folder)
    preprocess(save_folder)
