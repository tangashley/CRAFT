import glob
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

if __name__ == '__main__':
    # folder = '../results/Fraud/greedy/*_less/' # fraud detection result
    folder = '../results/UCI/greedy/*_less_features/'  # UCI result
    csv_files = glob.glob(folder + "metrics.csv")
    csv_files.sort(key=lambda f: int(re.sub('\D', '', f)))
    print(csv_files)
    # Read each CSV file into DataFrame
    # This creates a list of dataframes
    df_list = [pd.read_csv(file) for file in csv_files]
    # print(df_list)
    datasets = []
    for i, df in zip(range(22), df_list):
        datasets.append(pd.DataFrame({'ROC_AUC': df['ROC AUC'],
                                      # 'num_less_feature': [i] * len(df['ROC AUC']),
                                      'num of removed features': str(i)}))
    if isinstance(datasets, list):
        datasets = pd.concat(datasets, ignore_index=True)
    print(datasets)

    sns.set(style="darkgrid", font_scale=1.1)
    # plt.legend(ncol=2)
    # plt.ylim(-140, 0)
    # pal = sns.mpl_palette("tab20b", 20)[:4] + sns.mpl_palette("tab20b", 20)[-8:-4]
    sns.set_context("paper")

    sns.lineplot(data=datasets, x='num of removed features', y='ROC_AUC')
    plt.savefig('../plot/UCI_greedy_fewer_features.png', dpi=600)
    plt.show()
