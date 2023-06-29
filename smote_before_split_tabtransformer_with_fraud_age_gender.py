import argparse

import numpy as np
import pandas as pd
from numpy import where
from sklearn import metrics
import sklearn
from torch import optim

import tsai.metrics
from fastai.callback.tracker import SaveModelCallback, EarlyStoppingCallback
from fastai.tabular.core import Categorify, FillMissing
from fastai.tabular.data import TabularDataLoaders
from tsai.all import *
import torch
from imblearn.over_sampling import SMOTE, SMOTENC

## Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# set seaborn style because it prettier
from tsai.models.TabTransformerAttn import TabTransformerAttn

sns.set()

## Metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc

## Models
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

# from tsai.data.tabular import
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC


# Function for plotting ROC_AUC curve
def plot_roc_auc(y_test, preds, path):
    '''
    Takes actual and predicted(probabilities) as input and plots the Receiver
    Operating Characteristic (ROC) curve
    '''
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(path)
    plt.show()

def binary_acc(y_pred, y_true):
    y_pred_tag = torch.round(y_pred)
    correct_result_sum = (y_pred_tag == y_true).sum().float()
    acc = correct_result_sum/y_true.shape[0]
    acc = torch.round(acc * 100)
    return acc

def train_model(config):
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0") if use_gpu else 'cpu'

    data = pd.read_csv('data/kaggle_fraud_detection_age_gender/bs140513_032310.csv')
    # Create two dataframes with fraud and non-fraud data
    df_fraud = data.loc[data.fraud == 1]
    df_non_fraud = data.loc[data.fraud == 0]
    pos_weight = len(df_non_fraud)/len(df_fraud)


    # dropping zipcodeori and zipMerchant since they have only one unique value
    data_reduced = data.drop(['zipcodeOri', 'zipMerchant'], axis=1)

    y = data[['fraud']]
    data_reduced = data_reduced.drop(['fraud'], axis=1)

    # turning object columns type to categorical for easing the transformation process
    col_categorical = data_reduced.select_dtypes(include=['object']).columns
    col_num = data_reduced.select_dtypes(exclude=['object']).columns

    for col in col_categorical:
        data_reduced[col] = data_reduced[col].astype('category')
    data_reduced[col_categorical] = data_reduced[col_categorical].applymap(lambda x: x.replace("'", ""))

    # categorical values ==> numeric values
    # data_reduced[col_categorical] = data_reduced[col_categorical].apply(lambda x: x.cat.codes)



    # SMOTE for balancing training dataset
    idx = []
    for v in col_categorical.values:
        idx.append(data_reduced.columns.get_loc(v))
    sm = SMOTENC(categorical_features=idx, random_state=42)
    X, y = sm.fit_resample(data_reduced, y)
    y = pd.DataFrame(y)
    # summarize the new class distribution
    print('y values: ')
    print(y.value_counts())

    # scale the numerical columns
    scaler = MinMaxScaler()
    X[col_num] = scaler.fit_transform(X[col_num])

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, shuffle=True, test_size=0.3)

    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)


    if config.algo == 'xgboost':
        # categorical values ==> numeric values
        X_train[col_categorical] = X_train[col_categorical].apply(lambda x: x.cat.codes)
        X_val[col_categorical] = X_val[col_categorical].apply(lambda x: x.cat.codes)
        X_test[col_categorical] = X_test[col_categorical].apply(lambda x: x.cat.codes)

        XGBoost_CLF = xgb.XGBClassifier(max_depth=6, learning_rate=0.05, n_estimators=400,
                                        objective="binary:hinge", booster='gbtree',
                                        n_jobs=-1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0,
                                        subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1,
                                        base_score=0.5, random_state=42,
                                        scale_pos_weight=pos_weight)
        XGBoost_CLF.fit(X_train, y_train)
        y_pred = XGBoost_CLF.predict(X_test)
        print("Classification Report for XGBoost: \n", classification_report(y_test, y_pred))
        print("Confusion Matrix of XGBoost: \n", confusion_matrix(y_test, y_pred))
        plot_roc_auc(y_test, XGBoost_CLF.predict_proba(X_test)[:, 1], "plot/fraud_detect_age_gender_xgboost.pdf")

    elif config.algo == 'tabtransformerattn':
        torch.manual_seed(1)
        np.random.seed(1)

        X_merged = pd.concat([X_train, X_test])
        y_merged = pd.concat([y_train, y_test])
        data_merged = pd.concat([X_merged, y_merged], axis=1)
        val_index = list(range(len(X_train), len(data_merged)))

        # print("########################")
        # print(y_merged.head(10))
        # print("########################")

        dls = TabularDataLoaders.from_df(data_merged, y_names="fraud",
                                         cat_names=['age', 'category', 'customer', 'gender', 'merchant'],
                                         cont_names=['step', 'amount'],
                                         batch_size=1024,
                                         valid_idx=val_index,
                                         y_block=CategoryBlock(),
                                         procs=[Categorify])
        # x_cat, x_cont, yb = first(dls.train)
        model = TabTransformerAttn(dls.classes, dls.cont_names, 2, d_v=32, d_k=32)

        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Number of parameters are: " + str(params))

        model.to(device)

        # pos_weight = float(np.sqrt(pos_weight))
        loss_func = None #if config.do_smote else fastai.losses.CrossEntropyLossFlat(weight=torch.tensor([1.0, pos_weight]))
        learn = Learner(dls, model, lr=config.lr, loss_func=loss_func, metrics=[accuracy, Precision(), Recall(), F1Score()])
        # learn = Learner(dls, model, lr=config.lr, loss_func=loss_func)

        save_path = 'fraud_age_gender/tabtransformer_4'

        if config.monitor == 'valid_loss':
            comp = np.less
        else:
            comp = np.greater

        callbacks = [SaveModelCallback(monitor=config.monitor, comp=comp, min_delta=0.0,
                                       fname=save_path, every_epoch=False, at_end=False,
                                       with_opt=False, reset_on_fit=True),
                     EarlyStoppingCallback(monitor=config.monitor, comp=comp, min_delta=0.0,
                                           patience=5, reset_on_fit=True)]
        learn.fit(config.n_epochs, cbs=callbacks)

        # load the saved model which is the best model
        learn = learn.load(save_path, device=device)

        test_df = pd.concat([X_test, y_test], axis=1)
        test_dl = learn.dls.test_dl(test_df)
        preds, _ = learn.get_preds(dl=test_dl)
        final_preds = preds.numpy()
        final_preds = np.argmax(final_preds, axis=1)

        print("Classification Report for TabTransformerAttn: \n", classification_report(y_test, final_preds))
        print("Confusion Matrix of TabTransformerAttn: \n", confusion_matrix(y_test, final_preds))

    elif config.algo == 'lightgbm':
        fit_params = {"early_stopping_rounds": 10,
                      "eval_metric": 'auc',
                      "eval_set": [(X_val, y_val)],
                      'eval_names': ['valid'],
                      'verbose': 100,
                      'feature_name': 'auto',  # that's actually the default
                      'categorical_feature': 'auto'  # that's actually the default
                      }
        clf = lgb.LGBMClassifier(num_leaves=15, max_depth=-1,
                                 random_state=314,
                                 silent=True,
                                 metric='None',
                                 n_jobs=4,
                                 n_estimators=1000,
                                 colsample_bytree=0.9,
                                 subsample=0.9,
                                 learning_rate=0.1)
        clf.fit(X_train, y_train, **fit_params)
        y_pred = clf.predict(X_test)


        print("Classification Report for lightgbm: \n", classification_report(y_test, y_pred))
        print("Confusion Matrix of lightgbm: \n", confusion_matrix(y_test, y_pred))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("input_path", type=str, help="Path to the input data file")
    parser.add_argument("--algo", default="tabtransformerattn", type=str, help="algorithm name",
                        choices=['tabtransformerattn', 'xgboost', 'lightgbm'])
    parser.add_argument("--n_epochs", default=100, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--monitor", default="valid_loss", type=str)
    # parser.add_argument('--do_smote', action='store_true', default=False)
    config = parser.parse_args()


    print("Lr: " + str(config.lr))
    print("Monitoring: " + str(config.monitor))
    print('###################################################')
    print("SMOTE before splitting data")
    print('###################################################')
    # print("Do SMOTE: " + str(config.do_smote))

    train_model(config)
