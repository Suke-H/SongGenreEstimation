import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter

from models import NN, L2Softmax, EarlyStopping
from vis import loss_plot, acc_plot, visualize, visualize_with_model, visualize_with_model_list
from LOF import scatter_plot, outlier_detection, visalize_outlier, tSNE

def mean_norm(df):
    """ 標準化 """
    return df.apply(lambda x: (x-x.mean())/ x.std(), axis=0)

def preprocess(df_train):
    """ 前処理  """

    df = df_train.copy()

    # データと正解ラベルを分割
    X, y = df.drop(['genre', 'index'], axis=1), df["genre"]

    # 標準化
    X = mean_norm(X)

    # numpyに変換
    X, y = X.values.astype(np.float32), y.values

    return X, y

def PreProcess():
    #　trainデータ読み込み
    datapath = "Data/"
    df_train = pd.read_csv(datapath+"train_m.csv")
    X, y = preprocess(df_train)

    # モデル呼び出し
    data_dim = X.shape[1]
    num_classes = 11
    num_folds = 4
    model_list = [L2Softmax(data_dim, num_classes) for i in range(num_folds)]
    # model_list = [NN(data_dim, num_classes) for i in range(num_folds)]
    model_path_list = ['Model/NN/L2Softmax_' + str(i) + '.pth' for i in range(num_folds)]
    # model_path_list = ['Model/NN/NN_' + str(i) + '.pth' for i in range(num_folds)]
    for i in range(num_folds):
        model_list[i].load_state_dict(torch.load(model_path_list[i]))

    # NN -> tSNE
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=15)
    features = tSNE(model_list, X, y, skf)

    # 可視化
    # scatter_plot(features, y, "feature")

    # 外れ値検出
    outliers = outlier_detection(features, y)

    # 可視化
    visalize_outlier(features, outliers, y)

    # 外れ値除去
    down_indices = np.where(outliers == 0)[0]
    X_down, y_down = X[down_indices], y[down_indices]
    print(len(outliers), len(y_down))

    # オーバーサンプリング
    sm = SMOTE()
    X_up, y_up = sm.fit_resample(X_down, y_down)
    print(sorted(Counter(y_up).items()))

    features_up = tSNE(model_list, X_up, y_up, skf)
    scatter_plot(features_up, y_up, "feature_up")

    return X_up, y_up

if __name__ == '__main__':
    PreProcess()
    

