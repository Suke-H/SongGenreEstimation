import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

import torch

from models import NN, L2Softmax

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

def scatter_plot(X, y, name):
    """ 散布図の描画 """

    # 20色のカラーマップ
    cm = plt.cm.get_cmap('tab20')

    # クラスごとに可視化
    for label in np.unique(y):
        
        # 8, 10はスキップ
        if label in [8, 10]:
            continue

        plt.scatter(X[y == label, 0], X[y == label, 1], 
                    label=str(label), marker=".", color=cm.colors[label])
        
    plt.legend()
    plt.savefig("Results/"+name+".png")
    plt.close()

# def tSNE(model_list, X, y, skf):
#     # GPU
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     num_folds = len(model_list)
#     feature_list = np.zeros((X.shape[0], 100))

#     for fold, (indexes_trn, indexes_val) in enumerate(skf.split(X, y)):
#         print(f"------------------------------ fold {fold} ------------------------------")

#         X_train, y_train, X_val, y_val = X[indexes_trn], y[indexes_trn], X[indexes_val], y[indexes_val]

#         model = model_list[fold]
#         model = model.to(device)
#         model.eval() # 推論モードへ

#         # 20次元の特徴抽出
#         _, features = model(torch.from_numpy(X_val).to(device))
#         features = features.to('cpu').detach().numpy().copy()

#         feature_list[indexes_val] = features

#     # tSNE
#     tsne = TSNE(n_components=2, random_state=41, perplexity=i)
#     tsne_transformed = tsne.fit_transform(feature_list)
#     print(tsne_transformed.shape)

#     return tsne_transformed

def tSNE(model_list, X, y, skf):
    # GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.from_numpy(X).to(device)
    num_folds = len(model_list)
    feature_list = np.zeros((num_folds, X_tensor.shape[0], 20))

    for i, model in enumerate(model_list):

        model = model.to(device)
        model.eval() # 推論モードへ

        # 20次元の特徴抽出
        _, features = model(X_tensor)
        features = features.to('cpu').detach().numpy().copy()

        feature_list[i] = features

    # 平均
    features = np.mean(feature_list, axis=0)
    print(features.shape)

    # tSNE
    tsne = TSNE(n_components=2, random_state=41, perplexity=i)
    tsne_transformed = tsne.fit_transform(features)
    print(tsne_transformed.shape)

    return tsne_transformed

def outlier_detection(X, y):
    """ 外れ値検出 """
    outliers = np.zeros(len(y)).astype(int)

    # チューニング
    # n_neighbors_list = [5, 2, 5, 5, 2, 5, 5, 2, 5, 2, 5]
    n_neighbors_list = [2, 5, 2, 2, 5, 2, 2, 5, 5, 5, 5]
    contamination_list = [0.05, 0.2, 0.1, 0.1, 0.05, 0.1, 0.05, 0.2, 0.02, 0.05, 0.02]

    for label in np.unique(y):

        clf = LocalOutlierFactor(n_neighbors=n_neighbors_list[label], contamination=contamination_list[label])
        
        index = np.where(y == label)[0]
        # print(index, index.shape)
        pred = clf.fit_predict(X[index])
        # print(pred, pred.shape)

        unomaly_index = np.where(pred < 0)[0]

        outliers[index[unomaly_index]] = 1

    # print(list(unomaly_list))
    print(len(y), np.sum(outliers))

    return outliers

def visalize_outlier(features, outliers, y):
    """ 外れ値を可視化 """
    # 20色のカラーマップ
    cm = plt.cm.get_cmap('tab20')

    # normal: 0, outlier: 1
    # indices_NO = np.zeros(len(y)).astype(int)
    # indices_NO[outliers] = 1

    # クラスごとに可視化
    for label in np.unique(y):
        
        # 8, 10はスキップ
        # if label in [8, 10]:
        #     continue

        # 正解・不正解で場合分け

        indices_normal = np.where((y == label) & (outliers == 0))[0]
        indices_outlier = np.where((y == label) & (outliers == 1))[0]

        print("label: {}, normal: {}, outlier: {}".format(label, len(indices_normal), len(indices_outlier)))
        
        plt.scatter(features[indices_normal, 0], features[indices_normal, 1], 
                    label=str(label), marker=".", color=cm.colors[label])

        plt.scatter(features[indices_outlier, 0], features[indices_outlier, 1], 
                    marker="x", color=cm.colors[label])
        
    plt.legend()
    plt.savefig("Results/outlier_tune_x_on.png")
    plt.close()

if __name__ == '__main__':

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
    scatter_plot(features, y)

    # 外れ値検出
    outliers = outlier_detection(features, y)

    # 可視化
    visalize_outlier(features, outliers, y)



