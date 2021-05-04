import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from tqdm import tqdm
import pyarrow as pa
from time import time
import lightgbm as lgb

from models import NN, L2Softmax, EarlyStopping
from vis import loss_plot, acc_plot, visualize, visualize_with_model, visualize_with_model_list
from loss import SelfAdjDiceLoss
from nn import preprocess, preprocess_test

def calc_lgbm_cv_score(model, F, y=None):  # leave-one-out

    prediction_round = model.best_iteration+150 if num_round >= 1e8 else num_round  # おまじない
    preds = model.predict(F, num_iteration=prediction_round)

    # test
    if y is None:
        return preds

    # train
    else:
        score = f1_score(y, preds, average="macro")
        print(f"f1_score={score}\n")
        print(classification_report(y, preds))
        print(confusion_matrix(y, preds))

        return preds, score

def feature_extract(model, X):
    """ NNで学習データ(前処理済み)を20次元の特徴に抽出 """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.from_numpy(X).to(device)

    model = model.to(device)
    model.eval() # 推論モードへ

    # 20次元の特徴抽出
    _, features = model(X_tensor)
    features = features.to('cpu').detach().numpy().copy()

    return features

def mean_feature_extract(model_list, X):
    """ 複数のNNの平均値で、学習データ(前処理済み)を20次元の特徴に抽出 """

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

    return features

if __name__ == '__main__':

    #　trainデータ読み込み
    datapath = "Data/"
    df_train = pd.read_csv(datapath+"train_m.csv")
    X, y = preprocess(df_train)

    # testデータ読み込み
    df_test = pd.read_csv(datapath+"test_m.csv")
    X_test = preprocess_test(df_test)

    # 設定
    data_dim = X.shape[1]
    num_classes = 11
    num_folds = 4
    model_list = []
    model_path_list = ['Model/LGBM/normal_' + str(i) + '.txt' for i in range(num_folds)]
    learning_rate = 0.01
    lgb_params = {
        "objective": "multiclass",
        "num_class": num_classes,
        #"metric": "None",
        "learning_rate": learning_rate,
        "num_leaves": 3,
        "min_data_in_leaf": 40,
        #"colsample_bytree": 1.0,
        #"feature_fraction": 1.0,
        #"bagging_freq": 0,
        #"bagging_fraction": 1.0,
        "verbosity": 0,
        "seed": 42,
    }

    ### 特徴抽出 #####################################################

    # NNモデル呼び出し
    nn_list = [L2Softmax(data_dim, num_classes) for i in range(num_folds)]
    # model_list = [NN(data_dim, num_classes) for i in range(num_folds)]
    nn_path_list = ['Model/NN/L2Softmax_' + str(i) + '.pth' for i in range(num_folds)]

    for i in range(num_folds):
        nn_list[i].load_state_dict(torch.load(nn_path_list[i]))

    # 特徴抽出

    F_test = mean_feature_extract(nn_list, X_test)

    ### 学習（StratifiedKFoldを使用）#################################

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=15)
    # oof = np.zeros(len(y))
    oof = np.zeros((len(y), num_classes))
    f1_list = np.zeros(num_folds)
    # test_f1_list = np.zeros(num_folds)
    predictions = np.zeros((X_test.shape[0], num_classes))

    for fold, (indexes_trn, indexes_val) in enumerate(skf.split(X, y)):
        print(f"------------------------------ fold {fold} ------------------------------")
        
        # trainデータのみ（valデータ以外）で学習したNNで特徴抽出
        F = feature_extract(nn_list[fold], X)

        # trainデータとvalデータに分割
        F_train, y_train, F_val, y_val = F[indexes_trn], y[indexes_trn], F[indexes_val], y[indexes_val]

        lgb_train = lgb.Dataset(F_train, label=y_train)
        lgb_val = lgb.Dataset(F_val, label=y_val)

        lgb_params["learning_rate"] = learning_rate + np.random.random() * 0.001  # おまじない
        num_round = 999999999

        # 学習
        model = lgb.train(lgb_params, lgb_train, num_round, 
            valid_sets=lgb_val, verbose_eval=300,
            early_stopping_rounds=300 if num_round >= 1e8 else None, fobj=None)

        model_list.append(model)

        # 保存
        model.save_model(model_path_list[fold])

        # cv
        prediction_round = model.best_iteration+150 if num_round >= 1e8 else num_round  # おまじない
        oof[indexes_val] = model.predict(F_val, num_iteration=prediction_round) 
        # print(classification_report(y_val, oof[indexes_val].argmax(1)))
        score = f1_score(y_val, oof[indexes_val].argmax(1), average="macro")
        f1_list[fold] = score

        predictions += model.predict(F_test, num_iteration=prediction_round) / num_folds

    print("CV: {}".format(np.mean(f1_list)))

    ####################################################

    # モデル呼び出し
    model_list = [lgb.Booster(model_file=model_path_list[i]) for i in range(num_folds)]

    # # testデータ読み込み
    # df_test = pd.read_csv(datapath+"test_m.csv")
    # X_test = preprocess_test(df_test)

    # pred_list = np.zeros((num_folds, X_test.shape[0]))
    output_list = np.zeros((num_folds, X_test.shape[0], num_classes))

    ### testデータを算出 ####################################

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=15)
    f1_list = np.zeros(num_folds)
    # oof = np.zeros(len(y))
    oof = np.zeros((len(y), num_classes))
    predictions = np.zeros((X_test.shape[0], num_classes))

    for fold, (indexes_trn, indexes_val) in enumerate(skf.split(X, y)):
        print(f"------------------------------ fold {fold} ------------------------------")

        # trainデータのみ（valデータ以外）で学習したNNで特徴抽出
        F = feature_extract(nn_list[fold], X)

        # trainデータとvalデータに分割
        F_train, y_train, F_val, y_val = F[indexes_trn], y[indexes_trn], F[indexes_val], y[indexes_val]

        num_round = 999999999
        model = model_list[fold]

        # cv
        prediction_round = model.best_iteration+150 if num_round >= 1e8 else num_round  # おまじない
        oof[indexes_val] = model.predict(F_val, num_iteration=prediction_round)
        print(classification_report(y_val, oof[indexes_val].argmax(1)))
        score = f1_score(y_val, oof[indexes_val].argmax(1), average="macro")
        f1_list[fold] = score
        predictions += model.predict(F_test, num_iteration=prediction_round) / num_folds

        # _, f1_list[fold] = calc_lgbm_cv_score(model_list[fold], F_val, y_val)

    print("CV: {}".format(np.mean(f1_list)))

    # 推論
    y_preds = predictions.argmax(1)
    print(predictions)
    print(y_preds)
    print(y_preds.shape)

    # 提出
    df_sample_sub = pd.read_csv(datapath+"sample_submit.csv", header=None)
    df_sample_sub.columns = ["index", "genre"]

    df_submission = df_sample_sub.copy()
    df_submission['genre'] = y_preds

    print(df_submission)
    # assert not df_submission["genre"].isna().any()

    print("genre counts")
    # display(df_submission["genre"].value_counts().sort_index())
    print(df_submission["genre"].value_counts().sort_index())

    print("\nfirst 10 test data")
    # display(df_submission.head(10))
    print(df_submission.head(10))

    # make submission file
    df_submission.to_csv("Submit/lightGBM.csv", header=None, index=False)
