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

from models import NN, L2Softmax, EarlyStopping
from vis import loss_plot, acc_plot, visualize, visualize_with_model, visualize_with_model_list
from loss import SelfAdjDiceLoss

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

def preprocess_test(df_test):
    """ 前処理  """

    df = df_test.copy()

    # データと正解ラベルを分割
    X = df.drop(['index'], axis=1)

    # 標準化
    X = mean_norm(X)

    return X.values.astype(np.float32)

def make_dataloader(X, y, batch_size):
    """ Dataloader作成 """

    dataset = data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    return dataloader

def make_under_sampling_dataloader(X, y, batch_size):
    """ 8と10を半分削ったDataloader作成 """

    indices_8, indices_10 = np.where(y == 8)[0], np.where(y == 10)[0]
    perm_8, perm_10 = np.random.permutation(len(indices_8) // 2), np.random.permutation(len(indices_10) // 2)
    delete_8, delete_10 = indices_8[perm_8], indices_10[perm_10]
    X_tmp, y_tmp = np.delete(X, delete_8, axis=0), np.delete(y, delete_8)
    X_tmp, y_tmp = np.delete(X_tmp, delete_10, axis=0), np.delete(y_tmp, delete_10)

    dataset = data.TensorDataset(torch.from_numpy(X_tmp), torch.from_numpy(y_tmp))
    dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    return dataloader

def calc_loo_cv_score(model, X, y=None):  # leave-one-out
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, X = model.to(device), torch.from_numpy(X).to(device)
    model.eval() # 推論モードに

    outputs, _ = model(X)
    _, preds = torch.max(outputs, 1)
    outputs = outputs.to('cpu').detach().numpy().copy()
    preds = preds.to('cpu').detach().numpy().copy()

    # test
    if y is None:
        return outputs, preds

    # train
    else:
        score = f1_score(y, preds, average="macro")
        print(f"f1_score={score}\n")
        print(classification_report(y, preds))
        print(confusion_matrix(y, preds))

        return outputs, preds, score
    

def train(model, X_train, y_train, X_val, y_val, optimizer, criterion, num_epochs, path):
    """ 学習 """

    # ネットワークをGPUへ
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # earlystoppingインスタンス
    earlystopping = EarlyStopping(patience=300, verbose=True, path=path, param_name="validation loss")

    # lossやaccのプロット用
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "f1_macro": []}

    # epochのループ
    for epoch in range(num_epochs):

        model.train() # 訓練モードに

        epoch_loss = epoch_corrects = total = 0
        dataloader = make_dataloader(X_train, y_train, 100)
        # dataloader = make_under_sampling_dataloader(X_train, y_train, 100)

        # データローダーからミニバッチを取り出すループ
        for inputs, labels in dataloader:

            inputs, labels = inputs.to(device), labels.to(device)

            # optimizerを初期化
            optimizer.zero_grad()

            # 順伝搬（forward）計算
            outputs, _ = model(inputs)

            loss = criterion(outputs, labels)  # 損失を計算
            _, preds = torch.max(outputs, 1)  # ラベルを予測

            # 逆伝播（backward）計算
            loss.backward()
            optimizer.step()

            # イタレーション結果の計算
            mini_batch_size = inputs.size(0)

            epoch_loss += loss.item() * mini_batch_size
            epoch_corrects += torch.sum(preds == labels.data)
            total += mini_batch_size*1.0

        # epochごとのlossと正解率を表示
        epoch_loss = epoch_loss / total
        epoch_acc = epoch_corrects / total

        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)

        # val_loss算出
        model.eval() # 推論モードに

        outputs, _ = model(torch.from_numpy(X_val).to(device))
        val_loss = criterion(outputs, torch.from_numpy(y_val).to(device))
        val_loss = val_loss.item()
        history["val_loss"].append(val_loss)

        # val_acc算出
        _, preds = torch.max(outputs, 1)
        corrects = torch.sum(preds == torch.from_numpy(y_val).to(device))
        # print(corrects, type(corrects), type(len(y_val)))
        val_acc = corrects / (len(y_val) * 1.0)
        history["val_acc"].append(val_acc)

        # f1 macro算出
        f1_macro = f1_score(y_val, preds.to('cpu').detach().numpy().copy(), average="macro")
        history["f1_macro"].append(f1_macro)

        # earlystoppingの判定
        # earlystopping(val_loss, model, f1_macro) # 最小化
        earlystopping(-f1_macro, model, f1_macro) # 最大化
        if earlystopping.early_stop:
            print("Early Stopping on {} epoch.".format(epoch))
            break

        if (epoch+1) % 100 == 0:
            print('epoch: {}, train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, f1 macro: {:.4f}'\
                .format(epoch, epoch_loss, epoch_acc, val_loss, val_acc, f1_macro))

    # loss_plot(history["loss"])
    # acc_plot(history["acc"])


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

    # 最適アルゴリズム
    lr = 0.001
    beta1, beta2 = 0.0, 0.9
    optimizer_list = [torch.optim.Adam(model_list[i].parameters(), lr, [beta1, beta2]) for i in range(num_folds)]

    # 損失関数
    criterion = nn.CrossEntropyLoss(reduction='mean')
    # criterion = SelfAdjDiceLoss()

    # 学習回数
    num_epochs = 100000

    ### 学習（StratifiedKFoldを使用）#################################
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=15)
    oof = np.zeros(len(y))
    f1_list = np.zeros(num_folds)
    test_f1_list = np.zeros(num_folds)

    for fold, (indexes_trn, indexes_val) in enumerate(skf.split(X, y)):
        print(f"------------------------------ fold {fold} ------------------------------")

        X_train, y_train, X_val, y_val = X[indexes_trn], y[indexes_trn], X[indexes_val], y[indexes_val]

        # 学習
        train(model_list[fold], X_train, y_train, X_val, y_val, optimizer_list[fold], criterion, num_epochs, model_path_list[fold])

        # 各foldのvalidationデータの推定結果を合体させて、最終出力とする
        # oof[indexes_val] = model.predict(df_val, num_iteration=prediction_round)
        _, oof[indexes_val], f1_list[fold] = calc_loo_cv_score(model_list[fold], X_val, y_val)

        # test
        model_list[fold].load_state_dict(torch.load(model_path_list[fold]))
        _, _, test_f1_list[fold] = calc_loo_cv_score(model_list[fold], X_val, y_val)


    print("CV: {}".format(np.mean(f1_list)))
    print("test CV: {}".format(np.mean(test_f1_list)))

    ####################################################

    # モデル呼び出し
    for i in range(num_folds):
        model_list[i].load_state_dict(torch.load(model_path_list[i]))

    # testデータ読み込み
    df_test = pd.read_csv(datapath+"test_m.csv")
    X_test = preprocess_test(df_test)

    # pred_list = np.zeros((num_folds, X_test.shape[0]))
    output_list = np.zeros((num_folds, X_test.shape[0], num_classes))

    ### testデータを算出 ####################################

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=15)
    f1_list = np.zeros(num_folds)
    oof = np.zeros(len(y))
    for fold, (indexes_trn, indexes_val) in enumerate(skf.split(X, y)):
        print(f"------------------------------ fold {fold} ------------------------------")

        X_train, y_train, X_val, y_val = X[indexes_trn], y[indexes_trn], X[indexes_val], y[indexes_val]
        _, oof[indexes_val], f1_list[fold] = calc_loo_cv_score(model_list[fold], X_val, y_val)

        # 出力
        output_list[fold], _ = calc_loo_cv_score(model_list[fold], X_test)

    print("CV: {}".format(np.mean(f1_list)))

    # 平均値
    y_out = np.mean(output_list, axis=0)

    # 最終予測
    y_preds = np.argmax(y_out, axis=1)
    # print(y_preds)
    # print(y_preds.shape)

    ####################################################

    # 性能確認
    # calc_loo_cv_score(model, X, y)

    # 可視化
    # visualize(X, y)
    # visualize_with_model(model, X_test, y_preds)
    # visualize_with_model_list(model_list, X, oof.astype(np.int), y)
    # visualize_with_model_list(model_list, X_val, oof[indexes_val].astype(np.int), y_val)
