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
from collections import Counter
from imblearn.over_sampling import SMOTE

from models import NN, L2Softmax, EarlyStopping
from vis import loss_plot, acc_plot, visualize, visualize_with_model, visualize_with_model_list
from SMOTE import PreProcess

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

def encode_1st_data(y):
    """ 
    8, 10, それ以外の3クラスに正解ラベルを変換
    8 -> 0, 10 -> 1, else -> 2
    """

    # print(sorted(Counter(y).items()))

    # 8, 10, else
    indices_8, indices_10 = np.where(y == 8)[0], np.where(y == 10)[0]
    indices_else = np.where((y != 8) & (y != 10))[0]
    # print(len(y), len(indices_8), len(indices_10), len(indices_else))

    # 8 -> 0, 10 -> 1, else -> 2
    y_1st = np.copy(y)
    y_1st[indices_8] = 0
    y_1st[indices_10] = 1
    y_1st[indices_else] = 2

    return y_1st

def encode_2nd_data(y):
    """ 
    0, 1, 2, 3, 4, 5, 6, 7, 9の9クラスに正解ラベルを変換
    0->0, 1->1, ... , 7->7, 9->8なので、9のみ8に変換
    """

    # print(sorted(Counter(y).items()))

    # 0->0, 1->1, ... , 7->7, 9->8なので、9のみ8に変換
    y_2nd = np.copy(y)
    indices_9 = np.where(y == 9)[0]
    y_2nd[indices_9] = 8

    # print(sorted(Counter(y_2nd).items()))
    # a = input()
    
    return y_2nd

def make_1st_dataloader(X, y, batch_size):
    """ 8, 10, それ以外の3クラスにして、dataloader作成 """

    # 正解ラベルを変換
    y_1st = encode_1st_data(y)

    dataset = data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y_1st))
    dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    return dataloader

def make_2nd_dataloader(X, y, batch_size):
    """ 0, 1, 2, 3, 4, 5, 6, 7, 9の9クラスにして、dataloader作成 """

    # 正解ラベルを変換
    y_2nd = encode_2nd_data(y)

    dataset = data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y_2nd))
    dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    return dataloader

def calc_loo_cv_score(model_1st, model_2nd, X, y=None):  # leave-one-out
    
    # 推論
    preds = predict(model_1st, model_2nd, X)

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
    
def train(model, X_train, y_train, X_val, y_val, optimizer, criterion, num_epochs, path, phase):
    """ 学習 """

    # ネットワークをGPUへ
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # earlystoppingインスタンス
    earlystopping = EarlyStopping(patience=300, verbose=True, path=path, param_name="f1 macro")

    # lossやaccのプロット用
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "f1_macro": []}

    # validationデータの正解ラベルを変換
    if phase == "1st":
        y_val = encode_1st_data(y_val)

    elif phase == "2nd":
        y_val = encode_2nd_data(y_val)

    # epochのループ
    for epoch in range(num_epochs):

        model.train() # 訓練モードに

        epoch_loss = epoch_corrects = total = 0

        # 1stなら8, 10, elseの3クラス分類
        if phase == "1st":
            dataloader = make_1st_dataloader(X_train, y_train, 100)

        # 2ndなら0, 1, 2, 3, 4, 5, 6, 7, 9の9クラス分類
        elif phase == "2nd":
            dataloader = make_2nd_dataloader(X_train, y_train, 100)

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


def predict(model_1st, model_2nd, X):
    """ 1層, 2層のNNを組み合わせて推論 """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_1st, model_2nd = model_1st.to(device), model_2nd.to(device)
    model_1st.eval() # 推論モードに
    model_2nd.eval()

    # 最終推論結果
    preds = np.zeros(X.shape[0]).astype(int)
    
    # 最終出力
    num_classes = 11
    outputs = np.zeros((X.shape[0], num_classes))

    ### 1st ############################################
    X_tensor = torch.from_numpy(X).to(device)
    outputs_1st, _ = model_1st(X_tensor)
    _, preds_1st = torch.max(outputs_1st, 1)
    preds_1st = preds_1st.to('cpu').detach().numpy().copy().astype(int)
    print(sorted(Counter(preds_1st).items()))

    # 0 -> 8, 1 -> 10に変換
    indices_0 = np.where(preds_1st == 0)[0]
    indices_1 = np.where(preds_1st == 1)[0]
    preds[indices_0] = 8
    preds[indices_1] = 10

    # 2（else）を2ndに
    indices_else = np.where(preds_1st == 2)[0]
    X_else = X[indices_else]

    ### 2nd ############################################
    X_tensor = torch.from_numpy(X_else).to(device)
    outputs_2nd, _ = model_2nd(X_tensor)
    _, preds_2nd = torch.max(outputs_2nd, 1)
    preds_2nd = preds_2nd.to('cpu').detach().numpy().copy().astype(int)
    # print(sorted(Counter(preds_2nd).items()))

    # 0->0, 1->1, ... , 7->7, 8->9なので、8のみ9に変換
    indices_8 = np.where(preds_2nd == 8)[0]
    preds_2nd_tmp = np.copy(preds_2nd)
    preds_2nd_tmp[indices_8] = 9
    preds[indices_else] = preds_2nd_tmp

    # print(sorted(Counter(preds).items()))
    # a = input()

    return preds

if __name__ == '__main__':

    #　trainデータ読み込み
    datapath = "Data/"
    df_train = pd.read_csv(datapath+"train_m.csv")
    X, y = preprocess(df_train)

    ### LOF + SMOTE  #################################
    # X, y = PreProcess()

    # np.save(datapath+"X_up", X)
    # np.save(datapath+"y_up", y)

    # X = np.load(datapath+"X_up.npy")
    # y = np.load(datapath+"y_up.npy")
    ##################################################

    # モデル呼び出し
    data_dim = X.shape[1]
    num_classes = 11
    num_folds = 4

    model_1st_list = [L2Softmax(data_dim, 3) for i in range(num_folds)]
    model_1st_path_list = ['Model/NN_step/L2Softmax_1st_' + str(i) + '.pth' for i in range(num_folds)]
    # model_1st_list = [NN(data_dim, num_classes) for i in range(num_folds)]
    # model_1st_path_list = ['Model/NN/NN_down_' + str(i) + '.pth' for i in range(num_folds)]

    model_2nd_list = [L2Softmax(data_dim, 9) for i in range(num_folds)]
    model_2nd_path_list = ['Model/NN_step/L2Softmax_2nd_' + str(i) + '.pth' for i in range(num_folds)]
    # model_2nd_list = [NN(data_dim, num_classes) for i in range(num_folds)]
    # model_2nd_path_list = ['Model/NN/NN_down_' + str(i) + '.pth' for i in range(num_folds)]

    # 最適アルゴリズム
    lr = 0.001
    beta1, beta2 = 0.0, 0.9
    optimizer_1st_list = [torch.optim.Adam(model_1st_list[i].parameters(), lr, [beta1, beta2])\
                            for i in range(num_folds)]
    optimizer_2nd_list = [torch.optim.Adam(model_2nd_list[i].parameters(), lr, [beta1, beta2])\
                            for i in range(num_folds)]

    # 損失関数
    criterion = nn.CrossEntropyLoss(reduction='mean')
    # criterion = SelfAdjDiceLoss()

    # 学習回数
    num_epochs = 100000

    ### 学習（StratifiedKFoldを使用）#################################
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=15)
    oof = np.zeros(len(y))
    f1_list = np.zeros(num_folds)

    for fold, (indexes_trn, indexes_val) in enumerate(skf.split(X, y)):
        print(f"------------------------------ fold {fold} ------------------------------")

        X_train, y_train, X_val, y_val = X[indexes_trn], y[indexes_trn], X[indexes_val], y[indexes_val]

        ### 学習 ###
        # 1st
        train(model_1st_list[fold], X_train, y_train, X_val, y_val, optimizer_1st_list[fold], \
            criterion, num_epochs, model_1st_path_list[fold], "1st")

        # 8, 10のデータを抜かす
        indices_else_train = np.where((y_train != 8) & (y_train != 10))[0]
        X_2nd_train, y_2nd_train = X_train[indices_else_train], y_train[indices_else_train]
        # print(sorted(Counter(y_2nd_train).items()))
        # a = input()

        indices_else_val = np.where((y_val != 8) & (y_val != 10))[0]
        X_2nd_val, y_2nd_val = X_val[indices_else_val], y_val[indices_else_val]
        # print(sorted(Counter(y_2nd_val).items()))
        # a = input()

        # 2nd
        train(model_2nd_list[fold], X_2nd_train, y_2nd_train, X_2nd_val, y_2nd_val, \
            optimizer_2nd_list[fold], criterion, num_epochs, model_2nd_path_list[fold], "2nd")

        # 各foldのvalidationデータの推定結果を合体させて、最終出力とする
        # _, oof[indexes_val], f1_list[fold] = calc_loo_cv_score(model_1st_list[fold], model_2nd_list[fold], X_val, y_val)
        oof[indexes_val], f1_list[fold] = calc_loo_cv_score(model_1st_list[fold], model_2nd_list[fold], X_val, y_val)

    print("CV: {}".format(np.mean(f1_list)))

    ####################################################

    # 可視化
    # visualize(X, y)
    # visualize_with_model_list(model_list, X_test, y_preds, "Results/SMOTE/test/")
    # visualize_with_model_list(model_list, X, oof.astype(np.int), "Results/step/", y)
    # visualize_with_model_list(model_list, X_val, oof[indexes_val].astype(np.int), y_val)
    # print(confusion_matrix(y, oof.astype(np.int)))
