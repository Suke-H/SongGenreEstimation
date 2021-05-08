import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import random
from collections import Counter
from matplotlib import pyplot as plt

from models import NN, L2Softmax, EarlyStopping, L2Softmax_tuning

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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def predict_NN(model, X, y=None):  # leave-one-out
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, X = model.to(device), torch.from_numpy(X).to(device)
    model.eval() # 推論モードに

    outputs, _ = model(X)
    _, preds = torch.max(outputs, 1)

    # softmax
    m = nn.Softmax(dim=1)
    outputs = m(outputs)
    outputs = outputs.to('cpu').detach().numpy().copy()
    preds = preds.to('cpu').detach().numpy().copy()

    # test
    if y is None:
        return outputs, preds

    # train
    else:
        score = f1_score(y, preds, average="macro")
        print(f"f1_score={score}\n")
        # print(classification_report(y, preds))
        # print(confusion_matrix(y, preds))

        return outputs, preds, score
    
def train(model, X_train, y_train, X_val, y_val, optimizer, criterion, num_epochs, path):
    """ 学習 """

    # ネットワークをGPUへ
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # earlystoppingインスタンス
    earlystopping = EarlyStopping(patience=300, verbose=False, path=path, param_name="f1 macro")

    # lossやaccのプロット用
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "f1_macro": []}

    # epochのループ
    for epoch in range(num_epochs):

        model.train() # 訓練モードに
        epoch_loss = epoch_corrects = total = 0

        dataloader = make_dataloader(X_train, y_train, 100)

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

def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # Tensorflow
    # tf.random.set_seed(seed)

def run_NN(df_train, df_test):

    SEED = 42
    fix_seed(SEED)

    #　trainデータ前処理
    X, y = preprocess(df_train)

    # 設定
    data_dim = X.shape[1]
    num_classes = 11
    num_folds = 4

    # ハイパーパラメータ
    num_layers = 3
    num_units = [200, 600, 400]
    dropouts = [0.3, 0.3, 0.1]
    alpha = 32

    # モデル呼び出し
    model_list = [L2Softmax_tuning(data_dim, num_classes, num_layers, num_units, dropouts, alpha) for i in range(num_folds)]
    model_path_list = ['Model/NN_' + str(i) + '.pth' for i in range(num_folds)]

    # 最適アルゴリズム
    adam_lr = 0.00035338
    weight_decay = 0
    optimizer_list = [optim.Adam(model_list[i].parameters(), lr=adam_lr, weight_decay=weight_decay) for i in range(num_folds)]

    # 損失関数
    criterion = nn.CrossEntropyLoss(reduction='mean')

    # 学習回数
    num_epochs = 100000

    ### 学習（StratifiedKFoldを使用）#################################
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=15)
    oof = np.zeros(len(y))
    output_list = np.zeros((X.shape[0], num_classes))
    f1_list = np.zeros(num_folds)

    for fold, (indexes_trn, indexes_val) in enumerate(skf.split(X, y)):
        print(f"------------------------------ fold {fold} ------------------------------")

        X_train, y_train, X_val, y_val = X[indexes_trn], y[indexes_trn], X[indexes_val], y[indexes_val]

        # 学習
        train(model_list[fold], X_train, y_train, X_val, y_val, \
                optimizer_list[fold], criterion, num_epochs, model_path_list[fold])

        # 各foldのvalidationデータの推定結果を合体させて、最終出力とする
        output_list[indexes_val], oof[indexes_val], f1_list[fold] = predict_NN(model_list[fold], X_val, y_val)

    print("CV: {}".format(np.mean(f1_list)))
    ####################################################

    # モデル呼び出し
    for i in range(num_folds):
        model_list[i].load_state_dict(torch.load(model_path_list[i]))

    # testデータ前処理
    X_test = preprocess_test(df_test)

    # pred_list = np.zeros((num_folds, X_test.shape[0]))
    train_output_list = np.zeros((X.shape[0], num_classes))
    test_output_list = np.zeros((num_folds, X_test.shape[0], num_classes))

    ### testデータを算出 ####################################
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=15)
    f1_list = np.zeros(num_folds)
    oof = np.zeros(len(y))
    for fold, (indexes_trn, indexes_val) in enumerate(skf.split(X, y)):
        print(f"------------------------------ fold {fold} ------------------------------")

        X_train, y_train, X_val, y_val = X[indexes_trn], y[indexes_trn], X[indexes_val], y[indexes_val]
        train_output_list[indexes_val], oof[indexes_val], f1_list[fold] = predict_NN(model_list[fold], X_val, y_val)

        # 出力
        test_output_list[fold], _ = predict_NN(model_list[fold], X_test)

    print("CV: {}".format(np.mean(f1_list)))

    # 再現性チェック用
    # cm = confusion_matrix(y, oof)
    # cmp = ConfusionMatrixDisplay(cm, display_labels=[str(i) for i in range(11)])
    # cmp.plot(cmap=plt.cm.Blues)
    # plt.savefig("images/cm6.png", bbox_inches='tight')

    # 平均値
    y_out = np.mean(test_output_list, axis=0)

    # dfに変換
    df_share_train = pd.DataFrame(data=train_output_list, columns=['NN_'+str(i) for i in range(num_classes)])
    df_share_train.index = [i for i in range(4046)]

    df_share_test = pd.DataFrame(data=y_out, columns=['NN_'+str(i) for i in range(num_classes)])
    df_share_test.index = [i+4046 for i in range(4046)]

    # 最終予測
    y_preds_train = np.argmax(train_output_list, axis=1)
    y_preds_test = np.argmax(y_out, axis=1)

    return df_share_train, df_share_test

if __name__ == '__main__':
    train_df = pd.read_csv("Data/train_m.csv")
    test_df = pd.read_csv("Data/test_m.csv")
    train_pred, test_pred = run_NN(train_df, test_df)

    print(train_pred.shape)
    print(test_pred.shape)
