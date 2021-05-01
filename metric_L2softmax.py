import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import pandas as pd
from tqdm import tqdm
import pyarrow as pa

from models import Net, EarlyStopping
from vis import loss_plot, acc_plot, visualize, visualize_with_model
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

def calc_loo_cv_score(model, X, y):  # leave-one-out
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, X = model.to(device), torch.from_numpy(X).to(device)
    model.eval() # 推論モードに

    outputs, _ = model(X)
    _, preds = torch.max(outputs, 1)
    preds = preds.to('cpu').detach().numpy().copy()

    score = f1_score(y, preds, average="macro")
    print(f"f1_score={score}\n")
    print(classification_report(y, preds))
    cm = confusion_matrix(y, preds)
    print(cm)
    cm_sum = np.sum(cm, axis=1)
    print(cm_sum)

def train(model, X, y, optimizer, criterion, num_epochs, path):
    """ 学習 """

    # ネットワークをGPUへ
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train() # 訓練モードに

    # earlystoppingインスタンス
    earlystopping = EarlyStopping(patience=300, verbose=True, path=path)

    # lossやaccのプロット用
    history = {"loss": [], "acc": []}

    # epochのループ
    for epoch in range(num_epochs):

        epoch_loss = epoch_corrects = total = 0
        # dataloader = make_dataloader(X, y, 100)
        dataloader = make_under_sampling_dataloader(X, y, 100)

        # データローダーからミニバッチを取り出すループ
        for inputs, labels in dataloader:

            inputs, labels = inputs.to(device), labels.to(device)

            # optimizerを初期化
            optimizer.zero_grad()

            # 順伝搬（forward）計算
            outputs, feature = model(inputs)

            loss = criterion(outputs, labels)  # 損失を計算
            _, preds = torch.max(outputs, 1)  # ラベルを予測

            # print(preds)
            
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

        history["loss"].append(epoch_loss)
        history["acc"].append(epoch_acc)

        # print('epoch: {}, Loss: {:.4f} Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))

        # earlystoppingの判定
        earlystopping(epoch_loss, model)
        if earlystopping.early_stop:
            print("Early Stopping on {} epoch.".format(epoch))
            break

        if (epoch+1) % 100 == 0:
            print('epoch: {}, Loss: {:.4f} Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))

    loss_plot(history["loss"])
    acc_plot(history["acc"])


if __name__ == '__main__':

    #　読み込み
    datapath = "Data/"
    df_train = pd.read_csv(datapath+"train_m.csv")
    # df_test = pd.read_csv(datapath+"test_m.csv")

    # 前処理
    X, y = preprocess(df_train)
    # dataloader = make_dataloader(X, y, 100)

    # モデル呼び出し
    data_dim = X.shape[1]
    num_classes = 11
    model = Net(data_dim, num_classes)
    # model = L2ConstraintedNet(model)

    # 最適アルゴリズム
    lr = 0.001
    beta1, beta2 = 0.0, 0.9
    optimizer = torch.optim.Adam(model.parameters(), lr, [beta1, beta2])

    # 損失関数
    criterion = nn.CrossEntropyLoss(reduction='mean')
    # criterion = SelfAdjDiceLoss()

    # 学習回数
    num_epochs = 100000

    # earlystopping用のmodel保存先
    model_path = 'Model/NN/L2softmax.pth'

    # 学習
    # train(model, dataloader, optimizer, criterion, num_epochs, model_path)
    # train(model, X, y, optimizer, criterion, num_epochs, model_path)

    # モデル呼び出し
    model.load_state_dict(torch.load(model_path))

    # 性能確認
    calc_loo_cv_score(model, X, y)

    # 可視化
    # visualize(X, y)
    visualize_with_model(model, X, y)
