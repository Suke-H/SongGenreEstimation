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

def mean_norm(df):
    """ 標準化 """
    return df.apply(lambda x: (x-x.mean())/ x.std(), axis=0)

def preprocess(df_train):
    """ 前処理  """

    df = df_train.copy()

    # # ラベルエンコーディング
    # le1 = LabelEncoder()
    # encoded1 = le1.fit_transform(df['region'].values)
    # df['region'] = encoded1

    # le2 = LabelEncoder()
    # encoded2 = le2.fit_transform(df['tempo'].values)
    # df['tempo'] = encoded2

    # # 欠損値を平均で補完
    # df = df.fillna(df.mean())

    # print(df.isnull().sum())

    # データと正解ラベルを分割
    # X, y = df.drop(['genre', 'tempo'], axis=1), df["genre"]
    X, y = df.drop(['genre'], axis=1), df["genre"]

    # 標準化
    X = mean_norm(X)

    # numpyに変換
    X, y = X.values.astype(np.float32), y.values

    # 正解ラベルをone-hotに
    # n_labels = len(np.unique(y))
    # y = np.eye(n_labels)[y]

    return X, y

def make_dataloader(X, y, batch_size):
    """ Dataloader作成 """

    dataset = data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
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
    print(confusion_matrix(y, preds))

def train(model, dataloader, optimizer, criterion, num_epochs, path):
    """ 学習 """

    # ネットワークをGPUへ
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train() # 訓練モードに

    # earlystoppingインスタンス
    earlystopping = EarlyStopping(patience=100, verbose=True, path=path)

    # lossやaccのプロット用
    history = {"loss": [], "acc": []}

    # epochのループ
    for epoch in range(num_epochs):
        # print('Epoch {}/{}'.format(epoch+1, num_epochs))
        # print('-------------')

        epoch_loss = epoch_corrects = total = 0  

        # データローダーからミニバッチを取り出すループ
        for inputs, labels in dataloader:

            inputs, labels = inputs.to(device), labels.to(device)

            # optimizerを初期化
            optimizer.zero_grad()

            # 順伝搬（forward）計算
            outputs, features = model(inputs)

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

# def performance(model, dataloader):
#     # ネットワークをGPUへ
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     model.eval() # 推論モードに
#     correct = total = 0

#     with torch.no_grad():
#         for (inputs, labels) in dataloader:

#             inputs, labels = inputs.to(device), labels.to(device)

#             outputs, _ = model(inputs)
#             _, preds = torch.max(outputs, 1)
#             correct += torch.sum(preds == labels.data)
#             total += labels.size(0)*1.0
 
#     print("train Acc : %.4f" % (correct/total))


if __name__ == '__main__':

    #　読み込み
    datapath = "Data/"
    df_train = pd.read_csv(datapath+"train_m.csv")
    df_test = pd.read_csv(datapath+"test_m.csv")

    # 前処理
    X, y = preprocess(df_train)
    dataloader = make_dataloader(X, y, 100)

    # モデル呼び出し
    data_dim = X.shape[1]
    num_classes = 11
    model = Net(data_dim, num_classes)

    # 最適アルゴリズム
    lr = 0.001
    beta1, beta2 = 0.0, 0.9
    optimizer = torch.optim.Adam(model.parameters(), lr, [beta1, beta2])

    # 損失関数
    criterion = nn.CrossEntropyLoss(reduction='mean')

    # 学習回数
    num_epochs = 10000

    # earlystopping用のmodel保存先
    model_path = 'Model/NN/checkpoint_model.pth'

    # 学習
    # train(model, dataloader, optimizer, criterion, num_epochs, model_path)

    # モデル呼び出し
    # model_path = 'Model/NN/1.pth'
    model.load_state_dict(torch.load(model_path))

    # 性能確認
    # performance(model, dataloader)
    calc_loo_cv_score(model, X, y)

    # モデル保存
    # model_path = 'Model/NN/10000.pth'
    # torch.save(model.state_dict(), model_path)

    # 可視化
    # visualize(X, y)
    visualize_with_model(model, X, y)

    # features_path = "Data/Dataset_004/"
    # train_f = pd.read_parquet(features_path+"train.parq")

    # X_f, y_f = preprocess(train_f)

    # 可視化
    # visualize(model, X_f, y_f)
    # visualize(X, y)
