import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

from models import metric_Net, EarlyStopping
from vis import loss_plot, acc_plot, visualize, visualize_with_model
from loss import SelfAdjDiceLoss, CenterLoss

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


def train(model, X, y, num_epochs, path):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # NLLLoss
    # nllloss = nn.NLLLoss().to(device) #CrossEntropyLoss = log_softmax + NLLLoss
    nllloss = nn.CrossEntropyLoss(reduction='mean').to(device)
    # CenterLoss
    loss_weight = 1
    centerloss = CenterLoss(11, 2).to(device)
    
    # optimzer4nn
    # optimizer4nn = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    optimizer4nn = torch.optim.Adam(model.parameters(), 0.001, [0.0, 0.9])
    # sheduler = lr_scheduler.StepLR(optimizer4nn, 20, gamma=0.8)
    
    # optimzer4center
    # optimzer4center = optim.SGD(centerloss.parameters(), lr=0.5)
    optimzer4center = torch.optim.Adam(model.parameters(), 0.01, [0.0, 0.9])

    ip1_loader = []
    idx_loader = []

    model = model.to(device)
    model.train() # 訓練モードに

    # earlystoppingインスタンス
    earlystopping = EarlyStopping(patience=100, verbose=True, path=path)

    # lossやaccのプロット用
    history = {"loss": [], "acc": []}

    # epochのループ
    for epoch in range(num_epochs):

        epoch_loss = epoch_corrects = total = 0
        # dataloader = make_dataloader(X, y, 100)
        dataloader = make_under_sampling_dataloader(X, y, 100)

        for i, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            ip1, pred = model(data)
            # print(nllloss(pred, target).size())
            # print(centerloss(target, ip1).size())
            loss = nllloss(pred, target) + loss_weight * centerloss(target, ip1)

            optimizer4nn.zero_grad()
            optimzer4center.zero_grad()

            loss.backward()

            optimizer4nn.step()
            optimzer4center.step()

            ip1_loader.append(ip1)
            idx_loader.append((target))

            # イタレーション結果の計算
            mini_batch_size = data.size(0)

            epoch_loss += loss.item() * mini_batch_size
            # epoch_corrects += torch.sum(pred == target.data)
            total += mini_batch_size*1.0

        feat = torch.cat(ip1_loader, 0)
        labels = torch.cat(idx_loader, 0)
        # visualize(feat.data.cpu().numpy(),labels.data.cpu().numpy(),epoch)

        # epochごとのlossと正解率を表示
        epoch_loss = epoch_loss / total
        # epoch_acc = epoch_corrects / total

        history["loss"].append(epoch_loss)
        # history["acc"].append(epoch_acc)

        # earlystoppingの判定
        earlystopping(epoch_loss, model)
        if earlystopping.early_stop:
            print("Early Stopping on {} epoch.".format(epoch))
            break

        if (epoch+1) % 100 == 0:
            calc_loo_cv_score(model, X, y)

# def visualize(feat, labels, epoch):
#     plt.ion()
#     c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
#          '#ff00ff', '#990000', '#999900', '#009900', '#009999']
#     plt.clf()
#     for i in range(10):
#         plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
#     plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc = 'upper right')
#     plt.xlim(xmin=-8,xmax=8)
#     plt.ylim(ymin=-8,ymax=8)
#     plt.text(-7.8,7.3,"epoch=%d" % epoch)
#     # plt.savefig('./images/epoch=%d.jpg' % epoch)
#     plt.draw()
#     plt.pause(0.001)


if __name__ == '__main__':

    #　読み込み
    datapath = "Data/"
    df_train = pd.read_csv(datapath+"train_m.csv")
    df_test = pd.read_csv(datapath+"test_m.csv")

    # 前処理
    X, y = preprocess(df_train)
    # dataloader = make_dataloader(X, y, 100)

    # モデル呼び出し
    data_dim = X.shape[1]
    num_classes = 11
    model = metric_Net(data_dim, num_classes)

    # 最適アルゴリズム
    lr = 0.001
    beta1, beta2 = 0.0, 0.9
    optimizer = torch.optim.Adam(model.parameters(), lr, [beta1, beta2])

    # 損失関数
    # criterion = nn.CrossEntropyLoss(reduction='mean')
    criterion = nn.NLLLoss()
    # criterion = SelfAdjDiceLoss()

    # 学習回数
    num_epochs = 10000

    # earlystopping用のmodel保存先
    model_path = 'Model/NN/metric.pth'

    # 学習
    train(model, X, y, num_epochs, model_path)

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
    # visualize_with_model(model, X, y)
