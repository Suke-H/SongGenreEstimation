import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from  torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data

from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from loss import CenterLoss

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

class Net(torch.nn.Module):
    def __init__(self, data_dim, num_classes, feat_dim):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(data_dim, 100)
        self.dropout1 = torch.nn.Dropout(p=0.5)

        self.fc2 = torch.nn.Linear(100, 1000)
        self.dropout2 = torch.nn.Dropout(p=0.5)

        self.fc3 = torch.nn.Linear(1000, 1000)
        self.dropout3 = torch.nn.Dropout(p=0.5)

        self.fc4 = torch.nn.Linear(1000, 100)
        self.dropout4 = torch.nn.Dropout(p=0.25)

        self.fc5 = torch.nn.Linear(100, feat_dim)
        self.dropout5 = torch.nn.Dropout(p=0.25)

        self.fc6 = torch.nn.Linear(feat_dim, num_classes)
 
    def forward(self, x):
        # テンソルのリサイズ: (N, 1, 64, 64) -> (N, 64*64)
        # x = x.view(-1, 64 * 64)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = F.relu(self.fc3(x))
        x = self.dropout3(x)

        x = F.relu(self.fc4(x))
        x = self.dropout4(x)

        x = F.relu(self.fc5(x))
        ip1 = self.dropout5(x)

        x = F.relu(self.fc6(ip1))
        return ip1, x

def visualize(feat, labels, epoch):
    # plt.ion()
    # c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
    #      '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    # 20色のカラーマップ
    c = plt.cm.get_cmap('tab20').colors
    # plt.clf()
    for i in range(11):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], loc = 'upper right')
    plt.xlim(xmin=-8,xmax=8)
    plt.ylim(ymin=-8,ymax=8)
    plt.text(-7.8,7.3,"epoch=%d" % epoch)
    plt.savefig('images/epoch=%d.jpg' % epoch)
    # plt.draw()
    # plt.pause(0.001)
    plt.close()

def train(epoch, X, y):
    # print "Training... Epoch = %d" % epoch
    ip1_loader = []
    idx_loader = []

    train_loader = make_under_sampling_dataloader(X, y, 100)

    for i,(data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        ip1, pred = model(data)
        loss = nllloss(pred, target) + loss_weight * centerloss(target, ip1)

        optimizer4nn.zero_grad()
        optimzer4center.zero_grad()

        loss.backward()

        optimizer4nn.step()
        optimzer4center.step()

        ip1_loader.append(ip1)
        idx_loader.append((target))

    feat = torch.cat(ip1_loader, 0)
    labels = torch.cat(idx_loader, 0)
    # visualize(feat.data.cpu().numpy(),labels.data.cpu().numpy(),epoch)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#　読み込み
datapath = "Data/"
df_train = pd.read_csv(datapath+"train_m.csv")
df_test = pd.read_csv(datapath+"test_m.csv")

# 前処理
X, y = preprocess(df_train)

# Model
data_dim = X.shape[1]
num_classes = 11
feat_dim = 10
model = Net(data_dim, num_classes, feat_dim).to(device)

# NLLLoss
nllloss = nn.NLLLoss().to(device) #CrossEntropyLoss = log_softmax + NLLLoss
# CenterLoss
loss_weight = 1
centerloss = CenterLoss(num_classes, feat_dim).to(device)

# optimzer4nn
optimizer4nn = optim.SGD(model.parameters(),lr=0.001, momentum=0.9, weight_decay=0.0005)
# optimizer4nn = torch.optim.Adam(model.parameters(), 0.001, [0.0, 0.9])
sheduler = lr_scheduler.StepLR(optimizer4nn,20,gamma=0.8)

# optimzer4center
optimzer4center = optim.SGD(centerloss.parameters(), lr=0.5)
# optimzer4center = torch.optim.Adam(model.parameters(), 0.01, [0.0, 0.9])

for epoch in range(1000):
    sheduler.step()
    # print optimizer4nn.param_groups[0]['lr']
    train(epoch+1, X, y)

calc_loo_cv_score(model, X, y)