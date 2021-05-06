import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import torch.utils.data as data
from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay
from time import time

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

### モデルの定義 ##########################################

class Net(nn.Module):
    def __init__(self, data_dim, num_classes, \
                    trial, num_layers, num_units, dropouts):
        super(Net, self).__init__()

        # 活性化関数
        # self.activation = get_activation(trial)
        self.activation = F.relu

        # FC層
        self.fc = nn.ModuleList([])
        pre_units = data_dim
        for i in range(num_layers):
            self.fc.append(nn.Linear(pre_units, num_units[i]))
            pre_units = num_units[i]

        # Drouout
        self.dropouts = nn.ModuleList([])
        for i in range(num_layers):
            self.dropouts.append(nn.Dropout(p=dropouts[i]))

        # 最終層
        self.fc_last = nn.Linear(pre_units, num_classes)

        self.alpha = 16

    def forward(self, x):

        # FC層
        for (f, d) in zip(self.fc, self.dropouts):
            # print("f: {}\nd: {}".format(f, d))
            x = f(x)
            x = self.activation(x)
            x = d(x)

        # L2softmax層
        l2 = torch.sqrt((x**2).sum()) 
        feature = self.alpha * (x / l2)

        # 最終層
        x = self.fc_last(feature)

        return x, feature

### train  ############################################

def train(model, X_train, y_train, X_val, y_val, optimizer, criterion, num_epochs):
    """ 学習 """

    # ネットワークをGPUへ
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.train() # 訓練モードに
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

    model.eval() # 推論モードに

    # f1 macro算出
    outputs, _ = model(torch.from_numpy(X_val).to(device))
    _, preds = torch.max(outputs, 1)
    f1_macro = f1_score(y_val, preds.to('cpu').detach().numpy().copy(), average="macro")

    return -f1_macro

### 最適化アルゴリズム ##############################

def get_optimizer(trial, model):
    optimizer_names = ['Adam', 'MomentumSGD', 'rmsprop']
    optimizer_name = trial.suggest_categorical('optimizer', optimizer_names)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)

    # Adam
    if optimizer_name == optimizer_names[0]: 
        adam_lr = trial.suggest_loguniform('adam_lr', 1e-5, 1e-1)
        optimizer = optim.Adam(model.parameters(), lr=adam_lr, weight_decay=weight_decay)

    # MomentumSGD
    elif optimizer_name == optimizer_names[1]:
        momentum_sgd_lr = trial.suggest_loguniform('momentum_sgd_lr', 1e-5, 1e-1)
        optimizer = optim.SGD(model.parameters(), lr=momentum_sgd_lr, momentum=0.9, weight_decay=weight_decay)

    # RMSprop
    else:
        optimizer = optim.RMSprop(model.parameters())

    return optimizer

### パラメータチューニング用の目的関数 #############################

def objective(trial):

    EPOCH = 50 # 学習試行数
    data_dim = X.shape[1]
    num_classes = 11
    num_folds = 4

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # FC層の数（最終層を除く）
    num_layers = trial.suggest_int('num_layers', 3, 7)

    # FC層のユニット数
    num_units = [int(trial.suggest_discrete_uniform("num_units_"+str(i), 100, 1000, 100)) for i in range(num_layers)]

    # Dropoutの確率
    dropouts = [trial.suggest_discrete_uniform("dropout_"+str(i), 0, 0.5, 0.1) for i in range(num_layers)]

    model = Net(data_dim, num_classes, trial, num_layers, num_units, dropouts).to(device)
    # summary(model, input_size=(X.shape[1], ))

    # 最適アルゴリズム
    optimizer = get_optimizer(trial, model)

    # 損失関数
    criterion = nn.CrossEntropyLoss(reduction='mean')

    ### 学習（StratifiedKFoldを使用）#################################
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=15)
    f1_list = np.zeros(num_folds)

    for fold, (indexes_trn, indexes_val) in enumerate(skf.split(X, y)):
        X_train, y_train, X_val, y_val = X[indexes_trn], y[indexes_trn], X[indexes_val], y[indexes_val]

        # 学習
        for step in range(EPOCH):
            f1_list[fold] = train(model, X_train, y_train, X_val, y_val, optimizer, criterion, EPOCH)

    return np.mean(f1_list)

if __name__ == '__main__':

    # データ前処理
    datapath = "Data/"
    df = pd.read_csv(datapath+"train_m.csv")
    X, y = preprocess(df)

    # チューニング開始
    start = time()

    TRIAL_SIZE = 100 # チューニング試行数
    study = optuna.create_study()
    study.optimize(objective, n_trials=TRIAL_SIZE)
    print(study.best_params)

    end = time()
    print("time: {}m".format((end-start)/60))