import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self, data_dim, num_classes):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(data_dim, 100)
        self.dropout1 = torch.nn.Dropout(p=0.5)

        self.fc2 = torch.nn.Linear(100, 1000)
        self.dropout2 = torch.nn.Dropout(p=0.5)

        self.fc3 = torch.nn.Linear(1000, 1000)
        self.dropout3 = torch.nn.Dropout(p=0.5)

        self.fc4 = torch.nn.Linear(1000, 100)
        self.dropout4 = torch.nn.Dropout(p=0.25)

        self.fc5 = torch.nn.Linear(100, 100)
        self.dropout5 = torch.nn.Dropout(p=0.25)

        self.fc6 = torch.nn.Linear(100, num_classes)
 
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
        features = self.dropout5(x)

        x = F.relu(self.fc6(features))
        return x, features

class metric_Net(torch.nn.Module):
    def __init__(self, data_dim, num_classes):
        super(metric_Net, self).__init__()
        self.fc1 = torch.nn.Linear(data_dim, 100)
        self.dropout1 = torch.nn.Dropout(p=0.5)

        self.fc2 = torch.nn.Linear(100, 1000)
        self.dropout2 = torch.nn.Dropout(p=0.5)

        self.fc3 = torch.nn.Linear(1000, 1000)
        self.dropout3 = torch.nn.Dropout(p=0.5)

        self.fc4 = torch.nn.Linear(1000, 100)
        self.dropout4 = torch.nn.Dropout(p=0.25)

        self.fc5 = torch.nn.Linear(100, 2)
        self.dropout5 = torch.nn.Dropout(p=0.25)

        self.fc6 = torch.nn.Linear(2, num_classes)
 
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

class EarlyStopping:
    """earlystoppingクラス"""

    def __init__(self, patience=5, verbose=False, path='Model/checkpoint_model.pth'):
        """引数：最小値の非更新数カウンタ、表示設定、モデル格納path"""

        self.patience = patience    #設定ストップカウンタ
        self.verbose = verbose      #表示の有無
        self.counter = 0            #現在のカウンタ値
        self.best_score = None      #ベストスコア
        self.early_stop = False     #ストップフラグ
        self.val_loss_min = np.Inf   #前回のベストスコア記憶用
        self.path = path             #ベストモデル格納path

    def __call__(self, val_loss, model):
        """
        特殊(call)メソッド
        実際に学習ループ内で最小lossを更新したか否かを計算させる部分
        """
        score = -val_loss

        if self.best_score is None:  #1Epoch目の処理
            self.best_score = score   #1Epoch目はそのままベストスコアとして記録する
            self.checkpoint(val_loss, model)  #記録後にモデルを保存してスコア表示する
        elif score < self.best_score:  # ベストスコアを更新できなかった場合
            self.counter += 1   #ストップカウンタを+1
            # if self.verbose:  #表示を有効にした場合は経過を表示
            #     print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  #現在のカウンタを表示する 
            if self.counter >= self.patience:  #設定カウントを上回ったらストップフラグをTrueに変更
                self.early_stop = True
        else:  #ベストスコアを更新した場合
            self.best_score = score  #ベストスコアを上書き
            self.checkpoint(val_loss, model)  #モデルを保存してスコア表示
            self.counter = 0  #ストップカウンタリセット

    def checkpoint(self, val_loss, model):
        '''ベストスコア更新時に実行されるチェックポイント関数'''
        if self.verbose:  #表示を有効にした場合は、前回のベストスコアからどれだけ更新したか？を表示
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}) on {self.counter} out of {self.patience}.  Saving model ...')
        torch.save(model.state_dict(), self.path)  #ベストモデルを指定したpathに保存
        self.val_loss_min = val_loss  #その時のlossを記録する


