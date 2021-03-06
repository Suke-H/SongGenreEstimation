import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class NN(torch.nn.Module):
    def __init__(self, data_dim, num_classes):
        super(NN, self).__init__()
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

        # x = F.relu(self.fc6(features))
        x = self.fc6(features)
        return x, features

class L2Softmax(torch.nn.Module):
    def __init__(self, data_dim, num_classes, alpha=16):
        super(L2Softmax, self).__init__()
        self.alpha = alpha

        self.fc1 = torch.nn.Linear(data_dim, 100)
        self.dropout1 = torch.nn.Dropout(p=0.5)

        self.fc2 = torch.nn.Linear(100, 1000)
        self.dropout2 = torch.nn.Dropout(p=0.5)

        self.fc3 = torch.nn.Linear(1000, 1000)
        self.dropout3 = torch.nn.Dropout(p=0.5)

        self.fc4 = torch.nn.Linear(1000, 100)
        self.dropout4 = torch.nn.Dropout(p=0.25)

        self.fc5 = torch.nn.Linear(100, 20)
        self.dropout5 = torch.nn.Dropout(p=0.25)

        self.fc6 = torch.nn.Linear(20, num_classes)

        # self.fc6 = torch.nn.Linear(20, 20)
        # self.dropout6 = torch.nn.Dropout(p=0.25)

        # self.fc7 = torch.nn.Linear(20, num_classes)
 
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
        x = self.dropout5(x)

        # L2softmax
        l2 = torch.sqrt((x**2).sum()) 
        feature = self.alpha * (x / l2)

        x = self.fc6(x)

        # x = F.relu(self.fc6(feature))
        # x = self.dropout6(x)

        # x = self.fc7(x)
        
        return x, feature
        # return x

class L2Softmax_tuning(nn.Module):
    def __init__(self, data_dim, num_classes, num_layers, num_units, dropouts, alpha):
        super(L2Softmax_tuning, self).__init__()

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

        self.alpha = alpha

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

class EarlyStopping:
    """earlystoppingクラス"""

    def __init__(self, patience=5, verbose=False, path='Model/checkpoint_model.pth', param_name="val loss"):
        """引数：最小値の非更新数カウンタ、表示設定、モデル格納path"""

        self.patience = patience    #設定ストップカウンタ
        self.verbose = verbose      #表示の有無
        self.counter = 0            #現在のカウンタ値
        self.best_score = None      #ベストスコア
        self.early_stop = False     #ストップフラグ
        self.val_loss_min = np.Inf   #前回のベストスコア記憶用
        self.path = path             #ベストモデル格納path
        self.param_name = param_name

    def __call__(self, val_loss, model, f1_score):
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
            if self.counter >= self.patience:  #設定カウントを上回ったらストップフラグをTrueに変更
                self.early_stop = True
        else:  #ベストスコアを更新した場合
            self.best_score = score  #ベストスコアを上書き
            self.checkpoint(val_loss, model)  #モデルを保存してスコア表示
            self.counter = 0  #ストップカウンタリセット

    def checkpoint(self, val_loss, model):
        '''ベストスコア更新時に実行されるチェックポイント関数'''
        if self.verbose:  #表示を有効にした場合は、前回のベストスコアからどれだけ更新したか？を表示
            print(f'{self.param_name} decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}) on {self.counter} out of {self.patience}.  Saving model ...')
        torch.save(model.state_dict(), self.path)  #ベストモデルを指定したpathに保存
        self.val_loss_min = val_loss  #その時のlossを記録する
