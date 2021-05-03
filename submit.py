from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import seaborn as sns
import lightgbm as lgb
import matplotlib.pyplot as plt

import torch

from nn import preprocess_test, preprocess, calc_loo_cv_score
from models import L2Softmax

datapath = "Data/"
# df_train = pd.read_csv(datapath+"train_m.csv")
df_test = pd.read_csv(datapath+"test_m.csv")
test_x = preprocess_test(df_test)
print(df_test)

#　trainデータ読み込み
df_train = pd.read_csv(datapath+"train_m.csv")
X, y = preprocess(df_train)

# モデル
data_dim = test_x.shape[1]
num_classes = 11
num_folds = 4
# model = Net(data_dim, num_classes)
# model_path = 'Model/NN/L2softmax.pth'
# model.load_state_dict(torch.load(model_path))

model_list = [L2Softmax(data_dim, num_classes) for i in range(num_folds)]
model_path_list = ['Model/NN/L2Softmax_' + str(i) + '.pth' for i in range(num_folds)]

# モデル呼び出し
for i in range(num_folds):
    model_list[i].load_state_dict(torch.load(model_path_list[i]))

# pred_list = np.zeros((num_folds, X_test.shape[0]))
output_list = np.zeros((num_folds, test_x.shape[0], num_classes))

### testデータを算出 ####################################

skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=15)
f1_list = np.zeros(num_folds)
oof = np.zeros(test_x.shape[0])
for fold, (indexes_trn, indexes_val) in enumerate(skf.split(X, y)):
    print(f"------------------------------ fold {fold} ------------------------------")

    X_train, y_train, X_val, y_val = X[indexes_trn], y[indexes_trn], X[indexes_val], y[indexes_val]
    _, oof[indexes_val], f1_list[fold] = calc_loo_cv_score(model_list[fold], X_val, y_val)

    # 出力
    output_list[fold], _ = calc_loo_cv_score(model_list[fold], test_x)

print("CV: {}".format(np.mean(f1_list)))

# 平均値
y_out = np.mean(output_list, axis=0)

# 最終予測
y_preds = np.argmax(y_out, axis=1)

def test(model, X, df):
    # GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval() # 推論モードへ
    X_tensor = torch.from_numpy(X).to(device)

    # 出力
    outputs, features = model(X_tensor)
    _, preds = torch.max(outputs, 1)
    preds = preds.to('cpu').detach().numpy().copy()

    df['genre'] = preds

    return df

df_sample_sub = pd.read_csv(datapath+"sample_submit.csv", header=None)
df_sample_sub.columns = ["index", "genre"]
df_submission = df_sample_sub.copy()
print(df_submission)

# df_main = test(model, test_x, df_test)
df_test['genre'] = y_preds
# print(df_main)

# df_submission["genre"] = df_submission["index"].map(dict(df_main[["index", "genre"]].values))
df_submission["genre"] = df_submission["index"].map(dict(df_test[["index", "genre"]].values))
print(df_submission)
# assert not df_submission["genre"].isna().any()

print("genre counts")
# display(df_submission["genre"].value_counts().sort_index())
print(df_submission["genre"].value_counts().sort_index())

print("\nfirst 10 test data")
# display(df_submission.head(10))
print(df_submission.head(10))

# make submission file
df_submission.to_csv("Submit/L2Softmax.csv", header=None, index=False)
df_submission.to_csv("Submit/L2Softmax_0.csv", header=None, index=False)