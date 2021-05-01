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

from models import Net

def mean_norm(df):
    """ 標準化 """
    return df.apply(lambda x: (x-x.mean())/ x.std(), axis=0)

def preprocess(df_test):
    """ 前処理  """

    df = df_test.copy()

    # データと正解ラベルを分割
    X = df.drop(['index'], axis=1)

    # 標準化
    X = mean_norm(X)

    return X.values.astype(np.float32)

datapath = "Data/"
# df_train = pd.read_csv(datapath+"train_m.csv")
df_test = pd.read_csv(datapath+"test_m.csv")
test_x = preprocess(df_test)
print(df_test)

# モデル
data_dim = test_x.shape[1]
num_classes = 11
model = Net(data_dim, num_classes)
model_path = 'Model/NN/L2softmax.pth'
model.load_state_dict(torch.load(model_path))

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

df_main = test(model, test_x, df_test)
print(df_main)

df_submission["genre"] = df_submission["index"].map(dict(df_main[["index", "genre"]].values))
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