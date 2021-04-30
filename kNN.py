from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

INPUT = Path("Data")
df_train = pd.read_csv(INPUT / "train.csv")
df_test = pd.read_csv(INPUT / "test.csv")
df_sample_sub = pd.read_csv(INPUT / "sample_submit.csv", header=None)
df_sample_sub.columns = ["index", "genre"]
df_genre_labels = pd.read_csv(INPUT / "genre_labels.csv")

def merge_train_test(df_train, df_test):
    if "genre" not in df_test.columns.tolist():
        df_test["genre"] = -100
    res = pd.concat([df_train, df_test])
    res.reset_index(inplace=True, drop=True)
    return res

def split_train_test(df):
    df_train = df[df["genre"] != -100]
    df_test = df[df["genre"] == -100]
    df_train.reset_index(inplace=True, drop=True)
    df_test.reset_index(inplace=True, drop=True)
    return df_train, df_test

df = merge_train_test(df_train, df_test)

# feature engineering

df["genre_name"] = df["genre"].map(dict(df_genre_labels[["labels", "genre"]].values))

df["tempo"] = df["tempo"].map(lambda x: sum(map(int, x.split("-"))) / 2)

df = pd.concat([df, pd.get_dummies(df["region"]).rename(columns={"unknown": "region_unknown"})], axis=1)

df["num_nans"] = 0
for col in [
    "acousticness",
    "positiveness",
    "danceability",
    "energy",
    "liveness",
    "speechiness",
    "instrumentalness",
]:
    df["num_nans"] += df[col].isna()

# feature scaling

df["log_tempo"] = np.log(df["tempo"])
for col in [
    'popularity', 'duration_ms', 'acousticness',
    'positiveness', 'danceability', 'loudness', 'energy', 'liveness',
    'speechiness', 'instrumentalness', 'log_tempo', 'num_nans',
]:
    df["standardscaled_" + col] = StandardScaler().fit_transform(df[[col]])[:, 0]

df_train, df_test = split_train_test(df)
target = df_train["genre"]

# feature weights

features = [
   'region_A', 'region_B', 'region_C', 'region_D', 'region_E', 'region_F',
   'region_G', 'region_H', 'region_I', 'region_J', 'region_K', 'region_L',
   'region_M', 'region_N', 'region_O', 'region_P', 'region_Q', 'region_R',
   'region_S', 'region_T', 'region_unknown',
   'standardscaled_popularity', 'standardscaled_duration_ms',
   'standardscaled_acousticness', 'standardscaled_positiveness',
   'standardscaled_danceability', 'standardscaled_loudness',
   'standardscaled_energy', 'standardscaled_liveness',
   'standardscaled_speechiness', 'standardscaled_instrumentalness',
   'standardscaled_log_tempo', 'standardscaled_num_nans'
]

dict_feature_weights = {}

for col in [
    'region_A', 'region_B', 'region_C', 'region_D', 'region_E', 'region_F',
    'region_G', 'region_H', 'region_I', 'region_J', 'region_K', 'region_L',
    'region_M', 'region_N', 'region_O', 'region_P', 'region_Q', 'region_R',
    'region_S', 'region_T', 'region_unknown'
]:
    dict_feature_weights[col] = 100.0

for col in [
    'standardscaled_duration_ms',
    'standardscaled_acousticness', 'standardscaled_positiveness',
    'standardscaled_danceability', 'standardscaled_loudness',
    'standardscaled_energy', 'standardscaled_liveness',
    'standardscaled_speechiness', 'standardscaled_instrumentalness'
]:
    dict_feature_weights[col] = 1.0

dict_feature_weights["standardscaled_popularity"] = 8.0
dict_feature_weights["standardscaled_log_tempo"] = 0.001
dict_feature_weights["standardscaled_num_nans"] = 100.0

feature_weights = np.array([dict_feature_weights[col] for col in features])

# parameters

N_CLASSES = 11
n_neighbors = 1

# cross validation

def calc_loo_cv_score():  # leave-one-out
    model = KNeighborsClassifier(n_neighbors + 1)
    X = df_train[features].fillna(0.0).values * feature_weights
    model.fit(X, target)
    distances, indexes = model.kneighbors(X)
    distances = distances[:, 1:]
    indexes = indexes[:, 1:]
    labels = target.values[indexes]
    preds = np.array([np.bincount(labels_, distances_, N_CLASSES) for labels_, distances_ in zip(labels, distances)]).argmax(1)
    score = f1_score(target, preds, average="macro")
    print(f"f1_score={score}\n")
    print(classification_report(target, preds))
    
    
    def visualize_confusion_matrix(y_true,
                               pred_label,
                               height=.6,
                               labels=None):  # from https://signate.jp/competitions/449/discussions/lgbm-baseline-lb06240
        conf = confusion_matrix(y_true=y_true,
                                y_pred=pred_label,
                                normalize='true')

        n_labels = len(conf)
        size = n_labels * height
        fig, ax = plt.subplots(figsize=(size * 2, size * 1.5))
        sns.heatmap(conf, cmap='YlOrBr', ax=ax, annot=True, fmt='.2f')
        ax.set_ylabel('Label')
        ax.set_xlabel('Predict')

        if labels is not None:
            ax.set_yticklabels(labels)
            ax.set_xticklabels(labels)
            ax.tick_params('y', labelrotation=0)
            ax.tick_params('x', labelrotation=90)

        return fig

    fig = visualize_confusion_matrix(target, preds, labels=df_genre_labels["genre"])
    fig.show()

calc_loo_cv_score()

# prediction for test data

model = KNeighborsClassifier(n_neighbors)
X_train = df_train[features].fillna(0.0).values * feature_weights
X_test = df_test[features].fillna(0.0).values * feature_weights
model.fit(X_train, target)
df_test["prediction"] = model.predict(X_test)
df_submission = df_sample_sub.copy()
df_submission["genre"] = df_submission["index"].map(dict(df_test[["index", "prediction"]].values))
assert not df_submission["genre"].isna().any()
df_submission.head(10)

# make submission file
df_submission.to_csv("Submit/submission_kNN.csv", header=None, index=False)
