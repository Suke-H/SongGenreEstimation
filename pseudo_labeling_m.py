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


warnings.simplefilter('ignore', pd.core.common.SettingWithCopyWarning)
warnings.simplefilter('ignore', UserWarning)


N_CLASSES = 11


# INPUT = Path("../input")
INPUT = Path("Data")
df_train = pd.read_csv(INPUT / "train_m.csv")
df_test = pd.read_csv(INPUT / "test_m.csv")
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


# parameters

# def lgb_metric(preds, data):  
#     pred_labels = preds.reshape(N_CLASSES, -1).argmax(axis=0)
#     score = f1_score(data.get_label(), pred_labels, average="macro")
#     return "macro_f1", score, True

learning_rate = 0.01

lgb_params = {
    "objective": "multiclass",
    "num_class": N_CLASSES,
    #"metric": "None",
    "learning_rate": learning_rate,
    "num_leaves": 3,
    "min_data_in_leaf": 40,
    #"colsample_bytree": 1.0,
    #"feature_fraction": 1.0,
    #"bagging_freq": 0,
    #"bagging_fraction": 1.0,
    "verbosity": 0,
    "seed": 42,
}

knn_n_neighbors = 6


# parameters - knn feature weights

knn_features = [
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

knn_feature_weights = np.array([dict_feature_weights[col] for col in knn_features])

df_main = merge_train_test(df_train, df_test)


########################################

for pseudo_labeling_threshold in [0.95, 0.925, 0.9, 0.875, 0.85, -np.inf]:
    df = df_main.copy()
    
    
    # feature engineering

    # df["genre_name"] = df["genre"].map(dict(df_genre_labels[["labels", "genre"]].values))

    # df["tempo"] = df["tempo"].map(lambda x: sum(map(int, x.split("-"))) / 2)

    # df = pd.concat([df, pd.get_dummies(df["region"]).rename(columns={"unknown": "region_unknown"})], axis=1)

    # df["num_nans"] = 0
    # for col in [
    #     "acousticness",
    #     "positiveness",
    #     "danceability",
    #     "energy",
    #     "liveness",
    #     "speechiness",
    #     "instrumentalness",
    # ]:
    #     df["num_nans"] += df[col].isna()

    # class CountEncoder:
    #     def fit(self, series):
    #         self.counts = series.groupby(series).count()
    #         return self

    #     def transform(self, series):
    #         return series.map(self.counts).fillna(0)

    #     def fit_transform(self, series):
    #         return self.fit(series).transform(series)
    # columns_count_enc = ["region"]
    # for col in columns_count_enc:
    #     df["countenc_" + col] = CountEncoder().fit_transform(df[col])
    #     df.loc[df[col].isna().values, "countenc_" + col] = np.nan


    # columns_label_enc = ["region"]
    # for col in columns_count_enc:
    #     df["labelenc_" + col] = LabelEncoder().fit_transform(df[col])
    #     df.loc[df[col].isna().values, "labelenc_" + col] = np.nan


    # class GroupFeatureExtractor:  # 参考: https://signate.jp/competitions/449/discussions/lgbm-baseline-lb06240
    #     EX_TRANS_METHODS = ["deviation", "zscore"]

    #     def __init__(self, group_key, group_values, agg_methods):
    #         self.group_key = group_key
    #         self.group_values = group_values

    #         self.ex_trans_methods = [m for m in agg_methods if m in self.EX_TRANS_METHODS]
    #         self.agg_methods = [m for m in agg_methods if m not in self.ex_trans_methods]
    #         self.df_agg = None

    #     def fit(self, df_train, y=None):
    #         if not self.agg_methods:
    #             return
    #         dfs = []
    #         for agg_method in self.agg_methods:
    #             if callable(agg_method):
    #                 agg_method_name = agg_method.__name__
    #             else:
    #                 agg_method_name = agg_method
    #             df_agg = (df_train[[self.group_key] + self.group_values].groupby(self.group_key).agg(agg_method))
    #             df_agg.columns = self._get_column_names(agg_method_name)
    #             dfs.append(df_agg)
    #         self.df_agg = pd.concat(dfs, axis=1).reset_index()

    #     def transform(self, df_eval):
    #         key = self.group_key
    #         if self.agg_methods:
    #             df_features = pd.merge(df_eval[[self.group_key]], self.df_agg, on=self.group_key, how="left")
    #         else:
    #             df_features = df_eval[[self.group_key]].copy()
    #         if self.ex_trans_methods:
    #             if "deviation" in self.ex_trans_methods:
    #                 df_features[self._get_agg_column_names("deviation")] = df_eval[self.group_values] - df_eval[[key]+self.group_values].groupby(key).transform("mean")
    #             if "zscore" in self.ex_trans_methods:
    #                 df_features[self._get_column_names("zscore")] = (df_eval[self.group_values] - df_eval[[key]+self.group_values].groupby(key).transform("mean")) \
    #                                                                 / (df_eval[[key]+self.group_values].groupby(key).transform("std") + 1e-8)
    #         df_features.drop(self.group_key, axis=1, inplace=True)
    #         return df_features

    #     def _get_column_names(self, method):
    #         return [f"agg_{method}_{col}_grpby_{self.group_key}" for col in self.group_values]

    #     def fit_transform(self, df_train, y=None):
    #         self.fit(df_train, y=y)
    #         return self.transform(df_train)   

    # df["log_tempo"] = np.log(df["tempo"])
    # gfe = GroupFeatureExtractor(
    #     "region", 
    #     ['popularity', 'duration_ms', 'acousticness', 'positiveness', 'danceability', 'loudness', 'energy', 'liveness', 'speechiness', 'instrumentalness', 'log_tempo'],
    #     ["zscore"]
    # )
    # df = pd.concat([df, gfe.fit_transform(df)], axis=1)


    # class KNNFeatureExtractor:
    #     def __init__(self, n_neighbors=5):
    #         self.knn = KNeighborsClassifier(n_neighbors + 1)

    #     def fit(self, X, y):
    #         self.knn.fit(X, y)
    #         self.y = y if isinstance(y, np.ndarray) else np.array(y)
    #         return self

    #     def transform(self, X, is_train_data):
    #         distances, indexes = self.knn.kneighbors(X)
    #         distances = distances[:, 1:] if is_train_data else distances[:, :-1]
    #         indexes = indexes[:, 1:] if is_train_data else indexes[:, :-1]
    #         labels = self.y[indexes]
    #         score_columns = [f"knn_score_class{c:02d}" for c in range(N_CLASSES)]
    #         df_knn = pd.DataFrame(
    #             [np.bincount(labels_, distances_, N_CLASSES) for labels_, distances_ in zip(labels, 1.0 / distances)],
    #             columns=score_columns
    #         )
    #         df_knn["max_knn_scores"] = df_knn.max(1)
    #         for col in score_columns:
    #             df_knn[f"sub_max_knn_scores_{col}"] = df_knn["max_knn_scores"] - df_knn[col]
    #         for i, col1 in enumerate(score_columns):
    #             for j, col2 in enumerate(score_columns[i+1:], i+1):
    #                 if {i, j} & {8, 10}:
    #                     df_knn[f"sub_{col1}_{col2}"] = df_knn[col1] - df_knn[col2]
    #         df_knn["sum_knn_scores"] = df_knn.sum(1)

    #         return df_knn


    # # feature scaling

    # df["log_tempo"] = np.log(df["tempo"])
    # for col in [
    #     'popularity', 'duration_ms', 'acousticness',
    #     'positiveness', 'danceability', 'loudness', 'energy', 'liveness',
    #     'speechiness', 'instrumentalness', 'log_tempo', 'num_nans',
    # ]:
    #     df["standardscaled_" + col] = StandardScaler().fit_transform(df[[col]])[:, 0]


    df_train, df_test = split_train_test(df)
    target = df_train["genre"]
    
    # train
    
    N_SPLITS = 15
    SEED_SKF = 42
    np.random.seed(42)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED_SKF)
    oof = np.zeros((len(df_train), N_CLASSES))
    predictions = np.zeros((len(df_test), N_CLASSES))
    df_feature_importance = pd.DataFrame()

    features_numerical = [
        'popularity', 'duration_ms', 'acousticness',
        'positiveness', 'danceability', 'loudness', 'energy', 'liveness',
        'speechiness', 'instrumentalness', 'tempo',
        'region_A', 'region_B', 'region_C', 'region_D', 'region_E', 'region_F',
        'region_G', 'region_H', 'region_I', 'region_J', 'region_K', 'region_L',
        'region_M', 'region_N', 'region_O', 'region_P', 'region_Q', 'region_R',
        'region_S', 'region_T', 'region_unknown', 'countenc_region',
        'num_nans',
        'agg_zscore_popularity_grpby_region',
        'agg_zscore_duration_ms_grpby_region',
        'agg_zscore_acousticness_grpby_region',
        'agg_zscore_positiveness_grpby_region',
        'agg_zscore_danceability_grpby_region',
        'agg_zscore_loudness_grpby_region', 'agg_zscore_energy_grpby_region',
        'agg_zscore_liveness_grpby_region',
        'agg_zscore_speechiness_grpby_region',
        'agg_zscore_instrumentalness_grpby_region',
        'agg_zscore_log_tempo_grpby_region',
        'knn_score_class00', 'knn_score_class01',
        'knn_score_class02', 'knn_score_class03', 'knn_score_class04',
        'knn_score_class05', 'knn_score_class06', 'knn_score_class07',
        'knn_score_class08', 'knn_score_class09', 'knn_score_class10',
        'max_knn_scores',
        'sub_max_knn_scores_knn_score_class00',
        'sub_max_knn_scores_knn_score_class01',
        'sub_max_knn_scores_knn_score_class02',
        'sub_max_knn_scores_knn_score_class03',
        'sub_max_knn_scores_knn_score_class04',
        'sub_max_knn_scores_knn_score_class05',
        'sub_max_knn_scores_knn_score_class06',
        'sub_max_knn_scores_knn_score_class07',
        'sub_max_knn_scores_knn_score_class08',
        'sub_max_knn_scores_knn_score_class09',
        'sub_max_knn_scores_knn_score_class10',
        'sub_knn_score_class00_knn_score_class08',
        'sub_knn_score_class00_knn_score_class10',
        'sub_knn_score_class01_knn_score_class08',
        'sub_knn_score_class01_knn_score_class10',
        'sub_knn_score_class02_knn_score_class08',
        'sub_knn_score_class02_knn_score_class10',
        'sub_knn_score_class03_knn_score_class08',
        'sub_knn_score_class03_knn_score_class10',
        'sub_knn_score_class04_knn_score_class08',
        'sub_knn_score_class04_knn_score_class10',
        'sub_knn_score_class05_knn_score_class08',
        'sub_knn_score_class05_knn_score_class10',
        'sub_knn_score_class06_knn_score_class08',
        'sub_knn_score_class06_knn_score_class10',
        'sub_knn_score_class07_knn_score_class08',
        'sub_knn_score_class07_knn_score_class10',
        'sub_knn_score_class08_knn_score_class09',
        'sub_knn_score_class08_knn_score_class10',
        'sub_knn_score_class09_knn_score_class10',
        'sum_knn_scores'
    ]
    features_categorical = ["labelenc_region"]
    features = features_numerical + features_categorical

    for fold_, (indexes_trn, indexes_val) in enumerate(skf.split(df_train.values, target.values)):
        print(f"------------------------------ fold {fold_} ------------------------------")

        df_trn = df_train.loc[indexes_trn].reset_index(drop=True).astype('float32')
        df_val = df_train.loc[indexes_val].reset_index(drop=True).astype('float32')
        target_trn = target.loc[indexes_trn].reset_index(drop=True)
        target_val = target.loc[indexes_val].reset_index(drop=True)

        # print(df_trn)
        # a = input()

        # df_trn = df_trn.drop(["index", "genre", "tempo"], axis=1)
        # df_val = df_val.drop(["index", "genre", "tempo"], axis=1)
        df_trn = df_trn.drop(["index", "genre"], axis=1)
        df_val = df_val.drop(["index", "genre"], axis=1)

        # make knn features
        # X = df_trn[knn_features].fillna(0.0).values * knn_feature_weights
        # knn_feature_extractor = KNNFeatureExtractor(knn_n_neighbors).fit(X, target_trn)
        # df_trn = pd.concat([df_trn, knn_feature_extractor.transform(X, is_train_data=True)], axis=1)
        # X = df_val[knn_features].fillna(0.0).values * knn_feature_weights
        # df_val = pd.concat([df_val, knn_feature_extractor.transform(X, is_train_data=False)], axis=1)
        # X = df_test[knn_features].fillna(0.0).values * knn_feature_weights
        # df_test_knn_features = knn_feature_extractor.transform(X, is_train_data=False)
        # for col in df_test_knn_features.columns:
        #     df_test[col] = df_test_knn_features[col]

        # lgb_train = lgb.Dataset(
        #     df_trn.loc[:, features],
        #     label=target_trn,
        #     feature_name=features,
        #     categorical_feature=features_categorical
        # )
        # lgb_valid = lgb.Dataset(
        #     df_val.loc[:, features],
        #     label=target_val,
        #     feature_name=features,
        #     categorical_feature=features_categorical
        # )

        #####
        lgb_train = lgb.Dataset(
            df_trn,
            label=target_trn,
            feature_name='auto'
        )
        lgb_valid = lgb.Dataset(
            df_val,
            label=target_val,
            feature_name='auto'
        )

        lgb_params["learning_rate"] = learning_rate + np.random.random() * 0.001  # おまじない
        num_round = 999999999
        model = lgb.train(
            lgb_params,
            lgb_train, 
            num_round, 
            valid_sets=[lgb_train, lgb_valid], 
            verbose_eval=300,
            early_stopping_rounds=300 if num_round >= 1e8 else None,
            fobj=None,
            #feval=lgb_metric,
        )

        # cv
        prediction_round = model.best_iteration+150 if num_round >= 1e8 else num_round  # おまじない
        # oof[indexes_val] = model.predict(df_val[features], num_iteration=prediction_round)
        #####
        oof[indexes_val] = model.predict(df_val, num_iteration=prediction_round)

        # feature importance
        # df_fold_importance = pd.DataFrame()
        # df_fold_importance["feature"] = features
        # df_fold_importance["importance"] = model.feature_importance()
        # df_fold_importance["fold"] = fold_
        # df_feature_importance = pd.concat([df_feature_importance, df_fold_importance], axis=0)

        # prediction for test data
        # predictions += model.predict(df_test[features], num_iteration=prediction_round) / N_SPLITS
        #####
        # print(df_test)
        # a = input()
        df_test_tmp = df_test.drop(["index", "genre"], axis=1).astype('float32')
        predictions += model.predict(df_test_tmp, num_iteration=prediction_round) / N_SPLITS
        print()
    
    score = f1_score(target, oof.argmax(1), average="macro")
    print("CV score (not reliable!)")
    print(f"  f1: {score:8.5f}")
    print()
    print(classification_report(target, oof.argmax(1)))
    
    df_test["prediction"] = predictions.argmax(1)
    df_test["confidence"] = predictions.max(1)
    df_test["genre"] = np.where(predictions.max(1) > pseudo_labeling_threshold, predictions.argmax(1), -100)
    df = merge_train_test(df_train, df_test)
    df_main["genre"] = df_main["index"].map(dict(df[["index", "genre"]].values))
    print((df_test["confidence"] > pseudo_labeling_threshold).sum(), f"rows were filled. (confidence>{pseudo_labeling_threshold})")
    print("filled test labels:", np.bincount(df_test[df_test["genre"]!=-100]["genre"]))
    print("\n")

########
df_submission = df_sample_sub.copy()
df_main.to_csv("df_main.csv")
print(df_main[["index", "genre"]])
a = input()
df_submission["genre"] = df_submission["index"].map(dict(df_main[["index", "genre"]].values))
print(df_submission)
assert not df_submission["genre"].isna().any()

print("genre counts")
# display(df_submission["genre"].value_counts().sort_index())
print(df_submission["genre"].value_counts().sort_index())

print("\nfirst 10 test data")
# display(df_submission.head(10))
print(df_submission.head(10))

# make submission file
df_submission.to_csv("Submit/submission_m.csv", header=None, index=False)