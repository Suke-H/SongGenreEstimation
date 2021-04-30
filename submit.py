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


# INPUT = Path("../input")
INPUT = Path("Data")
df_main = pd.read_csv(INPUT / "df_main_0.csv")

df_sample_sub = pd.read_csv(INPUT / "sample_submit.csv", header=None)
df_sample_sub.columns = ["index", "genre"]

df_submission = df_sample_sub.copy()

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