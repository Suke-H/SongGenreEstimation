import numpy as np
from sklearn.metrics import classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
import pandas as pd

def target_encoding_count(df, index, columns):
    df_pivot = df.groupby([index, *columns]).count().iloc[:, 0].reset_index().pivot(index=columns, columns=index).fillna(0)
    columns.remove(index)
    name = '_'.join(columns)
    df_pivot.rename(columns=lambda x:str(index)+str(x)+f'_groupby{name}', inplace = True)
    df_pivot.columns = [i[1] for i in df_pivot.columns]
    return df_pivot.reset_index()

#　trainデータ読み込み
datapath = "Data/"
df_train = pd.read_csv(datapath+"train_m.csv")

pivot1 = target_encoding_count(df_train, 'genre', ['region'])
print(pivot1)

