import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from tqdm import tqdm
import pyarrow as pa
from time import time
from collections import Counter
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt

datapath = "Data/ensemble/"
honda_train = pd.read_csv(datapath+"NN_train_softmax.csv").drop(['index'], axis=1).values
# df_honda_test = pd.read_csv(datapath+"NN_test_softmax.csv")
matsuda_train = pd.read_csv(datapath+"pred_train.csv").values
# df_matsuda_test = pd.read_csv(datapath+"pred_test.csv")
# honda_train = pd.read_csv(datapath+"NN_train_softmax.csv").values
# honda_test = pd.read_csv(datapath+"NN_test_softmax.csv").values
# matsuda_train = pd.read_csv(datapath+"pred_train.csv").values
matsuda_test = pd.read_csv(datapath+"pred_test.csv").values

# print(matsuda_train)
# print(matsuda_train.shape)
print(matsuda_test)
print(matsuda_test.shape)
# print(honda_train)
# print(honda_train.shape)


