import numpy as np
import pandas as pd
from scipy.io import arff


def load_arff(path, num_labels):
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = df[col].str.decode('utf-8')
            except Exception:
                pass
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    X = df.iloc[:, :-num_labels].values.astype('float32')
    Y = (df.iloc[:, -num_labels:].values > 0).astype('float32')
    label_names = list(df.columns[-num_labels:])
    return X, Y, label_names


def ensure_binary_classes(Y_tr):
    for col in range(Y_tr.shape[1]):
        u = np.unique(Y_tr[:, col])
        if len(u) < 2:
            if 0.0 not in u:
                Y_tr[0, col] = 0.0
            if 1.0 not in u:
                Y_tr[0, col] = 1.0
