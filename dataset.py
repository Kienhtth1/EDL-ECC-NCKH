"""
dataset.py  --  Đọc dữ liệu ARFF và các hàm tiền xử lý dữ liệu đa nhãn
"""

import numpy as np
import pandas as pd
from scipy.io import arff


def load_arff(path, num_labels):
    """
    Tải và phân tách ma trận đặc trưng X và nhãn đa chiều Y từ file ARFF.
    
    Args:
        path (Path or str): Đường dẫn tới file .arff.
        num_labels (int): Số lượng nhãn ở các cột cuối cùng của bảng dữ liệu.
        
    Returns:
        X (np.ndarray): Ma trận đặc trưng dạng float32.
        Y (np.ndarray): Ma trận nhãn đa chiều nhị phân (0 hoặc 1).
        label_names (list): Danh sách tên các nhãn.
    """
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
    """
    Đảm bảo tập huấn luyện có đủ cả 2 lớp (0 và 1) cho từng nhãn
    để tránh lỗi của các bộ phân loại nhị phân (như LogisticRegression).
    
    Args:
        Y_tr (np.ndarray): Ma trận nhãn của tập train (được chỉnh sửa in-place nếu cần).
    """
    for col in range(Y_tr.shape[1]):
        u = np.unique(Y_tr[:, col])
        if len(u) < 2:
            if 0.0 not in u:
                Y_tr[0, col] = 0.0
            if 1.0 not in u:
                Y_tr[0, col] = 1.0
