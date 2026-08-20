import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from scipy.io import arff
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def dirichlet_kl_multiclass(alpha):
    S = torch.sum(alpha, dim=-1, keepdim=True)
    C = alpha.shape[-1]
    
    lnB = torch.lgamma(S) - torch.sum(torch.lgamma(alpha), dim=-1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(torch.ones_like(alpha)), dim=-1, keepdim=True) - torch.lgamma(torch.ones_like(S) * C)
    
    dg0 = torch.digamma(S)
    dg1 = torch.digamma(alpha)
    
    kl = lnB - lnB_uni + torch.sum((alpha - 1.0) * (dg1 - dg0), dim=-1, keepdim=True)
    return kl.squeeze(-1)

def edl_multiclass_mse_loss(alpha, target_class, epoch, C, annealing_step=5):
    S = torch.sum(alpha, dim=-1, keepdim=True)
    p = alpha / S
    y_onehot = F.one_hot(target_class, num_classes=C).float().to(alpha.device)
    
    mse = torch.sum((y_onehot - p) ** 2, dim=-1)
    var_term = torch.sum(p * (1.0 - p) / (S + 1.0), dim=-1)
    
    kl = dirichlet_kl_multiclass(alpha)
    lambda_t = min(1.0, epoch / max(1, annealing_step))
    
    return (mse + var_term + lambda_t * kl).mean()

class EDL_LP_Module(nn.Module):
    def __init__(self, in_dim, num_classes, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_classes)
        )
    
    def forward(self, x):
        evidence = F.relu(self.net(x))
        alpha = evidence + 1.0 + 1e-4
        return alpha

class Deep_Binary_Model:
    def __init__(self, in_dim, hidden=128, epochs=10, lr=1e-3):
        self.in_dim = in_dim
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.model = None

    def fit(self, X, y):
        self.model = EDL_LP_Module(self.in_dim, 2, self.hidden).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        y_t = torch.tensor(y, dtype=torch.long).to(device)
        
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        self.model.train()
        for epoch in range(1, self.epochs + 1):
            for xb, yb in loader:
                optimizer.zero_grad()
                alpha = self.model(xb)
                loss = edl_multiclass_mse_loss(alpha, yb, epoch, 2)
                loss.backward()
                optimizer.step()
        return self

    def predict_alpha(self, X):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            alpha = self.model(X_t)
        return alpha.cpu().numpy()

class Deep_BR:
    def __init__(self, epochs=10):
        self.epochs = epochs
        self.models_ = []
        
    def fit(self, X, Y):
        self.models_ = []
        n_features = X.shape[1]
        n_labels = Y.shape[1]
        for k in range(n_labels):
            mod = Deep_Binary_Model(n_features, epochs=self.epochs).fit(X, Y[:, k])
            self.models_.append(mod)
        return self
            
    def predict(self, X):
        preds = np.zeros((X.shape[0], len(self.models_)))
        for k, mod in enumerate(self.models_):
            alpha = mod.predict_alpha(X)
            S = np.sum(alpha, axis=1, keepdims=True)
            p = alpha / S
            preds[:, k] = (p[:, 1] > 0.5).astype(int)
        return preds

class Deep_EDL_CC:
    def __init__(self, order=None, beta=1.0, use_gate=True, epochs=10):
        self.order = order
        self.beta = beta
        self.use_gate = use_gate
        self.epochs = epochs
        self.models_ = []
        
    def fit(self, X, Y):
        self.models_ = []
        n_features = X.shape[1]
        n_labels = Y.shape[1]
        
        if self.order is None:
            self.order_ = list(range(n_labels))
        else:
            self.order_ = self.order
            
        X_current = X.copy()
        
        for k in self.order_:
            mod = Deep_Binary_Model(X_current.shape[1], epochs=self.epochs).fit(X_current, Y[:, k])
            self.models_.append(mod)
            
            alpha = mod.predict_alpha(X_current)
            S = np.sum(alpha, axis=1, keepdims=True)
            p = alpha / S
            p_k = p[:, 1]
            u_k = (2.0 / S).flatten()  # Uncertainty = K/S (K=2)
            
            if self.use_gate:
                p_out = p_k * np.exp(-self.beta * u_k)
                X_current = np.hstack([X_current, p_out.reshape(-1, 1), u_k.reshape(-1, 1)])
            else:
                p_out = (p_k > 0.5).astype(float)
                X_current = np.hstack([X_current, p_out.reshape(-1, 1)])
                
        return self

    def predict(self, X):
        X_current = X.copy()
        n_labels = len(self.order_)
        preds = np.zeros((X.shape[0], n_labels))
        
        for idx, k in enumerate(self.order_):
            mod = self.models_[idx]
            alpha = mod.predict_alpha(X_current)
            S = np.sum(alpha, axis=1, keepdims=True)
            p = alpha / S
            p_k = p[:, 1]
            u_k = (2.0 / S).flatten()
            
            preds[:, k] = (p_k > 0.5).astype(int)
            
            if self.use_gate:
                p_out = p_k * np.exp(-self.beta * u_k)
                X_current = np.hstack([X_current, p_out.reshape(-1, 1), u_k.reshape(-1, 1)])
            else:
                p_out = (p_k > 0.5).astype(float)
                X_current = np.hstack([X_current, p_out.reshape(-1, 1)])
                
        return preds

class Deep_EDL_ECC:
    def __init__(self, n_chains=3, beta=1.0, epochs=10):
        self.n_chains = n_chains
        self.beta = beta
        self.epochs = epochs
        self.chains = []
        
    def fit(self, X, Y):
        self.chains = []
        n_labels = Y.shape[1]
        for i in range(self.n_chains):
            order = np.random.permutation(n_labels).tolist()
            chain = Deep_EDL_CC(order=order, beta=self.beta, use_gate=True, epochs=self.epochs)
            chain.fit(X, Y)
            self.chains.append(chain)
        return self
        
    def predict(self, X):
        preds_sum = np.zeros((X.shape[0], self.chains[0].order_.__len__()))
        for chain in self.chains:
            preds_sum += chain.predict(X)
        return (preds_sum >= (self.n_chains / 2.0)).astype(int)

if __name__ == '__main__':
    datasets = {
        'CHD_49': 6,
        'emotions': 6,
        'scene': 6,
        'Yeast': 14,
        'Water-quality': 14,
        'HumanPseAAC': 14,
        'GpositivePseAAC': 4,
        'PlantPseAAC': 12,
        'VirusPseAAC': 6
    }
    data_dir = Path('/home/niektran/Downloads/pre-order-for-mlc/data')
    
    def load_arff(path, num_labels):
        data, meta = arff.loadarff(path)
        df = pd.DataFrame(data)
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    df[col] = df[col].str.decode('utf-8')
                except:
                    pass
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        X = df.iloc[:, :-num_labels].values.astype('float32')
        Y = df.iloc[:, -num_labels:].values.astype('float32')
        Y = (Y > 0).astype('float32')
        return X, Y
    
    all_results = []
    
    print(f"Bắt đầu huấn luyện Bóc tách (Ablation) bằng Deep Learning (PyTorch) trên thiết bị: {device}")
    
    for ds_name, num_labels in datasets.items():
        print(f"\n==========================================")
        print(f"Đang huấn luyện tập dữ liệu: {ds_name}")
        print(f"==========================================")
        try:
            path = data_dir / f"{ds_name}.arff"
            if not path.exists():
                path = data_dir / ds_name / f"{ds_name}.arff"
            X, Y = load_arff(path, num_labels)
        except Exception as e:
            print(f"Lỗi tải dữ liệu {ds_name}: {e}")
            continue
            
        X = StandardScaler().fit_transform(X)
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = {'Deep-BR': [], 'Deep-CC': [], 'Deep-EDL+CC': [], 'Deep-EDL+ECC': []}
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"  Fold {fold+1}/5...")
            X_train, Y_train = X[train_idx], Y[train_idx]
            X_val, Y_val = X[val_idx], Y[val_idx]
            
            # 1. BR
            br = Deep_BR(epochs=10).fit(X_train, Y_train)
            pred = br.predict(X_val)
            scores['Deep-BR'].append(f1_score(Y_val, pred, average='micro', zero_division=0))
            
            # 2. CC
            cc = Deep_EDL_CC(use_gate=False, epochs=10).fit(X_train, Y_train)
            pred = cc.predict(X_val)
            scores['Deep-CC'].append(f1_score(Y_val, pred, average='micro', zero_division=0))
            
            # 3. EDL-CC
            edl_cc = Deep_EDL_CC(use_gate=True, epochs=10).fit(X_train, Y_train)
            pred = edl_cc.predict(X_val)
            scores['Deep-EDL+CC'].append(f1_score(Y_val, pred, average='micro', zero_division=0))
            
            # 4. EDL-ECC
            edl_ecc = Deep_EDL_ECC(n_chains=3, epochs=10).fit(X_train, Y_train)
            pred = edl_ecc.predict(X_val)
            scores['Deep-EDL+ECC'].append(f1_score(Y_val, pred, average='micro', zero_division=0))
            
        for k, v in scores.items():
            print(f"    {k} Mean Micro-F1: {np.mean(v):.4f}")
            
        all_results.append({
            'Dataset': ds_name,
            'Deep-BR': np.mean(scores['Deep-BR']),
            'Deep-CC': np.mean(scores['Deep-CC']),
            'Deep-EDL+CC': np.mean(scores['Deep-EDL+CC']),
            'Deep-EDL+ECC': np.mean(scores['Deep-EDL+ECC'])
        })
    
    df_res = pd.DataFrame(all_results)
    df_res.to_csv('/home/niektran/Downloads/EDL_ECC_Project/ablation_dl_results.csv', index=False)
    print("\nHoàn tất! Bảng kết quả đã được lưu tại ablation_dl_results.csv")
