"""
PHÂN TÍCH SÂU: TẠI SAO EDL-ECC HIỆU QUẢ HƠN?

Ablation Study - So sánh các thành phần:
1. BR (Linear + Independent)
2. NN-ECC (Neural Network + Chain Dependency)
3. EDL-ECC NoUncertainty (EDL nhưng không dùng uncertainty)
4. EDL-ECC Full (EDL + Uncertainty + Chain)

Kết quả sẽ cho thấy:
- Đóng góp của Deep Learning
- Đóng góp của EDL framework
- Đóng góp của Uncertainty Quantification
- Đóng góp của Chain Dependency
"""

import warnings
warnings.filterwarnings('ignore')

import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import arff

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import f1_score, accuracy_score, hamming_loss, jaccard_score, confusion_matrix

# ============================================================================
# CONFIG
# ============================================================================
IS_KAGGLE   = Path('/kaggle/working').exists()
OUTPUTS_DIR = Path('/kaggle/working/outputs') if IS_KAGGLE else Path('./outputs')
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

DEEP_ANALYSIS_DIR = OUTPUTS_DIR / 'deep_analysis'
DEEP_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR_CANDIDATES = [Path('/kaggle/input'), Path('./data'), Path('.')]

DATASET_CONFIGS = {
    'Scene': {'file': 'Scene.arff', 'num_labels': 6, 'label_location': 'last'},
}

N_CHAINS, HIDDEN, DROPOUT = 5, 256, 0.2  # Reduced for speed
EPOCHS, BATCH_SIZE, LR, SEED, THRESHOLD = 50, 64, 1e-3, 42, 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# ============================================================================
# MODELS
# ============================================================================

def dirichlet_kl_binary(alpha):
    beta = torch.ones_like(alpha)
    S_alpha = alpha.sum(dim=-1, keepdim=True)
    S_beta = beta.sum(dim=-1, keepdim=True)
    lnB_a = torch.lgamma(alpha).sum(-1, keepdim=True) - torch.lgamma(S_alpha)
    lnB_b = torch.lgamma(beta).sum(-1, keepdim=True) - torch.lgamma(S_beta)
    dg_diff = torch.digamma(alpha) - torch.digamma(S_alpha)
    return ((alpha - beta) * dg_diff).sum(-1, keepdim=True).squeeze(-1) + lnB_b.squeeze(-1) - lnB_a.squeeze(-1)

def edl_binary_mse_loss(alpha, target, epoch, annealing_step=5):
    S = alpha.sum(dim=-1, keepdim=True)
    p = alpha / S
    y = torch.stack([1.0 - target.float(), target.float()], dim=-1)
    mse = ((y - p) ** 2).sum(dim=-1)
    var_term = (p * (1.0 - p) / (S + 1.0)).sum(dim=-1)
    pos_w = torch.where(target > 0, 2.0, 1.0)
    alpha_tilde = y + (1.0 - y) * alpha
    kl = dirichlet_kl_binary(alpha_tilde)
    lambda_t = min(1.0, epoch / max(1, annealing_step))
    return ((mse + var_term) * pos_w + lambda_t * kl).mean()

def predict_edl_binary(alpha):
    S = alpha.sum(dim=-1, keepdim=True)
    p_pos = alpha[..., 1:2] / S
    u = 2.0 / S
    return p_pos, u

class EDLModel(nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden // 2, 2)

    def forward(self, x):
        h = self.net(x)
        e = F.relu(self.out(h)) + 1e-4
        return e + 1.0

class StandardNNModel(nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))

class EDL_ECC_Full:
    def __init__(self, in_dim, num_labels, n_chains=N_CHAINS):
        self.num_labels = num_labels
        self.n_chains = n_chains
        self.chains = []

    def fit(self, X_tr, Y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR):
        for _ in range(self.n_chains):
            order = np.random.permutation(self.num_labels)
            X_current = torch.from_numpy(X_tr).float().to(device)
            Y_tr_t = torch.from_numpy(Y_tr).float().to(device)
            mods = []
            for lbl_idx in order:
                m = EDLModel(X_current.shape[1]).to(device)
                opt = torch.optim.Adam(m.parameters(), lr=lr)
                ld = DataLoader(TensorDataset(X_current, Y_tr_t[:, lbl_idx]), batch_size=batch_size, shuffle=True)
                for ep in range(1, epochs + 1):
                    m.train()
                    for xb, yb in ld:
                        a = m(xb)
                        loss = edl_binary_mse_loss(a, yb, ep)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                m.eval()
                with torch.no_grad():
                    ap = m(X_current)
                    pp, u = predict_edl_binary(ap)
                    X_current = torch.cat([X_current, pp, u], dim=-1)
                mods.append((lbl_idx, m))
            self.chains.append(mods)

    def predict_proba(self, X_val):
        all_probs = []
        X_val_t = torch.from_numpy(X_val).float().to(device)
        for mods in self.chains:
            X_curr = X_val_t.clone()
            chain_p = np.zeros((X_val.shape[0], self.num_labels), dtype='float32')
            for lbl_idx, m in mods:
                m.eval()
                with torch.no_grad():
                    a = m(X_curr)
                    pp, u = predict_edl_binary(a)
                    chain_p[:, lbl_idx] = pp.squeeze(-1).cpu().numpy()
                    X_curr = torch.cat([X_curr, pp, u], dim=-1)
            all_probs.append(chain_p)
        return np.mean(all_probs, axis=0)

    def predict(self, X_val, threshold=THRESHOLD):
        return (self.predict_proba(X_val) >= threshold).astype(int)

class EDL_ECC_NoUncertainty:
    def __init__(self, in_dim, num_labels, n_chains=N_CHAINS):
        self.num_labels = num_labels
        self.n_chains = n_chains
        self.chains = []

    def fit(self, X_tr, Y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR):
        for _ in range(self.n_chains):
            order = np.random.permutation(self.num_labels)
            X_current = torch.from_numpy(X_tr).float().to(device)
            Y_tr_t = torch.from_numpy(Y_tr).float().to(device)
            mods = []
            for lbl_idx in order:
                m = EDLModel(X_current.shape[1]).to(device)
                opt = torch.optim.Adam(m.parameters(), lr=lr)
                ld = DataLoader(TensorDataset(X_current, Y_tr_t[:, lbl_idx]), batch_size=batch_size, shuffle=True)
                for ep in range(1, epochs + 1):
                    m.train()
                    for xb, yb in ld:
                        a = m(xb)
                        loss = edl_binary_mse_loss(a, yb, ep)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                m.eval()
                with torch.no_grad():
                    ap = m(X_current)
                    pp, _ = predict_edl_binary(ap)
                    X_current = torch.cat([X_current, pp], dim=-1)
                mods.append((lbl_idx, m))
            self.chains.append(mods)

    def predict_proba(self, X_val):
        all_probs = []
        X_val_t = torch.from_numpy(X_val).float().to(device)
        for mods in self.chains:
            X_curr = X_val_t.clone()
            chain_p = np.zeros((X_val.shape[0], self.num_labels), dtype='float32')
            for lbl_idx, m in mods:
                m.eval()
                with torch.no_grad():
                    a = m(X_curr)
                    pp, _ = predict_edl_binary(a)
                    chain_p[:, lbl_idx] = pp.squeeze(-1).cpu().numpy()
                    X_curr = torch.cat([X_curr, pp], dim=-1)
            all_probs.append(chain_p)
        return np.mean(all_probs, axis=0)

    def predict(self, X_val, threshold=THRESHOLD):
        return (self.predict_proba(X_val) >= threshold).astype(int)

class NN_ECC_Standard:
    def __init__(self, in_dim, num_labels, n_chains=N_CHAINS):
        self.num_labels = num_labels
        self.n_chains = n_chains
        self.chains = []

    def fit(self, X_tr, Y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR):
        for _ in range(self.n_chains):
            order = np.random.permutation(self.num_labels)
            X_current = torch.from_numpy(X_tr).float().to(device)
            Y_tr_t = torch.from_numpy(Y_tr).float().to(device)
            mods = []
            for lbl_idx in order:
                m = StandardNNModel(X_current.shape[1]).to(device)
                opt = torch.optim.Adam(m.parameters(), lr=lr)
                ld = DataLoader(TensorDataset(X_current, Y_tr_t[:, lbl_idx]), batch_size=batch_size, shuffle=True)
                for ep in range(1, epochs + 1):
                    m.train()
                    for xb, yb in ld:
                        pred = m(xb).squeeze(-1)
                        loss = F.binary_cross_entropy(pred, yb)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                m.eval()
                with torch.no_grad():
                    pp = m(X_current)
                    X_current = torch.cat([X_current, pp], dim=-1)
                mods.append((lbl_idx, m))
            self.chains.append(mods)

    def predict_proba(self, X_val):
        all_probs = []
        X_val_t = torch.from_numpy(X_val).float().to(device)
        for mods in self.chains:
            X_curr = X_val_t.clone()
            chain_p = np.zeros((X_val.shape[0], self.num_labels), dtype='float32')
            for lbl_idx, m in mods:
                m.eval()
                with torch.no_grad():
                    pp = m(X_curr)
                    chain_p[:, lbl_idx] = pp.squeeze(-1).cpu().numpy()
                    X_curr = torch.cat([X_curr, pp], dim=-1)
            all_probs.append(chain_p)
        return np.mean(all_probs, axis=0)

    def predict(self, X_val, threshold=THRESHOLD):
        return (self.predict_proba(X_val) >= threshold).astype(int)

# ============================================================================
# ANALYZER
# ============================================================================

class DeepAnalyzer:
    def compute_metrics(self, y_true, y_pred):
        return {
            'micro_f1': float(f1_score(y_true, y_pred, average='micro', zero_division=0)),
            'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            'subset_acc': float(accuracy_score(y_true, y_pred)),
            'hamming': float(hamming_loss(y_true, y_pred)),
            'jaccard': float(jaccard_score(y_true, y_pred, average='samples', zero_division=0)),
        }
    
    def label_wise_analysis(self, Y_true, Y_pred_br, Y_pred_edl, label_names):
        n_labels = Y_true.shape[1]
        improvements = []
        for i in range(n_labels):
            br_f1 = f1_score(Y_true[:, i], Y_pred_br[:, i], zero_division=0)
            edl_f1 = f1_score(Y_true[:, i], Y_pred_edl[:, i], zero_division=0)
            improvements.append({
                'label': label_names[i] if i < len(label_names) else f'L{i}',
                'br_f1': round(br_f1, 4),
                'edl_f1': round(edl_f1, 4),
                'delta': round(edl_f1 - br_f1, 4),
            })
        return sorted(improvements, key=lambda x: x['delta'], reverse=True)

# ============================================================================
# HELPERS
# ============================================================================

def find_file(filename):
    target_lower = filename.lower()
    for base in DATA_DIR_CANDIDATES:
        if not base.exists():
            continue
        for p in base.rglob('*'):
            if p.is_file() and p.name.lower() == target_lower:
                return p
    return None

def load_arff(path, num_labels, label_location=None):
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = df[col].str.decode('utf-8')
            except:
                pass
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    if label_location == 'first':
        Y = (df.iloc[:, :num_labels].values > 0).astype('float32')
        X = df.iloc[:, num_labels:].values.astype('float32')
        label_names = [str(c) for c in df.columns[:num_labels]]
    else:
        X = df.iloc[:, :-num_labels].values.astype('float32')
        Y = (df.iloc[:, -num_labels:].values > 0).astype('float32')
        label_names = [str(c) for c in df.columns[-num_labels:]]
    
    return X, Y, label_names

# ============================================================================
# MAIN
# ============================================================================

def run_deep_analysis():
    print("\n" + "="*90)
    print("ABLATION STUDY: Phân tích đóng góp từng thành phần")
    print("="*90 + "\n")
    
    analyzer = DeepAnalyzer()
    all_results = {}
    
    for ds_name, cfg in DATASET_CONFIGS.items():
        arff_path = find_file(cfg['file'])
        if not arff_path or not arff_path.exists():
            print(f"[SKIP] {ds_name}")
            continue
        
        X, Y, label_names = load_arff(arff_path, cfg['num_labels'], cfg.get('label_location', 'last'))
        X = StandardScaler().fit_transform(X).astype('float32')
        
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        
        fold_results = {'ablation': [], 'label_wise': []}
        
        print(f"[{ds_name}] {X.shape[0]} samples, {X.shape[1]} features, {Y.shape[1]} labels\n")
        
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
            X_tr, X_va = X[tr_idx], X[va_idx]
            Y_tr, Y_va = Y[tr_idx], Y[va_idx]
            
            for col in range(Y_tr.shape[1]):
                u = np.unique(Y_tr[:, col])
                if len(u) < 2:
                    if 0.0 not in u: Y_tr[0, col] = 0.0
                    if 1.0 not in u: Y_tr[0, col] = 1.0
            
            print(f"  Fold {fold+1}/5...", end=' ', flush=True)
            
            # BR (baseline)
            base_lr = LogisticRegression(solver='lbfgs', max_iter=300, class_weight='balanced')
            br = OneVsRestClassifier(base_lr, n_jobs=-1)
            br.fit(X_tr, Y_tr)
            br_pred = br.predict(X_va)
            br_metrics = analyzer.compute_metrics(Y_va, br_pred)
            
            # NN-ECC Standard
            print("NN", end=' ', flush=True)
            nn_ecc = NN_ECC_Standard(X_tr.shape[1], Y_tr.shape[1], n_chains=N_CHAINS)
            nn_ecc.fit(X_tr, Y_tr, epochs=EPOCHS)
            nn_pred = nn_ecc.predict(X_va)
            nn_metrics = analyzer.compute_metrics(Y_va, nn_pred)
            
            # EDL-ECC NoUncertainty
            print("EDL-NoUnc", end=' ', flush=True)
            edl_no_unc = EDL_ECC_NoUncertainty(X_tr.shape[1], Y_tr.shape[1], n_chains=N_CHAINS)
            edl_no_unc.fit(X_tr, Y_tr, epochs=EPOCHS)
            edl_no_unc_pred = edl_no_unc.predict(X_va)
            edl_no_unc_metrics = analyzer.compute_metrics(Y_va, edl_no_unc_pred)
            
            # EDL-ECC Full
            print("EDL-Full", end=' ', flush=True)
            edl_full = EDL_ECC_Full(X_tr.shape[1], Y_tr.shape[1], n_chains=N_CHAINS)
            edl_full.fit(X_tr, Y_tr, epochs=EPOCHS)
            edl_full_pred = edl_full.predict(X_va)
            edl_full_metrics = analyzer.compute_metrics(Y_va, edl_full_pred)
            
            fold_results['ablation'].append({
                'BR': br_metrics,
                'NN-ECC': nn_metrics,
                'EDL-NoUnc': edl_no_unc_metrics,
                'EDL-Full': edl_full_metrics,
            })
            
            label_wise = analyzer.label_wise_analysis(Y_va, br_pred, edl_full_pred, label_names)
            fold_results['label_wise'].append(label_wise)
            
            print(" ✓")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        all_results[ds_name] = fold_results
    
    save_results(all_results)

def save_results(all_results):
    with open(DEEP_ANALYSIS_DIR / 'ablation_study.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    
    # Create summary CSV
    rows = []
    for ds_name, results in all_results.items():
        for fold_idx, fold_ablation in enumerate(results['ablation']):
            for variant, metrics in fold_ablation.items():
                rows.append({
                    'Dataset': ds_name,
                    'Fold': fold_idx + 1,
                    'Model': variant,
                    'Micro-F1': round(metrics['micro_f1'], 4),
                    'Macro-F1': round(metrics['macro_f1'], 4),
                    'Subset-Acc': round(metrics['subset_acc'], 4),
                })
    
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(DEEP_ANALYSIS_DIR / 'ablation_summary.csv', index=False)
    
    # Aggregate by model
    print("\n" + "="*90)
    print("ABLATION STUDY RESULTS")
    print("="*90)
    agg = summary_df.groupby(['Dataset', 'Model'])[['Micro-F1', 'Macro-F1', 'Subset-Acc']].mean()
    print(agg.round(4))
    print(f"\nSaved to: {DEEP_ANALYSIS_DIR}")

if __name__ == '__main__':
    run_deep_analysis()
