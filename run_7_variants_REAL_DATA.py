"""
7 VARIANTS ABLATION STUDY - WITH REAL DATA (FROM ARFF FILES)
Dung du lieu thuc tu cac file ARFF, khong synthetic
"""

import warnings
warnings.filterwarnings('ignore')

import gc
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import arff

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import f1_score, accuracy_score, hamming_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# ============================================================================
# CONFIG
# ============================================================================
IS_KAGGLE = Path('/kaggle/working').exists()
OUTPUTS_DIR = Path('/kaggle/working/outputs') if IS_KAGGLE else Path('./outputs')
CSI_DIR = OUTPUTS_DIR / 'csi_results'
CSI_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR_CANDIDATES = [
    Path('/kaggle/input'),
    Path('./data'),
    Path('.'),
]

DATASET_CONFIGS = {
    'Scene':           {'file': 'Scene.arff',           'num_labels': 6,  'label_location': 'last'},
    'Yeast':           {'file': 'Yeast.arff',           'num_labels': 14, 'label_location': 'first'},  
    'emotions':        {'file': 'emotions.arff',        'num_labels': 6,  'label_location': 'last'},
    'HumanPseAAC':     {'file': 'HumanPseAAC.arff',     'num_labels': 14, 'label_location': 'last'},
    'PlantPseAAC':     {'file': 'PlantPseAAC.arff',     'num_labels': 12, 'label_location': 'last'},
    'GpositivePseAAC': {'file': 'GpositivePseAAC.arff', 'num_labels': 4,  'label_location': 'last'},
    'VirusPseAAC':     {'file': 'VirusPseAAC.arff',     'num_labels': 6,  'label_location': 'last'},
    'Water-quality':   {'file': 'Water-quality.arff',   'num_labels': 14, 'label_location': 'first'},
    'CHD_49':          {'file': 'CHD_49.arff',          'num_labels': 6,  'label_location': 'first'},
}

N_CHAINS = 5  # Small for speed
HIDDEN = 356
DROPOUT = 0.2
EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3
SEED = 42
THRESHOLD = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# ============================================================================
# UTILITIES
# ============================================================================

def find_file(filename):
    """Tim file trong cac thu muc candidates"""
    for base in DATA_DIR_CANDIDATES:
        if not base.exists():
            continue
        for p in base.rglob('*'):
            if p.is_file() and p.name.lower() == filename.lower():
                return p
    return None

def load_arff(path, num_labels, label_location='last'):
    """Load ARFF file - copy from run1.py"""
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = df[col].str.decode('utf-8')
            except Exception:
                pass
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    if label_location == 'first':
        Y = (df.iloc[:, :num_labels].values > 0).astype('float32')
        X = df.iloc[:, num_labels:].values.astype('float32')
    else:
        X = df.iloc[:, :-num_labels].values.astype('float32')
        Y = (df.iloc[:, -num_labels:].values > 0).astype('float32')
    
    return X, Y

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

def compute_metrics(y_true, y_pred):
    return {
        'micro_f1': float(f1_score(y_true, y_pred, average='micro', zero_division=0)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'subset_acc': float(accuracy_score(y_true, y_pred)),
        'hamming': float(hamming_loss(y_true, y_pred)),
    }

# ============================================================================
# MODELS
# ============================================================================

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

# ============================================================================
# VARIANT 1: BR_Baseline
# ============================================================================

def variant_br_baseline(X_tr, Y_tr, X_va, Y_va):
    print("  [1/7] BR_Baseline...", end=' ', flush=True)
    start = time.time()
    
    model = OneVsRestClassifier(
        LogisticRegression(solver='lbfgs', max_iter=300, class_weight='balanced'),
        n_jobs=-1
    )
    model.fit(X_tr, Y_tr)
    Y_pred = model.predict(X_va)
    
    metrics = compute_metrics(Y_va, Y_pred)
    elapsed = time.time() - start
    
    print(f"F1={metrics['micro_f1']:.4f} ({elapsed:.1f}s)")
    return metrics

# ============================================================================
# VARIANT 2: ECC_Standard
# ============================================================================

def variant_ecc_standard(X_tr, Y_tr, X_va, Y_va, n_chains=N_CHAINS):
    print("  [2/7] ECC_Standard...", end=' ', flush=True)
    start = time.time()
    
    models = []
    for c in range(n_chains):
        cc = ClassifierChain(
            LogisticRegression(solver='lbfgs', max_iter=300, class_weight='balanced'),
            order='random',
            random_state=SEED + c * 100
        )
        cc.fit(X_tr, Y_tr)
        models.append(cc)
    
    chain_probs = [cc.predict(X_va) for cc in models]
    Y_pred = ((np.mean(chain_probs, axis=0)) >= THRESHOLD).astype(int)
    
    metrics = compute_metrics(Y_va, Y_pred)
    elapsed = time.time() - start
    
    print(f"F1={metrics['micro_f1']:.4f} ({elapsed:.1f}s)")
    return metrics

# ============================================================================
# VARIANT 3: NN_CC_Single
# ============================================================================

def variant_nn_cc_single(X_tr, Y_tr, X_va, Y_va):
    print("  [3/7] NN_CC_Single...", end=' ', flush=True)
    start = time.time()
    
    num_labels = Y_tr.shape[1]
    order = np.random.permutation(num_labels)
    X_current = torch.from_numpy(X_tr).float().to(device)
    Y_tr_t = torch.from_numpy(Y_tr).float().to(device)
    models = []
    
    for lbl_idx in order:
        m = StandardNNModel(X_current.shape[1]).to(device)
        opt = torch.optim.Adam(m.parameters(), lr=LR)
        ld = DataLoader(TensorDataset(X_current, Y_tr_t[:, lbl_idx]), batch_size=BATCH_SIZE, shuffle=True)
        
        for ep in range(1, EPOCHS + 1):
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
        models.append((lbl_idx, m))
    
    X_va_t = torch.from_numpy(X_va).float().to(device)
    X_curr = X_va_t.clone()
    Y_pred = np.zeros((X_va.shape[0], num_labels), dtype='float32')
    
    for lbl_idx, m in models:
        m.eval()
        with torch.no_grad():
            pp = m(X_curr)
            Y_pred[:, lbl_idx] = pp.squeeze(-1).cpu().numpy()
            X_curr = torch.cat([X_curr, pp], dim=-1)
    
    Y_pred = (Y_pred >= THRESHOLD).astype(int)
    metrics = compute_metrics(Y_va, Y_pred)
    elapsed = time.time() - start
    
    print(f"F1={metrics['micro_f1']:.4f} ({elapsed:.1f}s)")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics

# ============================================================================
# VARIANT 4: NN_ECC
# ============================================================================

def variant_nn_ecc(X_tr, Y_tr, X_va, Y_va, n_chains=N_CHAINS):
    print("  [4/7] NN_ECC...", end=' ', flush=True)
    start = time.time()
    
    num_labels = Y_tr.shape[1]
    all_chains = []
    
    for chain_idx in range(n_chains):
        order = np.random.permutation(num_labels)
        X_current = torch.from_numpy(X_tr).float().to(device)
        Y_tr_t = torch.from_numpy(Y_tr).float().to(device)
        chain_models = []
        
        for lbl_idx in order:
            m = StandardNNModel(X_current.shape[1]).to(device)
            opt = torch.optim.Adam(m.parameters(), lr=LR)
            ld = DataLoader(TensorDataset(X_current, Y_tr_t[:, lbl_idx]), batch_size=BATCH_SIZE, shuffle=True)
            
            for ep in range(1, EPOCHS + 1):
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
            chain_models.append((lbl_idx, m))
        
        all_chains.append(chain_models)
    
    X_va_t = torch.from_numpy(X_va).float().to(device)
    all_probs = []
    
    for chain_models in all_chains:
        X_curr = X_va_t.clone()
        chain_probs = np.zeros((X_va.shape[0], num_labels), dtype='float32')
        
        for lbl_idx, m in chain_models:
            m.eval()
            with torch.no_grad():
                pp = m(X_curr)
                chain_probs[:, lbl_idx] = pp.squeeze(-1).cpu().numpy()
                X_curr = torch.cat([X_curr, pp], dim=-1)
        
        all_probs.append(chain_probs)
    
    Y_pred = ((np.mean(all_probs, axis=0)) >= THRESHOLD).astype(int)
    metrics = compute_metrics(Y_va, Y_pred)
    elapsed = time.time() - start
    
    print(f"F1={metrics['micro_f1']:.4f} ({elapsed:.1f}s)")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics

# ============================================================================
# VARIANT 5: EDL_CC_Single
# ============================================================================

def variant_edl_cc_single(X_tr, Y_tr, X_va, Y_va):
    print("  [5/7] EDL_CC_Single...", end=' ', flush=True)
    start = time.time()
    
    num_labels = Y_tr.shape[1]
    order = np.random.permutation(num_labels)
    X_current = torch.from_numpy(X_tr).float().to(device)
    Y_tr_t = torch.from_numpy(Y_tr).float().to(device)
    models = []
    
    for lbl_idx in order:
        m = EDLModel(X_current.shape[1]).to(device)
        opt = torch.optim.Adam(m.parameters(), lr=LR)
        ld = DataLoader(TensorDataset(X_current, Y_tr_t[:, lbl_idx]), batch_size=BATCH_SIZE, shuffle=True)
        
        for ep in range(1, EPOCHS + 1):
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
        models.append((lbl_idx, m))
    
    X_va_t = torch.from_numpy(X_va).float().to(device)
    X_curr = X_va_t.clone()
    Y_pred = np.zeros((X_va.shape[0], num_labels), dtype='float32')
    
    for lbl_idx, m in models:
        m.eval()
        with torch.no_grad():
            ap = m(X_curr)
            pp, u = predict_edl_binary(ap)
            Y_pred[:, lbl_idx] = pp.squeeze(-1).cpu().numpy()
            X_curr = torch.cat([X_curr, pp, u], dim=-1)
    
    Y_pred = (Y_pred >= THRESHOLD).astype(int)
    metrics = compute_metrics(Y_va, Y_pred)
    elapsed = time.time() - start
    
    print(f"F1={metrics['micro_f1']:.4f} ({elapsed:.1f}s)")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics

# ============================================================================
# VARIANT 6: EDL_ECC_NoUnc
# ============================================================================

def variant_edl_ecc_nounc(X_tr, Y_tr, X_va, Y_va, n_chains=N_CHAINS):
    print("  [6/7] EDL_ECC_NoUnc...", end=' ', flush=True)
    start = time.time()
    
    num_labels = Y_tr.shape[1]
    all_chains = []
    
    for chain_idx in range(n_chains):
        order = np.random.permutation(num_labels)
        X_current = torch.from_numpy(X_tr).float().to(device)
        Y_tr_t = torch.from_numpy(Y_tr).float().to(device)
        chain_models = []
        
        for lbl_idx in order:
            m = EDLModel(X_current.shape[1]).to(device)
            opt = torch.optim.Adam(m.parameters(), lr=LR)
            ld = DataLoader(TensorDataset(X_current, Y_tr_t[:, lbl_idx]), batch_size=BATCH_SIZE, shuffle=True)
            
            for ep in range(1, EPOCHS + 1):
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
            chain_models.append((lbl_idx, m))
        
        all_chains.append(chain_models)
    
    X_va_t = torch.from_numpy(X_va).float().to(device)
    all_probs = []
    
    for chain_models in all_chains:
        X_curr = X_va_t.clone()
        chain_probs = np.zeros((X_va.shape[0], num_labels), dtype='float32')
        
        for lbl_idx, m in chain_models:
            m.eval()
            with torch.no_grad():
                ap = m(X_curr)
                pp, _ = predict_edl_binary(ap)
                chain_probs[:, lbl_idx] = pp.squeeze(-1).cpu().numpy()
                X_curr = torch.cat([X_curr, pp], dim=-1)
        
        all_probs.append(chain_probs)
    
    Y_pred = ((np.mean(all_probs, axis=0)) >= THRESHOLD).astype(int)
    metrics = compute_metrics(Y_va, Y_pred)
    elapsed = time.time() - start
    
    print(f"F1={metrics['micro_f1']:.4f} ({elapsed:.1f}s)")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics

# ============================================================================
# VARIANT 7: EDL_ECC_Full
# ============================================================================

def variant_edl_ecc_full(X_tr, Y_tr, X_va, Y_va, n_chains=N_CHAINS):
    print("  [7/7] EDL_ECC_Full...", end=' ', flush=True)
    start = time.time()
    
    num_labels = Y_tr.shape[1]
    all_chains = []
    
    for chain_idx in range(n_chains):
        order = np.random.permutation(num_labels)
        X_current = torch.from_numpy(X_tr).float().to(device)
        Y_tr_t = torch.from_numpy(Y_tr).float().to(device)
        chain_models = []
        
        for lbl_idx in order:
            m = EDLModel(X_current.shape[1]).to(device)
            opt = torch.optim.Adam(m.parameters(), lr=LR)
            ld = DataLoader(TensorDataset(X_current, Y_tr_t[:, lbl_idx]), batch_size=BATCH_SIZE, shuffle=True)
            
            for ep in range(1, EPOCHS + 1):
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
            chain_models.append((lbl_idx, m))
        
        all_chains.append(chain_models)
    
    X_va_t = torch.from_numpy(X_va).float().to(device)
    all_probs = []
    
    for chain_models in all_chains:
        X_curr = X_va_t.clone()
        chain_probs = np.zeros((X_va.shape[0], num_labels), dtype='float32')
        
        for lbl_idx, m in chain_models:
            m.eval()
            with torch.no_grad():
                ap = m(X_curr)
                pp, u = predict_edl_binary(ap)
                chain_probs[:, lbl_idx] = pp.squeeze(-1).cpu().numpy()
                X_curr = torch.cat([X_curr, pp, u], dim=-1)
        
        all_probs.append(chain_probs)
    
    Y_pred = ((np.mean(all_probs, axis=0)) >= THRESHOLD).astype(int)
    metrics = compute_metrics(Y_va, Y_pred)
    elapsed = time.time() - start
    
    print(f"F1={metrics['micro_f1']:.4f} ({elapsed:.1f}s)")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics

# ============================================================================
# MAIN
# ============================================================================

def run_7_variants():
    print("\n" + "="*100)
    print("7 VARIANTS ABLATION STUDY - WITH REAL DATA (FROM ARFF)")
    print("="*100 + "\n")
    
    all_results = {}
    
    for ds_name, cfg in DATASET_CONFIGS.items():
        arff_path = find_file(cfg['file'])
        if arff_path is None or not arff_path.exists():
            print(f"[SKIP] {ds_name}: File not found")
            continue
        
        print(f"[{ds_name}] Loading data...")
        X, Y = load_arff(arff_path, cfg['num_labels'], cfg.get('label_location', 'last'))
        X = StandardScaler().fit_transform(X).astype('float32')
        
        print(f"  Shape: X={X.shape}, Y={Y.shape}")
        
        kf = KFold(n_splits=2, shuffle=True, random_state=SEED)
        
        ds_results = {
            'dataset': ds_name,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_labels': Y.shape[1],
            'folds': []
        }
        
        print(f"  Running 2-fold evaluation...\n")
        
        for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X)):
            print(f"  Fold {fold_idx + 1}:")
            
            X_tr, X_va = X[tr_idx], X[va_idx]
            Y_tr, Y_va = Y[tr_idx], Y[va_idx]
            
            # Ensure balanced labels
            for col in range(Y_tr.shape[1]):
                u = np.unique(Y_tr[:, col])
                if len(u) < 2:
                    if 0.0 not in u: Y_tr[0, col] = 0.0
                    if 1.0 not in u: Y_tr[0, col] = 1.0
            
            fold_results = {}
            
            # Run all 7 variants
            fold_results['BR'] = variant_br_baseline(X_tr, Y_tr, X_va, Y_va)
            fold_results['ECC'] = variant_ecc_standard(X_tr, Y_tr, X_va, Y_va, n_chains=N_CHAINS)
            fold_results['NN_CC_Single'] = variant_nn_cc_single(X_tr, Y_tr, X_va, Y_va)
            fold_results['NN_ECC'] = variant_nn_ecc(X_tr, Y_tr, X_va, Y_va, n_chains=N_CHAINS)
            fold_results['EDL_CC_Single'] = variant_edl_cc_single(X_tr, Y_tr, X_va, Y_va)
            fold_results['EDL_ECC_NoUnc'] = variant_edl_ecc_nounc(X_tr, Y_tr, X_va, Y_va, n_chains=N_CHAINS)
            fold_results['EDL_ECC_Full'] = variant_edl_ecc_full(X_tr, Y_tr, X_va, Y_va, n_chains=N_CHAINS)
            
            ds_results['folds'].append(fold_results)
            print()
            gc.collect()
        
        # Summary
        print(f"  SUMMARY:")
        summary = {}
        for variant in ['BR', 'ECC', 'NN_CC_Single', 'NN_ECC', 'EDL_CC_Single', 'EDL_ECC_NoUnc', 'EDL_ECC_Full']:
            f1_scores = [fold[variant]['micro_f1'] for fold in ds_results['folds']]
            mean_f1 = np.mean(f1_scores)
            std_f1 = np.std(f1_scores)
            summary[variant] = {'mean': float(mean_f1), 'std': float(std_f1)}
            print(f"    {variant:20s}: {mean_f1:.4f} +/- {std_f1:.4f}")
        
        ds_results['summary'] = summary
        all_results[ds_name] = ds_results
        
        print()
    
    # Save all results
    result_file = CSI_DIR / '7_variants_real_data.json'
    with open(result_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*100)
    print(f"== Saved: {result_file}")
    print("="*100)

if __name__ == '__main__':
    run_7_variants()
