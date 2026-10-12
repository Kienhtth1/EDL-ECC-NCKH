"""
Ablation Study: Isolate Architecture vs Loss Function
======================================================
Compare 4 configurations:
1. ECC-LR:  Logistic Regression + Cross Entropy (baseline)
2. ECC-MLP: MLP [256,128] + Cross Entropy (architecture only)
3. EDL-LR:  Logistic Regression + EDL Loss (loss only) - optional
4. EDL-MLP: MLP [256,128] + EDL Loss (full proposed method)

Goal: Determine if improvement comes from MLP or EDL loss
"""

import numpy as np
import torch
import torch._utils  # Workaround: torch._dynamo needs torch._utils loaded explicitly (Kaggle, Python 3.13)
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, hamming_loss, jaccard_score
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# ============================================================================
# FIX for torch._dynamo AttributeError on Python 3.13 / Kaggle
# ============================================================================
# torch._dynamo.variables.torch tries to access torch._utils._get_device_index
# On Python 3.13, torch.__getattr__ fails to lazy-load torch._utils as an attribute.
# Solution: Manually attach _utils to torch module namespace.
import importlib
setattr(torch, '_utils', importlib.import_module('torch._utils'))
try:
    # Import dynamo now so any failure surfaces here with its real cause,
    # instead of inside torch.optim.Adam(...) later.
    import torch._dynamo  # noqa: F401
except Exception as _e:
    print(f"[warn] torch._dynamo import failed: {type(_e).__name__}: {_e}")
    print("[warn] If this mentions torch._utils, restart the Kaggle session "
          "(Run > Restart & clear outputs) so torch is not half-imported.")
# ============================================================================
from scipy.io import arff
import pandas as pd
import json
from pathlib import Path
import time
from tqdm import tqdm

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

CONFIG = {
    'datasets': [
        'CHD_49', 'emotions', 'GpositivePseAAC', 'HumanPseAAC',
        'PlantPseAAC', 'Scene', 'VirusPseAAC', 'Water-quality', 'Yeast'
    ],
    'data_dir': '/kaggle/input/datasets/fldragon/nckh-hvl',
    'output_dir': './outputs_ablation',
    'n_chains': 20,  # Match run1.py - EDL-MLP needs more chains for stability
    'n_folds': 5,   # 5-fold cross-validation on entire dataset
    'seed': 42,
    
    # MLP config
    'mlp_hidden': [256, 128],
    'mlp_dropout': 0.2,
    'mlp_epochs': 100,
    'mlp_lr': 1e-3,
    'mlp_batch_size': 64,
    
    # EDL config
    'edl_t_warmup': 5,
    'edl_w_pos': 2.0,
    
    # Logistic config
    'lr_max_iter': 1000,  # Increase to avoid convergence warnings
    'lr_solver': 'lbfgs',
}

SEED = CONFIG['seed']
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# 2. DATA LOADING
# ============================================================================

def load_arff_dataset(name, data_dir):
    """Load ARFF dataset (single file format for Kaggle)"""
    # Try single file first (e.g., Scene.arff)
    arff_path = Path(data_dir) / f"{name}.arff"
    
    if not arff_path.exists():
        # Try with folder structure (e.g., Scene/Scene.arff)
        arff_path = Path(data_dir) / name / f"{name}.arff"
    
    if not arff_path.exists():
        # Try train-test split format
        train_path = Path(data_dir) / name / f"{name}-train.arff"
        test_path = Path(data_dir) / name / f"{name}-test.arff"
        
        if train_path.exists() and test_path.exists():
            return _load_split_arff(train_path, test_path)
        else:
            raise FileNotFoundError(f"Cannot find ARFF file for dataset '{name}' in '{data_dir}'")
    
    # Load single ARFF file
    print(f"Loading single ARFF file: {arff_path}")
    with open(arff_path, 'r', encoding='utf-8') as f:
        data, meta = arff.loadarff(f)
    df = pd.DataFrame(data)
    
    # Decode bytes if needed
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = df[col].str.decode('utf-8')
            except:
                pass
    
    # Find label columns (last K columns with binary values)
    # Try to identify by checking unique values
    potential_label_cols = []
    for col in df.columns:
        unique_vals = set(df[col].dropna().unique())
        # Check if column contains only 0/1 or '0'/'1'
        if unique_vals <= {0, 1} or unique_vals <= {'0', '1'} or unique_vals <= {0.0, 1.0}:
            potential_label_cols.append(col)
    
    # Assume labels are at the end
    if len(potential_label_cols) > 0:
        label_cols = potential_label_cols
    else:
        # Fallback: assume last 6 columns are labels (adjust if needed)
        print(f"Warning: Could not identify label columns, assuming last 6 columns")
        label_cols = df.columns[-6:].tolist()
    
    feature_cols = [col for col in df.columns if col not in label_cols]
    
    print(f"  Features: {len(feature_cols)}, Labels: {len(label_cols)}")
    print(f"  Label columns: {label_cols}")
    
    # Convert to numpy
    X = df[feature_cols].astype(float).values
    Y = df[label_cols].astype(int).values
    
    print(f"  Data shape: X={X.shape}, Y={Y.shape}")
    
    return X, Y

def _load_split_arff(train_path, test_path):
    """Load split train/test ARFF files"""
    print(f"Loading split ARFF files: {train_path.name}, {test_path.name}")
    
    # Load train
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data, train_meta = arff.loadarff(f)
    df_train = pd.DataFrame(train_data)
    
    # Load test
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data, test_meta = arff.loadarff(f)
    df_test = pd.DataFrame(test_data)
    
    # Decode bytes
    for col in df_train.columns:
        if df_train[col].dtype == object:
            try:
                df_train[col] = df_train[col].str.decode('utf-8')
            except:
                pass
    for col in df_test.columns:
        if df_test[col].dtype == object:
            try:
                df_test[col] = df_test[col].str.decode('utf-8')
            except:
                pass
    
    # Find label columns
    potential_label_cols = []
    for col in df_train.columns:
        unique_vals = set(df_train[col].dropna().unique())
        if unique_vals <= {0, 1} or unique_vals <= {'0', '1'} or unique_vals <= {0.0, 1.0}:
            potential_label_cols.append(col)
    
    label_cols = potential_label_cols if potential_label_cols else df_train.columns[-6:].tolist()
    feature_cols = [col for col in df_train.columns if col not in label_cols]
    
    print(f"  Features: {len(feature_cols)}, Labels: {len(label_cols)}")
    
    # Convert to numpy
    X_train = df_train[feature_cols].astype(float).values
    Y_train = df_train[label_cols].astype(int).values
    X_test = df_test[feature_cols].astype(float).values
    Y_test = df_test[label_cols].astype(int).values
    
    # Combine for cross-validation
    X = np.vstack([X_train, X_test])
    Y = np.vstack([Y_train, Y_test])
    
    print(f"  Combined shape: X={X.shape}, Y={Y.shape}")
    
    return X, Y

# ============================================================================
# 3. MLP MODEL WITH CROSS ENTROPY
# ============================================================================

class MLPBinaryClassifier(nn.Module):
    """MLP for binary classification with Cross Entropy loss"""
    def __init__(self, in_dim, hidden=[256, 128], dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[1], 1)  # Single output for binary
        )
    
    def forward(self, x):
        logit = self.net(x)
        prob = torch.sigmoid(logit)
        return prob.squeeze(-1)

def train_mlp_ce(X_train, y_train, X_val, y_val, config):
    """Train MLP with Cross Entropy loss"""
    D = X_train.shape[1]
    model = MLPBinaryClassifier(
        in_dim=D,
        hidden=config['mlp_hidden'],
        dropout=config['mlp_dropout']
    ).to(DEVICE)
    
    # Dataset
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['mlp_batch_size'],
        shuffle=True
    )
    
    # Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config['mlp_lr'], weight_decay=1e-5)
    criterion = nn.BCELoss()
    
    # Class weights for imbalance
    pos_weight = config['edl_w_pos']
    
    # Training
    model.train()
    for epoch in range(config['mlp_epochs']):
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            # Forward
            probs = model(X_batch)
            
            # Weighted BCE
            loss = criterion(probs, y_batch)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    
    # Predict
    model.eval()
    with torch.no_grad():
        X_val_t = torch.FloatTensor(X_val).to(DEVICE)
        probs = model(X_val_t).cpu().numpy()
    
    return probs

# ============================================================================
# 4. MLP MODEL WITH EDL LOSS
# ============================================================================

class EDLBinaryClassifier(nn.Module):
    """MLP for binary classification with EDL loss"""
    def __init__(self, in_dim, hidden=[256, 128], dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[1], 2)  # 2 outputs for Dirichlet
        )
    
    def forward(self, x):
        z = self.net(x)
        # ReLU + small constant to ensure alpha > 0
        alpha = torch.relu(z) + 1e-4 + 1.0
        return alpha

def edl_binary_mse_loss(alpha, target, epoch, annealing_step=5):
    """EDL loss function - EXACT FROM run1.py"""
    S        = alpha.sum(dim=-1, keepdim=True)
    p        = alpha / S
    y        = torch.stack([1.0 - target.float(), target.float()], dim=-1)
    mse      = ((y - p) ** 2).sum(dim=-1)
    var_term = (p * (1.0 - p) / (S + 1.0)).sum(dim=-1)
    pos_w    = torch.where(target > 0, 2.0, 1.0)
    kl       = dirichlet_kl_binary(alpha)
    lambda_t = min(1.0, epoch / max(1, annealing_step))
    return ((mse + var_term) * pos_w + lambda_t * kl).mean()

class EDLLinearClassifier(nn.Module):
    """Linear (no hidden layer) evidential classifier: loss-only ablation."""
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Linear(in_dim, 2)  # 2 outputs for Dirichlet

    def forward(self, x):
        z = self.net(x)
        alpha = torch.relu(z) + 1e-4 + 1.0
        S = alpha.sum(dim=-1, keepdim=True)
        p = alpha[:, 1:2] / S
        u = 2.0 / S
        return p.squeeze(-1), u.squeeze(-1), alpha

def dirichlet_kl_binary(alpha):
    """KL Divergence between Dirichlet(alpha) and Dirichlet(1,1) - CORRECT VERSION"""
    beta = torch.ones_like(alpha)
    S_alpha = alpha.sum(dim=-1, keepdim=True)
    S_beta = beta.sum(dim=-1, keepdim=True)
    lnB_a = torch.lgamma(alpha).sum(-1, keepdim=True) - torch.lgamma(S_alpha)
    lnB_b = torch.lgamma(beta).sum(-1, keepdim=True) - torch.lgamma(S_beta)
    dg_diff = torch.digamma(alpha) - torch.digamma(S_alpha)
    return (
        ((alpha - beta) * dg_diff).sum(-1, keepdim=True).squeeze(-1)
        + lnB_b.squeeze(-1) - lnB_a.squeeze(-1)
    )

def edl_binary_mse_loss(alpha, target, epoch, annealing_step=5):
    """EDL Binary MSE + KL Regularization - CORRECT VERSION from run1.py"""
    S = alpha.sum(dim=-1, keepdim=True)
    p = alpha / S
    # One-hot encoding: [1-y, y]
    y = torch.stack([1.0 - target.float(), target.float()], dim=-1)
    # MSE loss for both classes
    mse = ((y - p) ** 2).sum(dim=-1)
    # Variance term
    var_term = (p * (1.0 - p) / (S + 1.0)).sum(dim=-1)
    # Weight positive samples more
    pos_w = torch.where(target > 0, 2.0, 1.0)
    # KL regularization
    kl = dirichlet_kl_binary(alpha)
    # Annealing
    lambda_t = min(1.0, epoch / max(1, annealing_step))
    
    return ((mse + var_term) * pos_w + lambda_t * kl).mean()

def train_lr_edl(X_train, y_train, X_val, y_val, config):
    """Train a linear model with EDL loss - same stabilization as train_mlp_edl"""
    D = X_train.shape[1]
    model = EDLLinearClassifier(D).to(DEVICE)

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=config['mlp_batch_size'],
        shuffle=True
    )
    # Optimizer with weight decay (stabilization)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['mlp_lr'], weight_decay=1e-5)

    model.train()
    for epoch in range(1, config['mlp_epochs'] + 1):
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            p, u, alpha = model(X_batch)
            loss = edl_binary_mse_loss(alpha, y_batch, epoch, config['edl_t_warmup'])
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        probs, uncertainty, _ = model(torch.FloatTensor(X_val).to(DEVICE))
    return probs.cpu().numpy(), uncertainty.cpu().numpy()

def train_mlp_edl(X_train, y_train, X_val, y_val, config):
    """Train MLP with EDL loss - CORRECT VERSION from run1.py"""
    D = X_train.shape[1]
    model = EDLBinaryClassifier(
        in_dim=D,
        hidden=config['mlp_hidden'],
        dropout=config['mlp_dropout']
    ).to(DEVICE)
    
    # Dataset
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['mlp_batch_size'],
        shuffle=True
    )
    
    # Optimizer with weight decay (stabilization for augmented features)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['mlp_lr'],
        weight_decay=1e-5
    )
    
    # Training
    model.train()
    for epoch in range(1, config['mlp_epochs'] + 1):
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            # Forward
            alpha = model(X_batch)
            
            # EDL Loss - CORRECT VERSION
            loss = edl_binary_mse_loss(alpha, y_batch, epoch, config['edl_t_warmup'])
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    
    # Predict
    model.eval()
    with torch.no_grad():
        X_val_t = torch.FloatTensor(X_val).to(DEVICE)
        alpha = model(X_val_t)
        S = alpha.sum(dim=-1, keepdim=True)
        probs = alpha[:, 1:2] / S
        uncertainty = 2.0 / S
        probs = probs.cpu().numpy()
        uncertainty = uncertainty.cpu().numpy()
    
    return probs.squeeze(-1), uncertainty.squeeze(-1)

# ============================================================================
# 5. ENSEMBLE CLASSIFIER CHAINS
# ============================================================================

def train_ecc_lr(X_train, Y_train, X_val, Y_val, n_chains, config):
    """ECC with Logistic Regression + Cross Entropy"""
    N_val, K = Y_val.shape
    all_preds = np.zeros((n_chains, N_val, K))
    
    for m in range(n_chains):
        order = np.random.permutation(K)
        preds = np.zeros((N_val, K))
        
        for i, j in enumerate(order):
            # Augmented features
            if i == 0:
                X_aug_train = X_train
                X_aug_val = X_val
            else:
                prev_labels = Y_train[:, order[:i]]
                X_aug_train = np.hstack([X_train, prev_labels])
                
                prev_preds = preds[:, order[:i]]
                X_aug_val = np.hstack([X_val, prev_preds])
            
            # Train Logistic Regression
            lr = LogisticRegression(
                solver=config['lr_solver'],
                max_iter=config['lr_max_iter'],
                class_weight='balanced'
            )
            lr.fit(X_aug_train, Y_train[:, j])
            
            # Predict probabilities
            preds[:, j] = lr.predict_proba(X_aug_val)[:, 1]
        
        all_preds[m] = preds
    
    # Average predictions
    final_preds = all_preds.mean(axis=0)
    return final_preds

def train_ecc_mlp_ce(X_train, Y_train, X_val, Y_val, n_chains, config):
    """ECC with MLP + Cross Entropy"""
    N_val, K = Y_val.shape
    all_preds = np.zeros((n_chains, N_val, K))
    
    for m in range(n_chains):
        order = np.random.permutation(K)
        preds = np.zeros((N_val, K))
        
        for i, j in enumerate(order):
            # Augmented features
            if i == 0:
                X_aug_train = X_train
                X_aug_val = X_val
            else:
                prev_labels = Y_train[:, order[:i]]
                X_aug_train = np.hstack([X_train, prev_labels])
                
                prev_preds = preds[:, order[:i]]
                X_aug_val = np.hstack([X_val, prev_preds])
            
            # Train MLP with CE
            probs = train_mlp_ce(
                X_aug_train, Y_train[:, j],
                X_aug_val, Y_val[:, j],
                config
            )
            preds[:, j] = probs
        
        all_preds[m] = preds
    
    # Average predictions
    final_preds = all_preds.mean(axis=0)
    return final_preds

def train_ecc_lr_edl(X_train, Y_train, X_val, Y_val, n_chains, config):
    """ECC with linear model + EDL loss - use predictions only for fair comparison"""
    N_val, K = Y_val.shape
    all_preds = np.zeros((n_chains, N_val, K))

    for m in range(n_chains):
        order = np.random.permutation(K)
        preds = np.zeros((N_val, K))
        uncertainties = np.zeros((N_val, K))

        for i, j in enumerate(order):
            if i == 0:
                X_aug_train = X_train
                X_aug_val = X_val
            else:
                # Augment features with previous predictions only (no uncertainty)
                # This ensures fair comparison: all methods use same features
                # EDL improves via better loss function, not via uncertainty propagation
                prev_labels = Y_train[:, order[:i]].astype(np.float32)
                X_aug_train = np.hstack([X_train, prev_labels])
                
                # Validation: use predicted probabilities only
                prev_preds = preds[:, order[:i]]
                X_aug_val = np.hstack([X_val, prev_preds])

            probs, uncert = train_lr_edl(
                X_aug_train, Y_train[:, j],
                X_aug_val, Y_val[:, j],
                config
            )
            preds[:, j] = probs
            uncertainties[:, j] = uncert

        all_preds[m] = preds

    return all_preds.mean(axis=0)

def train_ecc_mlp_edl(X_train, Y_train, X_val, Y_val, n_chains, config):
    """ECC with MLP + EDL Loss (proposed method) - use predictions only for fair comparison"""
    N_val, K = Y_val.shape
    all_preds = np.zeros((n_chains, N_val, K))
    
    for m in range(n_chains):
        order = np.random.permutation(K)
        preds = np.zeros((N_val, K))
        uncertainties = np.zeros((N_val, K))
        
        for i, j in enumerate(order):
            if i == 0:
                X_aug_train = X_train
                X_aug_val = X_val
            else:
                # Augment features with previous predictions only (no uncertainty)
                # This ensures fair comparison: all methods use same features
                # EDL improves via better loss function, not via uncertainty propagation
                prev_labels = Y_train[:, order[:i]].astype(np.float32)
                X_aug_train = np.hstack([X_train, prev_labels])
                
                # Validation: use predicted probabilities only
                prev_preds = preds[:, order[:i]]
                X_aug_val = np.hstack([X_val, prev_preds])
            
            # Train MLP with EDL
            probs, uncert = train_mlp_edl(
                X_aug_train, Y_train[:, j],
                X_aug_val, Y_val[:, j],
                config
            )
            preds[:, j] = probs
            uncertainties[:, j] = uncert
        
        all_preds[m] = preds
    
    # Average predictions
    final_preds = all_preds.mean(axis=0)
    return final_preds

# ============================================================================
# 6. EVALUATION METRICS
# ============================================================================

def compute_metrics(Y_true, Y_pred_probs, threshold=0.5):
    """Compute multi-label metrics"""
    Y_pred = (Y_pred_probs >= threshold).astype(int)
    
    # Micro-F1
    micro_f1 = f1_score(Y_true, Y_pred, average='micro', zero_division=0)
    
    # Macro-F1
    macro_f1 = f1_score(Y_true, Y_pred, average='macro', zero_division=0)
    
    # Hamming Loss
    hamming = hamming_loss(Y_true, Y_pred)
    
    # Jaccard
    jaccard = jaccard_score(Y_true, Y_pred, average='samples', zero_division=0)
    
    return {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'hamming_loss': hamming,
        'jaccard': jaccard
    }

# ============================================================================
# 7. MAIN EXPERIMENT
# ============================================================================

def run_ablation_study():
    """Run full ablation study"""
    from sklearn.preprocessing import StandardScaler
    results = {}
    
    for dataset_name in CONFIG['datasets']:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*80}")
        
        # Load data
        X, Y = load_arff_dataset(dataset_name, CONFIG['data_dir'])
        N, D = X.shape
        K = Y.shape[1]
        print(f"Samples: {N}, Features: {D}, Labels: {K}")
        
        # Normalize features (CRITICAL for EDL)
        X = StandardScaler().fit_transform(X).astype('float32')
        
        # Cross-validation
        kf = KFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=SEED)
        
        fold_results = {
            'ECC-LR': [],
            'ECC-MLP-CE': [],
            'EDL-LR': [],
            'EDL-MLP': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            print(f"\n--- Fold {fold}/{CONFIG['n_folds']} ---")
            
            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]
            
            # 1. ECC-LR (Baseline)
            print("Training ECC-LR...")
            t0 = time.time()
            preds_lr = train_ecc_lr(X_train, Y_train, X_val, Y_val, CONFIG['n_chains'], CONFIG)
            time_lr = time.time() - t0
            metrics_lr = compute_metrics(Y_val, preds_lr)
            metrics_lr['time'] = time_lr
            fold_results['ECC-LR'].append(metrics_lr)
            print(f"  Micro-F1: {metrics_lr['micro_f1']:.4f}, Time: {time_lr:.1f}s")
            
            # 2. ECC-MLP-CE (Architecture Effect)
            print("Training ECC-MLP-CE...")
            t0 = time.time()
            preds_mlp_ce = train_ecc_mlp_ce(X_train, Y_train, X_val, Y_val, CONFIG['n_chains'], CONFIG)
            time_mlp_ce = time.time() - t0
            metrics_mlp_ce = compute_metrics(Y_val, preds_mlp_ce)
            metrics_mlp_ce['time'] = time_mlp_ce
            fold_results['ECC-MLP-CE'].append(metrics_mlp_ce)
            print(f"  Micro-F1: {metrics_mlp_ce['micro_f1']:.4f}, Time: {time_mlp_ce:.1f}s")
            
            # 3. EDL-LR (Loss Effect)
            print("Training EDL-LR...")
            t0 = time.time()
            preds_edl_lr = train_ecc_lr_edl(X_train, Y_train, X_val, Y_val, CONFIG['n_chains'], CONFIG)
            time_edl_lr = time.time() - t0
            metrics_edl_lr = compute_metrics(Y_val, preds_edl_lr)
            metrics_edl_lr['time'] = time_edl_lr
            fold_results['EDL-LR'].append(metrics_edl_lr)
            print(f"  Micro-F1: {metrics_edl_lr['micro_f1']:.4f}, Time: {time_edl_lr:.1f}s")
            
            # 4. EDL-MLP (Full Proposed)
            print("Training EDL-MLP...")
            t0 = time.time()
            preds_edl = train_ecc_mlp_edl(X_train, Y_train, X_val, Y_val, CONFIG['n_chains'], CONFIG)
            time_edl = time.time() - t0
            metrics_edl = compute_metrics(Y_val, preds_edl)
            metrics_edl['time'] = time_edl
            fold_results['EDL-MLP'].append(metrics_edl)
            print(f"  Micro-F1: {metrics_edl['micro_f1']:.4f}, Time: {time_edl:.1f}s")
        
        # Aggregate results
        dataset_results = {}
        for method in fold_results:
            dataset_results[method] = {
                'micro_f1_mean': np.mean([m['micro_f1'] for m in fold_results[method]]),
                'micro_f1_std': np.std([m['micro_f1'] for m in fold_results[method]]),
                'macro_f1_mean': np.mean([m['macro_f1'] for m in fold_results[method]]),
                'macro_f1_std': np.std([m['macro_f1'] for m in fold_results[method]]),
                'time_mean': np.mean([m['time'] for m in fold_results[method]]),
            }
        
        results[dataset_name] = dataset_results
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"Summary for {dataset_name}:")
        print(f"{'='*80}")
        for method in ['ECC-LR', 'ECC-MLP-CE', 'EDL-LR', 'EDL-MLP']:
            print(f"{method:15s} | Micro-F1: {dataset_results[method]['micro_f1_mean']:.4f} ± {dataset_results[method]['micro_f1_std']:.4f} | "
                  f"Macro-F1: {dataset_results[method]['macro_f1_mean']:.4f} ± {dataset_results[method]['macro_f1_std']:.4f}")
    
    # Save results
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / 'ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to {output_dir / 'ablation_results.json'}")
    print(f"{'='*80}")
    
    return results

if __name__ == '__main__':
    results = run_ablation_study()
