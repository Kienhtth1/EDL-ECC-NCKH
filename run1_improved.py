import warnings
warnings.filterwarnings('ignore')

import gc
import time
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
from sklearn.metrics import f1_score, accuracy_score, hamming_loss, jaccard_score


IS_KAGGLE   = Path('/kaggle/working').exists()
OUTPUTS_DIR = Path('/kaggle/working/outputs') if IS_KAGGLE else Path('./outputs')
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


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

N_CHAINS      = 20
HIDDEN        = 256
DROPOUT       = 0.2
EPOCHS        = 100
BATCH_SIZE    = 64
LR            = 1e-3
SEED          = 42
THRESHOLD     = 0.5  
METRICS_NAMES = ['1-HL', 'SubAcc', 'Micro-F1', 'Macro-F1', 'Jaccard']

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Device: cuda ({torch.cuda.get_device_name(0)})\n")
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')
    print("Device: CPU\n")

# =============================================================================
# EDL Components
# =============================================================================

def dirichlet_kl_binary(alpha):
    beta    = torch.ones_like(alpha)
    S_alpha = alpha.sum(dim=-1, keepdim=True)
    S_beta  = beta.sum(dim=-1, keepdim=True)
    lnB_a   = torch.lgamma(alpha).sum(-1, keepdim=True) - torch.lgamma(S_alpha)
    lnB_b   = torch.lgamma(beta).sum(-1, keepdim=True)  - torch.lgamma(S_beta)
    dg_diff = torch.digamma(alpha) - torch.digamma(S_alpha)
    return (
        ((alpha - beta) * dg_diff).sum(-1, keepdim=True).squeeze(-1)
        + lnB_b.squeeze(-1) - lnB_a.squeeze(-1)
    )


def edl_binary_mse_loss(alpha, target, epoch, annealing_step=5):
    S        = alpha.sum(dim=-1, keepdim=True)
    p        = alpha / S
    y        = torch.stack([1.0 - target.float(), target.float()], dim=-1)
    mse      = ((y - p) ** 2).sum(dim=-1)
    var_term = (p * (1.0 - p) / (S + 1.0)).sum(dim=-1)
    pos_w    = torch.where(target > 0, 2.0, 1.0)
    alpha_tilde = y + (1.0 - y) * alpha
    kl       = dirichlet_kl_binary(alpha_tilde)
    lambda_t = min(1.0, epoch / max(1, annealing_step))
    return ((mse + var_term) * pos_w + lambda_t * kl).mean()


def predict_edl_binary(alpha):
    S     = alpha.sum(dim=-1, keepdim=True)
    p_pos = alpha[..., 1:2] / S
    u     = 2.0 / S
    return p_pos, u

# =============================================================================
# Standard NN Loss (for NN-CC/NN-ECC)
# =============================================================================

def standard_nn_loss(pred, target):
    """BCE loss with class weighting for standard NN"""
    pos_w = torch.where(target > 0, 2.0, 1.0)
    bce = F.binary_cross_entropy(pred, target.float(), reduction='none')
    return (bce * pos_w).mean()

# =============================================================================
# Model Architectures
# =============================================================================

class EDLModel(nn.Module):
    """EDL Model with Dirichlet output"""
    def __init__(self, in_dim, num_labels=1, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),      nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden // 2, num_labels * 2)

    def forward(self, x):
        h = self.net(x)
        e = F.relu(self.out(h)) + 1e-4
        return e + 1.0


class StandardNNModel(nn.Module):
    """Standard NN Model with Sigmoid output (NO EDL, NO Uncertainty)"""
    def __init__(self, in_dim, num_labels=1, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),      nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_labels),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))

# =============================================================================
# NN-CC: Neural Network Classifier Chains (Single Chain, No EDL)
# =============================================================================

class NN_CC:
    """Standard NN-based Classifier Chain (NO EDL, NO Uncertainty)"""
    def __init__(self, in_dim, num_labels, hidden=HIDDEN, dev=device):
        self.num_labels = num_labels
        self.hidden     = hidden
        self.dev        = dev
        self.models     = []
        self.order      = None

    def fit(self, X_tr, Y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR):
        self.order = np.random.permutation(self.num_labels)
        X_current = torch.from_numpy(X_tr).float().to(self.dev)
        Y_tr_t    = torch.from_numpy(Y_tr).float().to(self.dev)
        self.models = []

        for lbl_idx in self.order:
            m   = StandardNNModel(X_current.shape[1], num_labels=1, hidden=self.hidden).to(self.dev)
            opt = torch.optim.Adam(m.parameters(), lr=lr)
            ld  = DataLoader(
                TensorDataset(X_current, Y_tr_t[:, lbl_idx]),
                batch_size=batch_size, shuffle=True,
            )
            
            for ep in range(1, epochs + 1):
                m.train()
                for xb, yb in ld:
                    pred = m(xb).squeeze(-1)
                    loss = standard_nn_loss(pred, yb)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            # Augment features with predictions ONLY (no uncertainty)
            m.eval()
            with torch.no_grad():
                pp        = m(X_current)  # Just probability
                X_current = torch.cat([X_current, pp], dim=-1)
            
            self.models.append((lbl_idx, m))

    def predict_proba(self, X_val):
        X_curr  = torch.from_numpy(X_val).float().to(self.dev)
        chain_p = np.zeros((X_val.shape[0], self.num_labels), dtype='float32')
        
        for lbl_idx, m in self.models:
            m.eval()
            with torch.no_grad():
                pp                  = m(X_curr)
                chain_p[:, lbl_idx] = pp.squeeze(-1).cpu().numpy()
                X_curr              = torch.cat([X_curr, pp], dim=-1)
        
        return chain_p

    def predict(self, X_val, threshold=THRESHOLD):
        proba = self.predict_proba(X_val)
        return (proba >= threshold).astype(int)

# =============================================================================
# NN-ECC: Ensemble of NN Classifier Chains (NO EDL, NO Uncertainty)
# =============================================================================

class NN_ECC:
    """Ensemble of Standard NN Chains (NO EDL, NO Uncertainty)"""
    def __init__(self, in_dim, num_labels, n_chains=N_CHAINS, hidden=HIDDEN, dev=device):
        self.num_labels = num_labels
        self.n_chains   = n_chains
        self.hidden     = hidden
        self.dev        = dev
        self.chains     = []

    def fit(self, X_tr, Y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR):
        self.chains = []
        for _ in range(self.n_chains):
            chain = NN_CC(X_tr.shape[1], self.num_labels, hidden=self.hidden, dev=self.dev)
            chain.fit(X_tr, Y_tr, epochs=epochs, batch_size=batch_size, lr=lr)
            self.chains.append(chain)

    def predict_proba(self, X_val):
        all_probs = [chain.predict_proba(X_val) for chain in self.chains]
        return np.mean(all_probs, axis=0)

    def predict(self, X_val, threshold=THRESHOLD):
        proba = self.predict_proba(X_val)
        return (proba >= threshold).astype(int)

# =============================================================================
# EDL-ECC: Evidential Deep Learning ECC (With Uncertainty)
# =============================================================================

class EDL_ECC:
    """EDL-based Ensemble Classifier Chains (WITH Uncertainty)"""
    def __init__(self, in_dim, num_labels, n_chains=N_CHAINS, hidden=HIDDEN, dev=device):
        self.num_labels = num_labels
        self.n_chains   = n_chains
        self.hidden     = hidden
        self.dev        = dev
        self.chains     = []

    def fit(self, X_tr, Y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR):
        self.chains = []
        for _ in range(self.n_chains):
            order     = np.random.permutation(self.num_labels)
            X_current = torch.from_numpy(X_tr).float().to(self.dev)
            Y_tr_t    = torch.from_numpy(Y_tr).float().to(self.dev)
            mods      = []
            
            for lbl_idx in order:
                m   = EDLModel(X_current.shape[1], num_labels=1, hidden=self.hidden).to(self.dev)
                opt = torch.optim.Adam(m.parameters(), lr=lr)
                ld  = DataLoader(
                    TensorDataset(X_current, Y_tr_t[:, lbl_idx]),
                    batch_size=batch_size, shuffle=True,
                )
                
                for ep in range(1, epochs + 1):
                    m.train()
                    for xb, yb in ld:
                        a    = m(xb)
                        loss = edl_binary_mse_loss(a, yb, ep)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()

                # Augment features with BOTH probability and uncertainty
                m.eval()
                with torch.no_grad():
                    ap        = m(X_current)
                    pp, u     = predict_edl_binary(ap)
                    X_current = torch.cat([X_current, pp, u], dim=-1)
                
                mods.append((lbl_idx, m))
            
            self.chains.append(mods)

    def predict_proba(self, X_val):
        all_probs = []
        X_val_t   = torch.from_numpy(X_val).float().to(self.dev)
        
        for mods in self.chains:
            X_curr  = X_val_t.clone()
            chain_p = np.zeros((X_val.shape[0], self.num_labels), dtype='float32')
            
            for lbl_idx, m in mods:
                m.eval()
                with torch.no_grad():
                    a                   = m(X_curr)
                    pp, u               = predict_edl_binary(a)
                    chain_p[:, lbl_idx] = pp.squeeze(-1).cpu().numpy()
                    X_curr              = torch.cat([X_curr, pp, u], dim=-1)
            
            all_probs.append(chain_p)
        
        return np.mean(all_probs, axis=0)

    def predict(self, X_val, threshold=THRESHOLD):
        proba = self.predict_proba(X_val)
        return (proba >= threshold).astype(int)

# =============================================================================
# Evaluation & Utils
# =============================================================================

def evaluate(y_true, y_pred):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    return {
        '1-HL':     round(1 - hamming_loss(y_true, y_pred), 4),
        'SubAcc':   round(accuracy_score(y_true, y_pred), 4),
        'Micro-F1': round(f1_score(y_true, y_pred, average='micro',    zero_division=0), 4),
        'Macro-F1': round(f1_score(y_true, y_pred, average='macro',    zero_division=0), 4),
        'Jaccard':  round(jaccard_score(y_true, y_pred, average='samples', zero_division=0), 4),
    }


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
            except Exception:
                pass
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    if label_location is None:
        nominal_idx = [i for i, t in enumerate(meta.types()) if t == 'nominal']
        if nominal_idx == list(range(num_labels)):
            label_location = 'first'
        else:
            label_location = 'last'

    if label_location == 'first':
        Y = (df.iloc[:, :num_labels].values > 0).astype('float32')
        X = df.iloc[:, num_labels:].values.astype('float32')
        label_names = [str(c) for c in df.columns[:num_labels]]
    else:
        X = df.iloc[:, :-num_labels].values.astype('float32')
        Y = (df.iloc[:, -num_labels:].values > 0).astype('float32')
        label_names = [str(c) for c in df.columns[-num_labels:]]

    return X, Y, label_names

# =============================================================================
# Main Training Loop
# =============================================================================

def run():
    print("=" * 90)
    print(f"  IMPROVED COMPARISON: BR vs CC vs ECC vs NN-CC vs NN-ECC vs EDL-ECC")
    print(f"  Chains: M={N_CHAINS} | 5-Fold CV")
    print("=" * 90)
    print(f"  Output directory: {OUTPUTS_DIR.resolve()}\n")

    all_results     = {}
    all_std_results = {}
    # NEW: Added NN-CC and NN-ECC baselines
    models_to_run   = ['BR', 'CC', 'ECC', 'NN-CC', 'NN-ECC', 'EDL-ECC']

    for ds_name, cfg in DATASET_CONFIGS.items():
        arff_path = find_file(cfg['file'])
        if arff_path is None or not arff_path.exists():
            print(f"[SKIP] {ds_name}: Không tìm thấy file {cfg['file']}")
            continue

        ds_out = OUTPUTS_DIR / ds_name
        ds_out.mkdir(parents=True, exist_ok=True)

        num_labels = cfg['num_labels']
        label_location = cfg.get('label_location', 'last')
        X, Y, label_names = load_arff(arff_path, num_labels, label_location=label_location)
        X = StandardScaler().fit_transform(X).astype('float32')

        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        fold_scores = {m: [] for m in models_to_run}

        print(f"\n{'─'*90}")
        print(f"  [DATASET] {ds_name}  |  Samples: {len(X)}  Features: {X.shape[1]}  Labels: {num_labels}")
        print(f"{'─'*90}")

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
            X_tr, X_va = X[tr_idx], X[va_idx]
            Y_tr, Y_va = Y[tr_idx], Y[va_idx]

            # Ensure at least one positive/negative sample per label
            for col in range(Y_tr.shape[1]):
                u = np.unique(Y_tr[:, col])
                if len(u) < 2:
                    if 0.0 not in u: Y_tr[0, col] = 0.0
                    if 1.0 not in u: Y_tr[0, col] = 1.0

            tf = time.time()

            base_lr = LogisticRegression(solver='lbfgs', max_iter=300, class_weight='balanced')

            # 1. Binary Relevance (BR) - LogReg
            print(f"    [Fold {fold+1}] Training BR (LogReg)...", end=' ', flush=True)
            br = OneVsRestClassifier(base_lr, n_jobs=-1)
            br.fit(X_tr, Y_tr)
            br_pred = br.predict(X_va)
            fold_scores['BR'].append(evaluate(Y_va, br_pred))
            print(f"F1={fold_scores['BR'][-1]['Micro-F1']:.4f}")

            # 2. Classifier Chains (CC) - LogReg
            print(f"    [Fold {fold+1}] Training CC (LogReg)...", end=' ', flush=True)
            cc = ClassifierChain(base_lr, order='random', random_state=SEED + fold)
            cc.fit(X_tr, Y_tr)
            cc_pred = cc.predict(X_va)
            fold_scores['CC'].append(evaluate(Y_va, cc_pred))
            print(f"F1={fold_scores['CC'][-1]['Micro-F1']:.4f}")

            # 3. Ensemble of Classifier Chains (ECC) - LogReg
            print(f"    [Fold {fold+1}] Training ECC (LogReg, M={N_CHAINS})...", end=' ', flush=True)
            chain_probs = []
            for c in range(N_CHAINS):
                cc_m = ClassifierChain(base_lr, order='random', random_state=SEED + c * 100 + fold)
                cc_m.fit(X_tr, Y_tr)
                if hasattr(cc_m, "predict_proba"):
                    cp = cc_m.predict_proba(X_va)
                else:
                    cp = cc_m.predict(X_va)
                chain_probs.append(cp)
            ecc_pred = (np.mean(chain_probs, axis=0) >= THRESHOLD).astype(int)
            fold_scores['ECC'].append(evaluate(Y_va, ecc_pred))
            print(f"F1={fold_scores['ECC'][-1]['Micro-F1']:.4f}")

            # 4. NEW: NN-CC (Single NN Chain, NO Uncertainty)
            print(f"    [Fold {fold+1}] Training NN-CC (NN, no uncertainty)...", end=' ', flush=True)
            nn_cc = NN_CC(X_tr.shape[1], num_labels, dev=device)
            nn_cc.fit(X_tr, Y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR)
            nn_cc_pred = nn_cc.predict(X_va, threshold=THRESHOLD)
            fold_scores['NN-CC'].append(evaluate(Y_va, nn_cc_pred))
            print(f"F1={fold_scores['NN-CC'][-1]['Micro-F1']:.4f}")

            # 5. NEW: NN-ECC (Ensemble NN Chains, NO Uncertainty)
            print(f"    [Fold {fold+1}] Training NN-ECC (NN, M={N_CHAINS}, no uncertainty)...", end=' ', flush=True)
            nn_ecc = NN_ECC(X_tr.shape[1], num_labels, n_chains=N_CHAINS, dev=device)
            nn_ecc.fit(X_tr, Y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR)
            nn_ecc_pred = nn_ecc.predict(X_va, threshold=THRESHOLD)
            fold_scores['NN-ECC'].append(evaluate(Y_va, nn_ecc_pred))
            print(f"F1={fold_scores['NN-ECC'][-1]['Micro-F1']:.4f}")

            # 6. EDL-ECC (Proposed: EDL with Uncertainty)
            print(f"    [Fold {fold+1}] Training EDL-ECC (EDL+Unc, M={N_CHAINS})...", end=' ', flush=True)
            edl_ecc = EDL_ECC(X_tr.shape[1], num_labels, n_chains=N_CHAINS, dev=device)
            edl_ecc.fit(X_tr, Y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR)
            edl_pred = edl_ecc.predict(X_va, threshold=THRESHOLD)
            fold_scores['EDL-ECC'].append(evaluate(Y_va, edl_pred))
            print(f"F1={fold_scores['EDL-ECC'][-1]['Micro-F1']:.4f}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            elapsed = time.time() - tf
            mf1 = {m: fold_scores[m][-1]['Micro-F1'] for m in models_to_run}
            print(f"  Fold {fold+1}/5 [{elapsed:5.1f}s] Summary:")
            print(f"    BR={mf1['BR']:.4f}  CC={mf1['CC']:.4f}  ECC={mf1['ECC']:.4f}")
            print(f"    NN-CC={mf1['NN-CC']:.4f}  NN-ECC={mf1['NN-ECC']:.4f}  EDL-ECC={mf1['EDL-ECC']:.4f}")

        # Aggregate results
        df_mean = pd.DataFrame({m: pd.DataFrame(fold_scores[m]).mean() for m in models_to_run}).T
        df_std  = pd.DataFrame({m: pd.DataFrame(fold_scores[m]).std() for m in models_to_run}).T

        print(f"\n  [Mean Results 5-FOLD - {ds_name}]:")
        print(df_mean.round(4).to_string())

        df_mean.round(4).to_csv(ds_out / 'dataset_metrics_table.csv')

        all_results[ds_name] = {
            mn: {m: round(float(df_mean.loc[mn, m]), 4) for m in METRICS_NAMES}
            for mn in models_to_run
        }
        all_std_results[ds_name] = {
            mn: {m: round(float(df_std.loc[mn, m]), 4) for m in METRICS_NAMES}
            for mn in models_to_run
        }

    if not all_results:
        print("\n[Warning] No datasets found.")
        return

    # Create summary table
    summary_rows = []
    for ds_name, res in all_results.items():
        bv   = res['BR']['Micro-F1']
        cv   = res['CC']['Micro-F1']
        ecv  = res['ECC']['Micro-F1']
        nccv = res['NN-CC']['Micro-F1']
        necv = res['NN-ECC']['Micro-F1']
        edv  = res['EDL-ECC']['Micro-F1']
        
        best_val = max(bv, cv, ecv, nccv, necv, edv)
        winner = [m for m, v in [('BR', bv), ('CC', cv), ('ECC', ecv), 
                                  ('NN-CC', nccv), ('NN-ECC', necv), ('EDL-ECC', edv)] 
                  if v == best_val][0]

        summary_rows.append({
            'Dataset':             ds_name,
            'BR (LogReg)':         bv,
            'CC (LogReg)':         cv,
            'ECC (LogReg)':        ecv,
            'NN-CC':               nccv,
            'NN-ECC':              necv,
            'EDL-ECC':             edv,
            'Winner':              winner,
            'Δ(EDL-NN_ECC)':       round(edv - necv, 4),
            'Δ(EDL-ECC_LogReg)':   round(edv - ecv, 4),
            'Δ(NN_ECC-ECC_LogReg)': round(necv - ecv, 4),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUTS_DIR / 'overall_summary_table.csv', index=False)

    print(f"\n{'='*100}")
    print("   MICRO-F1 SUMMARY - ALL DATASETS")
    print(f"{'='*100}")
    print(summary_df.to_string(index=False))
    print(f"{'='*100}")

    # Contribution analysis
    print("\n" + "="*100)
    print("   CONTRIBUTION ANALYSIS")
    print("="*100)
    avg_br = np.mean([res['BR']['Micro-F1'] for res in all_results.values()])
    avg_ecc = np.mean([res['ECC']['Micro-F1'] for res in all_results.values()])
    avg_nn_cc = np.mean([res['NN-CC']['Micro-F1'] for res in all_results.values()])
    avg_nn_ecc = np.mean([res['NN-ECC']['Micro-F1'] for res in all_results.values()])
    avg_edl = np.mean([res['EDL-ECC']['Micro-F1'] for res in all_results.values()])
    
    print(f"  Average Micro-F1:")
    print(f"    BR (LogReg):           {avg_br:.4f}")
    print(f"    ECC (LogReg, M={N_CHAINS}):  {avg_ecc:.4f}  [+{avg_ecc-avg_br:.4f} from Ensemble]")
    print(f"    NN-CC:                 {avg_nn_cc:.4f}  [+{avg_nn_cc-avg_br:.4f} from Deep Learning]")
    print(f"    NN-ECC (M={N_CHAINS}):       {avg_nn_ecc:.4f}  [+{avg_nn_ecc-avg_nn_cc:.4f} from NN+Ensemble]")
    print(f"    EDL-ECC (M={N_CHAINS}):      {avg_edl:.4f}  [+{avg_edl-avg_nn_ecc:.4f} from EDL+Uncertainty]")
    print(f"\n  Key Insights:")
    print(f"    Deep Learning contribution:  +{avg_nn_cc-avg_br:.4f} ({(avg_nn_cc-avg_br)/(avg_edl-avg_br)*100:.1f}%)")
    print(f"    Ensemble contribution:       +{avg_nn_ecc-avg_nn_cc:.4f} ({(avg_nn_ecc-avg_nn_cc)/(avg_edl-avg_br)*100:.1f}%)")
    print(f"    EDL+Uncertainty contribution: +{avg_edl-avg_nn_ecc:.4f} ({(avg_edl-avg_nn_ecc)/(avg_edl-avg_br)*100:.1f}%)")
    print("="*100)

    out_json = OUTPUTS_DIR / 'results_run_improved.json'
    out_json.write_text(
        json.dumps({'mean': all_results, 'std': all_std_results}, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(f"\n✓ Results saved to: {OUTPUTS_DIR.resolve()}")


if __name__ == '__main__':
    run()
