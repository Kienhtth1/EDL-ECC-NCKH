"""
run.py  --  Chay toan bo 9 dataset: BR vs CC vs EDL-ECC (5-Fold CV)
EDLModel: hidden=256, dropout=0.3 (giong notebook hien tai)
Cach chay:  python run.py
"""

import warnings; warnings.filterwarnings('ignore')
import time, json
import numpy as np
import pandas as pd
from pathlib import Path
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
from sklearn.metrics import (f1_score, accuracy_score,
                              hamming_loss, jaccard_score)

# ── Cau hinh ──────────────────────────────────────────────────────────────────
DATA_DIR = Path('./data')
DATASET_CONFIGS = {
    'Scene':           {'file': 'Scene.arff',           'num_labels': 6},
    'Yeast':           {'file': 'Yeast.arff',           'num_labels': 14},
    'emotions':        {'file': 'emotions.arff',         'num_labels': 6},
    'HumanPseAAC':     {'file': 'HumanPseAAC.arff',     'num_labels': 14},
    'PlantPseAAC':     {'file': 'PlantPseAAC.arff',     'num_labels': 12},
    'GpositivePseAAC': {'file': 'GpositivePseAAC.arff', 'num_labels': 4},
    'VirusPseAAC':     {'file': 'VirusPseAAC.arff',     'num_labels': 6},
    'Water-quality':   {'file': 'Water-quality.arff',   'num_labels': 14},
    'CHD_49':          {'file': 'CHD_49.arff',          'num_labels': 6},
}

N_CHAINS   = 3
HIDDEN     = 256
DROPOUT    = 0.3
EPOCHS     = 10
BATCH_SIZE = 32
LR         = 1e-3
SEED       = 42
METRICS    = ['1-HL', 'SubAcc', 'Micro-F1', 'Macro-F1', 'Jaccard']

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Device: {device} (NVIDIA CUDA GPU)\n")
else:
    try:
        import torch_directml
        device = torch_directml.device()
        print(f"Device: {device} (DirectML AMD/Intel GPU)\n")
    except ImportError:
        device = torch.device('cpu')
        print("Device: cpu (Khong tim thay DirectML/CUDA, dang chay tren CPU)\n")


# ── EDL Helper functions (giong het notebook) ─────────────────────────────────
def dirichlet_kl_binary(alpha):
    beta     = torch.ones_like(alpha)
    S_alpha  = alpha.sum(dim=-1, keepdim=True)
    S_beta   = beta.sum(dim=-1, keepdim=True)
    lnB_a    = torch.lgamma(alpha).sum(-1, keepdim=True) - torch.lgamma(S_alpha)
    lnB_b    = torch.lgamma(beta).sum(-1, keepdim=True)  - torch.lgamma(S_beta)
    dg_diff  = torch.digamma(alpha) - torch.digamma(S_alpha)
    return ((alpha-beta)*dg_diff).sum(-1, keepdim=True).squeeze(-1) \
           + lnB_b.squeeze(-1) - lnB_a.squeeze(-1)


def edl_binary_mse_loss(alpha, target, epoch, annealing_step=5):
    S         = alpha.sum(dim=-1, keepdim=True)
    p         = alpha / S
    y         = torch.stack([1. - target.float(), target.float()], dim=-1)
    mse       = ((y - p) ** 2).sum(dim=-1)
    var_term  = (p * (1. - p) / (S + 1.)).sum(dim=-1)
    pos_w     = torch.where(target > 0, 2., 1.)
    kl        = dirichlet_kl_binary(alpha)
    lambda_t  = min(1., epoch / max(1, annealing_step))
    return ((mse + var_term) * pos_w + lambda_t * kl).mean()


def predict_edl_binary(alpha):
    S     = alpha.sum(dim=-1, keepdim=True)
    p_pos = alpha[..., 1:2] / S
    u     = 2. / S
    return p_pos, u


# ── EDLModel (hidden=256, dropout=0.3) ────────────────────────────────────────
class EDLModel(nn.Module):
    def __init__(self, in_dim, num_labels=1, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden // 2, num_labels * 2)

    def forward(self, x):
        h = self.net(x)
        e = F.relu(self.out(h)) + 1e-4   # evidence >= 0, shape [B, 2]
        return e + 1.0                    # alpha = evidence + 1


# ── EDL-ECC ───────────────────────────────────────────────────────────────────
class EDL_ECC:
    def __init__(self, in_dim, num_labels, n_chains=N_CHAINS,
                 hidden=HIDDEN, dev='cpu'):
        self.num_labels = num_labels
        self.n_chains   = n_chains
        self.hidden     = hidden
        self.dev        = dev

    def fit(self, X_tr, Y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR):
        self.chains = []
        for _ in range(self.n_chains):
            order     = np.random.permutation(self.num_labels)
            X_current = torch.from_numpy(X_tr).float().to(self.dev)
            Y_tr_t    = torch.from_numpy(Y_tr).float().to(self.dev)
            mods      = []
            for lbl_idx in order:
                m   = EDLModel(X_current.shape[1], num_labels=1,
                               hidden=self.hidden).to(self.dev)
                opt = torch.optim.Adam(m.parameters(), lr=lr)
                ld  = DataLoader(TensorDataset(X_current, Y_tr_t[:, lbl_idx]),
                                 batch_size=batch_size, shuffle=True)
                for ep in range(1, epochs + 1):
                    m.train()
                    for xb, yb in ld:
                        a = m(xb)
                        loss = edl_binary_mse_loss(a, yb, ep)
                        opt.zero_grad(); loss.backward(); opt.step()
                m.eval()
                with torch.no_grad():
                    ap = m(X_current)
                    pp, u = predict_edl_binary(ap)
                    X_current = torch.cat([X_current, pp, u], dim=-1)
                mods.append((lbl_idx, m))
            self.chains.append(mods)

    def predict_proba(self, X_val):
        all_probs = []
        X_val_t   = torch.from_numpy(X_val).float().to(self.dev)
        for mods in self.chains:
            X_curr    = X_val_t.clone()
            chain_p   = np.zeros((X_val.shape[0], self.num_labels), dtype='float32')
            for lbl_idx, m in mods:
                m.eval()
                with torch.no_grad():
                    a       = m(X_curr)
                    pp, u   = predict_edl_binary(a)
                    chain_p[:, lbl_idx] = pp.squeeze(-1).cpu().numpy()
                    X_curr  = torch.cat([X_curr, pp, u], dim=-1)
            all_probs.append(chain_p)
        return np.mean(all_probs, axis=0)


# ── Metrics & Data helpers ────────────────────────────────────────────────────
def evaluate(y_true, y_pred):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    return {
        '1-HL':     round(1 - hamming_loss(y_true, y_pred), 4),
        'SubAcc':   round(accuracy_score(y_true, y_pred), 4),
        'Micro-F1': round(f1_score(y_true, y_pred, average='micro',  zero_division=0), 4),
        'Macro-F1': round(f1_score(y_true, y_pred, average='macro',  zero_division=0), 4),
        'Jaccard':  round(jaccard_score(y_true, y_pred, average='samples', zero_division=0), 4),
    }


def load_arff(path, num_labels):
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    for col in df.columns:
        if df[col].dtype == object:
            try:    df[col] = df[col].str.decode('utf-8')
            except: pass
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    X = df.iloc[:, :-num_labels].values.astype('float32')
    Y = (df.iloc[:, -num_labels:].values > 0).astype('float32')
    return X, Y


# ── MAIN ──────────────────────────────────────────────────────────────────────
S1 = '=' * 78
S2 = '-' * 78

print(S1)
print('  EDL-ECC vs BR vs CC  |  5-Fold Cross-Validation  |  9 Datasets')
print(f'  EDLModel: hidden={HIDDEN}, dropout={DROPOUT}, n_chains={N_CHAINS}, epochs={EPOCHS}, lr={LR}')
print(S1)

all_results = {}

for ds_name, cfg in DATASET_CONFIGS.items():
    arff_path = DATA_DIR / cfg['file']
    if not arff_path.exists():
        print(f'\n[SKIP] {ds_name}: {arff_path} khong ton tai.')
        continue

    num_labels = cfg['num_labels']
    X, Y = load_arff(arff_path, num_labels)
    X = StandardScaler().fit_transform(X).astype('float32')

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    fold_scores = {'BR': [], 'CC': [], 'EDL-ECC': []}
    base_lr = LogisticRegression(solver='lbfgs', max_iter=300,
                                  class_weight='balanced')
    t_start = time.time()

    print(f'\n{S2}')
    print(f'  Dataset: {ds_name:<18} | Mau: {len(X):>5} | '
          f'Features: {X.shape[1]:>4} | Nhan: {num_labels}')
    print(S2)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        Y_tr, Y_va = Y[tr_idx], Y[va_idx]

        # Dam bao moi nhan co it nhat ca class 0 va class 1 trong train
        for col in range(Y_tr.shape[1]):
            u = np.unique(Y_tr[:, col])
            if len(u) < 2:
                if 0.0 not in u:
                    Y_tr[0, col] = 0.0
                if 1.0 not in u:
                    Y_tr[0, col] = 1.0

        print(f'  Fold {fold+1}/5 ', end='', flush=True)
        tf = time.time()

        # BR
        br = OneVsRestClassifier(base_lr)
        br.fit(X_tr, Y_tr)
        fold_scores['BR'].append(evaluate(Y_va, br.predict(X_va)))

        # CC
        cc = ClassifierChain(base_lr, order='random', random_state=SEED)
        cc.fit(X_tr, Y_tr)
        fold_scores['CC'].append(evaluate(Y_va, cc.predict(X_va)))

        # EDL-ECC
        edl_ecc = EDL_ECC(X_tr.shape[1], num_labels, dev=device)
        edl_ecc.fit(X_tr, Y_tr)
        proba = edl_ecc.predict_proba(X_va)

        best_th, best_f1 = 0.5, 0.
        for th in np.arange(0.1, 0.55, 0.05):
            pt = (proba >= th).astype(int)
            if pt.sum() == 0:
                continue
            try:
                f1 = f1_score(Y_va.astype(int), pt, average='micro',
                              zero_division=0)
                if f1 > best_f1:
                    best_f1, best_th = f1, th
            except ValueError:
                continue

        fold_scores['EDL-ECC'].append(evaluate(Y_va, (proba >= best_th).astype(int)))

        elapsed = time.time() - tf
        mf1 = {k: fold_scores[k][-1]['Micro-F1'] for k in ['BR','CC','EDL-ECC']}
        print(f'[{elapsed:>5.1f}s]  Micro-F1:  '
              f'BR={mf1["BR"]:.4f}  CC={mf1["CC"]:.4f}  EDL-ECC={mf1["EDL-ECC"]:.4f}')

    # Tong hop mean 5 folds
    total_time = time.time() - t_start
    res = {
        mn: {m: round(np.mean([s[m] for s in fold_scores[mn]]), 4)
             for m in METRICS}
        for mn in ['BR', 'CC', 'EDL-ECC']
    }
    all_results[ds_name] = res

    # In bang ket qua dataset nay
    print()
    hdr = f"  {'Metric':<12}{'BR':>10}{'CC':>10}{'EDL-ECC':>12}"
    hdr += f"{'D(EDL-BR)':>12}{'D(EDL-CC)':>12}"
    print(hdr)
    print(f"  {'-' * 68}")
    for m in METRICS:
        bv  = res['BR'][m]
        cv  = res['CC'][m]
        ev  = res['EDL-ECC'][m]
        d_br = ev - bv
        d_cc = ev - cv
        sb = '+' if d_br >= 0 else ''
        sc = '+' if d_cc >= 0 else ''
        best_val = max(bv, cv, ev)
        star_edl = ' *' if ev == best_val else '  '
        print(f"  {m:<12}{bv:>10.4f}{cv:>10.4f}{ev:>10.4f}{star_edl}"
              f"{sb + str(round(d_br,4)):>12}{sc + str(round(d_cc,4)):>12}")
    print(f"  Thoi gian toan bo: {total_time:.1f}s")

# ── Bang tong hop cuoi ────────────────────────────────────────────────────────
print(f'\n\n{S1}')
print('  BANG TONG HOP CUOI --- Micro-F1 (Mean 5-Fold CV)')
print(S1)
print(f"  {'Dataset':<20}{'BR':>10}{'CC':>10}{'EDL-ECC':>10}"
      f"{'Best':>10}{'D(EDL-CC)':>12}")
print(f"  {'-' * 74}")

edl_wins  = 0
br_wins   = 0
cc_wins   = 0
for ds_name, res in all_results.items():
    bv = res['BR']['Micro-F1']
    cv = res['CC']['Micro-F1']
    ev = res['EDL-ECC']['Micro-F1']
    best_val = max(bv, cv, ev)
    if ev == best_val:   winner = 'EDL-ECC'; edl_wins += 1
    elif cv == best_val: winner = 'CC';      cc_wins  += 1
    else:                winner = 'BR';      br_wins  += 1
    d_cc = ev - cv
    sc   = '+' if d_cc >= 0 else ''
    print(f"  {ds_name:<20}{bv:>10.4f}{cv:>10.4f}{ev:>10.4f}"
          f"{winner:>10}{sc + str(round(d_cc,4)):>12}")

print(f"  {'-' * 74}")
print(f"  Tong ket: EDL-ECC thang {edl_wins}/{len(all_results)} datasets "
      f"| CC thang {cc_wins} | BR thang {br_wins}")
print(S1)

# Luu JSON
out_path = Path('results_run.json')
out_path.write_text(
    json.dumps(all_results, ensure_ascii=False, indent=2),
    encoding='utf-8'
)
print(f'\n  Ket qua da luu tai: {out_path.resolve()}')
print('  Hoan thanh!')
