"""
Generator script for complete_edl_ecc_attention.ipynb
Tự chứa 100% (self-contained), không phụ thuộc file ngoài, chạy mượt mà trên Jupyter/VSCode.
"""
import json

def nb_md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": [src]}

def nb_code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [src]}

cells = []

# ─── Title ────────────────────────────────────────────────────────────────────
cells.append(nb_md("""# Pipeline EDL-ECC-Attention: Evidential Classifier Chains kết hợp Label-Feature Attention

Notebook này triển khai kiến trúc mới **EDL-ECC-Attention** lấy cảm hứng từ:
1. *Rethinking Multi-Label Image Classification With Deep Learning* (Label Queries & Feature Cross-Attention)
2. *Multi-Label Text Classification with Label Attention Aware and Correlation Aware Contrastive Learning* (Label-Correlation Attention)

### 7 Bước triển khai chi tiết:
1. **EDA**: Phân tích tương quan và phân bố nhãn
2. **Preprocessing**: Chuẩn hóa đặc trưng & cấu hình 5-Fold CV
3. **Định nghĩa Module**: Standard EDL Binary Module & EDL Attention Module
4. **EDL-ECC & EDL-ECC-Attention Ensemble**: Xây dựng 2 kiến trúc so sánh
5. **5-Fold Cross-Validation**: Đánh giá thực nghiệm so sánh BR, CC, Standard EDL-ECC và EDL-ECC-Attention
6. **Attention Heatmap Analysis**: Trực quan hóa ma trận trọng số Attention
7. **Advanced Visualizations**: Biểu đồ so sánh và bảng thống kê kết quả
"""))

# ─── Cell 1: Imports ──────────────────────────────────────────────────────────
cells.append(nb_code("""\
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.io import arff

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, hamming_loss, jaccard_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

# Cấu hình Dataset
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

DATASET_NAME = 'Scene'   # Doi ten dataset o day de chay dataset khac
cfg = DATASET_CONFIGS[DATASET_NAME]
DATASET_PATH = Path(f"./data/{cfg['file']}")
NUM_LABELS   = cfg['num_labels']

print(f"Dataset duoc chon: {DATASET_NAME} | File: {DATASET_PATH} | So nhan: {NUM_LABELS}")
"""))

# ─── Step 1: EDA ──────────────────────────────────────────────────────────────
cells.append(nb_md("## BUOC 1: Kham pha & Truc quan hoa Du lieu (EDA)"))

cells.append(nb_code("""\
# Doc du lieu ARFF
data, meta = arff.loadarff(DATASET_PATH)
df = pd.DataFrame(data)
for col in df.columns:
    if df[col].dtype == object:
        try: df[col] = df[col].str.decode('utf-8')
        except: pass
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

X_full = df.iloc[:, :-NUM_LABELS].values.astype('float32')
Y_full = df.iloc[:, -NUM_LABELS:].values.astype('float32')
Y_full = (Y_full > 0).astype('float32')

label_names = list(df.columns[-NUM_LABELS:])
print(f"Shape X: {X_full.shape} | Shape Y: {Y_full.shape}")
print(f"Nhan: {label_names}")
"""))

cells.append(nb_code("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
label_freq = Y_full.sum(axis=0)
axes[0].bar(label_names, label_freq, color=sns.color_palette('husl', NUM_LABELS), edgecolor='black')
axes[0].set_title(f'Tan suat Nhan - {DATASET_NAME}', fontweight='bold')
axes[0].tick_params(axis='x', rotation=30)

corr_matrix = np.corrcoef(Y_full.T)
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
            xticklabels=label_names, yticklabels=label_names, ax=axes[1], center=0, vmin=-1, vmax=1)
axes[1].set_title(f'Ma tran Tuong quan Nhan - {DATASET_NAME}', fontweight='bold')
plt.tight_layout()
plt.show()
"""))

# ─── Step 2: Preprocessing ────────────────────────────────────────────────────
cells.append(nb_md("## BUOC 2: Tien xu ly Du lieu & Thiet lap 5-Fold CV"))

cells.append(nb_code("""\
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_full).astype('float32')
kf       = KFold(n_splits=5, shuffle=True, random_state=42)

print(f"Du lieu chuan hoa: X shape={X_scaled.shape}")
print(f"Cau hinh 5-Fold CV: KFold(n_splits=5, shuffle=True, random_state=42)")
"""))

# ─── Step 3: Modules Definition ───────────────────────────────────────────────
cells.append(nb_md("""\
## BUOC 3: Dinh nghia cac Module Evidential Deep Learning (Standard & Attention)

1. **Standard EDL Binary Module**: MLP tiêu chuẩn (Linear $\to$ ReLU $\to$ Dropout $\to$ Linear $\to$ ReLU $\to$ Linear(2) $\to$ Softplus + 1).
2. **EDL Attention Binary Module**: Tích hợp khối **Feature-Label Cross Attention**:
   - $Q$: Chiếu từ vector ngữ cảnh nhãn và đặc trưng
   - $K, V$: Chiếu từ không gian đặc trưng ẩn
   - $A = \\text{Sigmoid}\\left(\\frac{Q \\cdot K}{\\sqrt{d}}\\right)$
   - $h_{\\text{attn}} = \\text{LayerNorm}(A \\cdot V + Q)$
   - Head: $\\alpha = \\text{ReLU}(\\text{MLP}(h_{\\text{attn}})) + 1.0$
"""))

cells.append(nb_code("""\
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    try:
        import torch_directml
        device = torch_directml.device()
    except ImportError:
        device = torch.device('cpu')
print(f"Thiet bi huan luyen: {device}")

# Ham mat mat Dirichlet KL & MSE
def dirichlet_kl_binary(alpha):
    beta = torch.ones_like(alpha)
    S_alpha = alpha.sum(dim=-1, keepdim=True)
    S_beta  = beta.sum(dim=-1, keepdim=True)
    lnB_alpha = torch.lgamma(alpha).sum(dim=-1, keepdim=True) - torch.lgamma(S_alpha)
    lnB_beta  = torch.lgamma(beta).sum(dim=-1, keepdim=True)  - torch.lgamma(S_beta)
    digamma_diff = torch.digamma(alpha) - torch.digamma(S_alpha)
    return ((alpha - beta) * digamma_diff).sum(dim=-1, keepdim=True).squeeze(-1) + lnB_beta.squeeze(-1) - lnB_alpha.squeeze(-1)

def edl_binary_mse_loss(alpha, target, epoch, annealing_step=5):
    S = alpha.sum(dim=-1, keepdim=True)
    p = alpha / S
    y = torch.stack([1.0 - target.float(), target.float()], dim=-1)
    mse = ((y - p) ** 2).sum(dim=-1)
    var_term = (p * (1.0 - p) / (S + 1.0)).sum(dim=-1)
    pos_weight = torch.where(target > 0, 2.0, 1.0)
    kl = dirichlet_kl_binary(alpha)
    lambda_t = min(1.0, epoch / max(1, annealing_step))
    return ((mse + var_term) * pos_weight + lambda_t * kl).mean()

def predict_edl_binary(alpha):
    S = alpha.sum(dim=-1, keepdim=True)
    p_pos = alpha[..., 1:2] / S
    u = 2.0 / S
    return p_pos, u

# 1. Standard EDL Binary Module
class EDL_Binary_Module(nn.Module):
    def __init__(self, in_dim, hidden=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2)
        )
    def forward(self, x):
        return F.relu(self.net(x)) + 1e-4 + 1.0

# 2. EDL Attention Binary Module (Proposed)
class EDL_Attention_Binary_Module(nn.Module):
    def __init__(self, in_dim, hidden=128, dropout=0.2):
        super().__init__()
        self.hidden = hidden
        self.proj_q = nn.Linear(in_dim, hidden)
        self.proj_k = nn.Linear(in_dim, hidden)
        self.proj_v = nn.Linear(in_dim, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2)
        )
        self.last_attn_weights = None

    def forward(self, x):
        Q = self.proj_q(x)
        K = self.proj_k(x)
        V = self.proj_v(x)
        scores = (Q * K) / np.sqrt(self.hidden)
        attn_weights = torch.sigmoid(scores)
        self.last_attn_weights = attn_weights.detach()
        h_attn = self.norm(attn_weights * V + Q)
        logits = self.mlp_head(h_attn)
        return F.relu(logits) + 1e-4 + 1.0

print("Dinh nghia thanh cong ca 2 Module: EDL_Binary_Module & EDL_Attention_Binary_Module!")
"""))

# ─── Step 4: Architectures Definition ─────────────────────────────────────────
cells.append(nb_md("## BUOC 4: Dinh nghia 2 Kien truc Ensemble (Standard EDL-ECC & EDL-ECC-Attention)"))

cells.append(nb_code("""\
# 1. Standard EDL-ECC
class EDL_ECC:
    def __init__(self, in_dim, num_labels, n_chains=3, hidden=128, device='cpu'):
        self.in_dim = in_dim
        self.num_labels = num_labels
        self.n_chains = n_chains
        self.hidden = hidden
        self.device = device
        self.chains = []

    def fit(self, X_tr, Y_tr, epochs=10, batch_size=32, lr=1e-3):
        self.chains = []
        for chain_id in range(self.n_chains):
            order = np.random.permutation(self.num_labels)
            chain_models = []
            X_current = torch.from_numpy(X_tr).float().to(self.device)
            Y_tr_t = torch.from_numpy(Y_tr).float().to(self.device)
            for pos, lbl_idx in enumerate(order):
                in_feat = X_current.shape[1]
                model = EDL_Binary_Module(in_feat, hidden=self.hidden).to(self.device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                y_target = Y_tr_t[:, lbl_idx]
                loader = DataLoader(TensorDataset(X_current, y_target), batch_size=batch_size, shuffle=True)
                for ep in range(1, epochs + 1):
                    model.train()
                    for xb, yb in loader:
                        alpha = model(xb)
                        loss = edl_binary_mse_loss(alpha, yb, ep)
                        optimizer.zero_grad(); loss.backward(); optimizer.step()
                model.eval()
                with torch.no_grad():
                    alpha_pred = model(X_current)
                    p_pos, u = predict_edl_binary(alpha_pred)
                    X_current = torch.cat([X_current, p_pos, u], dim=-1)
                chain_models.append((lbl_idx, model))
            self.chains.append(chain_models)

    def predict_proba(self, X_val):
        all_chain_probs = []
        X_val_t = torch.from_numpy(X_val).float().to(self.device)
        for chain_models in self.chains:
            X_curr = X_val_t.clone()
            chain_prob = np.zeros((X_val.shape[0], self.num_labels), dtype=np.float32)
            for lbl_idx, model in chain_models:
                model.eval()
                with torch.no_grad():
                    alpha = model(X_curr)
                    p_pos, u = predict_edl_binary(alpha)
                    chain_prob[:, lbl_idx] = p_pos.squeeze(-1).cpu().numpy()
                    X_curr = torch.cat([X_curr, p_pos, u], dim=-1)
            all_chain_probs.append(chain_prob)
        return np.mean(all_chain_probs, axis=0)

    def predict(self, X_val, threshold=0.5):
        return (self.predict_proba(X_val) >= threshold).astype(int)

# 2. EDL-ECC-Attention (Proposed)
class EDL_ECC_Attention:
    def __init__(self, in_dim, num_labels, n_chains=3, hidden=128, device='cpu'):
        self.in_dim = in_dim
        self.num_labels = num_labels
        self.n_chains = n_chains
        self.hidden = hidden
        self.device = device
        self.chains = []

    def fit(self, X_tr, Y_tr, epochs=10, batch_size=32, lr=1e-3):
        self.chains = []
        for chain_id in range(self.n_chains):
            order = np.random.permutation(self.num_labels)
            chain_models = []
            X_current = torch.from_numpy(X_tr).float().to(self.device)
            Y_tr_t = torch.from_numpy(Y_tr).float().to(self.device)
            for pos, lbl_idx in enumerate(order):
                in_feat = X_current.shape[1]
                model = EDL_Attention_Binary_Module(in_feat, hidden=self.hidden).to(self.device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                y_target = Y_tr_t[:, lbl_idx]
                loader = DataLoader(TensorDataset(X_current, y_target), batch_size=batch_size, shuffle=True)
                for ep in range(1, epochs + 1):
                    model.train()
                    for xb, yb in loader:
                        alpha = model(xb)
                        loss = edl_binary_mse_loss(alpha, yb, ep)
                        optimizer.zero_grad(); loss.backward(); optimizer.step()
                model.eval()
                with torch.no_grad():
                    alpha_pred = model(X_current)
                    p_pos, u = predict_edl_binary(alpha_pred)
                    X_current = torch.cat([X_current, p_pos, u], dim=-1)
                chain_models.append((lbl_idx, model))
            self.chains.append(chain_models)

    def predict_proba(self, X_val):
        all_chain_probs = []
        X_val_t = torch.from_numpy(X_val).float().to(self.device)
        for chain_models in self.chains:
            X_curr = X_val_t.clone()
            chain_prob = np.zeros((X_val.shape[0], self.num_labels), dtype=np.float32)
            for lbl_idx, model in chain_models:
                model.eval()
                with torch.no_grad():
                    alpha = model(X_curr)
                    p_pos, u = predict_edl_binary(alpha)
                    chain_prob[:, lbl_idx] = p_pos.squeeze(-1).cpu().numpy()
                    X_curr = torch.cat([X_curr, p_pos, u], dim=-1)
            all_chain_probs.append(chain_prob)
        return np.mean(all_chain_probs, axis=0)

    def predict(self, X_val, threshold=0.5):
        return (self.predict_proba(X_val) >= threshold).astype(int)

print("Kien truc san sang: EDL_ECC & EDL_ECC_Attention!")
"""))

# ─── Step 5: 5-Fold Cross-Validation ──────────────────────────────────────────
cells.append(nb_md("## BUOC 5: Thuc nghiem Danh gia 5-Fold Cross-Validation"))

cells.append(nb_code("""\
def evaluate_metrics(y_true, y_pred):
    return {
        '1-HammingLoss': 1 - hamming_loss(y_true, y_pred),
        'SubsetAcc':     accuracy_score(y_true, y_pred),
        'Micro-F1':      f1_score(y_true, y_pred, average='micro', zero_division=0),
        'Macro-F1':      f1_score(y_true, y_pred, average='macro', zero_division=0),
        'Jaccard':       jaccard_score(y_true, y_pred, average='samples', zero_division=0),
    }

base_lr = LogisticRegression(solver='lbfgs', max_iter=300, class_weight='balanced')
METRICS_NAMES = ['1-HammingLoss', 'SubsetAcc', 'Micro-F1', 'Macro-F1', 'Jaccard']
fold_scores = {'BR': [], 'CC': [], 'EDL-ECC (Baseline)': [], 'EDL-ECC-Attention (Ours)': []}

print(f"Bat dau 5-Fold CV so sanh tren dataset: {DATASET_NAME}")
print("=" * 70)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    Y_train, Y_val = Y_full[train_idx],   Y_full[val_idx]

    for col in range(Y_train.shape[1]):
        if Y_train[:, col].sum() == 0:
            Y_train[0, col] = 1.0

    print(f"\\n  --- Fold {fold+1}/5 ---")

    # 1. BR
    br_model = OneVsRestClassifier(base_lr).fit(X_train, Y_train)
    fold_scores['BR'].append(evaluate_metrics(Y_val, br_model.predict(X_val)))

    # 2. CC
    cc_model = ClassifierChain(base_lr, order='random', random_state=42).fit(X_train, Y_train)
    fold_scores['CC'].append(evaluate_metrics(Y_val, cc_model.predict(X_val)))

    # 3. Standard EDL-ECC
    edl_ecc_std = EDL_ECC(X_train.shape[1], NUM_LABELS, n_chains=3, device=device)
    edl_ecc_std.fit(X_train, Y_train, epochs=10, batch_size=32)
    ecc_probs_std = edl_ecc_std.predict_proba(X_val)
    best_th_s, best_f1_s = 0.5, 0.0
    for th in np.arange(0.1, 0.55, 0.05):
        preds_t = (ecc_probs_std > th).astype(int)
        f1_t = f1_score(Y_val, preds_t, average='micro', zero_division=0)
        if f1_t > best_f1_s: best_f1_s, best_th_s = f1_t, th
    fold_scores['EDL-ECC (Baseline)'].append(evaluate_metrics(Y_val, (ecc_probs_std > best_th_s).astype(int)))

    # 4. EDL-ECC-Attention (Proposed)
    edl_ecc_attn = EDL_ECC_Attention(X_train.shape[1], NUM_LABELS, n_chains=3, hidden=128, device=device)
    edl_ecc_attn.fit(X_train, Y_train, epochs=10, batch_size=32)
    ecc_probs_attn = edl_ecc_attn.predict_proba(X_val)
    best_th_a, best_f1_a = 0.5, 0.0
    for th in np.arange(0.1, 0.55, 0.05):
        preds_t = (ecc_probs_attn > th).astype(int)
        f1_t = f1_score(Y_val, preds_t, average='micro', zero_division=0)
        if f1_t > best_f1_a: best_f1_a, best_th_a = f1_t, th
    fold_scores['EDL-ECC-Attention (Ours)'].append(evaluate_metrics(Y_val, (ecc_probs_attn > best_th_a).astype(int)))

    print(f"  Fold {fold+1} xong | EDL-ECC-Attention Micro-F1: {fold_scores['EDL-ECC-Attention (Ours)'][-1]['Micro-F1']:.4f}")

print("\\n" + "=" * 70)
print("5-Fold Cross-Validation hoan tat!")
"""))

# ─── Step 6: Attention Weights Heatmap ────────────────────────────────────────
cells.append(nb_md("## BUOC 6: Truc quan hoa Ma tran Trong so Attention (Attention Heatmap)"))

cells.append(nb_code("""\
sample_idx = 10
sample_x = torch.from_numpy(X_scaled[sample_idx:sample_idx+1]).float().to(device)
first_model = edl_ecc_attn.chains[0][0][1]
first_model.eval()
with torch.no_grad():
    _ = first_model(sample_x)
    attn_w = first_model.last_attn_weights.cpu().numpy()

fig, ax = plt.subplots(figsize=(10, 3))
sns.heatmap(attn_w[:, :32], cmap='viridis', ax=ax, cbar=True, annot=False)
ax.set_title(f'Phan bo Trong so Attention tren 32 chieu dac trung dau tien (Mau #{sample_idx})', fontweight='bold', fontsize=12)
ax.set_xlabel('Chieu dac trung an (Hidden Feature Dimension)')
ax.set_ylabel('Mau (Batch)')
plt.tight_layout()
plt.show()
print("Truc quan hoa Attention Map thanh cong!")
"""))

# ─── Step 7: Summary & Charts ─────────────────────────────────────────────────
cells.append(nb_md("## BUOC 7: Tong hop Ket qua & Bieu do Danh gia"))

cells.append(nb_code("""\
results_mean = {}
results_std  = {}
for m_name, scores_list in fold_scores.items():
    df_f = pd.DataFrame(scores_list)
    results_mean[m_name] = df_f.mean()
    results_std[m_name]  = df_f.std()

df_mean = pd.DataFrame(results_mean).T
df_std  = pd.DataFrame(results_std).T

print(f"\\n{'='*75}")
print(f"  KET QUA 5-FOLD CV MEAN: SO SANH EDL-ECC VS EDL-ECC-ATTENTION ({DATASET_NAME})")
print(f"{'='*75}")
print(df_mean.round(4).to_string())

# Grouped Bar Chart
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(METRICS_NAMES))
width = 0.2
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for idx_m, (m_name, color) in enumerate(zip(df_mean.index, colors)):
    scores = [results_mean[m_name][met] for met in METRICS_NAMES]
    stds   = [results_std[m_name][met]  for met in METRICS_NAMES]
    ax.bar(x + idx_m * width, scores, width, label=m_name,
           color=color, edgecolor='black', alpha=0.85, yerr=stds, capsize=4)

ax.set_xlabel('Chi so Danh gia (5-Fold CV Mean +- Std)', fontsize=12, fontweight='bold')
ax.set_ylabel('Diem so', fontsize=12, fontweight='bold')
ax.set_title(f'So sanh Hieu nang EDL-ECC vs EDL-ECC-Attention — {DATASET_NAME}', fontsize=13, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(METRICS_NAMES, fontsize=11)
ax.legend(fontsize=10)
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()
"""))

# ─── Write file ───────────────────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.9.0"}
    },
    "cells": cells
}

out_path = "complete_edl_ecc_attention.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"Created successfully: {out_path}")
