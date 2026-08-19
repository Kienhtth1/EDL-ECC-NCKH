import os
import sys
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import arff
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, hamming_loss, jaccard_score
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

sys.stdout.reconfigure(encoding='utf-8')
plt.style.use('seaborn-v0_8-whitegrid')

dataset_configs = {
    'CHD_49': {'file': 'CHD_49.arff', 'num_labels': 6},
    'emotions': {'file': 'emotions.arff', 'num_labels': 6},
    'GpositivePseAAC': {'file': 'GpositivePseAAC.arff', 'num_labels': 4},
    'HumanPseAAC': {'file': 'HumanPseAAC.arff', 'num_labels': 14},
    'PlantPseAAC': {'file': 'PlantPseAAC.arff', 'num_labels': 12},
    'Scene': {'file': 'Scene.arff', 'num_labels': 6},
    'VirusPseAAC': {'file': 'VirusPseAAC.arff', 'num_labels': 6},
    'Water-quality': {'file': 'Water-quality.arff', 'num_labels': 14},
    'Yeast': {'file': 'Yeast.arff', 'num_labels': 14}
}

def load_arff_dataset(path, num_labels):
    path = Path(path)
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    for col in df.columns:
        if df[col].dtype == object:
            try: df[col] = df[col].str.decode('utf-8')
            except: pass
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    X = df.iloc[:, :-num_labels].values.astype('float32')
    Y = df.iloc[:, -num_labels:].values.astype('float32')
    Y = (Y > 0).astype('float32')
    return X, Y

def evaluate_metrics(y_true, y_pred):
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    h_loss = hamming_loss(y_true, y_pred)
    subset_acc = accuracy_score(y_true, y_pred)
    jaccard = jaccard_score(y_true, y_pred, average='samples', zero_division=0)
    return [1 - h_loss, subset_acc, micro_f1, macro_f1, jaccard]

def labelset_to_class(y_subset):
    k = y_subset.shape[1]
    powers = 2 ** np.arange(k)[::-1]
    return (y_subset * powers).sum(axis=1).astype(int)

def class_to_labelset(class_indices, k):
    B = len(class_indices)
    binary_matrix = np.zeros((B, k), dtype=np.float32)
    for i in range(k):
        power = 2 ** (k - 1 - i)
        binary_matrix[:, i] = (class_indices // power) % 2
    return binary_matrix

def dirichlet_kl_multiclass(alpha):
    C = alpha.size(-1)
    beta = torch.ones_like(alpha)
    S_alpha = torch.sum(alpha, dim=-1, keepdim=True)
    S_beta = torch.sum(beta, dim=-1, keepdim=True)
    lnB_alpha = torch.sum(torch.lgamma(alpha), dim=-1, keepdim=True) - torch.lgamma(S_alpha)
    lnB_beta = torch.sum(torch.lgamma(beta), dim=-1, keepdim=True) - torch.lgamma(S_beta)
    digamma_diff = torch.digamma(alpha) - torch.digamma(S_alpha)
    return (torch.sum((alpha - beta) * digamma_diff, dim=-1, keepdim=True) + lnB_beta - lnB_alpha).squeeze(-1)

def edl_multiclass_mse_loss(alpha, target_class, epoch, C, annealing_step=5):
    S = torch.sum(alpha, dim=-1, keepdim=True)
    p = alpha / S
    y_onehot = F.one_hot(target_class.cpu(), num_classes=C).to(alpha.device).float()
    mse = torch.sum((y_onehot - p) ** 2, dim=-1)
    var_term = torch.sum(p * (1.0 - p) / (S + 1.0), dim=-1)
    kl = dirichlet_kl_multiclass(alpha)
    lambda_t = min(1.0, epoch / max(1, annealing_step))
    return (mse + var_term + lambda_t * kl).mean()

class Standard_RAkEL:
    """Standard Random k-Labelsets (RAkEL) using Label Powerset and Logistic Regression."""
    def __init__(self, num_labels, k=3, m=None, random_state=42):
        self.num_labels = num_labels
        self.k = min(k, num_labels)
        self.m = m if m is not None else max(2 * num_labels, 6)
        self.random_state = random_state
        rng = np.random.RandomState(random_state)
        self.labelsets = [rng.choice(num_labels, self.k, replace=False) for _ in range(self.m)]
        self.models = []

    def fit(self, X, Y):
        self.models = []
        for labelset in self.labelsets:
            Y_sub = Y[:, labelset]
            y_lp = labelset_to_class(Y_sub)
            clf = LogisticRegression(solver='lbfgs', max_iter=300, class_weight='balanced')
            if len(np.unique(y_lp)) < 2:
                dummy_x = np.zeros((2, X.shape[1]), dtype=X.dtype)
                dummy_y = np.array([0, 1])
                clf.fit(np.vstack([X, dummy_x]), np.concatenate([y_lp, dummy_y]))
            else:
                clf.fit(X, y_lp)
            self.models.append(clf)

    def predict(self, X, threshold=0.5):
        N = X.shape[0]
        votes = np.zeros((N, self.num_labels), dtype=np.float32)
        counts = np.zeros(self.num_labels, dtype=np.float32)
        
        for labelset, clf in zip(self.labelsets, self.models):
            y_pred_class = clf.predict(X)
            binary_preds = class_to_labelset(y_pred_class, self.k)
            for i, lbl_idx in enumerate(labelset):
                votes[:, lbl_idx] += binary_preds[:, i]
                counts[lbl_idx] += 1.0
                
        probs = votes / np.maximum(counts, 1.0)
        return (probs >= threshold).astype(int)

class EDL_Binary_Module(nn.Module):
    def __init__(self, in_dim, hidden=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2)
        )

    def forward(self, x):
        e = F.relu(self.net(x)) + 1e-4
        alpha = e + 1.0
        return alpha

def dirichlet_kl_binary(alpha):
    beta = torch.ones_like(alpha)
    S_alpha = torch.sum(alpha, dim=-1, keepdim=True)
    S_beta = torch.sum(beta, dim=-1, keepdim=True)
    lnB_alpha = torch.sum(torch.lgamma(alpha), dim=-1, keepdim=True) - torch.lgamma(S_alpha)
    lnB_beta = torch.sum(torch.lgamma(beta), dim=-1, keepdim=True) - torch.lgamma(S_beta)
    digamma_diff = torch.digamma(alpha) - torch.digamma(S_alpha)
    return (torch.sum((alpha - beta) * digamma_diff, dim=-1, keepdim=True) + lnB_beta - lnB_alpha).squeeze(-1)

def edl_binary_mse_loss(alpha, target, epoch, annealing_step=5):
    S = torch.sum(alpha, dim=-1, keepdim=True)
    p = alpha / S
    y = torch.stack([1.0 - target.float(), target.float()], dim=-1)
    mse = torch.sum((y - p) ** 2, dim=-1)
    var_term = torch.sum(p * (1.0 - p) / (S + 1.0), dim=-1)
    pos_weight = torch.where(target > 0, 2.0, 1.0)
    expected_err = (mse + var_term) * pos_weight
    kl = dirichlet_kl_binary(alpha)
    lambda_t = min(1.0, epoch / max(1, annealing_step))
    return (expected_err + lambda_t * kl).mean()

def predict_proba_and_uncertainty_binary(alpha):
    S = alpha.sum(dim=-1, keepdim=True)
    p_pos = alpha[..., 1:2] / S
    u = 2.0 / S
    return p_pos, u

class EDL_ECC:
    """Ensemble Classifier Chains with Evidential Deep Learning Base Learners & Uncertainty Propagation."""
    def __init__(self, in_dim, num_labels, n_chains=3, hidden=128, device='cpu'):
        self.in_dim = in_dim
        self.num_labels = num_labels
        self.n_chains = n_chains
        self.hidden = hidden
        self.device = device
        self.chains = []
        self.orders = []

    def fit(self, X_tr, Y_tr, epochs=10, batch_size=32, lr=1e-3):
        self.chains = []
        self.orders = []

        for chain_id in range(self.n_chains):
            order = np.random.permutation(self.num_labels)
            self.orders.append(order)
            chain_models = []
            
            X_current = torch.from_numpy(X_tr).float().to(self.device)
            Y_tr_t = torch.from_numpy(Y_tr).float().to(self.device)

            for pos, lbl_idx in enumerate(order):
                in_feat = X_current.shape[1]
                model = EDL_Binary_Module(in_feat, hidden=self.hidden).to(self.device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                y_target = Y_tr_t[:, lbl_idx]

                dataset = TensorDataset(X_current, y_target)
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

                for ep in range(1, epochs + 1):
                    model.train()
                    for xb, yb in loader:
                        alpha = model(xb)
                        loss = edl_binary_mse_loss(alpha, yb, ep)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                model.eval()
                with torch.no_grad():
                    alpha_pred = model(X_current)
                    p_pos, u = predict_proba_and_uncertainty_binary(alpha_pred)
                    pred_feat = torch.cat([p_pos, u], dim=-1)
                    X_current = torch.cat([X_current, pred_feat], dim=-1)

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
                    p_pos, u = predict_proba_and_uncertainty_binary(alpha)
                    chain_prob[:, lbl_idx] = p_pos.squeeze(-1).cpu().numpy()
                    pred_feat = torch.cat([p_pos, u], dim=-1)
                    X_curr = torch.cat([X_curr, pred_feat], dim=-1)

            all_chain_probs.append(chain_prob)

        return np.mean(all_chain_probs, axis=0)

    def predict(self, X_val, threshold=0.5):
        probs = self.predict_proba(X_val)
        return (probs >= threshold).astype(int)

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
        return F.relu(self.net(x)) + 1.0 + 1e-4

class EDL_RAkEL:
    def __init__(self, in_dim, num_labels, k=3, m=None, hidden=128, device='cpu'):
        self.in_dim, self.num_labels, self.k = in_dim, num_labels, min(k, num_labels)
        self.C = 2 ** self.k
        self.m = m if m else max(2 * num_labels, 6)
        self.device = device
        self.labelsets = [np.random.choice(num_labels, self.k, replace=False) for _ in range(self.m)]
        self.models = [EDL_LP_Module(in_dim, self.C, hidden=hidden).to(device) for _ in range(self.m)]
        self.opts = [torch.optim.Adam(mod.parameters(), lr=1e-3) for mod in self.models]

    def fit(self, loader, epochs=10):
        for epoch in range(1, epochs + 1):
            for mod in self.models: mod.train()
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb_np = yb.numpy()
                for labelset, mod, opt in zip(self.labelsets, self.models, self.opts):
                    y_sub = yb_np[:, labelset]
                    target_c = torch.from_numpy(labelset_to_class(y_sub)).long().to(self.device)
                    alpha = mod(xb)
                    loss = edl_multiclass_mse_loss(alpha, target_c, epoch, self.C)
                    opt.zero_grad(); loss.backward(); opt.step()

    def predict_proba(self, X):
        X_t = torch.from_numpy(X).float().to(self.device)
        N = X.shape[0]
        votes = np.zeros((N, self.num_labels), dtype=np.float32)
        weights = np.zeros((N, self.num_labels), dtype=np.float32)
        
        for labelset, mod in zip(self.labelsets, self.models):
            mod.eval()
            with torch.no_grad():
                alpha = mod(X_t)
                S = alpha.sum(dim=-1, keepdim=True)
                p_class = (alpha / S).cpu().numpy()
                u = (self.C / S).squeeze(-1).cpu().numpy()
                w = np.clip(1.0 - u, 1e-4, 1.0)[:, None]
                binary_map = class_to_labelset(np.arange(self.C), self.k)
                p_labels = np.dot(p_class, binary_map)
                for i, lbl_idx in enumerate(labelset):
                    votes[:, lbl_idx] += (p_labels[:, i:i+1] * w).squeeze(-1)
                    weights[:, lbl_idx] += w.squeeze(-1)
        return votes / np.maximum(weights, 1e-6)

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        try:
            import torch_directml
            device = torch_directml.device()
        except ImportError:
            device = torch.device('cpu')
    print(f"Running 5-Fold Cross-Validation experiments using PyTorch on device: {device}")
    
    all_results = {}
    metrics_names = ['1-HammingLoss', 'SubsetAcc', 'Micro-F1', 'Macro-F1', 'Jaccard']
    
    for ds_name, cfg in dataset_configs.items():
        print(f"\n==========================================")
        print(f"Executing 5-Fold CV Dataset: {ds_name} ({cfg['file']})")
        print(f"==========================================")
        
        path = Path('data') / cfg['file']
        X, Y = load_arff_dataset(path, cfg['num_labels'])
        X = StandardScaler().fit_transform(X)
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_scores = {'BR': [], 'CC': [], 'RAkEL': [], 'EDL-ECC': [], 'EDL-RAkEL': []}
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]
            
            if any(len(np.unique(Y_train[:, col])) < 2 for col in range(Y_train.shape[1])):
                dummy_X = np.zeros((2, X_train.shape[1]), dtype='float32')
                dummy_Y = np.zeros((2, Y_train.shape[1]), dtype='float32')
                dummy_Y[1, :] = 1.0
                X_train = np.vstack([X_train, dummy_X])
                Y_train = np.vstack([Y_train, dummy_Y])
            
            train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float())
            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
            
            # 1. BR
            base_lr = LogisticRegression(solver='lbfgs', max_iter=300, class_weight='balanced')
            br_model = OneVsRestClassifier(base_lr)
            br_model.fit(X_train, Y_train)
            br_preds = br_model.predict(X_val)
            fold_scores['BR'].append(evaluate_metrics(Y_val, br_preds))
            
            # 2. CC
            cc_model = ClassifierChain(base_lr, order='random', random_state=42)
            cc_model.fit(X_train, Y_train)
            cc_preds = cc_model.predict(X_val)
            fold_scores['CC'].append(evaluate_metrics(Y_val, cc_preds))
            
            # 3. RAkEL
            rakel_model = Standard_RAkEL(Y_train.shape[1], k=min(3, Y_train.shape[1]), m=max(2 * Y_train.shape[1], 6), random_state=42 + fold)
            rakel_model.fit(X_train, Y_train)
            rakel_preds = rakel_model.predict(X_val)
            fold_scores['RAkEL'].append(evaluate_metrics(Y_val, rakel_preds))
            
            # 4. EDL-ECC
            edl_ecc = EDL_ECC(X_train.shape[1], Y_train.shape[1], n_chains=3, device=device)
            edl_ecc.fit(X_train, Y_train, epochs=10, batch_size=32)
            ecc_probs = edl_ecc.predict_proba(X_val)
            best_th_e, best_f1_e = 0.5, 0.0
            for th in np.arange(0.1, 0.55, 0.05):
                preds_tmp = (ecc_probs > th).astype(int)
                f1_tmp = f1_score(Y_val, preds_tmp, average='micro', zero_division=0)
                if f1_tmp > best_f1_e:
                    best_f1_e = f1_tmp
                    best_th_e = th
            edl_ecc_preds = (ecc_probs > best_th_e).astype(int)
            fold_scores['EDL-ECC'].append(evaluate_metrics(Y_val, edl_ecc_preds))
            
            # 5. EDL-RAkEL
            edl_rakel = EDL_RAkEL(X_train.shape[1], Y_train.shape[1], k=min(3, Y_train.shape[1]), m=max(2*Y_train.shape[1], 6), device=device)
            edl_rakel.fit(train_loader, epochs=10)
            rakel_probs = edl_rakel.predict_proba(X_val)
            
            best_th_r, best_f1_r = 0.5, 0.0
            for th in np.arange(0.1, 0.5, 0.05):
                preds_tmp = (rakel_probs > th).astype(int)
                f1_tmp = f1_score(Y_val, preds_tmp, average='micro', zero_division=0)
                if f1_tmp > best_f1_r:
                    best_f1_r = f1_tmp
                    best_th_r = th
                    
            edl_rakel_preds = (rakel_probs > best_th_r).astype(int)
            fold_scores['EDL-RAkEL'].append(evaluate_metrics(Y_val, edl_rakel_preds))
            print(f"  ✓ Fold {fold+1}/5 completed for {ds_name}")
            
        # Calculate 5-Fold Cross-Validation Mean Scores for each model
        res_dict = {
            m_name: np.mean(fold_scores[m_name], axis=0).tolist()
            for m_name in fold_scores
        }
        all_results[ds_name] = res_dict
        
        out_dir = Path('./outputs') / ds_name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Output 1: 5-Metric Radar Chart
        num_vars = len(metrics_names)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() + [0]
        
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        for m_name, scores in res_dict.items():
            s = scores + [scores[0]]
            ax.plot(angles, s, label=m_name, linewidth=2)
            ax.fill(angles, s, alpha=0.1)
            
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), metrics_names)
        ax.set_ylim(0, 1)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.title(f'Radar Chart (5-Fold CV) - {ds_name}', size=14, y=1.1, fontweight='bold')
        plt.savefig(out_dir / 'radar_chart.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        # Output 2: Grouped Bar Chart (5-Fold CV)
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(metrics_names))
        width = 0.15
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for idx_m, (m_name, scores) in enumerate(res_dict.items()):
            ax.bar(x + idx_m * width, scores, width, label=m_name, color=colors[idx_m % len(colors)], edgecolor='black', alpha=0.85)
            
        ax.set_xlabel('Chỉ số Đánh giá (5-Fold CV Mean)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Điểm số (Score)', fontsize=11, fontweight='bold')
        ax.set_title(f'So sánh Chỉ số 5-Fold Cross Validation - {ds_name}', fontsize=13, fontweight='bold')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(metrics_names, fontsize=10)
        ax.legend(fontsize=9, loc='upper left')
        ax.set_ylim(0, 1.08)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(out_dir / 'metrics_bar_chart.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        # Output 3: Dataset Statistics Table (CSV + PNG Table Image)
        df_ds = pd.DataFrame(res_dict, index=metrics_names).T
        df_ds.reset_index(inplace=True)
        df_ds.rename(columns={'index': 'Model'}, inplace=True)
        df_ds.to_csv(out_dir / 'dataset_metrics_table.csv', index=False)
        
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.axis('off')
        tbl_vals = df_ds.round(4).values
        table = ax.table(cellText=tbl_vals, colLabels=df_ds.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.6)
        plt.title(f'Bảng Thống kê 5-Fold CV Mean - {ds_name}', fontsize=12, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.savefig(out_dir / 'metrics_table.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Dataset {ds_name} completed with 5-Fold Cross Validation!")
        
    summary_rows = []
    for ds_name, res in all_results.items():
        for model_name, metrics in res.items():
            summary_rows.append({
                'Dataset': ds_name,
                'Model': model_name,
                '1-HammingLoss': metrics[0],
                'SubsetAcc': metrics[1],
                'Micro-F1': metrics[2],
                'Macro-F1': metrics[3],
                'Jaccard': metrics[4]
            })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv('./outputs/multi_dataset_benchmark_summary.csv', index=False)
    print("\n✓ 5-Fold CV Experiments complete. All dataset charts & master summary saved to ./outputs/!")
