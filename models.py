import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import f1_score

from config import HIDDEN, DROPOUT, N_CHAINS, EPOCHS, BATCH_SIZE, LR, SEED, DEVICE


# ── EDL Loss & Helper functions 
def dirichlet_kl_binary(alpha):
    """Tính khoảng cách KL Divergence cho phân bố Dirichlet nhị phân."""
    beta     = torch.ones_like(alpha)
    S_alpha  = alpha.sum(dim=-1, keepdim=True)
    S_beta   = beta.sum(dim=-1, keepdim=True)
    lnB_a    = torch.lgamma(alpha).sum(-1, keepdim=True) - torch.lgamma(S_alpha)
    lnB_b    = torch.lgamma(beta).sum(-1, keepdim=True)  - torch.lgamma(S_beta)
    dg_diff  = torch.digamma(alpha) - torch.digamma(S_alpha)
    return ((alpha - beta) * dg_diff).sum(-1, keepdim=True).squeeze(-1) \
           + lnB_b.squeeze(-1) - lnB_a.squeeze(-1)


def edl_binary_mse_loss(alpha, target, epoch, annealing_step=5):
    """Loss EDL Binary MSE + KL Regularization."""
    S         = alpha.sum(dim=-1, keepdim=True)
    p         = alpha / S
    y         = torch.stack([1.0 - target.float(), target.float()], dim=-1)
    mse       = ((y - p) ** 2).sum(dim=-1)
    var_term  = (p * (1.0 - p) / (S + 1.0)).sum(dim=-1)
    pos_w     = torch.where(target > 0, 2.0, 1.0)
    kl        = dirichlet_kl_binary(alpha)
    lambda_t  = min(1.0, epoch / max(1, annealing_step))
    return ((mse + var_term) * pos_w + lambda_t * kl).mean()


def predict_edl_binary(alpha):
    """(p_pos) and (u) -> alpha Dirichlet."""
    S     = alpha.sum(dim=-1, keepdim=True)
    p_pos = alpha[..., 1:2] / S
    u     = 2.0 / S
    return p_pos, u


# ── EDL Model Architecture
class EDLModel(nn.Module):
    """EDL Binary Classifier."""
    def __init__(self, in_dim, num_labels=1, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden // 2, num_labels * 2)

    def forward(self, x):
        h = self.net(x)
        e = F.relu(self.out(h)) + 1e-4   
        return e + 1.0                   


# ── EDL-ECC Classifier Ensemble
class EDL_ECC:
    """Ensemble of Classifier Chains + EDL."""
    def __init__(self, in_dim, num_labels, n_chains=N_CHAINS,
                 hidden=HIDDEN, dev='cpu'):
        self.in_dim     = in_dim
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


# ── Hàm Huấn luyện & Đánh giá từng mô hình
def train_eval_br(X_tr, Y_tr, X_va, Y_va, base_estimator=None):
    """Binary Relevance (BR)."""
    if base_estimator is None:
        base_estimator = LogisticRegression(solver='lbfgs', max_iter=300, class_weight='balanced')
    br = OneVsRestClassifier(base_estimator)
    br.fit(X_tr, Y_tr)
    br_preds = br.predict(X_va)
    return br_preds


def train_eval_cc(X_tr, Y_tr, X_va, Y_va, base_estimator=None, seed=SEED):
    """Classifier Chains (CC)."""
    if base_estimator is None:
        base_estimator = LogisticRegression(solver='lbfgs', max_iter=300, class_weight='balanced')
    cc = ClassifierChain(base_estimator, order='random', random_state=seed)
    cc.fit(X_tr, Y_tr)
    cc_preds = cc.predict(X_va)
    return cc_preds


def train_eval_edl_ecc(X_tr, Y_tr, X_va, Y_va, num_labels, dev=DEVICE,epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR):
    """
    EDL-ECC, kèm tối ưu ngưỡng và trích xuất độ bất định.
    """
    edl_ecc = EDL_ECC(X_tr.shape[1], num_labels, dev=dev)
    edl_ecc.fit(X_tr, Y_tr, epochs=epochs, batch_size=batch_size, lr=lr)
    proba = edl_ecc.predict_proba(X_va)
    best_th, best_f1 = 0.5, 0.0
    for th in np.arange(0.1, 0.55, 0.05):
        pt = (proba >= th).astype(int)
        if pt.sum() == 0:
            continue
        try:
            f1 = f1_score(Y_va.astype(int), pt, average='micro', zero_division=0)
            if f1 > best_f1:
                best_f1, best_th = f1, th
        except ValueError:
            continue

    edl_preds = (proba >= best_th).astype(int)
    unc_correct, unc_wrong = [], []
    try:
        X_va_t = torch.from_numpy(X_va).float().to(dev)
        first_chain = edl_ecc.chains[0]
        lbl_idx_0, model_0 = first_chain[0]
        model_0.eval()
        with torch.no_grad():
            alpha0 = model_0(X_va_t)
            _, u0 = predict_edl_binary(alpha0)
        u0_np = u0.squeeze(-1).cpu().numpy()
        correct_mask = (edl_preds[:, lbl_idx_0] == Y_va[:, lbl_idx_0])
        unc_correct = u0_np[correct_mask].tolist()
        unc_wrong = u0_np[~correct_mask].tolist()
    except Exception:
        pass

    return edl_preds, unc_correct, unc_wrong
