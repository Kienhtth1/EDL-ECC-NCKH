import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, hamming_loss, jaccard_score

from config import METRICS_NAMES, SEED, OUTPUTS_DIR, DEVICE
from dataset import ensure_binary_classes
from models import train_eval_br, train_eval_cc, train_eval_edl_ecc


def evaluate(y_true, y_pred):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    return {
        '1-HL':     1 - hamming_loss(y_true, y_pred),
        'SubAcc':   accuracy_score(y_true, y_pred),
        'Micro-F1': f1_score(y_true, y_pred, average='micro',  zero_division=0),
        'Macro-F1': f1_score(y_true, y_pred, average='macro',  zero_division=0),
        'Jaccard':  jaccard_score(y_true, y_pred, average='samples', zero_division=0),
    }


def print_dataset_report(res, total_time, ds_name, out_dir):
    print()
    hdr = f"  {'Metric':<12}{'BR':>10}{'CC':>10}{'EDL-ECC':>12}"
    hdr += f"{'Δ(EDL-BR)':>12}{'Δ(EDL-CC)':>12}"
    print(hdr)
    print(f"  {'-' * 68}")
    for m in METRICS_NAMES:
        bv   = res['BR'][m]
        cv   = res['CC'][m]
        ev   = res['EDL-ECC'][m]
        d_br = ev - bv
        d_cc = ev - cv
        sb = '+' if d_br >= 0 else ''
        sc = '+' if d_cc >= 0 else ''
        best_val = max(bv, cv, ev)
        star_edl = ' *' if ev == best_val else '  '
        print(f"  {m:<12}{bv:>10.4f}{cv:>10.4f}{ev:>10.4f}{star_edl}"
              f"{sb + str(round(d_br, 4)):>12}{sc + str(round(d_cc, 4)):>12}")


def generate_and_save_summary(all_results, all_std_results, out_dir=OUTPUTS_DIR, json_path=Path('results_run.json')):
    summary_rows = []
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
              f"{winner:>10}{sc + str(round(d_cc, 4)):>12}")
        summary_rows.append({
            'Dataset': ds_name,
            'BR': bv,
            'CC': cv,
            'EDL-ECC': ev,
            'Winner': winner,
            'Delta(EDL-CC)': round(d_cc, 4)
        })

    pd.DataFrame(summary_rows).to_csv(out_dir / 'overall_summary_table.csv', index=False)
    json_path.write_text(
        json.dumps({
            'mean': all_results,
            'std': all_std_results
        }, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )


def run_kfold_cv(X, Y, num_labels, ds_name, dev=DEVICE, n_splits=5, seed=SEED):
    S2 = '-' * 80
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    fold_scores = {'BR': [], 'CC': [], 'EDL-ECC': []}
    fold_unc_correct = []
    fold_unc_wrong   = []

    base_lr = LogisticRegression(solver='lbfgs', max_iter=300, class_weight='balanced')
    t_start = time.time()

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        Y_tr, Y_va = Y[tr_idx], Y[va_idx]

        # Đảm bảo mỗi nhãn có ít nhất 2 lớp 0 và 1 trong train
        ensure_binary_classes(Y_tr)

        print(f'  Fold {fold+1}/{n_splits} ', end='', flush=True)
        tf = time.time()

        # 1. Huấn luyện & Đánh giá BR
        br_preds = train_eval_br(X_tr, Y_tr, X_va, Y_va, base_estimator=base_lr)
        fold_scores['BR'].append(evaluate(Y_va, br_preds))

        # 2. Huấn luyện & Đánh giá CC
        cc_preds = train_eval_cc(X_tr, Y_tr, X_va, Y_va, base_estimator=base_lr, seed=seed)
        fold_scores['CC'].append(evaluate(Y_va, cc_preds))

        # 3. Huấn luyện & Đánh giá EDL-ECC
        edl_preds, unc_corr, unc_wrg = train_eval_edl_ecc(
            X_tr, Y_tr, X_va, Y_va, num_labels=num_labels, dev=dev
        )
        fold_scores['EDL-ECC'].append(evaluate(Y_va, edl_preds))
        fold_unc_correct.extend(unc_corr)
        fold_unc_wrong.extend(unc_wrg)

        elapsed = time.time() - tf
        mf1 = {k: fold_scores[k][-1]['Micro-F1'] for k in ['BR', 'CC', 'EDL-ECC']}
        print(f'[{elapsed:>5.1f}s]  Micro-F1:  '
              f'BR={mf1["BR"]:.4f}  CC={mf1["CC"]:.4f}  EDL-ECC={mf1["EDL-ECC"]:.4f}')

    total_time = time.time() - t_start

    df_mean_dict = {}
    df_std_dict  = {}
    for m_name in ['BR', 'CC', 'EDL-ECC']:
        df_fold_m = pd.DataFrame(fold_scores[m_name])
        df_mean_dict[m_name] = df_fold_m.mean()
        df_std_dict[m_name]  = df_fold_m.std()

    df_mean = pd.DataFrame(df_mean_dict).T
    df_std  = pd.DataFrame(df_std_dict).T

    return df_mean, df_std, fold_unc_correct, fold_unc_wrong, total_time
