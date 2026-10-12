import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, hamming_loss, jaccard_score

from config import (
    METRICS_NAMES, SEED, OUTPUTS_DIR, DEVICE, THRESHOLD,
    DATA_DIR, DATASET_CONFIGS, N_CHAINS, EPOCHS, BATCH_SIZE, LR,
    find_file
)
from dataset import load_arff, ensure_binary_classes
from models import train_eval_br, train_eval_cc, train_eval_ecc, train_eval_edl_ecc


def evaluate(y_true, y_pred):
    """
    Tính 5 chỉ số đánh giá tiêu chuẩn đa nhãn.
    Ngưỡng quyết định mặc định là 0.5 cho toàn bộ các mô hình.
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    return {
        '1-HL':     round(1 - hamming_loss(y_true, y_pred), 4),
        'SubAcc':   round(accuracy_score(y_true, y_pred), 4),
        'Micro-F1': round(f1_score(y_true, y_pred, average='micro', zero_division=0), 4),
        'Macro-F1': round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4),
        'Jaccard':  round(jaccard_score(y_true, y_pred, average='samples', zero_division=0), 4),
    }


# =========================================================================
# 1. HÀM ĐÁNH GIÁ TỪNG MÔ HÌNH RIÊNG BIỆT (X, Y INPUT)
# =========================================================================

def evaluate_br(X, Y, n_splits=5, seed=SEED, base_estimator=None):
    """
    Đánh giá riêng biệt mô hình Binary Relevance (BR) bằng 5-Fold CV.
    Ngưỡng phân loại: 0.5
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        Y_tr, Y_va = Y[tr_idx], Y[va_idx]
        ensure_binary_classes(Y_tr)

        br_preds = train_eval_br(X_tr, Y_tr, X_va, Y_va, base_estimator=base_estimator)
        scores.append(evaluate(Y_va, br_preds))

    df_scores = pd.DataFrame(scores)
    return df_scores.mean(), df_scores.std(), scores


def evaluate_cc(X, Y, n_splits=5, seed=SEED, base_estimator=None):
    """
    Đánh giá riêng biệt mô hình Classifier Chains (CC) bằng 5-Fold CV.
    Ngưỡng phân loại: 0.5
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        Y_tr, Y_va = Y[tr_idx], Y[va_idx]
        ensure_binary_classes(Y_tr)

        cc_preds = train_eval_cc(X_tr, Y_tr, X_va, Y_va, base_estimator=base_estimator, seed=seed + fold)
        scores.append(evaluate(Y_va, cc_preds))

    df_scores = pd.DataFrame(scores)
    return df_scores.mean(), df_scores.std(), scores


def evaluate_ecc(X, Y, n_chains=N_CHAINS, n_splits=5, seed=SEED, base_estimator=None):
    """
    Đánh giá riêng biệt mô hình Ensemble of Classifier Chains (ECC) bằng 5-Fold CV.
    Ngưỡng phân loại: 0.5
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        Y_tr, Y_va = Y[tr_idx], Y[va_idx]
        ensure_binary_classes(Y_tr)

        ecc_preds = train_eval_ecc(X_tr, Y_tr, X_va, Y_va, n_chains=n_chains,
                                   base_estimator=base_estimator, seed=seed, fold=fold)
        scores.append(evaluate(Y_va, ecc_preds))

    df_scores = pd.DataFrame(scores)
    return df_scores.mean(), df_scores.std(), scores


def evaluate_edl_ecc(X, Y, num_labels, n_chains=N_CHAINS, n_splits=5,
                     dev=DEVICE, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, seed=SEED):
    """
    Đánh giá riêng biệt mô hình Evidential Deep Learning - ECC (EDL-ECC) bằng 5-Fold CV.
    Ngưỡng phân loại: cố định 0.5 (loại bỏ data leakage và đảm bảo so sánh công bằng).
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    scores = []
    unc_correct = []
    unc_wrong = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        Y_tr, Y_va = Y[tr_idx], Y[va_idx]
        ensure_binary_classes(Y_tr)

        edl_preds, u_corr, u_wrg = train_eval_edl_ecc(
            X_tr, Y_tr, X_va, Y_va, num_labels=num_labels,
            n_chains=n_chains, dev=dev, epochs=epochs,
            batch_size=batch_size, lr=lr
        )
        scores.append(evaluate(Y_va, edl_preds))
        unc_correct.extend(u_corr)
        unc_wrong.extend(u_wrg)

    df_scores = pd.DataFrame(scores)
    return df_scores.mean(), df_scores.std(), scores, unc_correct, unc_wrong


# =========================================================================
# 2. HÀM ĐÁNH GIÁ TRÊN FILE TẬP DỮ LIỆU (DATASET-LEVEL WRAPPERS)
# =========================================================================

def _prepare_dataset(ds_name, cfg=None):
    if cfg is None:
        cfg = DATASET_CONFIGS[ds_name]
    arff_path = find_file(cfg['file'])
    if arff_path is None or not arff_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file {cfg['file']}")
    num_labels = cfg['num_labels']
    X, Y, _ = load_arff(arff_path, num_labels)
    X = StandardScaler().fit_transform(X).astype('float32')
    return X, Y, num_labels


def eval_br_dataset(ds_name, cfg=None, n_splits=5, seed=SEED):
    """Đánh giá BR trên dataset được chỉ định."""
    t0 = time.time()
    X, Y, _ = _prepare_dataset(ds_name, cfg)
    df_mean, df_std, _ = evaluate_br(X, Y, n_splits=n_splits, seed=seed)
    elapsed = time.time() - t0
    print(f"[BR] {ds_name} ({elapsed:.1f}s) -> Micro-F1: {df_mean['Micro-F1']:.4f}, Macro-F1: {df_mean['Macro-F1']:.4f}")
    return df_mean.to_dict()


def eval_cc_dataset(ds_name, cfg=None, n_splits=5, seed=SEED):
    """Đánh giá CC trên dataset được chỉ định."""
    t0 = time.time()
    X, Y, _ = _prepare_dataset(ds_name, cfg)
    df_mean, df_std, _ = evaluate_cc(X, Y, n_splits=n_splits, seed=seed)
    elapsed = time.time() - t0
    print(f"[CC] {ds_name} ({elapsed:.1f}s) -> Micro-F1: {df_mean['Micro-F1']:.4f}, Macro-F1: {df_mean['Macro-F1']:.4f}")
    return df_mean.to_dict()


def eval_ecc_dataset(ds_name, cfg=None, n_chains=N_CHAINS, n_splits=5, seed=SEED):
    """Đánh giá ECC trên dataset được chỉ định."""
    t0 = time.time()
    X, Y, _ = _prepare_dataset(ds_name, cfg)
    df_mean, df_std, _ = evaluate_ecc(X, Y, n_chains=n_chains, n_splits=n_splits, seed=seed)
    elapsed = time.time() - t0
    print(f"[ECC n={n_chains}] {ds_name} ({elapsed:.1f}s) -> Micro-F1: {df_mean['Micro-F1']:.4f}, Macro-F1: {df_mean['Macro-F1']:.4f}")
    return df_mean.to_dict()


def eval_edl_ecc_dataset(ds_name, cfg=None, n_chains=N_CHAINS, n_splits=5,
                         epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, seed=SEED, dev=DEVICE):
    """Đánh giá EDL-ECC trên dataset được chỉ định."""
    t0 = time.time()
    X, Y, num_labels = _prepare_dataset(ds_name, cfg)
    df_mean, df_std, _, unc_c, unc_w = evaluate_edl_ecc(
        X, Y, num_labels, n_chains=n_chains, n_splits=n_splits,
        dev=dev, epochs=epochs, batch_size=batch_size, lr=lr, seed=seed
    )
    elapsed = time.time() - t0
    print(f"[EDL-ECC n={n_chains}] {ds_name} ({elapsed:.1f}s) -> Micro-F1: {df_mean['Micro-F1']:.4f}, Macro-F1: {df_mean['Macro-F1']:.4f}")
    return df_mean.to_dict()


# =========================================================================
# 3. HÀM CHẠY ĐỒNG THỜI TOÀN BỘ CÁC MÔ HÌNH VỚI K-FOLD CHUNG
# =========================================================================

def run_kfold_cv(X, Y, num_labels, ds_name, dev=DEVICE, n_splits=5, seed=SEED,
                 n_chains=N_CHAINS, include_ecc=True):
    """
    Chạy K-Fold CV đồng thời cho BR, CC, ECC (tùy chọn) và EDL-ECC trên cùng fold splits.
    Tất cả các mô hình đều sử dụng ngưỡng phân loại cố định 0.5.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_names = ['BR', 'CC', 'EDL-ECC']
    if include_ecc:
        model_names.insert(2, 'ECC')

    fold_scores = {m: [] for m in model_names}
    fold_unc_correct = []
    fold_unc_wrong   = []

    base_lr = LogisticRegression(solver='lbfgs', max_iter=300, class_weight='balanced')
    t_start = time.time()

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        Y_tr, Y_va = Y[tr_idx], Y[va_idx]
        ensure_binary_classes(Y_tr)

        print(f'  Fold {fold+1}/{n_splits} ', end='', flush=True)
        tf = time.time()

        # 1. BR
        br_preds = train_eval_br(X_tr, Y_tr, X_va, Y_va, base_estimator=base_lr)
        fold_scores['BR'].append(evaluate(Y_va, br_preds))

        # 2. CC
        cc_preds = train_eval_cc(X_tr, Y_tr, X_va, Y_va, base_estimator=base_lr, seed=seed + fold)
        fold_scores['CC'].append(evaluate(Y_va, cc_preds))

        # 3. ECC (nếu bật)
        if include_ecc:
            ecc_preds = train_eval_ecc(X_tr, Y_tr, X_va, Y_va, n_chains=n_chains,
                                       base_estimator=base_lr, seed=seed, fold=fold)
            fold_scores['ECC'].append(evaluate(Y_va, ecc_preds))

        # 4. EDL-ECC (ngưỡng 0.5)
        edl_preds, unc_corr, unc_wrg = train_eval_edl_ecc(
            X_tr, Y_tr, X_va, Y_va, num_labels=num_labels,
            n_chains=n_chains, dev=dev
        )
        fold_scores['EDL-ECC'].append(evaluate(Y_va, edl_preds))
        fold_unc_correct.extend(unc_corr)
        fold_unc_wrong.extend(unc_wrg)

        elapsed = time.time() - tf
        log_str = " ".join([f"{m}={fold_scores[m][-1]['Micro-F1']:.4f}" for m in model_names])
        print(f'[{elapsed:>5.1f}s]  Micro-F1: {log_str}')

    total_time = time.time() - t_start

    df_mean_dict = {}
    df_std_dict  = {}
    for m_name in model_names:
        df_fold_m = pd.DataFrame(fold_scores[m_name])
        df_mean_dict[m_name] = df_fold_m.mean()
        df_std_dict[m_name]  = df_fold_m.std()

    df_mean = pd.DataFrame(df_mean_dict).T
    df_std  = pd.DataFrame(df_std_dict).T

    return df_mean, df_std, fold_unc_correct, fold_unc_wrong, total_time


# =========================================================================
# 4. BÁO CÁO & XUẤT KẾT QUẢ
# =========================================================================

def print_dataset_report(res, total_time, ds_name, out_dir):
    print()
    models = list(res.keys())
    hdr = f"  {'Metric':<12}" + "".join([f"{m:>12}" for m in models])
    if 'EDL-ECC' in res and 'CC' in res:
        hdr += f"{'Δ(EDL-CC)':>12}"
    print(hdr)
    print(f"  {'-' * len(hdr)}")
    for m in METRICS_NAMES:
        row = f"  {m:<12}"
        vals = [res[mdl][m] for mdl in models]
        best_val = max(vals)
        for mdl, v in zip(models, vals):
            star = '*' if (v == best_val and len(vals) > 1) else ' '
            row += f"{v:>11.4f}{star}"
        if 'EDL-ECC' in res and 'CC' in res:
            d_cc = res['EDL-ECC'][m] - res['CC'][m]
            sc = '+' if d_cc >= 0 else ''
            row += f"{sc + str(round(d_cc, 4)):>12}"
        print(row)


def generate_and_save_summary(all_results, all_std_results, out_dir=OUTPUTS_DIR, json_path=Path('results_run.json')):
    summary_rows = []
    first_ds = list(all_results.keys())[0]
    models = list(all_results[first_ds].keys())

    for ds_name, res in all_results.items():
        row = {'Dataset': ds_name}
        for m in models:
            row[m] = res[m]['Micro-F1']
        best_model = max(models, key=lambda m: res[m]['Micro-F1'])
        row['Winner'] = best_model
        if 'EDL-ECC' in res and 'CC' in res:
            row['Delta(EDL-CC)'] = round(res['EDL-ECC']['Micro-F1'] - res['CC']['Micro-F1'], 4)
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(out_dir / 'overall_summary_table.csv', index=False)
    json_path.write_text(
        json.dumps({
            'mean': all_results,
            'std': all_std_results
        }, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )
    print("\n--- OVERALL SUMMARY TABLE (Threshold = 0.5 for all models) ---")
    print(df_summary.to_string(index=False))


# =========================================================================
# 5. CHẠY STANDALONE (thay thế eval_ecc.py)
# =========================================================================

if __name__ == '__main__':
    import pprint
    import argparse

    parser = argparse.ArgumentParser(description='Đánh giá mô hình trên multi-label datasets')
    parser.add_argument('--model', type=str, default='all',
                        choices=['br', 'cc', 'ecc', 'edl-ecc', 'all'],
                        help='Mô hình cần đánh giá (default: all)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Tên dataset cụ thể (default: chạy tất cả)')
    args = parser.parse_args()

    datasets = {args.dataset: DATASET_CONFIGS[args.dataset]} if args.dataset else DATASET_CONFIGS
    results = {}

    for ds_name, cfg in datasets.items():
        print(f"\n{'─'*50}")
        print(f"  [DATASET] {ds_name}")
        print(f"{'─'*50}")
        try:
            if args.model in ('ecc', 'all'):
                res = eval_ecc_dataset(ds_name, cfg)
                if res:
                    results.setdefault(ds_name, {})['ECC'] = res
            if args.model in ('br', 'all'):
                res = eval_br_dataset(ds_name, cfg)
                if res:
                    results.setdefault(ds_name, {})['BR'] = res
            if args.model in ('cc', 'all'):
                res = eval_cc_dataset(ds_name, cfg)
                if res:
                    results.setdefault(ds_name, {})['CC'] = res
            if args.model in ('edl-ecc', 'all'):
                res = eval_edl_ecc_dataset(ds_name, cfg)
                if res:
                    results.setdefault(ds_name, {})['EDL-ECC'] = res
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")

    if results:
        print(f"\n{'='*50}")
        print("  KẾT QUẢ TỔNG HỢP")
        print(f"{'='*50}")
        pprint.pprint(results)
