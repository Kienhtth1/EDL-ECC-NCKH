import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import f1_score, accuracy_score, hamming_loss, jaccard_score

from config import DATA_DIR, DATASET_CONFIGS, SEED
from dataset import load_arff, ensure_binary_classes


def evaluate(y_true, y_pred):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    return {
        '1-HL':     round(1 - hamming_loss(y_true, y_pred), 4),
        'SubAcc':   round(accuracy_score(y_true, y_pred), 4),
        'Micro-F1': round(f1_score(y_true, y_pred, average='micro', zero_division=0), 4),
        'Macro-F1': round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4),
        'Jaccard':  round(jaccard_score(y_true, y_pred, average='samples', zero_division=0), 4),
    }


def eval_ecc_dataset(ds_name, cfg, n_chains=10, n_splits=5, seed=SEED):
    arff_path = DATA_DIR / cfg['file']
    if not arff_path.exists():
        return None

    num_labels = cfg['num_labels']
    X, Y, _ = load_arff(arff_path, num_labels)
    X = StandardScaler().fit_transform(X).astype('float32')

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    t0 = time.time()

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        Y_tr, Y_va = Y[tr_idx], Y[va_idx]
        ensure_binary_classes(Y_tr)

        # Ensemble of Classifier Chains
        chain_probs = []
        for c in range(n_chains):
            base_lr = LogisticRegression(solver='lbfgs', max_iter=300, class_weight='balanced')
            cc = ClassifierChain(base_lr, order='random', random_state=seed + c * 100 + fold)
            cc.fit(X_tr, Y_tr)
            if hasattr(cc, "predict_proba"):
                probs = cc.predict_proba(X_va)
            else:
                probs = cc.predict(X_va)
            chain_probs.append(probs)

        avg_probs = np.mean(chain_probs, axis=0)
        ecc_preds = (avg_probs >= 0.5).astype(int)
        scores.append(evaluate(Y_va, ecc_preds))

    df_res = pd.DataFrame(scores).mean()
    elapsed = time.time() - t0
    print(f"Done {ds_name} in {elapsed:.1f}s: Micro-F1={df_res['Micro-F1']:.4f}, Macro-F1={df_res['Macro-F1']:.4f}")
    return df_res.to_dict()


if __name__ == '__main__':
    results = {}
    for ds_name, cfg in DATASET_CONFIGS.items():
        res = eval_ecc_dataset(ds_name, cfg)
        if res:
            results[ds_name] = res
    print("\n--- ALL ECC RESULTS ---")
    import pprint
    pprint.pprint(results)
