import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import arff
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.base import clone
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings

warnings.filterwarnings('ignore')

datasets = {
    'CHD_49': 6,
    'emotions': 6,
    'scene': 6,
    'Yeast': 14,
    'Water-quality': 14,
    'HumanPseAAC': 14,
    'GpositivePseAAC': 4,
    'PlantPseAAC': 12,
    'VirusPseAAC': 6
}

data_dir = Path('/home/niektran/Downloads/pre-order-for-mlc/data')

def calculate_uncertainty(prob):
    # Maximum uncertainty is when probability is 0.5 (uncertainty = 0.5)
    # We multiply by 2 so that u goes from 0 to 1
    return 2 * (0.5 - np.abs(prob - 0.5))

class SafeLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, solver='lbfgs', max_iter=100, class_weight='balanced'):
        self.solver = solver
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.model = LogisticRegression(solver=self.solver, max_iter=self.max_iter, class_weight=self.class_weight)
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        if len(np.unique(y)) == 1:
            dummy_X = np.zeros((2, X.shape[1]), dtype='float32')
            dummy_y = np.array([0, 1], dtype='float32')
            X = np.vstack([X, dummy_X])
            y = np.hstack([y, dummy_y])
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

class EDL_CC:
    def __init__(self, base_estimator, order=None, beta=1.0):
        self.base_estimator = base_estimator
        self.order = order
        self.beta = beta
        self.estimators_ = []

    def fit(self, X, y):
        self.estimators_ = []
        n_samples, n_features = X.shape
        n_labels = y.shape[1]
        
        if self.order is None:
            self.order_ = list(range(n_labels))
        else:
            self.order_ = self.order
            
        X_current = X.copy()
        
        for k in self.order_:
            estimator = clone(self.base_estimator)
            y_k = y[:, k]
            
            if len(np.unique(y_k)) == 1:
                # Add dummy samples to avoid errors for rare classes in tiny splits
                dummy_X = np.zeros((2, X_current.shape[1]), dtype='float32')
                dummy_y = np.array([0, 1], dtype='float32')
                X_fit = np.vstack([X_current, dummy_X])
                y_fit = np.hstack([y_k, dummy_y])
                estimator.fit(X_fit, y_fit)
            else:
                estimator.fit(X_current, y_k)
                
            self.estimators_.append(estimator)
            
            p_k = estimator.predict_proba(X_current)[:, 1]
            u_k = calculate_uncertainty(p_k)
            
            # Uncertainty Gate
            p_gated = p_k * np.exp(-self.beta * u_k)
            
            X_current = np.hstack([X_current, p_gated.reshape(-1, 1), u_k.reshape(-1, 1)])
            
        return self

    def predict(self, X):
        X_current = X.copy()
        predictions = np.zeros((X.shape[0], len(self.order_)))
        
        for idx, estimator in enumerate(self.estimators_):
            p_k = estimator.predict_proba(X_current)[:, 1]
            pred_class = (p_k >= 0.5).astype(int)
            
            original_idx = self.order_[idx]
            predictions[:, original_idx] = pred_class
            
            u_k = calculate_uncertainty(p_k)
            p_gated = p_k * np.exp(-self.beta * u_k)
            
            X_current = np.hstack([X_current, p_gated.reshape(-1, 1), u_k.reshape(-1, 1)])
            
        return predictions

class EDL_ECC:
    def __init__(self, base_estimator, n_chains=3, beta=1.0, random_state=None):
        self.base_estimator = base_estimator
        self.n_chains = n_chains
        self.beta = beta
        self.random_state = random_state
        self.chains_ = []

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n_labels = y.shape[1]
        self.chains_ = []
        
        for i in range(self.n_chains):
            order = rng.permutation(n_labels).tolist()
            chain = EDL_CC(base_estimator=self.base_estimator, order=order, beta=self.beta)
            chain.fit(X, y)
            self.chains_.append(chain)
        return self

    def predict(self, X):
        n_samples = X.shape[0]
        n_labels = len(self.chains_[0].order_)
        votes = np.zeros((n_samples, n_labels))
        
        for chain in self.chains_:
            preds = chain.predict(X)
            votes += preds
            
        return (votes >= (self.n_chains / 2.0)).astype(int)

results = []

print("Starting Ablation Study...")
for name, num_labels in datasets.items():
    file_path = data_dir / f"{name}.arff"
    if not file_path.exists():
        print(f"Skipping {name}, file not found.")
        continue
        
    print(f"\nProcessing {name}...")
    try:
        data, meta = arff.loadarff(file_path)
        df = pd.DataFrame(data)
    except Exception as e:
        print(f"Failed to load {name}: {e}")
        continue
        
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = df[col].str.decode('utf-8')
            except:
                pass
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    X = df.iloc[:, :-num_labels].values.astype('float32')
    Y = df.iloc[:, -num_labels:].values.astype('float32')
    Y = (Y > 0).astype(int)
    
    X = StandardScaler().fit_transform(X)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    base_lr = SafeLogisticRegression(solver='lbfgs', max_iter=100, class_weight='balanced')
    
    micro_f1_br_list = []
    micro_f1_std_cc_list = []
    micro_f1_cc_list = []
    micro_f1_ecc_list = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        
        # BR
        br = MultiOutputClassifier(clone(base_lr))
        br.fit(X_train, y_train)
        pred_br = br.predict(X_test)
        micro_f1_br_list.append(f1_score(y_test, pred_br, average='micro', zero_division=0))
        
        # Standard CC
        std_cc = ClassifierChain(clone(base_lr), order='random', random_state=42)
        std_cc.fit(X_train, y_train)
        pred_std_cc = std_cc.predict(X_test)
        micro_f1_std_cc_list.append(f1_score(y_test, pred_std_cc, average='micro', zero_division=0))
        
        # EDL+CC
        edl_cc = EDL_CC(base_lr, beta=1.0)
        edl_cc.fit(X_train, y_train)
        pred_cc = edl_cc.predict(X_test)
        micro_f1_cc_list.append(f1_score(y_test, pred_cc, average='micro', zero_division=0))
        
        # EDL+ECC
        edl_ecc = EDL_ECC(base_lr, n_chains=3, beta=1.0, random_state=42)
        edl_ecc.fit(X_train, y_train)
        pred_ecc = edl_ecc.predict(X_test)
        micro_f1_ecc_list.append(f1_score(y_test, pred_ecc, average='micro', zero_division=0))
        
    micro_f1_br_mean = np.mean(micro_f1_br_list)
    micro_f1_std_cc_mean = np.mean(micro_f1_std_cc_list)
    micro_f1_cc_mean = np.mean(micro_f1_cc_list)
    micro_f1_ecc_mean = np.mean(micro_f1_ecc_list)
    
    print(f"    BR (5-Fold Micro-F1): {micro_f1_br_mean:.4f}")
    print(f"    CC (5-Fold Micro-F1): {micro_f1_std_cc_mean:.4f}")
    print(f"    EDL+CC (5-Fold Micro-F1): {micro_f1_cc_mean:.4f}")
    print(f"    EDL+ECC (5-Fold Micro-F1): {micro_f1_ecc_mean:.4f}")
    
    results.append({
        'Dataset': name,
        'BR': round(micro_f1_br_mean, 4),
        'CC': round(micro_f1_std_cc_mean, 4),
        'EDL+CC': round(micro_f1_cc_mean, 4),
        'EDL+ECC': round(micro_f1_ecc_mean, 4)
    })
    
results_df = pd.DataFrame(results)
out_path = '/home/niektran/Downloads/EDL_ECC_Project/ablation_results.md'
results_df.to_markdown(out_path, index=False)
print(f"\nSaved {out_path}")
