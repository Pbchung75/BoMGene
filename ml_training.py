import os
import logging
import warnings
import pandas as pd
import numpy as np
import time

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')

# Define input and output directories
input_folder = 'featurewiz_selected/'
output_folder = 'corrlimit_mrmr/'
os.makedirs(output_folder, exist_ok=True)

# Result file path
results_fp = os.path.join(output_folder, 'results.csv')

# Setup logging configuration
log_file = os.path.join(output_folder, 'experiment.log')
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("=== Experiment Started ===")

def load_data(fp):
    try:
        df = pd.read_csv(fp, header=0)
        logging.info(f"Loaded {fp}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {fp}")
        return None

def scale_data(X):
    return StandardScaler().fit_transform(X)

def choose_cv(y, k=10):
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=42) if len(y) >= 300 else LeaveOneOut()

def train_and_evaluate_models_cv(X, y):
    n_classes = len(np.unique(y))
    cv = choose_cv(y, k=10)

    model_builders = {
        'SVM': lambda: SVC(C=1e5, kernel='rbf', gamma='scale', class_weight='balanced', random_state=42),
        'RandomForest': lambda: RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        'GradientBoost': lambda: GradientBoostingClassifier(n_estimators=200, learning_rate=0.01, max_depth=10, random_state=42)
    }

    stats = {model: {'acc':[], 'prec':[], 'rec':[], 'f1':[], 'time':0.} for model in ['SVM', 'RandomForest', 'XGBoost', 'GradientBoost']}

    for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X, y), start=1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        for name, build in model_builders.items():
            model = build()
            t0 = time.time()
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            dt = time.time() - t0

            stats[name]['time'] += dt
            stats[name]['acc'].append(accuracy_score(y_te, y_pred))
            stats[name]['prec'].append(precision_score(y_te, y_pred, average='macro', zero_division=1))
            stats[name]['rec'].append(recall_score(y_te, y_pred, average='macro', zero_division=1))
            stats[name]['f1'].append(f1_score(y_te, y_pred, average='macro', zero_division=1))

            logging.info(f"Fold {fold_idx} | {name}: Acc={stats[name]['acc'][-1]:.4f}, F1={stats[name]['f1'][-1]:.4f}, Time={dt:.2f}s")

        # XGBoost using core API
        t0 = time.time()
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dtest = xgb.DMatrix(X_te)
        params = {
            'objective': 'multi:softprob',
            'num_class': n_classes,
            'learning_rate': 0.1,
            'max_depth': 10,
            'seed': 42,
            'verbosity': 0
        }
        bst = xgb.train(params, dtrain, num_boost_round=100, verbose_eval=False)
        preds = bst.predict(dtest)
        y_pred = np.argmax(preds, axis=1)
        dt = time.time() - t0

        stats['XGBoost']['time'] += dt
        stats['XGBoost']['acc'].append(accuracy_score(y_te, y_pred))
        stats['XGBoost']['prec'].append(precision_score(y_te, y_pred, average='macro', zero_division=1))
        stats['XGBoost']['rec'].append(recall_score(y_te, y_pred, average='macro', zero_division=1))
        stats['XGBoost']['f1'].append(f1_score(y_te, y_pred, average='macro', zero_division=1))

        logging.info(f"Fold {fold_idx} | XGBoost: Acc={stats['XGBoost']['acc'][-1]:.4f}, F1={stats['XGBoost']['f1'][-1]:.4f}, Time={dt:.2f}s")

    results = []
    for name, m in stats.items():
        results.append({
            'Model': name,
            'Accuracy (%)': np.mean(m['acc']) * 100,
            'Precision (%)': np.mean(m['prec']) * 100,
            'Recall (%)': np.mean(m['rec']) * 100,
            'F1 Score (%)': np.mean(m['f1']) * 100,
            'Train Time (s)': m['time']
        })
    return pd.DataFrame(results)

def append_results(df, dataset_name):
    df.insert(0, 'Dataset', dataset_name)
    header = not os.path.exists(results_fp)
    df.to_csv(results_fp, mode='a', header=header, index=False)
    logging.info(f"Results appended for dataset: {dataset_name}")

if __name__ == '__main__':
    if os.path.exists(results_fp):
        os.remove(results_fp)

    for fname in sorted(os.listdir(input_folder)):
        if not fname.endswith('_transformed.csv'):
            continue
        dataset_id = fname.split('_')[0]
        fp = os.path.join(input_folder, fname)
        print(f"\nProcessing dataset: {dataset_id}")

        df = load_data(fp)
        if df is None:
            continue

        X_all = df.iloc[:, :-1].values
        raw_y = df.iloc[:, -1].values
        y = LabelEncoder().fit_transform(raw_y)
        X_scaled = scale_data(X_all)

        result_df = train_and_evaluate_models_cv(X_scaled, y)
        append_results(result_df, dataset_id)

    print(f"\nDone! Results saved to: {results_fp}")
    logging.info("Experiment completed.")
