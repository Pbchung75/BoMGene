import os
import gzip
import pandas as pd
import numpy as np
import logging
import warnings
import time
import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')

# Define base directories for input and output
base_data_directory = 'Gene_dataset/'
output_base_directory = 'Bench/'
os.makedirs(output_base_directory, exist_ok=True)

feature_count_fp = os.path.join(output_base_directory, 'feature_count.csv')
results_fp = os.path.join(output_base_directory, 'results.csv')
log_file = os.path.join(output_base_directory, 'experiment.log')

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

def scale_data(X):
    return StandardScaler().fit_transform(X)

def choose_cv(y, k=10):
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=42) if len(y) >= 300 else LeaveOneOut()

def select_rf_features(X, y, n_feats):
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    idx = np.argsort(rf.feature_importances_)[::-1][:n_feats]
    return idx

def train_and_evaluate_models_cv(X, y):
    n_classes = len(np.unique(y))
    cv = choose_cv(y, k=10)

    model_builders = {
        'SVM': lambda: SVC(C=1e5, kernel='rbf', gamma='scale', class_weight='balanced', random_state=42),
        'RandomForest': lambda: RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoost': lambda: GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    stats = {name: {'acc':[], 'prec':[], 'rec':[], 'f1':[], 'time':0.} for name in ['SVM', 'RandomForest', 'XGBoost', 'GradientBoost']}

    for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X, y), start=1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # Train traditional ML models
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

        # Train XGBoost model
        t0 = time.time()
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dtest = xgb.DMatrix(X_te)
        params = {
            'objective': 'multi:softprob',
            'num_class': n_classes,
            'learning_rate': 0.01,
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

    rows = []
    for name, m in stats.items():
        rows.append({
            'Model': name,
            'Accuracy (%)': np.mean(m['acc']) * 100,
            'Precision (%)': np.mean(m['prec']) * 100,
            'Recall (%)': np.mean(m['rec']) * 100,
            'F1 Score (%)': np.mean(m['f1']) * 100,
            'Train Time (s)': m['time']
        })
    return pd.DataFrame(rows)

def append_results(df, dataset_name):
    df.insert(0, 'Dataset', dataset_name)
    header = not os.path.exists(results_fp)
    df.to_csv(results_fp, mode='a', header=header, index=False)
    logging.info(f"Appended results for dataset: {dataset_name}")

def read_gene_dataset(ds_folder):
    gz_path = os.path.join(ds_folder, 'data.trn.gz')
    norm_path = os.path.join(ds_folder, 'data.trn')
    df = None
    if os.path.exists(gz_path):
        with gzip.open(gz_path, 'rt') as f:
            df = pd.read_csv(f, sep='\s+', header=None)
    elif os.path.exists(norm_path):
        df = pd.read_csv(norm_path, sep='\s+', header=None)
    return df

# Read feature selection targets
feature_counts = pd.read_csv(feature_count_fp)
feat_map = dict(zip(feature_counts['Datasets'].astype(str), feature_counts['Num_select']))

if __name__ == '__main__':
    print("STARTING PIPELINE ...")
    if os.path.exists(results_fp):
        os.remove(results_fp)

    dataset_dirs = [d for d in os.listdir(base_data_directory) if os.path.isdir(os.path.join(base_data_directory, d))]
    print(f"Found {len(dataset_dirs)} datasets.")

    for ds in sorted(dataset_dirs, key=lambda x: int(x) if x.isdigit() else x):
        print(f"\nProcessing dataset: {ds}")
        ds_folder = os.path.join(base_data_directory, ds)
        df = read_gene_dataset(ds_folder)
        if df is None or df.shape[1] < 2:
            logger.warning(f"Invalid or missing data in {ds}")
            print(f"Skip: {ds} (no valid data found)")
            continue

        X_all = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_scaled = scale_data(X_all)

        n_feats = int(feat_map.get(ds, 0))
        if n_feats <= 0 or n_feats > X_all.shape[1]:
            logger.warning(f"Invalid feature count for {ds}: {n_feats}")
            print(f"Skip: {ds} (invalid feature count)")
            continue

        idx = select_rf_features(X_scaled, y, n_feats)
        X_selected = X_all[:, idx]

        transformed_fp = os.path.join(base_data_directory, ds, 'transformed_selected.csv')
        selected_df = pd.DataFrame(X_selected, columns=[f"f{i+1}" for i in range(X_selected.shape[1])])
        selected_df['label'] = y
        selected_df.to_csv(transformed_fp, index=False)

        X_sel_scaled = scale_data(X_selected)
        res_df = train_and_evaluate_models_cv(X_sel_scaled, y)
        append_results(res_df, ds)
        print(f"Done: {ds}")

    print(f"\nPipeline completed. Results saved to: {results_fp}")
