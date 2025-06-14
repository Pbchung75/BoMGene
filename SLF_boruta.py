import os
import logging
import warnings
import pandas as pd
import time

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

# Suppress warnings
warnings.filterwarnings('ignore')

# Define base paths
BASE_DATA_PATH = '/Gene_dataset/'
BASE_RESULTS_PATH = 'Boruta/'
os.makedirs(BASE_RESULTS_PATH, exist_ok=True)

def load_data(file_path):
    """
    Load the dataset from a .gz compressed file.
    """
    try:
        df = pd.read_csv(file_path, header=None, sep='\s+', compression='gzip')
        logging.info(f"Loaded data from: {file_path}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None

def scale_data(X):
    """
    Standardize the input features.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logging.info("Feature matrix standardized.")
    return X_scaled

def select_features_with_boruta(X_scaled, y):
    """
    Perform feature selection using the Boruta algorithm.
    """
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    boruta_selector = BorutaPy(
        rf,
        n_estimators=300,
        max_iter=200,
        alpha=0.01,
        perc=100,
        two_step=True,
        random_state=42,
        verbose=1
    )
    start_time = time.time()
    boruta_selector.fit(X_scaled, y)
    fs_duration = time.time() - start_time

    logging.info(f"Boruta runtime: {fs_duration:.2f}s | "
                 f"Confirmed: {boruta_selector.support_.sum()} | "
                 f"Tentative: {boruta_selector.support_weak_.sum()}")
    print(f"Time: {fs_duration:.2f}s | "
          f"Confirmed: {boruta_selector.support_.sum()} | "
          f"Tentative: {boruta_selector.support_weak_.sum()}")

    return boruta_selector, fs_duration

def save_selected_feature_names_to_csv(df_X, selector, out_dir):
    """
    Save selected feature names to CSV.
    """
    selected_cols = df_X.columns[selector.support_]
    output_path = os.path.join(out_dir, 'selected_feature_names.csv')
    pd.DataFrame(selected_cols, columns=['Selected Features']).to_csv(output_path, index=False)
    logging.info(f"Saved selected feature names to: {output_path}")

def save_boruta_summary_to_csv(selector, fs_time, out_dir):
    """
    Save Boruta feature selection summary.
    """
    confirmed = selector.support_.sum()
    tentative = selector.support_weak_.sum()
    rejected = len(selector.support_) - confirmed - tentative
    summary_df = pd.DataFrame([{
        'Confirmed': confirmed,
        'Tentative': tentative,
        'Rejected': rejected,
        'Feature Selection Time (s)': fs_time
    }])
    output_path = os.path.join(out_dir, 'boruta_summary.csv')
    summary_df.to_csv(output_path, index=False)
    logging.info(f"Saved Boruta summary to: {output_path}")
    return confirmed, tentative, rejected

# Initialize result aggregation
all_results = []

# List all dataset directories
dataset_dirs = [
    d for d in os.listdir(BASE_DATA_PATH)
    if os.path.isdir(os.path.join(BASE_DATA_PATH, d))
]

# Process each dataset
for dataset in sorted(dataset_dirs, key=lambda x: int(x) if x.isdigit() else x):
    data_file = os.path.join(BASE_DATA_PATH, dataset, 'data.trn.gz')
    if not os.path.isfile(data_file):
        print(f"Skipping {dataset}: 'data.trn.gz' not found.")
        continue

    # Setup logging per dataset
    out_dir = os.path.join(BASE_RESULTS_PATH, dataset)
    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, 'experiment_log.log')
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info(f"=== Processing dataset: {dataset} ===")

    # Load and preprocess data
    df = load_data(data_file)
    if df is None:
        continue
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1].values
    X_scaled = scale_data(X)

    # Run Boruta
    selector, fs_time = select_features_with_boruta(X_scaled, y)

    # Save results
    save_selected_feature_names_to_csv(X, selector, out_dir)
    confirmed, tentative, rejected = save_boruta_summary_to_csv(selector, fs_time, out_dir)

    # Append to global results
    all_results.append({
        'Dataset': dataset,
        'Confirmed': confirmed,
        'Tentative': tentative,
        'Rejected': rejected,
        'Feature Selection Time (s)': fs_time
    })

# Export global summary
summary_df = pd.DataFrame(all_results)
summary_path = os.path.join(BASE_RESULTS_PATH, 'ALL_boruta_summary.csv')
summary_df.to_csv(summary_path, index=False)
print(f"All-dataset summary saved to: {summary_path}")
logging.info(f"Saved overall summary to: {summary_path}")
