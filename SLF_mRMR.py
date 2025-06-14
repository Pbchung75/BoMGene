import pandas as pd
import gzip
import os
import gc
import logging
from featurewiz import featurewiz
from joblib import Parallel, delayed

base_folder = 'Gene/'
output_folder = 'featurewiz_selected/'
os.makedirs(output_folder, exist_ok=True)
log_path = os.path.join(output_folder, "featurewiz_log.txt")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.handlers = [file_handler, console_handler]

dataset_dirs = sorted([
    d for d in os.listdir(base_folder)
    if os.path.isdir(os.path.join(base_folder, d))
], key=lambda x: int(x) if x.isdigit() else x)
logger.info(f" Total number of detected datasets: {len(dataset_dirs)}")
logger.info(f"Dataset list: {dataset_dirs}")

def process_dataset(dataset_id):
    dataset_path = os.path.join(base_folder, dataset_id, 'data.trn.gz')
    if not os.path.isfile(dataset_path):
        logger.warning(f" File not found: {dataset_path}. Skipping.")
        return dataset_id, [], None
    try:
        logger.info(f" Processing dataset: {dataset_id}")
        with gzip.open(dataset_path, 'rt') as f:
            df = pd.read_csv(f, sep='\s+', header=None)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        X.columns = [str(i) for i in range(X.shape[1])]
        df_all = X.copy()
        df_all['label'] = y

        # Feature selection
        selected_features, df_transformed = featurewiz(
            df_all,
            target='label',
            feature_selection='mrmr',
            corr_limit=0.8,
            verbose=2
        )
        logger.info(f" {dataset_id}: selected {len(selected_features)} features.")
        return dataset_id, selected_features, df_transformed

    except Exception as e:
        logger.warning(f"❗ Error processing {dataset_id}: {e}")
        return dataset_id, [], None

    finally:
        gc.collect()

results = Parallel(n_jobs=-1, verbose=2)(
    delayed(process_dataset)(dataset_id) for dataset_id in dataset_dirs
)

for dataset_id, features, df_transformed in results:
    if features:
        feature_file = os.path.join(output_folder, f"{dataset_id}_selected_features.csv")
        pd.Series(features).to_csv(feature_file, index=False, header=False)
        logger.info(f" {dataset_id}: Saved selected features to: {feature_file}")

        transformed_file = os.path.join(output_folder, f"{dataset_id}_transformed.csv")
        df_transformed.to_csv(transformed_file, index=False)
        logger.info(f" {dataset_id}: Saved transformed dataframe to: {transformed_file}")
    else:
        logger.info(f" {dataset_id}: No features selected or processing failed.")

total = len(results)
success = sum(1 for _, features, _ in results if features)
fail = total - success
failed_ids = [dataset_id for dataset_id, features, _ in results if not features]

logger.info("\n SUMMARY:")
logger.info(f" - Total datasets: {total}")
logger.info(f" - Success (features selected): {success}")
logger.info(f" - Failed or no features selected: {fail}")
if failed_ids:
    logger.info(f" Failed datasets / no features: {failed_ids}")
