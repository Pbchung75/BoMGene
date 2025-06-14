import pandas as pd
import gzip
import os
from featurewiz import featurewiz

# Step 1: Define input and output directories
base_folder = '/Gene_dataset/'
output_folder = 'featurewiz_selected/'
os.makedirs(output_folder, exist_ok=True)

# Step 2: Initialize dictionary to store selected features for each dataset
selected_features_all = {}

# Step 3: Retrieve and sort all dataset subdirectories
dataset_dirs = sorted([
    d for d in os.listdir(base_folder)
    if os.path.isdir(os.path.join(base_folder, d))
], key=lambda x: int(x) if x.isdigit() else x)

# Step 4: Process each dataset individually
for dataset_id in dataset_dirs:
    dataset_path = os.path.join(base_folder, dataset_id, 'data.trn.gz')

    if not os.path.isfile(dataset_path):
        print(f"File not found: {dataset_path}. Skipping.")
        continue

    try:
        # Load dataset from gzip-compressed file
        with gzip.open(dataset_path, 'rt') as f:
            df = pd.read_csv(f, sep='\s+', header=None)

        # Separate features (X) and label (y)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        X.columns = [str(i) for i in range(X.shape[1])]
        df_all = X.copy()
        df_all['label'] = y

        print(f"\nProcessing dataset {dataset_id}...")

        # Apply featurewiz with mRMR-based feature selection
        selected_features, df_transformed = featurewiz(
            df_all,
            target='label',
            feature_selection='mrmr',
            corr_limit=0.70,
            verbose=2
        )

        # Store selected features
        selected_features_all[dataset_id] = selected_features
        print(f"{dataset_id}: {len(selected_features)} features selected.")

    except Exception as e:
        print(f"Error processing {dataset_id}: {e}")
        continue

# Step 5: Save selected features to individual CSV files
for dataset_id, features in selected_features_all.items():
    feature_file = os.path.join(output_folder, f"{dataset_id}_selected_features.csv")
    pd.Series(features).to_csv(feature_file, index=False, header=False)
    print(f"Saved selected features to: {feature_file}")
