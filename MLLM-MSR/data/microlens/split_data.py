"""
Split MicroLens-50k_pairs.csv into train/val/test sets with negative sampling.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import random

# Set random seed for reproducibility
SEED = 2024
random.seed(SEED)
np.random.seed(SEED)

# Paths
# Directory structure (relative to project root):
#   PROJECT_ROOT/
#   ├── MLLM-MSR/data/microlens/split_data.py  (this script)
#   ├── MLLM-MSR/data/microlens/               (source data)
#   └── data/MicroLens-50k/Split/              (output dir)
#
SCRIPT_DIR = Path(__file__).resolve().parent
MLLM_MSR_PATH = SCRIPT_DIR.parent.parent  # microlens -> data -> MLLM-MSR
PROJECT_ROOT = MLLM_MSR_PATH.parent  # MLLM-MSR -> project root
INPUT_FILE = SCRIPT_DIR / "MicroLens-50k_pairs.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "MicroLens-50k" / "Split"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading data...")
df = pd.read_csv(INPUT_FILE)
print(f"Total interactions: {len(df)}")

# Get all unique items for negative sampling
all_items = set(df['item'].unique())
print(f"Unique items: {len(all_items)}")
print(f"Unique users: {df['user'].nunique()}")

# Create user-item interaction set for fast lookup
user_items = df.groupby('user')['item'].apply(set).to_dict()

# Sort by timestamp for temporal split
df = df.sort_values('timestamp')

# Add label=1 for all positive samples
df['label'] = 1

# Function to generate negative samples for a user
def generate_negatives(user, n_neg=1):
    """Generate n_neg negative samples for a user."""
    interacted = user_items.get(user, set())
    available = list(all_items - interacted)
    if len(available) >= n_neg:
        return random.sample(available, n_neg)
    return available

print("Generating negative samples...")
# For each positive sample, generate 1 negative sample
negative_samples = []
for idx, row in df.iterrows():
    neg_items = generate_negatives(row['user'], n_neg=1)
    for neg_item in neg_items:
        negative_samples.append({
            'user': row['user'],
            'item': neg_item,
            'timestamp': row['timestamp'],
            'label': 0
        })

df_neg = pd.DataFrame(negative_samples)
print(f"Generated {len(df_neg)} negative samples")

# Combine positive and negative samples
df_all = pd.concat([df, df_neg], ignore_index=True)
df_all = df_all.sort_values('timestamp')
print(f"Total samples after negative sampling: {len(df_all)}")

# Temporal split: 80% train, 10% val, 10% test
n_total = len(df_all)
train_end = int(n_total * 0.8)
val_end = int(n_total * 0.9)

df_train = df_all.iloc[:train_end]
df_val = df_all.iloc[train_end:val_end]
df_test = df_all.iloc[val_end:]

# Shuffle within each split
df_train = df_train.sample(frac=1, random_state=SEED).reset_index(drop=True)
df_val = df_val.sample(frac=1, random_state=SEED).reset_index(drop=True)
df_test = df_test.sample(frac=1, random_state=SEED).reset_index(drop=True)

print(f"\nSplit sizes:")
print(f"  Train: {len(df_train)} (pos: {(df_train['label']==1).sum()}, neg: {(df_train['label']==0).sum()})")
print(f"  Val:   {len(df_val)} (pos: {(df_val['label']==1).sum()}, neg: {(df_val['label']==0).sum()})")
print(f"  Test:  {len(df_test)} (pos: {(df_test['label']==1).sum()}, neg: {(df_test['label']==0).sum()})")

# Save to CSV (keep only user, item, label columns as expected by the scripts)
df_train[['user', 'item', 'label']].to_csv(OUTPUT_DIR / "train_pairs.csv", index=False)
df_val[['user', 'item', 'label']].to_csv(OUTPUT_DIR / "val_pairs.csv", index=False)
df_test[['user', 'item', 'label']].to_csv(OUTPUT_DIR / "test_pairs.csv", index=False)

print(f"\nFiles saved to {OUTPUT_DIR}:")
print(f"  - train_pairs.csv")
print(f"  - val_pairs.csv")
print(f"  - test_pairs.csv")

# Also copy/link titles file to the expected location
titles_src = SCRIPT_DIR / "MicroLens-50k_titles.csv"
titles_dst = OUTPUT_DIR.parent / "MicroLens-50k_titles.csv"
if titles_src.exists() and not titles_dst.exists():
    import shutil
    shutil.copy(titles_src, titles_dst)
    print(f"  - Copied MicroLens-50k_titles.csv to {titles_dst}")

print(f"\nDone! You can now run dataset_create.py and multi_col_dataset.py")
print(f"\nIMPORTANT: Before running dataset_create.py, ensure:")
print(f"  1. MicroLens-50k_covers folder exists at:")
print(f"     - {OUTPUT_DIR.parent / 'MicroLens-50k_covers'}")
print(f"     - OR at: {SCRIPT_DIR / 'MicroLens-50k_covers'}")
print(f"  2. Run preferece_inference_recurrent.py first to generate user preferences")
