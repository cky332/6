from datasets import Dataset, Image, DatasetDict
from pathlib import Path
import pandas as pd
import os
import sys

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"

# Get the directory where this script is located
# Directory structure (relative to project root):
#   PROJECT_ROOT/
#   ├── MLLM-MSR/train/microlens/dataset_create.py  (this script)
#   ├── MLLM-MSR/data/microlens/                    (source data)
#   └── data/MicroLens-50k/                         (processed data)
#
SCRIPT_DIR = Path(__file__).resolve().parent
MLLM_MSR_PATH = SCRIPT_DIR.parent.parent  # train -> MLLM-MSR
PROJECT_ROOT = MLLM_MSR_PATH.parent  # MLLM-MSR -> project root

# Data paths - check multiple possible locations
DATA_PATH = PROJECT_ROOT / "data" / "MicroLens-50k"
MICROLENS_DATA_PATH = MLLM_MSR_PATH / "data" / "microlens"


def get_file_full_paths_and_names(folder_path):
    folder_path = Path(folder_path)
    full_paths = []
    file_names = []
    for file_path in folder_path.glob('*'):
        if file_path.is_file():
            full_paths.append(str(file_path.absolute()))
            file_names.append(file_path.stem)
    return full_paths, file_names


def check_required_files():
    """Check if all required files exist and provide helpful error messages."""
    errors = []

    # Check train/val pairs
    train_pair_path = DATA_PATH / "Split" / "train_pairs.csv"
    val_pair_path = DATA_PATH / "Split" / "val_pairs.csv"
    if not train_pair_path.exists() or not val_pair_path.exists():
        errors.append(
            f"Train/Val pairs not found at {DATA_PATH / 'Split'}.\n"
            f"   Please run: python MLLM-MSR/data/microlens/split_data.py"
        )

    # Check titles file - try multiple locations
    titles_path = DATA_PATH / "MicroLens-50k_titles.csv"
    alt_titles_path = MICROLENS_DATA_PATH / "MicroLens-50k_titles.csv"
    if not titles_path.exists() and not alt_titles_path.exists():
        errors.append(
            f"Titles file not found.\n"
            f"   Expected at: {titles_path}\n"
            f"   Or at: {alt_titles_path}"
        )

    # Check covers folder - try multiple locations
    covers_path = DATA_PATH / "MicroLens-50k_covers"
    alt_covers_path = MICROLENS_DATA_PATH / "MicroLens-50k_covers"
    if not covers_path.exists() and not alt_covers_path.exists():
        errors.append(
            f"Image covers folder not found.\n"
            f"   Expected at: {covers_path}\n"
            f"   Please download MicroLens-50k_covers from:\n"
            f"   https://github.com/westlake-repl/MicroLens"
        )

    # Check user preference file
    user_pref_path = PROJECT_ROOT / "user_preference_recurrent.csv"
    alt_user_pref_path = MLLM_MSR_PATH / "user_preference_recurrent.csv"
    if not user_pref_path.exists() and not alt_user_pref_path.exists():
        errors.append(
            f"User preference file not found.\n"
            f"   Expected at: {user_pref_path}\n"
            f"   Please run first: python MLLM-MSR/Inference/microlens/preferece_inference_recurrent.py"
        )

    if errors:
        print("=" * 60)
        print("ERROR: Missing required data files!")
        print("=" * 60)
        for i, err in enumerate(errors, 1):
            print(f"\n{i}. {err}")
        print("\n" + "=" * 60)
        print("Please ensure all data files are in place before running.")
        print("=" * 60)
        sys.exit(1)


# Run file checks
print("Checking required files...")
print(f"Project root: {PROJECT_ROOT}")
print(f"Data path: {DATA_PATH}")
check_required_files()
print("All required files found.\n")

# Load train/val pairs
train_pair_file_path = DATA_PATH / "Split" / "train_pairs.csv"
df_train = pd.read_csv(train_pair_file_path)
df_train['item'] = df_train['item'].astype(str)
df_train['user'] = df_train['user'].astype(str)

val_pair_file_path = DATA_PATH / "Split" / "val_pairs.csv"
df_val = pd.read_csv(val_pair_file_path)
df_val['item'] = df_val['item'].astype(str)
df_val['user'] = df_val['user'].astype(str)

# Load user preferences - check multiple locations
user_pref_file_path = PROJECT_ROOT / "user_preference_recurrent.csv"
if not user_pref_file_path.exists():
    user_pref_file_path = MLLM_MSR_PATH / "user_preference_recurrent.csv"

# Check if file has header by reading first line
with open(user_pref_file_path, 'r') as f:
    first_line = f.readline().strip()
has_header = first_line.startswith('user') or ',' in first_line and not first_line.split(',')[0].isdigit()

if has_header:
    user_pref_df = pd.read_csv(user_pref_file_path)
    # Rename columns if needed
    if 'user_id' in user_pref_df.columns:
        user_pref_df = user_pref_df.rename(columns={'user_id': 'user'})
else:
    user_pref_df = pd.read_csv(user_pref_file_path, header=None, names=["user", "preference"])
user_pref_df['user'] = user_pref_df['user'].astype(str)

# Load item titles - check multiple locations with header detection
item_title_file_path = DATA_PATH / "MicroLens-50k_titles.csv"
if not item_title_file_path.exists() or item_title_file_path.stat().st_size == 0:
    item_title_file_path = MICROLENS_DATA_PATH / "MicroLens-50k_titles.csv"

# Check if file has header
with open(item_title_file_path, 'r') as f:
    first_line = f.readline().strip()
has_header = first_line.startswith('item') or first_line == 'item,title'

if has_header:
    item_title_df = pd.read_csv(item_title_file_path)
else:
    item_title_df = pd.read_csv(item_title_file_path, header=None, names=["item", "title"])
item_title_df['item'] = item_title_df['item'].astype(str)

# Load image covers - check multiple locations
folder_path = DATA_PATH / "MicroLens-50k_covers"
if not folder_path.exists():
    folder_path = MICROLENS_DATA_PATH / "MicroLens-50k_covers"

file_paths, file_names = get_file_full_paths_and_names(folder_path)
image_df = pd.DataFrame({"image": file_paths, "item": file_names})
image_df['item'] = image_df['item'].astype(str)


# Debug: Print sample data to understand key formats
print("=== DEBUG: Data key samples ===")
print(f"df_train items (first 5): {df_train['item'].head().tolist()}")
print(f"df_train users (first 5): {df_train['user'].head().tolist()}")
print(f"image_df items (first 5): {image_df['item'].head().tolist()}")
print(f"item_title_df items (first 5): {item_title_df['item'].head().tolist()}")
print(f"user_pref_df users (first 5): {user_pref_df['user'].head().tolist()}")
print()

# Debug: Check data counts before merge
print(f"df_train count: {len(df_train)}")
print(f"image_df count: {len(image_df)}")
print(f"item_title_df count: {len(item_title_df)}")
print(f"user_pref_df count: {len(user_pref_df)}")
print()

# Debug: Step-by-step merge with counts
df_train = pd.merge(df_train, image_df, on="item", how="inner")
print(f"After merge with image_df: {len(df_train)}")

df_train = pd.merge(df_train, item_title_df, on="item", how="inner")
print(f"After merge with item_title_df: {len(df_train)}")

df_train = pd.merge(df_train, user_pref_df, on="user", how="inner")
print(f"After merge with user_pref_df: {len(df_train)}")

df_val = pd.merge(df_val, image_df, on="item", how="inner")
df_val = pd.merge(df_val, item_title_df, on="item", how="inner")
df_val = pd.merge(df_val, user_pref_df, on="user", how="inner")

prompt_text = "Based on the previous interaction history, the user's preference can be summarized as: {}" \
              "Please predict whether this user would interact with the video at the next opportunity. The video's title is'{}', and the given image is this video's cover? " \
              "Please only response 'yes' or 'no' based on your judgement, do not include any other content including words, space, and punctuations in your response."


df_train['prompt'] = df_train.apply(lambda x: prompt_text.format(x['preference'], x['title']), axis=1)
df_train['ground_truth'] = df_train.apply(lambda x: 'Yes' if x['label'] == 1 else 'No', axis=1)
df_train = df_train[['prompt', 'image', 'ground_truth']]

df_val['prompt'] = df_val.apply(lambda x: prompt_text.format(x['preference'], x['title']), axis=1)
df_val['ground_truth'] = df_val.apply(lambda x: 'Yes' if x['label'] == 1 else 'No', axis=1)
df_val = df_val[['prompt', 'image', 'ground_truth']]

# Remove any rows with missing image paths
df_train = df_train.dropna(subset=['image'])
df_val = df_val.dropna(subset=['image'])

print(f"Train samples after merge: {len(df_train)}")
print(f"Val samples after merge: {len(df_val)}")

train_dataset = Dataset.from_pandas(df_train)
train_dataset = train_dataset.cast_column("image", Image())
train_dataset = train_dataset.select(range(25000))
train_dataset = train_dataset.shuffle(seed=2024)

val_dataset = Dataset.from_pandas(df_val)
val_dataset = val_dataset.cast_column("image", Image())
val_dataset = val_dataset.select(range(1000))

dataset = DatasetDict({"train": train_dataset, "validation": val_dataset})
print(dataset)
dataset.save_to_disk("MicroLens-50k-training-recurrent")
