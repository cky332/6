from datasets import Dataset, Image
from pathlib import Path
import pandas as pd
import os
import sys

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"

# Get the directory where this script is located
# Directory structure can be one of:
#   Option 1: data inside project
#     PROJECT_ROOT/
#     ├── MLLM-MSR/test/microlens/multi_col_dataset.py
#     └── data/MicroLens-50k/
#
#   Option 2: data alongside project (sibling folders)
#     PARENT_DIR/
#     ├── 6-main/MLLM-MSR/test/microlens/multi_col_dataset.py
#     └── data/MicroLens-50k/
#
SCRIPT_DIR = Path(__file__).resolve().parent
MLLM_MSR_PATH = SCRIPT_DIR.parent.parent  # test -> MLLM-MSR
PROJECT_ROOT = MLLM_MSR_PATH.parent  # MLLM-MSR -> project root (e.g., 6-main)
PARENT_DIR = PROJECT_ROOT.parent  # Parent of project root (for sibling data folder)

# Data paths - check multiple possible locations
DATA_PATH = PROJECT_ROOT / "data" / "MicroLens-50k"  # data inside project
DATA_PATH_SIBLING = PARENT_DIR / "data" / "MicroLens-50k"  # data alongside project
MICROLENS_DATA_PATH = MLLM_MSR_PATH / "data" / "microlens"


def get_file_full_paths_and_names(folder_path, image_extensions=None):
    """Get image file paths and names from a folder.

    Args:
        folder_path: Path to the folder containing images
        image_extensions: Set of valid image extensions (default: common image formats)
    """
    if image_extensions is None:
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}

    folder_path = Path(folder_path)
    full_paths = []
    file_names = []
    for file_path in folder_path.glob('*'):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            full_paths.append(str(file_path.absolute()))
            file_names.append(file_path.stem)
    return full_paths, file_names


def check_required_files():
    """Check if all required files exist and provide helpful error messages."""
    errors = []

    # Check test pairs - try both data locations
    test_pair_path = DATA_PATH / "Split" / "test_pairs.csv"
    test_pair_path_sibling = DATA_PATH_SIBLING / "Split" / "test_pairs.csv"
    if not test_pair_path.exists() and not test_pair_path_sibling.exists():
        errors.append(
            f"Test pairs not found.\n"
            f"   Checked: {DATA_PATH / 'Split'}\n"
            f"   Checked: {DATA_PATH_SIBLING / 'Split'}\n"
            f"   Please run: python MLLM-MSR/data/microlens/split_data.py"
        )

    # Check titles file - try multiple locations
    titles_paths = [
        DATA_PATH / "MicroLens-50k_titles.csv",
        DATA_PATH_SIBLING / "MicroLens-50k_titles.csv",
        MICROLENS_DATA_PATH / "MicroLens-50k_titles.csv",
    ]
    if not any(p.exists() for p in titles_paths):
        errors.append(
            f"Titles file not found.\n"
            f"   Checked locations:\n" +
            "\n".join(f"   - {p}" for p in titles_paths)
        )

    # Check covers folder - try multiple locations
    def has_images(path):
        if not path.exists():
            return False
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
        return any(f.suffix.lower() in image_extensions for f in path.glob('*') if f.is_file())

    image_locations = [
        DATA_PATH / "MicroLens-50k_covers",
        DATA_PATH_SIBLING / "MicroLens-50k_covers",
        MICROLENS_DATA_PATH / "MicroLens-50k_covers",
        DATA_PATH,  # Images directly in folder
        DATA_PATH_SIBLING,  # Images directly in sibling folder
    ]

    if not any(has_images(p) for p in image_locations):
        errors.append(
            f"Image files not found.\n"
            f"   Checked locations:\n" +
            "\n".join(f"   - {p}" for p in image_locations) +
            f"\n   Please download MicroLens-50k images from:\n"
            f"   https://github.com/westlake-repl/MicroLens"
        )

    # Check user preference file - try multiple locations
    user_pref_paths = [
        PROJECT_ROOT / "user_preference_recurrent.csv",
        MLLM_MSR_PATH / "user_preference_recurrent.csv",
        MLLM_MSR_PATH / "inference" / "Microlens" / "user_preference_recurrent.csv",
        MLLM_MSR_PATH / "Inference" / "microlens" / "user_preference_recurrent.csv",
    ]
    if not any(p.exists() for p in user_pref_paths):
        errors.append(
            f"User preference file not found.\n"
            f"   Checked locations:\n" +
            "\n".join(f"   - {p}" for p in user_pref_paths) +
            f"\n   Please run first: python MLLM-MSR/Inference/microlens/preferece_inference_recurrent.py"
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
print(f"Data path (inside project): {DATA_PATH}")
print(f"Data path (sibling): {DATA_PATH_SIBLING}")
check_required_files()
print("All required files found.\n")

# Helper function to find existing file from candidates
def find_existing_file(*candidates):
    for path in candidates:
        if path.exists() and (path.is_file() and path.stat().st_size > 0 or path.is_dir()):
            return path
    return candidates[0]  # Fallback

# Load test pairs - check both locations
pair_file_path = find_existing_file(
    DATA_PATH / "Split" / "test_pairs.csv",
    DATA_PATH_SIBLING / "Split" / "test_pairs.csv"
)
print(f"Loading test pairs from: {pair_file_path}")
df = pd.read_csv(pair_file_path)
df['item'] = df['item'].astype(str)
df['user'] = df['user'].astype(str)

# Load user preferences - check multiple locations
user_pref_file_path = find_existing_file(
    PROJECT_ROOT / "user_preference_recurrent.csv",
    MLLM_MSR_PATH / "user_preference_recurrent.csv",
    MLLM_MSR_PATH / "inference" / "Microlens" / "user_preference_recurrent.csv",
    MLLM_MSR_PATH / "Inference" / "microlens" / "user_preference_recurrent.csv",
)
print(f"Loading user preferences from: {user_pref_file_path}")

# Check if file has header by reading first line
with open(user_pref_file_path, 'r') as f:
    first_line = f.readline().strip()
has_header = first_line.startswith('user') or ',' in first_line and not first_line.split(',')[0].isdigit()

if has_header:
    user_pref_df = pd.read_csv(user_pref_file_path)
    print(f"User pref columns (original): {user_pref_df.columns.tolist()}")
    # Rename first column to 'user' if needed
    if 'user_id' in user_pref_df.columns:
        user_pref_df = user_pref_df.rename(columns={'user_id': 'user'})
    elif user_pref_df.columns[0] != 'user':
        user_pref_df = user_pref_df.rename(columns={user_pref_df.columns[0]: 'user'})
    # Rename second column to 'preference' if needed
    if len(user_pref_df.columns) >= 2 and 'preference' not in user_pref_df.columns:
        second_col = user_pref_df.columns[1]
        user_pref_df = user_pref_df.rename(columns={second_col: 'preference'})
else:
    user_pref_df = pd.read_csv(user_pref_file_path, header=None, names=["user", "preference"])
user_pref_df['user'] = user_pref_df['user'].astype(str)
print(f"User pref columns (final): {user_pref_df.columns.tolist()}")

# Load item titles - check multiple locations with header detection
item_title_file_path = find_existing_file(
    DATA_PATH / "MicroLens-50k_titles.csv",
    DATA_PATH_SIBLING / "MicroLens-50k_titles.csv",
    MICROLENS_DATA_PATH / "MicroLens-50k_titles.csv"
)
print(f"Loading titles from: {item_title_file_path}")

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
image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}

def find_image_folder():
    """Find the folder containing image files."""
    candidates = [
        DATA_PATH / "MicroLens-50k_covers",
        DATA_PATH_SIBLING / "MicroLens-50k_covers",
        MICROLENS_DATA_PATH / "MicroLens-50k_covers",
        DATA_PATH,  # Images directly in MicroLens-50k folder
        DATA_PATH_SIBLING,  # Images directly in sibling MicroLens-50k folder
    ]
    for path in candidates:
        if path.exists():
            # Check if this folder has image files
            has_imgs = any(f.suffix.lower() in image_extensions for f in path.glob('*') if f.is_file())
            if has_imgs:
                return path
    return candidates[0]  # Fallback

folder_path = find_image_folder()
print(f"Loading images from: {folder_path}")

file_paths, file_names = get_file_full_paths_and_names(folder_path)
image_df = pd.DataFrame({"image": file_paths, "item": file_names})
image_df['item'] = image_df['item'].astype(str)

# Debug: Print sample data to understand key formats
print("=== DEBUG: Data key samples ===")
print(f"df items (first 5): {df['item'].head().tolist()}")
print(f"df users (first 5): {df['user'].head().tolist()}")
print(f"image_df items (first 5): {image_df['item'].head().tolist()}")
print(f"item_title_df items (first 5): {item_title_df['item'].head().tolist()}")
print(f"user_pref_df users (first 5): {user_pref_df['user'].head().tolist()}")
print()

# Debug: Check data counts before merge
print(f"df count: {len(df)}")
print(f"image_df count: {len(image_df)}")
print(f"item_title_df count: {len(item_title_df)}")
print(f"user_pref_df count: {len(user_pref_df)}")
print()

# Merge dataframes
df = pd.merge(df, image_df, on="item", how="inner")
print(f"After merge with image_df: {len(df)}")

df = pd.merge(df, item_title_df, on="item", how="inner")
print(f"After merge with item_title_df: {len(df)}")

df = pd.merge(df, user_pref_df, on="user", how="inner")
print(f"After merge with user_pref_df: {len(df)}")

prompt_text = "[INST]<image>\nBased on the previous interaction history, the user's preference can be summarized as: {}" \
              "Please predict whether this user would interact with the video at the next opportunity. The video's title is'{}', and the given image is this video's cover? " \
              "Please only response 'yes' or 'no' based on your judgement, do not include any other content including words, space, and punctuations in your response. [/INST]"

df['prompt'] = df.apply(lambda x: prompt_text.format(x['preference'], x['title']), axis=1)

df = df[['user', 'prompt', 'image', 'label']]

print(f"\nTest samples: {len(df)}")

# Create dataset
dataset = Dataset.from_pandas(df)
dataset = dataset.cast_column("image", Image())

# Check dataset structure
print(dataset)
dataset.save_to_disk("MicroLens-50k-test-recurrent")
print(f"\nDataset saved to: MicroLens-50k-test-recurrent")
