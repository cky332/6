import torch
from multiprocess import set_start_method
from transformers import AutoProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
from datasets import load_dataset
from torchvision import transforms
from PIL import ImageOps
from torch.cuda.amp import autocast
import os
import pandas as pd

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"  # Use 4 GPUs

# ============ Configuration ============
# Set your Hugging Face cache directory here, or use environment variable HF_HOME
# Default: ~/.cache/huggingface
CACHE_DIR = os.environ.get('HF_HOME', None)  # Set to None to use default cache

# Set your MicroLens data directory here
# The directory should contain image files (jpg, png, etc.)
DATA_DIR = os.environ.get('MICROLENS_DATA_DIR', '/home/mlsnrs/data/cky/data/MicroLens-50k/MicroLens-50k_covers')

# Memory optimization settings
USE_4BIT = True  # Enable 4-bit quantization to reduce memory usage
BATCH_SIZE = 2   # Reduce batch size to save memory
NUM_PROC = 1     # Use single process with device_map="auto"
# =======================================

#model_id  = "lmms-lab/llama3-llava-next-8b"
model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

# Load model with memory optimization
if USE_4BIT:
    # 4-bit quantization config - significantly reduces memory usage
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        cache_dir=CACHE_DIR,
        quantization_config=bnb_config,
        device_map="auto",  # Automatically distribute across GPUs
        torch_dtype=torch.float16,
    ).eval()
else:
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        cache_dir=CACHE_DIR,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()

prompt = "[INST] <image>\nPlease describe this image, which is a cover of a video." \
         " Provide a detailed description in one continuous paragraph, including content information and visual features such as colors, objects, text," \
         " and any notable elements present in the image.[/INST]"


def add_image_file_path(example):
    file_path = example['image'].filename
    filename = os.path.splitext(os.path.basename(file_path))[0]
    example['item_id'] = filename
    return example

# Use DATA_DIR from configuration
img_dir = DATA_DIR
dataset = load_dataset("imagefolder", data_dir=img_dir)
dataset = dataset.map(lambda x: add_image_file_path(x))
print(dataset)

processor = AutoProcessor.from_pretrained(model_id, return_tensors=torch.float16)


def gpu_computation(batch, rank=0):
    # With device_map="auto", model is already distributed across GPUs
    # Get the device of the first model parameter
    device = next(model.parameters()).device

    batch_images = batch['image']

    max_width = max(img.width for img in batch_images)
    max_height = max(img.height for img in batch_images)

    padded_images = []
    for img in batch_images:
        if img.width == max_width and img.height == max_height:
            padded_images.append(img)
            continue
        else:
            delta_width = max_width - img.width
            delta_height = max_height - img.height

            padding = (
            delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))

            new_img = ImageOps.expand(img, border=padding, fill='black')
            padded_images.append(new_img)

    batch['image'] = padded_images

    # Process inputs
    model_inputs = processor(text=[prompt for i in range(len(batch['image']))], images=batch['image'], return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        with autocast():
            outputs = model.generate(**model_inputs, max_new_tokens=200)

    ans = processor.batch_decode(outputs, skip_special_tokens=True)
    ans = [a.split("[/INST]")[1] for a in ans]

    # Clear cache to free memory
    torch.cuda.empty_cache()

    return {"summary": ans}

#f.close()

if __name__ == "__main__":
    if NUM_PROC > 1:
        set_start_method("spawn")

    print(f"Starting inference with batch_size={BATCH_SIZE}, num_proc={NUM_PROC}, 4bit={USE_4BIT}")
    print(f"Processing {len(dataset['train'])} images...")

    updated_dataset = dataset.map(
        gpu_computation,
        batched=True,
        batch_size=BATCH_SIZE,
        with_rank=(NUM_PROC > 1),
        num_proc=NUM_PROC if NUM_PROC > 1 else None,  # None = single process
    )

    train_dataset = updated_dataset['train']
    item_id = train_dataset['item_id']
    summary = train_dataset['summary']
    df = pd.DataFrame({'item_id': item_id, 'summary': summary})
    df.to_csv('image_summary.csv', index=False)

