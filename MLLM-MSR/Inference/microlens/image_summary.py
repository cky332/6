import torch
from transformers import AutoProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
from datasets import load_dataset
from PIL import ImageOps
import os
import pandas as pd
from tqdm import tqdm
import gc
import time

os.environ['CURL_CA_BUNDLE'] = ''
# 注意：如果你通过命令行设置 CUDA_VISIBLE_DEVICES，这里会被覆盖
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"  # 使用4张4090

# ============ Configuration ============
CACHE_DIR = os.environ.get('HF_HOME', None)
DATA_DIR = os.environ.get('MICROLENS_DATA_DIR', '/home/mlsnrs/data/cky/data/MicroLens-50k')
BATCH_SIZE = 1  # 减小到1，更稳定
OUTPUT_FILE = 'image_summary.csv'
# =======================================

print(f"Using {torch.cuda.device_count()} GPUs")
print(f"Loading model...")

model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

# 使用4bit量化大幅减少显存占用
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id,
    cache_dir=CACHE_DIR,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
    device_map="auto",
    low_cpu_mem_usage=True,
).eval()

print(f"Model loaded! Device: {model.device}")

processor = AutoProcessor.from_pretrained(model_id)

prompt = "[INST] <image>\nPlease describe this image, which is a cover of a video." \
         " Provide a detailed description in one continuous paragraph, including content information and visual features such as colors, objects, text," \
         " and any notable elements present in the image.[/INST]"


def get_item_id(example):
    file_path = example['image'].filename
    filename = os.path.splitext(os.path.basename(file_path))[0]
    return filename


def process_single_image(image):
    """处理单张图片"""
    device = model.device

    model_inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            outputs = model.generate(**model_inputs, max_new_tokens=200)

    ans = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    ans = ans.split("[/INST]")[1] if "[/INST]" in ans else ans

    # 清理
    del model_inputs, outputs
    gc.collect()
    torch.cuda.empty_cache()

    return ans


def load_existing_results(output_file):
    """加载已有结果，支持断点续传"""
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        return set(df['item_id'].astype(str).tolist())
    return set()


if __name__ == "__main__":
    print(f"Loading dataset from {DATA_DIR}...")
    dataset = load_dataset("imagefolder", data_dir=DATA_DIR)
    train_data = dataset['train']
    total = len(train_data)
    print(f"Total images: {total}")

    # 加载已处理的结果（断点续传）
    processed_ids = load_existing_results(OUTPUT_FILE)
    print(f"Already processed: {len(processed_ids)} images")

    # 准备结果列表
    results = []

    # 如果有已处理的，先加载
    if processed_ids:
        existing_df = pd.read_csv(OUTPUT_FILE)
        results = existing_df.to_dict('records')

    # 预热模型
    print("Warming up model with first image...")
    start_time = time.time()
    first_example = train_data[0]
    first_id = get_item_id(first_example)
    if first_id not in processed_ids:
        _ = process_single_image(first_example['image'])
    print(f"Warmup done in {time.time() - start_time:.2f}s")

    # 处理所有图片
    print("\nProcessing images...")
    skipped = 0

    for i in tqdm(range(total), desc="Processing"):
        example = train_data[i]
        item_id = get_item_id(example)

        # 跳过已处理的
        if item_id in processed_ids:
            skipped += 1
            continue

        try:
            summary = process_single_image(example['image'])
            results.append({'item_id': item_id, 'summary': summary})
            processed_ids.add(item_id)

            # 每100张保存一次（断点续传）
            if len(results) % 100 == 0:
                df = pd.DataFrame(results)
                df.to_csv(OUTPUT_FILE, index=False)
                tqdm.write(f"Saved {len(results)} results")

        except Exception as e:
            tqdm.write(f"Error processing {item_id}: {e}")
            continue

    # 最终保存
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDone! Processed {len(results)} images, skipped {skipped} (already done)")
    print(f"Results saved to {OUTPUT_FILE}")
