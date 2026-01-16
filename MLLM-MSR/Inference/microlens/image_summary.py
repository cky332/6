import torch
import torch.multiprocessing as mp
from transformers import AutoProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
from datasets import load_dataset
from PIL import ImageOps
import os
import pandas as pd
from tqdm import tqdm
import gc
import time
from filelock import FileLock

os.environ['CURL_CA_BUNDLE'] = ''

# ============ Configuration ============
CACHE_DIR = os.environ.get('HF_HOME', None)
DATA_DIR = os.environ.get('MICROLENS_DATA_DIR', '/home/mlsnrs/data/cky/data/MicroLens-50k')
OUTPUT_FILE = 'image_summary.csv'
NUM_GPUS = 4  # 使用的GPU数量
BATCH_SIZE = 2  # 每个GPU的batch size
SAVE_EVERY = 50  # 每个进程处理多少张后保存
# =======================================

model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

prompt = "[INST] <image>\nPlease describe this image, which is a cover of a video." \
         " Provide a detailed description in one continuous paragraph, including content information and visual features such as colors, objects, text," \
         " and any notable elements present in the image.[/INST]"


def get_item_id(example):
    """从example中提取item_id"""
    file_path = example['image'].filename
    filename = os.path.splitext(os.path.basename(file_path))[0]
    return filename


def load_existing_results(output_file):
    """加载已有结果，支持断点续传"""
    if os.path.exists(output_file):
        try:
            df = pd.read_csv(output_file)
            return set(df['item_id'].astype(str).tolist())
        except:
            return set()
    return set()


def save_results_safely(results, output_file):
    """线程安全地保存结果"""
    lock_file = output_file + '.lock'
    with FileLock(lock_file):
        # 读取现有结果
        existing = []
        if os.path.exists(output_file):
            try:
                existing_df = pd.read_csv(output_file)
                existing = existing_df.to_dict('records')
            except:
                pass

        # 合并结果（去重）
        existing_ids = {r['item_id'] for r in existing}
        for r in results:
            if r['item_id'] not in existing_ids:
                existing.append(r)
                existing_ids.add(r['item_id'])

        # 保存
        df = pd.DataFrame(existing)
        df.to_csv(output_file, index=False)
        return len(existing)


def worker_process(rank, num_gpus, dataset_indices, return_dict):
    """
    每个GPU上运行的worker进程
    rank: 当前进程的GPU编号
    num_gpus: 总GPU数量
    dataset_indices: 该进程需要处理的数据索引列表
    return_dict: 用于返回结果的共享字典
    """
    # 设置当前进程使用的GPU
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    print(f"[GPU {rank}] Starting worker, processing {len(dataset_indices)} images")

    # 每个进程加载自己的4-bit量化模型到指定GPU
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
        device_map={"": device},  # 指定加载到当前GPU
        low_cpu_mem_usage=True,
    ).eval()

    processor = AutoProcessor.from_pretrained(model_id)

    print(f"[GPU {rank}] Model loaded on {device}")

    # 加载数据集
    dataset = load_dataset("imagefolder", data_dir=DATA_DIR)
    train_data = dataset['train']

    # 加载已处理的ID
    processed_ids = load_existing_results(OUTPUT_FILE)

    # 处理分配给该进程的图片
    results = []
    processed_count = 0
    skipped_count = 0

    # 使用tqdm显示进度
    pbar = tqdm(dataset_indices, desc=f"GPU {rank}", position=rank)

    for idx in pbar:
        example = train_data[idx]
        item_id = get_item_id(example)

        # 跳过已处理的
        if item_id in processed_ids:
            skipped_count += 1
            continue

        try:
            # 处理单张图片
            image = example['image']
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

            results.append({'item_id': item_id, 'summary': ans})
            processed_ids.add(item_id)
            processed_count += 1

            # 清理显存
            del model_inputs, outputs
            gc.collect()
            torch.cuda.empty_cache()

            # 定期保存
            if processed_count % SAVE_EVERY == 0:
                total_saved = save_results_safely(results, OUTPUT_FILE)
                pbar.set_postfix({'saved': total_saved})
                results = []  # 清空已保存的结果

        except Exception as e:
            tqdm.write(f"[GPU {rank}] Error processing {item_id}: {e}")
            continue

    # 保存剩余结果
    if results:
        save_results_safely(results, OUTPUT_FILE)

    return_dict[rank] = {
        'processed': processed_count,
        'skipped': skipped_count
    }
    print(f"[GPU {rank}] Done! Processed {processed_count}, skipped {skipped_count}")


def main():
    print(f"=" * 60)
    print(f"Multi-GPU Image Summary Generator")
    print(f"=" * 60)
    print(f"Using {NUM_GPUS} GPUs")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"=" * 60)

    # 检查GPU数量
    available_gpus = torch.cuda.device_count()
    if available_gpus < NUM_GPUS:
        print(f"Warning: Requested {NUM_GPUS} GPUs but only {available_gpus} available")
        num_gpus = available_gpus
    else:
        num_gpus = NUM_GPUS

    # 加载数据集获取总数
    print("Loading dataset to get total count...")
    dataset = load_dataset("imagefolder", data_dir=DATA_DIR)
    total_images = len(dataset['train'])
    print(f"Total images: {total_images}")

    # 加载已处理的数量
    processed_ids = load_existing_results(OUTPUT_FILE)
    print(f"Already processed: {len(processed_ids)} images")
    print(f"Remaining: {total_images - len(processed_ids)} images")

    # 将数据索引分配给各个GPU
    all_indices = list(range(total_images))

    # 均匀分配
    indices_per_gpu = [[] for _ in range(num_gpus)]
    for i, idx in enumerate(all_indices):
        indices_per_gpu[i % num_gpus].append(idx)

    print(f"\nDistribution:")
    for i, indices in enumerate(indices_per_gpu):
        print(f"  GPU {i}: {len(indices)} images")

    # 启动多进程
    print(f"\nStarting {num_gpus} worker processes...")
    mp.set_start_method('spawn', force=True)

    manager = mp.Manager()
    return_dict = manager.dict()

    processes = []
    start_time = time.time()

    for rank in range(num_gpus):
        p = mp.Process(
            target=worker_process,
            args=(rank, num_gpus, indices_per_gpu[rank], return_dict)
        )
        p.start()
        processes.append(p)
        time.sleep(2)  # 错开启动时间，避免同时加载模型

    # 等待所有进程完成
    for p in processes:
        p.join()

    elapsed_time = time.time() - start_time

    # 统计结果
    total_processed = sum(r['processed'] for r in return_dict.values())
    total_skipped = sum(r['skipped'] for r in return_dict.values())

    print(f"\n" + "=" * 60)
    print(f"All done!")
    print(f"Total processed: {total_processed}")
    print(f"Total skipped: {total_skipped}")
    print(f"Time elapsed: {elapsed_time/60:.2f} minutes")
    print(f"Speed: {total_processed/elapsed_time:.2f} images/second")
    print(f"Results saved to: {OUTPUT_FILE}")
    print(f"=" * 60)


if __name__ == "__main__":
    main()
