"""
Multi-GPU Image Summarization Module for LLaVA-based Video Cover Analysis

This module implements a parallelized, production-grade pipeline for generating
textual descriptions of video cover images using the LLaVA (Large Language and Vision
Assistant) multimodal language model. It's designed for processing large-scale datasets
efficiently across multiple GPUs with batch processing, 4-bit quantization, and
fault-tolerant resume capability.

Key Features:
    - Multi-GPU Parallel Processing: Distributes image processing across multiple GPUs
    - True Batch Processing: Processes multiple images per GPU simultaneously
    - 4-bit Quantization: Uses BitsAndBytes NF4 quantization to reduce memory usage
    - Resume Capability: Automatically skips already-processed images (idempotent)
    - Thread-Safe File I/O: Uses file locking to prevent data corruption
    - Memory Efficient: Aggressive garbage collection and CUDA cache management
    - Progress Tracking: Per-GPU progress bars with real-time statistics

Architecture:
    - Main process: Distributes workload and coordinates worker processes
    - Worker processes: Each worker loads a 4-bit quantized model on a dedicated GPU
    - Batch processing: Images are padded to uniform size and processed in batches
    - Safe persistence: Results are periodically saved with file locking

Typical Use Case:
    Generate detailed textual descriptions for 50,000+ video cover images to create
    training data for multimodal sequential recommendation systems.

Example:
    $ python image_summary.py
    # Processes all images in DATA_DIR across NUM_GPUS GPUs
    # Outputs results to OUTPUT_FILE (image_summary.csv)

Authors: MLLM-MSR Research Team
License: Academic Research Use
"""

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
# Environment-based configuration with sensible defaults
CACHE_DIR = os.environ.get('HF_HOME', None)  # Hugging Face cache directory
DATA_DIR = os.environ.get('MICROLENS_DATA_DIR', '/home/mlsnrs/data/cky/data/MicroLens-50k')  # Image dataset location
OUTPUT_FILE = 'image_summary.csv'  # Output CSV file for image summaries
NUM_GPUS = 3  # Number of GPUs to use for parallel processing
BATCH_SIZE = 8  # Batch size per GPU (true batching for efficiency)
SAVE_EVERY = 50  # Save results to disk every N images (per process)
# =======================================

# Model configuration
model_id = "llava-hf/llava-v1.6-mistral-7b-hf"  # LLaVA-NeXT 1.6 with Mistral 7B LLM backbone

# Instruction prompt template for image description generation
# Uses the LLaVA instruction format: [INST] <image>\n{instructions} [/INST]
prompt = "[INST] <image>\nPlease describe this image, which is a cover of a video." \
         " Provide a detailed description in one continuous paragraph, including content information and visual features such as colors, objects, text," \
         " and any notable elements present in the image.[/INST]"


def get_item_id(example):
    """
    Extract the item ID from a dataset example by parsing its image filename.

    The item ID is the filename without extension, used as a unique identifier
    for each video/image in the dataset. This enables tracking which images
    have been processed and prevents duplicate processing.

    Args:
        example (dict): A dataset example dictionary containing an 'image' key
            with a PIL Image object that has a 'filename' attribute.

    Returns:
        str: The item ID (filename without extension).
            Example: "/path/to/video_12345.jpg" -> "video_12345"

    Example:
        >>> example = {'image': PIL.Image with filename="/data/covers/item_001.jpg"}
        >>> get_item_id(example)
        'item_001'
    """
    file_path = example['image'].filename
    filename = os.path.splitext(os.path.basename(file_path))[0]
    return filename


def load_existing_results(output_file):
    """
    Load previously processed item IDs from the output CSV file.

    This function enables resume capability by reading the output file and
    extracting all item IDs that have already been processed. Worker processes
    use this set to skip already-completed images, making the pipeline idempotent
    and resilient to interruptions.

    Args:
        output_file (str): Path to the CSV output file containing columns
            ['item_id', 'summary'].

    Returns:
        set: A set of item IDs (as strings) that have been previously processed.
            Returns an empty set if the file doesn't exist or cannot be read.

    Notes:
        - Silently handles file read errors by returning empty set
        - Converts all item_ids to strings for consistent comparison
        - This function is called by each worker process independently

    Example:
        >>> processed = load_existing_results('image_summary.csv')
        >>> 'video_001' in processed
        True
        >>> len(processed)
        15432
    """
    if os.path.exists(output_file):
        try:
            df = pd.read_csv(output_file)
            return set(df['item_id'].astype(str).tolist())
        except:
            return set()
    return set()


def save_results_safely(results, output_file):
    """
    Thread-safe function to append new results to the output CSV file.

    Uses file locking (FileLock) to ensure that multiple worker processes can
    safely write to the same output file concurrently without data corruption
    or race conditions. The function performs atomic read-modify-write operations
    with automatic deduplication.

    Algorithm:
        1. Acquire exclusive file lock
        2. Read existing results from CSV
        3. Merge new results with existing (skip duplicates by item_id)
        4. Write merged results back to CSV
        5. Release file lock

    Args:
        results (list[dict]): List of result dictionaries to save. Each dict
            should contain {'item_id': str, 'summary': str}.
        output_file (str): Path to the output CSV file.

    Returns:
        int: Total number of unique items in the output file after merging.

    Thread Safety:
        This function is safe for concurrent calls from multiple processes.
        Uses FileLock to prevent race conditions and data corruption.

    Notes:
        - Automatically creates output_file if it doesn't exist
        - Silently handles CSV read errors (treats as empty file)
        - Deduplicates by item_id (first occurrence wins)
        - Lock file: {output_file}.lock

    Example:
        >>> results = [
        ...     {'item_id': 'vid_001', 'summary': 'A cat playing piano'},
        ...     {'item_id': 'vid_002', 'summary': 'Mountain landscape at sunset'}
        ... ]
        >>> total = save_results_safely(results, 'summaries.csv')
        >>> print(f"Total items: {total}")
        Total items: 15434
    """
    lock_file = output_file + '.lock'
    with FileLock(lock_file):
        # Read existing results
        existing = []
        if os.path.exists(output_file):
            try:
                existing_df = pd.read_csv(output_file)
                existing = existing_df.to_dict('records')
            except:
                pass

        # Merge results (deduplicate by item_id)
        existing_ids = {r['item_id'] for r in existing}
        for r in results:
            if r['item_id'] not in existing_ids:
                existing.append(r)
                existing_ids.add(r['item_id'])

        # Save merged results
        df = pd.DataFrame(existing)
        df.to_csv(output_file, index=False)
        return len(existing)


def pad_images_to_same_size(images):
    """
    Pad a batch of images to uniform dimensions for batch processing.

    Batch inference requires all images to have identical dimensions. This function
    finds the maximum width and height across all images and pads smaller images
    with black borders, centering the original content.

    Padding Strategy:
        - Find max width and max height across all images
        - For each image smaller than max dimensions:
            - Calculate padding needed on each side
            - Apply symmetric padding (centered) with black fill
            - Left/top padding: delta // 2
            - Right/bottom padding: delta - (delta // 2) [handles odd deltas]

    Args:
        images (list[PIL.Image.Image]): List of PIL Image objects to pad.
            Can be of varying dimensions.

    Returns:
        list[PIL.Image.Image]: List of padded images, all with identical dimensions
            equal to (max_width, max_height).

    Performance:
        - Single image: Returns immediately without modification
        - Already uniform: Minimal overhead (dimension check only)
        - Requires padding: O(n) where n = number of images

    Example:
        >>> from PIL import Image
        >>> images = [
        ...     Image.new('RGB', (100, 50)),  # Small
        ...     Image.new('RGB', (200, 150))  # Large
        ... ]
        >>> padded = pad_images_to_same_size(images)
        >>> all(img.size == (200, 150) for img in padded)
        True
    """
    if len(images) == 1:
        return images

    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)

    padded_images = []
    for img in images:
        if img.width == max_width and img.height == max_height:
            padded_images.append(img)
        else:
            delta_width = max_width - img.width
            delta_height = max_height - img.height
            padding = (
                delta_width // 2,
                delta_height // 2,
                delta_width - (delta_width // 2),
                delta_height - (delta_height // 2)
            )
            new_img = ImageOps.expand(img, border=padding, fill='black')
            padded_images.append(new_img)

    return padded_images


def process_batch(model, processor, images, device):
    """
    Process a batch of images through LLaVA model to generate descriptions.

    This is the core inference function that handles batch processing of images
    through the multimodal LLaVA model. It performs padding, tokenization,
    model inference with automatic mixed precision, and decoding.

    Processing Pipeline:
        1. Pad images to uniform size (required for batching)
        2. Prepare batch inputs: replicate prompt for each image
        3. Tokenize text and process images through processor
        4. Run model generation with FP16 automatic mixed precision
        5. Decode generated tokens to text descriptions
        6. Extract answers by removing instruction prefix
        7. Aggressively clean up GPU memory

    Args:
        model (LlavaNextForConditionalGeneration): The loaded LLaVA model
            in evaluation mode.
        processor (AutoProcessor): The LLaVA processor for tokenization and
            image preprocessing.
        images (list[PIL.Image.Image]): Batch of PIL images to process.
        device (str): Target device (e.g., 'cuda:0', 'cuda:1').

    Returns:
        list[str]: List of generated text descriptions, one per input image.
            Descriptions are cleaned (instruction prefix removed).

    Memory Management:
        - Uses torch.no_grad() to disable gradient computation
        - Uses autocast for FP16 automatic mixed precision
        - Explicitly deletes intermediate tensors
        - Calls garbage collector and clears CUDA cache
        - Critical for processing thousands of images without OOM

    Notes:
        - Max generation length: 200 tokens (~150 words)
        - Batch size limited by GPU memory (typically 4-16)
        - Processing time: ~0.5-2 seconds per image depending on hardware

    Example:
        >>> batch = [image1, image2, image3]
        >>> descriptions = process_batch(model, processor, batch, 'cuda:0')
        >>> descriptions[0]
        'This video cover features a red sports car on a mountain road at sunset...'
    """
    # Pad images to same size (required for batch processing)
    padded_images = pad_images_to_same_size(images)

    # Prepare batch inputs: same prompt for all images
    prompts = [prompt] * len(padded_images)

    model_inputs = processor(
        text=prompts,
        images=padded_images,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            outputs = model.generate(**model_inputs, max_new_tokens=200)

    # Decode all outputs to text
    answers = processor.batch_decode(outputs, skip_special_tokens=True)
    # Remove instruction prefix to extract clean answer
    answers = [ans.split("[/INST]")[1] if "[/INST]" in ans else ans for ans in answers]

    # Aggressively clean GPU memory (critical for long-running processes)
    del model_inputs, outputs
    gc.collect()
    torch.cuda.empty_cache()

    return answers


def worker_process(rank, num_gpus, dataset_indices, return_dict):
    """
    Worker process function that runs on a single GPU.

    This is the main execution function for each parallel worker. Each worker:
    - Loads a 4-bit quantized LLaVA model on its assigned GPU
    - Processes its assigned subset of images in batches
    - Implements resume logic by skipping already-processed images
    - Periodically saves results with thread-safe file I/O
    - Reports progress via per-GPU progress bars
    - Handles errors gracefully with fallback to single-image processing

    Architecture:
        - Process isolation: Each worker is a separate Python process
        - GPU binding: Exclusively uses assigned GPU (cuda:{rank})
        - Memory efficient: 4-bit NF4 quantization (~3-4GB per model)
        - Fault tolerant: Try batch processing, fall back to single on error
        - Resume capable: Skips processed images based on existing CSV

    Processing Flow:
        1. Load 4-bit quantized model on assigned GPU
        2. Load dataset and existing results (for resume)
        3. Iterate through assigned image indices
        4. Accumulate images into batches of size BATCH_SIZE
        5. Process each batch (or fall back to single if error)
        6. Save results every SAVE_EVERY images
        7. Report final statistics

    Args:
        rank (int): GPU rank/ID for this worker (0-indexed).
            Example: rank=0 uses 'cuda:0', rank=1 uses 'cuda:1'
        num_gpus (int): Total number of GPUs being used (informational,
            not actively used in current implementation).
        dataset_indices (list[int]): List of dataset indices this worker
            should process. Assigned by main process for load balancing.
        return_dict (multiprocessing.Manager.dict): Shared dictionary for
            returning statistics to main process. Updates with:
            {rank: {'processed': int, 'skipped': int}}

    Returns:
        None. Results are written to OUTPUT_FILE and statistics to return_dict.

    Error Handling:
        - Batch processing errors: Falls back to single-image processing
        - Single image errors: Logs error and continues with next image
        - CSV errors: Silently handled (treats as empty file)

    Performance:
        - Typical throughput: 50-200 images/minute per GPU
        - Memory usage: ~4-6GB per GPU (with 4-bit quantization)
        - Batch size: Configurable (default 8), higher = faster but more memory

    Example Process:
        >>> # Spawned by main process
        >>> worker_process(
        ...     rank=0,
        ...     num_gpus=4,
        ...     dataset_indices=[0, 4, 8, 12, ...],  # Every 4th image
        ...     return_dict=shared_dict
        ... )
        [GPU 0] Starting worker, processing 12500 images, batch_size=8
        [GPU 0] Model loaded on cuda:0
        GPU 0: 100%|██████████| 1562/1562 [15:23<00:00, 1.69batch/s]
        [GPU 0] Done! Processed 12500, skipped 0
    """
    # 设置当前进程使用的GPU
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    print(f"[GPU {rank}] Starting worker, processing {len(dataset_indices)} images, batch_size={BATCH_SIZE}")

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

    # 收集待处理的batch
    batch_indices = []
    batch_images = []
    batch_item_ids = []

    # 计算总batch数用于进度条
    total_batches = (len(dataset_indices) + BATCH_SIZE - 1) // BATCH_SIZE
    pbar = tqdm(total=total_batches, desc=f"GPU {rank}", position=rank)

    for idx in dataset_indices:
        example = train_data[idx]
        item_id = get_item_id(example)

        # 跳过已处理的
        if item_id in processed_ids:
            skipped_count += 1
            continue

        # 收集到batch中
        batch_indices.append(idx)
        batch_images.append(example['image'])
        batch_item_ids.append(item_id)

        # 当batch满了，处理这个batch
        if len(batch_images) >= BATCH_SIZE:
            try:
                answers = process_batch(model, processor, batch_images, device)

                for item_id, answer in zip(batch_item_ids, answers):
                    results.append({'item_id': item_id, 'summary': answer})
                    processed_ids.add(item_id)
                    processed_count += 1

                pbar.update(1)
                pbar.set_postfix({'processed': processed_count, 'batch': BATCH_SIZE})

            except Exception as e:
                tqdm.write(f"[GPU {rank}] Batch error: {e}, falling back to single processing")
                # 如果批量处理失败，逐个处理
                for img, item_id in zip(batch_images, batch_item_ids):
                    try:
                        answers = process_batch(model, processor, [img], device)
                        results.append({'item_id': item_id, 'summary': answers[0]})
                        processed_ids.add(item_id)
                        processed_count += 1
                    except Exception as e2:
                        tqdm.write(f"[GPU {rank}] Error processing {item_id}: {e2}")

            # 清空batch
            batch_indices = []
            batch_images = []
            batch_item_ids = []

            # 定期保存
            if processed_count % SAVE_EVERY == 0 and results:
                total_saved = save_results_safely(results, OUTPUT_FILE)
                results = []  # 清空已保存的结果

    # 处理剩余的不足一个batch的图片
    if batch_images:
        try:
            answers = process_batch(model, processor, batch_images, device)

            for item_id, answer in zip(batch_item_ids, answers):
                results.append({'item_id': item_id, 'summary': answer})
                processed_ids.add(item_id)
                processed_count += 1

            pbar.update(1)

        except Exception as e:
            tqdm.write(f"[GPU {rank}] Final batch error: {e}, falling back to single processing")
            for img, item_id in zip(batch_images, batch_item_ids):
                try:
                    answers = process_batch(model, processor, [img], device)
                    results.append({'item_id': item_id, 'summary': answers[0]})
                    processed_ids.add(item_id)
                    processed_count += 1
                except Exception as e2:
                    tqdm.write(f"[GPU {rank}] Error processing {item_id}: {e2}")

    pbar.close()

    # 保存剩余结果
    if results:
        save_results_safely(results, OUTPUT_FILE)

    return_dict[rank] = {
        'processed': processed_count,
        'skipped': skipped_count
    }
    print(f"[GPU {rank}] Done! Processed {processed_count}, skipped {skipped_count}")


def main():
    """
    Main orchestration function for multi-GPU parallel image summarization.

    This function coordinates the entire pipeline:
    - Validates GPU availability
    - Loads dataset to determine workload
    - Distributes work evenly across available GPUs
    - Spawns worker processes for parallel execution
    - Monitors progress and collects statistics
    - Reports final performance metrics

    Workflow:
        1. Print configuration and validate GPU count
        2. Load dataset to get total image count
        3. Load existing results to determine remaining work
        4. Distribute dataset indices across GPUs (round-robin)
        5. Spawn worker process for each GPU with 2s stagger
        6. Wait for all workers to complete
        7. Aggregate and report statistics

    Configuration (module-level constants):
        - NUM_GPUS: Number of GPUs to use
        - BATCH_SIZE: Batch size per GPU
        - DATA_DIR: Path to image dataset
        - OUTPUT_FILE: Output CSV path
        - SAVE_EVERY: Save frequency

    Load Distribution:
        Images are distributed round-robin across GPUs:
        - GPU 0: indices [0, 4, 8, 12, ...]  (if 4 GPUs)
        - GPU 1: indices [1, 5, 9, 13, ...]
        - GPU 2: indices [2, 6, 10, 14, ...]
        - GPU 3: indices [3, 7, 11, 15, ...]

        This ensures even distribution and similar workload per GPU.

    Process Management:
        - Uses 'spawn' start method for clean process isolation
        - 2-second stagger between launches to avoid OOM during model loading
        - Shared return_dict for collecting worker statistics
        - Blocking wait for all workers before final report

    Error Handling:
        - If NUM_GPUS > available GPUs: Uses available count with warning
        - Worker errors: Handled within worker, don't crash main process
        - Resume capability: Automatically continues from existing results

    Performance Metrics Reported:
        - Total images processed this run
        - Total images skipped (already done)
        - Total elapsed time
        - Processing speed (images/second)

    Example Output:
        ============================================================
        Multi-GPU Image Summary Generator
        ============================================================
        Using 3 GPUs
        Batch size per GPU: 8
        Data directory: /data/MicroLens-50k
        Output file: image_summary.csv
        ============================================================
        Loading dataset to get total count...
        Total images: 50000
        Already processed: 12000 images
        Remaining: 38000 images

        Distribution:
          GPU 0: 16667 images (2083 batches)
          GPU 1: 16667 images (2083 batches)
          GPU 2: 16666 images (2083 batches)

        Starting 3 worker processes...
        [Progress bars for each GPU]

        ============================================================
        All done!
        Total processed: 38000
        Total skipped: 12000
        Time elapsed: 45.23 minutes
        Speed: 14.01 images/second
        Results saved to: image_summary.csv
        ============================================================
    """
    print(f"=" * 60)
    print(f"Multi-GPU Image Summary Generator")
    print(f"=" * 60)
    print(f"Using {NUM_GPUS} GPUs")
    print(f"Batch size per GPU: {BATCH_SIZE}")
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
        print(f"  GPU {i}: {len(indices)} images ({len(indices)//BATCH_SIZE} batches)")

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
    if elapsed_time > 0:
        print(f"Speed: {total_processed/elapsed_time:.2f} images/second")
    print(f"Results saved to: {OUTPUT_FILE}")
    print(f"=" * 60)


if __name__ == "__main__":
    main()
