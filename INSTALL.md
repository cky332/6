# MLLM-MSR 项目部署指南

本文档提供从零开始部署 MLLM-MSR (Multimodal LLM for Multimodal Sequential Recommendation) 项目的完整步骤。

## 系统要求

- **操作系统**: Linux (推荐 Ubuntu 20.04/22.04)
- **GPU**: NVIDIA GPU，显存 >= 24GB (推荐 A100 40GB/80GB 或 RTX 4090)
- **CUDA**: 11.8 或 12.1+
- **内存**: >= 64GB RAM
- **磁盘**: >= 100GB 可用空间

## 一、安装 Anaconda

如果尚未安装 Anaconda，请按以下步骤安装：

```bash
# 下载 Anaconda 安装脚本
wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh

# 运行安装脚本
bash Anaconda3-2024.02-1-Linux-x86_64.sh

# 按提示完成安装，然后初始化 conda
source ~/.bashrc

# 验证安装
conda --version
```

## 二、创建 Conda 环境

### 方法一：使用 environment.yml（推荐）

```bash
# 进入项目目录
cd /path/to/MLLM-MSR

# 创建环境（这会自动安装所有依赖）
conda env create -f environment.yml

# 激活环境
conda activate mllm-msr
```

### 方法二：手动创建环境

```bash
# 1. 创建新环境
conda create -n mllm-msr python=3.10 -y

# 2. 激活环境
conda activate mllm-msr

# 3. 安装 PyTorch（根据你的 CUDA 版本选择）
# CUDA 12.1
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 或 CUDA 11.8
# conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 4. 安装其他依赖
pip install -r requirements.txt
```

## 三、安装 Flash Attention（重要）

Flash Attention 可以显著加速训练和推理，但需要单独编译安装：

```bash
# 确保已安装编译工具
sudo apt-get update
sudo apt-get install -y build-essential

# 安装 flash-attn（需要几分钟编译时间）
pip install flash-attn --no-build-isolation

# 如果上述命令失败，可以尝试：
pip install flash-attn==2.3.6 --no-build-isolation
```

**注意**: Flash Attention 需要 Ampere 架构或更新的 GPU (如 A100, RTX 30xx, RTX 40xx)。

## 四、配置 Hugging Face

项目使用 Hugging Face 模型，需要登录以下载 LLaVA 和 LLaMA 模型：

```bash
# 安装 huggingface-cli
pip install huggingface_hub[cli]

# 登录 Hugging Face
huggingface-cli login
# 输入你的 Hugging Face token
```

**获取 Token**: 访问 https://huggingface.co/settings/tokens 创建 Access Token。

**注意**: 使用 LLaMA 模型需要先在 Hugging Face 上申请访问权限：
- LLaMA 3: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

## 五、下载 NLTK 数据

```bash
python -c "import nltk; nltk.download('punkt')"
```

## 六、准备数据集

### 6.1 MicroLens 数据集

1. 从 [MicroLens GitHub](https://github.com/westlake-repl/MicroLens) 下载数据
2. 将数据放置到 `MLLM-MSR/data/microlens/` 目录

### 6.2 Amazon Review 数据集

1. 从 [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/#grouped-by-category) 下载数据
2. 运行数据处理脚本：

```bash
cd MLLM-MSR/data/amazon
python process_data.py
python download_images.py
```

## 七、运行项目

按照以下顺序执行：

### 7.1 图像摘要生成

```bash
cd MLLM-MSR
python Inference/microlens/image_summary.py
```

### 7.2 用户偏好推理

```bash
python Inference/microlens/preferece_inference_recurrent.py
```

### 7.3 创建训练数据集

```bash
python train/microlens/dataset_create.py
```

### 7.4 创建测试数据集

```bash
python test/microlens/multi_col_dataset.py
```

### 7.5 训练模型

```bash
python train/microlens/train_llava_sft.py
```

### 7.6 测试模型

```bash
python test/microlens/test_with_llava_sft.py
```

## 八、配置说明

### 修改数据路径

代码中的数据路径可能需要根据你的实际情况修改：

- `train_llava_sft.py` 中的 `cache_dir` 和 `SAVE_DIR`
- `image_summary.py` 中的 `img_dir`
- 各文件中的 `CUDA_VISIBLE_DEVICES` 环境变量

### GPU 配置

根据可用 GPU 数量修改训练脚本中的参数：

```python
# train_llava_sft.py
trainer = L.Trainer(
    devices=6,  # 改为你的 GPU 数量
    ...
)
```

### 内存优化

如果 GPU 显存不足，可以启用量化：

```python
# 在 train_llava_sft.py 中
USE_QLORA = True  # 启用 4-bit 量化
```

## 九、常见问题

### Q1: CUDA out of memory

- 减小 `batch_size`
- 启用 `USE_QLORA = True`
- 减少 `num_proc` 参数

### Q2: Flash Attention 安装失败

- 确保 CUDA 版本与 PyTorch 匹配
- 尝试安装预编译版本：`pip install flash-attn --no-build-isolation`
- 如果仍失败，可以在代码中注释掉 `_attn_implementation="flash_attention_2"` 参数

### Q3: 模型下载慢

设置 Hugging Face 镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 十、环境验证

运行以下脚本验证环境是否正确配置：

```python
import torch
import transformers
import lightning
import peft

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"Transformers version: {transformers.__version__}")
print(f"Lightning version: {lightning.__version__}")
print(f"PEFT version: {peft.__version__}")

# 测试 Flash Attention
try:
    from flash_attn import flash_attn_func
    print("Flash Attention: Available")
except ImportError:
    print("Flash Attention: Not installed")
```

## 联系方式

如有问题，请参考论文或在 GitHub 仓库提交 Issue。
