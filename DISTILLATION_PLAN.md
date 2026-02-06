# Qwen2.5-72B → 7B 知识蒸馏规划

## 目标

将 Qwen2.5-72B-Instruct 的佛经问答能力蒸馏到 Qwen2.5-7B-Instruct，生成轻量级佛经领域专家模型。

## 硬件环境

- **平台**: 双节点 DGX Spark (GX10)
- **每节点**: 128GB 统一内存, GB10 GPU (Blackwell sm_121)
- **总内存**: 256GB
- **互联**: 200Gbps RoCE

## 工具选择

| 工具 | KD 支持 | 推荐度 | 备注 |
|------|---------|--------|------|
| **PyTorch/torchtune** | 原生支持 | ⭐⭐⭐ | 有 Qwen 官方配置，支持 FSDP |
| LLaMA-Factory | 不支持 | ⭐⭐ | 需用 SFT 模拟离线蒸馏 |
| NeMo-Aligner | 支持 | ⭐ | 配置复杂，适合大规模集群 |

**推荐方案**: 
- **方案 A (简单)**: 离线蒸馏 - 用 72B 生成回答，用 LLaMA-Factory SFT 训练 7B
- **方案 B (专业)**: 在线蒸馏 - 用 torchtune 的 KD loss (Forward KL + CE)

本文档以**方案 A (离线蒸馏)** 为主，附录提供 torchtune 方案 B 的配置。

---

## 内存需求分析

| 模型 | FP16/BF16 | INT8 | INT4 (NF4) |
|------|-----------|------|------------|
| Qwen2.5-72B | ~144GB | ~72GB | **~40GB** |
| Qwen2.5-7B | ~14GB | ~7GB | ~4GB |
| 训练开销 (7B LoRA) | ~10GB | | |

**结论**: 单节点 128GB 可容纳 INT4 量化的 72B (~40GB) + FP16 的 7B 训练 (~24GB)

---

## 蒸馏流程

```
┌────────────────────────────────────────────────────────────────┐
│  阶段 0: 当前 32B LoRA 训练 (进行中)                            │
│  预计完成: 4-6 小时                                             │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│  阶段 1: 下载模型                                               │
│  - Qwen2.5-72B-Instruct (~140GB, 37 分片)                      │
│  - Qwen2.5-7B-Instruct (~14GB, 4 分片)                         │
│  预计时间: 6-8 小时 (可与阶段0并行)                              │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│  阶段 2: 生成蒸馏数据                                           │
│  - 加载 72B (INT4 量化, ~40GB)                                  │
│  - 对 23,683 条佛经训练数据生成 Teacher 回答                     │
│  预计时间: 8-12 小时                                            │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│  阶段 3: 训练 Student 模型                                      │
│  - 用蒸馏数据训练 Qwen2.5-7B                                    │
│  - LoRA 或全量微调                                              │
│  预计时间: 2-4 小时                                             │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│  阶段 4: 评估与部署                                             │
│  - 对比 7B vs 72B 回答质量                                      │
│  - 量化部署版本                                                 │
│  预计时间: 1-2 小时                                             │
└────────────────────────────────────────────────────────────────┘

总计: ~24-36 小时
```

---

## 阶段 1: 下载模型

### 1.1 下载 72B Teacher

```bash
#!/bin/bash
# ~/distill/download_72b.sh

MODEL_DIR=~/models/Qwen2.5-72B-Instruct
mkdir -p $MODEL_DIR
cd $MODEL_DIR

BASE_URL="https://modelscope.cn/models/Qwen/Qwen2.5-72B-Instruct/resolve/master"

# 下载配置文件
for f in config.json generation_config.json tokenizer.json tokenizer_config.json \
         vocab.json merges.txt model.safetensors.index.json; do
    wget -c "$BASE_URL/$f"
done

# 并行下载 37 个模型分片 (每个约 4GB)
for i in $(seq -w 1 37); do
    wget -c --tries=20 --retry-connrefused --timeout=120 \
        -O "model-000${i}-of-00037.safetensors" \
        "$BASE_URL/model-000${i}-of-00037.safetensors" &
    
    # 限制并发数为 4
    if (( $(jobs -r | wc -l) >= 4 )); then
        wait -n
    fi
done
wait

echo "72B model download complete!"
du -sh $MODEL_DIR
```

### 1.2 下载 7B Student

```bash
#!/bin/bash
# ~/distill/download_7b.sh

MODEL_DIR=~/models/Qwen2.5-7B-Instruct
mkdir -p $MODEL_DIR
cd $MODEL_DIR

BASE_URL="https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct/resolve/master"

# 下载配置文件
for f in config.json generation_config.json tokenizer.json tokenizer_config.json \
         vocab.json merges.txt model.safetensors.index.json; do
    wget -c "$BASE_URL/$f"
done

# 下载 4 个模型分片
for i in $(seq -w 1 4); do
    wget -c --tries=20 --retry-connrefused --timeout=120 \
        -O "model-000${i}-of-00004.safetensors" \
        "$BASE_URL/model-000${i}-of-00004.safetensors" &
done
wait

echo "7B model download complete!"
du -sh $MODEL_DIR
```

---

## 阶段 2: 生成蒸馏数据

### 2.1 蒸馏数据生成脚本

```python
#!/usr/bin/env python3
# ~/distill/generate_distill_data.py
"""
使用 Qwen2.5-72B (INT4) 为佛经训练数据生成高质量回答
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import time

# 配置
MODEL_PATH = os.path.expanduser("~/models/Qwen2.5-72B-Instruct")
TRAIN_DATA = os.path.expanduser("~/train/data/buddhist_train_alpaca.json")
OUTPUT_DIR = os.path.expanduser("~/distill/data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "buddhist_distill.json")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "distill_checkpoint.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# INT4 量化配置 (72B 约需 40GB)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

print("=" * 60)
print("Loading Qwen2.5-72B-Instruct (INT4 quantized)...")
print("Expected memory usage: ~40GB")
print("=" * 60)

start_time = time.time()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

load_time = time.time() - start_time
print(f"Model loaded in {load_time:.1f}s")

# 加载训练数据
print(f"Loading training data from {TRAIN_DATA}")
with open(TRAIN_DATA, "r", encoding="utf-8") as f:
    train_data = json.load(f)
print(f"Total samples: {len(train_data)}")

# 检查是否有检查点
start_idx = 0
distill_data = []
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)
        distill_data = checkpoint.get("data", [])
        start_idx = checkpoint.get("next_idx", 0)
    print(f"Resuming from checkpoint: {start_idx}/{len(train_data)}")

# 生成函数
def generate_response(instruction: str, input_text: str = "") -> str:
    """使用 72B 模型生成回答"""
    
    # 构建 Qwen 格式的 prompt
    messages = [
        {"role": "system", "content": "你是一位精通佛学的大师，对佛经有深入的理解和研究。请用准确、专业且易懂的方式回答问题。"},
        {"role": "user", "content": instruction + ("\n" + input_text if input_text else "")}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    )
    return response.strip()

# 主循环
print(f"\nGenerating distillation data...")
print(f"Starting from index {start_idx}")

try:
    for idx in tqdm(range(start_idx, len(train_data)), initial=start_idx, total=len(train_data)):
        sample = train_data[idx]
        
        # 生成 Teacher 回答
        teacher_response = generate_response(
            sample["instruction"],
            sample.get("input", "")
        )
        
        distill_data.append({
            "instruction": sample["instruction"],
            "input": sample.get("input", ""),
            "output": teacher_response,
            "original_output": sample["output"],  # 保留原始标注供参考
        })
        
        # 每 50 条保存检查点
        if (idx + 1) % 50 == 0:
            checkpoint = {"data": distill_data, "next_idx": idx + 1}
            with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, ensure_ascii=False)
            
            # 估算剩余时间
            elapsed = time.time() - start_time
            samples_done = idx + 1 - start_idx
            if samples_done > 0:
                time_per_sample = elapsed / samples_done
                remaining = len(train_data) - idx - 1
                eta = remaining * time_per_sample / 3600
                print(f"\n[Checkpoint] {idx+1}/{len(train_data)} | ETA: {eta:.1f}h")

except KeyboardInterrupt:
    print("\nInterrupted! Saving checkpoint...")
    checkpoint = {"data": distill_data, "next_idx": len(distill_data)}
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False)
    print(f"Saved {len(distill_data)} samples to checkpoint")
    exit(1)

# 保存最终结果
print(f"\nSaving {len(distill_data)} distillation samples to {OUTPUT_FILE}")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(distill_data, f, ensure_ascii=False, indent=2)

# 创建 dataset_info.json
dataset_info = {
    "buddhist_distill": {
        "file_name": "buddhist_distill.json",
        "formatting": "alpaca"
    }
}
with open(os.path.join(OUTPUT_DIR, "dataset_info.json"), "w") as f:
    json.dump(dataset_info, f, indent=2)

print("=" * 60)
print("Distillation data generation complete!")
print(f"Total samples: {len(distill_data)}")
print(f"Output: {OUTPUT_FILE}")
print("=" * 60)
```

### 2.2 运行蒸馏数据生成

```bash
# 在 Node A 运行
source ~/miniforge3/etc/profile.d/conda.sh
conda activate buddhist-train

# 确保安装了 bitsandbytes
pip install bitsandbytes

# 使用 screen/tmux 防止断连
screen -S distill
python ~/distill/generate_distill_data.py

# 预计 8-12 小时，生成 23,683 条蒸馏数据
```

---

## 阶段 3: 训练 Student 模型

### 3.1 训练配置

```yaml
# ~/distill/train_7b_config.yaml
### Model
model_name_or_path: ~/models/Qwen2.5-7B-Instruct

### Method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 64
lora_alpha: 128
lora_dropout: 0.05

### Dataset
dataset: buddhist_distill
dataset_dir: ~/distill/data
template: qwen
cutoff_len: 2048
preprocessing_num_workers: 8

### Output
output_dir: ~/distill/output_7b
logging_steps: 10
save_steps: 500
save_total_limit: 3

### Training
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 5.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
gradient_checkpointing: true

### Evaluation
val_size: 0.1
per_device_eval_batch_size: 4
eval_strategy: steps
eval_steps: 500
```

### 3.2 Accelerate 配置 (单节点)

```yaml
# ~/distill/accelerate_config.yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: 'NO'
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
use_cpu: false
```

### 3.3 启动训练

```bash
#!/bin/bash
# ~/distill/train_student.sh

source ~/miniforge3/etc/profile.d/conda.sh
conda activate buddhist-train

cd ~/distill

accelerate launch --config_file accelerate_config.yaml \
    -m llamafactory.train train_7b_config.yaml

echo "Student training complete!"
```

---

## 阶段 4: 评估与部署

### 4.1 评估脚本

```python
#!/usr/bin/env python3
# ~/distill/evaluate.py
"""对比 72B Teacher vs 7B Student 的回答质量"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 测试问题
TEST_QUESTIONS = [
    "请解释《心经》中'色即是空,空即是色'的含义",
    "什么是四圣谛?",
    "禅宗的'明心见性'是什么意思?",
    "《金刚经》的核心思想是什么?",
    "佛教中的'无我'概念如何理解?",
]

def load_student_model():
    """加载训练后的 7B Student 模型"""
    base_model = AutoModelForCausalLM.from_pretrained(
        "~/models/Qwen2.5-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, "~/distill/output_7b")
    tokenizer = AutoTokenizer.from_pretrained("~/models/Qwen2.5-7B-Instruct")
    return model, tokenizer

def generate_answer(model, tokenizer, question):
    messages = [
        {"role": "system", "content": "你是一位精通佛学的大师。"},
        {"role": "user", "content": question}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
    
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

# 运行评估
model, tokenizer = load_student_model()

print("=" * 60)
print("7B Student Model Evaluation")
print("=" * 60)

for q in TEST_QUESTIONS:
    print(f"\n问: {q}")
    print(f"答: {generate_answer(model, tokenizer, q)}")
    print("-" * 40)
```

### 4.2 导出合并模型

```python
# ~/distill/export_merged.py
"""将 LoRA 权重合并到基座模型并导出"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# 加载基座和 LoRA
base_model = AutoModelForCausalLM.from_pretrained(
    "~/models/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(base_model, "~/distill/output_7b")

# 合并权重
merged_model = model.merge_and_unload()

# 保存
output_path = "~/models/Qwen2.5-7B-Buddhist"
merged_model.save_pretrained(output_path)
AutoTokenizer.from_pretrained("~/models/Qwen2.5-7B-Instruct").save_pretrained(output_path)

print(f"Merged model saved to {output_path}")
```

---

## 目录结构

```
~/distill/
├── download_72b.sh           # 下载 72B 模型
├── download_7b.sh            # 下载 7B 模型
├── generate_distill_data.py  # 生成蒸馏数据
├── train_7b_config.yaml      # LLaMA-Factory 训练配置
├── accelerate_config.yaml    # Accelerate 配置
├── train_student.sh          # 启动训练脚本
├── evaluate.py               # 评估脚本
├── export_merged.py          # 导出合并模型
├── data/
│   ├── buddhist_distill.json    # 蒸馏数据
│   └── dataset_info.json
└── output_7b/                # 训练输出
    └── adapter_model.safetensors
```

---

## 可选优化

### 混合训练 (推荐)

混合使用 Teacher 输出和原始标注，提高模型鲁棒性:

```python
import random

mixed_data = []
for item in distill_data:
    if random.random() < 0.7:  # 70% 用 Teacher 输出
        mixed_data.append({
            "instruction": item["instruction"],
            "input": item["input"],
            "output": item["output"],  # Teacher's response
        })
    else:  # 30% 用原始标注
        mixed_data.append({
            "instruction": item["instruction"],
            "input": item["input"],
            "output": item["original_output"],  # Original annotation
        })
```

### 使用更大的 LoRA Rank

对于知识蒸馏，更大的 rank 可能效果更好:

```yaml
lora_rank: 128
lora_alpha: 256
```

### 全量微调 (如果内存允许)

7B 全量微调效果可能更好:

```yaml
finetuning_type: full
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
```

---

## 附录: 方案 B - torchtune 在线蒸馏

如果需要更专业的蒸馏效果 (使用 KL 散度 loss)，可以使用 PyTorch/torchtune。

### 安装 torchtune

```bash
pip install torchtune
```

### 配置文件 (基于官方 Qwen3 模板)

```yaml
# ~/distill/torchtune_kd_config.yaml

# Model configs
model:
  _component_: torchtune.models.qwen2_5.lora_qwen2_5_7b_instruct
  lora_attn_modules: ['q_proj', 'k_proj', 'v_proj', 'o_proj']
  lora_rank: 64
  lora_alpha: 128

teacher_model:
  _component_: torchtune.models.qwen2_5.qwen2_5_72b_instruct

tokenizer:
  _component_: torchtune.models.qwen2_5.qwen2_5_tokenizer
  path: ~/models/Qwen2.5-7B-Instruct/

# Checkpoints
checkpointer:
  _component_: torchtune.training.FullModelHFCheckpointer
  checkpoint_dir: ~/models/Qwen2.5-7B-Instruct/
  output_dir: ~/distill/torchtune_output/
  model_type: QWEN2

teacher_checkpointer:
  _component_: torchtune.training.FullModelHFCheckpointer
  checkpoint_dir: ~/models/Qwen2.5-72B-Instruct/
  model_type: QWEN2

# Dataset
dataset:
  _component_: torchtune.datasets.alpaca_dataset
  source: ~/distill/data/buddhist_train_alpaca.json
  train_on_input: False

# Loss - 核心蒸馏配置
loss:
  _component_: torchtune.modules.loss.CEWithChunkedOutputLoss

kd_loss:
  _component_: torchtune.modules.loss.ForwardKLWithChunkedOutputLoss

kd_ratio: 0.5  # 50% CE loss + 50% KD loss

# Training
batch_size: 1
epochs: 3
max_steps_per_epoch: null
gradient_accumulation_steps: 8
optimizer:
  _component_: torch.optim.AdamW
  lr: 5e-5
  weight_decay: 0.01

lr_scheduler:
  _component_: torchtune.training.lr_schedulers.get_cosine_schedule_with_warmup
  num_warmup_steps: 100

# Memory optimization
enable_activation_checkpointing: True
compile: False
dtype: bf16

# Distributed (FSDP for 72B teacher)
fsdp:
  cpu_offload: True  # 必需，72B 太大
```

### 运行训练

```bash
# 单节点 (使用 CPU offload)
tune run knowledge_distillation_single_device \
    --config ~/distill/torchtune_kd_config.yaml

# 多节点 FSDP (推荐)
tune run --nproc_per_node=1 --nnodes=2 \
    knowledge_distillation_distributed \
    --config ~/distill/torchtune_kd_config.yaml
```

### torchtune vs 离线蒸馏对比

| 方面 | 离线蒸馏 (方案 A) | torchtune (方案 B) |
|------|-------------------|-------------------|
| 复杂度 | 低 | 中 |
| 内存效率 | 高 (分阶段) | 中 (需同时加载) |
| 蒸馏质量 | 好 | 更好 (KL loss) |
| 灵活性 | 高 | 需适配配置 |
| 依赖 | LLaMA-Factory | torchtune |

**建议**: 先用方案 A 快速验证，效果不理想再用方案 B 优化。

---

## 检查清单

- [ ] 当前 32B LoRA 训练完成
- [ ] 下载 Qwen2.5-72B-Instruct
- [ ] 下载 Qwen2.5-7B-Instruct
- [ ] 安装 bitsandbytes (用于 INT4 量化)
- [ ] 生成蒸馏数据 (23,683 条)
- [ ] 训练 7B Student
- [ ] 评估模型质量
- [ ] 导出合并模型

---

## 更新日志

- **2026-02-06**: 初始规划完成，添加 torchtune 方案作为附录
