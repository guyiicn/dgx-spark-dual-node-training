# DGX Spark 双节点分布式训练环境搭建指南

本文档详细记录了在双节点 NVIDIA DGX Spark (GX10) 集群上搭建 Qwen2.5-32B-Instruct LoRA 微调环境的完整过程。

## 目录

1. [硬件配置](#硬件配置)
2. [网络拓扑](#网络拓扑)
3. [软件环境搭建](#软件环境搭建)
4. [NCCL 编译](#nccl-编译)
5. [训练配置](#训练配置)
6. [模型下载](#模型下载)
7. [启动训练](#启动训练)
8. [故障排除](#故障排除)

---

## 硬件配置

### 节点规格

| 项目 | 规格 |
|------|------|
| 型号 | NVIDIA DGX Spark (ASUS GX10) |
| CPU | ARM64 (aarch64) |
| GPU | NVIDIA GB10 (Blackwell 架构, sm_121) |
| 内存 | 128GB 统一内存 (GPU/CPU 共享) |
| CUDA | 13.0 |
| 操作系统 | Ubuntu 24.04 |

### 节点信息

| 节点 | 主机名 | 管理网 IP | 200G RoCE IP | 角色 |
|------|--------|-----------|--------------|------|
| Node A | gx10-beee | 192.168.34.100 | 172.16.100.1 | Ray Head / Rank 0 |
| Node B | gx10-c6fb | 192.168.34.101 | 172.16.100.2 | Ray Worker / Rank 1 |

### 互联网络

- **200Gbps RoCE** via NVIDIA ConnectX-7
- 网段: 172.16.100.0/24
- 用途: NCCL 集合通信

---

## 网络拓扑

```
┌─────────────────┐     200G RoCE      ┌─────────────────┐
│   Node A        │◄──────────────────►│   Node B        │
│   gx10-beee     │   172.16.100.x     │   gx10-c6fb     │
│   GB10 GPU      │                    │   GB10 GPU      │
│   128GB Unified │                    │   128GB Unified │
└─────────────────┘                    └─────────────────┘
       │                                      │
       │ 1GbE 管理网                          │ 1GbE 管理网
       │ 192.168.34.100                       │ 192.168.34.101
       └──────────────────┬───────────────────┘
                          │
                    ┌─────▼─────┐
                    │  交换机    │
                    └───────────┘
```

---

## 软件环境搭建

### 1. 安装 Miniforge (两节点)

DGX Spark 使用 ARM64 架构，需要使用 Miniforge 而非 Anaconda:

```bash
# 下载 ARM64 版本
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh

# 安装到 ~/miniforge3
bash Miniforge3-Linux-aarch64.sh -b -p ~/miniforge3

# 初始化
~/miniforge3/bin/conda init bash
source ~/.bashrc
```

### 2. 创建训练环境 (Node A)

```bash
# 创建 Python 3.12 环境
conda create -n buddhist-train python=3.12 -y
conda activate buddhist-train

# 安装 PyTorch 2.9.0 with CUDA 13.0 (nightly)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

# 安装训练依赖
pip install transformers==4.57.1
pip install peft==0.17.1
pip install accelerate==1.11.0
pip install datasets
pip install trl
pip install scipy

# 安装 LLaMA-Factory
pip install llamafactory==0.9.4
```

### 3. 复制环境到 Node B

使用 conda-pack 打包并通过 200G 链路传输:

```bash
# Node A: 打包环境
conda activate buddhist-train
pip install conda-pack
conda pack -n buddhist-train -o buddhist-train.tar.gz

# 通过 200G 链路传输 (约 754 MB/s)
rsync -avP --compress buddhist-train.tar.gz gx10@172.16.100.2:~/

# Node B: 解压环境
ssh gx10@192.168.34.101
mkdir -p ~/miniforge3/envs/buddhist-train
tar -xzf buddhist-train.tar.gz -C ~/miniforge3/envs/buddhist-train
conda activate buddhist-train
conda-unpack  # 修复硬编码路径
```

### 4. 验证环境一致性

```bash
# 两节点分别执行
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"
python -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')"
```

---

## NCCL 编译

### 为什么需要自编译 NCCL

DGX Spark 使用 Blackwell 架构 (sm_121)，官方预编译的 NCCL 可能不包含此架构支持。需要从源码编译。

### 编译步骤 (两节点)

```bash
# 克隆 NCCL 源码
git clone https://github.com/NVIDIA/nccl.git ~/nccl
cd ~/nccl

# 编译 (支持 sm_121)
make -j$(nproc) src.build NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_121,code=sm_121"

# 验证
ls ~/nccl/build/lib/libnccl.so*
```

### 配置环境变量

创建 `~/train/scripts/env_setup.sh`:

```bash
#!/bin/bash

# NCCL 库路径
export LD_LIBRARY_PATH=$HOME/nccl/build/lib:$LD_LIBRARY_PATH

# NCCL RoCE 配置
export NCCL_IB_HCA=mlx5_0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=enp5s0  # 200G 网卡
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET

# 分布式训练
export MASTER_ADDR=172.16.100.1
export MASTER_PORT=29500

# PyTorch 配置
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 验证 NCCL 通信

创建测试脚本 `test_nccl.py`:

```python
import os
import torch
import torch.distributed as dist

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda")
    
    # 创建测试张量
    tensor = torch.ones(1024 * 1024 * 256, dtype=torch.float32, device=device)  # 1GB
    
    # 测试 allreduce
    import time
    start = time.time()
    for _ in range(10):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    if rank == 0:
        bandwidth = (10 * 1024 * 2) / elapsed  # MB/s (双向)
        print(f"AllReduce bandwidth: {bandwidth:.1f} MB/s ({bandwidth/1024:.1f} GB/s)")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

运行测试:

```bash
# Node A
source ~/train/scripts/env_setup.sh
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
    --master_addr=172.16.100.1 --master_port=29500 \
    test_nccl.py

# Node B (同时执行)
source ~/train/scripts/env_setup.sh
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 \
    --master_addr=172.16.100.1 --master_port=29500 \
    test_nccl.py
```

预期输出: `AllReduce bandwidth: ~12.4 GB/s`

---

## 训练配置

### 目录结构

```
~/train/
├── accelerate_config_node0.yaml  # Node A FSDP 配置
├── accelerate_config_node1.yaml  # Node B FSDP 配置
├── train_config.yaml             # LLaMA-Factory 配置
├── data/
│   ├── buddhist_train_alpaca.json
│   ├── buddhist_val_alpaca.json
│   └── dataset_info.json
└── scripts/
    ├── env_setup.sh
    ├── launch_train.sh
    ├── test_sdpa.py
    └── test_nccl.py
```

### Accelerate FSDP 配置

`accelerate_config_node0.yaml` (Node A, rank=0):

```yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
enable_cpu_affinity: false
fsdp_config:
  fsdp_activation_checkpointing: true
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: false
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: true
machine_rank: 0
main_process_ip: 172.16.100.1
main_process_port: 29500
main_training_function: main
mixed_precision: bf16
num_machines: 2
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

Node B 配置相同，仅 `machine_rank: 1`。

### 关键配置决策

| 配置项 | 选择 | 原因 |
|--------|------|------|
| `fsdp_offload_params: false` | 不 offload | 统一内存架构，offload 无意义 |
| `fsdp_state_dict_type: SHARDED_STATE_DICT` | 分片保存 | 避免检查点 OOM |
| `mixed_precision: bf16` | BF16 混合精度 | Blackwell 原生支持 |
| `fsdp_activation_checkpointing: true` | 启用 | 32B 模型必需 |

### LLaMA-Factory 训练配置

`train_config.yaml`:

```yaml
### Model
model_name_or_path: ~/models/Qwen2.5-32B-Instruct

### Method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 64
lora_alpha: 128
lora_dropout: 0.05

### Dataset
dataset: buddhist_train
dataset_dir: ~/train/data
template: qwen
cutoff_len: 2048
preprocessing_num_workers: 8

### Output
output_dir: ~/train/output
logging_steps: 10
save_steps: 500
save_total_limit: 3

### Training
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 2.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
gradient_checkpointing: true

### Evaluation
val_size: 0.0
per_device_eval_batch_size: 1
eval_strategy: "no"
```

### 数据集配置

`data/dataset_info.json`:

```json
{
  "buddhist_train": {
    "file_name": "buddhist_train_alpaca.json",
    "formatting": "alpaca"
  },
  "buddhist_val": {
    "file_name": "buddhist_val_alpaca.json",
    "formatting": "alpaca"
  }
}
```

---

## 模型下载

### 模型信息

- **模型**: Qwen2.5-32B-Instruct
- **大小**: ~62GB (17 个 safetensors 分片)
- **来源**: ModelScope (国内访问更快)

### 下载脚本

```bash
#!/bin/bash
MODEL_DIR=~/models/Qwen2.5-32B-Instruct
mkdir -p $MODEL_DIR
cd $MODEL_DIR

BASE_URL="https://modelscope.cn/models/Qwen/Qwen2.5-32B-Instruct/resolve/master"

# 下载配置文件
for f in config.json generation_config.json tokenizer.json tokenizer_config.json \
         vocab.json merges.txt model.safetensors.index.json; do
    wget -c "$BASE_URL/$f"
done

# 并行下载模型分片 (4 并发)
for i in $(seq -w 1 17); do
    wget -c --tries=20 --retry-connrefused --timeout=120 \
        -O "model-000${i}-of-00017.safetensors" \
        "$BASE_URL/model-000${i}-of-00017.safetensors" &
    
    # 限制并发数
    if (( $(jobs -r | wc -l) >= 4 )); then
        wait -n
    fi
done
wait
```

### 同步到 Node B

```bash
# 使用 200G 链路同步 (约 754 MB/s)
rsync -avP --compress ~/models/Qwen2.5-32B-Instruct/ \
    gx10@172.16.100.2:~/models/Qwen2.5-32B-Instruct/
```

---

## 启动训练

### 启动脚本

`scripts/launch_train.sh`:

```bash
#!/bin/bash
NODE_RANK=${1:-0}

# 加载环境
source ~/train/scripts/env_setup.sh
source ~/miniforge3/etc/profile.d/conda.sh
conda activate buddhist-train

cd ~/train

# 选择配置文件
if [ "$NODE_RANK" = "0" ]; then
    CONFIG=accelerate_config_node0.yaml
else
    CONFIG=accelerate_config_node1.yaml
fi

# 启动训练
accelerate launch --config_file $CONFIG \
    --main_process_ip 172.16.100.1 \
    --main_process_port 29500 \
    --machine_rank $NODE_RANK \
    --num_machines 2 \
    --num_processes 2 \
    -m llamafactory.train train_config.yaml
```

### 执行步骤

**Node A (Rank 0)**:
```bash
source ~/train/scripts/env_setup.sh
source ~/miniforge3/etc/profile.d/conda.sh
conda activate buddhist-train
bash ~/train/scripts/launch_train.sh 0
```

**Node B (Rank 1)** - 同时执行:
```bash
source ~/train/scripts/env_setup.sh
source ~/miniforge3/etc/profile.d/conda.sh
conda activate buddhist-train
bash ~/train/scripts/launch_train.sh 1
```

或从 Node A 远程启动 Node B:
```bash
ssh gx10@192.168.34.101 "source ~/train/scripts/env_setup.sh && \
    source ~/miniforge3/etc/profile.d/conda.sh && \
    conda activate buddhist-train && \
    bash ~/train/scripts/launch_train.sh 1"
```

---

## 故障排除

### 常见问题

#### 1. sm_121 架构警告

```
UserWarning: CUDA Architecture sm_121 is not yet fully supported
```

**解决**: 忽略此警告。PyTorch 2.9.0 在 sm_121 上正常工作。

#### 2. NCCL 初始化失败

```
NCCL error: unhandled system error
```

**检查项**:
- 确认 200G 网络连通: `ping 172.16.100.2`
- 确认 NCCL 环境变量已设置
- 确认防火墙开放端口 29500

#### 3. FlashAttention 编译失败

**原因**: FlashAttention 不支持 sm_121 (Blackwell)

**解决**: 使用 PyTorch 原生 SDPA (Scaled Dot Product Attention):
```python
# 验证 SDPA 可用
import torch
print(torch.backends.cuda.flash_sdp_enabled())  # 应返回 True
```

#### 4. DeepSpeed 初始化失败

**原因**: DeepSpeed C++/CUDA JIT 在 ARM64+CUDA13+sm_121 未经测试

**解决**: 使用 FSDP 替代 DeepSpeed

#### 5. 内存不足 (OOM)

**调整**:
- 减小 `cutoff_len` (如 1024)
- 减小 `per_device_train_batch_size` (如 1)
- 确认 `gradient_checkpointing: true`

### 监控命令

```bash
# GPU 使用情况
nvidia-smi -l 1

# NCCL 调试日志
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# 训练进度
tail -f ~/train/output/trainer_log.jsonl
```

---

## 附录

### 验证测试脚本

#### SDPA 测试 (test_sdpa.py)

```python
import torch
import torch.nn.functional as F

def test_sdpa():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    
    B, H, S, D = 2, 32, 1024, 128
    q = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
    
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True, 
        enable_math=False, 
        enable_mem_efficient=False
    ):
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        loss = out.sum()
        loss.backward()
    
    print(f"SDPA test passed!")
    print(f"  Output shape: {out.shape}")
    print(f"  flash_sdp enabled: {torch.backends.cuda.flash_sdp_enabled()}")

if __name__ == "__main__":
    test_sdpa()
```

### 环境版本参考

| 组件 | 版本 |
|------|------|
| Python | 3.12 |
| PyTorch | 2.9.0+cu130 |
| Transformers | 4.57.1 |
| PEFT | 0.17.1 |
| Accelerate | 1.11.0 |
| LLaMA-Factory | 0.9.4 |
| CUDA | 13.0 |
| NCCL | 2.28.9 (自编译) |

---

## 双节点训练 32B 模型问题总结

> **重要结论**: 对于 Qwen2.5-32B LoRA 微调任务，**双节点 DDP 是最优选择**。虽然每 step 速度相同，但双节点的 batch size 翻倍，总 steps 减半，训练时间缩短一半。FSDP 模式因通信开销过大，不推荐使用。

### 性能对比

| 训练模式 | 速度 (s/step) | Steps 数 | 预估总时间 | 推荐 |
|----------|---------------|----------|------------|------|
| 单节点 | ~24 | 1,482 | ~10 小时 | |
| **双节点 DDP** | ~24 | **741** | **~5 小时** | ✅ 推荐 |
| 双节点 FSDP (FULL_SHARD) | ~67 | 741 | ~13.8 小时 | ❌ |
| 双节点 FSDP (SHARD_GRAD_OP) | ~114 | 741 | ~23 小时 | ❌ |

**关键发现**: 双节点 DDP 和单节点每 step 速度相同 (~24s)，但因为 world_size=2 使得有效 batch size 翻倍，总 steps 从 1,482 减少到 741，训练时间缩短一半！

### 遇到的问题

#### 1. NCCL 通信初始化问题

**现象**: 训练卡在 `ncclCommInitRankConfig ... Init START`

**原因**: Node B 上存在僵尸进程，导致新旧进程冲突

**解决**: 强制杀死所有 Python 进程后重启
```bash
ssh gx10@172.16.100.2 'pkill -9 -f python; pkill -9 -f accelerate'
```

#### 2. MTU 配置不当

**现象**: 网络带宽未充分利用

**原因**: 200G RoCE 网卡默认 MTU=1500，应使用 Jumbo Frames

**解决**: 修改 MTU 为 9000
```bash
sudo ip link set enp1s0f0np0 mtu 9000  # 两节点都需执行
```

**效果**: Ping 延迟从 ~1ms 降到 ~0.26ms，但对训练速度影响有限

#### 3. GPU Direct RDMA 不可用

**现象**: NCCL 日志显示 `GPU Direct RDMA Disabled for HCA`

**原因**: Blackwell (sm_121) + ARM64 平台 GDR 支持不完整

**影响**: 数据传输需经过 CPU 复制，增加延迟

**状态**: 等待 NVIDIA 驱动/固件更新

#### 4. Node B 数据文件缺失

**现象**: `ValueError: File /home/gx10/train/data/buddhist_train_alpaca.json not found`

**原因**: Node A 上的数据文件是符号链接，rsync 同步后 Node B 上链接目标不存在

**解决**: 复制实际文件而非符号链接
```bash
scp ~/buddhist-llm-finetune/output/buddhist_train_alpaca.json gx10@172.16.100.2:~/train/data/
```

#### 5. FSDP 通信开销过大

**现象**: 双节点训练比单节点慢 70+ 倍

**分析**:
- 32B 模型 (bf16) = 64GB
- FSDP FULL_SHARD: 每个 step 需 all-gather (64GB) + reduce-scatter (64GB) = 128GB 通信
- 200Gbps = 25GB/s 理论带宽
- 理论最小通信时间 = 128GB / 25GB/s ≈ 5s/step
- 实际 67s/step，说明还有其他开销 (CPU 复制、同步等待)

**根本原因**: 
- 每节点仅 1 个 GPU，通信/计算比过高
- 训练数据量小 (3,947 样本)，无法摊薄通信成本

#### 6. DDP 模式是最优解

**尝试**: 改用 DDP 模式 (只同步 LoRA 梯度 134MB，而非整个模型 64GB)

**结果**: 
- 每 step 速度: ~24s (与单节点相同)
- 总 steps: 741 (单节点的一半，因为 batch size 翻倍)
- 总时间: ~5 小时 (单节点的一半)

**原因**: DDP 只需同步梯度，通信量小 (134MB vs FSDP 的 64GB)，通信开销可忽略不计

### 何时应该使用双节点

| 场景 | 推荐方案 |
|------|----------|
| LoRA 微调 32B 模型 | **双节点 DDP** ✅ |
| 模型 > 128GB (如 70B 全参) | 双节点 FSDP |
| 单节点内存不足 | 双节点 FSDP |
| 节点内有多 GPU (8×H100) | 多节点 DDP/FSDP |

### 优化建议

1. **LoRA 微调用 DDP**: 通信量小，双节点能有效加速
2. **全参微调用 FSDP**: 模型太大必须分片
3. **避免 FSDP 用于 LoRA**: 通信开销远大于收益
4. **检查 MTU**: 确保使用 Jumbo Frames (MTU 9000)
5. **清理僵尸进程**: 启动前确保两节点无残留进程

### 配置文件参考

#### 单节点配置 (推荐)

`accelerate_config_single.yaml`:
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
mixed_precision: bf16
num_machines: 1
num_processes: 1
use_cpu: false
```

#### 双节点 DDP 配置 (如需要)

`accelerate_config_ddp_node0.yaml`:
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
machine_rank: 0
main_process_ip: 172.16.100.1
main_process_port: 29500
mixed_precision: bf16
num_machines: 2
num_processes: 2
rdzv_backend: static
same_network: true
use_cpu: false
```

---

## 训练结果

### Qwen2.5-32B-Instruct 佛经知识 LoRA 微调

**训练配置**:
- 基础模型: Qwen2.5-32B-Instruct
- 微调方法: LoRA (r=8, alpha=16)
- 训练数据: 3,947 条佛经知识问答 (CBETA 语料)
- 验证数据: 439 条
- 训练模式: 双节点 DDP (2× NVIDIA GB10)

**训练参数**:
| 参数 | 值 |
|------|-----|
| Epochs | 3 |
| Batch Size (per device) | 1 |
| Gradient Accumulation | 8 |
| Effective Batch Size | 16 (2 nodes × 1 × 8) |
| Learning Rate | 4e-5 |
| Warmup Ratio | 0.1 |
| Max Length | 2048 |
| LoRA Rank | 8 |
| LoRA Alpha | 16 |
| Trainable Params | 134,217,728 (0.41%) |

**训练结果**:
| 指标 | 值 |
|------|-----|
| 训练时长 | 5:46:55 |
| 总 Steps | 741 |
| Train Loss | 1.2977 |
| Eval Loss | 1.4564 |
| 速度 | ~24 s/step |
| 训练样本/秒 | 0.569 |

**输出文件**:
```
~/train/saves/qwen2.5-32b-buddhist-lora/
├── adapter_config.json
├── adapter_model.safetensors
├── chat_template.jinja
├── special_tokens_map.json
├── tokenizer_config.json
└── tokenizer.json
```

**Loss 曲线分析**:
- 初始 Loss: ~2.15
- 最终 Train Loss: 1.30
- Eval Loss: 1.46 (轻微过拟合，但可接受)

**验证结论**:
- ✅ 双节点 DDP 训练成功完成
- ✅ 训练时间符合预期 (~5.8h vs 预估 5h)
- ✅ Loss 正常收敛
- ✅ 推理测试通过 (见下文)

---

## 推理评估结果

### 关键修复：禁止 CPU Offload

**问题**: 默认加载方式会将部分参数 offload 到 CPU，导致推理极慢

**解决方案**: 设置 `max_memory = {0: '100GiB', 'cpu': '0GiB'}` 禁止 CPU offload

```python
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    max_memory={0: '100GiB', 'cpu': '0GiB'},  # 关键！
    trust_remote_code=True
)
```

### 评估结果汇总

#### 1. 困惑度对比 (50 样本)

| 模型 | 困惑度 | Loss |
|------|--------|------|
| 基础模型 | 8.88 | 2.18 |
| 微调模型 | **5.02** | 1.61 |
| **改进** | **43.5%** ⬇️ | |

困惑度降低 43.5%，说明模型对佛经语料的理解能力显著提升。

#### 2. 佛经问答测试 (20 样本)

| 指标 | 值 |
|------|-----|
| BLEU-4 | 6.16% |
| ROUGE-L | 21.93% |
| 平均响应时间 | 45.1s/样本 |
| GPU 内存占用 | 61.5 GB |

**样例输出**:

Q: 请解释《楞伽经》中'五法'的内在逻辑关系

A: 《楞伽经》所立'五法'（相、名、妄想、正智、如如）并非并列分类，而是揭示心识从迷到悟的完整认知次第。首先，'相'指一切可被感知的现象形态（如色声香味触等），本无自性，唯是因缘假立；'名'则是人为安立的概念标签...

#### 3. 推理性能

| 指标 | 值 |
|------|-----|
| 生成速度 | ~3.3 tokens/s |
| 100 tokens 时间 | ~30s |
| GPU 内存 | 61.5 / 119.6 GB |
| CPU Offload | 无 |

### 评估脚本

测试脚本位于 `~/train/eval/`:

| 脚本 | 用途 |
|------|------|
| `buddhist_qa_test_v2.py` | 佛经问答测试 (优化版) |
| `perplexity_compare_v2.py` | 困惑度对比 (优化版) |
| `eval_bleu_rouge.py` | BLEU/ROUGE 评估 |

**运行示例**:
```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate buddhist-train
cd ~/train/eval

# 佛经问答测试
python buddhist_qa_test_v2.py --max_samples 20

# 困惑度对比
python perplexity_compare_v2.py --max_samples 50
```

---

## 更新日志

- **2026-02-05**: 初始环境搭建完成
- **2026-02-06**: 模型下载完成
- **2026-02-06**: 双节点训练测试完成，发现 FSDP 性能问题
- **2026-02-06**: 确认双节点 DDP 是最优方案 (5小时 vs 单节点10小时)，修正文档
- **2026-02-06**: Qwen2.5-32B 佛经 LoRA 微调完成 (5:46:55, train_loss=1.30, eval_loss=1.46)
- **2026-02-07**: 解决推理 offload 问题，完成模型评估 (困惑度降低43.5%, BLEU-4=6.16%, ROUGE-L=21.93%)
