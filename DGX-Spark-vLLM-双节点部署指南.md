# DGX Spark 双节点 vLLM 分布式推理部署指南

## 概述

本文档记录在两台 NVIDIA DGX Spark (GB10 Blackwell GPU) 上使用 Pipeline Parallel (PP=2) 部署 Qwen2.5-32B-Instruct 模型的完整过程。

**部署日期**: 2026年2月8日

---

## 1. 硬件环境

### 节点配置

| 项目 | Node A (Head) | Node B (Worker) |
|------|---------------|-----------------|
| 主机名 | gx10-beee | gx10-c6fb |
| 外网 IP | 192.168.34.100 | 192.168.34.101 |
| 内网 IP (200G 堆栈) | 172.16.100.1 | 172.16.100.2 |
| GPU | NVIDIA GB10 (Blackwell) | NVIDIA GB10 (Blackwell) |
| GPU 显存 | 128 GB | 128 GB |
| CUDA Capability | 12.1 (sm_121) | 12.1 (sm_121) |
| CPU | ARM64 (aarch64) | ARM64 (aarch64) |
| 内存 | ~80 GB | ~80 GB |

### 网络配置

- **分布式通信**: 必须使用 172.16.100.x 网络 (200Gbps 堆栈互联)
- **外部访问**: 192.168.34.x 网络
- **SSH 用户**: gx10

---

## 2. 软件环境

### 系统信息

```
OS: Ubuntu (Linux 6.14.0-1015-nvidia)
Architecture: aarch64 (ARM64)
CUDA Driver: 580.95.05
CUDA Toolkit: 13.0
```

### Python 环境

```
Python: 3.12.12
Conda: Miniforge3
环境名称: buddhist-train
环境路径: ~/miniforge3/envs/buddhist-train/
```

### 关键依赖版本

| 包名 | 版本 | 备注 |
|------|------|------|
| PyTorch | 2.9.0+cu130 | 需从源码编译支持 CUDA 13.0 |
| vLLM | 0.11.3.dev0+g275de3417.d20260208.cu130 | 从源码编译 |
| Ray | 2.51.1 | pip 安装 |
| flash-attn | 2.7.4.post1+25.11 | NGC 容器提取 |
| flashinfer-python | 0.5.1 | NGC 容器提取 |
| flashinfer-cubin | 0.5.1+28c4ab8a.nv25.11 | NGC 容器提取 |
| nvidia-nccl-cu13 | 2.27.7 | pip 安装 |
| transformers | 4.57.1 | pip 安装 |
| triton | 3.5.0+git8daff01a | 从源码编译 |
| safetensors | 0.6.2 | pip 安装 |

---

## 3. 环境搭建

### 3.1 创建 Conda 环境

```bash
# 安装 Miniforge (ARM64 版本)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh

# 创建环境
conda create -n buddhist-train python=3.12 -y
conda activate buddhist-train
```

### 3.2 安装 PyTorch (CUDA 13.0)

由于 PyTorch 官方不提供 CUDA 13.0 的预编译包，需要从 NVIDIA NGC 容器提取或从源码编译：

```bash
# 方法1: 从 NGC 容器提取 (推荐)
# 使用 vllm/vllm-pytorch-native:v0.12.0-dgx-spark 容器中的 wheel

# 方法2: 从源码编译
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git checkout v2.9.0
pip install -r requirements.txt
USE_CUDA=1 TORCH_CUDA_ARCH_LIST="12.1" python setup.py bdist_wheel
pip install dist/torch-*.whl
```

### 3.3 安装 Flash Attention 和 FlashInfer

```bash
# 从 NGC 容器提取预编译的 wheel
# flash_attn-2.7.4.post1+25.11-*.whl
# flashinfer_cubin-0.5.1+28c4ab8a.nv25.11-*.whl
# flashinfer_python-0.5.1-*.whl

pip install flash_attn-*.whl
pip install flashinfer_cubin-*.whl flashinfer_python-*.whl
```

### 3.4 配置 FlashInfer 版本检查绕过

```bash
# 创建 conda activate 钩子
mkdir -p ~/miniforge3/envs/buddhist-train/etc/conda/activate.d
echo 'export FLASHINFER_DISABLE_VERSION_CHECK=1' > \
    ~/miniforge3/envs/buddhist-train/etc/conda/activate.d/flashinfer.sh
```

### 3.5 编译 vLLM

```bash
# 克隆 vLLM 源码
mkdir -p ~/build/vllm-build
cd ~/build/vllm-build
git clone https://github.com/vllm-project/vllm.git
cd vllm

# 设置编译环境变量
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="12.1"
export VLLM_TARGET_DEVICE=cuda
export MAX_JOBS=16

# 安装编译依赖
pip install ninja cmake wheel packaging

# 编译安装 (开发模式)
pip install -e . --no-build-isolation

# 验证安装
python -c "import vllm; print(vllm.__version__)"
```

### 3.6 安装其他依赖

```bash
pip install ray==2.51.1
pip install nvidia-nccl-cu13==2.27.7
pip install transformers safetensors sentencepiece
```

---

## 4. 模型准备

### 4.1 下载模型

```bash
# 使用 huggingface-cli 下载
huggingface-cli download Qwen/Qwen2.5-32B-Instruct --local-dir ~/models/Qwen2.5-32B-Instruct

# 或使用 modelscope
modelscope download --model Qwen/Qwen2.5-32B-Instruct --local_dir ~/models/Qwen2.5-32B-Instruct
```

### 4.2 模型信息

| 属性 | 值 |
|------|-----|
| 模型大小 | 32B 参数 |
| 精度 | BF16 |
| 磁盘占用 | ~64 GB (17 个 safetensors 分片) |
| 加载后显存 | ~30.5 GB per GPU (PP=2) |

---

## 5. 双节点部署

### 5.1 启动 Ray 集群

**Node A (Head 节点):**

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate buddhist-train

ray start --head \
    --node-ip-address=172.16.100.1 \
    --port=6379 \
    --dashboard-host=0.0.0.0 \
    --num-gpus=1
```

**Node B (Worker 节点):**

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate buddhist-train

ray start \
    --address=172.16.100.1:6379 \
    --num-gpus=1
```

**验证集群状态:**

```bash
ray status

# 预期输出:
# Active: 2 nodes
# Resources: 40.0 CPU, 2.0 GPU
```

### 5.2 启动 vLLM 服务

**关键环境变量:**

```bash
export RAY_ADDRESS=172.16.100.1:6379
export VLLM_HOST_IP=172.16.100.1

# 解决 Ray Compiled DAG bug 的关键设置
export VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE=shm
```

**启动命令:**

```bash
python -m vllm.entrypoints.openai.api_server \
    --model ~/models/Qwen2.5-32B-Instruct \
    --pipeline-parallel-size 2 \
    --distributed-executor-backend ray \
    --enforce-eager \
    --gpu-memory-utilization 0.80 \
    --max-model-len 4096 \
    --host 0.0.0.0 \
    --port 8000
```

**后台运行:**

```bash
nohup python -m vllm.entrypoints.openai.api_server \
    --model ~/models/Qwen2.5-32B-Instruct \
    --pipeline-parallel-size 2 \
    --distributed-executor-backend ray \
    --enforce-eager \
    --gpu-memory-utilization 0.80 \
    --max-model-len 4096 \
    --host 0.0.0.0 \
    --port 8000 > /tmp/vllm.log 2>&1 &
```

### 5.3 参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `--pipeline-parallel-size` | 2 | 跨 2 个节点进行 Pipeline 并行 |
| `--distributed-executor-backend` | ray | 使用 Ray 作为分布式执行后端 |
| `--enforce-eager` | - | 禁用 CUDA Graph，避免兼容性问题 |
| `--gpu-memory-utilization` | 0.80 | 使用 80% GPU 显存 |
| `--max-model-len` | 4096 | 最大序列长度 |

---

## 6. 问题排查与解决

### 6.1 Ray Compiled DAG Bug

**问题描述:**
在 ARM64 多节点环境下，Ray Compiled DAG 执行时崩溃：
```
Check failed: object_manager_->WriteAcquire(...) 
Status not OK: ChannelError: Channel closed
```

**解决方案:**
```bash
export VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE=shm
```

### 6.2 FlashInfer 版本检查失败

**问题描述:**
FlashInfer 与 PyTorch 版本不匹配导致启动失败。

**解决方案:**
```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
# 或在 conda activate.d 中配置
```

### 6.3 GPU 资源未释放

**问题描述:**
之前的 vLLM 进程未正常退出，导致 GPU 显存被占用。

**解决方案:**
```bash
# 查看 GPU 进程
nvidia-smi

# 强制杀死残留进程
pkill -9 -f 'vllm'

# 重启 Ray 集群
ray stop --force
ray start ...
```

### 6.4 网络配置错误

**问题描述:**
使用外网 IP (192.168.34.x) 导致分布式通信失败。

**解决方案:**
- Ray 集群必须使用内网 IP (172.16.100.x)
- 设置 `VLLM_HOST_IP=172.16.100.1`

---

## 7. 性能测试结果

### 7.1 测试环境

- **模型**: Qwen2.5-32B-Instruct (BF16)
- **配置**: PP=2, 2x GB10 GPU
- **测试工具**: vLLM bench serve

### 7.2 测试结果

#### 测试 1: 低并发 (30 请求, 2 RPS)

| 指标 | 数值 |
|------|------|
| 输入长度 | 512 tokens |
| 输出长度 | 256 tokens |
| 成功率 | 100% (30/30) |
| 输出吞吐量 | 66.55 tok/s |
| 峰值吞吐量 | 90.00 tok/s |
| 总吞吐量 | 199.64 tok/s |
| 平均 TTFT | 4,970 ms |
| 平均 TPOT | 394.71 ms |
| P99 TPOT | 432.37 ms |

#### 测试 2: 高并发短输出 (50 请求, 无限速率)

| 指标 | 数值 |
|------|------|
| 输入长度 | 256 tokens |
| 输出长度 | 128 tokens |
| 成功率 | 100% (50/50) |
| 输出吞吐量 | 115.37 tok/s |
| 峰值吞吐量 | 150.00 tok/s |
| 总吞吐量 | 346.11 tok/s |
| 平均 TTFT | 4,272 ms |
| 平均 TPOT | 401.32 ms |
| P99 TPOT | 417.56 ms |

#### 测试 3: 高并发长输出 (100 请求, 无限速率)

| 指标 | 数值 |
|------|------|
| 输入长度 | 512 tokens |
| 输出长度 | 512 tokens |
| 成功率 | 100% (100/100) |
| **输出吞吐量** | **198.00 tok/s** |
| **峰值吞吐量** | **264.00 tok/s** |
| **总吞吐量** | **396.00 tok/s** |
| 平均 TTFT | 12,801 ms |
| 平均 TPOT | 477.63 ms |
| P99 TPOT | 493.55 ms |

### 7.3 性能总结

| 指标 | 最佳值 |
|------|--------|
| 最大输出吞吐量 | 264 tok/s |
| 最大总吞吐量 | 396 tok/s |
| 单 token 生成延迟 | ~400-480 ms |
| 最大并发请求 | 100 |
| 成功率 | 100% |

### 7.4 资源利用

| 节点 | GPU 显存使用 | 模型分片 |
|------|-------------|---------|
| Node A | ~94 GB | 前半层 (layers 0-31) |
| Node B | ~95 GB | 后半层 (layers 32-63) |

**KV Cache 容量:**
- 单节点 (TP=1): 152,160 tokens
- 双节点 (PP=2): **511,392 tokens** (3.4x 提升)

---

## 8. API 使用示例

### 8.1 健康检查

```bash
curl http://192.168.34.100:8000/health
```

### 8.2 获取模型列表

```bash
curl http://192.168.34.100:8000/v1/models
```

### 8.3 聊天补全

```bash
curl http://192.168.34.100:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/home/gx10/models/Qwen2.5-32B-Instruct",
    "messages": [
      {"role": "system", "content": "你是一个友好的AI助手。"},
      {"role": "user", "content": "你好！"}
    ],
    "max_tokens": 500,
    "temperature": 0.7
  }'
```

### 8.4 Python 客户端

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://192.168.34.100:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="/home/gx10/models/Qwen2.5-32B-Instruct",
    messages=[
        {"role": "user", "content": "用Python写一个快速排序算法"}
    ],
    max_tokens=500
)

print(response.choices[0].message.content)
```

---

## 9. 运维命令

### 9.1 查看服务状态

```bash
# 查看 Ray 集群状态
ray status

# 查看 GPU 使用情况
nvidia-smi

# 查看 vLLM 日志
tail -f /tmp/vllm.log
```

### 9.2 停止服务

```bash
# 停止 vLLM
pkill -f 'vllm.entrypoints'

# 停止 Ray 集群 (两个节点都要执行)
ray stop
```

### 9.3 重启服务

```bash
# 1. 停止所有服务
pkill -9 -f 'vllm'
ray stop --force  # 两个节点都执行

# 2. 重启 Ray 集群
# Node A:
ray start --head --node-ip-address=172.16.100.1 --port=6379 --dashboard-host=0.0.0.0 --num-gpus=1

# Node B:
ray start --address=172.16.100.1:6379 --num-gpus=1

# 3. 启动 vLLM
export RAY_ADDRESS=172.16.100.1:6379
export VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE=shm
python -m vllm.entrypoints.openai.api_server ...
```

---

## 10. 注意事项

### 必须遵守

1. ❌ **不要使用 Docker** - 本指南使用原生 Conda 环境
2. ❌ **不要使用 PyPI 预编译的 vLLM** - 不兼容 CUDA 13.0
3. ❌ **不要使用 192.168.34.x 进行分布式通信** - 必须用 172.16.100.x
4. ❌ **不要忘记 `--enforce-eager`** - 防止 CUDA Graph 问题
5. ❌ **不要忘记 `VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE=shm`** - 解决 Ray bug

### 建议配置

1. ✅ 使用 `--gpu-memory-utilization 0.80` 避免显存不足
2. ✅ 使用 `--max-model-len 4096` 限制序列长度以节省显存
3. ✅ 将 FlashInfer 版本检查绕过写入 conda activate.d
4. ✅ 使用 200G 堆栈网络进行节点间通信

---

## 附录 A: 完整启动脚本

### start_cluster.sh (在两个节点上都放置)

```bash
#!/bin/bash
# 双节点 vLLM 集群启动脚本

NODE_A_IP="172.16.100.1"
NODE_B_IP="172.16.100.2"
MODEL_PATH="$HOME/models/Qwen2.5-32B-Instruct"

# 初始化 Conda
source ~/miniforge3/etc/profile.d/conda.sh
conda activate buddhist-train

# 设置环境变量
export RAY_ADDRESS=${NODE_A_IP}:6379
export VLLM_HOST_IP=${NODE_A_IP}
export VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE=shm

# 检测当前节点
CURRENT_IP=$(hostname -I | awk '{print $2}')

if [ "$CURRENT_IP" == "$NODE_A_IP" ]; then
    echo "Starting as HEAD node..."
    ray stop --force 2>/dev/null
    ray start --head \
        --node-ip-address=${NODE_A_IP} \
        --port=6379 \
        --dashboard-host=0.0.0.0 \
        --num-gpus=1
    
    echo "Waiting for worker node to join..."
    sleep 10
    
    echo "Starting vLLM server..."
    python -m vllm.entrypoints.openai.api_server \
        --model ${MODEL_PATH} \
        --pipeline-parallel-size 2 \
        --distributed-executor-backend ray \
        --enforce-eager \
        --gpu-memory-utilization 0.80 \
        --max-model-len 4096 \
        --host 0.0.0.0 \
        --port 8000
else
    echo "Starting as WORKER node..."
    ray stop --force 2>/dev/null
    ray start --address=${NODE_A_IP}:6379 --num-gpus=1
    echo "Worker node joined the cluster."
fi
```

---

## 附录 B: 依赖版本锁定

```
# requirements.txt
torch==2.9.0+cu130
ray==2.51.1
nvidia-nccl-cu13==2.27.7
transformers==4.57.1
safetensors==0.6.2
sentencepiece==0.2.1
# vLLM 从源码编译
# flash-attn 和 flashinfer 从 NGC 容器提取
```

---

*文档版本: 1.0*
*最后更新: 2026-02-08*
