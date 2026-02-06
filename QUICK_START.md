# DGX Spark 双节点训练 - 快速参考卡

## 节点信息

| 节点 | 管理网 IP | 200G IP | 用户/密码 |
|------|-----------|---------|----------|
| Node A (Head) | 192.168.34.100 | 172.16.100.1 | gx10 / Password01! |
| Node B (Worker) | 192.168.34.101 | 172.16.100.2 | gx10 / Password01! |

## 常用命令

### SSH 连接

```bash
# 直连 Node A
ssh gx10@192.168.34.100

# 通过 Node A 跳转到 Node B
ssh gx10@192.168.34.100 "ssh gx10@192.168.34.101 'command'"
```

### 激活环境

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate buddhist-train
source ~/train/scripts/env_setup.sh
```

### 启动训练

```bash
# Node A (终端 1)
bash ~/train/scripts/launch_train.sh 0

# Node B (终端 2，同时执行)
bash ~/train/scripts/launch_train.sh 1
```

### 监控

```bash
# GPU 状态
nvidia-smi -l 1

# 训练日志
tail -f ~/train/output/trainer_log.jsonl

# NCCL 通信调试
export NCCL_DEBUG=INFO
```

## 文件位置

| 内容 | 路径 |
|------|------|
| Conda 环境 | ~/miniforge3/envs/buddhist-train |
| 训练配置 | ~/train/*.yaml |
| 训练数据 | ~/train/data/*.json |
| 模型 | ~/models/Qwen2.5-32B-Instruct/ |
| 自编译 NCCL | ~/nccl/build/lib/ |
| 训练输出 | ~/train/output/ |

## 关键技术选择

| 项目 | 选择 | 原因 |
|------|------|------|
| 分布式策略 | FSDP | DeepSpeed JIT 在 ARM64+sm_121 不稳定 |
| 注意力 | SDPA | FlashAttention 不支持 sm_121 |
| CPU Offload | 禁用 | 统一内存架构无意义 |
| 检查点格式 | SHARDED_STATE_DICT | 避免 OOM |

## 故障快速排查

| 问题 | 解决 |
|------|------|
| sm_121 警告 | 忽略，PyTorch 正常工作 |
| NCCL 超时 | 检查 200G 网络: `ping 172.16.100.2` |
| OOM | 减小 batch_size 或 cutoff_len |
| 模型加载慢 | 确认从 NVMe 加载，非 NFS |

## 200G 文件传输

```bash
# Node A → Node B (约 754 MB/s)
rsync -avP --compress ~/models/ gx10@172.16.100.2:~/models/
```
