#!/bin/bash
NODE_RANK=${1:-0}

echo "============================================================"
echo "善知识 v5 32B LoRA 微调启动"
echo "============================================================"
echo "Node Rank: $NODE_RANK"
echo ""

# 加载环境
source ~/train/scripts/env_setup.sh
source ~/miniforge3/etc/profile.d/conda.sh
conda activate buddhist-train

cd ~/train

# 选择配置文件
if [ "$NODE_RANK" = "0" ]; then
    CONFIG=accelerate_config_node0.yaml
    echo "Using config: $CONFIG (Head Node)"
else
    CONFIG=accelerate_config_node1.yaml
    echo "Using config: $CONFIG (Worker Node)"
fi

# 检查配置文件是否存在
if [ ! -f "$CONFIG" ]; then
    echo "Error: $CONFIG not found"
    exit 1
fi

# 检查训练数据
if [ ! -f "data/buddhist_train_alpaca.json" ]; then
    echo "Error: Training data not found"
    echo "Please run: bash ~/v5_scripts/prepare_v5_data.sh"
    exit 1
fi

echo "Starting training..."
echo "============================================================"

# 启动训练
accelerate launch --config_file $CONFIG \
    --main_process_ip 172.16.100.1 \
    --main_process_port 29500 \
    --machine_rank $NODE_RANK \
    --num_machines 2 \
    --num_processes 2 \
    -m llamafactory.train ~/v5_scripts/train_config_v5.yaml
