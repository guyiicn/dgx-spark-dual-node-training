#!/bin/bash

# 善知识 v5 32B 训练数据准备脚本
# 从 GitHub 拉取 v5 数据并转换为 Alpaca 格式

set -e

echo "============================================================"
echo "善知识 v5 数据准备 (24,553 条)"
echo "============================================================"

# 加载环境
source ~/train/scripts/env_setup.sh 2>/dev/null || true

# 创建数据目录
mkdir -p ~/train/data

# 进入数据目录
cd ~/train/data

# 检查是否已克隆
if [ ! -d "buddhist-llm-finetune" ]; then
    echo "Cloning repository from GitHub..."
    git clone git@github.com:guyiicn/buddhist-llm-finetune.git
else
    echo "Repository already exists, pulling latest..."
    cd buddhist-llm-finetune
    git pull origin main
    cd ..
fi

# 检查数据文件
TRAIN_FILE="buddhist-llm-finetune/32b/train_all.jsonl"
if [ ! -f "$TRAIN_FILE" ]; then
    echo "Error: $TRAIN_FILE not found"
    exit 1
fi

echo "Found data file: $TRAIN_FILE"
wc -l "$TRAIN_FILE"

# 转换数据格式
echo "Converting to Alpaca format..."
python3 ~/v5_scripts/convert_v5_to_alpaca.py

# 验证输出
if [ -f "buddhist_train_alpaca.json" ]; then
    echo "✓ Training data ready: buddhist_train_alpaca.json"
    ls -lh buddhist_train_alpaca.json buddhist_val_alpaca.json
else
    echo "✗ Conversion failed"
    exit 1
fi

# 更新 dataset_info.json
cat > dataset_info.json << 'JSON'
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
JSON

echo "✓ dataset_info.json updated"

echo ""
echo "============================================================"
echo "数据准备完成!"
echo "============================================================"
