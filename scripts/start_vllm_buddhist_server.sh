#!/bin/bash
# vLLM 双机 PP=2 API Server - 佛经微调模型 (基于 DGX Spark 部署指南)
set -e

export RAY_ADDRESS=172.16.100.1:6379
export VLLM_HOST_IP=172.16.100.1
export VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE=shm
export FLASHINFER_DISABLE_VERSION_CHECK=1

source ~/miniforge3/etc/profile.d/conda.sh
conda activate buddhist-train

MODEL_PATH="${1:-/home/gx10/models/Qwen2.5-32B-Buddhist-Merged}"
MODEL_NAME="${2:-buddhist-merged}"

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${MODEL_NAME}" \
    --pipeline-parallel-size 2 \
    --distributed-executor-backend ray \
    --enforce-eager \
    --gpu-memory-utilization 0.80 \
    --max-model-len 4096 \
    --host 0.0.0.0 \
    --port 8000 \
    2>&1 | tee ~/vllm_buddhist_server.log
