# 善知识 v5 32B LoRA 微调

**版本**: v5.0  
**目标模型**: Qwen2.5-32B-Instruct  
**训练数据**: 24,553 条 (v4.1 的 6.2 倍)  
**预计时长**: ~19 小时 (双节点 DDP)

---

## 训练配置 (方案 B)

| 参数 | 值 | 说明 |
|------|-----|------|
| LoRA Rank | 16 | v4.1 为 8，增加模型表达能力 |
| LoRA Alpha | 32 | Alpha = 2×Rank |
| Learning Rate | 2e-5 | v4.1 为 4e-5，降低防止过拟合 |
| Epochs | 2 | v4.1 为 3，数据量大可减少 |
| Batch Size | 1 (per device) | 保持 |
| Gradient Accumulation | 8 | 保持 |
| Effective Batch | 16 (2 nodes × 1 × 8) | 保持 |
| Max Length | 2048 | 保持 |
| Trainable Params | ~268M (0.83%) | v4.1 为 134M |

---

## 数据对比

| 维度 | v4.1 | v5 | 变化 |
|------|------|-----|------|
| 训练数据 | 3,947 | 24,553 | +622% |
| P0 核心数据 | - | 4,743 | 新增 |
| P1 扩展数据 | - | 6,459 | 新增 |
| P2 补充数据 | - | 13,402 | 新增 |

---

## 执行步骤

### 1. 准备数据 (两节点都需要)

在 Node A (Rank 0) 上执行:

```bash
cd ~/train/data
git clone git@github.com:guyiicn/buddhist-llm-finetune.git
```

数据转换脚本会自动:
1. 从 GitHub 拉取数据
2. 转换 jsonl → Alpaca JSON
3. 按 98:2 切分 train/val

### 2. 同步数据到 Node B

```bash
# Node A 执行
rsync -avP ~/train/data/buddhist_*.json ~/train/data/dataset_info.json \
    gx10@172.16.100.2:~/train/data/
```

### 3. 复制配置文件

```bash
# Node A
cd ~/v5_scripts
cp train_config_v5.yaml ~/train/
cp accelerate_config_node0.yaml ~/train/

# Node B (从 Node A 复制)
scp train_config_v5.yaml gx10@172.16.100.2:~/train/
scp accelerate_config_node1.yaml gx10@172.16.100.2:~/train/
```

### 4. 启动训练

**Node A (Rank 0)** - 主节点:
```bash
bash ~/v5_scripts/launch_v5_train.sh 0
```

**Node B (Rank 1)** - 工作节点 (同时执行):
```bash
bash ~/v5_scripts/launch_v5_train.sh 1
```

---

## 训练监控

### GPU 使用
```bash
# 两节点分别执行
nvidia-smi -l 1
```

### 训练日志
```bash
# Node A
tail -f ~/train/output/v5_32b_lora/trainer_log.jsonl
```

### Loss 曲线
```bash
# Node A
tail -f ~/train/output/v5_32b_lora/trainer_state.json
```

---

## 预期结果

| 指标 | 值 |
|------|-----|
| 训练时长 | ~19 小时 |
| 总 Steps | ~3,700 |
| Train Loss | ~1.2-1.4 |
| Eval Loss | ~1.3-1.5 |

---

## 文件结构

```
~/v5_scripts/
├── convert_v5_to_alpaca.py    # 数据转换脚本
├── train_config_v5.yaml         # 训练配置
├── launch_v5_train.sh           # 启动脚本
├── prepare_v5_data.sh           # 数据准备脚本
└── README.md                    # 本文件
```

---

## 与 v4.1 的区别

| 项目 | v4.1 | v5 |
|------|------|-----|
| 数据量 | 3,947 | 24,553 (+622%) |
| LoRA Rank | 8 | 16 (+100%) |
| Learning Rate | 4e-5 | 2e-5 (-50%) |
| Epochs | 3 | 2 (-33%) |
| 预估时长 | 5.8h | 19h (+228%) |
| Trainable Params | 134M (0.41%) | 268M (0.83%) |

v5 数据量大幅增加，虽然训练时长增加，但模型知识覆盖面更广，质量应该有显著提升。
