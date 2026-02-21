#!/usr/bin/env python3
"""
Convert v5 training data from GitHub to Alpaca format for LLaMA-Factory
GitHub: git@github.com:guyiicn/buddhist-llm-finetune.git
"""

import json
from pathlib import Path
import sys

# Input paths
DATA_DIR = Path("/home/gx10/train/data/buddhist-llm-finetune/32b")
TRAIN_FILE = DATA_DIR / "train_all.jsonl"

# Output paths
OUTPUT_DIR = Path("/home/gx10/train/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_OUTPUT = OUTPUT_DIR / "buddhist_train_alpaca.json"
VAL_OUTPUT = OUTPUT_DIR / "buddhist_val_alpaca.json"

VAL_RATIO = 0.02  # 2% validation split

def convert_jsonl_to_alpaca(input_file: Path, train_file: Path, val_file: Path, val_ratio: float = 0.02):
    """Convert jsonl with instruction/output to Alpaca JSON format"""
    
    data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                if "instruction" in item and "output" in item:
                    # Convert to Alpaca format
                    alpaca_item = {
                        "instruction": item["instruction"],
                        "input": "",
                        "output": item["output"]
                    }
                    data.append(alpaca_item)
            except Exception as e:
                print(f"Error parsing line: {e}")
                continue
    
    print(f"Loaded {len(data)} items from {input_file}")
    
    # Shuffle and split
    import random
    random.seed(42)
    random.shuffle(data)
    
    split_idx = int(len(data) * (1 - val_ratio))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Write train
    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    # Write val
    with open(val_file, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved train to {train_file}")
    print(f"Saved val to {val_file}")
    
    return len(train_data), len(val_data)

if __name__ == "__main__":
    if not TRAIN_FILE.exists():
        print(f"Error: {TRAIN_FILE} not found")
        print("Please clone the repo first:")
        print("  cd /home/gx10/train/data")
        print("  git clone git@github.com:guyiicn/buddhist-llm-finetune.git")
        sys.exit(1)
    
    train_count, val_count = convert_jsonl_to_alpaca(TRAIN_FILE, TRAIN_OUTPUT, VAL_OUTPUT, VAL_RATIO)
    
    print(f"\nConversion complete!")
    print(f"  Total: {train_count + val_count:,}")
    print(f"  Train: {train_count:,}")
    print(f"  Val: {val_count:,}")
