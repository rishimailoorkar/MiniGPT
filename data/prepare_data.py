"""
Data Preparation & Splitting Script for MiniGPT
Reads the local raw corpus, computes descriptive statistics,
and partitions the dataset into train.txt and val.txt strictly offline.
"""

import os
import sys

def prepare_dataset(raw_path: str, data_dir: str, train_ratio: float = 0.9):
    print("=" * 60)
    print("MiniGPT: Offline Data Preparation")
    print("=" * 60)
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw corpus not found at: {raw_path}")
    
    with open(raw_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Basic cleaning
    text = text.strip()
    total_chars = len(text)
    unique_chars = sorted(list(set(text)))
    vocab_size = len(unique_chars)
    
    print(f"[*] Raw corpus loaded from: {raw_path}")
    print(f"[*] Total characters: {total_chars:,}")
    print(f"[*] Total unique characters (Vocab size): {vocab_size}")
    print(f"[*] Unique character set preview: {''.join(unique_chars[:40])}...")
    
    # Train / Val partition
    split_idx = int(total_chars * train_ratio)
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    train_path = os.path.join(data_dir, "train.txt")
    val_path = os.path.join(data_dir, "val.txt")
    
    with open(train_path, "w", encoding="utf-8") as f:
        f.write(train_text)
        
    with open(val_path, "w", encoding="utf-8") as f:
        f.write(val_text)
        
    print(f"[*] Training dataset saved to: {train_path} ({len(train_text):,} chars, {train_ratio*100:.0f}%)")
    print(f"[*] Validation dataset saved to: {val_path} ({len(val_text):,} chars, {(1-train_ratio)*100:.0f}%)")
    print("[+] Data preparation complete!\n")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_corpus = os.path.join(current_dir, "raw", "corpus.txt")
    prepare_dataset(raw_corpus, current_dir)
