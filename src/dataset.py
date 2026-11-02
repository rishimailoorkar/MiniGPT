"""
PyTorch Dataset and DataLoader Module for MiniGPT
Implements next-token prediction sequence slicing (x -> y shifted by 1).
"""

import os
import sys

# Ensure project root is in sys.path for both direct execution and module imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Ensure UTF-8 console output for Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from src.tokenizer import MiniTokenizer

class TextDataset(Dataset):
    """
    PyTorch Dataset for autoregressive language modeling.
    Given a 1D tensor of token IDs, extracts input-target pairs:
    x = tokens[i : i + block_size]
    y = tokens[i + 1 : i + block_size + 1]
    
    Here, y is the identical sequence shifted by 1 token to the right.
    """
    def __init__(self, text_path: str, tokenizer: MiniTokenizer, block_size: int = 128):
        self.block_size = block_size
        self.tokenizer = tokenizer
        
        if not os.path.exists(text_path):
            raise FileNotFoundError(f"Dataset file not found at: {text_path}")
            
        with open(text_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
            
        # Encode raw text into integer token IDs
        token_ids = self.tokenizer.encode(raw_text)
        self.data = torch.tensor(token_ids, dtype=torch.long)
        
        print(f"[TextDataset] Loaded '{os.path.basename(text_path)}' -> {len(self.data):,} tokens (block_size: {block_size}).")
        
    def __len__(self) -> int:
        # Number of full block_size chunks available
        if len(self.data) <= self.block_size:
            return 1
        return len(self.data) - self.block_size
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Handle small dataset edge case with padding
        if len(self.data) <= self.block_size:
            chunk = self.data
            pad_len = (self.block_size + 1) - len(chunk)
            padded = torch.cat([chunk, torch.full((pad_len,), self.tokenizer.pad_id, dtype=torch.long)])
            x = padded[:self.block_size]
            y = padded[1:self.block_size + 1]
            return x, y
            
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def get_random_batch(
    data_tensor: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    High-performance random batch sampler for CPU/GPU training.
    Selects 'batch_size' random starting indices from the data tensor.
    """
    max_idx = len(data_tensor) - block_size - 1
    if max_idx <= 0:
        raise ValueError("Data tensor is too short for the specified block_size.")
        
    indices = torch.randint(0, max_idx, (batch_size,))
    x = torch.stack([data_tensor[i : i + block_size] for i in indices])
    y = torch.stack([data_tensor[i + 1 : i + block_size + 1] for i in indices])
    
    return x.to(device), y.to(device)


# -------------------------------------------------------------
# Standalone Unit Test
# -------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Testing TextDataset and Next-Token Prediction Slicing")
    print("=" * 60)
    
    vocab_path = "checkpoints/vocab.json"
    train_path = "data/train.txt"
    
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Run step 2 first: {vocab_path} missing.")
        
    tokenizer = MiniTokenizer.load(vocab_path)
    dataset = TextDataset(train_path, tokenizer=tokenizer, block_size=16)
    
    print(f"Total samples in dataset: {len(dataset)}")
    x, y = dataset[0]
    
    print("\nSample 0 Inspection:")
    print(f"x shape: {x.shape}, y shape: {y.shape}")
    print(f"x token IDs: {x.tolist()}")
    print(f"y token IDs: {y.tolist()}")
    
    # Check that y is indeed x shifted by 1
    assert torch.equal(x[1:], y[:-1]), "Target y is not properly shifted by 1 relative to x!"
    print("[+] Test 1 Passed: Perfect causal 1-step token alignment (y[t] == x[t+1])!")
    
    # Check decoding alignment
    print("\nDecoded alignment check:")
    for t in range(5):
        context = tokenizer.decode(x[:t+1].tolist())
        target = tokenizer.decode([y[t].item()])
        print(f"  Step {t+1}: Context '{context}' -> Target next token: '{target}'")
        
    # Test DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch_x, batch_y = next(iter(loader))
    print(f"\nDataLoader Batch x shape: {batch_x.shape} (Batch_size, Block_size)")
    print(f"DataLoader Batch y shape: {batch_y.shape}")
    assert batch_x.shape == (4, 16)
    print("[+] Test 2 Passed: PyTorch DataLoader batching verified!")
    
    # Test Random Batch Sampler
    rx, ry = get_random_batch(dataset.data, block_size=16, batch_size=8)
    print(f"Random Batch Sampler rx shape: {rx.shape}, ry shape: {ry.shape}")
    assert rx.shape == (8, 16)
    print("[+] Test 3 Passed: Fast random batch sampler verified!")
    
    print("\n[+] All Dataset Unit Tests PASSED successfully!")
