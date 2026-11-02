"""
MiniGPT Configuration Module
Contains all hyperparameters and file paths for offline training and inference.
"""

from dataclasses import dataclass, asdict
import os

@dataclass
class MiniGPTConfig:
    # --- Model Architecture ---
    vocab_size: int = 74         # Size of vocabulary (dynamically set by tokenizer)
    block_size: int = 64         # Sequence context window (T tokens)
    d_model: int = 192           # Embedding / hidden dimension size
    n_heads: int = 6             # Number of attention heads (192 / 6 = 32 per head)
    n_layers: int = 6            # Number of stacked Transformer decoder blocks
    dropout: float = 0.1         # Dropout probability for regularization
    bias: bool = False           # True: use bias in Linear layers; False: cleaner & faster
    
    # --- Training Hyperparameters (Optimized for Intel CPU) ---
    batch_size: int = 16         # Micro-batch size per optimization step
    learning_rate: float = 8e-4  # Peak learning rate with AdamW
    min_learning_rate: float = 8e-5 # Minimum learning rate for cosine decay
    max_iters: int = 1000        # Total training iterations
    warmup_iters: int = 100      # Linear warmup steps
    eval_interval: int = 100     # Evaluate validation loss every N steps
    eval_iters: int = 10         # Number of batches to estimate validation loss
    weight_decay: float = 1e-1   # L2 weight decay for AdamW
    grad_clip: float = 1.0       # Gradient norm clipping threshold
    
    # --- System & Offline Storage Paths ---
    device: str = "cpu"          # Target device: "cpu"
    data_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    checkpoint_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
    plots_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")
    
    # Checkpoint filenames
    model_checkpoint_path: str = os.path.join(checkpoint_dir, "minigpt_model.pt")
    tokenizer_vocab_path: str = os.path.join(checkpoint_dir, "vocab.json")
    training_metrics_path: str = os.path.join(checkpoint_dir, "metrics.json")
    
    def to_dict(self):
        return asdict(self)
        
    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)
