"""
MiniGPT Offline Training Pipeline
Trains the Decoder-Only Transformer using Cross-Entropy next-token prediction.
Logs loss metrics, computes perplexity, and saves checkpoints to local disk.
"""

import json
import math
import os
import sys
import time

# Ensure project root is in sys.path
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
import torch.nn as nn
from src.config import MiniGPTConfig
from src.tokenizer import MiniTokenizer
from src.dataset import TextDataset, get_random_batch
from src.model import MiniGPT


def configure_optimizers(model: MiniGPT, weight_decay: float, learning_rate: float, betas=(0.9, 0.95)):
    """
    Separates parameters into decay and no-decay groups.
    2D weight matrices (Linear weights) get weight decay.
    1D parameters (biases, LayerNorm scale/shift) do not receive weight decay.
    Handles weight tying cleanly.
    """
    decay = []
    no_decay = []
    
    for pn, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)

    optim_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    return optimizer


def get_lr(it: int, config: MiniGPTConfig) -> float:
    """
    Cosine learning rate schedule with linear warmup.
    """
    # 1) Linear warmup for warmup_iters steps
    if it < config.warmup_iters:
        return config.learning_rate * (it + 1) / (config.warmup_iters + 1)
    # 2) If it > max_iters, return min learning rate
    if it > config.max_iters:
        return config.min_learning_rate
    # 3) In between, use cosine decay down to min_learning_rate
    decay_ratio = (it - config.warmup_iters) / (config.max_iters - config.warmup_iters)
    assert 0.0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_learning_rate + coeff * (config.learning_rate - config.min_learning_rate)


@torch.no_grad()
def estimate_loss(model: MiniGPT, train_data: torch.Tensor, val_data: torch.Tensor, config: MiniGPTConfig):
    """
    Computes averaged train and validation loss over micro-batches.
    """
    out = {}
    model.eval()
    for split_name, data_tensor in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            x, y = get_random_batch(data_tensor, config.block_size, config.batch_size, device=config.device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split_name] = losses.mean().item()
    model.train()
    return out


def train():
    print("=" * 70, flush=True)
    print("MiniGPT: Offline Transformer Training Pipeline", flush=True)
    print("=" * 70, flush=True)
    
    # 1. Load configuration and tokenizer
    config = MiniGPTConfig()
    
    # Utilize Intel Core CPU threads
    torch.set_num_threads(8)
    
    # Load vocabulary
    if not os.path.exists(config.tokenizer_vocab_path):
        raise FileNotFoundError(f"Vocabulary file missing at: {config.tokenizer_vocab_path}. Run step 2 first.")
    tokenizer = MiniTokenizer.load(config.tokenizer_vocab_path)
    config.vocab_size = tokenizer.vocab_size
    
    # 2. Load dataset splits
    train_path = os.path.join(config.data_dir, "train.txt")
    val_path = os.path.join(config.data_dir, "val.txt")
    
    train_dataset = TextDataset(train_path, tokenizer, block_size=config.block_size)
    val_dataset = TextDataset(val_path, tokenizer, block_size=config.block_size)
    
    # 3. Instantiate model
    print(f"\n[*] Initializing MiniGPT model architecture on {config.device.upper()}...", flush=True)
    model = MiniGPT(config)
    model.to(config.device)
    total_params = model.get_num_params(non_embedding=False)
    print(f"[*] Total Trainable Parameters: {total_params:,} (~{total_params / 1e6:.2f}M)", flush=True)
    
    # 4. Setup optimizer
    optimizer = configure_optimizers(model, config.weight_decay, config.learning_rate)
    
    # Metrics tracking dictionary
    metrics = {
        "step": [],
        "train_loss": [],
        "val_loss": [],
        "val_perplexity": [],
        "lr": [],
        "elapsed_time": []
    }
    
    best_val_loss = float("inf")
    start_time = time.time()
    
    print("\n" + "-" * 70, flush=True)
    print(f"{'Step':<8}{'Train Loss':<14}{'Val Loss':<14}{'Val Perplexity':<18}{'LR':<12}{'Time (s)':<10}", flush=True)
    print("-" * 70, flush=True)
    
    # Training Loop
    for step in range(config.max_iters + 1):
        # Evaluation step
        if step % config.eval_interval == 0 or step == config.max_iters:
            losses = estimate_loss(model, train_dataset.data, val_dataset.data, config)
            val_ppl = math.exp(losses["val"]) if losses["val"] < 20 else float("inf")
            current_lr = get_lr(step, config)
            elapsed = time.time() - start_time
            
            print(f"{step:<8}{losses['train']:<14.4f}{losses['val']:<14.4f}{val_ppl:<18.2f}{current_lr:<12.2e}{elapsed:<10.1f}", flush=True)
            
            metrics["step"].append(step)
            metrics["train_loss"].append(losses["train"])
            metrics["val_loss"].append(losses["val"])
            metrics["val_perplexity"].append(val_ppl)
            metrics["lr"].append(current_lr)
            metrics["elapsed_time"].append(round(elapsed, 2))
            
            # Save checkpoint if validation loss improved or at final iteration
            if losses["val"] < best_val_loss or step == config.max_iters:
                best_val_loss = losses["val"]
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config.to_dict(),
                    "step": step,
                    "val_loss": best_val_loss,
                    "val_perplexity": val_ppl
                }
                torch.save(checkpoint, config.model_checkpoint_path)
                
            # Quick sample preview at key milestones
            if step in [0, 200, 500, 800, config.max_iters]:
                prompt = "User: What is AI?\nAssistant:"
                prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=config.device)
                sample_ids = model.generate(prompt_ids, max_new_tokens=40, temperature=0.7, top_k=10)
                decoded_sample = tokenizer.decode(sample_ids[0].tolist())
                print(f"  >>> Sample Generation [Step {step}]:\n  {decoded_sample.strip()}\n", flush=True)
                
        if step == config.max_iters:
            break
            
        # Adjust learning rate for current step
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
            
        # Micro-batch forward and backward pass
        x, y = get_random_batch(train_dataset.data, config.block_size, config.batch_size, device=config.device)
        logits, loss = model(x, y)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient clipping prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        
    total_training_time = time.time() - start_time
    print("=" * 70, flush=True)
    print(f"[+] Training completed in {total_training_time:.2f} seconds ({total_training_time / 60:.2f} minutes)!", flush=True)
    print(f"[+] Best Validation Loss: {best_val_loss:.4f} (Perplexity: {math.exp(best_val_loss):.2f})", flush=True)
    print(f"[+] Checkpoint saved to: {config.model_checkpoint_path}", flush=True)
    
    # Save training metrics to JSON for evaluation and plotting
    with open(config.training_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[+] Metrics saved to: {config.training_metrics_path}", flush=True)


if __name__ == "__main__":
    train()
