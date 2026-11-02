"""
MiniGPT Architecture: Decoder-Only Transformer from Scratch
Implements:
1. Learned Token & Positional Embeddings
2. Multi-Head Causal Self-Attention with Scaled Dot-Product
3. Transformer Blocks with Pre-LayerNorm and GELU Feed-Forward Networks
4. Language Modeling Head with optional Weight Tying
"""

import math
import os
import sys
from typing import Tuple, Optional

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
import torch.nn.functional as F
from src.config import MiniGPTConfig


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal (Masked) Self-Attention.
    Allows tokens to attend only to current and preceding tokens,
    preventing any future token information leakage.
    """
    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        
        # Combined Linear projection for Query, Key, and Value
        # Projecting from d_model to 3 * d_model in a single matrix multiplication
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        
        # Output projection back to residual stream
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Lower-triangular causal mask buffer (registered as persistent buffer, not a parameter)
        # 1 means attend, 0 means mask out
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("bias", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size() # Batch size, Sequence length, Embedding dimension (d_model)
        
        # Compute Query, Key, Value projections in a single step
        qkv = self.c_attn(x) # (B, T, 3 * C)
        q, k, v = qkv.split(self.d_model, dim=2) # Each is (B, T, C)
        
        # Reshape for multi-head attention: (B, n_heads, T, head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention: (Q @ K^T) / sqrt(head_dim)
        # att shape: (B, n_heads, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # Apply causal mask: replace positions where bias == 0 with -infinity
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        
        # Softmax normalizes weights along the key dimension
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Multiply attention weights with Value vectors: (B, n_heads, T, T) @ (B, n_heads, T, head_dim)
        y = att @ v # (B, n_heads, T, head_dim)
        
        # Concatenate heads back into (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Final linear projection into residual stream
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """
    Position-Wise Feed-Forward Network.
    Expands embedding dimension 4x with GELU activation, then projects back.
    """
    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, 4 * config.d_model, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.d_model, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    A single Transformer Decoder Block.
    Uses Pre-Layer Normalization (Pre-LN) for superior gradient stability.
    Output = x + Attention(LN1(x))
    Output = Output + MLP(LN2(Output))
    """
    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection around self-attention
        x = x + self.attn(self.ln_1(x))
        # Residual connection around feed-forward network
        x = x + self.mlp(self.ln_2(x))
        return x


class MiniGPT(nn.Module):
    """
    The Full Decoder-Only MiniGPT Language Model.
    Architecture:
    - Token Embedding Table (WTE)
    - Positional Embedding Table (WPE)
    - N Stacked Transformer Blocks
    - Final Layer Normalization (LN_f)
    - Language Model Head (LM_Head)
    """
    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model),
            wpe = nn.Embedding(config.block_size, config.d_model),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.d_model),
        ))
        
        # Linear projection from d_model back to vocabulary space
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying: share parameters between token embedding and LM head
        # (Proven to improve performance and reduce parameter redundancy)
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize all model weights according to GPT-2 standards
        self.apply(self._init_weights)
        
        # Apply special scaled initialization to residual projections
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Calculates total trainable parameters.
        non_embedding=True subtracts positional embeddings.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass:
        Args:
            idx: Tensor of shape (B, T) with token IDs.
            targets: Optional Tensor of shape (B, T) with ground-truth next token IDs.
        Returns:
            logits: Tensor of shape (B, T, vocab_size)
            loss: Optional cross-entropy scalar loss (if targets is provided)
        """
        device = idx.device
        b, t = idx.size()
        
        assert t <= self.config.block_size, f"Cannot forward sequence length {t}, block_size is {self.config.block_size}"
        
        # Generate position indices: [0, 1, 2, ..., t-1]
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # (1, t)
        
        # 1. Forward Token & Positional Embeddings
        tok_emb = self.transformer.wte(idx) # (B, T, d_model)
        pos_emb = self.transformer.wpe(pos) # (1, T, d_model)
        x = self.transformer.drop(tok_emb + pos_emb) # (B, T, d_model)
        
        # 2. Forward through stacked Transformer blocks
        for block in self.transformer.h:
            x = block(x)
            
        # 3. Final Layer Normalization
        x = self.transformer.ln_f(x)
        
        # 4. Compute Language Model Logits
        if targets is not None:
            # Training mode: compute logits for all positions and calculate loss
            logits = self.lm_head(x) # (B, T, vocab_size)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        else:
            # Inference mode: only compute logits for the very last token position to save compute
            logits = self.lm_head(x[:, [-1], :]) # (B, 1, vocab_size)
            loss = None
            
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
        eos_id: int = None
    ) -> torch.Tensor:
        """
        Autoregressively generates new tokens given a conditioning prefix sequence 'idx'.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop current sequence if it exceeds block_size context window
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward pass to get logits for the last token
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] # (B, vocab_size)
            
            # Apply temperature scaling
            if temperature > 0:
                logits = logits / temperature
            else:
                # Greedy decoding: pick highest logit directly
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                idx = torch.cat((idx, next_token), dim=1)
                if eos_id is not None and (next_token == eos_id).all():
                    break
                continue
                
            # Optional Top-K filtering
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
                
            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample next token from the multinomial probability distribution
            next_token = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            # Append sampled token to running sequence
            idx = torch.cat((idx, next_token), dim=1)
            
            # Terminate if end-of-sequence token is generated
            if eos_id is not None and (next_token == eos_id).all():
                break
                
        return idx


# -------------------------------------------------------------
# Standalone Architecture Unit Test
# -------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Testing MiniGPT Architecture from Scratch")
    print("=" * 60)
    
    # Configure model
    config = MiniGPTConfig(
        vocab_size=74,    # Matches our local vocabulary
        block_size=128,   # Context window
        d_model=192,      # Hidden dimension
        n_heads=6,        # Number of heads (192 / 6 = 32 per head)
        n_layers=6,       # Stacked layers
        dropout=0.1
    )
    
    model = MiniGPT(config)
    total_params = model.get_num_params(non_embedding=False)
    non_emb_params = model.get_num_params(non_embedding=True)
    
    print(f"Total Parameters:          {total_params:,} (~{total_params / 1e6:.2f} Million)")
    print(f"Non-Embedding Parameters:  {non_emb_params:,}")
    print(f"Device:                    CPU (Optimized for Intel Core i5-14500)")
    
    # Test Forward Pass with Loss Calculation
    B, T = 4, 32 # Micro-batch: 4 sequences, length 32
    dummy_x = torch.randint(0, config.vocab_size, (B, T))
    dummy_y = torch.randint(0, config.vocab_size, (B, T))
    
    logits, loss = model(dummy_x, dummy_y)
    print(f"\nForward Pass Verification:")
    print(f"Input shape:  {dummy_x.shape}")
    print(f"Logits shape: {logits.shape} (Batch, Time, Vocab)")
    print(f"Loss value:   {loss.item():.4f}")
    
    assert logits.shape == (B, T, config.vocab_size), "Logits shape mismatch!"
    # Theoretical initial loss for uniform random distribution is -ln(1 / 74) = 4.304
    expected_initial_loss = -math.log(1.0 / config.vocab_size)
    print(f"Expected Initial Random Loss: -ln(1/{config.vocab_size}) = {expected_initial_loss:.4f}")
    assert abs(loss.item() - expected_initial_loss) < 1.0, "Initial loss deviates unusually from expected uniform baseline!"
    print("[+] Test 1 Passed: Forward pass & initial Cross-Entropy loss verified!")
    
    # Test Autoregressive Generation
    prefix = torch.tensor([[10, 20, 30]]) # Prefix prompt of 3 tokens
    generated = model.generate(prefix, max_new_tokens=15, temperature=0.8, top_k=10)
    print(f"\nAutoregressive Sampling Verification:")
    print(f"Prefix tokens:    {prefix.tolist()[0]}")
    print(f"Generated tokens: {generated.tolist()[0]}")
    print(f"Total length:     {generated.shape[1]} (3 prefix + 15 generated)")
    assert generated.shape == (1, 18), "Generated token sequence length mismatch!"
    print("[+] Test 2 Passed: Autoregressive text sampling verified!")
    
    print("\n[+] All MiniGPT Architecture Unit Tests PASSED successfully!")
