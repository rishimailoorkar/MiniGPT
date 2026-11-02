"""
MiniGPT Offline Text Generation Engine
Generates text autoregressively from a user prompt using trained local weights.
Supports:
- Greedy Decoding
- Temperature Scaling
- Top-K Sampling
- Top-P (Nucleus) Sampling
- Interactive CLI Chat Mode
"""

import argparse
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
import torch.nn.functional as F
from src.config import MiniGPTConfig
from src.tokenizer import MiniTokenizer
from src.model import MiniGPT


class TextGenerator:
    """
    Offline Text Generation Controller for MiniGPT.
    Loads checkpoint and vocabulary directly from disk.
    """
    def __init__(self, checkpoint_path: str = None, vocab_path: str = None, device: str = "cpu"):
        config_default = MiniGPTConfig()
        self.device = device
        self.checkpoint_path = checkpoint_path or config_default.model_checkpoint_path
        self.vocab_path = vocab_path or config_default.tokenizer_vocab_path
        
        # 1. Load Tokenizer
        if not os.path.exists(self.vocab_path):
            raise FileNotFoundError(f"Vocabulary not found at: {self.vocab_path}. Run training first.")
        self.tokenizer = MiniTokenizer.load(self.vocab_path)
        
        # 2. Load Model Checkpoint
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {self.checkpoint_path}. Run training first.")
            
        print(f"[*] Loading MiniGPT checkpoint from: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        # Handle config format
        if isinstance(checkpoint["config"], dict):
            self.config = MiniGPTConfig.from_dict(checkpoint["config"])
        else:
            self.config = checkpoint["config"]
            
        self.config.vocab_size = self.tokenizer.vocab_size
        self.config.device = self.device
        
        # Instantiate and restore weights
        self.model = MiniGPT(self.config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        
        step = checkpoint.get("step", "N/A")
        val_loss = checkpoint.get("val_loss", 0.0)
        print(f"[+] Model loaded successfully! (Trained to step {step}, Val Loss: {val_loss:.4f})")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 120,
        temperature: float = 0.7,
        top_k: int = 15,
        top_p: float = 0.9,
        greedy: bool = False
    ) -> str:
        """
        Autoregressively generates text continuations for a given prompt string.
        """
        self.model.eval()
        
        # Encode user prompt
        token_ids = self.tokenizer.encode(prompt)
        if len(token_ids) == 0:
            token_ids = [self.tokenizer.bos_id]
            
        x = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        
        start_time = time.time()
        generated_count = 0
        
        for _ in range(max_new_tokens):
            # Crop to block_size context window
            x_cond = x if x.size(1) <= self.config.block_size else x[:, -self.config.block_size:]
            
            # Forward pass to obtain logits for the last position
            logits, _ = self.model(x_cond)
            logits = logits[:, -1, :] # (1, vocab_size)
            
            # 1. Greedy Decoding
            if greedy or temperature <= 0.0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # 2. Temperature Scaling
                logits = logits / temperature
                
                # 3. Top-K Filtering
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                    
                # 4. Top-P (Nucleus) Filtering
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep the first token above threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[0, indices_to_remove] = float("-inf")
                    
                # 5. Softmax to Probability Distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
            # Append next token
            x = torch.cat((x, next_token), dim=1)
            generated_count += 1
            
            # Stop if EOS token generated
            if next_token.item() == self.tokenizer.eos_id:
                break
                
        elapsed = time.time() - start_time
        tokens_per_sec = generated_count / max(elapsed, 1e-5)
        
        # Decode generated IDs back to string
        full_tokens = x[0].tolist()
        output_text = self.tokenizer.decode(full_tokens, skip_special_tokens=True)
        
        return output_text, generated_count, tokens_per_sec


def interactive_chat():
    """Starts an interactive CLI session where users can chat offline with MiniGPT."""
    print("=" * 70)
    print("MiniGPT: Offline Interactive Generation Console")
    print("Type your prompt and press Enter. Type 'exit' or 'quit' to stop.")
    print("=" * 70)
    
    generator = TextGenerator()
    
    while True:
        try:
            user_input = input("\nEnter prompt > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting MiniGPT. Goodbye!")
                break
                
            # Format prompt with conversational markers if not already formatted
            if not user_input.startswith("User:"):
                formatted_prompt = f"User: {user_input}\nAssistant:"
            else:
                formatted_prompt = user_input
                
            print("\nGenerating response...", end="\r", flush=True)
            response, count, speed = generator.generate(
                formatted_prompt,
                max_new_tokens=100,
                temperature=0.7,
                top_k=15,
                top_p=0.9
            )
            
            print("-" * 50)
            print(response.strip())
            print("-" * 50)
            print(f"[{count} tokens generated | Speed: {speed:.1f} tokens/sec | Offline CPU]")
            
        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            break
        except Exception as e:
            print(f"[!] Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniGPT Offline Text Generation")
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt to complete")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=15, help="Top-K filtering cutoff")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P nucleus threshold")
    parser.add_argument("--greedy", action="store_true", help="Use deterministic greedy decoding")
    
    args = parser.parse_args()
    
    if args.prompt is not None:
        generator = TextGenerator()
        text, count, speed = generator.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            greedy=args.greedy
        )
        print("\n" + "=" * 60)
        print("MiniGPT Generated Output:")
        print("=" * 60)
        print(text.strip())
        print("=" * 60)
        print(f"Generated {count} tokens in {count/speed:.2f}s ({speed:.1f} tokens/sec offline)")
    else:
        # Default to interactive chat
        interactive_chat()
