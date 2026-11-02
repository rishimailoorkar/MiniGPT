"""
Offline Tokenizer Module for MiniGPT
Built from scratch without external dependencies.
Converts text strings into discrete integer token sequences and back.
"""

import json
import os
import sys
from typing import List, Dict, Optional

# Ensure UTF-8 output encoding for Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

class MiniTokenizer:
    """
    A robust, self-contained character-level tokenizer with special token support.
    Ideal for offline small language models (< 5M parameters):
    - 100% deterministic & lossless round-trip decoding.
    - Zero external library dependencies (no HuggingFace / cloud).
    - Can save to and load from local JSON files.
    """
    
    # Standard special tokens
    PAD_TOKEN = "<pad>"  # Padding token (ID: 0)
    UNK_TOKEN = "<unk>"  # Unknown character fallback (ID: 1)
    BOS_TOKEN = "<bos>"  # Beginning of sequence (ID: 2)
    EOS_TOKEN = "<eos>"  # End of sequence (ID: 3)
    
    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        if vocab is not None:
            self.char_to_id = vocab
            self.id_to_char = {idx: ch for ch, idx in self.char_to_id.items()}
        else:
            self.char_to_id: Dict[str, int] = {}
            self.id_to_char: Dict[int, str] = {}
            
    @property
    def vocab_size(self) -> int:
        """Returns the total number of unique tokens in the vocabulary."""
        return len(self.char_to_id)
        
    @property
    def pad_id(self) -> int:
        return self.char_to_id.get(self.PAD_TOKEN, 0)

    @property
    def unk_id(self) -> int:
        return self.char_to_id.get(self.UNK_TOKEN, 1)

    @property
    def bos_id(self) -> int:
        return self.char_to_id.get(self.BOS_TOKEN, 2)

    @property
    def eos_id(self) -> int:
        return self.char_to_id.get(self.EOS_TOKEN, 3)

    def train_from_text(self, text: str):
        """
        Builds vocabulary from a raw training text corpus.
        Assigns IDs 0-3 to special tokens, followed by sorted unique characters.
        """
        special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        
        # Extract unique characters from the corpus
        unique_chars = sorted(list(set(text)))
        
        # Build mappings
        self.char_to_id = {}
        for idx, token in enumerate(special_tokens):
            self.char_to_id[token] = idx
            
        offset = len(special_tokens)
        for idx, char in enumerate(unique_chars):
            self.char_to_id[char] = offset + idx
            
        self.id_to_char = {idx: ch for ch, idx in self.char_to_id.items()}
        print(f"[MiniTokenizer] Vocabulary successfully built. Size: {self.vocab_size} tokens.")

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Converts a string of text into a list of integer token IDs.
        If a character is not in vocabulary, falls back to <unk>.
        """
        unk = self.unk_id
        token_ids = [self.char_to_id.get(char, unk) for char in text]
        
        if add_special_tokens:
            token_ids = [self.bos_id] + token_ids + [self.eos_id]
            
        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Converts a list of integer token IDs back into human-readable text.
        """
        special_ids = {self.pad_id, self.unk_id, self.bos_id, self.eos_id}
        chars = []
        for idx in token_ids:
            if skip_special_tokens and idx in special_ids:
                continue
            chars.append(self.id_to_char.get(idx, ""))
        return "".join(chars)

    def save(self, filepath: str):
        """Saves the vocabulary mapping to a local JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.char_to_id, f, ensure_ascii=False, indent=2)
        print(f"[MiniTokenizer] Saved vocabulary ({self.vocab_size} tokens) to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "MiniTokenizer":
        """Loads vocabulary from a local JSON file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Tokenizer vocabulary file not found at: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        tokenizer = cls(vocab=vocab)
        print(f"[MiniTokenizer] Loaded vocabulary ({tokenizer.vocab_size} tokens) from: {filepath}")
        return tokenizer


# -------------------------------------------------------------
# Standalone Unit Test
# -------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Testing MiniTokenizer from Scratch")
    print("=" * 60)
    
    sample_corpus = "Hello, MiniGPT! Machine Learning runs completely offline on Windows."
    tokenizer = MiniTokenizer()
    tokenizer.train_from_text(sample_corpus)
    
    # Test Encoding & Decoding
    test_phrase = "Hello, MiniGPT!"
    encoded = tokenizer.encode(test_phrase, add_special_tokens=True)
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    
    print(f"Original text: '{test_phrase}'")
    print(f"Encoded IDs:   {encoded}")
    print(f"Decoded text:  '{decoded}'")
    assert test_phrase == decoded, "Decoded text does not match original!"
    print("[+] Test 1 Passed: Perfect reconstruction!")
    
    # Test Unknown Character Fallback
    unknown_phrase = "Hello, MiniGPT! 🚀"  # Emoji not in training corpus
    encoded_unk = tokenizer.encode(unknown_phrase)
    print(f"With unknown character '{unknown_phrase}' -> Encoded: {encoded_unk}")
    assert tokenizer.unk_id in encoded_unk, "<unk> token was not assigned to unknown character!"
    print("[+] Test 2 Passed: <unk> fallback works correctly!")
    
    # Test Save & Load
    test_vocab_path = "checkpoints/test_vocab.json"
    tokenizer.save(test_vocab_path)
    loaded_tokenizer = MiniTokenizer.load(test_vocab_path)
    assert loaded_tokenizer.vocab_size == tokenizer.vocab_size
    print("[+] Test 3 Passed: Save and Load from disk works offline!")
    
    if os.path.exists(test_vocab_path):
        os.remove(test_vocab_path)
    print("\n[+] All Tokenizer Unit Tests PASSED successfully!")
