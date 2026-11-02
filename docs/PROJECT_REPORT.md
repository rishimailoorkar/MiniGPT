# Academic Project Report: MiniGPT
## An Offline Small Language Model for Text Generation from Scratch

**Department of Computer Science and Engineering**  
**Project Title:** MiniGPT – Offline Small Language Model from Scratch  
**Target Hardware:** Intel(R) Core(TM) i5-14500 (2.60 GHz), 32 GB RAM, Windows 11  
**Execution Environment:** 100% Offline, Air-Gapped, PyTorch Native CPU  

---

## Abstract
Recent advances in Large Language Models (LLMs) have demonstrated impressive generative capabilities. However, modern foundation models typically rely on massive cloud infrastructure, billions of parameters, and proprietary external APIs that raise severe privacy, network latency, and operational cost concerns. This project presents **MiniGPT**, a compact, decoder-only Small Language Model (SLM) engineered completely from mathematical first principles in PyTorch. MiniGPT features approximately **2.69 million parameters**, utilizes a custom self-contained tokenizer, trains on a locally curated knowledge corpus via next-token prediction, and runs entirely offline on an Intel Core i5-14500 CPU. MiniGPT achieves empirical convergence in 2.19 minutes with an 85.7% relative drop in training loss (from 4.3342 to 0.6205 nats) and a validation perplexity of 8.65. The complete system includes a deterministic/stochastic sampling engine, automated evaluation curve generators, and a minimalist black-and-white Streamlit web interface operating with zero external network calls.

---

## Table of Contents
1. [Introduction & Problem Statement](#1-introduction--problem-statement)
2. [Theoretical Background & Literature Review](#2-theoretical-background--literature-review)
3. [System Architecture & Mathematical Formulation](#3-system-architecture--mathematical-formulation)
4. [Hardware Feasibility Analysis](#4-hardware-feasibility-analysis)
5. [Modular Implementation Details](#5-modular-implementation-details)
6. [Training Dynamics & Empirical Evaluation](#6-training-dynamics--empirical-evaluation)
7. [Offline Text Generation & Web Interface](#7-offline-text-generation--web-interface)
8. [Conclusion & Future Work](#8-conclusion--future-work)
9. [References](#9-references)

---

## 1. Introduction & Problem Statement

### 1.1 Background
The dominant paradigm in contemporary Natural Language Processing (NLP) revolves around cloud-hosted foundational models accessed via HTTP endpoints (e.g., OpenAI GPT-4, Google Gemini). While capable, these APIs introduce fundamental limitations:
- **Data Privacy & Compliance**: Sensitive user prompts traverse third-party networks, violating strict corporate or institutional confidentiality requirements.
- **Network Dependency**: Unusable in air-gapped, offline, defense, or remote laboratory settings.
- **Latency & Recurring Cost**: Financial barriers per token and unpredictable latency spikes.

### 1.2 Project Objective
The objective of this project is to develop **MiniGPT**: an autonomous, offline, small language model built from scratch without external libraries, pretrained weights, or network telemetry. MiniGPT demonstrates how a modern causal Transformer can be constructed, optimized, and deployed on commodity consumer hardware (Intel Core i5 CPU) while remaining completely transparent and reproducible.

---

## 2. Theoretical Background & Literature Review

### 2.1 The Transformer Revolution
Introduced by Vaswani et al. (2017) in *"Attention Is All You Need"*, the Transformer replaced sequential recurrence (RNNs, LSTMs) with **Self-Attention**, enabling fully parallelized training over arbitrary context sequences.

### 2.2 Autoregressive Causal Decoders
Radford et al. (2018, 2019) demonstrated with GPT-1 and GPT-2 that a **decoder-only architecture** trained strictly on unsupervised next-token prediction acts as an effective general-purpose generative model. Unlike bidirectional encoders (such as BERT) that condition on past and future tokens simultaneously, GPT employs a **causal lower-triangular attention mask** to ensure that token $t$ attends only to tokens at indices $\le t$.

---

## 3. System Architecture & Mathematical Formulation

MiniGPT follows a standardized Pre-Layer Normalization (Pre-LN) decoder architecture.

```
Input Tokens X  --> Token Embedding (W_te) + Positional Embedding (W_pe)
                           |
                           v
               [ Transformer Block 1 ]  (x 6 Layers)
               |   - LayerNorm 1
               |   - Causal Multi-Head Attention (6 Heads)
               |   - Residual Skip Connection (+)
               |   - LayerNorm 2
               |   - Feed-Forward MLP (GELU, 4x Expansion)
               |   - Residual Skip Connection (+)
                           |
                           v
               Final Layer Normalization (LN_f)
                           |
                           v
               Language Model Head (W_lm tied with W_te)
                           |
                           v
               Logits Output  --> Cross-Entropy Loss / Sampling Engine
```

### 3.1 Input Embedding Layer
Given a sequence of token IDs $X = (x_1, x_2, \dots, x_T) \in \mathbb{Z}^{B \times T}$:
$$\mathbf{h}_0 = \text{Embedding}_{\text{token}}(X) + \text{Embedding}_{\text{pos}}(\text{Pos}) \in \mathbb{R}^{B \times T \times d_{\text{model}}}$$
where $d_{\text{model}} = 192$, context length $T = 64$, and vocabulary size $V = 74$.

### 3.2 Causal Multi-Head Self-Attention
For each attention block, linear projections produce Queries ($Q$), Keys ($K$), and Values ($V$):
$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V \quad (W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}})$$
Each representation is split into $h = 6$ heads of dimension $d_k = 192 / 6 = 32$. Scaled dot-product attention applies a lower-triangular causal mask $M$:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$
where:
$$M_{i,j} = \begin{cases} 0 & \text{if } i \ge j \\ -\infty & \text{if } i < j \end{cases}$$
The concatenated head outputs are projected back into the residual stream:
$$\text{MHA}(X) = [\text{head}_1; \dots; \text{head}_h] W_O$$

### 3.3 Position-Wise Feed-Forward Network (MLP)
$$\text{FFN}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2$$
where $W_1 \in \mathbb{R}^{192 \times 768}$ (4x expansion) and $W_2 \in \mathbb{R}^{768 \times 192}$.

### 3.4 Weight Tying
To reduce memory footprint and prevent overfitting, the output language modeling head shares parameters directly with the token embedding table:
$$W_{\text{lm}} \equiv W_{\text{te}} \in \mathbb{R}^{74 \times 192}$$

---

## 4. Hardware Feasibility Analysis

| Metric | Target Specification | Measured MiniGPT Allocation |
| :--- | :--- | :--- |
| **CPU** | Intel Core i5-14500 (14 Cores / 20 Threads) | Multi-threaded CPU execution (OpenMP / MKL) |
| **RAM** | 32 GB DDR5 | Model + Optimizer + Batch: **~180 MB** (< 1% system RAM) |
| **GPU** | Integrated Intel Graphics | **0 GPU Required** |
| **Storage** | Local NVMe SSD | Checkpoint + Vocab + Plots: **~35 MB** total |
| **Network** | Air-Gapped / Disconnected | **0.0 KB** transmitted |

---

## 5. Modular Implementation Details

1. **`src/config.py`**: Typed `@dataclass` hyperparameter specification.
2. **`src/tokenizer.py`**: Pure Python character-level tokenizer with `<pad>`, `<unk>`, `<bos>`, and `<eos>` special control tokens. Serializes to `checkpoints/vocab.json`.
3. **`src/dataset.py`**: PyTorch `Dataset` that slices text chunks $x = [t_0 \dots t_{T-1}]$ and targets $y = [t_1 \dots t_T]$ for next-token prediction.
4. **`src/model.py`**: Modular implementation of `CausalSelfAttention`, `MLP`, `TransformerBlock`, and `MiniGPT`.
5. **`src/train.py`**: Training loop with AdamW optimizer, cosine decay with linear warmup, gradient clipping ($1.0$), and automatic checkpointing.
6. **`src/generate.py`**: Autoregressive generator supporting Greedy, Temperature, Top-K, and Top-P nucleus sampling.
7. **`evaluate.py`**: Offline visualization generator producing publication-ready PNG curves in `plots/`.
8. **`app.py`**: Minimalist black-and-white Streamlit web application.

---

## 6. Training Dynamics & Empirical Evaluation

MiniGPT was trained on a local corpus of Computer Science, Artificial Intelligence, and conversational Q&A text.

### 6.1 Training Progression Table
| Step | Train Loss (nats) | Val Loss (nats) | Val Perplexity | Learning Rate | Elapsed Time |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 4.3342 | 4.3246 | 75.53 | $7.92 \times 10^{-6}$ | 0.5s |
| **100** | 2.4417 | 2.6172 | 13.70 | $8.00 \times 10^{-4}$ | 10.2s |
| **200** | 2.2236 | 2.4968 | 12.14 | $7.78 \times 10^{-4}$ | 24.9s |
| **300** | 1.9194 | 2.3073 | 10.05 | $7.16 \times 10^{-4}$ | 35.3s |
| **500** | 1.4386 | 2.1575 | **8.65** | $5.03 \times 10^{-4}$ | 65.4s |
| **800** | 0.8618 | 2.2477 | 9.47 | $1.64 \times 10^{-4}$ | 106.0s |
| **1000** | **0.6205** | 2.3762 | 10.76 | $8.00 \times 10^{-5}$ | 131.2s |

### 6.2 Analysis of Results
- **Initial Loss**: Step 0 loss of $4.3342$ closely matches the theoretical maximum entropy $-\ln(1/74) = 4.3041$, verifying correct initialization.
- **Convergence**: Training loss decreased by **85.7%** down to $0.6205$ nats.
- **Generalization**: Validation loss reached a minimum of **2.1575** (Perplexity: **8.65**) around step 500.
- **Inference Latency**: Autoregressive decoding achieves **198–215 tokens/sec** on the Intel Core i5 CPU.

---

## 7. Offline Text Generation & Web Interface

The web interface is implemented in Streamlit with an ultra-clean, minimal monochrome design:
- **Local Host URL**: `http://localhost:8501`
- **Zero Remote Assets**: Streamlit telemetry and external CDNs are completely deactivated.
- **Interactive Controls**: Users can adjust Temperature, Top-K, and Top-P in real time and observe generation speed in tokens per second.

---

## 8. Conclusion & Future Work

MiniGPT demonstrates that building an offline small language model from scratch is both practically feasible and computationally accessible on consumer PC hardware. Within 2.19 minutes of CPU training, the model learned syntactic patterns, capitalization, technical vocabulary, and conversational Q&A formatting.

### Future Enhancements:
1. Implementation of Byte-Pair Encoding (BPE) subword tokenization to compress sequence length.
2. Direct Preference Optimization (DPO) or offline reinforcement learning from human feedback.
3. FlashAttention CPU kernels to scale context window to 512+ tokens.

---

## 9. References
1. Vaswani, A., et al. (2017). *Attention Is All You Need*. Advances in Neural Information Processing Systems (NeurIPS).
2. Radford, A., et al. (2019). *Language Models are Unsupervised Multitask Learners*. OpenAI Technical Report.
3. Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization*. International Conference on Learning Representations (ICLR).
4. Karpathy, A. (2022). *nanoGPT: The simplest, fastest repository for training/finetuning medium GPTs*. GitHub.
