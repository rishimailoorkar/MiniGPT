# MiniGPT: Presentation Slides & Defense Guide

**Project Title:** MiniGPT: An Offline Small Language Model from Scratch  
**Presenter:** Computer Science & Engineering Student  
**Target Hardware:** Intel(R) Core(TM) i5-14500 (CPU Native), 32 GB RAM  
**Format:** 12-Slide Academic Deck with Speaking Notes and Examiner Q&A  

---

## Slide 1: Title & Overview
### Slide Content:
- **Title:** MiniGPT – Offline Small Language Model for Text Generation
- **Sub-headline:** Building a Decoder-Only Transformer from Scratch in PyTorch on Windows
- **Key Badges:** 100% Offline | Zero Cloud APIs | 2.69M Parameters | PyTorch CPU
- **Presenter Name & University Details**

> **Speaker Notes:**
> *"Good morning respected professors and examiners. Today I am presenting MiniGPT, a project focused on designing, training, and running a Generative Pre-trained Transformer completely offline on local hardware, without relying on cloud APIs, external weights, or internet connections."*

---

## Slide 2: Motivation & Problem Statement
### Slide Content:
- **The Cloud LLM Problem:**
  - Data privacy violations (confidential prompts sent over public networks).
  - High subscription costs and per-token API charges.
  - Inability to run in air-gapped, defense, or offline environments.
  - Black-box opacity: Proprietary weights with no internal inspectability.
- **Project Solution:**
  - An open, mathematically transparent Small Language Model (SLM).
  - Operates autonomously on consumer CPU hardware.

> **Speaker Notes:**
> *"Modern AI is increasingly locked behind proprietary APIs. MiniGPT addresses this by prioritizing complete offline independence, zero-latency network reliance, and 100% data privacy."*

---

## Slide 3: System Architecture
### Slide Content:
- **Model Type:** Decoder-Only Autoregressive Transformer
- **Input Pipeline:**
  - Discrete Tokens $\to$ Word Token Embeddings ($W_{\text{te}}$) + Word Position Embeddings ($W_{\text{pe}}$)
- **Processing Stack:**
  - 6 Stacked Transformer Decoder Blocks
  - Pre-Layer Normalization (Pre-LN) for gradient stability
  - Multi-Head Causal Self-Attention (6 heads, $d_k = 32$)
  - Position-wise Feed-Forward Network (GELU, 4x expansion)
- **Output:**
  - Final Layer Normalization $\to$ Tied Language Model Head $\to$ Logits

> **Speaker Notes:**
> *"Here is the architectural schematic of MiniGPT. Like GPT-2, it uses a decoder-only architecture. Notice the Pre-LayerNorm placement, which ensures smooth gradient flow during backpropagation."*

---

## Slide 4: Mathematical Foundations
### Slide Content:
1. **Causal Attention Formula:**
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$
   - Upper-triangular elements masked with $-\infty$ to enforce autoregressive causality.
2. **Next-Token Prediction Objective (Cross-Entropy Loss):**
   $$\mathcal{L} = -\frac{1}{T} \sum_{t=1}^T \log P(x_t \mid x_{<t})$$
3. **Weight Tying Principle:**
   $$W_{\text{lm}} \equiv W_{\text{te}}$$

> **Speaker Notes:**
> *"The critical mathematical element is the causal mask matrix M. It strictly prevents tokens from peeking into the future, enabling autoregressive generation."*

---

## Slide 5: Hardware & Computational Budget
### Slide Content:
- **Processor:** Intel(R) Core(TM) i5-14500 (14 Cores / 20 Threads)
- **Memory:** 32 GB RAM (MiniGPT consumes < 180 MB)
- **GPU:** Intel Integrated Graphics (Not required; runs natively on CPU)
- **Parameter Breakdown:**
  - Embeddings: $74 \times 192 = 14,208$
  - Position: $64 \times 192 = 12,288$
  - 6 Blocks $\times 443,136 \approx 2,658,816$
  - **Total Parameters:** **2,685,696 (~2.69 Million)**

> **Speaker Notes:**
> *"We specifically designed the parameter count to be ~2.69 million. This perfectly matches the 1–5 million parameter requirement, allowing full multi-threaded training on an Intel CPU without overheating or memory bottlenecks."*

---

## Slide 6: Dataset & Offline Tokenizer
### Slide Content:
- **Offline Tokenizer:**
  - Implemented from scratch in pure Python (zero HuggingFace dependencies).
  - Handles `<pad>`, `<unk>`, `<bos>`, `<eos>` special control tokens.
  - Serializes to local `vocab.json`.
- **Domain Knowledge Corpus:**
  - Curated text spanning Computer Science fundamentals, Deep Learning, and Q&A dialogues.
  - 90% Training split (`train.txt`) / 10% Validation split (`val.txt`).

> **Speaker Notes:**
> *"Rather than downloading black-box tokenizers, we authored MiniTokenizer from scratch. It builds its vocabulary strictly from local files, ensuring 100% offline reproducibility."*

---

## Slide 7: Training Dynamics & Optimization
### Slide Content:
- **Optimizer:** AdamW ($\beta_1=0.9, \beta_2=0.95$, weight decay $=0.1$)
  - Weight decay applied only to 2D matrices (Linear layers); biases and LayerNorms excluded.
- **LR Schedule:** Cosine Annealing with linear warmup (peak LR: $8 \times 10^{-4}$).
- **Regularization:** Gradient norm clipping ($1.0$) and dropout ($0.1$).
- **Total Training Duration:** **131 seconds (2.19 minutes)** on local CPU!

> **Speaker Notes:**
> *"We decoupled weight decay so normalization vectors are not penalized. With cosine annealing, the model smoothly transitioned from random initialization to sharp convergence."*

---

## Slide 8: Empirical Results & Loss Progression
### Slide Content:
- **Initial Random Loss:** **4.3342 nats** (matches theoretical baseline $-\ln(1/74) = 4.3041$)
- **Final Training Loss:** **0.6205 nats** (85.7% error reduction)
- **Best Validation Loss:** **2.1575 nats**
- **Best Validation Perplexity:** **8.65** (down from initial 75.53)
- **Visuals:** [Figure: Evaluation Dashboard and Loss Curves]

> **Speaker Notes:**
> *"As shown on the evaluation dashboard, our training loss dropped sharply in the first 300 steps. The validation perplexity reached a strong score of 8.65, demonstrating high predictive confidence."*

---

## Slide 9: Text Generation & Sampling Strategies
### Slide Content:
- **Greedy Decoding:** Deterministic $\arg\max$, fast and consistent.
- **Temperature Scaling ($T$):** Controls distribution sharpness ($0.65$ optimal).
- **Top-K Sampling ($K=12$):** Eliminates low-probability noise tokens.
- **Top-P (Nucleus, $P=0.90$):** Dynamic cumulative probability thresholding.
- **CPU Throughput:** **198–215 tokens per second** (instantaneous streaming).

> **Speaker Notes:**
> *"For inference, we implemented both greedy decoding and stochastic Top-K/Top-P sampling. On the Intel Core i5 CPU, generation runs at over 200 tokens per second."*

---

## Slide 10: Streamlit Minimalist Interface
### Slide Content:
- **Design Philosophy:** Minimalist, high-contrast black-and-white theme.
- **Features:**
  - Conversational interface with chat memory.
  - Dynamic generation controls (Temperature, Top-K, Top-P sliders).
  - Integrated performance metrics tab (viewing generated loss curves).
  - One-click launcher (`run_offline.bat`) for Windows.

> **Speaker Notes:**
> *"To make MiniGPT user-friendly, we built a clean monochrome Streamlit interface. It can be launched offline with a single click via run_offline.bat."*

---

## Slide 11: Summary of Deliverables
### Slide Content:
1. Complete, self-contained Python source code (`src/`).
2. Custom offline Tokenizer and Dataset modules.
3. 2.69M parameter Decoder-Only Transformer architecture.
4. Trained local checkpoint (`checkpoints/minigpt_model.pt`).
5. Evaluation metrics and 4 publication-quality offline plots (`plots/`).
6. Streamlit Chatbot application (`app.py`).
7. Comprehensive Academic Report and Presentation Deck.

---

## Slide 12: Examiner Q&A Defense Guide
### Common Questions & Prepared Answers:

**Q1: Why did you choose a Decoder-only architecture instead of Encoder-Decoder?**  
*Answer:* Decoder-only architectures (GPT) are naturally autoregressive and excel at next-token text generation. Encoder-Decoder architectures (T5) are heavier and better suited for sequence-to-sequence translation.

**Q2: How does the model run so fast on a CPU without a GPU?**  
*Answer:* We constrained the parameter count to 2.69 million and embedding dimension to 192. At this scale, modern CPU vector instructions (AVX-2/AVX-512) and multi-threaded PyTorch BLAS libraries execute matrix multiplications faster than the overhead of transferring small tensors to GPU VRAM.

**Q3: How do you guarantee the model is 100% offline?**  
*Answer:* We implemented our own tokenizer without HuggingFace, disabled Streamlit telemetry, load all checkpoints and weights strictly from local disk paths, and tested execution with network adapters disabled.

---
## Thank You! Questions & Discussion.
