# MiniGPT: Offline Small Language Model from Scratch

[![Platform](https://img.shields.io/badge/Platform-Windows-0078D6?style=flat-square&logo=windows)](https://microsoft.com)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch_CPU-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![Offline](https://img.shields.io/badge/Mode-100%25_Air--Gapped_Offline-000000?style=flat-square)](https://github.com)
[![Parameters](https://img.shields.io/badge/Parameters-2.69_Million-black?style=flat-square)](https://github.com)

A compact, decoder-only Transformer language model built completely from mathematical first principles in PyTorch. Runs **100% offline on Windows** using native CPU multithreading on an **Intel(R) Core(TM) i5-14500** with **zero external cloud APIs, zero pretrained model weights, and zero internet calls**.

---

## Quick Start (One-Click)

To launch the offline chatbot web interface:
1. Double-click **`run_offline.bat`** in Windows File Explorer.
2. Your browser will automatically open to **`http://localhost:8501`**.

Or start via command line:
```cmd
streamlit run app.py
```

---

## Project Highlights

- **Custom Decoder-Only Transformer**: Learned token embeddings, positional encodings, multi-head causal self-attention, GELU feed-forward networks, pre-layer normalization, and weight tying.
- **Self-Contained Tokenizer**: Built from scratch in pure Python with special tokens (`<pad>`, `<unk>`, `<bos>`, `<eos>`).
- **Hardware Optimized**: Fully trained in **2.19 minutes** on an Intel Core i5-14500 CPU (< 180 MB RAM footprint).
- **Inference Speed**: Blazing-fast **~210 tokens/second** streaming directly on CPU.
- **Empirical Convergence**: Training loss dropped from **4.3342 down to 0.6205 nats** (85.7% drop); validation perplexity reached **8.65**.
- **Minimalist Monochrome UI**: Clean, high-contrast black-and-white Streamlit web interface with interactive sampling sliders (Temperature, Top-K, Top-P).
- **Academic Package**: Includes full college project report (`docs/PROJECT_REPORT.md`) and 12-slide presentation deck with defense answers (`docs/PRESENTATION_SLIDES.md`).

---

## Project Structure

```
MiniGPT/
│
├── data/
│   ├── raw/
│   │   └── corpus.txt              # Local training knowledge base
│   ├── train.txt                   # 90% Training split
│   ├── val.txt                     # 10% Validation split
│   └── prepare_data.py             # Data preprocessing & splitting script
│
├── src/
│   ├── __init__.py                 # Package marker
│   ├── config.py                   # Dataclass hyperparameters & file paths
│   ├── tokenizer.py                # Self-contained offline Tokenizer
│   ├── dataset.py                  # PyTorch Dataset & next-token prediction
│   ├── model.py                    # Decoder-Only Transformer architecture
│   ├── train.py                    # Offline training engine with AdamW & Cosine LR
│   └── generate.py                 # Sampling engine (Greedy, Temp, Top-K, Top-P)
│
├── checkpoints/
│   ├── minigpt_model.pt            # Trained local weights checkpoint (~31 MB)
│   ├── vocab.json                  # Serialized vocabulary dictionary
│   ├── metrics.json                # Step-by-step training metrics
│   └── evaluation_summary.json     # Empirical evaluation summary
│
├── plots/
│   ├── evaluation_dashboard.png    # 2x2 multi-panel performance dashboard
│   ├── loss_curves.png             # Training vs Validation cross-entropy loss
│   ├── perplexity_curve.png        # Validation perplexity progression
│   └── learning_rate_schedule.png  # Cosine annealing schedule
│
├── docs/
│   ├── PROJECT_REPORT.md           # Full academic project submission report
│   └── PRESENTATION_SLIDES.md      # 12-slide presentation deck & defense Q&A
│
├── .streamlit/
│   └── config.toml                 # Streamlit offline configuration & styling
│
├── app.py                          # Minimalist black & white Streamlit Web App
├── evaluate.py                     # Evaluation benchmark & plot generator
├── run_offline.bat                 # One-click Windows offline launcher
├── requirements.txt                # Offline dependency list & wheel guide
└── README.md                       # Master documentation
```

---

## Step-by-Step CLI Execution Guide

### 1. Offline Environment Setup
If moving to a completely disconnected PC:
```cmd
pip install -r requirements.txt
```
*(See `requirements.txt` for instructions on bundling wheel files for air-gapped systems).*

### 2. Prepare Corpus
Splits the raw domain corpus into `data/train.txt` and `data/val.txt`:
```cmd
python data/prepare_data.py
```

### 3. Build Offline Vocabulary
```cmd
python -c "from src.tokenizer import MiniTokenizer; tok = MiniTokenizer(); tok.train_from_text(open('data/train.txt', encoding='utf-8').read()); tok.save('checkpoints/vocab.json')"
```

### 4. Train MiniGPT Offline
```cmd
python src/train.py
```
*(Trains in ~2 minutes on Intel CPU; logs metrics to `checkpoints/metrics.json` and weights to `checkpoints/minigpt_model.pt`).*

### 5. Generate Text in CLI
```cmd
python src/generate.py --prompt "User: What is AI?`nAssistant:" --max_tokens 80 --temperature 0.65 --top_k 12
```

### 6. Generate Publication Graphs & Evaluation Summary
```cmd
python evaluate.py
```

### 7. Launch Minimalist Web UI
```cmd
streamlit run app.py
```

---

## Model Architecture Specifications

| Hyperparameter | Specification |
| :--- | :--- |
| **Model Type** | Decoder-Only Causal Transformer |
| **Total Parameters** | 2,685,696 (~2.69 Million) |
| **Vocabulary Size** | 74 tokens |
| **Context Window ($T$)** | 64 tokens |
| **Embedding Dimension ($d_{\text{model}}$)** | 192 |
| **Transformer Blocks ($N$)** | 6 layers |
| **Attention Heads ($h$)** | 6 heads (32 dim per head) |
| **MLP Expansion** | 4x ($192 \to 768 \to 192$) |
| **Weight Tying** | Enabled ($W_{\text{lm}} \equiv W_{\text{te}}$) |
| **Activation Function** | Gaussian Error Linear Unit (GELU) |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.95$, weight decay $=0.1$) |
| **Peak Learning Rate** | $8.0 \times 10^{-4}$ (with linear warmup & cosine decay) |
| **Target Hardware** | Intel(R) Core(TM) i5-14500 (CPU Native) |

---

## Academic Deliverables Included
- **Comprehensive Project Report**: See [`docs/PROJECT_REPORT.md`](file:///c:/Users/20231COM0209/Downloads/MiniGPT/docs/PROJECT_REPORT.md)
- **Presentation Deck & Defense Q&A**: See [`docs/PRESENTATION_SLIDES.md`](file:///c:/Users/20231COM0209/Downloads/MiniGPT/docs/PRESENTATION_SLIDES.md)
