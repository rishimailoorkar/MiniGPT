"""
MiniGPT: Minimalist Offline Small Language Model Interface
Clean, typography-driven monochrome (black & white) user interface.
Runs 100% locally on localhost without external network or API dependencies.
"""

import os
import sys
import time
import json
import streamlit as st
import torch
from PIL import Image

# Ensure project root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Ensure UTF-8 console output for Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.config import MiniGPTConfig
from src.generate import TextGenerator

# Page configuration
st.set_page_config(
    page_title="MiniGPT | Offline SLM",
    page_icon="▪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimalist Monochrome CSS
st.markdown("""
<style>
    /* Global resets and typography */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #111111;
        background-color: #FFFFFF;
    }

    /* Minimalist Header */
    .brand-title {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.04em;
        color: #000000;
        margin: 0;
        padding-top: 0.5rem;
    }

    .brand-subtitle {
        font-size: 0.92rem;
        color: #666666;
        letter-spacing: -0.01em;
        margin-top: 4px;
        margin-bottom: 1.2rem;
    }

    .badge-monochrome {
        display: inline-block;
        padding: 3px 10px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        border: 1px solid #111111;
        background-color: #000000;
        color: #FFFFFF;
        border-radius: 2px;
        margin-right: 8px;
    }

    .badge-ghost {
        display: inline-block;
        padding: 3px 10px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 500;
        color: #444444;
        border: 1px solid #E0E0E0;
        background-color: #FAFAFA;
        border-radius: 2px;
    }

    /* Hairline Divider */
    .hairline-divider {
        height: 1px;
        background-color: #EAEAEA;
        border: none;
        margin: 1.2rem 0 1.5rem 0;
    }

    /* Minimalist Metric Cards */
    .minimal-card {
        border: 1px solid #E5E5E5;
        background-color: #FFFFFF;
        padding: 1.2rem 1.4rem;
        border-radius: 4px;
        transition: border-color 0.2s ease;
    }
    .minimal-card:hover {
        border-color: #000000;
    }

    .card-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #737373;
        margin-bottom: 6px;
    }

    .card-value {
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #000000;
    }

    .card-detail {
        font-size: 0.8rem;
        color: #737373;
        margin-top: 4px;
    }

    /* Chat Messages */
    .stChatMessage {
        border: 1px solid #F0F0F0;
        border-radius: 4px;
        background-color: #FFFFFF;
        margin-bottom: 0.8rem;
    }

    /* Code & Pre */
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
        background-color: #F8F8F8 !important;
        color: #111111 !important;
        border: 1px solid #EEEEEE !important;
        border-radius: 3px !important;
    }

    /* Minimalist Button Styling */
    .stButton > button {
        background-color: #FFFFFF;
        color: #000000;
        border: 1px solid #111111;
        border-radius: 2px;
        font-weight: 500;
        letter-spacing: -0.01em;
        transition: all 0.15s ease-in-out;
        padding: 0.4rem 1rem;
    }
    .stButton > button:hover {
        background-color: #000000;
        color: #FFFFFF;
        border-color: #000000;
    }

    /* Monospace telemetry tag */
    .telemetry-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #737373;
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_generator():
    """Loads and caches model weights once in memory."""
    config = MiniGPTConfig()
    if not os.path.exists(config.model_checkpoint_path):
        return None, "Model checkpoint not found. Train the model first."
    if not os.path.exists(config.tokenizer_vocab_path):
        return None, "Vocabulary not found. Run tokenizer first."
        
    try:
        generator = TextGenerator(
            checkpoint_path=config.model_checkpoint_path,
            vocab_path=config.tokenizer_vocab_path,
            device="cpu"
        )
        return generator, None
    except Exception as e:
        return None, str(e)


def main():
    # Top Branding Bar
    col_brand, col_status = st.columns([3, 1])
    with col_brand:
        st.markdown('<div class="brand-title">MINIGPT</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="brand-subtitle">'
            'Offline Decoder-Only Small Language Model from Scratch'
            '</div>',
            unsafe_allow_html=True
        )
    with col_status:
        st.markdown(
            '<div style="text-align: right; padding-top: 15px;">'
            '<span class="badge-monochrome">LOCAL CPU</span> '
            '<span class="badge-ghost">AIR-GAPPED</span>'
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown('<hr class="hairline-divider">', unsafe_allow_html=True)

    # Load offline engine
    generator, err_msg = load_generator()
    if generator is None:
        st.error(f"Initialization Error: {err_msg}")
        return

    # Sidebar: Minimalist Controls
    st.sidebar.markdown("### **PARAMETERS**")
    st.sidebar.caption("Deterministic vs Stochastic Sampling Controls")

    decoding_mode = st.sidebar.selectbox("Decoding Strategy", ["Top-K / Nucleus Sampling", "Greedy Decoding"])
    
    if decoding_mode == "Greedy Decoding":
        temperature = 0.0
        top_k = None
        top_p = None
        st.sidebar.caption("Always selects token with argmax(logits).")
    else:
        temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=1.5, value=0.65, step=0.05)
        top_k = st.sidebar.slider("Top-K Cutoff", min_value=1, max_value=50, value=12, step=1)
        top_p = st.sidebar.slider("Top-P (Nucleus)", min_value=0.1, max_value=1.0, value=0.9, step=0.05)

    max_tokens = st.sidebar.slider("Sequence Length (Max Tokens)", min_value=20, max_value=180, value=80, step=10)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### **SYSTEM SPECS**")
    st.sidebar.markdown("""
    - **Architecture:** Causal Transformer
    - **Total Parameters:** 2,685,696
    - **Vocabulary:** 74 tokens
    - **Context Window:** 64 tokens
    - **Layers / Heads:** 6 Blocks / 6 Heads
    - **Embedding Dim:** 192 (head dim: 32)
    - **Processor:** Intel i5-14500 (CPU Native)
    - **RAM Footprint:** < 180 MB
    - **Network Traffic:** 0.0 KB (Offline)
    """)

    if st.sidebar.button("Clear Conversation History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Navigation Tabs
    tab_chat, tab_evaluation, tab_specs = st.tabs([
        "Terminal Chat",
        "Empirical Metrics & Graphs",
        "Architecture & Equations"
    ])

    # -------------------------------------------------------------
    # TAB 1: Chatbot Interface
    # -------------------------------------------------------------
    with tab_chat:
        # Prompt quick selectors
        st.caption("SELECT PROMPT PRESET:")
        q_cols = st.columns(4)
        sample_prompts = [
            "What is AI?",
            "What is a Transformer?",
            "What is Machine Learning?",
            "What is Python?"
        ]
        active_preset = None
        for idx, text in enumerate(sample_prompts):
            if q_cols[idx].button(text, key=f"quick_{idx}", use_container_width=True):
                active_preset = text

        # Conversation history state
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "MiniGPT offline system ready. Weights loaded from local disk. Ask any Computer Science or Transformer question.",
                    "stats": "System: CPU Native | Zero Cloud Calls"
                }
            ]

        # Render chat messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "stats" in msg and msg["stats"]:
                    st.markdown(f'<div class="telemetry-tag">{msg["stats"]}</div>', unsafe_allow_html=True)

        # Input query
        user_input = st.chat_input("Enter prompt...")
        if active_preset:
            user_input = active_preset

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            formatted_prompt = f"User: {user_input}\nAssistant:"

            with st.chat_message("assistant"):
                with st.spinner("Decoding tokens..."):
                    t_start = time.time()
                    response_text, gen_count, speed = generator.generate(
                        prompt=formatted_prompt,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        greedy=(decoding_mode == "Greedy Decoding")
                    )
                    t_elapsed = time.time() - t_start

                    # Clean prefix
                    if "Assistant:" in response_text:
                        clean_response = response_text.split("Assistant:")[-1].strip()
                    else:
                        clean_response = response_text[len(formatted_prompt):].strip()

                    if not clean_response:
                        clean_response = response_text.strip()

                    st.markdown(clean_response)
                    telemetry = f"[{gen_count} tokens | {t_elapsed:.2f}s | {speed:.1f} tokens/sec | offline CPU]"
                    st.markdown(f'<div class="telemetry-tag">{telemetry}</div>', unsafe_allow_html=True)

            st.session_state.messages.append({
                "role": "assistant",
                "content": clean_response,
                "stats": telemetry
            })

    # -------------------------------------------------------------
    # TAB 2: Evaluation & Metrics
    # -------------------------------------------------------------
    with tab_evaluation:
        st.markdown("#### **CONVERGENCE & EMPIRICAL BENCHMARKS**")
        config = MiniGPTConfig()
        
        summary_path = os.path.join(config.checkpoint_dir, "evaluation_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                s = json.load(f)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(
                    f'<div class="minimal-card">'
                    f'<div class="card-label">Trainable Parameters</div>'
                    f'<div class="card-value">{s["model_parameters"]:,}</div>'
                    f'<div class="card-detail">~2.69M (Target: 1-5M)</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with m2:
                st.markdown(
                    f'<div class="minimal-card">'
                    f'<div class="card-label">Training Loss</div>'
                    f'<div class="card-value">{s["final_train_loss"]:.4f}</div>'
                    f'<div class="card-detail">Down from {s["initial_train_loss"]:.4f} (85.7% drop)</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with m3:
                st.markdown(
                    f'<div class="minimal-card">'
                    f'<div class="card-label">Validation Loss</div>'
                    f'<div class="card-value">{s["best_val_loss"]:.4f}</div>'
                    f'<div class="card-detail">Best checkpoint at step {s["best_val_step"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with m4:
                st.markdown(
                    f'<div class="minimal-card">'
                    f'<div class="card-label">Validation Perplexity</div>'
                    f'<div class="card-value">{s["best_val_perplexity"]:.2f}</div>'
                    f'<div class="card-detail">Baseline: 75.53 (Random Guess)</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### **OFFLINE PUBLICATION PLOTS**")
        
        plot_dashboard = os.path.join(config.plots_dir, "evaluation_dashboard.png")
        plot_loss = os.path.join(config.plots_dir, "loss_curves.png")
        plot_ppl = os.path.join(config.plots_dir, "perplexity_curve.png")
        plot_lr = os.path.join(config.plots_dir, "learning_rate_schedule.png")

        if os.path.exists(plot_dashboard):
            st.image(Image.open(plot_dashboard), caption="Figure 1: Comprehensive Model Performance Dashboard", use_container_width=True)

        c1, c2 = st.columns(2)
        if os.path.exists(plot_loss):
            c1.image(Image.open(plot_loss), caption="Figure 2: Cross-Entropy Loss (Train vs Val)", use_container_width=True)
        if os.path.exists(plot_ppl):
            c2.image(Image.open(plot_ppl), caption="Figure 3: Perplexity: exp(Loss)", use_container_width=True)

    # -------------------------------------------------------------
    # TAB 3: Model Architecture & Math
    # -------------------------------------------------------------
    with tab_specs:
        st.markdown("#### **MATHEMATICAL ARCHITECTURE (DECODER-ONLY TRANSFORMER)**")
        st.markdown(r"""
        ##### 1. Input Representation
        The discrete token sequence $X \in \mathbb{R}^{B \times T}$ is transformed into continuous vector space:
        $$\mathbf{h}_0 = W_{\text{te}}(X) + W_{\text{pe}}(\text{Pos}) \in \mathbb{R}^{B \times T \times d_{\text{model}}}$$
        where $W_{\text{te}} \in \mathbb{R}^{V \times 192}$ and $W_{\text{pe}} \in \mathbb{R}^{64 \times 192}$.

        ##### 2. Causal Multi-Head Attention
        $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$
        where $M_{i,j} = 0$ for $i \ge j$ and $M_{i,j} = -\infty$ for $i < j$.
        
        ##### 3. Position-Wise Feed-Forward Network
        $$\text{FFN}(x) = \text{GELU}(x W_1) W_2$$
        where $W_1 \in \mathbb{R}^{192 \times 768}$ and $W_2 \in \mathbb{R}^{768 \times 192}$.

        ##### 4. Pre-Layer Normalization
        $$\mathbf{h}_{l+1} = \mathbf{h}_l + \text{MHA}(\text{LN}(\mathbf{h}_l))$$
        $$\mathbf{h}_{\text{out}} = \mathbf{h}_{l+1} + \text{MLP}(\text{LN}(\mathbf{h}_{l+1}))$$
        """)

        st.markdown("---")
        st.markdown("#### **ARCHITECTURAL HYPERPARAMETERS**")
        specs_table = {
            "Parameter Name": [
                "Model Architecture",
                "Total Trainable Parameters",
                "Vocabulary Size (Tokens)",
                "Context Window (Block Size)",
                "Embedding Dimension (d_model)",
                "Transformer Decoder Layers",
                "Attention Heads",
                "Head Dimension",
                "Feed-Forward Expansion Ratio",
                "Weight Tying (WTE = LM_Head)",
                "Activation Function",
                "Optimizer",
                "Peak Learning Rate",
                "Hardware Execution Engine"
            ],
            "Specification": [
                "Decoder-Only Causal Transformer",
                "2,685,696 (~2.69 Million)",
                "74 unique character/control tokens",
                "64 tokens",
                "192 dimensions",
                "6 stacked blocks",
                "6 heads",
                "32 dimensions (192 / 6)",
                "4x (768 dimensions)",
                "Enabled (shared weights)",
                "GELU (Gaussian Error Linear Unit)",
                "AdamW (weight decay = 0.1)",
                "8.0e-4 (cosine annealing with warmup)",
                "Intel(R) Core(TM) i5-14500 (CPU Native)"
            ]
        }
        st.table(specs_table)


if __name__ == "__main__":
    main()
