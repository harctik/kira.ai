# kira.ai: Autonomous Neural Architecture & Continuous Agent Engine

A privacy-preserving neural model family and agent architecture designed strictly from scratch in PyTorch with zero synthetic AI contamination, featuring custom Byte-Pair Encoding (BPE), scalable decoder-only transformers from 13.8M to 2.84B parameters, dynamic INT8 CPU quantization, and continuous memory evolution.

![kira.ai Banner](banner.png)

---

## Technical Overview

kira.ai addresses two fundamental constraints in modern machine learning systems:
1. Low-latency private execution on consumer CPU hardware without reliance on external cloud APIs or GPUs. This is achieved through dynamic INT8 quantization, Grouped-Query Attention (GQA), and RMSNorm pre-normalization.
2. Prevention of synthetic model degradation. Modern language models suffer from degradation when trained on synthetic data produced by other commercial LLMs. Kira's training sets are filtered strictly to verified human-authored code and original human dialogue corpora.

---

## System Architecture

The architecture partitions into four functional subsystems surrounding a modern decoder-only transformer:

1. Human Knowledge Engine: Curates verified human-written Python, PyTorch, and algorithms code without synthetic contamination.
2. Local Document Indexer: Offline semantic indexer and retrieval-augmented generation engine operating over private local files.
3. Operating System Bridge: Sandboxed execution layer for terminal commands, hardware monitoring, and cross-device local network communication.
4. Continuous Evolution Engine: Consolidates user feedback and coding interactions into permanent episodic memory for ongoing fine-tuning exports.

---

## Model Scaling Specifications

The architecture is implemented in `model_scale.py` as five presets (parameter counts measured with `python -m benchmarks.int8_cpu_latency`, weights tied, vocabulary 8,192):

| Preset | Parameters | Hidden Dimension | Layers | Attention Heads | KV Heads (GQA) |
|---|---|---|---|---|---|
| `15m` | 13.8M | 384 | 6 | 6 | 6 (MHA) |
| `125m` | 91.2M | 768 | 12 | 12 | 12 (MHA) |
| `500m` | 272.6M | 1,024 | 24 | 16 | 4 |
| `1b` | 985.7M | 2,048 | 22 | 16 | 4 |
| `3b` | 2,843.9M | 3,072 | 28 | 24 | 8 |

### Mathematical Design Highlights

Rotary Position Embeddings (RoPE) replace fixed absolute positional encodings, enabling length generalization across extended sequence contexts.

SwiGLU gated linear units replace standard ReLU activations:

$$\text{SwiGLU}(x) = (x W_1) \otimes \sigma(x W_2) W_3$$

Grouped-Query Attention (GQA) reduces key-value cache memory overhead during autoregressive generation by sharing key-value projections across head subsets.

RMSNorm pre-normalization eliminates mean-centering calculations to improve training speed and stability.

---

## Training Convergence & Verification

### 1. Training Loss Trajectory
Validation loss trajectories demonstrate stable convergence during pre-training on human-authored corpora without learning rate collapse or gradient explosions.

![Pre-Training Loss Convergence](docs/assets/loss_curve.png)

### 2. Runtime Benchmarks & Unit Tests
Dynamic INT8 quantization (`quantize_for_cpu`) was measured with `python -m benchmarks.int8_cpu_latency` (2 threads, 128-token context) on a 13th Gen Intel Core i7-1360P, using PyTorch 2.12.1 (CPU build) and Python 3.10.2 on Windows 11. Each run averages 15 forward passes after 3 warm-up passes. Over two runs, the 13.8M preset took 41-44 ms in FP32 and 27-29 ms in INT8 (about 1.5x speedup), and the 91.2M preset took 294-297 ms in FP32 and 145-147 ms in INT8 (about 2.0x speedup). Timings depend on hardware and system load, so rerun the benchmark on your own machine. The model has no KV cache yet, so each pass recomputes the full context.

---

## Zero-Contamination Corpus

All tokenization and data preprocessing scripts run independently:

1. `prepare_data.py`: Trains an 8,192-token custom Byte-Pair Encoding (BPE) tokenizer and formats human-written Wikipedia articles using memory-mapped binary files.
2. `prepare_code_data.py`: Extracts verified human-authored codebases from open-source algorithmic repositories.
3. `prepare_sft_data.py`: Prepares multi-turn human dialogue records for instruction fine-tuning.

---

## Local Development & Quick Start

### 1. Environment Setup
```bash
git clone https://github.com/harctik/kira.ai.git
cd kira.ai
python -m venv venv

# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
pytest tests/ -v
```

### 3. Launch Local Network Bridge
```bash
python server.py
# Access http://localhost:8080 in your browser
```

---

## Author and Engineering Attribution

All architecture design, PyTorch neural modeling, tokenizer training, INT8 quantization benchmarks, and agent infrastructure in this repository were engineered end-to-end by Avvari Naga Harthikanand (Harthik).

Developer: Avvari Naga Harthikanand
Location: Bangalore, Karnataka, India
LinkedIn: https://www.linkedin.com/in/harthik
GitHub: https://github.com/harctik
Email: harthikanand90@gmail.com
License: MIT License
