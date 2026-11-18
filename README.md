<div align="center">

![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Architecture](https://img.shields.io/badge/Architecture-RoPE%20%7C%20SwiGLU%20%7C%20GQA-blueviolet?style=for-the-badge)
![Quantization](https://img.shields.io/badge/Inference-INT8%20Quantized-00A67E?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

# 🧠 Kira: Autonomous Self-Evolving AI System

**A privacy-preserving, local neural model and agent architecture built 100% from scratch with Zero AI Contamination.**

[Architecture](#-system-architecture) • [Core Modules](#-core-modules) • [Scaling Law](#-scaled-model-configurations) • [Getting Started](#-how-to-run)

</div>

---

## 🏛️ System Architecture

```
                          ┌───────────────────────────┐
                          │         KIRA CORE         │
                          └─────────────┬─────────────┘
                                        │
    ┌──────────────────┬────────────────┼────────────────┬──────────────────┐
    ▼                  ▼                ▼                ▼                  ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  1. Coding & │ │   5. Scaled  │ │4. Continuous │ │  3. Laptop & │ │  2. Personal │
│  AI Engineer │ │  Architecture│ │  Evolution   │ │ Mobile Bridge│ │  Data Engine │
│  Knowledge   │ │   (15M->3B)  │ │(Self-Learning│ │  (OS Server) │ │ (Local RAG)  │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

---

## 📁 Core Modules

| Module | Purpose |
| :--- | :--- |
| **`model_scale.py`** | Modern Llama/Mistral architecture (RoPE, RMSNorm, SwiGLU, GQA) scalable from 15M to 3B with CPU INT8 quantization. |
| **`evolution.py`** | Self-learning memory loop, daily experience consolidation, and continuous fine-tuning exports. |
| **`tools.py`** | Sandboxed OS execution engine: inspect/edit files, hardware status, and shell command execution. |
| **`server.py`** | Mobile and laptop bridge server. Connect from any device via home Wi-Fi (`http://<ip>:8080`). |
| **`personal_data.py`** | 100% private local document & code indexer with offline semantic search. |
| **`prepare_code_data.py`** | Curates human-written Python, PyTorch, & Stack Overflow code without synthetic LLM contamination. |
| **`prepare_data.py`** | Streams articles & trains custom 8k Byte-Pair Encoding (BPE) tokenizer from scratch. |
| **`train.py`** | Memory-mapped pre-training & fine-tuning engine with mixed precision & cosine learning rate scheduling. |
| **`chat.py`** | Interactive terminal session for conversational text generation and tool invocation. |

---

## 📐 Scaled Model Configurations

| Configuration | Parameters | Hidden Dim | Layers | Heads | KV Heads | Context Window |
|---|---|---|---|---|---|---|
| **Kira-Nano** | 15M | 288 | 6 | 6 | 2 | 1,024 |
| **Kira-Micro** | 30M | 384 | 8 | 6 | 2 | 2,048 |
| **Kira-Small** | 70M | 512 | 10 | 8 | 2 | 2,048 |
| **Kira-Base** | 300M | 1,024 | 16 | 16 | 4 | 4,096 |
| **Kira-Pro** | 1.1B | 2,048 | 22 | 16 | 4 | 4,096 |
| **Kira-Max** | 3.2B | 3,072 | 28 | 24 | 8 | 8,192 |

---

## ⚡ How to Run

### 1. Launch the Control Server

```bash
python server.py
# Open: http://localhost:8080 in your browser
```

### 2. Interactive Terminal Chat

```bash
python chat.py
```

### 3. Pretraining / Fine-Tuning

```bash
python train.py config/train_kira_15m.py
```

---

## 📄 License

This project is licensed under the MIT License.
