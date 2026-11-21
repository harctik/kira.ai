#!/usr/bin/env bash
# Full Kira training pipeline for Colab (free T4). Run from the repo root after mounting Drive.
# Checkpoints are written straight to Drive so a runtime reset cannot lose them.
# Re-running skips stages that already finished.
set -euo pipefail

D=/content/drive/MyDrive/kira
mkdir -p "$D"

# 1. Data + tokenizer (local disk, quick to rebuild)
if [ ! -f data/kira/train.bin ]; then
  python prepare_data.py --docs-per-source 50000 --vocab-size 8192 --output-dir data/kira
fi
cp data/kira/tokenizer.json data/kira/meta.pkl "$D/"

# 2. Pre-training (2500 iters, cosine decay matched to the run length)
if [ ! -f "$D/out/PRETRAIN_DONE" ]; then
  python train.py --data-dir data/kira --out-dir "$D/out" --model-size 15m \
    --batch-size 32 --gradient-accumulation-steps 4 --max-iters 2500 --lr-decay-iters 2500
  touch "$D/out/PRETRAIN_DONE"
fi

# 3. SFT data
if [ ! -f data/kira_sft/train.bin ]; then
  python prepare_sft_data.py --tokenizer-path data/kira/tokenizer.json --output-dir data/kira_sft
fi

# 4. Fine-tune FROM the pre-trained checkpoint
python train.py --data-dir data/kira_sft --out-dir "$D/out_sft" --init-from "$D/out/ckpt.pt" \
  --model-size 15m --batch-size 16 --gradient-accumulation-steps 4 \
  --max-iters 800 --lr-decay-iters 800 --warmup-iters 50 --learning-rate 2e-4

ls -la "$D/out" "$D/out_sft"
echo "ALL DONE: $D/out_sft/ckpt.pt"
