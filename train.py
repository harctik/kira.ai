"""
train.py - Pre-training loop for Kira LLM
Runs on Google Colab (free T4 GPU), Kaggle, or local machine.
Uses memory-mapped binary files (train.bin / val.bin) for instant loading and minimal RAM usage.
"""

import os
import time
import math
import pickle
import argparse
from contextlib import nullcontext
import numpy as np
import torch

from model import GPTConfig, GPT


def parse_args():
    parser = argparse.ArgumentParser(description="Train Kira LLM from scratch")
    parser.add_argument("--data-dir", type=str, default="data/kira", help="Directory with train.bin, val.bin, meta.pkl")
    parser.add_argument("--out-dir", type=str, default="out", help="Directory to save checkpoints")
    
    # Model architecture options (~15M or ~30M)
    parser.add_argument("--model-size", type=str, default="15m", choices=["15m", "30m", "toy"],
                        help="Pre-configured model size: '15m' (~15M params), '30m' (~31M params), 'toy' (~2M for testing)")
    parser.add_argument("--block-size", type=int, default=512, help="Context window length (default: 512 tokens)")
    
    # Training hyperparams
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per micro-step")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation steps (effective batch: batch_size * grad_accum)")
    parser.add_argument("--max-iters", type=int, default=5000, help="Total training iterations")
    parser.add_argument("--learning-rate", type=float, default=6e-4, help="Peak learning rate")
    parser.add_argument("--min-lr", type=float, default=6e-5, help="Minimum learning rate after cosine decay")
    parser.add_argument("--warmup-iters", type=int, default=200, help="Linear warmup iterations")
    parser.add_argument("--lr-decay-iters", type=int, default=5000, help="Learning rate decay iterations")
    parser.add_argument("--weight-decay", type=float, default=1e-1, help="AdamW weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm")
    
    # Initialization and checkpoint loading
    parser.add_argument("--init-from", type=str, default="scratch",
                        help="'scratch' to start from zero, or path to a checkpoint (e.g. 'out/ckpt.pt') to continue/fine-tune")
    
    # Evaluation and logging
    parser.add_argument("--eval-interval", type=int, default=250, help="Evaluation every N iterations")
    parser.add_argument("--eval-iters", type=int, default=50, help="Number of batches to evaluate on")
    parser.add_argument("--log-interval", type=int, default=10, help="Log to terminal every N iterations")
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'cuda', or 'cpu'")
    
    return parser.parse_args()


def get_model_config(model_size, block_size, vocab_size):
    if model_size == "toy":
        return GPTConfig(block_size=block_size, vocab_size=vocab_size, n_layer=4, n_head=4, n_embd=128)
    elif model_size == "15m":
        # ~15.4M parameters: 6 layers, 6 heads, 384 embedding dim
        return GPTConfig(block_size=block_size, vocab_size=vocab_size, n_layer=6, n_head=6, n_embd=384)
    elif model_size == "30m":
        # ~31.2M parameters: 8 layers, 8 heads, 512 embedding dim
        return GPTConfig(block_size=block_size, vocab_size=vocab_size, n_layer=8, n_head=8, n_embd=512)
    else:
        raise ValueError(f"Unknown model size: {model_size}")


def get_batch(data, batch_size, block_size, device, device_type):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # Pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Device configuration
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    device_type = 'cuda' if 'cuda' in device else 'cpu'

    # Mixed precision setup
    dtype = 'bfloat16' if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    print("=" * 60)
    print("  Kira LLM - Pre-Training (100% Human Dataset)")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if device_type == 'cuda' else 'CPU'})")
    print(f"  Precision: {dtype if device_type == 'cuda' else 'float32'}")
    print("=" * 60)

    # 2. Load dataset metadata and memory-map .bin files
    train_bin = os.path.join(args.data_dir, "train.bin")
    val_bin = os.path.join(args.data_dir, "val.bin")
    meta_path = os.path.join(args.data_dir, "meta.pkl")

    if not os.path.exists(train_bin) or not os.path.exists(val_bin):
        raise FileNotFoundError(f"Missing data in {args.data_dir}. Run prepare_data.py first!")

    vocab_size = 8192
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
            vocab_size = meta["vocab_size"]
            print(f"Loaded metadata from {meta_path}: vocab_size = {vocab_size}")

    train_data = np.memmap(train_bin, dtype=np.uint16, mode='r')
    val_data = np.memmap(val_bin, dtype=np.uint16, mode='r')
    print(f"Training tokens:   {len(train_data):,}")
    print(f"Validation tokens: {len(val_data):,}")

    # 3. Initialize Model
    config = get_model_config(args.model_size, args.block_size, vocab_size)
    model = GPT(config)

    if args.init_from != "scratch":
        if os.path.exists(args.init_from):
            print(f"\n[CONTINUAL LEARNING] Loading weights from existing checkpoint: {args.init_from}...")
            ckpt = torch.load(args.init_from, map_location=device)
            state_dict = ckpt['model']
            # Unwanted prefix _orig_mod. if compiled
            unwanted_prefix = '_orig_mod.'
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            model.load_state_dict(state_dict, strict=False)
            print("Successfully loaded pre-trained foundation weights! Continuing training...")
        else:
            print(f"Warning: Checkpoint {args.init_from} not found, initializing from scratch.")

    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel initialized ({args.model_size.upper()}):")
    print(f"  Layers: {config.n_layer}, Heads: {config.n_head}, Embd: {config.n_embd}, Block: {config.block_size}")
    print(f"  Total Parameters: {total_params / 1e6:.2f}M")
    print(f"  Effective Batch Size: {args.batch_size * args.gradient_accumulation_steps * config.block_size:,} tokens/iter")

    # 4. Optimizer and Learning Rate Scheduler
    optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate, (0.9, 0.95), device_type)

    def get_lr(it):
        # 1) Linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            return args.learning_rate * it / args.warmup_iters
        # 2) If it > lr_decay_iters, return min learning rate
        if it > args.lr_decay_iters:
            return args.min_lr
        # 3) In between, use cosine decay down to min learning rate
        decay_ratio = (it - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return args.min_lr + coeff * (args.learning_rate - args.min_lr)

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split, data in [('train', train_data), ('val', val_data)]:
            losses = torch.zeros(args.eval_iters)
            for k in range(args.eval_iters):
                X, Y = get_batch(data, args.batch_size, config.block_size, device, device_type)
                with ctx:
                    _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    # 5. Training Loop
    print("\nStarting training loop...")
    best_val_loss = float('inf')
    t0 = time.time()

    for iter_num in range(1, args.max_iters + 1):
        # Update learning rate
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Evaluate periodically
        if iter_num % args.eval_interval == 0 or iter_num == 1:
            losses = estimate_loss()
            print(f"Step {iter_num:5d} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f} | LR: {lr:.2e}")
            if losses['val'] < best_val_loss or iter_num == args.max_iters:
                best_val_loss = losses['val']
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
                torch.save(checkpoint, ckpt_path)
                print(f"  >>> Saved best checkpoint to {ckpt_path} (val_loss: {best_val_loss:.4f})")

        # Forward & Backward with Gradient Accumulation
        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(args.gradient_accumulation_steps):
            X, Y = get_batch(train_data, args.batch_size, config.block_size, device, device_type)
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

        # Clip gradients & step optimizer
        if args.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        # Timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % args.log_interval == 0:
            tokens_per_sec = (args.batch_size * args.gradient_accumulation_steps * config.block_size) / max(dt, 1e-5)
            loss_val = loss.item() * args.gradient_accumulation_steps
            print(f"Iter {iter_num:5d} | Loss: {loss_val:.4f} | Time: {dt*1000:.1f}ms | {tokens_per_sec:,.0f} tok/s")

    print("\nTraining completed successfully!")
    print(f"Final checkpoint: {os.path.join(args.out_dir, 'ckpt.pt')}")


if __name__ == "__main__":
    main()
