"""
prepare_sft_data.py - 100% Human Chat / SFT Data Preparation
Sources (Verified Human-Authored, Zero Synthetic AI):
  1. databricks/databricks-dolly-15k (Authored by 5,000+ Databricks employees)
  2. OpenAssistant/oasst1 (Crowd-sourced multi-turn human conversations)

Outputs:
  - data/kira_sft/train.bin
  - data/kira_sft/val.bin
"""

import os
import argparse
import pickle
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare 100% human SFT data using custom tokenizer.")
    parser.add_argument("--tokenizer-path", type=str, default="data/kira/tokenizer.json",
                        help="Path to tokenizer.json trained by prepare_data.py")
    parser.add_argument("--meta-path", type=str, default="data/kira/meta.pkl",
                        help="Path to meta.pkl")
    parser.add_argument("--output-dir", type=str, default="data/kira_sft",
                        help="Directory to save train.bin and val.bin for SFT")
    parser.add_argument("--include-dolly", action="store_true", default=True,
                        help="Include databricks-dolly-15k (default: True)")
    parser.add_argument("--include-oasst", action="store_true", default=True,
                        help="Include OpenAssistant oasst1 (default: True)")
    parser.add_argument("--max-samples", type=int, default=30000,
                        help="Max total conversational samples to pack")
    return parser.parse_args()


def load_dolly():
    from datasets import load_dataset
    print("\nLoading databricks/databricks-dolly-15k (streaming mode)...")
    try:
        ds = load_dataset("databricks/databricks-dolly-15k", split="train", streaming=True)
    except Exception:
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    dialogues = []
    for item in ds:
        instruction = item.get("instruction", "").strip()
        context = item.get("context", "").strip()
        response = item.get("response", "").strip()
        
        if not instruction or not response:
            continue
            
        if context:
            user_msg = f"{instruction}\n\nContext:\n{context}"
        else:
            user_msg = instruction
            
        formatted = f"<|user|>\n{user_msg}\n<|bot|>\n{response}<|endoftext|>"
        dialogues.append(formatted)
        if len(dialogues) >= 15000:
            break
        
    print(f"Loaded {len(dialogues):,} samples from Dolly-15k.")
    return dialogues


def load_oasst1():
    from datasets import load_dataset
    print("\nLoading OpenAssistant/oasst1 (100% crowd-sourced human dialogue)...")
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    
    # Filter for English messages
    en_messages = {}
    for item in ds:
        if item.get("lang") == "en":
            en_messages[item["message_id"]] = item

    # Pair prompter and assistant replies
    dialogues = []
    for msg_id, item in en_messages.items():
        if item.get("role") == "assistant" and item.get("parent_id") in en_messages:
            parent = en_messages[item["parent_id"]]
            if parent.get("role") == "prompter":
                user_msg = parent.get("text", "").strip()
                bot_msg = item.get("text", "").strip()
                if user_msg and bot_msg:
                    formatted = f"<|user|>\n{user_msg}\n<|bot|>\n{bot_msg}<|endoftext|>"
                    dialogues.append(formatted)

    print(f"Loaded {len(dialogues):,} human dialogue turns from OASST1.")
    return dialogues


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        from tokenizers import Tokenizer
    except ImportError:
        raise ImportError("Please run: pip install tokenizers datasets")

    if not os.path.exists(args.tokenizer_path):
        raise FileNotFoundError(f"Custom tokenizer not found at {args.tokenizer_path}. Run prepare_data.py first!")

    print(f"Loading custom tokenizer from {args.tokenizer_path}...")
    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    # Copy meta.pkl to SFT dir for easy reference
    if os.path.exists(args.meta_path):
        with open(args.meta_path, "rb") as f:
            meta = pickle.load(f)
        with open(os.path.join(args.output_dir, "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)

    # Collect conversational data
    samples = []
    if args.include_dolly:
        samples.extend(load_dolly())
    if args.include_oasst:
        samples.extend(load_oasst1())

    # Shuffle and trim
    np.random.seed(42)
    np.random.shuffle(samples)
    if len(samples) > args.max_samples:
        samples = samples[:args.max_samples]

    print(f"\nTotal curated human conversation pairs: {len(samples):,}")

    # Split 95% train, 5% val
    num_val = max(1, int(len(samples) * 0.05))
    val_samples = samples[:num_val]
    train_samples = samples[num_val:]

    def tokenize_all(dataset_samples, desc):
        """Returns (token ids, loss mask). Mask is 1 only on the assistant reply (+ end token)."""
        all_ids, all_mask = [], []
        marker = "\n<|bot|>\n"
        for text in tqdm(dataset_samples, desc=desc):
            cut = text.find(marker)
            if cut < 0:
                continue
            cut += len(marker)
            prefix_ids = tokenizer.encode(text[:cut]).ids
            reply_ids = tokenizer.encode(text[cut:]).ids
            all_ids.extend(prefix_ids + reply_ids)
            all_mask.extend([0] * len(prefix_ids) + [1] * len(reply_ids))
        return np.array(all_ids, dtype=np.uint16), np.array(all_mask, dtype=np.uint8)

    val_tokens, val_mask = tokenize_all(val_samples, "Tokenizing SFT validation")
    train_tokens, train_mask = tokenize_all(train_samples, "Tokenizing SFT training")

    train_bin = os.path.join(args.output_dir, "train.bin")
    val_bin = os.path.join(args.output_dir, "val.bin")

    train_tokens.tofile(train_bin)
    val_tokens.tofile(val_bin)
    train_mask.tofile(os.path.join(args.output_dir, "train_mask.bin"))
    val_mask.tofile(os.path.join(args.output_dir, "val_mask.bin"))

    print("\n" + "=" * 60)
    print("  SFT DATA PREPARATION COMPLETE!")
    print(f"  Train SFT tokens: {len(train_tokens):,} (~{len(train_tokens)*2/1024**2:.2f} MB)")
    print(f"  Val SFT tokens:   {len(val_tokens):,} (~{len(val_tokens)*2/1024**2:.2f} MB)")
    print(f"  Files generated in {args.output_dir}:")
    print(f"    - {train_bin}")
    print(f"    - {val_bin}")
    print("=" * 60)


if __name__ == "__main__":
    main()
