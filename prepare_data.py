"""
prepare_data.py - 100% Human Data Preparation & Custom Tokenizer Trainer
Strict "No Other Model" rule: Zero synthetic data, zero AI filtering.
Sources:
  - English Wikipedia (wikimedia/wikipedia)
  - Optional: Project Gutenberg (human public-domain books)

Outputs:
  - tokenizer.json (Custom BPE tokenizer trained from scratch)
  - meta.pkl (Vocab mappings and metadata for nanoGPT)
  - train.bin & val.bin (uint16 binary token arrays ready for nanoGPT training)
"""

import os
import sys
import argparse
import pickle
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare pure human pre-training data and train tokenizer.")
    parser.add_argument("--docs-per-source", type=int, default=50000,
                        help="Number of Wikipedia articles to fetch (default: 50,000 -> ~100M-150M tokens)")
    parser.add_argument("--include-books", action="store_true",
                        help="Also fetch public-domain books from PG-19 / Project Gutenberg")
    parser.add_argument("--books-count", type=int, default=500,
                        help="Number of Gutenberg books to fetch if --include-books is set")
    parser.add_argument("--vocab-size", type=int, default=8192,
                        help="Vocabulary size for custom BPE tokenizer (default: 8192, max 65535 for uint16)")
    parser.add_argument("--output-dir", type=str, default="data/kira",
                        help="Directory to save tokenizer, meta.pkl, and .bin files")
    parser.add_argument("--val-ratio", type=float, default=0.05,
                        help="Validation set ratio (default: 0.05)")
    return parser.parse_args()


def train_bpe_tokenizer(text_iterator, vocab_size, output_dir):
    """
    Trains a custom Byte-Pair Encoding (BPE) tokenizer from scratch.
    Does NOT use GPT-2 or GPT-4 tokenizer files.
    """
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    except ImportError:
        raise ImportError("Please install tokenizers: pip install tokenizers")

    print(f"\n[1/3] Training custom Byte-Level BPE tokenizer (vocab_size={vocab_size})...")
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    special_tokens = [
        "<|endoftext|>",
        "<|pad|>",
        "<|user|>",
        "<|bot|>",
        "<|system|>"
    ]

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet()
    )

    tokenizer.train_from_iterator(text_iterator, trainer=trainer)

    os.makedirs(output_dir, exist_ok=True)
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"Saved custom tokenizer to {tokenizer_path}")

    # Build meta.pkl for nanoGPT compatibility
    vocab = tokenizer.get_vocab()
    meta = {
        "vocab_size": len(vocab),
        "itos": {idx: token for token, idx in vocab.items()},
        "stoi": vocab,
        "special_tokens": {tok: tokenizer.token_to_id(tok) for tok in special_tokens}
    }
    meta_path = os.path.join(output_dir, "meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"Saved nanoGPT metadata to {meta_path} (actual vocab: {len(vocab)})")

    return tokenizer, meta


def fetch_wikipedia_stream(num_docs):
    """
    Stream Wikipedia articles from Hugging Face datasets.
    Streaming mode avoids downloading the entire 20GB+ compressed dump.
    """
    from datasets import load_dataset
    print(f"\nConnecting to Hugging Face: wikimedia/wikipedia (streaming mode)...")
    
    # Try current english split; fallback to 20231101.en if date unspecified
    dataset = None
    configs = ["20231101.en", "20220301.en"]
    for cfg in configs:
        try:
            print(f"Trying config '{cfg}'...")
            dataset = load_dataset("wikimedia/wikipedia", cfg, split="train", streaming=True, trust_remote_code=True)
            break
        except Exception as e:
            print(f"Config {cfg} failed: {e}. Trying fallback...")

    if dataset is None:
        # Generic fallback
        dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True, trust_remote_code=True)

    print(f"Streaming {num_docs:,} articles...")
    docs = []
    pbar = tqdm(total=num_docs, desc="Fetching Wikipedia")
    for item in dataset:
        text = item.get("text", "").strip()
        if len(text) > 200:  # Skip trivial stubs
            docs.append(text)
            pbar.update(1)
            if len(docs) >= num_docs:
                break
    pbar.close()
    return docs


def fetch_gutenberg_stream(num_books):
    """
    Optionally stream public domain books from PG-19.
    100% human-authored pre-1919 literature.
    """
    from datasets import load_dataset
    print(f"\nStreaming {num_books:,} books from PG-19 (Project Gutenberg)...")
    try:
        dataset = load_dataset("deepmind/pg19", split="train", streaming=True, trust_remote_code=True)
    except Exception:
        dataset = load_dataset("emozilla/pg19", split="train", streaming=True, trust_remote_code=True)

    books = []
    pbar = tqdm(total=num_books, desc="Fetching Gutenberg")
    for item in dataset:
        text = item.get("text", "").strip()
        if len(text) > 1000:
            books.append(text)
            pbar.update(1)
            if len(books) >= num_books:
                break
    pbar.close()
    return books


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  Kira LLM - 100% Human Data Preparation")
    print("  Strict 'No Other Model' Rule: No Synthetic, No AI-Filtered")
    print("=" * 60)

    # 1. Fetch text data
    all_texts = []
    wiki_texts = fetch_wikipedia_stream(args.docs_per_source)
    all_texts.extend(wiki_texts)

    if args.include_books:
        book_texts = fetch_gutenberg_stream(args.books_count)
        all_texts.extend(book_texts)

    total_chars = sum(len(t) for t in all_texts)
    print(f"\nCollected {len(all_texts):,} documents ({total_chars / 1e6:.2f} MB of raw text).")

    # 2. Train tokenizer on a representative sample of documents
    sample_size = min(len(all_texts), 25000)
    sample_iterator = (all_texts[i] for i in range(sample_size))
    tokenizer, meta = train_bpe_tokenizer(sample_iterator, args.vocab_size, args.output_dir)

    eot_id = meta["special_tokens"]["<|endoftext|>"]

    # 3. Tokenize all texts into train and val splits
    print(f"\n[2/3] Tokenizing {len(all_texts):,} documents with custom tokenizer...")
    num_val = max(1, int(len(all_texts) * args.val_ratio))
    val_docs = all_texts[:num_val]
    train_docs = all_texts[num_val:]

    def tokenize_and_pack(docs, desc):
        all_token_ids = []
        for doc in tqdm(docs, desc=desc):
            enc = tokenizer.encode(doc)
            all_token_ids.extend(enc.ids)
            all_token_ids.append(eot_id)  # Separate documents with <|endoftext|>
        return np.array(all_token_ids, dtype=np.uint16)

    val_tokens = tokenize_and_pack(val_docs, "Tokenizing validation set")
    train_tokens = tokenize_and_pack(train_docs, "Tokenizing training set")

    # 4. Save .bin files
    train_path = os.path.join(args.output_dir, "train.bin")
    val_path = os.path.join(args.output_dir, "val.bin")

    print(f"\n[3/3] Saving binary files for nanoGPT...")
    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)

    print("\n" + "=" * 60)
    print("  DATA PREPARATION COMPLETE!")
    print(f"  Training tokens:   {len(train_tokens):,} tokens (~{len(train_tokens) * 2 / 1024**2:.2f} MB)")
    print(f"  Validation tokens: {len(val_tokens):,} tokens (~{len(val_tokens) * 2 / 1024**2:.2f} MB)")
    print(f"  Total tokens:      {len(train_tokens) + len(val_tokens):,}")
    print(f"  Files generated:")
    print(f"    - {train_path}")
    print(f"    - {val_path}")
    print(f"    - {os.path.join(args.output_dir, 'tokenizer.json')}")
    print(f"    - {os.path.join(args.output_dir, 'meta.pkl')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
