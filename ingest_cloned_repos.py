"""
ingest_cloned_repos.py - Ingest & Tokenize Cloned Human GitHub Repositories
Extracts clean Python, PyTorch, algorithm, and system code from:
  - TheAlgorithms_Python
  - pytorch_examples
  - learn_python
  - nanoGPT
  - dolly

Generates:
  - data/repos_corpus.txt (Clean raw text corpus of all human code)
  - data/kira_code/train.bin & val.bin (if tokenizer is available)
"""

import os
import argparse
import pickle
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Extract and tokenize code from cloned repos.")
    parser.add_argument("--repos-dir", type=str, default="data/repos", help="Directory where repos are cloned")
    parser.add_argument("--output-corpus", type=str, default="data/repos_corpus.txt", help="Output text corpus file")
    parser.add_argument("--output-tokens-dir", type=str, default="data/kira_code", help="Output directory for binary tokens")
    parser.add_argument("--tokenizer-path", type=str, default="data/kira/tokenizer.json", help="Path to custom tokenizer.json")
    return parser.parse_args()


IGNORE_DIRS = {".git", ".github", "__pycache__", ".pytest_cache", "venv", ".venv", "build", "dist"}
ALLOWED_EXTS = {".py", ".md", ".jsonl"}


def main():
    args = parse_args()
    print("=" * 60)
    print("  Kira LLM - Ingesting Cloned Human GitHub Repositories")
    print(f"  Source Directory: {args.repos_dir}")
    print("=" * 60)

    if not os.path.exists(args.repos_dir):
        raise FileNotFoundError(f"Directory {args.repos_dir} does not exist!")

    extracted_files = []
    total_chars = 0

    print("\nScanning repositories...")
    for root, dirs, files in os.walk(args.repos_dir):
        # Prune ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in ALLOWED_EXTS:
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, args.repos_dir)
                try:
                    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()

                    # Skip trivial files (<50 chars) or minified code
                    if len(content.strip()) > 50 and not file.endswith(".min.js"):
                        header = f"\n\n# ============================================================\n"
                        header += f"# FILE: {rel_path}\n"
                        header += f"# ============================================================\n\n"
                        extracted_files.append(header + content)
                        total_chars += len(content)
                except Exception as e:
                    print(f"Skipping {rel_path}: {e}")

    print(f"\nExtracted {len(extracted_files):,} clean human code files!")
    print(f"Total raw text size: {total_chars / 1e6:.2f} MB (~{total_chars / 4 / 1e6:.1f} Million tokens)")

    # Save to unified corpus file
    os.makedirs(os.path.dirname(args.output_corpus), exist_ok=True)
    with open(args.output_corpus, "w", encoding="utf-8") as f:
        for block in extracted_files:
            f.write(block)
    print(f"Saved complete corpus to: {args.output_corpus}")

    # Tokenize if tokenizer.json exists
    if os.path.exists(args.tokenizer_path):
        try:
            from tokenizers import Tokenizer
            print(f"\nTokenizing with custom tokenizer ({args.tokenizer_path})...")
            tokenizer = Tokenizer.from_file(args.tokenizer_path)
            eot_id = tokenizer.token_to_id("<|endoftext|>") or 0

            # 95% train / 5% val
            num_val = max(1, int(len(extracted_files) * 0.05))
            val_files = extracted_files[:num_val]
            train_files = extracted_files[num_val:]

            def pack_tokens(file_list, desc):
                tokens = []
                for content in tqdm(file_list, desc=desc):
                    enc = tokenizer.encode(content)
                    tokens.extend(enc.ids)
                    tokens.append(eot_id)
                return np.array(tokens, dtype=np.uint16)

            val_tokens = pack_tokens(val_files, "Tokenizing validation")
            train_tokens = pack_tokens(train_files, "Tokenizing training")

            os.makedirs(args.output_tokens_dir, exist_ok=True)
            train_bin = os.path.join(args.output_tokens_dir, "train.bin")
            val_bin = os.path.join(args.output_tokens_dir, "val.bin")

            train_tokens.tofile(train_bin)
            val_tokens.tofile(val_bin)

            print("\n" + "=" * 60)
            print("  CODE REPO TOKENIZATION COMPLETE!")
            print(f"  Training tokens:   {len(train_tokens):,}")
            print(f"  Validation tokens: {len(val_tokens):,}")
            print(f"  Saved to: {train_bin} & {val_bin}")
            print("=" * 60)

        except ImportError:
            print("Note: 'tokenizers' not installed locally. Run in Colab or install with 'pip install tokenizers'.")
    else:
        print(f"\nNote: Tokenizer not found at {args.tokenizer_path}. Run prepare_data.py first to tokenize.")


if __name__ == "__main__":
    main()
