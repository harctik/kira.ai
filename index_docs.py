"""
index_docs.py - Index local files/folders for Kira to answer from.

Usage: python index_docs.py PATH [PATH ...] [--index memory/index.json]
"""

import argparse
import os

from retrieval import DocumentIndex


def main():
    ap = argparse.ArgumentParser(description="Index files or folders into Kira's local search index.")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--index", default=os.path.join("memory", "index.json"))
    args = ap.parse_args()

    idx = DocumentIndex(args.index)
    for p in args.paths:
        if os.path.isdir(p):
            stats = idx.add_directory(p)
            print(f"{p}: {stats['files']} files, {stats['chunks']} chunks")
        elif os.path.isfile(p):
            print(f"{p}: {idx.add_file(p)} chunks")
        else:
            print(f"{p}: not found")
    idx.save()
    print(f"Saved {len(idx)} chunks to {args.index}")


if __name__ == "__main__":
    main()
