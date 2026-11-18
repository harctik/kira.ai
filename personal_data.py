"""
personal_data.py - 100% Private Local Personal Data & RAG Engine
Scans and indexes your personal files (markdown, text, code, notes, pdfs) on your laptop.
Zero external API calls. 100% private and offline.
"""

import os
import re
import math
from typing import List, Dict, Tuple


class LocalDocumentChunk:
    def __init__(self, filepath: str, chunk_id: int, content: str):
        self.filepath = filepath
        self.chunk_id = chunk_id
        self.content = content
        self.tokens = self._tokenize(content)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())


class PersonalDataEngine:
    """Offline, local search engine for personal documents, notes, and code"""

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".py", ".json", ".csv", ".js", ".html", ".env", ".log"}

    def __init__(self, index_file: str = "memory/doc_index.json"):
        self.index_file = index_file
        self.chunks: List[LocalDocumentChunk] = []

    def index_directory(self, root_dir: str, chunk_size: int = 500) -> int:
        """Scan a personal directory and index all text and code files"""
        if not os.path.exists(root_dir):
            print(f"Directory not found: {root_dir}")
            return 0

        indexed_count = 0
        self.chunks.clear()

        for root, _, files in os.walk(root_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in self.SUPPORTED_EXTENSIONS:
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                            text = f.read()

                        if len(text.strip()) > 20:
                            # Split into overlapping chunks
                            words = text.split()
                            for i in range(0, len(words), chunk_size - 100):
                                chunk_text = " ".join(words[i:i + chunk_size])
                                self.chunks.append(LocalDocumentChunk(filepath, len(self.chunks), chunk_text))
                            indexed_count += 1
                    except Exception as e:
                        print(f"Skipping {filepath}: {e}")

        print(f"Indexed {indexed_count} personal files into {len(self.chunks)} searchable chunks.")
        return len(self.chunks)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        """Search personal indexed documents for matching context"""
        query_tokens = set(re.findall(r"\w+", query.lower()))
        if not query_tokens or not self.chunks:
            return []

        scores: List[Tuple[float, LocalDocumentChunk]] = []

        for chunk in self.chunks:
            # BM25-style term frequency overlap
            match_count = sum(1 for tok in query_tokens if tok in chunk.tokens)
            if match_count > 0:
                score = match_count / math.sqrt(len(chunk.tokens) + 1)
                scores.append((score, chunk))

        scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, chunk in scores[:top_k]:
            results.append({
                "score": f"{score:.3f}",
                "file": chunk.filepath,
                "snippet": chunk.content[:400] + ("..." if len(chunk.content) > 400 else "")
            })

        return results
