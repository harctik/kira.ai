"""
retrieval.py - Free, dependency-free local retrieval for Kira (chunking + BM25).

No embeddings model, no API, no network. Works with any generation backend.
"""

import json
import math
import os
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_STOPWORDS = frozenset(
    "a an and are as at be but by for from has have i in is it its of on or that the this to was were will with "
    "you your what who how why when where which do does did can could should would about into than then so".split()
)
TEXT_EXTENSIONS = {".txt", ".md", ".py", ".rst", ".json", ".yml", ".yaml", ".toml", ".js", ".ts", ".html", ".css"}


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens; splits snake_case and keeps identifiers searchable."""
    out = []
    for tok in _TOKEN_RE.findall(text.lower()):
        if tok in _STOPWORDS:
            continue
        out.append(tok)
        if "_" in tok:
            out.extend(p for p in tok.split("_") if p and p not in _STOPWORDS)
    return out


def chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks, preferring paragraph then sentence boundaries."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    overlap = max(0, min(overlap, max_chars // 2))
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []

    pieces: List[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            pieces.append(para)
        else:
            pieces.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", para) if s.strip())

    chunks: List[str] = []
    current = ""
    for piece in pieces:
        while len(piece) > max_chars:  # hard-split an oversized sentence/line
            if current:
                chunks.append(current)
                current = ""
            chunks.append(piece[:max_chars])
            piece = piece[max_chars - overlap:]
        if current and len(current) + 1 + len(piece) > max_chars:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = (tail + " " + piece).strip() if tail else piece
        else:
            current = (current + "\n" + piece).strip() if current else piece
    if current:
        chunks.append(current)
    return chunks


class BM25Index:
    """Okapi BM25 over a list of documents (each a dict with 'text' plus any metadata)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs: List[Dict] = []
        self._tf: List[Counter] = []
        self._df: Counter = Counter()
        self._len_total = 0

    def __len__(self) -> int:
        return len(self.docs)

    def add(self, doc: Dict) -> None:
        tokens = tokenize(doc["text"])
        tf = Counter(tokens)
        self.docs.append(doc)
        self._tf.append(tf)
        self._df.update(tf.keys())
        self._len_total += len(tokens)

    def search(self, query: str, k: int = 3, min_score: float = 0.0) -> List[Dict]:
        q_tokens = list(dict.fromkeys(tokenize(query)))
        n = len(self.docs)
        if not q_tokens or n == 0:
            return []
        avgdl = (self._len_total / n) or 1.0
        scored = []
        for i, tf in enumerate(self._tf):
            dl = sum(tf.values())
            score = 0.0
            for t in q_tokens:
                f = tf.get(t)
                if not f:
                    continue
                idf = math.log(1 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5))
                score += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / avgdl))
            if score > min_score:
                scored.append((score, i))
        scored.sort(key=lambda x: -x[0])
        return [dict(self.docs[i], score=round(s, 4)) for s, i in scored[:k]]


class DocumentIndex:
    """Persistent local document index: add files/folders, search by meaning-of-words (BM25)."""

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.index = BM25Index()
        self._sources = set()
        if path and os.path.exists(path):
            self._load()

    def __len__(self) -> int:
        return len(self.index)

    def add_text(self, text: str, source: str, max_chars: int = 800) -> int:
        n = 0
        for i, chunk in enumerate(chunk_text(text, max_chars=max_chars)):
            self.index.add({"text": chunk, "source": source, "chunk": i})
            n += 1
        self._sources.add(source)
        return n

    def add_file(self, filepath: str, max_bytes: int = 2_000_000) -> int:
        try:
            if os.path.getsize(filepath) > max_bytes:
                return 0
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return self.add_text(f.read(), source=os.path.abspath(filepath))
        except OSError:
            return 0

    def add_directory(self, root: str, extensions: Iterable[str] = TEXT_EXTENSIONS) -> Dict[str, int]:
        exts = {e.lower() for e in extensions}
        files = chunks = 0
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in {"__pycache__", "node_modules", "venv"}]
            for name in filenames:
                if os.path.splitext(name)[1].lower() in exts:
                    added = self.add_file(os.path.join(dirpath, name))
                    if added:
                        files += 1
                        chunks += added
        return {"files": files, "chunks": chunks}

    def search(self, query: str, k: int = 3) -> List[Dict]:
        return self.index.search(query, k=k)

    def save(self) -> None:
        if not self.path:
            raise ValueError("No path set for this index")
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.index.docs, f)

    def _load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                docs = json.load(f)
        except (OSError, ValueError):
            return
        for d in docs:
            if isinstance(d, dict) and "text" in d:
                self.index.add(d)
                self._sources.add(d.get("source", ""))
