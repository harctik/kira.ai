"""
assistant.py - Memory + retrieval assistant: ask() = recall memory + search docs + generate.
"""

from typing import Dict, Optional

from backends import ExtractiveBackend
from retrieval import DocumentIndex


class Assistant:
    def __init__(self, evolution, index: Optional[DocumentIndex] = None, backend=None, k: int = 3):
        self.evolution = evolution
        self.index = index or DocumentIndex()
        self.backend = backend or ExtractiveBackend()
        self.k = k

    def build_prompt(self, question: str, memories, docs) -> str:
        parts = [f"[memory] {m['text']}" for m in memories]
        parts += [f"[{d['source']}] {d['text']}" for d in docs]
        context = "\n---\n".join(parts)
        return (
            "Answer using only the context. If it is not there, say you don't know.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )

    def ask(self, question: str) -> Dict:
        memories = self.evolution.recall(question, k=self.k)
        docs = self.index.search(question, k=self.k)
        prompt = self.build_prompt(question, memories, docs)
        try:
            answer = self.backend.generate(prompt)
        except Exception as e:  # backend down/missing: degrade to extractive, never crash the server
            answer = ExtractiveBackend().generate(prompt) + f"\n(backend {self.backend.name} failed: {type(e).__name__})"
        return {
            "answer": answer,
            "sources": [d["source"] for d in docs] + ["memory"] * len(memories),
        }
