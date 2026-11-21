"""
Tests for retrieval, memory recall, and the assistant.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from assistant import Assistant
from backends import ExtractiveBackend
from evolution import EvolutionEngine
from retrieval import BM25Index, DocumentIndex, chunk_text, tokenize


def test_tokenize_splits_snake_case_and_drops_stopwords():
    toks = tokenize("The get_system_status function")
    assert "get_system_status" in toks and "status" in toks and "the" not in toks


def test_chunk_text_respects_limit_and_keeps_content():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 40 for i in range(10))
    chunks = chunk_text(text, max_chars=300)
    assert len(chunks) > 1
    assert all(len(c) <= 300 + 100 for c in chunks)
    assert "Paragraph 9" in " ".join(chunks)


def test_chunk_text_handles_empty_and_giant_line():
    assert chunk_text("   ") == []
    assert all(len(c) <= 100 for c in chunk_text("x" * 1000, max_chars=100))


def test_bm25_ranks_relevant_doc_first():
    idx = BM25Index()
    idx.add({"text": "Cats purr and chase mice around the house"})
    idx.add({"text": "The transformer uses rotary position embeddings"})
    idx.add({"text": "Bananas are yellow fruit"})
    hits = idx.search("rotary embeddings", k=2)
    assert hits and "rotary" in hits[0]["text"]
    assert idx.search("zzzz nothing") == []


def test_document_index_roundtrip(tmp_path):
    doc = tmp_path / "notes.md"
    doc.write_text("Kira uses grouped query attention to shrink the KV cache.", encoding="utf-8")
    path = str(tmp_path / "idx.json")
    idx = DocumentIndex(path)
    assert idx.add_directory(str(tmp_path))["files"] == 1
    idx.save()
    reloaded = DocumentIndex(path)
    assert len(reloaded) == len(idx) > 0
    assert "KV cache" in reloaded.search("kv cache")[0]["text"]


def test_recall_finds_taught_facts_and_past_chats(tmp_path):
    ev = EvolutionEngine(memory_dir=str(tmp_path / "m"))
    ev.teach_fact("note", "deadline", "project deadline is Friday")
    ev.log_interaction("what editor do I use", "You use Neovim")
    assert "Friday" in ev.recall("when is the deadline")[0]["text"]
    assert "Neovim" in ev.recall("which editor")[0]["text"]


def test_assistant_answers_from_docs_and_memory(tmp_path):
    ev = EvolutionEngine(memory_dir=str(tmp_path / "m"))
    ev.teach_fact("note", "pet", "my dog is called Biscuit")
    idx = DocumentIndex()
    idx.add_text("Kira tokenizer has 8192 BPE tokens.", source="readme")
    a = Assistant(ev, idx, ExtractiveBackend())
    assert "Biscuit" in a.ask("what is my dog called")["answer"]
    res = a.ask("how many tokens in the tokenizer")
    assert "8192" in res["answer"] and "readme" in res["sources"]


def test_assistant_says_unknown_when_nothing_found(tmp_path):
    a = Assistant(EvolutionEngine(memory_dir=str(tmp_path / "m")), DocumentIndex(), ExtractiveBackend())
    assert "don't have" in a.ask("quantum banana")["answer"]


def test_assistant_survives_broken_backend(tmp_path):
    class Broken:
        name = "broken"

        def generate(self, prompt, max_new_tokens=200):
            raise ConnectionError("down")

    ev = EvolutionEngine(memory_dir=str(tmp_path / "m"))
    ev.teach_fact("note", "x", "the secret word is plum")
    out = Assistant(ev, DocumentIndex(), Broken()).ask("secret word")
    assert "plum" in out["answer"] and "failed" in out["answer"]


def test_reindexing_replaces_instead_of_duplicating(tmp_path):
    doc = tmp_path / "a.md"
    doc.write_text("alpha beta gamma", encoding="utf-8")
    idx = DocumentIndex()
    idx.add_file(str(doc))
    idx.add_file(str(doc))
    assert len(idx) == 1
    doc.write_text("delta epsilon", encoding="utf-8")
    idx.add_file(str(doc))
    assert len(idx) == 1 and idx.search("alpha") == []


def test_directory_scan_skips_memory_data_and_own_index(tmp_path):
    (tmp_path / "keep.md").write_text("keep this", encoding="utf-8")
    for d in ("memory", "data"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "skip.md").write_text("skip this", encoding="utf-8")
    idx = DocumentIndex(str(tmp_path / "idx.json"))
    assert idx.add_directory(str(tmp_path))["files"] == 1
