"""
Tests for evaluation, the MCP tool server, and the multi-agent orchestrator.
"""

import io
import json
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import evaluate
import mcp_tools
from agents import Orchestrator, split_questions
from assistant import Assistant
from backends import ExtractiveBackend
from evolution import EvolutionEngine
from retrieval import DocumentIndex


@pytest.fixture()
def assistant(tmp_path):
    ev = EvolutionEngine(memory_dir=str(tmp_path / "m"))
    idx = DocumentIndex()
    idx.add_text("The auth header is X-Kira-Token.", source="server")
    idx.add_text("Memory is stored in interactions.jsonl.", source="evo")
    idx.add_text("Bananas are yellow.", source="noise")
    return Assistant(ev, idx, ExtractiveBackend())


def test_token_f1_and_contains():
    assert evaluate.token_f1("auth header token", "auth header token") == 1.0
    assert evaluate.token_f1("apple", "banana") == 0.0
    assert evaluate.contains_expected("The  Header IS X-Kira-Token", "x-kira-token")


def test_retrieval_metrics(assistant):
    golden = [{"question": "what is the auth header", "expected": "X-Kira-Token"},
              {"question": "where is memory stored", "expected": "interactions.jsonl"},
              {"question": "unanswerable zzz", "expected": "nothing"}]
    r = evaluate.evaluate_retrieval(assistant.index, golden, k=2)
    assert r["hit@2"] == pytest.approx(2 / 3) and r["mrr"] == pytest.approx(2 / 3)
    a = evaluate.evaluate_answers(assistant, golden[:2])
    assert a["contains_expected"] == 1.0


def test_eval_cli_exit_code(tmp_path, assistant):
    idx_path = str(tmp_path / "i.json")
    assistant.index.path = idx_path
    assistant.index.save()
    g = tmp_path / "g.json"
    g.write_text(json.dumps([{"question": "auth header", "expected": "X-Kira-Token"}]))
    assert evaluate.main([str(g), "--index", idx_path, "--min-hit", "1.0"]) == 0
    g.write_text(json.dumps([{"question": "auth header", "expected": "nope"}]))
    assert evaluate.main([str(g), "--index", idx_path, "--min-hit", "1.0"]) == 1


def test_mcp_lists_only_readonly_by_default(tmp_path):
    reg = mcp_tools.ToolRegistry(str(tmp_path))
    assert reg.names() == ["get_system_status", "list_directory", "read_file"]
    assert "write_file" not in reg.names()


def test_mcp_never_exposes_shell_even_if_allowed(tmp_path):
    reg = mcp_tools.ToolRegistry(str(tmp_path), allow={"execute_shell", "open_url_or_app", "read_file"})
    assert reg.names() == ["read_file"]
    with pytest.raises(PermissionError):
        reg.call("execute_shell", {"command": "echo hi"})


def test_mcp_blocks_path_escape(tmp_path):
    (tmp_path / "ok.txt").write_text("hello")
    reg = mcp_tools.ToolRegistry(str(tmp_path))
    assert "hello" in reg.call("read_file", {"path": "ok.txt"})
    for bad in ["../secret.txt", os.path.abspath(os.sep)]:
        with pytest.raises((PermissionError, ValueError)):
            reg.call("read_file", {"path": bad})


def test_mcp_write_requires_opt_in(tmp_path):
    reg = mcp_tools.ToolRegistry(str(tmp_path), allow={"write_file"})
    reg.call("write_file", {"path": "a.txt", "content": "x"})
    assert (tmp_path / "a.txt").read_text() == "x"
    with pytest.raises(PermissionError):
        mcp_tools.ToolRegistry(str(tmp_path)).call("write_file", {"path": "b.txt", "content": "x"})


def test_mcp_protocol_over_stdio(tmp_path):
    (tmp_path / "f.txt").write_text("data")
    msgs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "read_file", "arguments": {"path": "f.txt"}}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "execute_shell", "arguments": {}}},
        {"jsonrpc": "2.0", "id": 5, "method": "nope"},
    ]
    stdin = io.StringIO("\n".join(json.dumps(m) for m in msgs) + "\nnot json\n")
    out = io.StringIO()
    mcp_tools.serve(mcp_tools.ToolRegistry(str(tmp_path)), stdin, out)
    r = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [x["id"] for x in r] == [1, 2, 3, 4, 5, None]  # notification got no reply
    assert r[0]["result"]["serverInfo"]["name"] == "kira-tools"
    assert len(r[1]["result"]["tools"]) == 3
    assert r[2]["result"]["content"][0]["text"] == "data"
    assert r[3]["result"]["isError"] is True
    assert r[4]["error"]["code"] == -32601 and r[5]["error"]["code"] == -32700


def test_split_questions():
    qs = split_questions("What is the auth header? How is memory stored")
    assert len(qs) == 2


def test_orchestrator_runs_end_to_end(assistant):
    out = Orchestrator(assistant).run("What is the auth header? Where is memory stored")
    assert len(out["plan"]) == 2
    assert "X-Kira-Token" in out["final"] and "interactions.jsonl" in out["final"]
