"""
Security tests for the Kira bridge server: auth, shell gating, input validation.
"""

import json
import os
import sys
import threading
import urllib.error
import urllib.request

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import server
from evolution import EvolutionEngine
from http.server import HTTPServer


@pytest.fixture()
def base_url(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "evolution", EvolutionEngine(memory_dir=str(tmp_path / "mem")))
    monkeypatch.setattr(server, "AUTH_TOKEN", "test-token")
    monkeypatch.setattr(server, "ALLOW_SHELL", False)
    httpd = HTTPServer(("127.0.0.1", 0), server.KiraServerHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def post(url, payload, token=None, raw=None):
    data = raw if raw is not None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Kira-Token"] = token
    req = urllib.request.Request(url + "/api/chat", data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_default_bind_is_localhost(monkeypatch):
    monkeypatch.delenv("KIRA_HOST", raising=False)
    captured = {}

    class FakeHTTPD:
        def __init__(self, addr, handler):
            captured["addr"] = addr

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            pass

    monkeypatch.setattr(server, "HTTPServer", FakeHTTPD)
    server.start_server(port=9999)
    assert captured["addr"] == ("127.0.0.1", 9999)


def test_chat_without_token_is_rejected(base_url):
    status, body = post(base_url, {"message": "status"})
    assert status == 401


def test_chat_with_wrong_token_is_rejected(base_url):
    status, _ = post(base_url, {"message": "status"}, token="nope")
    assert status == 401


def test_status_endpoint_requires_token(base_url):
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(base_url + "/api/status")
    assert exc.value.code == 401


def test_run_command_is_disabled_by_default(base_url, tmp_path):
    marker = tmp_path / "pwned.txt"
    status, body = post(base_url, {"message": f"run: echo hi > {marker}"}, token="test-token")
    assert status == 200
    assert "disabled" in body["reply"].lower()
    assert body["tool_output"] is None
    assert not marker.exists()


def test_run_command_works_when_explicitly_enabled(base_url, monkeypatch):
    monkeypatch.setattr(server, "ALLOW_SHELL", True)
    status, body = post(base_url, {"message": "run: echo kira-ok"}, token="test-token")
    assert status == 200
    assert "kira-ok" in body["tool_output"]


def test_invalid_json_returns_400(base_url):
    status, _ = post(base_url, None, token="test-token", raw=b"{not json")
    assert status == 400


def test_oversized_body_returns_413(base_url):
    # Declare an oversized body without sending it; the server must refuse before reading.
    import http.client

    host, port = base_url.replace("http://", "").split(":")
    conn = http.client.HTTPConnection(host, int(port))
    conn.putrequest("POST", "/api/chat")
    conn.putheader("X-Kira-Token", "test-token")
    conn.putheader("Content-Length", str(server.MAX_BODY_BYTES + 1))
    conn.endheaders()
    resp = conn.getresponse()
    assert resp.status == 413
    conn.close()


def test_ui_does_not_use_innerhtml_for_replies():
    assert "innerHTML = data.reply" not in server.HTML_UI
    assert "innerHTML +=" not in server.HTML_UI
