"""
mcp_tools.py - Expose Kira's laptop tools as an MCP server over stdio (stdlib only).

Safe by default: only read-only tools are enabled, and file access is confined to a
workspace root. Enable more with KIRA_MCP_ALLOW=read_file,list_directory,write_file,...
Shell and app-launching tools are never exposed. This is an allowlist, not a sandbox.

Run:  python mcp_tools.py            (root = current directory, or KIRA_MCP_ROOT)
"""

import json
import os
import sys
from typing import Callable, Dict, Optional

from tools import LaptopTools

DEFAULT_ALLOW = {"get_system_status", "read_file", "list_directory"}
NEVER_EXPOSE = {"execute_shell", "open_url_or_app"}
PROTOCOL_VERSION = "2024-11-05"


def _in_root(root: str, path: str) -> str:
    full = os.path.realpath(os.path.join(root, path))
    base = os.path.realpath(root)
    if os.path.commonpath([full, base]) != base:
        raise PermissionError("path is outside the workspace root")
    return full


class ToolRegistry:
    def __init__(self, root: str = ".", allow: Optional[set] = None):
        self.root = os.path.realpath(root)
        allow = set(allow) if allow is not None else set(DEFAULT_ALLOW)
        self.allow = allow - NEVER_EXPOSE
        self._impl: Dict[str, Callable] = {
            "get_system_status": lambda a: LaptopTools.get_system_status(),
            "read_file": lambda a: LaptopTools.read_file(_in_root(self.root, a["path"]), int(a.get("max_lines", 200))),
            "list_directory": lambda a: LaptopTools.list_directory(_in_root(self.root, a.get("path", "."))),
            "write_file": lambda a: LaptopTools.write_file(_in_root(self.root, a["path"]), a["content"]),
        }
        self._schemas = {
            "get_system_status": ("CPU, RAM and battery status.", {}, []),
            "read_file": ("Read a text file inside the workspace.",
                          {"path": {"type": "string"}, "max_lines": {"type": "integer"}}, ["path"]),
            "list_directory": ("List a directory inside the workspace.", {"path": {"type": "string"}}, []),
            "write_file": ("Write a text file inside the workspace.",
                           {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
        }

    def names(self):
        return sorted(n for n in self._impl if n in self.allow)

    def list_tools(self):
        out = []
        for n in self.names():
            desc, props, req = self._schemas[n]
            out.append({"name": n, "description": desc,
                        "inputSchema": {"type": "object", "properties": props, "required": req}})
        return out

    def call(self, name: str, arguments: dict):
        if name not in self.allow or name not in self._impl:
            raise PermissionError(f"tool '{name}' is not enabled")
        result = self._impl[name](arguments or {})
        return result if isinstance(result, str) else json.dumps(result, indent=2, default=str)


def handle(registry: ToolRegistry, msg: dict) -> Optional[dict]:
    """Handle one JSON-RPC message; returns a response dict, or None for notifications."""
    mid, method = msg.get("id"), msg.get("method")
    if mid is None:
        return None  # notification

    def ok(result):
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def err(code, text):
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": text}}

    if method == "initialize":
        return ok({"protocolVersion": PROTOCOL_VERSION, "capabilities": {"tools": {}},
                   "serverInfo": {"name": "kira-tools", "version": "0.1.0"}})
    if method == "ping":
        return ok({})
    if method == "tools/list":
        return ok({"tools": registry.list_tools()})
    if method == "tools/call":
        params = msg.get("params") or {}
        try:
            text = registry.call(params.get("name", ""), params.get("arguments") or {})
            return ok({"content": [{"type": "text", "text": text}], "isError": False})
        except Exception as e:  # tool failures are reported in-band per MCP
            return ok({"content": [{"type": "text", "text": f"{type(e).__name__}: {e}"}], "isError": True})
    return err(-32601, f"method not found: {method}")


def serve(registry: ToolRegistry, stdin=sys.stdin, stdout=sys.stdout):
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            resp = handle(registry, json.loads(line))
        except ValueError:
            resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        if resp is not None:
            stdout.write(json.dumps(resp) + "\n")
            stdout.flush()


if __name__ == "__main__":
    allow = os.environ.get("KIRA_MCP_ALLOW")
    serve(ToolRegistry(os.environ.get("KIRA_MCP_ROOT", "."),
                       set(a.strip() for a in allow.split(",")) if allow else None))
