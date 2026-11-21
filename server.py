"""
server.py - Kira Mobile & Laptop Bridge Server
Runs locally on your laptop. Accessible from both laptop browser and mobile phone on Wi-Fi.
Zero extra dependencies required (uses Python standard library http.server).
"""

import os
import json
import hmac
import socket
import secrets
from http.server import HTTPServer, BaseHTTPRequestHandler
from tools import LaptopTools
from evolution import EvolutionEngine

evolution = EvolutionEngine(memory_dir="memory")

# Security model:
#  - Binds to 127.0.0.1 unless KIRA_HOST is set (e.g. KIRA_HOST=0.0.0.0 for phone access).
#  - Every /api/* call needs the secret token (header X-Kira-Token). Set KIRA_TOKEN or one is
#    generated per start and printed in the console.
#  - Raw shell execution ("run:") is OFF unless KIRA_ALLOW_SHELL=1.
MAX_BODY_BYTES = 16 * 1024
AUTH_TOKEN = os.environ.get("KIRA_TOKEN") or secrets.token_urlsafe(24)
ALLOW_SHELL = os.environ.get("KIRA_ALLOW_SHELL") == "1"


INDEX_PATH = os.path.join("memory", "index.json")
_doc_index = None
_backend = None


def get_assistant():
    """Lazily build the assistant; the index and backend are cached, memory is read fresh."""
    global _doc_index, _backend
    from assistant import Assistant
    from backends import pick_backend
    from retrieval import DocumentIndex

    if _doc_index is None:
        _doc_index = DocumentIndex(INDEX_PATH)
    if _backend is None:
        _backend = pick_backend()
    return Assistant(evolution, _doc_index, _backend)


def token_ok(supplied: str) -> bool:
    return bool(supplied) and hmac.compare_digest(supplied.encode("utf-8"), AUTH_TOKEN.encode("utf-8"))


def get_local_ip():
    """Find the laptop's LAN IP address for mobile connection"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


HTML_UI = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>Kira - Personal AI</title>
<style>
  :root {
    --bg: #0b0f19;
    --card: #131b2e;
    --accent: #6366f1;
    --text: #f8fafc;
    --muted: #94a3b8;
    --border: #1e293b;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  body { background: var(--bg); color: var(--text); display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
  header { background: var(--card); border-bottom: 1px solid var(--border); padding: 12px 16px; display: flex; justify-content: space-between; align-items: center; }
  .logo { font-size: 1.2rem; font-weight: 700; color: var(--accent); display: flex; align-items: center; gap: 8px; }
  .status-badge { font-size: 0.75rem; background: #064e3b; color: #34d399; padding: 4px 8px; border-radius: 999px; }
  #chat-container { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
  .message { max-width: 85%; padding: 10px 14px; border-radius: 12px; font-size: 0.95rem; line-height: 1.4; word-break: break-word; }
  .user { align-self: flex-end; background: var(--accent); color: white; border-bottom-right-radius: 2px; }
  .bot { align-self: flex-start; background: var(--card); border: 1px solid var(--border); color: var(--text); border-bottom-left-radius: 2px; }
  .tool-output { font-family: monospace; font-size: 0.8rem; background: #000; padding: 8px; border-radius: 6px; margin-top: 6px; overflow-x: auto; color: #a5f3fc; }
  footer { background: var(--card); border-top: 1px solid var(--border); padding: 10px 14px; display: flex; gap: 8px; }
  input[type="text"] { flex: 1; background: #1e293b; border: 1px solid var(--border); color: white; padding: 12px 14px; border-radius: 8px; font-size: 1rem; outline: none; }
  input[type="text"]:focus { border-color: var(--accent); }
  button { background: var(--accent); color: white; border: none; padding: 12px 18px; border-radius: 8px; font-weight: 600; cursor: pointer; }
  .quick-bar { display: flex; gap: 6px; padding: 8px 16px; background: #0f172a; overflow-x: auto; border-bottom: 1px solid var(--border); }
  .quick-chip { font-size: 0.75rem; background: var(--card); border: 1px solid var(--border); padding: 4px 10px; border-radius: 99px; white-space: nowrap; cursor: pointer; color: var(--muted); }
  .quick-chip:hover { color: white; border-color: var(--accent); }
</style>
</head>
<body>
<header>
  <div class="logo">✦ Kira Core</div>
  <div class="status-badge" id="sys-status">Laptop Connected</div>
</header>
<div class="quick-bar">
  <div class="quick-chip" onclick="sendCmd('System status')">💻 Status</div>
  <div class="quick-chip" onclick="sendCmd('List files in current folder')">📁 Files</div>
  <div class="quick-chip" onclick="sendCmd('Consolidate memory')">🧠 Memory</div>
</div>
<div id="chat-container">
  <div class="message bot">Hello! I am Kira. Connected to your laptop. You can ask me to run commands, check files, access your notes, or write code.</div>
</div>
<footer>
  <input type="text" id="userInput" placeholder="Ask Kira or control laptop..." onkeypress="if(event.key==='Enter') sendMessage()">
  <button onclick="sendMessage()">Send</button>
</footer>

<script>
// Token comes from the URL fragment (#token=...), which is never sent to the server.
const TOKEN = new URLSearchParams(location.hash.slice(1)).get("token") || "";
async function sendMessage() {
  const input = document.getElementById("userInput");
  const msg = input.value.trim();
  if (!msg) return;
  input.value = "";
  
  appendMessage(msg, "user");
  const botDiv = appendMessage("Thinking...", "bot");
  
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Kira-Token": TOKEN },
      body: JSON.stringify({ message: msg })
    });
    const data = await res.json();
    if (!res.ok) { botDiv.textContent = data.error || ("Request failed: " + res.status); return; }
    botDiv.textContent = data.reply;
    if (data.tool_output) {
      const out = document.createElement("div");
      out.className = "tool-output";
      out.textContent = data.tool_output;
      botDiv.appendChild(out);
    }
  } catch (err) {
    botDiv.textContent = "Error communicating with Kira: " + err;
  }
}

function sendCmd(cmd) {
  document.getElementById("userInput").value = cmd;
  sendMessage();
}

function appendMessage(text, sender) {
  const c = document.getElementById("chat-container");
  const d = document.createElement("div");
  d.className = `message ${sender}`;
  d.textContent = text;
  c.appendChild(d);
  c.scrollTop = c.scrollHeight;
  return d;
}
</script>
</body>
</html>
"""


class KiraServerHandler(BaseHTTPRequestHandler):
    def _send_json(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        if token_ok(self.headers.get("X-Kira-Token", "")):
            return True
        self._send_json(401, {"error": "Missing or invalid token."})
        return False

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(HTML_UI.encode("utf-8"))
        elif self.path == "/api/status":
            if not self._authorized():
                return
            self._send_json(200, LaptopTools.get_system_status())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/chat":
            if not self._authorized():
                return
            try:
                content_length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                return self._send_json(400, {"error": "Bad Content-Length."})
            if content_length < 0 or content_length > MAX_BODY_BYTES:
                return self._send_json(413, {"error": "Request too large."})
            try:
                req = json.loads(self.rfile.read(content_length).decode("utf-8"))
                user_msg = str(req.get("message", "")).strip()
            except (ValueError, AttributeError):
                return self._send_json(400, {"error": "Invalid JSON body."})

            reply = ""
            tool_output = None

            # Tool triggers / actions
            lower_msg = user_msg.lower()
            if "status" in lower_msg or "battery" in lower_msg or "cpu" in lower_msg or "ram" in lower_msg:
                status = LaptopTools.get_system_status()
                reply = "Here is your laptop's current live system status:"
                tool_output = json.dumps(status, indent=2)
            elif "list files" in lower_msg or "show files" in lower_msg or "dir" in lower_msg:
                files = LaptopTools.list_directory(".")
                reply = f"Here are the files in current directory ({len(files)} items):"
                tool_output = "\n".join(files[:25])
            elif "consolidate memory" in lower_msg or "evolve" in lower_msg:
                summary = evolution.consolidate_memory()
                reply = "Memory consolidation complete! Kira updated its evolution log."
                tool_output = summary
            elif lower_msg.startswith("run:"):
                cmd = user_msg[4:].strip()
                if ALLOW_SHELL:
                    reply = f"Executed shell command: `{cmd}`"
                    tool_output = LaptopTools.execute_shell(cmd)
                else:
                    reply = "Shell execution is disabled. Start the server with KIRA_ALLOW_SHELL=1 to enable it."
            elif lower_msg.startswith("research:"):
                from agents import Orchestrator
                task = user_msg[9:].strip()
                if task:
                    out = Orchestrator(get_assistant()).run(task)
                    reply = out["final"]
                    steps = [f"- {q}" for q in out["plan"]]
                    tool_output = "Plan:" + chr(10) + chr(10).join(steps)
                else:
                    reply = "Give me a task, e.g. research: what is the auth header and how is memory stored?"
            elif lower_msg.startswith("remember:"):
                fact = user_msg[9:].strip()
                if fact:
                    evolution.teach_fact("note", fact[:40], fact)
                    reply = "Got it, I'll remember that."
                else:
                    reply = "Tell me what to remember, e.g. remember: my project deadline is Friday."
            else:
                result = get_assistant().ask(user_msg)
                reply = result["answer"]
                if result["sources"]:
                    tool_output = "Sources:\n" + "\n".join(dict.fromkeys(result["sources"]))

            # Log interaction to evolution engine
            evolution.log_interaction(user_msg, reply + ("\n" + tool_output if tool_output else ""))

            self._send_json(200, {"reply": reply, "tool_output": tool_output})
        else:
            self.send_response(404)
            self.end_headers()


def start_server(port: int = 8080, host: str = None):
    host = host or os.environ.get("KIRA_HOST", "127.0.0.1")
    httpd = HTTPServer((host, port), KiraServerHandler)

    print("=" * 60)
    print("  KIRA MOBILE & LAPTOP BRIDGE SERVER")
    print(f"  Laptop URL:  http://localhost:{port}/#token={AUTH_TOKEN}")
    if host in ("0.0.0.0", "::"):
        print(f"  Mobile URL:  http://{get_local_ip()}:{port}/#token={AUTH_TOKEN}")
        print("  WARNING: reachable by every device on your network; keep the token private.")
    else:
        print(f"  Bound to {host} only. Set KIRA_HOST=0.0.0.0 to allow your phone.")
    print(f"  Shell execution: {'ENABLED' if ALLOW_SHELL else 'disabled (set KIRA_ALLOW_SHELL=1)'}")
    print("=" * 60)
    print("Press Ctrl+C to stop the server.")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        httpd.server_close()


if __name__ == "__main__":
    start_server()
