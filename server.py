"""
server.py - Kira Mobile & Laptop Bridge Server
Runs locally on your laptop. Accessible from both laptop browser and mobile phone on Wi-Fi.
Zero extra dependencies required (uses Python standard library http.server).
"""

import os
import json
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler
from tools import LaptopTools
from evolution import EvolutionEngine

evolution = EvolutionEngine(memory_dir="memory")


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
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: msg })
    });
    const data = await res.json();
    botDiv.innerHTML = data.reply;
    if (data.tool_output) {
      botDiv.innerHTML += `<div class="tool-output">${data.tool_output}</div>`;
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
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_UI.encode("utf-8"))
        elif self.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            status = LaptopTools.get_system_status()
            self.wfile.write(json.dumps(status).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/chat":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            req = json.loads(post_data.decode("utf-8"))
            user_msg = req.get("message", "").strip()

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
                reply = f"Executed shell command: `{cmd}`"
                tool_output = LaptopTools.execute_shell(cmd)
            else:
                reply = f"Kira received: \"{user_msg}\". Ready for local model generation or OS command."

            # Log interaction to evolution engine
            evolution.log_interaction(user_msg, reply + ("\n" + tool_output if tool_output else ""))

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"reply": reply, "tool_output": tool_output}).encode("utf-8"))


def start_server(port: int = 8080):
    ip = get_local_ip()
    server_address = ("0.0.0.0", port)
    httpd = HTTPServer(server_address, KiraServerHandler)

    print("=" * 60)
    print("  KIRA MOBILE & LAPTOP BRIDGE SERVER")
    print(f"  Laptop URL:  http://localhost:{port}")
    print(f"  Mobile URL:  http://{ip}:{port}  (Open this on your phone!)")
    print("=" * 60)
    print("Press Ctrl+C to stop the server.")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        httpd.server_close()


if __name__ == "__main__":
    start_server()
