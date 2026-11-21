"""
backends.py - Pluggable, free text generation backends.

Every backend exposes generate(prompt, max_new_tokens) -> str. None needs a paid API.
  ExtractiveBackend : no model at all; answers by quoting the retrieved context.
  OllamaBackend     : a free local Ollama server (http://localhost:11434), stdlib only.
  KiraBackend       : your own trained checkpoint (needs torch + a checkpoint).
"""

import json
import os
import re
import urllib.request


class ExtractiveBackend:
    name = "extractive"

    def generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        m = re.search(r"Context:\n(.*?)\n\nQuestion:", prompt, re.S)
        context = m.group(1).strip() if m else ""
        if not context:
            return "I don't have anything in memory or your indexed documents about that yet."
        return "From what I have indexed:\n" + context[:1200]


class OllamaBackend:
    name = "ollama"

    def __init__(self, model: str = "llama3.2", host: str = "http://localhost:11434", timeout: int = 120):
        self.model, self.host, self.timeout = model, host.rstrip("/"), timeout

    def generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        body = json.dumps({
            "model": self.model, "prompt": prompt, "stream": False,
            "options": {"num_predict": max_new_tokens},
        }).encode("utf-8")
        req = urllib.request.Request(self.host + "/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())["response"].strip()


class KiraBackend:
    name = "kira"

    def __init__(self, ckpt: str, tokenizer_path: str = "data/kira/tokenizer.json", device: str = "cpu"):
        import torch
        from tokenizers import Tokenizer
        from model import GPT, GPTConfig

        self.torch = torch
        self.device = device
        checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
        cfg = checkpoint["config"]
        self.model = GPT(cfg if isinstance(cfg, GPTConfig) else GPTConfig(**cfg))
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(device).eval()
        self.tok = Tokenizer.from_file(tokenizer_path)

    def generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        torch = self.torch
        ids = self.tok.encode(f"<|user|>\n{prompt}\n<|bot|>\n").ids
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model.generate(x, max_new_tokens=max_new_tokens, temperature=0.7, top_k=40)
        text = self.tok.decode(out[0].tolist()[len(ids):])
        return re.sub(r"<\|[^|]*\|>", "", text).strip()


def pick_backend():
    """KIRA_BACKEND = extractive (default) | ollama | kira. Never touches the network unless asked."""
    choice = os.environ.get("KIRA_BACKEND", "extractive").lower()
    if choice == "ollama":
        return OllamaBackend(model=os.environ.get("KIRA_OLLAMA_MODEL", "llama3.2"))
    if choice == "kira":
        return KiraBackend(os.environ.get("KIRA_CKPT", "out/ckpt.pt"))
    return ExtractiveBackend()
