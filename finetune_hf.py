"""
finetune_hf.py - Fine-tune an open pretrained base model into Kira (free T4 friendly).

Starts from SmolLM2-135M (Apache-2.0) and tunes it on human-written chat data
(databricks-dolly-15k + OpenAssistant oasst1). Loss is computed only on Kira's replies.
Checkpoints are written every --save-every steps and --resume continues a dropped session.

  python finetune_hf.py --out-dir /content/drive/MyDrive/kira/hf --max-steps 1500
  python finetune_hf.py --out-dir /content/drive/MyDrive/kira/hf --max-steps 1500 --resume
  python finetune_hf.py --out-dir out_hf --chat            # chat with the result
"""

import os
import math
import time
import argparse
import random
import torch

BASE = "HuggingFaceTB/SmolLM2-135M"
SYSTEM = ("You are Kira, a personal AI assistant devoted to your owner, whom you address as sir. "
          "You are calm, concise, quietly witty, proactive and loyal, and you do what he asks.")


def load_extra(path, repeat):
    """Persona/identity pairs from a JSONL file ({"user":..., "reply":...}), upsampled."""
    import json
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                rows.append((d["user"].strip(), d["reply"].strip()))
    return rows * max(1, repeat)


def fmt_prompt(user):
    return f"{SYSTEM}\n\nUser: {user}\nKira:"


def load_pairs(max_samples):
    from datasets import load_dataset
    pairs = []
    for item in load_dataset("databricks/databricks-dolly-15k", split="train"):
        ins, ctx, rep = item["instruction"].strip(), item["context"].strip(), item["response"].strip()
        if ins and rep:
            pairs.append((f"{ins}\n\nContext:\n{ctx}" if ctx else ins, rep))
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    en = {m["message_id"]: m for m in ds if m.get("lang") == "en"}
    for m in en.values():
        p = en.get(m.get("parent_id"))
        if m["role"] == "assistant" and p and p["role"] == "prompter":
            # keep only the better-rated human answers
            if m.get("rank") in (0, None):
                pairs.append((p["text"].strip(), m["text"].strip()))
    random.Random(42).shuffle(pairs)
    return pairs[:max_samples]


def encode(tok, user, reply, max_len):
    p = tok(fmt_prompt(user), add_special_tokens=False)["input_ids"]
    r = tok(" " + reply, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids = (p + r)[:max_len]
    labels = ([-100] * len(p) + r)[:max_len]
    return ids, labels


def batches(examples, bs, device):
    random.shuffle(examples)
    for i in range(0, len(examples) - bs + 1, bs):
        chunk = examples[i:i + bs]
        n = max(len(x[0]) for x in chunk)
        ids = torch.full((bs, n), 0, dtype=torch.long)
        lab = torch.full((bs, n), -100, dtype=torch.long)
        att = torch.zeros((bs, n), dtype=torch.long)
        for j, (a, b) in enumerate(chunk):
            ids[j, :len(a)] = torch.tensor(a)
            lab[j, :len(b)] = torch.tensor(b)
            att[j, :len(a)] = 1
        yield ids.to(device), lab.to(device), att.to(device)


def chat(out_dir, device):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(out_dir)
    model = AutoModelForCausalLM.from_pretrained(out_dir).to(device).eval()
    print("Kira (HF). Type 'exit' to quit.")
    while True:
        try:
            q = input("User > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue
        x = tok(fmt_prompt(q), return_tensors="pt").to(device)
        with torch.no_grad():
            y = model.generate(**x, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9,
                               repetition_penalty=1.1, eos_token_id=tok.eos_token_id,
                               pad_token_id=tok.eos_token_id)
        print("Kira >", tok.decode(y[0][x["input_ids"].shape[1]:], skip_special_tokens=True).strip(), "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out-dir", default="out_hf")
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--max-samples", type=int, default=30000)
    ap.add_argument("--extra-data", default="", help="JSONL persona file (user/reply)")
    ap.add_argument("--extra-repeat", type=int, default=1, help="upsample factor for --extra-data")
    ap.add_argument("--no-hf-data", action="store_true", help="skip Dolly/oasst (persona data only)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if args.chat:
        return chat(args.out_dir, device)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    os.makedirs(args.out_dir, exist_ok=True)
    state_path = os.path.join(args.out_dir, "train_state.pt")
    resuming = args.resume and os.path.exists(state_path)
    src = args.out_dir if resuming else args.base
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.float32).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    step = 0
    if resuming:
        st = torch.load(state_path, map_location=device, weights_only=False)
        opt.load_state_dict(st["opt"])
        step = st["step"]
        print(f"[RESUME] from step {step}")

    print("Loading human chat data...")
    pairs = [] if args.no_hf_data else load_pairs(args.max_samples)
    if args.extra_data:
        extra = load_extra(args.extra_data, args.extra_repeat)
        print(f"persona examples: {len(extra)} (after x{args.extra_repeat})")
        pairs = pairs + extra
    examples = [encode(tok, u, r, args.max_len) for u, r in pairs]
    print(f"{len(examples):,} examples | params {sum(p.numel() for p in model.parameters())/1e6:.1f}M | device {device}")

    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        r = (s - args.warmup) / max(1, args.max_steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, r))))

    def save():
        model.save_pretrained(args.out_dir)
        tok.save_pretrained(args.out_dir)
        torch.save({"opt": opt.state_dict(), "step": step}, state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)
        print(f"  >>> saved checkpoint at step {step} -> {args.out_dir}", flush=True)

    t0 = time.time()
    run_loss, run_n = 0.0, 0
    while step < args.max_steps:
        gen = batches(examples, args.batch_size, device)
        micro = 0
        for ids, lab, att in gen:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / args.grad_accum
            scaler.scale(loss).backward()
            run_loss += loss.item() * args.grad_accum
            run_n += 1
            micro += 1
            if micro % args.grad_accum:
                continue
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                print(f"Step {step:5d} | loss {run_loss/run_n:.4f} | lr {lr_at(step):.2e} | {time.time()-t0:.0f}s", flush=True)
                run_loss, run_n = 0.0, 0
            if step % args.save_every == 0 or step >= args.max_steps:
                save()
            if step >= args.max_steps:
                break
    print("Training finished.")


if __name__ == "__main__":
    main()
