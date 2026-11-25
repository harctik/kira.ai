"""FP32 vs dynamic-INT8 CPU forward-pass latency for the ScaledTransformer presets.
Usage: python -m benchmarks.int8_cpu_latency [--threads 2] [--seq 128]
Reports parameter counts for every preset and latency for the presets that fit in RAM."""
import argparse, time, torch
from model_scale import ScaledTransformer, get_preset_config

ap = argparse.ArgumentParser()
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--seq", type=int, default=128)
ap.add_argument("--presets", nargs="+", default=["15m", "125m"])
a = ap.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(0)

print("preset  params")
for name in ["15m", "125m", "500m", "1b", "3b"]:
    with torch.device("meta"):
        n = sum(p.numel() for p in ScaledTransformer(get_preset_config(name)).parameters())
    print(f"{name:6s}  {n/1e6:8.1f}M")

def bench(model, x, n=15):
    with torch.no_grad():
        for _ in range(3): model(x)
        t = time.perf_counter()
        for _ in range(n): model(x)
        return (time.perf_counter() - t) / n * 1000

print(f"\nforward pass, {a.seq}-token context, {a.threads} threads")
for name in a.presets:
    cfg = get_preset_config(name, block_size=max(256, a.seq))
    m = ScaledTransformer(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, a.seq))
    fp = bench(m, x); iq = bench(m.quantize_for_cpu().eval(), x)
    print(f"{name:6s} FP32 {fp:7.1f} ms   INT8 {iq:7.1f} ms   speedup {fp/iq:.2f}x")
