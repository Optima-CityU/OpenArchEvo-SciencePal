#!/usr/bin/env python3
"""CPU micro-proxy training for one architecture program (protocol: cpu-micro-v1).

A deliberately small, fully specified training run that fits in a CPU sandbox:
byte-level WikiText-2 (raw), 2 blocks, d_model 64, BPTT LIF neurons, fixed seed,
fixed number of optimizer steps. It reports validation bits-per-byte (bpb) on a
fixed slice of the validation split. Test split is never read.

This is NOT the research protocol (GPU, SDN, GPT-2 BPE, 6 x 256, 30 epochs) and
its numbers are never mixed with research PPL. Its purpose is (a) cheap feedback
when no GPU is available and (b) a demo that SciencePal can run training itself.
Whether its ranking agrees with the research protocol is an empirical question
that must be measured, not assumed.

Usage: python micro_train.py PROGRAM.py --data DIR [--steps 300] [--json out.json]
DIR must contain wt2_train.txt and wt2_validation.txt (raw text).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTOCOL = "cpu-micro-v1"
CFG = dict(d_model=64, n_layers=2, num_heads=4, d_ffn_mult=2, dropout=0.0, neuron_type="bptt",
           seq_len=128, batch=16, lr=3e-3, weight_decay=0.01, warmup=30, grad_clip=1.0,
           val_batches=24, seed=0)


def main() -> int:
    sys.path[:0] = [str(HERE / "lib")]
    from cpu_check import _setup_paths  # noqa: E402
    _setup_paths()
    import torch
    import torch.nn.functional as F
    from openarchevo.models.program import load_block_class
    from openarchevo.models.wrapper import SequenceModel

    ap = argparse.ArgumentParser()
    ap.add_argument("program", type=Path)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=CFG["seed"])
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    cfg = dict(CFG, steps=a.steps, seed=a.seed)
    torch.set_num_threads(int(os.environ.get("MICRO_THREADS", "2")))
    torch.manual_seed(cfg["seed"])

    def load(split):
        b = (a.data / f"wt2_{split}.txt").read_bytes()
        return torch.frombuffer(bytearray(b), dtype=torch.uint8).long()

    train, val = load("train"), load("validation")
    src = a.program.read_text(encoding="utf-8")
    res = {"protocol": PROTOCOL, "program": a.program.name,
           "code_sha256": hashlib.sha256(src.encode()).hexdigest(), "config": cfg, "ok": False}
    t0 = time.perf_counter()
    try:
        cls = load_block_class(a.program)
        model = SequenceModel(cls, vocab_size=256, n_outputs=256, d_model=cfg["d_model"],
                              n_layers=cfg["n_layers"], num_heads=cfg["num_heads"],
                              d_ffn_mult=cfg["d_ffn_mult"], dropout=cfg["dropout"],
                              neuron_type=cfg["neuron_type"], pad_token_id=0, is_lm=True)
        res["params"] = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                                betas=(0.9, 0.95))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg["warmup"]) * 0.5 * (
            1 + math.cos(math.pi * min(1.0, s / cfg["steps"]))))
        g = torch.Generator().manual_seed(cfg["seed"])
        L, B = cfg["seq_len"], cfg["batch"]
        losses = []
        model.train()
        for step in range(cfg["steps"]):
            idx = torch.randint(0, len(train) - L - 1, (B,), generator=g)
            x = torch.stack([train[i:i + L] for i in idx])
            y = torch.stack([train[i + 1:i + L + 1] for i in idx])
            logits, reg, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
            total = loss + (reg if reg is not None else 0.0)
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            sched.step()
            losses.append(float(loss))
        res["train_seconds"] = round(time.perf_counter() - t0, 1)
        res["train_loss_last50"] = sum(losses[-50:]) / len(losses[-50:])
        model.eval()
        n = cfg["val_batches"]
        stride = (len(val) - L - 1) // (n * B)
        tot, cnt, fires = 0.0, 0, []
        with torch.no_grad():
            for vb in range(n):
                starts = [(vb * B + j) * stride for j in range(B)]
                x = torch.stack([val[s:s + L] for s in starts])
                y = torch.stack([val[s + 1:s + L + 1] for s in starts])
                logits, _, spikes = model(x)
                tot += float(F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1), reduction="sum"))
                cnt += y.numel()
                if spikes:
                    fires.append(float(torch.cat([s.float().flatten() for s in spikes]).mean()))
        res["val_bpb"] = tot / cnt / math.log(2)
        res["val_fire_rate"] = sum(fires) / len(fires) if fires else None
        res["ok"] = True
    except Exception as exc:
        res["error_type"], res["error"] = type(exc).__name__, str(exc)[:500]
    res["seconds"] = round(time.perf_counter() - t0, 1)
    text = json.dumps(res, indent=1)
    if a.json:
        a.json.write_text(text)
    print(text)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
