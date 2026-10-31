#!/usr/bin/env python3
"""CPU smoke check for one OpenArchEvo architecture program (protocol: cpu-demo-v1).

Purpose: tell quickly whether a candidate `SpikingFLABlock` program obeys the
interface contract on CPU. It is a *demo / feasibility* protocol, deliberately
separate from the research protocol (GPU, SDN neurons, d_model=256, WT2 proxy).
Numbers produced here are never mixed with research metrics.

Checks (all at tiny size, BPTT LIF neurons, fp32, CPU):
  1. imports resolve (with the bundled CPU shims for `fla` / `transformers`)
  2. block builds with the contract kwargs
  3. forward returns (y, reg_loss, spikes) with y shaped like x and finite
  4. spikes (if returned) are binary
  5. backward produces finite gradients for all trainable parameters that get one
  6. causality: gradient of output position t w.r.t. inputs > t is ~0, and a
     perturbation of a future token does not change past outputs
  7. every nn.Linear receives binary (0/1) inputs (contract rule 3)
  8. deterministic structure stats: parameter count per block, fire rate

Usage:
  python cpu_check.py PROGRAM.py [--json out.json] [--d-model 64] [--seq-len 64]
Exit code 0 = pass, 1 = fail (reason in JSON), 2 = usage error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTOCOL = "cpu-demo-v1"


def _setup_paths() -> None:
    # bundled shims first only for packages that are not installed for real
    shim = HERE / "cpu_shims"
    for pkg in ("fla", "transformers"):
        try:
            __import__(pkg)
        except Exception:
            if str(shim) not in sys.path:
                sys.path.insert(0, str(shim))
    lib = HERE / "lib"  # vendored openarchevo subset
    if lib.exists() and str(lib) not in sys.path:
        sys.path.insert(0, str(lib))


def check(program: Path, d_model: int, seq_len: int, n_heads: int, seed: int) -> dict:
    _setup_paths()
    import torch
    from openarchevo.contract import block_kwargs
    from openarchevo.models.program import load_block_class

    torch.manual_seed(seed)
    torch.set_num_threads(int(os.environ.get("CPU_CHECK_THREADS", "2")))
    src = program.read_text(encoding="utf-8")
    res: dict = {"protocol": PROTOCOL, "program": program.name,
                 "code_sha256": hashlib.sha256(src.encode()).hexdigest(),
                 "config": {"d_model": d_model, "seq_len": seq_len, "num_heads": n_heads,
                            "d_ffn_mult": 2, "neuron_type": "bptt", "batch": 2, "seed": seed},
                 "passed": False, "stage": "import"}
    t0 = time.perf_counter()
    try:
        cls = load_block_class(program, check_imports=True)
        res["stage"] = "build"
        kw = block_kwargs(d_model, n_heads, 2, 0.0, "bptt", layer_idx=0)
        block = cls(**kw)
        block.eval()
        n_params = sum(p.numel() for p in block.parameters())
        res["params_per_block"] = int(n_params)

        res["stage"] = "forward"
        x = torch.randn(2, seq_len, d_model, requires_grad=True)
        out = block(x)
        if not (isinstance(out, (tuple, list)) and len(out) == 3):
            raise ValueError(f"forward must return (y, reg_loss, spikes); got {type(out).__name__}"
                             f"{'' if not isinstance(out, (tuple, list)) else ' of len %d' % len(out)}")
        y, reg, spikes = out
        if tuple(y.shape) != tuple(x.shape):
            raise ValueError(f"y shape {tuple(y.shape)} != input shape {tuple(x.shape)}")
        if not torch.isfinite(y).all():
            raise ValueError("y has non-finite values")
        if spikes is not None:
            s = spikes.detach().float()
            if s.numel() and not torch.all((s == 0) | (s == 1)):
                raise ValueError("spikes tensor is not binary")
            res["fire_rate"] = float(s.mean()) if s.numel() else None
            res["n_spikes_elems"] = int(s.numel())
        else:
            res["fire_rate"] = None
        res["has_reg_loss"] = reg is not None

        res["stage"] = "backward"
        loss = y.pow(2).mean() + (reg if reg is not None else 0.0)
        loss.backward()
        bad = [n for n, p in block.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        if bad:
            raise ValueError(f"non-finite gradients in {bad[:3]}")
        res["params_with_grad_frac"] = round(
            sum(p.grad is not None for p in block.parameters()) / max(1, len(list(block.parameters()))), 3)

        res["stage"] = "spiking_contract"
        # contract rule 3: every nn.Linear must receive binary spikes (0/1) as input
        nonbin = {}
        hooks = []
        for name, mod in block.named_modules():
            if isinstance(mod, torch.nn.Linear):
                def _hook(m, args, _n=name):
                    xx = args[0].detach().float()
                    nonbin[_n] = max(nonbin.get(_n, 0.0), float(((xx != 0) & (xx != 1)).float().mean()))
                hooks.append(mod.register_forward_pre_hook(_hook))
        with torch.no_grad():
            block(torch.randn(2, seq_len, d_model))
        for h in hooks:
            h.remove()
        bad = {k: round(v, 3) for k, v in nonbin.items() if v > 1e-6}
        res["linear_inputs_checked"] = len(nonbin)
        if bad:
            raise ValueError("non-binary input to nn.Linear (contract rule 3): " + json.dumps(bad))

        res["stage"] = "causality"
        res["causality"] = _causality(block, d_model, seq_len, torch)
        if not res["causality"]["ok"]:
            raise ValueError("causality violation: " + json.dumps(res["causality"]))
        res["passed"] = True
        res["stage"] = "done"
    except Exception as exc:  # report, never crash the caller
        res["error_type"] = type(exc).__name__
        res["error"] = str(exc)[:600]
        res["traceback_tail"] = traceback.format_exc().splitlines()[-4:]
    res["seconds"] = round(time.perf_counter() - t0, 3)
    return res


def _causality(block, d_model: int, seq_len: int, torch) -> dict:
    block.eval()
    pivots = sorted({seq_len // 4, seq_len // 2, seq_len - 2})
    max_grad_future, max_pert = 0.0, 0.0
    g = torch.Generator().manual_seed(1234)
    for t in pivots:
        x = torch.randn(1, seq_len, d_model, generator=g).requires_grad_(True)
        y = block(x)[0]
        y[:, : t + 1].sum().backward()
        max_grad_future = max(max_grad_future, float(x.grad[:, t + 1:].abs().max()))
        with torch.no_grad():
            x2 = x.detach().clone()
            x2[:, t + 1] += torch.randn(d_model, generator=g) * 3.0
            ya, yb = block(x.detach())[0], block(x2)[0]
            max_pert = max(max_pert, float((ya[:, : t + 1] - yb[:, : t + 1]).abs().max()))
    tol = 1e-5
    return {"ok": max_grad_future <= tol and max_pert <= tol, "max_grad_future": max_grad_future,
            "max_perturbation_change": max_pert, "tol": tol, "pivots": pivots}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("program", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if not a.program.exists():
        print(f"no such file: {a.program}", file=sys.stderr)
        return 2
    res = check(a.program, a.d_model, a.seq_len, a.num_heads, a.seed)
    text = json.dumps(res, indent=1)
    if a.json:
        a.json.write_text(text)
    print(text)
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
