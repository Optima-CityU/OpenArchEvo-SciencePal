#!/usr/bin/env python3
"""Functional-equivalence test between two architecture programs (protocol: cpu-demo-v1).

Two programs are *functionally equivalent under shared weights* if, after building both blocks with the
contract kwargs and copying every parameter/buffer of A into the same-named, same-shaped tensor of B,
the outputs (y and spikes) of B equal those of A on random inputs, and B has no parameters that A lacks
(or the extra ones receive exactly zero gradient, i.e. are dead).

This catches LLM "edits" that only add dead code, rename locals or reorder equivalent operations — the
R11 failure mode where 79–98 changed lines produced bit-identical training curves.

Usage: python func_equiv.py A.py B.py [--json out.json]
Exit 0 = equivalent, 1 = different, 2 = error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cpu_check import _setup_paths  # noqa: E402


def _pair(a_path, b_path, neuron, d_model, seq_len, n_heads, trials, dropout):
    import torch
    from openarchevo.contract import block_kwargs
    from openarchevo.models.program import load_block_class
    kw = block_kwargs(d_model, n_heads, 2, dropout, neuron, layer_idx=0)
    torch.manual_seed(0)
    A = load_block_class(a_path, check_imports=True)(**kw)
    torch.manual_seed(1)
    B = load_block_class(b_path, check_imports=True)(**kw)
    sa, sb = A.state_dict(), B.state_dict()
    copied, shape_mismatch = [], []
    for k, v in sa.items():
        if k in sb:
            if sb[k].shape == v.shape:
                sb[k] = v.clone()
                copied.append(k)
            else:
                shape_mismatch.append(k)
    B.load_state_dict(sb)
    only_a, only_b = sorted(set(sa) - set(sb)), sorted(set(sb) - set(sa))
    r = {"n_params_a": sum(p.numel() for p in A.parameters()), "n_params_b": sum(p.numel() for p in B.parameters()),
         "copied": len(copied), "only_a": only_a[:20], "only_b": only_b[:20], "shape_mismatch": shape_mismatch[:20]}
    g = torch.Generator().manual_seed(123)
    max_dy = max_ds = max_dreg = max_dgrad = 0.0
    for t in range(trials):
        x = torch.randn(2, seq_len, d_model, generator=g)
        # eval forward
        A.eval(); B.eval()
        with torch.no_grad():
            ya, rega, spa = A(x)
            yb, regb, spb = B(x)
        max_dy = max(max_dy, float((ya - yb).abs().max()))
        if spa is not None and spb is not None and spa.shape == spb.shape:
            max_ds = max(max_ds, float((spa.float() - spb.float()).abs().max()))
        elif (spa is None) != (spb is None) or (spa is not None and spa.shape != spb.shape):
            max_ds = float("inf")
        # train-mode forward + backward with identical RNG state (dropout), including the regulariser
        A.train(); B.train(); A.zero_grad(); B.zero_grad()
        torch.manual_seed(1000 + t)
        ya, rega, _ = A(x)
        torch.manual_seed(1000 + t)
        yb, regb, _ = B(x)
        max_dy = max(max_dy, float((ya - yb).abs().max().detach()))
        ra = float(rega) if rega is not None else 0.0
        rb = float(regb) if regb is not None else 0.0
        max_dreg = max(max_dreg, abs(ra - rb))
        (ya.pow(2).mean() + (rega if rega is not None else 0.0)).backward()
        (yb.pow(2).mean() + (regb if regb is not None else 0.0)).backward()
        na, nb = dict(A.named_parameters()), dict(B.named_parameters())
        for k in na:
            if k in nb and na[k].shape == nb[k].shape:
                ga = na[k].grad if na[k].grad is not None else torch.zeros_like(na[k])
                gb = nb[k].grad if nb[k].grad is not None else torch.zeros_like(nb[k])
                max_dgrad = max(max_dgrad, float((ga - gb).abs().max()))
        dead_extra = None
        if only_b:
            dead_extra = all(nb[k].grad is None or float(nb[k].grad.abs().max()) == 0.0 for k in only_b if k in nb)
    tol = 1e-6
    r.update(max_abs_dy=max_dy, max_abs_dspikes=max_ds, max_abs_dreg=max_dreg, max_abs_dgrad=max_dgrad,
             extra_params_dead=dead_extra if only_b else None, spikes_equal=bool(max_ds <= tol))
    # equivalent = same outputs (eval and train), same regulariser and same gradients with shared weights.
    r["equivalent"] = bool(max_dy <= tol and max_dreg <= tol and max_dgrad <= 1e-5 and not shape_mismatch
                           and not only_a and (not only_b or dead_extra))
    return r


def compare(a_path: Path, b_path: Path, d_model=64, seq_len=160, n_heads=4, trials=2,
            neurons=("bptt", "sltt"), dropout=0.1) -> dict:
    """Shared-weight equivalence: identical outputs (eval + train mode with the same dropout RNG), identical
    regulariser and identical parameter gradients, for every neuron type in `neurons`. seq_len 160 crosses
    chunk boundaries (chunk 64). Initialisation differences are deliberately NOT compared (weights are shared):
    two programs that differ only in initialisation are reported as `equivalent` with the note `init_may_differ`."""
    _setup_paths()
    import torch
    torch.set_num_threads(2)
    res: dict = {"protocol": "cpu-demo-v1/func-equiv-v2", "a": a_path.name, "b": b_path.name, "per_neuron": {}}
    for nt in neurons:
        try:
            res["per_neuron"][nt] = _pair(a_path, b_path, nt, d_model, seq_len, n_heads, trials, dropout)
        except Exception as exc:
            res["per_neuron"][nt] = {"error": f"{type(exc).__name__}: {exc}"[:300], "equivalent": False}
    res["equivalent"] = all(v.get("equivalent") for v in res["per_neuron"].values())
    res["spikes_equal"] = all(v.get("spikes_equal") for v in res["per_neuron"].values())
    first = next(iter(res["per_neuron"].values()))
    res["max_abs_dy"] = max(v.get("max_abs_dy", float("inf")) for v in res["per_neuron"].values())
    res["note"] = "init_may_differ" if res["equivalent"] else ""
    # weights can only be shared where parameter names and shapes match; if most of A's tensors have no counterpart
    # (typical for a cross-family recombination donor), the verdict says little: report it as not comparable
    n_a = first.get("copied", 0) + len(first.get("only_a") or []) + len(first.get("shape_mismatch") or [])
    res["shared_fraction"] = round(first.get("copied", 0) / max(1, n_a), 3)
    if not res["equivalent"] and res["shared_fraction"] < 0.5:
        res["note"] = "not_comparable: fewer than half of A's tensors could be shared (different family?)"
    for k in ("only_a", "only_b", "copied", "n_params_a", "n_params_b"):
        res[k] = first.get(k)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    try:
        r = compare(a.a, a.b)
    except Exception as exc:
        r = {"error": f"{type(exc).__name__}: {exc}"[:500], "equivalent": None}
    t = json.dumps(r, indent=1)
    if a.json:
        a.json.write_text(t)
    print(t)
    return 2 if r.get("equivalent") is None else (0 if r["equivalent"] else 1)


if __name__ == "__main__":
    sys.exit(main())
