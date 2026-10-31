#!/usr/bin/env python3
"""Arm driver for the evolutionary loop (E40: arms V / O / L; earlier: B2 rule-based islands vs M island-manager agents).

  python evo/run_arm2.py --state runs/evo_M --rounds 1 2 3 --manager
Per round: [M only: dossier -> manager agents (SciencePal) -> directives] -> propose_batch (DeepSeek, Skill v0.2)
-> fingerprints (SciencePal GPU) -> select k (NSGA-II) -> train (SciencePal GPU) -> ingest.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("SKILL_MD_PATH", str(ROOT / "skills" / "v0.7.0" / "SKILL.md"))  # E40 used skills/v0.5.1 (see scripts/launch_e40.sh)


def sh(*cmd, timeout=None):
    print(time.strftime("%H:%M:%S"), " ".join(map(str, cmd[1:]))[:200], flush=True)
    p = subprocess.run(list(map(str, cmd)), cwd=str(ROOT), capture_output=True, text=True, timeout=timeout)
    tail = "\n".join(l for l in (p.stdout + p.stderr).splitlines() if not any(s in l for s in ("INFO", "Warn", "Loading", "warn")))
    print(tail[-2000:], flush=True)
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--rounds", type=int, nargs="+", required=True)
    ap.add_argument("--manager", action="store_true")
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--diverse", action="store_true", help="diversity-aware island managers")
    ap.add_argument("--mode", choices=["naive", "diverse", "moead", "moead_div", "moead_rp"], default=None)
    ap.add_argument("--lead", action="store_true", help="V2: one research-lead agent per generation (evo/lead.py)")
    ap.add_argument("--weights-only", action="store_true", help="MD ablation: MOEA/D weight vectors go straight to the proposer, no manager agents")
    ap.add_argument("--no-train", action="store_true", help="proposal-only study (CPU fingerprints, no selection/training)")
    ap.add_argument("--resume-first", action="store_true", help="first round: re-attach to fingerprint jobs already enqueued")
    a = ap.parse_args()
    ctrl = [sys.executable, str(ROOT / "evo" / "controller.py"), "--state", a.state]
    mgr = [sys.executable, str(ROOT / "evo" / "manager.py")]
    for r in a.rounds:
        extra = []
        if a.manager:
            if not (Path(a.state) / f"directives_r{r:02d}.json").exists():
                sh(*mgr, "dossier", "--state", a.state, "--round", r, *(["--diverse"] if a.diverse else []), *(["--mode", a.mode] if a.mode else []))
                sh(*mgr, "run", "--state", a.state, "--round", r, timeout=3 * 3600)
                sh(*mgr, "collect", "--state", a.state, "--round", r, *(["--mode", a.mode] if a.mode else []))
            extra = ["--directives", str(Path(a.state) / f"directives_r{r:02d}.json")]
        if a.lead:
            if not (Path(a.state) / f"directives_r{r:02d}.json").exists():
                ld = [sys.executable, str(ROOT / "evo" / "lead.py")]
                sh(*ld, "dossier", "--state", a.state, "--round", r)
                sh(*ld, "run", "--state", a.state, "--round", r, timeout=2 * 3600)
                sh(*ld, "collect", "--state", a.state, "--round", r)
            extra = ["--directives", str(Path(a.state) / f"directives_r{r:02d}.json")]
        if a.weights_only:
            sys.path.insert(0, str(ROOT / "evo"))
            from manager import weights
            ROLES = ["explore", "recombine", "simplify"]
            dirs = [{"island": i, "slots": [{"role": ROLES[(i + k) % 3], "parents": [], "direction": "(no manager: rule-based parents)",
                                              "weight": weights(i)[k]} for k in range(3)], "avoid": [], "manager_run": None} for i in range(3)]
            dp = Path(a.state) / f"directives_r{r:02d}.json"
            dp.write_text(json.dumps(dirs, indent=1))
            extra = ["--directives", str(dp)]
        res = ["--resume"] if (a.resume_first and r == a.rounds[0]) else []
        sh(*ctrl, "propose_batch", "--n", a.n, "--round", r, "--workers", 3, "--fp-timeout", 8 * 3600, *extra, *res, timeout=9 * 3600)
        if a.no_train:
            continue
        sh(*ctrl, "select", "--k", a.k)
        sh(*ctrl, "train_lab", "--timeout", 3 * 3600, timeout=7 * 3600)
        sh(*ctrl, "report")


if __name__ == "__main__":
    main()
