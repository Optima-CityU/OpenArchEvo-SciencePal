#!/usr/bin/env python3
"""E32: SciencePal model panel blindly designs a cheap 30-epoch predictor from 2-epoch logs; scored on held-out archs."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evo"))
from sp_panel import run_panel, MODELS
B = ROOT / "prompts" / "panels" / "e32"
files = {n: (B / n).read_bytes() for n in ("README.md", "train.csv", "test.csv", "programs.zip")}
run_panel("e32", files, (B / "prompt.txt").read_text(), [(k, MODELS[k]) for k in MODELS], fetch=("predictor.py",), par=2)
