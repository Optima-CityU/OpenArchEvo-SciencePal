#!/usr/bin/env python3
"""E44: SciencePal model panel independently re-analyses round-8 raw data."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evo"))
from sp_panel import run_panel, MODELS
B = ROOT / "prompts" / "panels" / "e44"
names = ("README.md", "trainings.csv", "curves_2ep.csv", "e29_pick.json", "e29_step2.json", "e32_train.csv", "e32_test.csv", "prereg_round8.md")
files = {n: (B / n).read_bytes() for n in names}
run_panel("e44", files, (B / "prompt.txt").read_text(), [(k, MODELS[k]) for k in MODELS], fetch=("analysis.py",), par=2)
