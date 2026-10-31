#!/usr/bin/env python3
"""E35: SciencePal model panel explains the sigmoid effect, predicts E33, proposes a new gate activation."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evo"))
from sp_panel import run_panel, MODELS
B = ROOT / "prompts" / "panels" / "e35"
files = {n: (B / n).read_bytes() for n in ("README.md", "curves.csv", "r02a-e29092.py")}
run_panel("e35", files, (B / "prompt.txt").read_text(), [(k, MODELS[k]) for k in MODELS], par=2)
