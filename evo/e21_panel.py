#!/usr/bin/env python3
"""E21: hypothesis panel — 5 SciencePal models analyse the E18 curves, propose falsifiable hypotheses + interventions."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evo"))
from sp_panel import run_panel, MODELS
B = ROOT / "prompts" / "panels" / "e21"
files = {n: (B / n).read_bytes() for n in ("README.md", "curves.csv", "r03a-263bee.py", "i00-deltanet.py", "r04v2-bf4d6a.py")}
run_panel("e21", files, (B / "prompt.txt").read_text(), [(k, MODELS[k]) for k in MODELS], fetch=("h1.py", "h2.py"), par=2)
