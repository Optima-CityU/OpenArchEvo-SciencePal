#!/usr/bin/env python3
"""E22: independent re-analysis of E7/E10/E13 raw logs by a panel of SciencePal agents (5 models x 1)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evo"))
from sp_panel import run_panel, MODELS
B = ROOT / "prompts" / "panels" / "e22"
files = {"runs.tar.gz": (B / "runs.tar.gz").read_bytes(), "README.md": (B / "README.md").read_bytes()}
run_panel("e22", files, (B / "prompt.txt").read_text(), [(k, MODELS[k]) for k in MODELS], fetch=("analysis.py",), par=2)
