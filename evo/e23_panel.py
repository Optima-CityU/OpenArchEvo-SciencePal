#!/usr/bin/env python3
"""E23: literature grounding panel (3 SciencePal models); every cited work is verified by us before use."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evo"))
from sp_panel import run_panel, MODELS
B = ROOT / "prompts" / "panels" / "e23"
run_panel("e23", {"finding.md": (B / "finding.md").read_bytes()}, (B / "prompt.txt").read_text(),
          [(k, MODELS[k]) for k in ("gpt61", "kimik3", "glm53")], par=1)
