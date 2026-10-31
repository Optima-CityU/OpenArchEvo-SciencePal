#!/usr/bin/env bash
# Fixed CPU Skill benchmark used in E15/E19/E27/E43: 6 expert seeds x 3 roles x 8 proposals = 24 proposals, 2 repetitions.
# usage: scripts/skill_benchmark.sh <out_dir> <baseline SKILL.md> [candidate SKILL.md ...]
# Then:  LADDER_DIR=<out_dir> python3 evo/ladder_stats.py ; LADDER_DIR=<out_dir> python3 evo/sibling_equiv.py <names>  (distinct_new)
# Required env: OAE_LLM_BASE_URL, OAE_LLM_API_KEY (proposer model, DeepSeek-V4-Flash in our runs).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
OUT="$1"; shift
SEEDS="i09-s0557 i03-s0335 i00-deltanet i00-hgrn2 i00-mamba2-ssd i10-s0337"
run() { name=$1; path=$2; for rep in 0 1; do
  [ -f "$OUT/${name}_r${rep}/DONE" ] && continue
  SKILL_MD_PATH="$ROOT/$path" nice -n 15 python3 evo/batch_propose.py --out "$OUT/${name}_r${rep}" --seeds $SEEDS --per-role 8 --workers 2 --seed $rep
  touch "$OUT/${name}_r${rep}/DONE"
done; }
for p in "$@"; do run "$(basename "$(dirname "$p")")" "$p" & done
wait
