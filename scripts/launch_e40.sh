#!/usr/bin/env bash
# E40 main experiment: arms V (2-epoch val fitness), O (OAF fitness), L (OAF + research-lead agent), 3 seed-paired runs each.
# Each run: 4 generations x 9 proposals, 3 trained per generation on SciencePal GPU labs. Skill v0.5.1 for all arms.
# Required env (see .env.example): SCIENCEPAL_API_KEY, OAE_LLM_BASE_URL, OAE_LLM_API_KEY; optional OAE_LLM_MODEL,
# OPENARCHEVO_DIR (default ../OpenArchEvo), OAE_LABS_JSON (default config/labs.json).
# Each run directory needs a config.json (e.g. config/e40_L_config.example.json; set "fitness": "val" for arm V, "oaf" for O/L)
# and the seed population (controller.py init).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
OUT="${E40_OUT:-runs/e40}"; mkdir -p "$OUT/logs"
for arm in V O L; do for k in 1 2 3; do
  extra=""; [ "$arm" = L ] && extra="--lead"
  SKILL_MD_PATH="$ROOT/skills/v0.5.1/SKILL.md" nohup python3 evo/run_arm2.py --state "$OUT/${arm}_s$k" --rounds 1 2 3 4 --k 3 $extra \
    >> "$OUT/logs/${arm}_s$k.log" 2>&1 < /dev/null &
done; done
wait
# then deliver each run's top-2 children (own fitness) for 30 epochs x seeds 0,1:
python3 scripts/deliver_e40.py --out "$OUT"
