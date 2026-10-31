#!/usr/bin/env bash
# Start one feeder per SciencePal GPU lab (all draw from the shared pool "pool"), plus the result collector loop.
# Labs are read from $OAE_LABS_JSON (default config/labs.json): {"lab1": "<thread id>", ...}.
# Required env: SCIENCEPAL_API_KEY.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
LABS="${OAE_LABS_JSON:-config/labs.json}"
mkdir -p runs
python3 -c "import json,sys; [print(k, v) for k, v in json.load(open(sys.argv[1])).items()]" "$LABS" | while read -r name thread; do
  feed=$([ "$name" = lab1 ] && echo "" || echo "--feed $name")
  nohup python3 lab/feeder.py --thread "$thread" $feed --fp-par 2 --max-train 1 --pool pool >> "runs/feeder_$name.log" 2>&1 < /dev/null &
done
# collector: pulls finished job results from every lab into runs/lab_store.json every 10 minutes
nohup bash -c 'while true; do timeout 900 python3 lab/collect.py >> runs/collect.log 2>&1; sleep 600; done' > /dev/null 2>&1 < /dev/null &
# optional: lab supervisor agent (restarts dead workers through a SciencePal agent turn)
# nohup python3 harness/lab_supervise.py --lab lab1=new --lab lab2=new >> runs/lab_supervise.log 2>&1 &
