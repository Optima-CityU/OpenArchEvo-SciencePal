Lab supervision session (OpenArchEvo GPU jobs). Work unattended; never ask questions; no web search. Keep every message very short.

Only /workspace persists. Always use /workspace/venv/bin/python. Do not modify /workspace/lab/lab_worker.py or lab_setup.sh.

1. If /workspace/venv/bin/python cannot `import torch, fla` with CUDA available, run `bash /workspace/lab/lab_setup.sh > /workspace/lab/setup.out 2>&1` and poll /workspace/lab/setup.log every 60 s until SETUP_DONE (max 25 min). If fla import fails with a triton error, run `/workspace/venv/bin/python -m pip install --no-deps "fla-core==0.5.2" "flash-linear-attention==0.5.2"` once.
2. cd /workspace/lab; delete STOP if present. If no live `lab_worker.py` process exists, start it: `LAB_MAX_PARALLEL=1 LAB_GPU_AUTO=1 setsid nohup /workspace/venv/bin/python lab_worker.py >> worker.log 2>&1 &` and save its PID to worker.pid.
3. SUPERVISE: repeat the following loop for as long as this session allows (aim for at least 3 hours): `sleep 300`, then run ONE short command that prints the `ts` and `running` fields of /workspace/lab/heartbeat.json and restarts the worker exactly as in step 2 if its process is gone. Do not print anything else, do not summarise, do not stop early because nothing seems to happen — the controller adds jobs over time.
4. Only if you are about to be cut off, print `SUPERVISOR_END` and end.
