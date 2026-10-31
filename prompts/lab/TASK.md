Lab session (OpenArchEvo GPU jobs). Work unattended; never ask questions; no web search. Keep messages short.

Only /workspace persists between sessions; the controller keeps a virtual environment at /workspace/venv. Always use /workspace/venv/bin/python. Do not pip install into the system Python and do not modify /workspace/lab/lab_setup.sh or lab_worker.py.

1. Run `bash /workspace/lab/lab_setup.sh > /workspace/lab/setup.out 2>&1` in the background and poll every 60 seconds until the line SETUP_DONE appears in /workspace/lab/setup.log (usually seconds; at most 20 minutes). Print the last 8 lines of setup.log. If torch is not CUDA-enabled or fla is missing, try once to fix it inside /workspace/venv only.
2. cd /workspace/lab, delete a file named STOP if present, kill any old lab_worker.py process, then start `/workspace/venv/bin/python lab_worker.py` with nohup and setsid, environment variable LAB_MAX_PARALLEL=2, stdout/stderr appended to worker.log, in the background; save its PID to worker.pid.
3. Wait 90 seconds, then print /workspace/lab/heartbeat.json, `nvidia-smi --query-gpu=name,uuid,memory.total,memory.used --format=csv`, and `free -m | head -2`.
4. Write /workspace/lab/supervise_report.json with: torch/fla versions, cuda available, GPU name and uuid, worker PID, start time. Then END this session immediately (do not supervise; the worker keeps running on its own). Print the report path as the last line.
