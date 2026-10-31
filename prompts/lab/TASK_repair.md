Lab REPAIR session. Work unattended; never ask questions; keep messages very short. Only /workspace persists.
1. `touch /workspace/lab/STOP`; kill any running lab_worker.py process.
2. Repair the venv (run each command, wait for it; total may take 15–25 min):
   - `rm -rf ~/.cache/pip`
   - `/workspace/venv/bin/python -m pip uninstall -y torch triton fla-core flash-linear-attention`
   - `/workspace/venv/bin/python -m pip install --no-cache-dir "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128`
   - `/workspace/venv/bin/python -m pip install --no-cache-dir --no-deps "fla-core==0.5.2" "flash-linear-attention==0.5.2"`
   - `/workspace/venv/bin/python -m pip install --no-cache-dir einops`
3. Verify: `/workspace/venv/bin/python -c "import torch, fla, fla.modules; from fla.modules import ShortConvolution, RMSNorm; print(torch.__version__, torch.cuda.is_available(), fla.__version__)"` must print True and 0.5.2. If it fails, try once to fix inside /workspace/venv only.
4. Write /workspace/lab/repair_report.json: {"torch": ..., "cuda": ..., "fla": ..., "verify_ok": true/false, "error": ...}.
5. Only if verify_ok: `rm -f /workspace/lab/STOP` and start the worker: `cd /workspace/lab && LAB_MAX_PARALLEL=1 LAB_GPU_AUTO=1 setsid nohup /workspace/venv/bin/python lab_worker.py >> worker.log 2>&1 &`.
6. End. Print the report as the last line.
