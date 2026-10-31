You are the lab technician for an evolutionary architecture-search experiment (OpenArchEvo, native spiking sequence models). Work unattended; do not ask questions; do not use web search.

Two files are attached to this message: `lab_setup.sh` and `lab_worker.py`. First locate them (they are usually in /workspace; otherwise `find / -name lab_worker.py -not -path '*/proc/*' 2>/dev/null`) and copy both into `/workspace/lab/`. Then:

1. Run `bash /workspace/lab/lab_setup.sh` in the background with output to `/workspace/lab/setup.out` and poll it until `SETUP_DONE` appears in `/workspace/lab/setup.log` (it installs a CUDA 12.8 PyTorch build, flash-linear-attention, clones the public OpenArchEvo repository and prepares WikiText-2; it may take ~10 minutes). Then show the last 15 lines of `/workspace/lab/setup.log`. The root disk is only 10 GB: never use the pip cache (`--no-cache-dir`), and do not install other CUDA builds of torch. If torch with CUDA still is not available afterwards, diagnose and fix it with a cu128 wheel, then continue.
2. Verify the GPU path with: `cd /workspace/lab/OpenArchEvo && python -c "import torch,fla;print(torch.__version__,torch.cuda.is_available(),fla.__version__)"`.
3. Start the queue worker detached so that it keeps running after you finish:
   `cd /workspace/lab && LAB_MAX_PARALLEL=4 nohup python3 lab_worker.py > /workspace/lab/worker.log 2>&1 &`
   Wait 40 seconds and print `/workspace/lab/heartbeat.json` and the tail of `worker.log`.
4. Write `/workspace/lab/lab_start.json` with: torch/fla versions, cuda availability, disk free, worker PID, heartbeat content, and any problems you fixed. Print its path as the last line.

If a command fails repeatedly, do not loop: record the error in lab_start.json and finish. Remove any core dump files (`core.*`) you create.
