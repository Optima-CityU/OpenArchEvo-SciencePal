Unattended capability probe for an engineering integration. Do not ask me questions; if something fails, record the failure and continue. Do not use web search. Keep it short: run commands, do not write a report.

One zip file is attached: openarchevo_snn_proposer.zip (a Skill package with Python scripts). WikiText-2 raw text is NOT attached: in Step 2 download it with python from https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/{train,validation}-00000-of-00001.parquet (concatenate the `text` column into oae_probe/data/wt2_train.txt and wt2_validation.txt; pip install pyarrow if needed). If the download fails, skip the micro_train command and record why.

Step 1. In your sandbox terminal, in a new directory named `oae_probe`, run and capture the output of:
  pwd; whoami; uname -a; nproc; free -m; df -h .; (nvidia-smi || echo NO_GPU); python3 --version; python3 -c "import torch,sys;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'threads',torch.get_num_threads())" ; python3 -c "import numpy,sklearn,einops;print('np',numpy.__version__,'sk',sklearn.__version__,'einops ok')"
  for u in https://huggingface.co https://pypi.org/simple/ https://download.pytorch.org/whl/cpu/; do curl -s -o /dev/null -w "%{http_code} $u\n" --max-time 15 $u; done
Step 2. Find the attached zip file (search the filesystem if needed, e.g. `find / -name "openarchevo_snn_proposer.zip" 2>/dev/null`), record their absolute paths, and unzip it into `oae_probe/skill`.
Step 3. If torch is missing, run `pip install torch --index-url https://download.pytorch.org/whl/cpu` and record how long it took. If einops is missing, `pip install einops`.
Step 4. Inside `oae_probe/skill`, run with `time`:
  python3 scripts/cpu_check.py references/library/code/i09-s0557.py --json check_neurogate.json
  python3 scripts/micro_train.py references/library/code/i00-deltanet.py --data ../data --steps 20 --json micro20.json
Step 5. Write `oae_probe/probe.json` containing: every command above with its exit code and (truncated to 2000 chars) output, the zip paths, the absolute path of `oae_probe`, the wall time of each step, and the full content of check_neurogate.json and micro20.json. Then print the absolute path of probe.json as the last line of your answer.
