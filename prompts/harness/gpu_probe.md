Unattended GPU feasibility test for an open-source research code base. Do not ask me questions; if a step fails, record the error and continue with the next step. Do not use web search. Do not write a report; just run the steps and save the files.

Work in `/workspace/oae_gpu`. For every step record: command, exit code, wall seconds, last 1500 characters of output. Save them incrementally to `/workspace/oae_gpu/gpu_probe.json` (rewrite the file after each step, so partial results survive).

Step 1. Environment: `nvidia-smi; nproc; free -g; python3 --version`.
Step 2. Install (record time): `pip install torch einops pyyaml scikit-learn "transformers>=4.40" "datasets>=2.16" pyarrow flash-linear-attention` then print `python3 -c "import torch,fla;print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0), fla.__version__)"`.
Step 3. Code: `git clone --depth 1 https://github.com/Optima-CityU/OpenArchEvo.git` then `cd OpenArchEvo && git log --oneline -1 && pip install -e .`.
Step 4. Data (WikiText-2, tokenized v1 release): with python and pyarrow, download
   https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-v1/{train,validation,test}-00000-of-00001.parquet
   and write the `text` column, one row per line exactly as stored, to `OpenArchEvo/data/wikitext-2/wiki.train.tokens`, `wiki.valid.tokens`, `wiki.test.tokens`. Record line counts.
Step 5. One-epoch timing, foreground, from inside `OpenArchEvo`:
   `timeout 1500 python training/train_wikitext2.py --program zoo/experts/deltanet.py --neuron_type sltt --epochs 1 --seed 0 --output_dir /workspace/oae_gpu/out_sltt --result_json /workspace/oae_gpu/r_deltanet_sltt_e1.json`
   Record the wall time and the json content. While it runs, also record one `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv` sample.
Step 6. Same with `--neuron_type bptt` (timeout 1500 s, output files with `bptt` in the name). If it times out, record that.
Step 7. Background survival test: start a detached process that keeps running after you finish:
   `cd /workspace/oae_gpu/OpenArchEvo && nohup python training/train_wikitext2.py --program zoo/discovered/iter_09/i09-s0557_neurogate.py --neuron_type sltt --epochs 3 --seed 0 --output_dir /workspace/oae_gpu/out_bg --result_json /workspace/oae_gpu/r_neurogate_sltt_e3.json > /workspace/oae_gpu/bg_train.log 2>&1 &`
   and also `nohup bash -c 'for i in $(seq 1 240); do echo "$(date -u +%FT%TZ) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)" >> /workspace/oae_gpu/heartbeat.log; sleep 60; done' > /dev/null 2>&1 &`
   Record the PIDs. Do not wait for them to finish.
Step 8. Finalise gpu_probe.json and print its absolute path as the last line of your answer.
