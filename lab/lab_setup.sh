#!/usr/bin/env bash
# Idempotent environment setup for the SciencePal sandbox lab (H20, driver 570 / CUDA 12.8).
# Facts learned on 2026-10-21:
#  * system packages (/opt/venv) are RESET between agent runs; only /workspace persists
#    -> everything lives in a private venv at /workspace/venv (no system site-packages)
#  * default PyPI torch (2.14) is built for a newer CUDA than the driver -> use cu128 wheels
#  * the root disk is 10 GB and pip's cache fills it -> always --no-cache-dir
#  * the sandbox cgroup has ~4 GiB RAM -> keep job parallelism low
set -u
LAB=${LAB_DIR:-/workspace/lab}
VENV=/workspace/venv
mkdir -p "$LAB" && cd "$LAB"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LAB/setup.log"; }
export PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
# the sandbox pip config points at a slow mirror (~100 kB/s observed); pypi.org is reachable
export PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.org/simple}

log "disk before: $(df -h /workspace | tail -1)  mem limit: $(cat /sys/fs/cgroup/memory.max 2>/dev/null)"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV" && log "created venv"
fi
PY="$VENV/bin/python"
if ! "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  log "installing torch cu128 into venv (no pip cache: the 10 GB disk fills up and leaves a broken torch)"
  rm -rf "$HOME/.cache/pip" 2>/dev/null
  "$PY" -m pip uninstall -y torch >>"$LAB/setup.log" 2>&1
  timeout 1200 "$PY" -m pip install --no-cache-dir "torch==2.8.0" >>"$LAB/setup.log" 2>&1 \
  || "$PY" -m pip install --no-cache-dir "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128 >>"$LAB/setup.log" 2>&1 \
    || "$PY" -m pip install --no-cache-dir "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu128 >>"$LAB/setup.log" 2>&1
fi
"$PY" -c "import einops, yaml, sklearn, transformers, datasets, pyarrow" 2>/dev/null || \
  "$PY" -m pip install --no-cache-dir einops pyyaml scikit-learn numpy "transformers>=4.40,<5" "datasets>=2.16" pyarrow >>"$LAB/setup.log" 2>&1
# --no-deps so fla cannot replace the cu128 torch
# pin fla 0.5.2: fla 0.6.0 needs a newer triton (lab5/lab6 failures, 10-25)
"$PY" -c "import fla.modules" 2>/dev/null || "$PY" -m pip install --no-cache-dir --no-deps "flash-linear-attention==0.5.2" "fla-core==0.5.2" >>"$LAB/setup.log" 2>&1
log "torch: $("$PY" -c 'import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())' 2>&1 | tail -1)"
log "fla:   $("$PY" -c 'import fla;print(fla.__version__)' 2>&1 | tail -1)"

[ -d OpenArchEvo ] || git clone --depth 1 https://github.com/Optima-CityU/OpenArchEvo.git >>"$LAB/setup.log" 2>&1
log "repo: $(cd OpenArchEvo && git log --oneline -1)"

D=OpenArchEvo/data/wikitext-2
if [ ! -s "$D/wiki.test.tokens" ]; then
  mkdir -p "$D"
  "$PY" - <<'EOF' >>"$LAB/setup.log" 2>&1
import io, urllib.request, pyarrow.parquet as pq
base = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-v1/"
for split, name in (("train", "train"), ("validation", "valid"), ("test", "test")):
    raw = urllib.request.urlopen(base + f"{split}-00000-of-00001.parquet", timeout=120).read()
    rows = pq.read_table(io.BytesIO(raw)).column("text").to_pylist()
    with open(f"OpenArchEvo/data/wikitext-2/wiki.{name}.tokens", "w", encoding="utf-8") as f:
        f.write("".join(rows))
    print(split, len(rows))
EOF
fi
log "data: $(wc -l $D/wiki.*.tokens 2>/dev/null | tail -1)"
if [ ! -d OpenArchEvo/data/cache ] || [ -z "$(ls -A OpenArchEvo/data/cache 2>/dev/null)" ]; then
  (cd "$LAB" && "$PY" prep_data.py >>"$LAB/setup.log" 2>&1) && log "tokenized cache built"
fi
log "disk after: $(df -h /workspace | tail -1)"
log "SETUP_DONE"
