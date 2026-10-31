"""Build the tokenized WikiText-2 cache once (GPT-2 BPE, seq 512), so parallel training jobs do not race."""
import sys, os, yaml
sys.path.insert(0, "/workspace/lab/OpenArchEvo/training"); sys.path.insert(0, "/workspace/lab/OpenArchEvo")
os.chdir("/workspace/lab/OpenArchEvo")
from train_wikitext2 import load_data
cfg = yaml.safe_load(open("configs/training/wikitext2.yaml"))
d = load_data(cfg)
print({k: len(v) for k, v in d.items()})
