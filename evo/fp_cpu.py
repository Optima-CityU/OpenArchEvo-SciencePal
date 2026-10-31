import sys, json, time
import os
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path[:0]=[os.environ.get('OAE_SKILL_DIR', str(_ROOT / 'skills' / 'runtime')) + '/scripts/cpu_shims',
              os.environ.get('OPENARCHEVO_DIR', str(_ROOT.parent / 'OpenArchEvo'))]
import torch; torch.set_num_threads(1)
from openarchevo.config import FingerprintConfig, ProbeConfig, WrapperConfig
from openarchevo.fingerprint.extract import extract_fingerprint
from openarchevo.fingerprint.schema import FEATURE_KEYS
cfg = FingerprintConfig(seed=42, neuron_type='bptt', devices=['cpu'],
    probe=ProbeConfig(batch_size=int(sys.argv[3]) if len(sys.argv)>3 else 8, seq_len=int(sys.argv[2]) if len(sys.argv)>2 else 256),
    wrapper=WrapperConfig(), structure_seq_len=int(sys.argv[4]) if len(sys.argv)>4 else 512)
t=time.time(); fp=extract_fingerprint(open(sys.argv[1]).read(), cfg, device='cpu')
print(json.dumps({'s':round(time.time()-t,1), **dict(zip(FEATURE_KEYS, fp))}))
