# One sigmoid, 4.6 perplexity: why?

Native spiking sequence-model blocks (every nn.Linear is fed binary spikes through `SpikingLinear`), DeltaNet family,
trained on WikiText-2 (seq 128, SLTT neurons). Metric: best validation perplexity over 30 epochs (lower is better).

`r02a-e29092.py` is the new champion. After the output projection it computes `gate = self.gate_proj(hidden_states)`
(a SpikingLinear: spikes of the block input -> linear, so the gate is real-valued, can be negative or > 1) and returns
`o = o * gate`. The old "output gate" r03a-263bee is the *identical* program with one extra line `gate = torch.sigmoid(gate)`.
Same parameter count (38,224 per block at demo size; DeltaNet 34,128).

| variant (where the gate sits, its activation) | 30-epoch val PPL (seeds) | 2-epoch val PPL |
|---|---|---|
| no gate (DeltaNet) | 71.82 (6 runs, 3 seeds) | 120.64 |
| DeltaNet built-in gated norm (`use_gate`) | 70.66 (3) | — |
| DeltaNet widened to the same params, no gate | 70.74 (3) | — |
| after o_proj, sigmoid (old output gate) | 73.54 (3) | 117.66 |
| after o_proj, 1 + tanh | 71.96 (2) | — |
| before o_proj (per head), sigmoid | 70.72 (2) | 118.83 |
| **after o_proj, identity (champion)** | **68.95 (3: 69.02, 68.88, 68.95)** | 117.70 |

`curves.csv`: per-epoch train and val PPL of these 30-epoch runs (train PPL is the running average with dropout on).

Three more variants of the champion are queued on the GPU (results unknown to anyone yet), changing only the gate's
activation after o_proj: `2*sigmoid(g)`, `tanh(g)`, `abs(g)`.
