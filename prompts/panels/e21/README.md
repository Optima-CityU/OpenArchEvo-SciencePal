# Data for a mechanism study: why does a "spiking output gate" learn fast but end up worse?

Task: spiking (SNN) language model on WikiText-2, 2 layers, d_model 128-ish, SLTT spiking neurons; all linear layers take binary spikes.
The training script and its hyper-parameters (optimizer, LR schedule, dropout, batch size, 30 epochs) are FIXED; only the
block program (`SpikingFLABlock` in a .py file) can change.

Architectures (all trained with the same 30-epoch schedule, seeds 0/1/2):
- `DeltaNet` = `i00-deltanet.py` (expert design). 30-epoch val PPL 71.84 (mean of 3 seeds).
- `OutputGate` = `r03a-263bee.py`: DeltaNet + a spiking gate `o = o * sigmoid(SpikingLinear(hidden_states))` (+12% params).
  Found by an evolutionary search that scored children by 2-epoch val PPL. 30-epoch val 73.54.
- `elu+1` = `r04v2-bf4d6a.py`: DeltaNet with elu+1 feature map on q/k (no extra params). 30-epoch val 70.75.
- `ParamMatched` = DeltaNet with a wider MLP (+12% params). 30-epoch val 70.74.
- `UseGate` = DeltaNet with its built-in (non-spiking-input) output gate switched on (+12% params). 30-epoch val 70.66.

`curves.csv`: arch, seed, epoch, train_ppl, val_ppl for every epoch of every run (train ppl is measured with dropout on).
In separate short runs: 2-epoch val PPL DeltaNet 120.64, OutputGate 117.66; 6-epoch (3 seeds) DeltaNet 94.50, OutputGate 91.51.
