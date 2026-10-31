# Predicting 30-epoch quality from a 2-epoch training run

Context: an evolutionary search over native spiking sequence-model blocks (programs defining `SpikingFLABlock`) uses a
cheap fitness: validation perplexity after 2 epochs of WikiText-2 training (protocol `sp-wt2-sltt-s128-e2`; lower is
better). The quantity we care about is validation perplexity after 30 epochs (`sp-wt2-sltt-s128-e30-rank`, seed 0).
On architectures we have checked, the 2-epoch ranking is a poor guide to the 30-epoch ranking.

Files:
- `train.csv`: 17 architectures with their 2-epoch run (train and val PPL after epoch 1 and 2; train PPL is the running
  average over the epoch with dropout on, val PPL is measured at the end of the epoch), CPU features of the program at
  initialisation (`params_per_block`, `fire_rate_init`), and the 30-epoch result `val_ppl_30ep`.
- `test.csv`: 21 further architectures, same columns, **without** the 30-epoch result. Their 30-epoch training is running
  now; nobody knows the results yet. Your predictor will be scored on them.
- `programs.zip`: the program source of every architecture in both files (`programs/<arch>.py`). Parse it with code if
  you use it; do not print whole files.
