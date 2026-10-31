# Finding to check against the literature
Spiking (SNN) language model, WikiText-2, small scale. An evolutionary architecture search scored candidates by 2-epoch
validation perplexity. Its best discovery added a sigmoid output gate (input-dependent, per-channel, computed from the
block input through a spiking linear layer) to a DeltaNet (delta-rule linear attention) block.
- 2 and 6 epochs: gate ~3 PPL better than DeltaNet; at epoch 5 of a 30-epoch run, 9 PPL better.
- 30 epochs: gate 1.7 PPL WORSE than DeltaNet; training PPL stays 9 lower (overfitting).
- Parameter-matched controls (+12% params elsewhere, or DeltaNet's own built-in output gate) end ~1.1 PPL better than
  DeltaNet; a parameter-free change (elu+1 feature map on q/k) also ends 1.1 better.
Conclusion we draw: short-training proxies reward fast early learners; multi-fidelity confirmation is needed.
