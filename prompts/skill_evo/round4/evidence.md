# New GPU evidence for revising the OpenArchEvo proposer Skill (v0.5.1-cards), 10/29–10/30

All numbers: WikiText-2, SLTT neurons, seq 128; "30-epoch" = best validation PPL within 30 epochs (protocol sp-wt2-sltt-s128-e30-rank); lower is better. DeltaNet expert: 71.8.

## 1. Confirmed architecture (3 seeds each)
- r02a-e29092 = DeltaNet + spiking multiplicative gate AFTER o_proj with NO activation (gate = Linear(spike(h))): 68.95 (69.02/68.88/68.95). Matched-parameter controls: widened DeltaNet 70.74, DeltaNet use_gate 70.66.
- The same program with `gate = sigmoid(gate)` (r03a-263bee): 73.54. Their 2-epoch scores are equal (117.70 vs 117.66).
- Other confirmed: pre-o_proj sigmoid gate + state-update gate 70.40; elu+1 q/k feature map (no extra params) 70.75.

## 2. Gate activation after o_proj (30 epochs; seeds in brackets)
identity 68.95 (3); g/sqrt(1+0.1 g^2) 69.20 (3); abs 69.70 (2); tanh 70.16 (2); 1+g 70.54 / 71.04; SiLU g*sigmoid(g) 71.75 (1);
1+tanh 71.96 (2); sigmoid(g)-0.5 72.88 (1); sigmoid 73.54 (3); 2*sigmoid 74.14 (1).
Pre-registered explanations that FAILED: "signed gates are needed", "amplifiable gates suffice", "zero-resting gates suffice".
The SiLU gate was proposed by 4 of 5 models and failed. Only "no sigmoid at this position" is established.

## 3. Proxy fitness
- 2-epoch val PPL vs 30-epoch: Spearman 0.25 (n=12) and 0.00 over 25 search runs' delivered architectures; 6 epochs copies the 2-epoch ranking (0.94).
- Overfitting-aware score OAF = val2 + 0.5*(val2 - train2) from the same 2-epoch log: Spearman 0.63 on 28 held-out architectures (val2: -0.02).
  Prospective selection of 4 from 16 candidates: OAF picks average 69.9 (4/4 beat DeltaNet); 2-epoch picks 74.7 (1/4).
- Implication for proposers: an idea that only speeds up early learning (lower 2-epoch val with much lower train PPL) is a warning sign; argue about behaviour over 30 epochs.

## 4. v0.5.1 in use
- Fixed CPU benchmark: first-try pass 33/47 (70%) vs v0.5 36/48 (75%); distinct_new 40 vs 37; 12/47 proposals cited cards.
- One generation from the champion (12 proposals, 2-epoch, OAF top-2 to 30 epochs): 68.90 and 69.34 (both recombinations).
- Search runs whose delivered architectures diverged or overfitted at 30 epochs (e.g. 114.5, 117.8) were all 2-epoch "good" children.
