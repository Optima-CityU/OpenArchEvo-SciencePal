# Research-lead dossier, generation 2

## Results table (all candidates; score = overfitting-aware 2-epoch fitness OAF = val + 0.5*(val - train), lower is better; it predicts 30-epoch quality better than val PPL; delta = vs first parent; params at CPU size)

| id | gen | role | parents | val PPL | delta vs parent | params/block | status | mechanism / hypothesis |
|---|---|---|---|---|---|---|---|---|
| i00-deltanet | 0 | seed |  | 82.03 |  | 34128 | seed |  |
| i00-hgrn2 | 0 | seed |  | 86.84 |  | 29568 | seed |  |
| i00-mamba2-ssd | 0 | seed |  | 88.77 |  | 58568 | seed |  |
| i03-s0335 | 0 | seed |  | 85.27 |  | 58698 | seed |  |
| i09-s0557 | 0 | seed |  | 89.26 |  | 34144 | seed |  |
| i10-s0337 | 0 | seed |  | 85.35 |  | 81488 | seed |  |
| r01e40l-5bc96d | 1 | simplify | i03-s0335 | 83.36 | -1.91 | 50506 | trained | {"added": ["single shared SpikingLinear for B and C"], "removed": ["separate SpikingLinear for C", "d_state reduced from 64 to 32"], "inherited": {"i03-s0335": ["spiking in_proj and out_proj", "negative dt_bias initializ |
| r01e40l-87f5b6 | 1 | explore | i03-s0335 | 84.28 | -0.98 | 33802 | trained | {"added": ["GLA-style vector-state recurrence (S = a*S + k*v, output = q*S) replacing Mamba2 SSD matrix-state recurrence"], "removed": ["Mamba2 SSD chunked matrix-state recurrence", "expand factor of 2 (d_inner = 2*d_mod |
| r01e40l-98a122 | 1 | explore | i03-s0335 |  |  | 75336 | not trained | {"added": ["ShortConvolution on the value path", "SiLU-gated output projection (gate before o_proj)", "Spike-activity regularization term"], "removed": ["SpikingResidualGate modules for attention and MLP blocks", "Sigmoi |
| r01e40l-a21814 | 1 | simplify | i00-deltanet | 82.62 | 0.6 | 29776 | trained | {"added": ["shared qk_proj replacing separate q_proj and k_proj"], "removed": ["b_proj", "use_beta", "expand_k reduction from 1.0 to 1.0 (explicitly set)"], "inherited": {"i00-deltanet": ["delta_rule_chunkwise", "short c |
| r01e40l-cf625f | 1 | explore | i10-s0337 |  |  | 34128 | not trained | {"added": ["beta_proj: SpikingLinear(d_model, num_heads) for the per-head delta-rule decay gate", "delta_rule_chunkwise recurrence for the state update and output", "identity-centred output gate (1 + tanh) applied before |

## Knowledge
No archive knowledge is provided in this run: base your directions only on the results table above.

## Fixed slots for this generation

- slot 1: island 0, role explore, parents ['i09-s0557'], weight [quality, size] = [1.0, 0.0]
- slot 2: island 0, role recombine, parents ['i00-deltanet', 'r01e40l-87f5b6'], weight [quality, size] = [0.938, 0.062]
- slot 3: island 0, role simplify, parents ['r01e40l-87f5b6'], weight [quality, size] = [0.875, 0.125]
- slot 4: island 1, role recombine, parents ['i09-s0557', 'i03-s0335'], weight [quality, size] = [0.812, 0.188]
- slot 5: island 1, role simplify, parents ['i03-s0335'], weight [quality, size] = [0.75, 0.25]
- slot 6: island 1, role explore, parents ['i00-deltanet'], weight [quality, size] = [0.688, 0.312]
- slot 7: island 2, role simplify, parents ['i09-s0557'], weight [quality, size] = [0.625, 0.375]
- slot 8: island 2, role explore, parents ['i00-mamba2-ssd'], weight [quality, size] = [0.562, 0.438]
- slot 9: island 2, role recombine, parents ['r01e40l-5bc96d', 'i10-s0337'], weight [quality, size] = [0.5, 0.5]
