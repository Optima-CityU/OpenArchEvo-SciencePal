# Research-lead dossier, generation 4

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
| r02e40l-084566 | 2 | explore | i09-s0557 |  |  | 34144 | not trained | {"added": ["spike_decay_gate: per-head multiplicative gate derived from aggregated spikes, applied inside delta_rule_chunkwise to scale v and k before state update"], "removed": ["post-recurrence multiplicative modulatio |
| r02e40l-31b232 | 2 | recombine | r01e40l-5bc96d,i10-s0337 | 85.6 | 2.24 | 58698 | trained | {"added": ["g_proj: SpikingLinear(d_inner, num_heads*d_head) for per-head SiLU gating before o_proj"], "removed": [], "inherited": {"r01e40l-5bc96d": ["shared B/C projection (single SpikingLinear split into B and C)", "d |
| r02e40l-355427 | 2 | explore | i00-mamba2-ssd |  |  | 58632 | not trained | {"added": ["Spike-activity residual gate: computes a scalar gate from the spike outputs of the SSD branch and multiplies the SSD output before the residual addition."], "removed": ["Reduced d_state from 64 to 32."], "inh |
| r02e40l-3c1104 | 2 | explore | i00-deltanet | 83.56 | 1.53 | 34136 | trained | {"added": ["Spike-based output gate: reuse b_proj spikes to compute a per-head gate, applied via FusedRMSNormGated before o_proj."], "removed": [], "inherited": {"i00-deltanet": ["SpikingLinear for q,k,v,o projections",  |
| r02e40l-514346 | 2 | simplify | i09-s0557 |  |  | 34140 | not trained | {"added": [], "removed": ["spike_gate_mod_proj", "spike_gate_modulation computation"], "inherited": {"i09-s0557": ["DeltaNet core with delta_rule_chunkwise", "SpikingLinear for q/k/v projections", "spike_beta_mod_proj (m |
| r02e40l-898c52 | 2 | recombine | i09-s0557,i03-s0335 |  |  | 58584 | not trained | {"added": ["per-head spike-driven beta modulation (from i09) applied to Mamba2 decay a"], "removed": ["attn_residual_gate", "mlp_residual_gate"], "inherited": {"i03-s0335": ["mamba2_ssd_parallel_stable", "SpikingMamba2 c |
| r02e40l-ab6fe4 | 2 | simplify | i03-s0335 | 87.3 | 2.03 | 42154 | trained | {"added": [], "removed": ["Separate B and C projections", "d_inner expansion from 2 to 1.5"], "inherited": {"i03-s0335": ["SpikingResidualGate", "SpikingMamba2 structure", "ShortConvolution", "RMSNorm", "SpikingMLP with  |
| r02e40l-bef273 | 2 | recombine | i00-deltanet,r01e40l-87f5b6 |  |  | 34128 | not trained | {"added": ["vector-state recurrence S = a*S + k*v per head (from 87f5b6)"], "removed": ["DeltaNet's matrix-state recurrence S = S + k^T @ u"], "inherited": {"i00-deltanet": ["spiking q/k/v/b projections", "ShortConvoluti |
| r02e40l-c1f943 | 2 | simplify | r01e40l-87f5b6 | 85.61 | 1.33 | 33546 | trained | {"added": [], "removed": ["ShortConvolution on q", "separate D parameter"], "inherited": {"r01e40l-87f5b6": ["GLA-style vector-state recurrence", "single shared projection", "SpikingResidualGate modules", "SpikingMLP", " |
| r03e40l-06c60f | 3 | explore | r02e40l-ab6fe4 |  |  | 48298 | not trained | {"added": ["SpikingValueGate: a SpikingLinear (d_model -> d_model) that reads in_proj spikes and modulates v before the Mamba2 state update"], "removed": [], "inherited": {"r02e40l-ab6fe4": ["Shared B/C projections", "d_ |
| r03e40l-1fc3a4 | 3 | simplify | r01e40l-a21814 | 81.77 | -0.85 | 28624 | trained | {"added": [], "removed": [], "inherited": {"r01e40l-a21814": ["shared qk_proj", "no b_proj", "delta_rule_chunkwise core", "value path unchanged"]}} |
| r03e40l-42c0fb | 3 | recombine | r01e40l-5bc96d,r02e40l-514346 |  |  | 50774 | not trained | {"added": ["Aggregation of spike activity from in_proj, conv, and out_proj into three scalar means", "dt_proj now takes hidden_states concatenated with aggregated spike activity (input dim d_model + 3)", "Spike-condition |
| r03e40l-618244 | 3 | explore | i00-deltanet |  |  | 38224 | not trained | {"added": ["SpikeActivityProj (SpikingLinear from hidden_size to hidden_size) and aggregation of q/k/v spikes for beta modulation"], "removed": [], "inherited": {"i00-deltanet": ["DeltaNet block structure", "SpikingLinea |
| r03e40l-8c56f7 | 3 | simplify | r01e40l-5bc96d |  |  | 50441 | not trained | {"added": [], "removed": ["SpikingResidualGate module from MLP path (proj_in SpikingLinear(1, d_model//2), proj_out SpikingLinear(d_model//2, 1), bias parameter)"], "inherited": {"r01e40l-5bc96d": ["SpikingMamba2 attenti |
| r03e40l-946697 | 3 | recombine | r02e40l-ab6fe4,i09-s0557 | 88.28 | 0.98 | 42538 | trained | {"added": ["spike_gate_mod_proj: SpikingLinear(3 -> d_inner) that maps aggregated q/k/v spike mean activity to a gate modulation signal", "Multiplicative output gate modulation: y = y * (1 + sigmoid(spike_gate_mod)) appl |
| r03e40l-fb0977 | 3 | explore | r01e40l-5bc96d |  |  | 50522 | not trained | {"added": ["SpikingLinear gate projection from in_proj spikes to per-head decay modulation"], "removed": [], "inherited": {"r01e40l-5bc96d": ["shared B/C projection", "d_state=32", "SpikingResidualGate modules", "Mamba2  |

## Knowledge
No archive knowledge is provided in this run: base your directions only on the results table above.

## Fixed slots for this generation

- slot 1: island 0, role explore, parents ['i00-deltanet'], weight [quality, size] = [1.0, 0.0]
- slot 2: island 0, role recombine, parents ['r01e40l-cf625f', 'r01e40l-87f5b6'], weight [quality, size] = [0.938, 0.062]
- slot 3: island 0, role simplify, parents ['r02e40l-ab6fe4'], weight [quality, size] = [0.875, 0.125]
- slot 4: island 1, role recombine, parents ['r03e40l-1fc3a4', 'r02e40l-31b232'], weight [quality, size] = [0.812, 0.188]
- slot 5: island 1, role simplify, parents ['r02e40l-bef273'], weight [quality, size] = [0.75, 0.25]
- slot 6: island 1, role explore, parents ['i03-s0335'], weight [quality, size] = [0.688, 0.312]
- slot 7: island 2, role simplify, parents ['r01e40l-cf625f'], weight [quality, size] = [0.625, 0.375]
- slot 8: island 2, role explore, parents ['r03e40l-1fc3a4'], weight [quality, size] = [0.562, 0.438]
- slot 9: island 2, role recombine, parents ['i03-s0335', 'r03e40l-1fc3a4'], weight [quality, size] = [0.5, 0.5]
