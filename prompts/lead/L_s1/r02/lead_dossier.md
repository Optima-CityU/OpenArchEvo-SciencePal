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
| r01e40l-0b5d91 | 1 | simplify | i03-s0335 |  |  | 58568 | not trained | {"added": [], "removed": ["SpikingResidualGate module class", "attn_residual_gate module", "mlp_residual_gate module", "spike-activity collection and averaging", "modulation factor computation and multiplication", "relat |
| r01e40l-54554b | 1 | simplify | i09-s0557 |  |  | 34128 | not trained | {"added": [], "removed": ["spike_beta_mod_proj and spike_gate_mod_proj SpikingLinear layers", "aggregated_spikes computation and sigmoid-based modulation of beta and gate"], "inherited": {"i09-s0557": ["DeltaNet with spi |
| r01e40l-61eaa7 | 1 | simplify | i00-deltanet | 83.03 | 1.0 | 29836 | trained | {"added": [], "removed": ["25% of the width of q_proj, k_proj, v_proj, and o_proj projections"], "inherited": {"i00-deltanet": ["delta_rule_chunkwise", "SpikingLinear wrappers", "ShortConvolution", "RMSNorm", "o_proj", " |
| r01e40l-6ecf88 | 1 | explore | i10-s0337 |  |  | 29760 | not trained | {"added": ["HGRN2-style compact recurrent state (S_t = decay_t * S_{t-1} + k_t * v_t)"], "removed": ["q_proj, f_proj, g_proj projections", "gsa_chunkwise operator", "g_norm"], "inherited": {"i10-s0337": ["SpikingLinear w |
| r01e40l-cabf9f | 1 | recombine | i03-s0335,i10-s0337 | 82.27 | -3.0 | 81618 | trained | {"added": ["GSA attention mechanism with five separate projections", "Per-head gated norm"], "removed": ["Mamba2 attention with large in_proj", "ShortConvolution for Mamba2"], "inherited": {"i03-s0335": ["SpikingResidual |
| r01e40l-cb6155 | 1 | recombine | i09-s0557,i10-s0337 |  |  | 42320 | not trained | {"added": ["GSA attention with num_slots=16 instead of 64"], "removed": ["DeltaNet's chunkwise delta rule with beta modulation"], "inherited": {"i09-s0557": ["SpikingMLP structure", "spike regularization mechanism", "blo |
| r01e40l-e7979c | 1 | explore | i03-s0335 | 85.42 | 0.16 | 62794 | trained | {"added": ["A per-channel input modulation gate (SpikingLinear d_model -> d_model/2 -> d_model with sigmoid) that scales x_attn based on input spike activity"], "removed": [], "inherited": {"i03-s0335": ["SpikingMamba2 a |

## Knowledge
No archive knowledge is provided in this run: base your directions only on the results table above.

## Fixed slots for this generation

- slot 1: island 0, role explore, parents ['r01e40l-61eaa7'], weight [quality, size] = [1.0, 0.0]
- slot 2: island 0, role recombine, parents ['i00-mamba2-ssd', 'r01e40l-cabf9f'], weight [quality, size] = [0.938, 0.062]
- slot 3: island 0, role simplify, parents ['r01e40l-cabf9f'], weight [quality, size] = [0.875, 0.125]
- slot 4: island 1, role recombine, parents ['r01e40l-54554b', 'r01e40l-cb6155'], weight [quality, size] = [0.812, 0.188]
- slot 5: island 1, role simplify, parents ['r01e40l-cabf9f'], weight [quality, size] = [0.75, 0.25]
- slot 6: island 1, role explore, parents ['r01e40l-cabf9f'], weight [quality, size] = [0.688, 0.312]
- slot 7: island 2, role simplify, parents ['r01e40l-54554b'], weight [quality, size] = [0.625, 0.375]
- slot 8: island 2, role explore, parents ['i00-deltanet'], weight [quality, size] = [0.562, 0.438]
- slot 9: island 2, role recombine, parents ['i00-mamba2-ssd', 'i00-deltanet'], weight [quality, size] = [0.5, 0.5]
