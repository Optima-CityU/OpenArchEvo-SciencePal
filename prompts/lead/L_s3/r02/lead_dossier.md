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
| r01e40l-13cc3c | 1 | simplify | i00-hgrn2 |  |  | 26160 | not trained | {"added": [], "removed": ["conv_i (input-branch ShortConvolution)", "25% of hidden_dim via expand_ratio 0.75"], "inherited": {"i00-hgrn2": ["element-wise recurrence hgrn2_chunkwise", "logsigmoid gate", "SpikingLinear pro |
| r01e40l-1d2d98 | 1 | explore | i03-s0335 |  |  | 58572 | not trained | {"added": [], "removed": ["Two-layer SpikingResidualGate, replaced by single spiking linear gate with one SpikingLinear layer per residual path"], "inherited": {"i03-s0335": ["SpikingMamba2 attention block", "SpikingMLP  |
| r01e40l-33e606 | 1 | simplify | i03-s0335 |  |  | 58601 | not trained | {"added": ["shared single SpikingResidualGate with hidden width d/4"], "removed": ["second SpikingResidualGate for MLP path"], "inherited": {"i03-s0335": ["SpikingMamba2 attention block", "SpikingMLP block", "RMSNorm pre |
| r01e40l-5f6403 | 1 | simplify | i00-mamba2-ssd |  |  | 54312 | not trained | {"added": ["shared B/C projection SpikingLinear(d_model, d_inner) with two analog heads (d_inner -> d_inner) for B and C"], "removed": ["separate B and C projections in in_proj (width d_inner each)", "expand=2 in favor o |
| r01e40l-9149c3 | 1 | recombine | i10-s0337,i03-s0335 | 86.01 | 0.66 | 68497 | trained | {"added": ["SpikingResidualGate module for attention residual, transplanted from i03-s0335", "Mean spike activity computation from attention spikes", "Modulation of x_attn before residual addition"], "removed": ["Num_slo |
| r01e40l-ab6c1d | 1 | recombine | i00-mamba2-ssd,i03-s0335 |  |  | 58573 | not trained | {"added": ["SpikingResidualGate for state-decay modulation", "spike-activity calculation for in_proj and out_proj", "decay modulation factor applied to the a tensor"], "removed": ["SpikingResidualGate applied to residual |
| r01e40l-b09a30 | 1 | recombine | i00-deltanet,i03-s0335 | 83.35 | 1.32 | 34193 | trained | {"added": ["SpikingResidualGate module applied to the attention residual path, computing a modulation factor from the mean spiking activity of the attention block"], "removed": [], "inherited": {"i00-deltanet": ["DeltaNe |
| r01e40l-b4a8ba | 1 | explore | i00-deltanet |  |  | 34128 | train failed | {"added": ["sum-norm applied to q/k instead of l2norm", "beta folded into the value projection"], "removed": ["b_proj SpikingLinear", "l2norm on q/k"], "inherited": {"i00-deltanet": ["delta_rule_chunkwise recurrence", "S |

## Knowledge
No archive knowledge is provided in this run: base your directions only on the results table above.

## Fixed slots for this generation

- slot 1: island 0, role explore, parents ['i00-mamba2-ssd'], weight [quality, size] = [1.0, 0.0]
- slot 2: island 0, role recombine, parents ['i09-s0557', 'i10-s0337'], weight [quality, size] = [0.938, 0.062]
- slot 3: island 0, role simplify, parents ['i10-s0337'], weight [quality, size] = [0.875, 0.125]
- slot 4: island 1, role recombine, parents ['r01e40l-1d2d98', 'r01e40l-b4a8ba'], weight [quality, size] = [0.812, 0.188]
- slot 5: island 1, role simplify, parents ['i00-deltanet'], weight [quality, size] = [0.75, 0.25]
- slot 6: island 1, role explore, parents ['i03-s0335'], weight [quality, size] = [0.688, 0.312]
- slot 7: island 2, role simplify, parents ['r01e40l-5f6403'], weight [quality, size] = [0.625, 0.375]
- slot 8: island 2, role explore, parents ['i10-s0337'], weight [quality, size] = [0.562, 0.438]
- slot 9: island 2, role recombine, parents ['r01e40l-b09a30', 'i00-hgrn2'], weight [quality, size] = [0.5, 0.5]
