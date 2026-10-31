# Contract and protocols (reference)

## Block contract

See SKILL.md §1. The block is wrapped by a fixed model:
`embedding -> n_layers x SpikingFLABlock -> RMSNorm -> linear head`.
Only the block is searched. `forward(x)` must return `(y, reg_loss, spikes)`.

Constructor arguments passed by every harness:
`d_model, num_heads, d_ffn_mult, dropout, neuron_type, prenorm=True, use_spiking_mlp=True, layer_idx=<int>`.

Neurons: `from src.models.spike.neuron import registry as neuron_registry` then
`neuron_registry[neuron_type](surrogate_function=piecewise_quadratic_surrogate())`.
Neurons take tensors whose LAST dimension is time, e.g. `(B, D, L)`, and return binary
spikes of the same shape. Available types: `bptt`, `sltt` (CPU and GPU), `sdn` (GPU research
protocol; needs a checkpoint that is not available here).

## Protocols (never mix their numbers)

| protocol | where | size | neuron | data | metric |
|---|---|---|---|---|---|
| `cpu-demo-v1` | SciencePal sandbox / any CPU | d_model 64, 1 block, L 64, B 2 | bptt | random tensors | pass/fail, params_per_block, fire_rate at init, causality |
| `cpu-micro-v1` | SciencePal sandbox / any CPU | 2 blocks, d_model 64, L 128, B 16, 300 steps | bptt | WikiText-2 raw, bytes | validation bits-per-byte (bpb) |
| `research-wt2` | GPU (controller) | 6 blocks, d_model 256 | sdn | WikiText-2, GPT-2 BPE | best validation PPL; test only after freeze |

Historical numbers in `library/index.json` (`hist_wt2_ppl`) come from the 2026 OpenArchEvo
paper's archive; their split provenance is under audit. They are background only.
