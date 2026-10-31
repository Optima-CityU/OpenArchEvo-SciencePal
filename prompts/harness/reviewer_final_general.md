The attached review_final.zip contains two folders: review/arm_B (population search with islands, trained candidates of generations 2-3) and review/arm_A (best-only search that always mutates the current best, trained candidates of generations 1-2). Each has generation.json and code/*.py. Find the zip (usually in /workspace) and extract it under /workspace so that /workspace/review/arm_A/generation.json and /workspace/review/arm_B/generation.json exist.

Work unattended; do not ask questions; do not use web search. Do not load any skill. The block contract is:


```python
class SpikingFLABlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ffn_mult, dropout, neuron_type,
                 prenorm=True, use_spiking_mlp=True, layer_idx=None, **kwargs): ...
    def forward(self, x):          # x: (B, L, d_model)
        return y, reg_loss, spikes # y: (B, L, d_model); reg_loss: scalar tensor or None;
                                   # spikes: 1-D tensor of ALL binary spikes in the block, or None
```

Hard rules (a candidate violating any of them is rejected):
1. Causal: output at position t may depend only on inputs at positions <= t.
2. Sub-quadratic in sequence length (no full softmax attention over L x L).
3. Spiking: inputs to linear projections are binary spikes from the provided neurons
   (`neuron_registry[neuron_type]` from `src.models.spike.neuron`, or `SpikingLinear`).
   Keep `neuron_type` a constructor argument; never hard-code `"sdn"` or `"bptt"`.
4. Works for any batch size and sequence length; no hard-coded shapes; no `torch.compile`, no fp16.
5. Allowed imports: `torch`, `einops`, `math`, `typing`, `fla.modules` (RMSNorm,
   FusedRMSNormGated, ShortConvolution, l2norm), `fla.models.utils`, `fla.layers.utils`,
   and the legacy `src.models.*` helpers already used by the parents. Nothing else.
6. Keep the class name `SpikingFLABlock` and keep `**kwargs` in `__init__`.


 Context is limited: never print whole large files (never cat references/library.jsonl or generation.json in full); inspect code files one at a time with grep or partial reads, and keep notes short.

Important new rule: contract rule 3 requires every nn.Linear to receive binary spikes. Our new audit found candidates that feed analog activations to nn.Linear (e.g. replacing the spiking MLP with a standard MLP); such candidates are invalid even if their perplexity is better. Check this explicitly for every candidate by reading the code.

For each candidate in both arms: (1) does the code implement the hypothesis (yes/partly/no); (2) contract risks, especially non-binary inputs to linear layers, future leakage, hidden O(L^2); (3) novelty given the similarity values (genuinely different / re-parameterisation / near-copy); (4) verdict keep / keep-but-fix / discard with one reason.

Then write a comparison (<= 250 words) of what each arm explored: diversity of mechanisms, how many failed or diverged, which mechanism families look promising, and what you would do in the next generation of each arm. Quote measured numbers exactly; do not invent results; a single run per arm is exploratory evidence only.

Write /workspace/review/review_final.json with fields arm_A: [...], arm_B: [...], comparison. Print its path as the last line.
