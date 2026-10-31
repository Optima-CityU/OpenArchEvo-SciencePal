---
name: openarchevo-snn-proposer
description: Propose, implement and CPU-check a new native spiking sequence-model block (SpikingFLABlock) from OpenArchEvo parent architectures. Use when asked to explore, recombine or simplify spiking/linear-attention architecture mechanisms, or to produce an OpenArchEvo candidate.json.
version: 0.2.0
---

# OpenArchEvo SNN Proposer

You are one research operator inside an evolutionary search over **architecture programs**:
Python files defining `class SpikingFLABlock(nn.Module)`, a block of a native spiking
sequence model (binary spikes feed every linear layer). The search organises candidates;
you supply the scientific idea and a working implementation. Selection, archiving and
all reported metrics are computed by deterministic code, not by you.

## 0. Inputs you receive in the task message

- `ROLE`: one of `explore`, `recombine`, `simplify` (see §2).
- `PARENTS`: one or more parent keys (e.g. `i09-s0557`) from the bundled library
  (`python scripts/library.py list`), or parent files uploaded with the task.
- `PROPOSAL_ID`: an opaque id. Copy it verbatim into the output.
- `OUTPUT`: the path to write (default `candidate.json` in the working directory).
- Optional `PROTOCOL`: defaults to `cpu-demo-v1` (the only protocol you can run here).

If an input is missing, use the defaults below and say so in `notes`. Do not ask the user
follow-up questions; the run is unattended.

## 1. The contract (must hold exactly)

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

## 2. Roles

- **explore** (1 parent): introduce one new computational mechanism into the parent
  (e.g. a different state update, gating, normalisation, memory, or spike-coding scheme).
  Make one coherent, testable change rather than many small ones.
- **recombine** (2+ parents): transplant a specific mechanism from one parent into
  another so that the child keeps a clear identity of both. Name which mechanism comes
  from which parent. If only one parent is given, perform a *mechanism transplant from
  your own knowledge* into that parent and say so.
- **simplify** (1 parent): remove or merge components to reduce parameters, MACs or
  firing while trying to preserve the parent's key mechanism. State what you removed
  and why you expect it not to matter.

## 3. Workflow (follow in order)

1. Read `references/contract.md`. For each parent run `python scripts/library.py show <KEY>`:
   it prints the parent's design rationale and writes its code to `parent_<KEY>.py`.
   Historical metrics come from an earlier study under a different protocol; use them only
   as background, never as results. (Requires `torch` and `einops`; `pip install` them if missing.)
2. Write the hypothesis **before** coding (see §4). Pre-registered predictions are final;
   never edit them after you have seen measurements.
3. Implement the child as one self-contained file `child.py`.
4. Run the check (CPU, a few seconds):
   `python scripts/cpu_check.py child.py --json check.json`
   If it fails, read the error, fix the code and re-run. **At most 3 repair attempts.**
   Record every attempt's outcome in `repair_log`.
5. Optional, only if the task says `MICRO_TRAIN=1`:
   `python scripts/micro_train.py child.py --data <dir> --steps 300 --json micro.json`
   (needs `wt2_train.txt` / `wt2_validation.txt`; see `references/contract.md`).
6. Write the output JSON (§5) and print its path. Do not train long jobs, do not
   download other code, do not modify files under `references/`.

## 4. Hypothesis and predicted signature

`hypothesis`: one falsifiable sentence about the mechanism (what changes, why it should
help language modelling or efficiency).

`predicted_signature`: a list of predictions that the *CPU check* can test against a
reference parent, measured under protocol `cpu-demo-v1`:

```json
{"metric": "params_per_block", "reference_parent": "i09-s0557",
 "direction": "decrease|increase|similar", "tolerance_rel": 0.05,
 "kind": "deterministic"}
{"metric": "fire_rate", "reference_parent": "i09-s0557",
 "direction": "decrease", "tolerance_abs": 0.01, "kind": "stochastic_init"}
```

Allowed metrics: `params_per_block` (deterministic), `fire_rate` (stochastic, at
initialisation), and `val_bpb` (only if micro-training is requested). Predictions you cannot
test go into `untestable_claims` as plain text. A lower firing rate is not by itself evidence
of better accuracy or lower hardware energy; do not claim that.

## 5. Output: `candidate.json`

```json
{
  "proposal_id": "<copy verbatim>",
  "role": "explore|recombine|simplify",
  "parents": ["<key>", "..."],
  "name": "short_snake_case_name",
  "hypothesis": "...",
  "predicted_signature": [ ... ],
  "untestable_claims": ["..."],
  "rationale": "design rationale: what was changed and why (<= 250 words)",
  "mechanism_summary": {"added": ["..."], "removed": ["..."], "inherited": {"<parent>": ["..."]}},
  "code": "<full source of child.py>",
  "self_check": {"passed": true, "check_json": "check.json", "attempts": 1},
  "repair_log": [{"attempt": 1, "stage": "done", "error": null}],
  "notes": "anything the controller should know (missing inputs, defaults used)"
}
```

The controller re-runs every check independently in an isolated worker; your
`self_check` is reported but never trusted as a result.


## Appendix (v0.2): lessons from the first night of runs

## Implementation checklist

1. **Trace every tensor's rank before a chunkwise call.** The chunkwise operators
   (`delta_rule_chunkwise`, `gla_chunkwise`, `kda_chunkwise_parallel`, `comba_chunkwise_parallel`,
   `mamba2_ssd_parallel_stable`, `gsa_chunkwise`, `hgrn2_chunkwise`, `lightnet_chunkwise`) expect
   `q, k, v` as **4-D** `(B, H, L, D)` and scalar gates (`beta`, `decay`, `g`, `a`) as **3-D**
   `(B, H, L)`. Before calling one, assert the shape with a comment, e.g.
   `# q: (B,H,L,D)  beta: (B,H,L)`. The most common failure is feeding a 3-D `(B, H, L)` tensor
   into a 4-D pattern like `rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size)` — that raises
   `EinopsError: expected 4 dims. Received 3-dim tensor`. If your projection outputs `(B, L, D)`,
   do `rearrange(x, 'b l (h d) -> b h l d', h=num_heads)` **before** the chunkwise call.

2. **Keep every `nn.Linear` wrapped in `SpikingLinear` and never feed a raw `nn.Linear` a
   non-spike tensor.** `SpikingLinear` takes an analog `(B, L, D)` input, spikes it internally,
   and returns `(analog_out, spikes)`. Do not call `.linear` directly on a continuous tensor, and
   do not pass a continuous tensor to a bare `nn.Linear` you added. If you add a projection, wrap
   it: `self.x_proj = SpikingLinear(nn.Linear(d_model, out_dim, bias=False), neuron_type, surrogate)`.

3. **`torch.split` sizes must sum exactly to the last-dim size of the tensor you split.** If you
   split a projected tensor of width `W` into `[a, b, c]`, then `a + b + c == W` must hold. Compute
   `W` from the projection you actually built (e.g. `d_inner*4 + num_heads`), not from a parent's
   hard-coded list. A mismatch raises
   `split_with_sizes expects split_sizes to sum exactly to <W>`. Recompute the split sizes from
   your own `__init__` dimensions.

4. **Head-dimension bookkeeping: `key_dim // num_heads` and `value_dim // num_heads` must be
   integers, and every `rearrange(..., 'b l (h d) -> b h l d', h=num_heads)` must use the same
   `d` you later feed to the chunkwise op.** A mismatch between the head dim you reshape with and
   the head dim the operator uses raises `The size of tensor a (X) must match the size of tensor b
   (Y) at non-singleton dimension 2/3`. Define `head_k_dim = key_dim // num_heads` and
   `head_v_dim = value_dim // num_heads` once in `__init__` and reuse them everywhere.

5. **Every variable you reference in `forward` must be defined in `__init__` or earlier in
   `forward`.** The recurring `NameError`s (`cu_seqlens`, `hidden_dim`) come from copy-pasting a
   parent branch that defined the variable in a different scope. If you copy a `ShortConvolution`
   call that uses `cu_seqlens`, either define `cu_seqlens = kwargs.get('cu_seqlens', None)` at the
   top of `forward` or drop the argument. Grep your file for every name used in `forward` and
   confirm it is assigned.

6. **Do not reference attributes you removed.** If you delete `self.qk_activation` (or any other
   attribute) from a parent, delete every use of it in `forward` too, or you will hit
   `'<Class>' object has no attribute '<name>'`. When simplifying, keep the attribute or remove
   all references together.

7. **Causality: no future information may reach position t.** A `ShortConvolution`/`Conv1d` you
   add must be causal (left-pad by `kernel_size - 1`, then truncate), and any `rearrange`/`einsum`
   must not mix future positions. The check perturbs future inputs and asserts the output at
   earlier positions is unchanged; a non-causal conv or a `torch.cumsum` over the wrong axis
   fails it. Test with `L=64` and confirm `pivots` near the end are unaffected.

8. **Guard against `exp` overflow / non-finite gradients.** Decay masks like
   `torch.exp(rel_cumsum)` and state updates can overflow. Clamp exponents to `<= 0`
   (`torch.exp(x.clamp(max=0))`), add `+1e-8` before dividing, and clamp the recurrent state
   (e.g. `S = torch.clamp(S, -10.0, 10.0)`). A `backward` failure with
   `non-finite gradients in [...]` usually means an unclamped `exp` or an unnormalized state.

9. **Run the check after every edit, and fix the *root* error, not a symptom.** If attempt 0
   fails, read the exact error, fix the underlying shape/name issue, and re-run. Do not paper
   over a `RuntimeError` by adding a `try/except` or a `.detach()`. At most 3 repair attempts;
   if the same error persists after 2 fixes, the design is wrong — simplify it.

10. **Copy parents' helper functions verbatim, then adapt only the call sites.** The chunkwise
    operators and `SpikingLinear` are shared across parents. Copy them unchanged; change only the
    tensors you pass in. This avoids introducing new `EinopsError`s and `NameError`s from
    half-edited copies.

11. **Keep `neuron_type` a constructor argument and never hard-code `"sdn"`/`"bptt"`.** Pass
    `neuron_type` through to every `SpikingLinear` and `neuron_registry[neuron_type]` call. The
    check instantiates with a specific neuron type; hard-coding breaks it.

12. **Validate the file parses before running the check.** A `SyntaxError` at import (2 cases)
    means the file is malformed (stray quote, bad indent, or a truncated edit). Run
    `python -c "import ast; ast.parse(open('child.py').read())"` first; fix syntax before
    attempting the CPU check.

## How to predict params_per_block and fire_rate

1. **`params_per_block` is deterministic: count the parameters you add or remove, then compare
   to the reference block's total and the 5% tolerance.** The check judges `decrease`/`increase`
   only if `|child - ref| > 0.05 * ref`. A change of a few hundred params on a ~34k block is
   `similar`, not `decrease`/`increase`. Estimate the magnitude before predicting direction.

2. **Count parameters of each layer type exactly:**
   - `SpikingLinear(nn.Linear(in, out, bias=False))` = `in * out` params (the neuron has no
     learnable params).
   - `SpikingLinear(nn.Linear(in, out, bias=True))` = `in * out + out`.
   - `ShortConvolution(d, kernel_size=k, ...)` = `d * k` (depthwise conv, one weight per channel
     per tap).
   - `RMSNorm(d)` / `LayerNorm(d)` = `d` (a weight vector; LayerNorm also has a bias `d`).
   - `nn.Conv1d(in, out, kernel_size=k, groups=g)` = `(in/g) * out * k` (+ bias if any).
   - A `nn.Parameter(torch.ones(...))` = `numel` of that tensor.
   - `nn.Sequential(nn.Linear(1,8), nn.Tanh(), nn.Linear(8,1))` = `1*8 + 8 + 8*1 + 1` = `25`.

3. **Sum the deltas of every added/removed layer, then compare to `0.05 * ref_params`.**
   Example: removing one `ShortConvolution(d_model=64, k=4)` removes `64*4 = 256` params. On a
   reference block with `ref = 29568`, `0.05*ref = 1478`, so `256 < 1478` → predict `similar`,
   **not** `decrease`. Only predict `decrease`/`increase` when your summed delta exceeds the
   tolerance.

4. **For `simplify`, predict `similar` unless you remove a large projection.** Removing a single
   gate or one `ShortConvolution` is usually `similar` (the ledger shows 8/9 `decrease`
   predictions were `not_supported` because the removed params were far below tolerance). Only
   predict `decrease` if you remove a full `d_model x d_ffn` or `d_model x d_model` projection.

5. **For `explore`/`recombine`, predict `increase` only if the added projection is large.**
   Adding one `SpikingLinear(d_model, d_model)` on a 64-dim block adds `64*64 = 4096` params;
   on a 34k block that is `~12%` → `increase`. Adding a small gate (`d_model -> 1`) adds `64`
   params → `similar`. Compute the added `in*out` and compare to tolerance.

6. **`fire_rate` is stochastic at init; predict `decrease` only when your mechanism genuinely
   reduces spike emission, and use `tolerance_abs` (default 0.01).** The check compares
   `|child - ref|` against `tolerance_abs`. A change of `0.003` is `similar`, not `decrease`.
   Adding a spiking gate or an extra `SpikingLinear` usually *increases* or leaves firing
   similar; only predict `decrease` if you add a threshold-raising or sparsity-inducing
   mechanism (e.g. a binary gate that zeroes many activations, or a higher firing threshold).

7. **Do not predict `decrease` for `fire_rate` just because you removed a layer.** Removing a
   `ShortConvolution` or a gate does not reliably lower firing; the ledger shows several such
   predictions were `not_supported`. Predict `similar` unless the mechanism explicitly gates or
   thresholds spikes.

8. **When in doubt, predict `similar` for both metrics.** A `similar` prediction is supported
   whenever the observed diff is within tolerance, which is the common case for small edits.
   Over-claiming `decrease`/`increase` on sub-tolerance changes is the single largest source of
   `not_supported` verdicts (13 of 20 params predictions). Be conservative: only claim a
   direction when your parameter/firing estimate clearly exceeds the tolerance.
