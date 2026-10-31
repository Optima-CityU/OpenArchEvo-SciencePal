---
name: openarchevo-snn-proposer
description: Propose, implement, audit and CPU-check a new native spiking sequence-model block (SpikingFLABlock) from OpenArchEvo parent architectures. Use when asked to explore, recombine or simplify spiking/linear-attention architecture mechanisms, or to produce an OpenArchEvo candidate.json. Also use it to demo the OpenArchEvo discovery loop ("agents propose, evolution decides"): replay a real generation of the v2 loop, recompute the v1-vs-v2 evidence from raw data, replay the audits, or summarise the case studies.
version: 0.3.4
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
   Use `python scripts/library.py list` to browse; **never open `scripts/library_code.zip` or print the
   whole index** — the library is ~1 MB of code and would overflow your context.
   Historical metrics come from an earlier study under a different protocol; use them only
   as background, never as results. (Requires `torch` and `einops`. If missing, install them **separately**, without cache (the sandbox disk is 10 GB) and from the official indexes (the default mirror is ~100 KB/s): `PIP_INDEX_URL=https://pypi.org/simple pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu` then `PIP_INDEX_URL=https://pypi.org/simple pip install --no-cache-dir einops`. Takes 5–25 minutes in a fresh sandbox. Start it once, in the background, only when needed.)
2. Write the hypothesis **before** coding (see §4). Pre-registered predictions are final;
   never edit them after you have seen measurements.
3. Implement the child as one self-contained file `child.py`.
4. Run the check (CPU, a few seconds):
   `python scripts/cpu_check.py child.py --json check.json`
   If it fails, read the error, fix the code and re-run. **At most 3 repair attempts.**
   Record every attempt's outcome in `repair_log`.
4b. Run the functional-equivalence test against every parent:
   `python scripts/func_equiv.py parent_<KEY>.py child.py --json equiv_<KEY>.json`
   `"equivalent": true` means that, with the parent's weights copied in, your child computes exactly the
   same output: the edit is dead code or an equivalent rewrite and would train to the identical result.
   Revise the change so it actually alters the computation (this counts as a repair attempt). Record the
   result in `self_check.func_equiv`. Exit code 0 means *equivalent* (bad for a proposal), 1 means *different*
   (what a real change should give), 2 means the test itself failed. The test is only meaningful against the parent
   whose code you modified (the *base* parent); against a cross-family donor most weights cannot be shared and the
   result is reported as `note: not_comparable` — say so instead of reading it as a real difference.
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
  "self_check": {"passed": true, "check_json": "check.json", "attempts": 1,
                 "func_equiv": {"<parent>": false}},
  "repair_log": [{"attempt": 1, "stage": "done", "error": null}],
  "notes": "anything the controller should know (missing inputs, defaults used)"
}
```

The controller re-runs every check independently in an isolated worker; your
`self_check` is reported but never trusted as a result.


## 6. Demo mode (a person, e.g. a judge, is trying the system)

Use this section when the user asks to *see*, *try* or *demo* the system ("run one generation", "show me how it
works", "show the audit", "show the case studies") instead of giving ROLE/PARENTS/PROPOSAL_ID. Keep the whole demo
under ~10 minutes of CPU work. Never start GPU training in demo mode.

Pick the mode from the user's words; if unclear, offer D, E and B in one short sentence and start with **E** (fastest).
Never install anything for E or D. Start the torch/einops install (§3 step 1) in the background only for A, B, the live
variant of D, or `replay --live`; tell the user it can take 5–25 minutes in a fresh sandbox.

D. **Replay one real generation of the v2 loop** ("show me how it works", "run a generation", "agents propose, evolution
   decides"). Needs no torch (gate verdicts are precomputed; `replay --live` re-runs them after the install).
   Data: generation 6 of a real pre-registered run (`references/generation6.json`). Run in order and show each
   table: `python scripts/generation.py brief` (what the research-lead agent read and wrote; parents and roles come from the
   evolutionary rules), `python scripts/generation.py replay` (the contract and functional-equivalence gate verdicts on the 9 real
   proposals, next to what the real run decided; ~1 s, or ~1 min with `--live`), `python scripts/generation.py select` (NSGA-II on the survivors,
   what was actually trained, and what the best child did later). Then explain in ≤5 sentences who did what: agents
   proposed (directions + code), deterministic gates and evolutionary selection decided. If the user wants to try,
   offer the live variant: pick a slot, `python scripts/generation.py parent <base> --out parent.py`, write your own child
   following the slot's direction and §1, then `python scripts/generation.py check child.py --slot N`. Every command appends its table to
   `generation_report.md` in the working directory; point to it.
E. **Recompute the evidence** ("is it real?", "show the results", "what did you find?"). `python scripts/evidence.py`
   (pure Python, ~1 s; add `--budget 17` for the pre-registered budget of the replication runs, `--run V2_s1` to list one
   run). It rebuilds the per-run table, the wasted share of trainings (v1 ~24% vs v2 ~7% at 15 trainings per run) and exact
   permutation p-values from the raw outcomes. State the honest notes it prints: the best PPL is about the same in both
   arms; "good" children are only borderline more frequent; small 2-epoch proxy. It writes `evidence.md` (and `evidence.png` if matplotlib is present) to the working directory; show them.
A. **One live audited proposal** (the user wants the agent to invent a new child now):
   1. `python scripts/library.py demo`, then pick the parent the user names or the first demo parent; choose the role
      the user names or `explore`. Make up a PROPOSAL_ID (`demo-<6 hex>`).
   2. Run the normal workflow §3 steps 1–4b (hypothesis and predictions written BEFORE code; CPU check including the
      spiking-contract audit; functional-equivalence test against the parent).
   3. Compare each pre-registered prediction with the measured CPU value and say plainly whether it held. The
      `reference_parent` of every prediction is the base parent (the one whose code you edited); for a recombination
      also state which parent donated the transplanted mechanism.
   4. Write `candidate_card.md`: lineage (parent → role → child), hypothesis, predictions vs measurements, the code
      diff essentials (≤40 lines, `diff -u` of parent and child, comments stripped), every check with pass/fail and the
      exact numbers, the functional-equivalence verdict, and "what happens next in the full system" — GPU behavioural
      fingerprint (protocol `sp-fp-sltt-v1`), NSGA-II selection on predicted fitness × novelty, training under
      `sp-wt2-sltt-s128-e2` (WikiText-2, 6×256, **SLTT** neurons, seq 128, 2 epochs; the paper's SDN protocol is not
      used here), paired-seed re-tests and a reviewer agent — clearly marked as *not run in this demo*.
   5. Also write `candidate.json` (§5). Show the card to the user.
B. **Audit replay** ("show the audit", "how do you catch cheating"). Run these exact commands in order (no chaining with
   `&&` after commands that may exit non-zero) and show each verdict next to the expected one:

   | step | command | expected |
   |---|---|---|
   | 1 | `python scripts/library.py case r02a-4fdb1d --out cheat.py` then `python scripts/cpu_check.py cheat.py` | FAIL at stage `spiking_contract`: `mlp.fc1`, `mlp.fc2` get non-binary input (case study §2) |
   | 2 | `python scripts/library.py case r01a-66286b --out dup_a.py`; `python scripts/library.py case r01a-b7e4ea --out dup_b.py`; `python scripts/func_equiv.py dup_a.py dup_b.py` | `"equivalent": true` — 98 changed lines, same function (§3) |
   | 3 | `python scripts/library.py case r01b-c8c630 --out c8.py`; `python scripts/library.py show i00-hgrn2 --out hgrn2.py`; `python scripts/func_equiv.py hgrn2.py c8.py` | `"equivalent": false`, 2 new live parameters — a real change (§1) |

   Write `audit_replay.md` with the three outputs and one sentence each on why it matters.
C. **Case studies** ("what did you find?"): summarise `references/case_studies.md` faithfully, with protocol names,
   seed noise and the stated limits. Never claim a state-of-the-art architecture or energy savings.


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

## Version history

- 0.1.0 (10/21 00:50) first platform Skill: contract, roles, CPU check, parent library.
- 0.2.0 (10/21 morning) implementation checklist written by a SciencePal agent from the night's failure logs
  (first-try pass 73% → 83%); spiking-contract audit after an analog-MLP candidate won on PPL.
- 0.3.0 (10/21 22:00) functional-equivalence test after 79–98-line edits trained to bit-identical PPL; library split
  into index + zip after two runs overflowed their context; demo mode and case studies.
- 0.3.1 (10/21 22:20) fixes from a simulated judge trial on SciencePal (7 min, 2.7M tokens): separate torch/einops
  install, exit-code meaning of `func_equiv.py`, an exact audit-replay table, correct protocol names in the card.
- 0.3.2 (10/21 22:35) from a second (sceptical) judge trial, 12 min: reference parent = base parent; equivalence
  against a cross-family donor reported as not comparable; `library.py demo` prints the next steps.
- 0.3.3 (10/22 14:50) case studies updated with multi-fidelity results (c8c630 ties its parent after 30 epochs; the
  2-epoch protocol mis-ranks architectures) and the Skill-ladder numbers. SKILL text otherwise unchanged.
- 0.3.4 (10/25) demo only: modes D (replay a real v2 generation: agent directions → gates → NSGA-II selection) and E
  (recompute the v1-vs-v2 evidence from bundled raw outcomes, exact permutation tests); faster, cache-free torch install.
  Proposer instructions (§0–§5) unchanged; experiments keep using 0.3.3.
- Rejected: candidate 0.4.0 (10/22), written by a SciencePal agent from the ladder evidence (parameter-estimation procedure +
  3 checklist items; its own back-test claimed 71/90 correct). On the fixed benchmark (4 × 24 proposals) it did worse than
  0.3.2: first-try pass 65% vs 77%, repairs 46 vs 31, parameter-direction predictions 37% vs 38%. Not promoted.
- Rejected: a platform-proposed upgrade (10/21 09:06) mixing GPU-lab supervision lessons into this Skill.
