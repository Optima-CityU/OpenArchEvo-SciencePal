---
name: openarchevo-snn-proposer
description: Propose, implement, audit and CPU-check a new native spiking sequence-model block (SpikingFLABlock) from OpenArchEvo parent architectures. Use when asked to explore, recombine or simplify spiking/linear-attention architecture mechanisms, or to produce an OpenArchEvo candidate.json. Also use it to demo the OpenArchEvo discovery loop, replay its spiking-contract and functional-duplicate audits, or summarise its case studies.
version: 0.5.0-gpt61
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

## 2. Roles: require a live mechanism, not a textual edit

Every child must add, remove or change a **live computation path reaching returned `y`**.
Before coding, trace `x -> changed operation -> downstream operations -> y` through the
actual `SpikingFLABlock` constructor and forward call. A changed helper, parameter count,
line count, name, annotation, spike log or `reg_loss` alone is not sufficient.

- **explore** (1 parent): add one coherent mechanism on that live path: a state update,
  gate, normalisation, memory or spike-coding change. State where its result is consumed.
- **recombine** (2+ parents): name the base and the donor's specific mechanism; transplant
  its computation, not its imports or class scaffolding. Keep both identities. With only
  one parent, identify a mechanism transplant from your own knowledge and say so.
- **simplify** (1 parent): remove or merge a component that the base actually executes
  and that affects `y`, while preserving its key mechanism. Deleting unused neurons,
  helpers, assertions or allocations, moving assignments, and equivalent
  re-parameterisations do not qualify, even if parameters decrease.

Follow a supplied per-slot research direction; do not substitute a generic cleanup.
Without one, choose a concrete role-specific mechanism and its insertion/removal point
before editing. Keep the known-working base structure; novelty is not a reason to rewrite
unrelated helpers. Do not copy another slot's mechanism under a new name.

## 3. Workflow (follow in order)

1. Read `references/contract.md`. For each parent run `python scripts/library.py show <KEY>`:
   it prints the rationale and writes `parent_<KEY>.py`. Use `python scripts/library.py list`
   to browse; **never open `scripts/library_code.zip` or print the whole index**. Historical
   metrics are background from another protocol, not results here. If needed, install
   dependencies separately: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
   then `pip install einops`. Identify the base file and preserve it for comparison.
2. Write the hypothesis and predictions **before** coding (§4); never revise predictions
   after measurements. Include in the existing `rationale` a short live-path plan: base,
   operation before/after, enabling constructor values, tensor shapes and route to `y`.
   Inspect the values actually passed by the wrapper: adding `num_groups=1` or changing an
   unused default does not activate grouped computation. Reject a plan that is identity,
   disabled, overwritten downstream, or confined to an uncalled branch.
3. Copy the base to self-contained `child.py`; make only the planned mechanism edit.
   Preserve imports, neuron wiring, helper definitions and shape conventions unless the
   mechanism requires changing them. Trace the edited value through subsequent spiking:
   a change erased by binarisation is not a live output change. Review the diff against
   the plan; renames, comments, annotation cleanup and assignment reordering are not novelty.
   Use the import/forward checklist below before running the check.
4. Run `python scripts/cpu_check.py child.py --json check.json`.
   Read the reported stage and exact error; repair its cause, not the checker. **At most
   3 repair attempts total**, shared with step 4b; record each outcome in `repair_log`.
   Never hide a failure with an exception fallback, detach, non-spiking bypass or a return
   of the parent's output. Re-run the full check after every code revision, including a
   novelty repair. Preserve the working interface instead of making unrelated edits.
4b. Run against every parent:
   `python scripts/func_equiv.py parent_<KEY>.py child.py --json equiv_<KEY>.json`.
   Exit 0 / `"equivalent": true` rejects the mechanism: the tested child reproduces its
   parent despite source edits. Exit 1 / false supports a difference on the tested inputs,
   **not** novelty across the population, better accuracy or a guarantee on all inputs.
   Exit 2 is a test failure, not evidence of difference; inspect the JSON as well as status.
   The base comparison is essential. A cross-family donor marked `note: not_comparable`
   is not a novelty pass; report it in `notes`, not as a verified false equivalence.
   For a false base verdict, check that the planned operation actually causes the output
   difference, rather than changed initialisation order or unmatched weights. If unclear,
   compare mechanism-enabled and disabled/ablated versions with identical shared weights,
   inputs, eval settings and reset neuron state, using multiple inputs. If disabling the
   mechanism leaves `y` unchanged, repair the live path; do not add noise or rename weights
   to fool the test. Inspect any supplied prior candidates for the same mechanism; a
   base-only difference cannot clear a known near-duplicate. Record actual verdicts in
   `self_check.func_equiv`, and unresolved comparability in `notes`.
5. Only if `MICRO_TRAIN=1`:
   `python scripts/micro_train.py child.py --data <dir> --steps 300 --json micro.json`
   (needs `wt2_train.txt` / `wt2_validation.txt`; see `references/contract.md`).
6. Write the output JSON (§5) and print its path. Its code must match the final checked
   `child.py`; report unsuccessful or exhausted repairs honestly, never manufacture a pass.
   `self_check.passed` reports the actual CPU result; an equivalent child is not an accepted
   proposal even if CPU-valid. Do not train long jobs, download other code or modify
   `references/`. Use existing fields for the live-path audit; add no output fields.

## 4. Hypothesis and predicted signature

`hypothesis`: one falsifiable sentence about the mechanism (what changes, why it should
help language modelling or efficiency). Name the live operation and its expected effect,
not merely a new class or parameter. In `rationale`, state how disabling/removing that
operation would change `y`; keep this within its existing 250-word limit. A prediction
about parameter count or firing alone does not establish a functionally different child.

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

A. **One audited generation** (default demo):
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


## Appendix: focused live-path and CPU checklist

The revision evidence reports `forward` (119) and `import` (69) as the largest CPU failure
stages. It also shows parent-equivalent edits ranging from 3 to 48 changed non-comment
lines. Therefore prioritise a small live mechanism with targeted checks, not longer edits.

### Before implementation: reject these non-mechanisms

- Persisting a local flag as `self.use_gate` without changing the executed branch
  (r02a-c5fe99); moving an assignment or deleting an unused neuron (r02a-a27ae0).
- Adding grouped-query scaffolding while `num_groups=1` leaves the path unchanged
  (r04a-b7742d). Trace the wrapper's actual arguments, not just helper defaults.
- Import/docstring/type cleanup, unused helper deletion or changing a default neuron
  value not used by the wrapper (r05a-7da371). Do not remove required imports while cleaning.
- A zero-strength addition, multiply-by-one, disabled branch, overwritten result, or
  algebraically equivalent rewrite. Parameter changes alone do not establish novelty.

Evidence reports fewer near-duplicates/parent-identical proposals with per-slot directions
(41% versus 60% without); this is not a guarantee for a new instruction set. The week's
spiking output gate on DeltaNet was confirmed with three seeds and a parameter-matched
control. Use it as an example of a live output path, not a direction to repeat everywhere.

### Import preflight

1. Check the complete saved file for syntax, truncation and indentation before CPU execution
   (for example, parse it with `ast` in a separate local validation file). This is a source
   check, not an architecture import; §1 still governs imports inside `child.py`.
2. Preserve the base's allowed imports and helper definitions; ensure every symbol used
   by annotations, constructors and forward is defined/imported. Copy only the donor's
   needed mechanism and dependencies, not unavailable packages or a partial class.
3. Do not add dependencies, change neuron defaults or broadly rewrite helper code to
   repair an import failure. Read the actual error; syntax success alone is not import
   success. Let `cpu_check.py` test the real import environment.

### Forward preflight

1. Trace changed tensors from projection to returned output: shape, head axis, split widths,
   gate broadcast and residual width. Derive dimensions from constructor arguments; never
   fix a test by hard-coding batch or sequence length. Split sizes must sum to the projected
   width; head dimensions and every reshape must agree with the operator's interface.
2. Inspect the actual copied helper's input/output ranks before its call. Where it expects
   `(B,H,L,D)` q/k/v and `(B,H,L)` gates, do not supply a 3-D tensor to a 4-D rearrange.
   For `(B,L,H*D)`, use the appropriate head reshape before the call. Preserve the helper's
   tested layout instead of guessing from another family; carry output shapes through
   the output projection and residual. Check singleton and non-chunk-multiple lengths
   when touching chunking, without weakening the any-length contract.
3. Resolve every local name and `self` attribute on the executed branch. If a component
   is removed, update all uses together. Do not paste references such as `cu_seqlens` or
   `hidden_dim` from a scope that does not define them; do not invent kwargs in a forward
   signature that has none.
4. Keep every `nn.Linear` input binary via the parent's neuron/`SpikingLinear` wiring.
   Unpack `(analog_out, spikes)` correctly; never call `.linear` on continuous tensors.
   Pass `neuron_type` through all added neurons and collect their spikes in the existing
   return structure. A continuous gate cannot bypass the next projection's spiking input.
5. Preserve causal padding, recurrent update order and numeric safeguards. Check that
   changed reductions/broadcasts do not mix future positions and that introduced
   divisions/exponentials remain finite. Avoid gratuitous clamps that erase the proposed
   mechanism; use the full CPU check rather than assuming a forward pass proves causality.
6. Fix the reported root cause and rerun the complete CPU and base-equivalence checks
   on the final bytes. Do not switch to an identity operation just to recover CPU validity.
   If two fixes repeat the same failure, simplify the implementation of the same live
   mechanism within the remaining repair budget, not the contract or the novelty gate.

### Predict only what the protocol measures

Count added/removed parameters from actual layer dimensions and compare the net delta to
`params_per_block` and the declared tolerance before predicting increase/decrease. A small
change within tolerance is `similar`; that says nothing about functional novelty. For a
linear layer count `in*out` plus bias if present, and include other changed parameters.
Do not change architecture merely to make a metric prediction easier to satisfy.

Predict `fire_rate` from the changed spike-emission path, not from parameter count or a
removed layer; retain its stochastic-at-initialisation qualification. Place accuracy,
energy or other unmeasured expectations in `untestable_claims`. A passed check and a
non-equivalence verdict are not a measured language-modelling improvement.

### Final audit and limits

Confirm the slot's named mechanism reaches `y`, CPU checks passed on the final source,
base-equivalence is resolved, and predictions remain pre-registered. Document actual
failures and unknowns using §5, including known duplicate mechanisms. Keep all script
names, checker settings and thresholds unchanged. Functional difference on the base
comparison is necessary, not sufficient for downstream fingerprint novelty.

The previous v0.4 instruction candidate was rejected after worse fixed-benchmark repairs
and first-try pass despite its own back-test claims. This revision likewise makes no
measured pass-rate claim: evaluate CPU pass/repair rates and duplicate rejections on the
same benchmark before promotion; do not trade CPU reliability for apparent novelty.

## Version history

- 0.5.0-gpt61: candidate revision from the supplied weekly evidence; live-path planning,
  base-comparison interpretation, targeted import/forward preflight and honest final audit.
  Not benchmark-validated or promoted by writing this file.
