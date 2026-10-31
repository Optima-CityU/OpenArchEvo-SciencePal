# OpenArchEvo SNN Proposer — v0.3.2 → v0.4.0

Every change below is grounded in `evidence.json` (the fixed 24-proposal
benchmark, 4 runs per Skill version, DeepSeek proposer reading the SKILL as its
system prompt, controller running the CPU checks). The contract, roles,
workflow, output format, demo mode and prior version history are unchanged.

## Change 1 — params_per_block prediction rewritten as a concrete code-based audit

**Evidence.** `v032_param_predictions` (90 rows) shows the v0.3.2 proposer
predicted a direction on 61/90 rows but was correct on only 34/90 (~38%,
matching the "~40%" calibration noted in the task). The ground truth is heavily
skewed: 68/90 rows are within the 5% tolerance (`similar`), 11 are `increase`,
11 are `decrease` (integer-count classification). The dominant error is
over-claiming `decrease`/`increase` on sub-tolerance changes.

**Change.** Replaced the old items 4–5 of "How to predict params_per_block and
fire_rate" with a concrete, checkable estimation procedure at the cpu-demo-v1
sizes (`d_model=64, num_heads=4, d_ffn_mult=2` → `d_ffn=128`):

- A full `d_model x d_model` `SpikingLinear` adds `64*64 = 4096` params; a full
  `d_model x d_ffn` one adds `64*128 = 8192`. On a ~34k block `0.05*ref ≈ 1700`,
  so either is `increase`/`decrease`.
- A small gate (`d_model -> num_heads` = `256`, `d_model -> 1` = `64`) and a
  `ShortConvolution(d,k)` (`256`) are all **below** tolerance → `similar`.
- Default to `similar`; only claim a direction when the counted `|net_delta|`
  clearly exceeds `0.05 * ref_params`.

**Back-test accuracy.** I reconstructed each row's net_delta from its
`mechanism_summary` text using the same layer-type counts and compared to
`0.05 * ref_params`. The rule predicts **71/90** rows correctly, versus the
v0.3.2 proposer's **34/90** and the always-`similar` baseline's **68/90**. It
also correctly identifies 11 of the 22 non-`similar` rows (the baseline
identifies 0). This is a conservative lower bound: the summary text is
free-form and ambiguous (e.g. it does not always state exact projection
dimensions), whereas the rule as written in the SKILL is a *code-based* audit
the proposer performs on the actual `child.py`/parent code it writes and reads,
which resolves those ambiguities. Reproducible script: `backtest.py`.

## Change 2 — three checklist items for the dominant CPU-check failures

**Evidence.** `v032_fail_counts` and `v032_fail_examples` (25 examples):
`forward:RuntimeError` (16) and `import:SyntaxError` (10) dominate; the rest are
`forward:NameError` (2), `forward:TypeError` (2), `forward:AttributeError` (1),
`causality:ValueError` (1), `build:KeyError` (1), `forward:ValueError` (1),
`build:NameError` (1).

**Change.** Added exactly three checklist items (13–15), each tied to concrete
failure examples:

- **Item 13 (RuntimeError shape errors, 16 cases).** At cpu-demo-v1
  `d_model=64, num_heads=4, head_dim=16, d_ffn=128`, a per-head tensor is
  `(B,H,L,16)`/`(B,H,L)` and a d_model tensor is `(B,L,64)`. Mixing them raises
  `The size of tensor a (4) must match the size of tensor b (64) at
  non-singleton dimension 1` (e.g. `d-exp-5804ef`, `d-exp-779bd1`). Before any
  `rearrange`/`einsum`/broadcast, confirm both operands use head dim 16 and
  never mix a per-head tensor with a d_model tensor.
- **Item 14 (SyntaxError at import, 10 cases).** Run
  `python -c "import ast; ast.parse(open('child.py').read())"` after writing
  `child.py` and after every repair. `SyntaxError: invalid syntax (<unknown>,
  line 1)` (e.g. `d-rec-27a7b0`, `d-sim-86d251`, `d-exp-65073d`) means the file
  is malformed; fix it before the CPU check.
- **Item 15 (copy-paste kwargs/names, TypeError/NameError/AttributeError).**
  `delta_rule_chunkwise() got an unexpected keyword argument 'initial_state'`
  (e.g. `d-rec-f85c7d`), `name 'mlp_spikes' is not defined` (e.g.
  `d-sim-86d251`), `'DeltaNet' object has no attribute 'use_beta'` (e.g.
  `d-rec-16fe85`). Check the operator's signature before passing extra kwargs.

## Change 3 — version history line for 0.4.0

Appended a 0.4.0 line to the version history citing the evidence numbers
(68/90 within tolerance, back-test 71/90 vs 34/90, 16 RuntimeError + 10
SyntaxError failures) and stating that everything else is unchanged.

## Size validation

`SKILL.md` is **22,890 bytes** (≤ 23,000) and still contains the required
sections `## 1. The contract`, `## 5. Output`, and `## 6. Demo mode`.
