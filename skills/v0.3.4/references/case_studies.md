# Case studies from the hackathon runs (2026-10-21)

Real records from the OpenArchEvo × SciencePal runs. Protocol names matter: numbers under different protocols
must never be compared. `sp-wt2-sltt-s128-e2` = WikiText-2, GPT-2 BPE, 6×256 SNN, SLTT neurons, seq 128,
2 epochs; val PPL (lower is better). Seed-to-seed noise of one architecture is 0.5–1.0 PPL.
When you present these to a user, quote the numbers exactly and keep the caveats.

## 1. Following one descendant: `r01b-c8c630` (HGRN2 + data-dependent spiking output gate)

- Found in generation 1 by the population arm (3 islands, NSGA-II on predicted fitness × novelty).
- Role `explore`, parent `i00-hgrn2`. Pre-registered hypothesis: "a data-dependent output gate that modulates
  the recurrent state output with a spiking sigmoid gate will let the model selectively emphasise or suppress
  information from the recurrent state at each timestep, while keeping sub-quadratic complexity."
- Pre-registered predictions (cpu-demo-v1): params increase (observed 29,568 → 33,920: supported);
  fire rate similar ±0.01 (0.107 → 0.112: supported).
- Code diff (essence): one new `SpikingLinear` projection `o_gate_proj` (d_model → hidden, spike input), a causal
  `ShortConvolution` on it, `o = o * sigmoid(o_gate)` before the output projection.
- Three views vs parent: code similarity high; behavioural fingerprint cosine 0.92 (distinct behaviour);
  functional-equivalence test: NOT equivalent (max |Δy| 0.39 with shared weights; 2 new live parameters).
- A SciencePal reviewer agent initially called it a "near copy with small expected gain" from its
  initialisation fingerprint. Training disagreed:

| | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| r01b-c8c630 | 120.91 | 121.50 | 121.27 | 121.23 |
| HGRN2 (parent) | 122.39 | 123.37 | 122.46 | 122.74 |

  Multi-fidelity re-tests (val PPL, child vs parent): 2 epochs −1.5 (3 seeds); 6 epochs −0.85 (3 seeds; test −0.75);
  **30 epochs: 81.41 vs 81.43 — a tie** (test 72.42 vs 72.86). The gate makes HGRN2 learn faster early; the advantage
  disappears with long training. HGRN2 itself is the weakest of five seeds after 30 epochs.
  Lessons: initial behaviour similarity ≠ trained outcome; short protocols reward fast early learning; only claim
  "better" under the long protocol.

## 2. Specification gaming caught: `r02a-4fdb1d`

- Best-only arm, generation 2, parent DeltaNet. Hypothesis: "Removing the SpikingMLP and using a standard MLP …
  will reduce parameters and firing rate while maintaining similar performance."
- Result: **119.33**, the best score of the night (DeltaNet 120.64).
- But `mlp.fc1`/`mlp.fc2` were plain `nn.Linear` layers fed analog activations: contract rule 3 violated. The gain
  came from adding non-spiking computation. The original CPU check passed it (it only checked interface, shapes,
  gradients and causality).
- Two independent detectors then flagged it: (a) a deterministic audit (forward pre-hooks on every `nn.Linear`
  asserting binary inputs), and (b) a SciencePal reviewer agent ("HARD VIOLATION rule 3 — drop"), which did not
  see the audit result. Marked invalid; never used as a parent.
- Audit over everything: 3 of 71 new candidates violate rule 3; 16 of 157 archive programs are flagged for human
  review (CPU/BPTT conditions; not a verdict).
- Fix fed back: the audit became stage `spiking_contract` of `cpu_check.py` (Skill v0.2).

## 3. Functional duplicates: edits that change nothing

| pair | changed lines | val PPL | functional-equivalence test |
|---|---|---|---|
| r01a-66286b vs r01a-b7e4ea | 98 | 121.885 / 121.885 | equivalent (Δy = 0) |
| r04b-0c5e28 vs r01b-4d71a8 | 79 | 120.643 / 120.643 | equivalent (Δy = 0) |
| r01b-4d71a8 vs DeltaNet | "shared state across heads" | 120.643 / 120.641 | not equivalent (Δy 0.83) |

- Same seed + same function ⇒ bit-identical training. Exact dedup (fingerprint ε=1e-6) missed both pairs;
  their fingerprint cosine was 0.974. The weight-sharing equivalence test (`scripts/func_equiv.py`) catches them
  in ~2 s on CPU. Across all 43 child–parent and equal-PPL pairs of the night, exactly these 2 were equivalent.

## 4. Skills that evolve — under review

- Fixed benchmark ("Skill ladder"): the same 24-proposal plan (6 seeds × 3 roles), 4 repetitions per version, DeepSeek
  proposer reading only the SKILL text, controller CPU checks. Pooled over 96 proposals per version:

| version | first-try pass | repairs | contract violations | functionally identical to parent |
|---|---|---|---|---|
| v0.1 | 67% | 52 | 4 | 11% |
| v0.2 | 74% | 35 | 1 | 13% |
| v0.3.2 | 77% | 31 | 0 | 4% |

  Prediction calibration did not improve (parameter-count direction right ~40–50%). Run-to-run variation is about ±8
  points, so the first-try trend is suggestive; repairs, violations and duplicates are the clearer gains.
- Where the versions came from: v0.2 = a 12-item checklist that a SciencePal agent wrote from the first night's failure
  logs + the spiking-contract audit (after case 2); v0.3 = functional-equivalence test + library split (two runs had
  overflowed their context); v0.3.1/v0.3.2 = fixes from simulated judge trials; v0.4 = a parameter-estimation procedure and
  three checklist items written by a SciencePal agent from the ladder evidence (being evaluated on the same benchmark).
- SciencePal's post-task learning auto-proposed an upgrade from a GPU-supervision run. It was reviewed and rejected
  (based on an outdated version; wrong responsibility; referenced a non-existent script), and archived.

## 5. Does the cheap search protocol rank like long training? (protocol-fidelity check)

| architecture | 2-epoch rank | 30-epoch val / test PPL | 30-epoch rank |
|---|---|---|---|
| DeltaNet | 1 | 71.72 / 63.67 | 1 |
| HomeoResSSM | 5 | 73.23 / 65.70 | 2 |
| NeuroGate | 4 | 75.24 / 67.27 | 3 |
| Mamba2 | 3 | 79.78 / 71.24 | 4 |
| HGRN2 | 2 | 81.43 / 72.86 | 5 |

Kendall τ between 2-epoch and 30-epoch ranks: −0.2 (6 epochs: 0.33 on four architectures). The 30-epoch protocol reproduces
the two-tier structure of the original OpenArchEvo archive ({DeltaNet, HomeoResSSM, NeuroGate} ahead of {Mamba2, HGRN2});
the 2-epoch protocol inverts it. The system therefore moved to multi-fidelity evaluation: 2 epochs to screen, 6 epochs to
promote, 30 epochs before any claim.

## 6. Calibration facts worth knowing

- LLM pre-registered predictions are poorly calibrated: parameter-count direction right 7/20, firing-rate direction
  14/20. Deterministic checks are not optional.
- Behavioural fingerprint (18 stable features, protocol-calibrated): same architecture with a different probe seed
  cosine ≥ 0.999; different architectures median 0.22.

## 7. v1 vs v2 under the same GPU budget (E7 + pre-registered replication E9, 10-23/24)

- Two arms, 5 independent runs each, identical expert seeds, selection rule and per-generation training budget
  (`sp-wt2-sltt-s128-e2`, 2 trainings per generation after the first). v1 = rule islands + Skill v0.1 (LLM as mutation
  operator); v2 = research-lead agent + Skill v0.3.3 + functional-equivalence gate.
- At 15 trainings per run: wasted (failed / diverged / PPL > 130) trainings 3.6 vs 1.0 per run, i.e. ~24% vs ~7% of the
  GPU budget (exact one-sided permutation p = 0.004; this test was added after the first night). Good children
  (< 122.64) 4.2 vs 5.8 — borderline (exact p = 0.024 at the pre-registered budget, 0.056 at a uniform 15). Best 2-epoch
  PPL about equal (118.40 vs 119.26): the pre-registered "v2 finds a lower best" failed.
- v2 children beat their own parent more often (31% vs 22%) and change parameter counts less — partly more cautious edits.
- Cost: the research lead used ~10M platform tokens per run; at this 2-epoch scale that is more than the GPU it saved.
- Run `python scripts/evidence.py` to recompute these numbers from the raw outcomes.
