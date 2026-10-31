# Proposer Skills and their evolution

The proposer Skill (`SKILL.md`) is the instruction set an LLM reads before it writes a new spiking sequence-model block.
In OpenArchEvo v1.5 the Skill is itself an evolving object: SciencePal agents read evidence from the search and
propose revisions, and a fixed CPU benchmark decides which revision survives. Machine-readable details:
[`lineage.json`](lineage.json).

```
v0.1 ─▶ v0.2 ─▶ v0.3.x ─▶ v0.4 ✗ ─┐
                   │               (rejected by the ladder benchmark)
                   └──────────────▶ v0.5 ✓ ─▶ v0.6 ×6 ✗ ─┐
                                      │       (no promotion)
                                      └─────▶ v0.5.1 ✓ ─▶ v0.7 ✓ (current)
```

| Version | Proposed by | Evidence it was given | Benchmark (first-try pass · diversity) | Verdict |
|---|---|---|---|---|
| [v0.1](v0.1/SKILL.md) | project authors | — | ladder: 67% · honesty 0.85 | initial |
| [v0.2](v0.2/SKILL.md) | SciencePal agent | first night's failure logs | ladder: 74% · honesty 0.86 | promoted |
| v0.3.2 → [v0.3.3](v0.3.3/SKILL.md) | authors + two simulated-judge trials on SciencePal | duplicate trainings, judge feedback | ladder: 77% · honesty 0.96 | promoted |
| [v0.3.4](v0.3.4/SKILL.md) | authors | — | demo modes only | demo release |
| [v0.4.0](v0.4.0/SKILL.md) | SciencePal agent (GPT-6) | ladder evidence ([CHANGES](v0.4.0/CHANGES.md)) | ladder: 65% · repairs 46 vs 31 | **rejected** |
| [v0.5.0](v0.5.0/SKILL.md) | DeepSeek-V4-Flash (E15, 5-model panel) | proposal fate, failure stages, duplicate examples | E15: 64% · effective-new 43 vs 39, parent-equivalent 19% → 2% | **promoted** |
| [v0.6.0 ×6](candidates/v0.6.0_round3/) | Flash, GLM-5.3, Kimi K3, GPT-6.1, DeepSeek-Pro | v0.5 inside the GPU loop | E19: best +3 distinct_new but −15 pp pass | all rejected |
| [v0.5.1](v0.5.1/SKILL.md) | experience cards C1–C6 from GPU experiments | E12/E18/E20/E21/E24 | E27: 70% vs 75% · distinct_new 40 vs 37 | **promoted** (no regression) |
| [v0.7.0](v0.7.0/SKILL.md) | GPT-6.1 (E43, 5-model panel) | v0.5.1 + new GPU evidence (gates, OAF) | E43: 79% vs 79% · distinct_new 42 vs 32 | **promoted (current)** |

Benchmarks:
- **Ladder (E4):** 24 CPU-checked proposals × 4 repetitions; first-try feasibility and "honesty" (no functional
  duplicate, no contract violation), 90% Wilson intervals.
- **Fixed CPU benchmark (E15/E19/E27/E43):** 6 expert seeds × 3 roles × 8 proposals × 2 repetitions, DeepSeek-V4-Flash
  writes the code. `distinct_new` counts functional-equivalence classes that differ from the parent. The baseline is
  re-run in every round; promotion rules were written down before each run (`docs/PREREG_night4.md`).

Layout:
- `v*/SKILL.md` — every released or evaluated version (v0.4.0 kept as the first rejected agent revision).
- `candidates/<round>/<model>/SKILL.md` — all agent-written candidates of rounds 2–4, including the rejected ones.
- `runtime/` — the executable part of the Skill used by the controller: `scripts/cpu_check.py` (spiking contract,
  causality, shapes), `scripts/func_equiv.py` (functional-equivalence test), `scripts/lib` (vendored OpenArchEvo subset
  with its license), and `references/library/` (parent programs and their design rationales).
- The evidence files given to each round's agents are in [`../prompts/skill_evo/`](../prompts/skill_evo/).
