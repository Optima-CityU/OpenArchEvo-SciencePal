# Results: searched architectures and evolution records

All numbers are WikiText-2 perplexities of native spiking sequence models (SLTT neurons, seq 128, the hackathon
protocol defined in `lab/wikitext2_lab.yaml`), trained on SciencePal GPU sandbox labs.

| File | Contents |
|---|---|
| `architectures/<id>.py` | Complete PyTorch programs (one `SpikingFLABlock` each) for the champion, the earlier sigmoid-gate champion, the expert DeltaNet seed, the E41/E33/E35/E36 ablation and gate variants, the same-parameter controls, every E40 delivered architecture and every E45 counterfactual pick. `architectures/index.json` lists role and 30-epoch results per seed. |
| `e40_runs.json` | Full record of the E40 main experiment (see schema below). |
| `long_runs.json` | Every GPU training of ≥ 6 epochs: architecture, seed, epochs, protocol, experiment label, best val PPL, test PPL, best epoch, test firing rate. |
| `three_view/<id>/` | Three views of four key architectures: `code.py`, `rationale.md` (the proposing agent's design rationale), `fingerprint.json` (21-feature OpenArchEvo behavioural fingerprint: zero-cost proxies + architecture statistics). |
| `panels/<exp>/<model>.json` | Raw outputs of SciencePal multi-model panels: E21 hypotheses, E22/E44 independent re-analyses, E23 literature, E32 30-epoch predictors (`pred/*.py`), E35 gate proposals. |
| `data/lab_jobs.json` | Trimmed store of all 1,551 GPU jobs (train + fingerprint): protocol, args, metrics or fingerprint. Input of `analysis/*.py` and `figures/scripts/*.py`. |
| `data/curves_30ep.json`, `data/curves_2ep.json` | Per-epoch `[epoch, train_ppl, val_ppl]` curves for 30-epoch and 2-epoch trainings. |
| `data/e29_pick.json`, `data/e29_step2.json`, `data/fidelity_curve.json` | Inputs/outputs of the prospective-selection and rank-fidelity analyses. |

## `e40_runs.json` schema

```
runs.<V_s1..L_s3>:
  arm, arm_description, paired_seed_index, skill, config{...}
  seed_architectures: [{id, measured_val2}]
  delivered: [id, id]                 # top-2 by the arm's own fitness, trained 30 epochs
  counterfactual_picks_e45: [id, id]  # top-2 under the other arm's rule (V/O runs only)
  generations: [
    { generation: 1..4,
      research_lead_directives: [...],   # L runs: per-island slots {role, parents, weight, direction}
      research_lead_analysis: "...",     # L runs: the lead agent's written analysis
      candidates: [
        { id, generation, role, parents, directive, name, rationale (<=700 chars), hypothesis,
          cpu_attempts, cpu_passed, cpu_fail_stages,
          status: registered | rejected | parse_error | llm_error, reject_reason,
          island, predicted_fitness,
          parent_similarity: {parent: [s_code, s_rationale, s_fingerprint]},
          functional_equivalence_gate: [{parent, equivalent, spikes_equal, max_abs_dy, note}],
          selection_objectives: [-predicted_fitness, novelty], selection_rule,
          trained, val2, train2, oaf, fitness, test2, fire_rate2, train_error,
          delivered, counterfactual_pick_e45,
          epoch30_best_val_ppl_by_seed: {E40: {seed: ppl}, E45: {seed: ppl}} } ] } ]
```

`fitness` is the score the arm actually selected on (V: `val2`; O and L: `oaf = val2 + 0.5·(val2 − train2)`).
For V runs `oaf` is computed post hoc from the same epoch-2 log and marked with `oaf_note`.

Counts: 9 runs, 324 proposals, 202 registered, 122 rejected before training (62 near-duplicate, 32 behaviour-identical
to parent, 21 infeasible, 7 fingerprint failures), 108 trained for 2 epochs, 18 delivered architectures × 3 seeds × 30
epochs, 12 counterfactual picks (3 coincide with deliveries; the other 9 trained 2 seeds × 30 epochs).

Regenerate the reported tables with `python analysis/e40_analysis.py` and `python analysis/e45_analysis.py`.
