# Raw logs: 15 evolution runs of a spiking language-model architecture search

Three arms, 5 independent runs each (same 6 expert seed architectures, different random seeds):
- **V1**: rule-based island model; an LLM writes child programs following Skill v0.1.
- **A**: same as V1, but the LLM follows a revised Skill v0.3.3.
- **V2**: Skill v0.3.3 plus a "research lead" LLM agent that writes per-slot research directions each generation, plus extra deterministic gates.

Each generation proposes 9 child programs; most are rejected by CPU checks or duplicate filters (event `proposal_rejected`); a few are selected and trained on GPU for 2 epochs on WikiText-2 (event `train_submitted`), and the result comes back as `measured` (validation perplexity, lower is better) or `train_failed`.

Files per run (`runs/<arm>_<seed>/`):
- `events.jsonl`: one JSON per line, with `ts` (unix time) and `event`. Relevant events: `train_submitted` (id, job), `measured` (id, value = val PPL), `train_failed`, `infra_retrain` (an infrastructure failure, not the model's fault, was retrained with the same settings; the retrained result arrives later as `measured` / `train_failed` for the same id), `selected_for_training`, `proposal_start`, `proposal_rejected` (reason).
- `population_slim.jsonl`: final population (id, round, parents, fitness = −val PPL, role).

The expert seed architectures (ids starting with `i`) are not trainings of the run. The best expert seed has val PPL 120.641.
