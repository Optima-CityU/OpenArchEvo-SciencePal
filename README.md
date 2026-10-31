# OpenArchEvo v1.5 — Agents propose, evolution decides (众智提出，演化择优)

Large language models are becoming designers of algorithms and architectures, and architectures are the foundation of
progress in AI. [OpenArchEvo](https://github.com/Optima-CityU/OpenArchEvo) ([arXiv 2609.40258](https://arxiv.org/abs/2609.40258))
uses an LLM as the mutation operator of an evolutionary search over complete PyTorch programs for native spiking
sequence models. **OpenArchEvo v1.5** (SciencePal-assisted OpenArchEvo, our entry to the SciencePal Global AI4Science
Hackathon 2026) shows how agents on the SciencePal platform take part in and accelerate this workflow: GPU sandbox
labs train candidates, multi-model panels (GPT-6.1, Kimi K3, GLM-5.3, DeepSeek-V4) propose hypotheses, Skills and
evaluation rules, and a research-lead agent steers each generation. Agents propose; pre-registered evolutionary
selection decides what survives.

**Project page:** `site/` (deployed to GitHub Pages by `.github/workflows/pages.yml`), with a 4:07 video (English voice, bilingual subtitles: `video/subs_*.srt`, script `video/script2.json`, procedurally composed background music `video/compose.py`).

## Four co-evolving objects

| Object | Who proposes | How evolution decides | Where |
|---|---|---|---|
| **Architectures** (complete PyTorch `SpikingFLABlock` programs) | DeepSeek-V4-Flash proposer reading the Skill; a SciencePal research-lead agent sets per-island directions | CPU contract check → behavioural-fingerprint and functional-equivalence gates → NSGA-II selection → GPU training on SciencePal labs | `evo/controller.py`, `evo/lead.py`, `results/` |
| **Proposer Skill** (`SKILL.md`) | Panels of 5 SciencePal models read evidence from the search and rewrite the Skill | Fixed CPU benchmark with pre-registered promotion rules | `evo/skill_evo*.py`, `skills/` |
| **Scientific hypotheses** | Multi-model panels (E21 gate hypotheses, E35 gate activations, E22/E44 independent re-analyses) | 30-epoch GPU trainings, multiple seeds | `evo/e*_panel.py`, `prompts/panels/`, `results/panels/` |
| **Fitness / evaluation rule** | Panels design cheap predictors of 30-epoch quality (E32) | Out-of-sample rank agreement, then a full search per rule (E40) and counterfactual selection (E45) | `evo/controller.py` (`fitness: "oaf"`), `analysis/` |

The evaluation rule that won is the overfitting-aware fitness **OAF = val₂ + 0.5·(val₂ − train₂)**, read from the
2-epoch training log.

## Key results

| Result | Value |
|---|---|
| Champion `r02a-e29092` (post-`o_proj` spiking multiplicative gate), 30 epochs, 3 seeds | **68.95** val PPL vs expert DeltaNet **71.8**; same-parameter controls 70.74 / 70.66 |
| E41 ablation of the champion (one change each) | gate before `o_proj` 69.57 · gate reads `o_proj` output 70.60 · non-spiking (ANN) gate 71.43 · old sigmoid gate 73.54 |
| E40 main experiment, run mean of delivered architectures (3 paired runs, 30 epochs) | V (2-epoch val) 88.84 · O (OAF) 72.02 · **L (OAF + research lead) 70.90** |
| E40 deliveries worse than DeltaNet by > 3 PPL | V 6/12 · O 2/12 · **L 0/12**; best L delivery `r04e40l-93af64` 69.92 (seed 2: 69.54) |
| E45 counterfactual selection inside the same pools | OAF picks better than val₂ in 2/3 V pools; val₂ picks worse than OAF in 2/3 O pools |
| E42 transfer to seq 512 (15 epochs, 2 seeds) | champion 74.30 · DeltaNet 74.66 · same-parameter control 75.19 |
| Skill lineage | v0.4 ✗ → v0.5 ✓ (parent-equivalent proposals 19% → 2%) → v0.6 ×6 ✗ → v0.5.1 ✓ → **v0.7 ✓** (distinct new proposals 42 vs 32) |

Figures: [`figures/`](figures/) (F1 system, F3 rank fidelity, F4/F5 selection rules and OAF, F7 gate ladder,
F8 Skill lineage, F9 champion ablation, F10 E40 arms, F11 E45 counterfactual). Pre-registrations and round reports
(Chinese originals): [`docs/`](docs/), starting with `docs/PREREG_night4.md` and `docs/RESULTS_round9_1031.md`.

## Repository layout

```
evo/        evolutionary controller (OAF fitness, FE gate, NSGA-II), arm driver run_arm2.py, research lead (lead.py),
            island managers, multi-model panels (sp_panel.py, e*_panel.py), Skill evolution (skill_evo2-4.py),
            Skill benchmark (batch_propose.py, ladder_stats.py, sibling_equiv.py), CPU fingerprint (fp_cpu.py)
lab/        SciencePal GPU-lab harness: lab_client, feeder (job queue + shared pool), lab_worker, collector, keeper,
            watchdog, sandbox setup, training configs (wikitext2_lab.yaml, wt2_s512.yaml)
harness/    SciencePal API client with a call-budget ledger (sp_client.py), agent task/turn runners, lab supervisor,
            OpenAI-compatible proposer client (llm_propose.py)
prompts/    agent role definitions: research-lead dossiers and prompts, lab start/supervise tasks, panel prompts and
            inputs (E21-E44), Skill-evolution evidence per round, proposer/reviewer system prompts
skills/     every Skill version, all agent-written candidates, lineage.json, and the Skill runtime (CPU checks,
            functional-equivalence test, parent library)
results/    searched architectures (code), e40_runs.json, long_runs.json, three views, panel outputs, analysis data
analysis/   scripts that produce the reported tables (E40, E45, fidelity, final analysis, Skill Pareto)
figures/    F1-F11 (png + svg) and their scripts
scripts/    launch_e40.sh, deliver_e40.py, skill_benchmark.sh, start_lab_daemons.sh
config/     labs.example.json, an E40 run config example
docs/       pre-registration log and round results
```

## Reproduce

Requirements: Python ≥ 3.10, `numpy scipy matplotlib httpx torch einops sentence-transformers`, and a checkout of
[OpenArchEvo](https://github.com/Optima-CityU/OpenArchEvo) next to this repository (or set `OPENARCHEVO_DIR`).

Environment variables (names only; see `.env.example`):

| Variable | Purpose |
|---|---|
| `SCIENCEPAL_API_KEY` | SciencePal agent runs, sandbox file API (labs, panels, research lead) |
| `SCIENCEPAL_BASE_URL` | optional API base override |
| `OAE_LLM_BASE_URL`, `OAE_LLM_API_KEY`, `OAE_LLM_MODEL` | OpenAI-compatible endpoint for the proposer model (DeepSeek-V4-Flash) |
| `OPENARCHEVO_DIR` | OpenArchEvo checkout (default `../OpenArchEvo`) |
| `OAE_LABS_JSON` | `{lab_name: SciencePal thread id}` of your GPU labs (default `config/labs.json`) |
| `SKILL_MD_PATH`, `OAE_SKILL_DIR` | proposer Skill text (default `skills/v0.7.0/SKILL.md`) and runtime (default `skills/runtime`) |

Reported tables from the released data (no API access needed):

```bash
python analysis/e40_analysis.py      # E40 arms V / O / L
python analysis/e45_analysis.py      # E45 counterfactual selection
python analysis/fidelity_curve.py    # rank agreement of epoch-k vs 30-epoch PPL
python analysis/final_analysis.py    # E29-E39 selection-rule analyses
python figures/scripts/f10_e40.py    # regenerate a figure
```

Full loop on SciencePal:

```bash
# 1. start GPU labs (prompts/lab/TASK_start.md is the lab agent's start task), list their threads in config/labs.json
python harness/lab_supervise.py --lab lab1=new --lab lab2=new
scripts/start_lab_daemons.sh                 # feeders + result collector
# 2. one evolutionary run: OAF fitness + research lead, Skill v0.5.1 (E40 arm L)
python evo/controller.py --state runs/e40/L_s1 init --cfg "$(cat config/e40_L_config.example.json)" \
  --seeds i09-s0557 i03-s0335 i00-deltanet i00-hgrn2 i00-mamba2-ssd i10-s0337
SKILL_MD_PATH=skills/v0.5.1/SKILL.md python evo/run_arm2.py --state runs/e40/L_s1 --rounds 1 2 3 4 --k 3 --lead
#    all 9 E40 runs and the 30-epoch deliveries:
scripts/launch_e40.sh
# 3. one round of Skill evolution: pack evidence, let 5 models rewrite the Skill, then benchmark the candidates
python evo/skill_evo4.py run && python evo/skill_evo4.py extract
scripts/skill_benchmark.sh runs/ladder skills/v0.5.1/SKILL.md build/skill_evo4/cand_*/SKILL.md
```

Run configs: `config/e40_L_config.example.json` (`"fitness": "val"` for arm V, `"oaf"` for arms O and L). Panels,
the research lead and Skill evolution continue existing SciencePal threads (one sandbox per thread); list reusable
thread ids in `runs/lead_threads.json` or let `lead.py` create new runs.

## Scope

Results are for the hackathon protocol (WikiText-2, seq 128, 30-epoch confirmation, 2–3 seeds); see `docs/` for the
pre-registered predictions and their outcomes.

## Citation and license

Built on OpenArchEvo: code <https://github.com/Optima-CityU/OpenArchEvo>, paper arXiv:2609.40258.
Released under the [Apache License 2.0](LICENSE). `skills/runtime/scripts/lib` vendors a subset of OpenArchEvo under
its own license notice.
