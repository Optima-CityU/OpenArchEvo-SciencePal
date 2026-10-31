Meta-review task (unattended; do not ask questions; no web search).

The attached meta_review.zip contains meta/data.json: the current SKILL.md of our architecture-proposer agent, every CPU-check failure our proposers produced tonight (stage, error type, message), and a ledger of pre-registered predictions versus measurements (params_per_block, fire_rate) with verdicts. Extract it under /workspace/meta.

Goal: improve the instructions so that a language model following them (a) passes the CPU contract check on the first attempt more often and (b) makes better-calibrated predictions about params_per_block and fire_rate, without making proposals less ambitious.

1. Classify the failures into a few root causes with counts (quote error messages exactly).
2. Diagnose why the params_per_block direction was predicted wrongly so often (look at the ledger details).
3. Write /workspace/meta/skill_patch.md: an "Implementation checklist" section (at most 12 bullet points) and a "How to predict params_per_block and fire_rate" section (at most 8 bullet points) that can be appended to SKILL.md. Be concrete (shapes, einops patterns, how to count parameters of SpikingLinear / projections / convolutions), and do not change the contract or the output schema.
4. Write /workspace/meta/meta_review.json with: failure_classes [{cause, count, examples}], prediction_diagnosis, and the patch text.
Print the path of meta_review.json as the last line.
