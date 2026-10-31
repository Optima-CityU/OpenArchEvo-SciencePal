# Review task: one generation of OpenArchEvo candidates

You are the scientific reviewer of an evolutionary search over native spiking sequence-model blocks.
Use the `openarchevo-snn-proposer` skill for the contract (SKILL.md section 1) and the parent library.
Work unattended; do not ask questions; do not use web search.

Input: `/workspace/review/generation.json` (written by the controller). For every candidate it contains:
id, role, parents, hypothesis, pre-registered predicted_signature, rationale, code path, and the
controller's measurements: CPU contract check, three-view similarity to each parent
(code / rationale / fingerprint, each in [0,1]; fingerprint = initialisation-time behaviour under a fixed probe),
and, if trained, validation perplexity under a short training protocol.

For each candidate (read its code; keep your notes short):
1. Mechanism check: does the code actually implement the mechanism the hypothesis describes? (yes / partly / no + one sentence)
2. Contract risks the CPU check could miss (future leakage through normalisation or pooling, non-binary inputs to linear layers, hidden O(L^2) ops, hard-coded neuron type). List concrete line-level issues or "none found".
3. Novelty judgement: given the three-view similarities, is this a genuinely different design from its parents,
   a re-parameterisation, or a near-copy? Note any case where code similarity and fingerprint similarity disagree, and explain why from the code.
4. One-line verdict: keep / keep-but-fix / discard, and why.

Then write a short generation summary (<= 200 words): which mechanisms look promising for the next generation and
which directions look exhausted. Measured numbers must be quoted exactly as given; do not invent results.

Output: `/workspace/review/review.json` with fields `candidates: [{id, mechanism_check, contract_risks, novelty, verdict, notes}]`
and `summary`. Print its path as the last line.
