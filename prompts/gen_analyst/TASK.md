Generation Analyst task (OpenArchEvo v2). Work unattended; never ask questions; no web search.

Attached (find under /workspace; otherwise use find, excluding /proc): gen_results.json.
It contains this night's trained candidates of an evolutionary search over native spiking sequence-model blocks:
- "children": each trained child with id, arm (which coordination scheme proposed it), parent id(s), family seed, e6 val/test PPL
  (protocol sp-wt2-sltt-s128-e6-confirm: 6 epochs WikiText-2, seq 128, SLTT neurons, single seed; seed noise about 0.3-0.5 PPL),
  params per block, the proposer's hypothesis and mechanism summary, and a unified diff vs its first parent.
- "references": e6 values of the seeds/parents (some are 3-seed means), and 30-epoch results for architecture A (r03b-422493)
  vs DeltaNet over 3 seeds.

Your job (like an "analyzer" in an automated research loop): explain what this generation tells us.
1. For every child: compare with its parent and siblings (other children of the same parent or family). Which code change is the
   most likely cause of the difference? Is the difference larger than noise (0.5 PPL)? Did the proposer's hypothesis hold?
2. Across children: which mechanisms look promising, which failed, and which results are not interpretable (confounds such as
   parameter count changes).
3. Propose 5 concrete next experiments (parent + change + what result would confirm/refute), ranked.

Write /workspace/analysis.json:
{"per_child": [{"id": ..., "parent": ..., "delta_vs_parent": <number you computed>, "beyond_noise": true|false,
  "likely_cause": "...", "hypothesis_held": "yes|no|unclear", "confounds": "..."}],
 "patterns": ["<=40 words each", ...], "next_experiments": [{"parent": ..., "change": ..., "confirm_if": ...}, x5]}
Every number must be computed from gen_results.json (use Python). Validate that the JSON loads. Print it as your final message.
