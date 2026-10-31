You are the research lead of island-based evolutionary search over native spiking sequence-model blocks (programs define
`SpikingFLABlock`). Work unattended; never ask questions; no web search.

Find the attached lead_dossier.md and code.zip (under /workspace; otherwise use find, excluding /proc). Read the dossier fully;
open code from the zip only where needed.

The next generation has 9 proposal slots. Their parents, roles and preference weights are FIXED by the evolutionary rules (listed
in the dossier). A separate proposer writes the code from your direction; deterministic code then checks, fingerprints, selects
and trains (2-epoch WikiText-2 validation PPL, lower is better; params per block is the second objective).

For EACH slot write a concrete research direction (2-4 sentences): which mechanism to change in the given parent(s), why the
evidence (results table, checked knowledge) supports it, and what you expect. Respect the slot's weight: [1,0] = only quality
matters; [0,1] = make the block smaller/cheaper while keeping quality; in between = trade-off. Make the 9 directions different
from each other and from directions that already failed (see the table). Do not invent numbers: cite only numbers in the dossier.

Write /workspace/directives.json:
{"analysis": "<=150 words: what the measured results so far suggest (cite ids and numbers from the table)",
 "slots": [{"slot": 1, "direction": "..."}, ... 9 entries in the dossier's slot order]}
Validate it with python (json.load, 9 slots). Print the file content as your final message.
