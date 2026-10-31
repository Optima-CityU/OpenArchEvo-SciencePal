"""Predict 30-epoch validation PPL from a 2-epoch run plus the program source.

score(row, code) -> float, lower = predicted better 30-epoch PPL.

Formula (2 fitted constants: 100, 0.21; marker weight fixed at 1.0):
    score = 100 * (val_ppl_ep2 / train_ppl_ep2) + 0.21 * val_ppl_ep2
            + 1.0 * (1 if 'hgrn2_chunkwise' in code else 0)

Rationale: val_ppl_ep2 alone ranks 30-epoch PPL poorly (Spearman rho ~0.34 on
train). The epoch-2 generalization margin val_ppl_ep2/train_ppl_ep2 (train PPL
is a dropout-on running average) is a stronger early signal (rho ~0.50 alone);
combining it with val_ppl_ep2 reaches rho ~0.70. The 'hgrn2_chunkwise' source
marker flags HGRN2-family blocks that early-saturate (all three such train
architectures land at 30-epoch PPL 81-84), adding a fixed penalty that lifts
train rho to ~0.75 and is robust under leave-one-out (mean ~0.75).
"""


def score(row: dict, code: str) -> float:
    val_ppl_ep2 = float(row["val_ppl_ep2"])
    train_ppl_ep2 = float(row["train_ppl_ep2"])
    margin = val_ppl_ep2 / train_ppl_ep2
    hgrn2_penalty = 1.0 if "hgrn2_chunkwise" in code else 0.0
    return 100.0 * margin + 0.21 * val_ppl_ep2 + hgrn2_penalty
