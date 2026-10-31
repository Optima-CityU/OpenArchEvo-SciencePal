"""Cheap 30-epoch val PPL predictor from 2-epoch signals + program source.

Mechanistic idea: the epoch-2 generalization margin (val_ppl_ep2 / train_ppl_ep2,
train PPL measured with dropout on) reflects how fast an architecture memorises
training noise relative to generalising. Architectures with a large margin keep
losing ground over the remaining 28 epochs, whatever their absolute early val
level. The element-wise HGRN2 recurrence family additionally lags at 30 epochs.

score = 100 * (val_ppl_ep2 / train_ppl_ep2) + 0.21 * val_ppl_ep2 + 1.0 * hgrn2
lower score = predicted lower (better) 30-epoch val PPL.
Fitted constants: 0.21 (weight of val_ppl_ep2 vs ratio) and 1.0 (HGRN2 penalty),
grid-searched for Spearman rho on the 17 training rows; LOO-refit picks the same
constants in 11/17 folds. Train rho = 0.7517.
"""

HGRN2_MARKER = "hgrn2_chunkwise"
W_VAL = 0.21
HGRN2_PENALTY = 1.0


def score(row: dict, code: str) -> float:
    """Predict the 30-epoch validation perplexity rank of one architecture.

    Args:
        row: dict of test.csv columns as strings (needs train_ppl_ep2,
             val_ppl_ep2).
        code: the program source text.

    Returns:
        float, lower = predicted better 30-epoch val PPL.
    """
    v2 = float(row["val_ppl_ep2"])
    t2 = float(row["train_ppl_ep2"])
    ratio = v2 / t2
    hgrn2 = 1.0 if HGRN2_MARKER in code else 0.0
    return 100.0 * ratio + W_VAL * v2 + HGRN2_PENALTY * hgrn2
