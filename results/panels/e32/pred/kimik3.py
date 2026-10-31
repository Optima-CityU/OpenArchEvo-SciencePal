"""Cheap predictor of 30-epoch validation PPL from a 2-epoch run.

score(row, code) -> float, lower = predicted better 30-epoch PPL.

Mechanism: architectures that are still far from converged at epoch 1
(high train_ppl_ep1 = large remaining learning potential) keep improving
through epoch 30, while fast learners that already reach low train PPL by
epoch 1 plateau early. val_ppl_ep2 alone ranks poorly because it rewards
fast early convergence. We subtract a fraction of train_ppl_ep1 (the
learning-potential signal) from val_ppl_ep2 to reward slow learners.

Formula: score = val_ppl_ep2 - 0.1 * train_ppl_ep1
One fitted constant (0.1). Uses only 2-epoch columns; `code` is accepted
but not required.
"""


def score(row: dict, code: str) -> float:
    val_ep2 = float(row["val_ppl_ep2"])
    train_ep1 = float(row["train_ppl_ep1"])
    return val_ep2 - 0.1 * train_ep1
