"""Parameter-free rank predictor from the two-epoch run.

Lower scores predict better 30-epoch validation PPL. The score is in log
units and is not a calibrated prediction of the final perplexity.

Mechanistic hypothesis: conditional on endpoint validation loss, a larger
running-train/endpoint-validation loss gap indicates optimization headroom
and/or robustness to dropout, predicting better longer-run performance.
This is an association to test, not an established causal explanation.

Only train_ppl_ep2 and val_ppl_ep2 are accessed. The program source, architecture
ID, epoch-30 labels, other rows, and filesystem are deliberately unused. Training
PPL is a dropout-on epoch average, not an endpoint training-set measurement.
"""
import math


def score(row: dict, code: str) -> float:
    """Return log(val2) minus the log(train2/val2) headroom proxy."""
    train2 = float(row['train_ppl_ep2'])
    val2 = float(row['val_ppl_ep2'])
    if not (math.isfinite(train2) and math.isfinite(val2)
            and train2 > 0.0 and val2 > 0.0):
        raise ValueError('Epoch-2 perplexities must be finite and positive.')
    return 2.0 * math.log(val2) - math.log(train2)
