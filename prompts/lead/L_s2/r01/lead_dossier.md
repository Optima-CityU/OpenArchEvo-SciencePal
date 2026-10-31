# Research-lead dossier, generation 1

## Results table (all candidates; score = overfitting-aware 2-epoch fitness OAF = val + 0.5*(val - train), lower is better; it predicts 30-epoch quality better than val PPL; delta = vs first parent; params at CPU size)

| id | gen | role | parents | val PPL | delta vs parent | params/block | status | mechanism / hypothesis |
|---|---|---|---|---|---|---|---|---|
| i00-deltanet | 0 | seed |  | 82.03 |  | 34128 | seed |  |
| i00-hgrn2 | 0 | seed |  | 86.84 |  | 29568 | seed |  |
| i00-mamba2-ssd | 0 | seed |  | 88.77 |  | 58568 | seed |  |
| i03-s0335 | 0 | seed |  | 85.27 |  | 58698 | seed |  |
| i09-s0557 | 0 | seed |  | 89.26 |  | 34144 | seed |  |
| i10-s0337 | 0 | seed |  | 85.35 |  | 81488 | seed |  |

## Knowledge
No archive knowledge is provided in this run: base your directions only on the results table above.

## Fixed slots for this generation

- slot 1: island 0, role explore, parents ['i10-s0337'], weight [quality, size] = [1.0, 0.0]
- slot 2: island 0, role recombine, parents ['i03-s0335', 'i00-mamba2-ssd'], weight [quality, size] = [0.938, 0.062]
- slot 3: island 0, role simplify, parents ['i03-s0335'], weight [quality, size] = [0.875, 0.125]
- slot 4: island 1, role recombine, parents ['i00-deltanet', 'i00-mamba2-ssd'], weight [quality, size] = [0.812, 0.188]
- slot 5: island 1, role simplify, parents ['i00-deltanet'], weight [quality, size] = [0.75, 0.25]
- slot 6: island 1, role explore, parents ['i03-s0335'], weight [quality, size] = [0.688, 0.312]
- slot 7: island 2, role simplify, parents ['i03-s0335'], weight [quality, size] = [0.625, 0.375]
- slot 8: island 2, role explore, parents ['i03-s0335'], weight [quality, size] = [0.562, 0.438]
- slot 9: island 2, role recombine, parents ['i00-deltanet', 'i03-s0335'], weight [quality, size] = [0.5, 0.5]
