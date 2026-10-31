# Independent re-analysis of round 8 (10/29-10/30)
- trainings.csv: every successful training job. job_id = tr-<architecture id>-<suffix>; protocol sp-wt2-sltt-s128-e2 / -e6 / -e30-rank (2 / 6 / 30 epochs, WikiText-2, seq 128); val_ppl_best = best validation PPL over the run (the reported metric); test_ppl at that checkpoint.
- curves_2ep.csv: per-epoch train/val PPL of 2-epoch runs (train PPL = running average over the epoch with dropout; val at epoch end). Search-time logs have job ids tr-<arch>-<6 hex>; re-runs carry tags like e31be2s0 / e39e2s1.
- e29_pick.json: the 16 candidates of the prospective selection test (E29/E31) with their search-time 2-epoch val; e29_step2.json: their 6-epoch values and the 6-epoch top-4.
- e32_train.csv / e32_test.csv: the data given to the fitness-design panel (E32).
- prereg_round8.md: the pre-registration and claimed verdicts, in Chinese, with timestamps.
DeltaNet expert = architecture id i00-deltanet.
