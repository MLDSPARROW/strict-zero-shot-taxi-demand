# Helper-gated ridge tilt of the final method: second look at the held-out targets (frozen BEFORE any held-out pseudo-target run and before any held-out number of the tilt)

Written: 2026-10-26. Committed and time-stamped (`PROTOCOL_V14.timestamp`) with `runner_v14.py` and `v14_gate.py` before the pseudo-target runs below start.

## Why

PROTOCOL_V13 (development targets only, helper data only) found that a ridge tilt of the shares of the final method v1 is stable and useful when NYC is the target (helper folds +16.3% and +10.4%, all coefficients positive,
then NYC 7.454 -> 6.733 MAE) and harmful when Chicago is the target (helper folds -22.6% and -4.0%, coefficient signs flip). The check that separates the two cities uses helper data only. This protocol freezes it as a
rule and tests it as a second look on the held-out targets.

## Method (fixed now)

v1, the tilt (features, ridge, clipping, airport regions untouched) and the pseudo-targets are those of PROTOCOL_V13 / `v13_tilt.py`, with three changes:
1. lambda is fixed at 1 (no selection on any target; it was chosen in V13 for NYC and is not re-chosen).
2. Leave-one-helper-out gate with k folds (k = 2 for SF, sf_zcta; k = 2 for Chicago, NYC; k = 3 for DC, dc_cluster, Austin): in fold j the tilt is fitted on the pseudo-predictions of the other share helpers and applied to the
   pseudo-prediction of share helper j. The tilt is applied to the target only if (a) the relative gain of the allocation MAE (true slot totals of the helper) is at least 2% in every fold, and (b) the coefficients of log jobs and log residents
   are positive in every fold fit. Otherwise v1 is used unchanged for that target. The rule is not relaxed after the results.
3. If the gate passes, the tilt is fitted on all share helpers with lambda = 1 and applied to the v1 of the target.
The 2% threshold, lambda = 1 and the sign rule were fixed after the V13 development results were known (an external review round, 2026-10-26, advised a threshold above zero); they are frozen here, before any held-out number.

Pseudo-target predictions for the held-out targets: 13 configurations (`runner_v14.py`), heads eh and eo, runs r = 1..5 (helper partition r, seed r - 1), trained on the other share helpers only, scale-matched to the target as in the paper.
Held-out v1 predictions are those already on disk (`out_cal/`), unchanged.

## Evaluation (`v14_gate.py`)

`python v14_gate.py dev` reproduces the gate for Chicago and NYC with the frozen rule. `python v14_gate.py heldout` scores v1, the tilt without the gate and the gated tilt on SF, DC (hourly), sf_zcta, dc_cluster (hourly) and Austin
(248 days), with the city totals and bootstrap of `v10_heldout.py`. A target whose gate fails counts as no change. Adoption rule (as PROTOCOL_V10): the mean MAE change and the mean RMSE change over the five targets are both above zero,
and the gated tilt is better than v1 in MAE in at least three of the five and in RMSE in at least three of the five. Otherwise v1 stays. This is a SECOND LOOK: the held-out numbers of the jobs-and-residents allocation were known
(PROTOCOL_V10), the helper data are the same cities, and the gate with threshold 2% was written after the development results. No untouched target with taxi demand exists.
Diagnostics printed for every target: the fold table (gain, jobs, residents, busyness and area coefficients), the mass moved by the tilt and the share of regions whose share rises.

## Disclosure

The development numbers of v1, of the fixed mixes (PROTOCOL_V12) and of the tilt (PROTOCOL_V13) were known. No pseudo-target prediction and no held-out number of the tilt existed when this was written.
