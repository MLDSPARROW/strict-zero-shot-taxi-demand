# Helper-based hard selector among v1, XGBoost, jobs-and-residents and the tilt (frozen BEFORE any pseudo-target XGBoost prediction exists and before any selector number)

Written: 2026-10-26. Committed and time-stamped (`PROTOCOL_V15.timestamp`) with `xgb_pseudo.py` and `v15_selector.py` before they are run. The rule follows an external review round (a language model,
asked with the numbers of this project, 2026-10-26); it is a challenger to v1, which stays the final method unless the adoption rule below holds.

## Question

PROTOCOL_V10 to V14 showed that no single allocation wins in every city (v1: Chicago; scale-matched XGBoost: NYC and coarse SF; jobs-and-residents: SF, DC and the coarse targets). Can a rule that uses only the share
helpers of the target choose among them?

## Candidates (shares per slot, the same city totals as the paper)

v1 (final method), g (scale-matched XGBoost, ten-run ensemble, `out_xgb/`), jr (jobs + residents, public), tilt (ridge tilt of v1, lambda 1, `v14_gate.py`; the sign rule of the V14 gate is NOT used here).

## Rule (fixed now)

For a target T the share helpers are its pseudo-targets (each predicted by models trained on the other share helpers only: v1 and tilt from the pseudo-target runs of `runner_v14.py` and PROTOCOL_V2 addendum 6,
g from `xgb_pseudo.py` with the configuration of `xgb_scale.py`, 5 runs; jr needs no training). The error of a candidate on a pseudo-target is its allocation MAE with the true slot totals of the pseudo-target on its
observed slots. A challenger (g, jr, tilt) is eligible if its mean pseudo-target error is at least 5% below that of v1 and at least 2% below on every pseudo-target. Among the eligible challengers the lowest mean
error wins; the order g, jr, tilt breaks ties (relative difference below 1e-6); if none is eligible, v1 is used. No threshold is tuned.

## Honest limits

- With two or three pseudo-targets the ranking is noisy and the pseudo-targets are predicted by models trained on fewer helpers than the deployed ones (this favours the public jr). A learning-curve check of this bias
  (pseudo-targets with one and with two training helpers) is NOT run, because it needs many more neural runs; the limit is stated in the paper if the selector is used.
- The thresholds were fixed after the development results of V10 to V14 were known, and after seeing the held-out numbers of jr and XGBoost (V10). The held-out evaluation is a second look, not a confirmation.
- No untouched target with taxi demand exists.

## Evaluation

`python v15_selector.py dev` (Chicago and NYC; strict trips with the city totals of the paper) needs only the dev pseudo-targets and `python xgb_pseudo.py dev`. `python v15_selector.py heldout` scores SF, DC (hourly),
sf_zcta, dc_cluster (hourly) and Austin (248 days) after the pseudo-target runs of `runner_v14.py` and `python xgb_pseudo.py heldout` are complete, with the city totals and the 7-day block bootstrap of V10.
Adoption (held-out, second look): the mean MAE change and the mean RMSE change of the selected allocator against v1 over the five held-out targets are both above zero, the selector is better than v1 in MAE in at
least three targets and in RMSE in at least three, and no target has an MAE change below -5%. Otherwise v1 stays. Every number is reported either way, with the choice of the selector per target, the pseudo-target error table
and the candidate with the lowest MAE on the target (to show when the selector is wrong).

## Disclosure

All development and held-out numbers of v1, XGBoost and jr were known. No pseudo-target XGBoost prediction and no selector decision existed when this was written.
