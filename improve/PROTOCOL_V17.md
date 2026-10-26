# Goal 4: our own forecaster ALONE below Chronos-Bolt-small, with the previous 7 days as input (frozen BEFORE any run of the candidates below)

Written: 2026-10-27, before any forecast by the candidates C5 below. Source: answer of ChatGPT (see the scratchpad question q5) and the decision of the first author
(2026-10-26): no mix with Chronos in any form; our own models only (XGBoost, a neural network, a weekly average, mixes of these); training only on helper cities
(Chicago <- NYC, SF; NYC <- Chicago, SF; as PROTOCOL_V8). Not zero-shot: the previous 336 slots of the target region are an input, as for Chronos-Bolt.

## Setting
As PROTOCOL_V8 (context 336 slots, horizon 6 slots, MAE/RMSE of the mean forecast, development October and November 2021 with origins every 12 slots, December 2021 once).
The baseline for the comparison is C2 of PROTOCOL_V8 (saved `out_xc/pred_dev_C2_*`): Chicago 1.226 / 3.503, NYC 1.738 / 5.511; Chronos-Bolt-small 1.157 / 3.209 and 1.577 / 4.958.

## Step 1: diagnostics (read-only on the saved development forecasts of C2 and Chronos-Bolt; no training)
A. MAE and RMSE per forecast step j = 1..6 of C2 and Chronos, and the relative gap.
B. The same by region size (five equal-count bins of the mean count of the region in its own previous 336 slots).
C. The same by hour of the day of the forecast slot and by weekday / weekend.
D. The paired absolute-error difference (C2 minus Chronos) against the level-shift statistic (|mean of last 12 slots - mean of the preceding 48 slots| / (std of the 336 slots + 0.1)), in deciles.
Printed only; they decide where to look, not which candidate wins.

## Step 2: candidate C5 (C2 with a richer description of the 336-slot context; nothing else changed)
C2 plus these features, all from counts before the origin: the seven same-slot counts of the previous seven days as separate inputs (log1p); the 42 means of consecutive
8-slot chunks of the 336 slots, divided by the weekly mean (log of ratio with +0.1); the standard deviation of the 336 slots, the zero fraction, the mean of the last 24 and
of the last 48 slots (log1p), the ratio of the last 12 slots to the preceding 48 (level shift), and its product with j. Same model (Poisson, offset log lambda0, depth 6,
learning rate 0.05, at most 600 trees, early stopping on held-out helper days), same helper data, partitions r = 1, 2, no map features.
Leak test (as `xc7d.py leak`) on the new features before any forecast of the target.
Rule: C5 passes the screen if the mean over the two cities of MAE(C5)/MAE(C2) <= 0.98, no city above 1.01, and no RMSE above 1.02 of C2.
Goal rule (the real goal 4): MAE and RMSE of the own forecaster below Chronos-Bolt-small in both cities on this period.

## Step 3 (only if C5 does not reach the goal rule; a separate protocol PROTOCOL_V18 will be written before it)
A small neural forecaster on the 336-slot context (log1p, median/IQR scaling of the window, MLP 336-256-128-6 with a seasonal-baseline residual, Huber + MAE loss, trained on
helper cities only), and an own-model blend (weights fitted on helper validation only). Fine-tuning Chronos-Bolt on helper cities is NOT an own model (ceiling check only, not now).

## Step 4: December and other foundation models
December 2021 once, only for a candidate that reaches the goal rule on October and November. TimesFM and Moirai small may be run as extra baselines under the same input
(a separate protocol); OpenCity, UrbanGPT and UniST are quoted from their papers, not compared as a win.
Wording in the paper (honest): cross-city transfer with target-region history, not zero-shot.
