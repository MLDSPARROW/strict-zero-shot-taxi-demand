# Ours + 7 days, version 2: averages of the foundation model and the cross-city forecasters (frozen BEFORE any average is computed)

Written: 2026-10-26, after a request of the first author ("what if we use the last 7 days of data, predict the next 3 hours, and our error is better than the foundation model") and after the failure of PROTOCOL_V8.
Committed and time-stamped (`PROTOCOL_V16.timestamp`) with `v16_dev.py` before any number below was computed.

## Why

PROTOCOL_V8 ended with the best cross-city forecaster (C2) 5.9% (Chicago) and 10.2% (NYC) behind Chronos-Bolt-small on October and November 2021, and fixed that no variant be tried without a new protocol. Two different
forecasters that read the same seven days (a pretrained foundation model, and an XGBoost forecaster trained on helper cities only) often make partly different errors, and their average is then better than both. This protocol
tests exactly that, with no new training: the saved development forecasts of both families are averaged with fixed weights. Setting, origins, slots, horizon (6 slots = 3 hours), point forecast (the mean forecast), metrics
and the bootstrap are those of PROTOCOL_V6 / V8. Version E of the cost table would use target data (the previous seven days), exactly as Chronos-Bolt; it is not a zero-shot result.

## Rules (fixed now; mean forecasts per region and slot; no weight is tuned)

S1 = (Chronos-Bolt + C2) / 2, S2 = (2 Chronos-Bolt + C2) / 3, S3 = (Chronos-Bolt + C1 + C2 + C3 + C4) / 5, with C1 to C4 the candidates of PROTOCOL_V8 (scale-matched helpers, two partitions, `out_xc/pred_dev_*`).

## Development rule (October and November 2021, `v16_dev.py`)

A rule passes if, in both cities, its MAE and its RMSE are below those of Chronos-Bolt-small and the upper end of the 95% interval (7-day block bootstrap, 2,000 draws) of the mean daily MAE difference (rule minus Chronos-Bolt) is below zero.
If more than one rule passes, the one with the smaller mean over the two cities of MAE(rule) / MAE(Chronos-Bolt) is chosen. If none passes, nothing is run for December, nothing of this goes into the paper, and the result is recorded here.

## Test (December 2021, once, only if a rule passes)

The cross-city forecasters needed by the chosen rule (C2 for S1 and S2; C1 to C4 for S3) are trained for December as in PROTOCOL_V8 (ten scale-matched partitions, before 1 December 2021, origins every 6 slots), and averaged with the
December forecasts of Chronos-Bolt-small saved by PROTOCOL_V6. Hypothesis H1: the chosen average has a lower MAE and RMSE than Chronos-Bolt-small in both cities with the upper end of the 95% interval of the daily MAE difference
below zero. If H1 holds, a version E ("ours + Chronos-Bolt, previous seven days") is added to the cost table of the paper next to version D, labelled as using the previous seven days of the target and as a weighted average of two
forecasters. If H1 fails, the paper keeps version D alone, as now.

## Disclosure

The December forecasts of Chronos-Bolt and the development results of C1 to C4 were known. No average of the two families had been computed. The weights 1/2, 2/3 and 1/5 were not chosen from any number.
