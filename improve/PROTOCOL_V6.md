# A pre-trained foundation model with target history: version D of the cost-of-strictness table (frozen BEFORE any real forecast)

Written: 2026-10-22, before any forecast of any city was made with the model. Committed and time-stamped (`PROTOCOL_V6.timestamp`) with the plan.

## Why

The first author asked for one table that holds the strictly zero-shot method and the versions that use target data, and for a comparison
with a trustworthy pre-trained foundation model. The "Cost of strictness" table of the paper (versions A, B and C, December 2021,
Chicago and New York City) is extended by a version D: a pre-trained time-series foundation model, used as released, that reads the
previous seven days of each region. Simple history baselines (last week, four-week average) were computed for context
(`results/history_baselines.txt`, PROTOCOL_V5.md) and are not part of the paper's table.

## Model and setting (fixed now)

- Model: Chronos-Bolt (small), Ansari et al. 2024 / Amazon, Apache-2.0, 47.7 million parameters, weights 190.9 MB, from Hugging Face
  (`amazon/chronos-bolt-small`); package chronos-forecasting 1.5.0, transformers 4.46.3, torch 2.1.1 on CPU, float32, in the separate
  environment `.venv_fm` (the environment of all other experiments is not changed). Used as released: zero-shot, no fine-tuning, no
  training on any data of this study, univariate (each region on its own).
- Targets and period: Chicago (77 community areas) and New York City (263 zones), December 2021, 30-minute slots, all 1,488 slots, the
  setting of the existing table (versions A, B, C).
- Input: for each region, the previous 336 observed half-hour counts (seven days, reaching back into November for the first origins).
  Output: the next 6 slots (three hours). Origins at 00:00, 03:00, ... of every day of December, so each December slot is forecast exactly
  once and the horizon steps 1 to 6 are used equally. The context length 336, the horizon 6 (the first-six-steps setting of the OpenCity
  comparison) and the origins were fixed before any real forecast and are not tuned.
- Point forecast: the model returns the quantiles 0.1, 0.2, ..., 0.9. The point forecast for MAE and RMSE is the mean of the nine
  quantiles ("quantile mean", comparable with the mean forecasts of A, B and C), clipped at zero. The MAE of the median forecast (quantile
  0.5) is also reported. A non-finite forecast would be replaced by zero and counted.
- Scoring (`fm_chronos.py`): MAE and RMSE over all region-slots of December 2021, and the MAE by horizon step. As a consistency check, version
  A is recomputed here for the same slots from the archived predictions of the final method with the strict city totals; it must agree with
  the table (Chicago 2.18 / 6.31, New York City 9.44 / 21.02) within 0.02, otherwise this is reported. The cost of strictness against D is
  computed like the other rows of the table.
- Prediction, fixed now: D has a lower MAE than A in both cities. No other prediction. All results are reported as obtained.

## Disclosure

- The model was run only on synthetic Poisson series (speed, shape, finiteness and all-zero checks) before this protocol; no demand of any
  city was forecast.
- Version D uses the target's own demand of the previous seven days; it is not zero-shot in the sense of this study and is labelled so,
  like B and C.
- Other foundation models (OpenCity, UniST, TimesFM, Moirai), other contexts or horizons, and other cities are not part of this protocol.

## Addendum 1 (2026-10-22, before any result of this protocol was looked at)

By decision of the first author, the result of this protocol is reported in the comparison with published results and foundation models (the
table with published results, with the column "target data used" = the previous seven days), and not as a version D of the cost-of-strictness
table, which stays unchanged. The model, the setting, the scoring, the consistency check of version A and the prediction above are not
changed. The simple history baselines of PROTOCOL_V5.md (comparison 2) are not put in the paper.
