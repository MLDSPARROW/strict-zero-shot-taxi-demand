# "Ours + 7 days": a scale-matched cross-city forecaster that reads the previous seven days, against a foundation model on equal terms (frozen BEFORE any forecast of the target by this method)

Written: 2026-10-22, before any forecast of Chicago or New York City by the method below and before any run of Chronos-Bolt on the development
period. Committed and time-stamped (`PROTOCOL_V8.timestamp`) with the plan and the two scripts (`xc7d.py`, `fm_chronos_dev.py`).

## Why

Decision of the first author (2026-10-22): the comparison with the foundation model is put in the paper only if the method of this study, given
the same previous seven days, is more accurate than the foundation model. Otherwise neither the foundation-model result nor the extra versions
are put in the paper. The results of PROTOCOL_V6, V7 and this protocol stay in this repository whatever the outcome.

The first attempt (PROTOCOL_V7: the strict forecast plus a week of data, Gamma-Poisson shrinkage) failed: the selected weight of the strict forecast
was zero, so it was a plain weekly average (December 2021, MAE / RMSE: Chicago 1.293 / 3.952, NYC 1.831 / 6.024), clearly worse than Chronos-Bolt-small
(1.083 / 2.887 and 1.505 / 4.690, PROTOCOL_V6). The strict forecast is a poor way to use the helper cities once a week of data exists. A better way is
to let the helper cities teach how a region's counts continue from its previous week: a forecaster is trained on the helper cities only, with the same
input as the foundation model (the previous seven days of the region), and is applied to the target without any training on target data. Scale
matching is applied to the helpers as in the strict method. The target data enter only as the input of the forecast, as for Chronos-Bolt.

## Method (fixed now; `xc7d.py`)

- Setting as PROTOCOL_V6: Chicago (77 community areas) and New York City (263 zones), 30-minute slots, context = the previous 336 slots of the
  region, horizon 6 slots, point forecast = the mean of the predictive distribution (the quantile mean for Chronos-Bolt, as in PROTOCOL_V6).
- Training data: helper cities only. Chicago <- New York City and San Francisco; New York City <- Chicago and San Francisco (as the final method).
  Scale-matched condition: partition r of the re-aggregated helpers of `data/agg` (the blocks of the strict method); native condition: the original
  regions. The 2021 helpers are used only for the days before the first day of the evaluation period (development: before 1 October 2021; test:
  before 1 December 2021); San Francisco (2023) is used for the whole year. Training origins: every 12 slots with alternating offsets, from day 8; every
  12th day (day of the year mod 12 = 5) is held out for early stopping on helper data.
- Features of region r, forecast slot t = t0 + j (j = 0..5), computed from counts before t0 only (`xc7d.py leak` verified that replacing all counts
  from t0 on changes no feature, offset or label-free quantity): the mean of the same slot on the days of the same type (weekday / weekend) among the
  previous 7 days and its three-slot smooth; the counts of the same slot one week and one day before; the mean count of the week; the separable estimate
  (week mean x city profile of the slot); the number of same-type days; the city-level ratios of the last 6 and 48 slots to their weekly-average
  expectation; the region-level ratios of the last 6 and 48 slots to theirs; the last count; the counts of the last 6 and 12 slots; j; sine and cosine
  of the slot of the day and of the weekday; weekend; federal holiday, the day before and the day after; and, unless stated, the 18 map features
  of the region, scaled by the median and interquartile range pooled over the helper regions (as the XGBoost baseline of the paper).
- Model: XGBoost, Poisson objective, on top of an offset log(lambda0) with lambda0 = rho6 x (n x weekly average + 1 x separable estimate) / (n + 1), where
  rho6 is the city-level ratio of the last 6 slots to their weekly-average expectation clipped to [0.5, 2] (the level of today of PROTOCOL_V7), floor
  0.002. Fixed: learning rate 0.05, subsample 0.8, column subsample 0.8, min_child_weight 20, lambda 1, 128 bins, at most 600 trees, early stopping 30 rounds
  on the held-out helper days.
- Ensemble: the mean forecasts of the runs on the partitions r = 1, 2 (development) and r = 1..10 (test), model seed r - 1.
- Candidates, fixed now: C1 depth 6 with the map features; C2 depth 6 without the map features; C3 depth 4 with the map features; C4 depth 8 with the map features.
- Code checks done before this protocol, without any accuracy of the new model on the target: the leak test above; the weekly average of the code reproduces
  PROTOCOL_V7 on October and November (MAE 1.400 Chicago, 1.955 NYC; RMSE 4.327, 6.367); one model trained on helper data only (Chicago-target helpers,
  partition 1, before 1 October) lowered the MAE on the held-out helper days from 2.098 (offset only) to 1.776.

## Development (October and November 2021; Chicago and New York City)

- Chronos-Bolt-small is run on this period with the model, context, horizon and point forecast of PROTOCOL_V6 (`fm_chronos_dev.py`), on origins every 12 slots
  with alternating offsets (even days 0, 12, 24, 36; odd days 6, 18, 30, 42), and the candidates are scored on the same slots.
- Choice: the candidate with the smallest mean over the two cities of MAE(candidate) / MAE(Chronos-Bolt); ties to the smaller index. Go rule: the chosen
  candidate has a lower MAE than Chronos-Bolt in both cities on this period. If not, December is not run under this protocol, and the result is recorded.
- C1 with native helpers is also run, for information, and is not used for the choice.
- Changes of the method after seeing these results are allowed only under a dated addendum written before the changed runs; all development results are
  reported; December is not touched during development.

## Test (December 2021, once, with the chosen candidate)

- Origins every 6 slots, horizon 6, all slots of December (each slot once), exactly the slots of PROTOCOL_V6; ten runs on the scale-matched partitions and ten
  runs (seeds) on the native helpers; mean-forecast MAE and RMSE; Chronos-Bolt-small forecasts are those saved by PROTOCOL_V6 (`out_cal/fm_chunks`).
- Hypotheses, fixed now. H1 (primary): the scale-matched forecaster has a lower MAE and a lower RMSE than Chronos-Bolt-small in both cities, with the 95%
  interval of the mean daily MAE difference (7-day block bootstrap, 2,000 draws) below zero in both cities. H2: scale-matched helpers give a lower MAE than
  native helpers in both cities (interval excludes zero). H3, information only: the Poisson-median forecast against the median forecast of Chronos-Bolt
  (1.055 and 1.481).
- Reading rule for the paper: if H1 holds, the foundation-model comparison goes into the table with published results, with the label "uses the previous
  seven days of the region", with both rows. If H1 does not hold, nothing of PROTOCOL_V6, V7 and V8 goes into the paper (decision of the first author above).

## Disclosure

- The December forecasts of Chronos-Bolt (PROTOCOL_V6) are known; no October or November forecast of Chronos-Bolt has been looked at. The idea was formed after
  PROTOCOL_V7 failed. The held-out cities (San Francisco, Washington, DC, the coarse versions, Austin) are not touched; if the extra is carried to them, it
  will be under a new addendum, once, with the frozen candidate.
- The Poisson-median MAE is information only; the primary comparison uses the mean forecast of both methods, as in PROTOCOL_V6.
- Other foundation models (larger Chronos-Bolt models, TimesFM, Moirai, OpenCity) are not part of this protocol.
