# Hub-and-city ensemble of the final method and gradient boosting (frozen BEFORE any held-out run)

Written: 2026-10-24, after a descriptive analysis of the development targets Chicago and NYC only (listed under "Disclosure") and before any
XGBoost model was trained for, or any prediction was made of, a held-out target. Committed and time-stamped (`PROTOCOL_V10.timestamp`) before
those steps.

## Why (development targets only)

1. Every allocation change tested since PROTOCOL_V2 that helps NYC hurts Chicago (people exposure, ACS inputs, bike-share teachers, Way 2 with
   bike-share). Of 231 fixed mixes of the three trained heads eh, eo and ep, the best gains at most 0.7% on the weakest of the four numbers
   (MAE and RMSE in both cities; `results/v11_pareto.txt`).
2. The reason, measured in `results/v11_law2.txt` and `results/v11_xgb_compare.txt`: NYC yellow-taxi trips follow LODES jobs with an
   elasticity of 1.80 (airports left out), against 1.01 for Chicago and 0.82 for SF at the NYC scale, while 18.0% of Chicago's trips start at
   its airports against 5.6% in NYC. Each development target differs from its helpers in a way the helpers cannot teach.
3. The scale-matched XGBoost baseline (PROTOCOL_XGB_SCALE.md, configuration unchanged) is the most accurate allocation for NYC (6.29 / 16.95)
   but has no airport mechanism: O'Hare receives 0.6% of Chicago's trips (true 14.6%) and JFK 0.07% of NYC's (true 3.4%). Outside the
   airports its yearly split of NYC is closer to the truth than that of v1 (total-variation distance 0.42 against 0.53).
4. Two model families with different inductive biases make partly different errors, and their average has a lower error than the average
   of their errors (ambiguity decomposition, Krogh and Vedelsby, 1995). The factorisation below takes the airport mass from the model that
   has an airport mechanism (the final method, whose airport share is 15.1% in Chicago and 5.9% in NYC against true 18.0% and 5.6%), and
   averages the two families for the split of the rest of the city.

## Method (fixed now): hub-and-city ensemble HC

For target T and slot t, A is the set of airport regions of T (enplanements > 0 in `data/enpl_<T>.npy`: O'Hare and Midway in Chicago, JFK and
LaGuardia in NYC, AUS in Austin; none in SF, DC, sf_zcta and dc_cluster).
- v1[t] = the shares of the final method, unchanged (equal average of the eh and eo ten-run ensembles).
- g[t] = the shares of the XGBoost baseline of the paper, unchanged (18 map and 9 calendar features, Tweedie loss, 400 trees, depth 6,
  learning rate 0.05, subsample 0.8, column sample 0.8, 800 random slots per helper, median/IQR scaling pooled over the helper regions),
  trained on the same helpers as the final method for T; run r uses partition r and seeds r - 1 (SF: native helpers, seeds 0-9); ten-run
  ensemble.
- HC[t, A] = v1[t, A]; HC[t, i] for i not in A = (1 - sum over A of v1[t]) x (1/2 x v1[t, i] / sum over non-A of v1[t] + 1/2 x g[t, i] /
  sum over non-A of g[t]). If A is empty, HC = (v1 + g) / 2.
- Trips = HC x the strict city total per slot of the paper (level x rhythm, unchanged). No target taxi data enter any step.
- The weights 1/2 and 1/2 are the paper's equal-average rule and are not tuned. Disclosed: on the development grid
  (`results/v11_hybrid.txt`) an XGBoost weight of 0.2 in the city split beats v1 on all four development numbers (Chicago 1.826 / 4.972,
  NYC 7.046 / 17.011), but it was read off the development targets and is not used.

## Development result (designed on these numbers; `v10_dev.py`, `results/v10_dev.txt`)

Chicago: v1 1.829 / 5.207, HC 1.865 / 4.837 (MAE 1.97% worse, daily MAE difference v1 - HC -0.0360 [-0.0569, -0.0172], so significantly worse
in MAE; RMSE 7.10% better). NYC: v1 7.454 / 17.590, HC 6.557 / 16.464 (MAE 12.03% better, +0.8970 [+0.8217, +0.9823], better on 100% of
days; RMSE 6.40% better). Fresh development runs (the confirmation step of earlier protocols) are not available, because the design used all
ten runs; the held-out evaluation is the only independent test.

## Held-out evaluation (run once, after this protocol is committed)

- Targets: the five held-out targets of the paper, SF, DC (scored hourly), sf_zcta, dc_cluster (scored hourly) and Austin (248 days), with the
  v1 predictions already on disk (`out_cal/`, as in `v4h_eval.py`, which reproduces the paper) and the strict city totals of `coarse_eval.py`
  and `austin_eval.py`. Porto is not used.
- `xgb_heldout.py` trains the XGBoost runs (helper data only, the target demand is never read); `v10_heldout.py` scores v1, XGBoost alone, HC,
  jobs-proportional and jobs + residents, with the 7-day block bootstrap (2,000 draws) of the daily MAE difference v1 - HC.
- Adoption rule: HC replaces v1 as the final method if (a) the mean over the five held-out targets of the relative MAE change (v1 - HC) / v1
  is positive, (b) the same holds for RMSE, and (c) HC is better than v1 in MAE in at least three of the five targets and in RMSE in at least
  three of the five. Otherwise v1 stays. Every number is reported whether or not the rule holds.
- If HC is adopted, the paper is updated in one pass: HC as the final method, v1 and the scale-matched XGBoost as its components and as
  comparison rows on every target, and the Chicago and NYC numbers labelled as development results.

## Disclosure (development targets only, 2026-10-24, before this protocol)

Descriptive scripts and their outputs: `v11_diag.py` (pooling of heads; arithmetic and geometric averages; region-level errors),
`v11_law.py` and `v11_law2.py` (cross-sectional laws of yearly shares against public covariates), `v11_envelope.py` (tilting the shares of
v1 towards the range of the helper laws: NYC MAE -1.0%, RMSE +0.2%, not pursued), `v11_pareto.py` (231 mixes of eh, eo, ep), `v11_hybrid.py`
(286 mixes with XGBoost, plain and hub-and-city), `v11_xgb_compare.py`, and `xgb_save.py` (a re-run of the published scale-matched XGBoost
baseline that saves its predictions; it reproduces 2.45 / 6.99 and 6.29 / 16.95 exactly). A level check showed that pooling the public units
of the level step worsens both cities, so the level step is unchanged. No SF, DC, sf_zcta, dc_cluster or Austin number was read for any of
these, and XGBoost has never been run on a held-out target. Since PROTOCOL_V4, eight allocation variants have been tried on the same two
development targets (m, d, md, k, dk, q, t and HC); the held-out evaluation decides.
