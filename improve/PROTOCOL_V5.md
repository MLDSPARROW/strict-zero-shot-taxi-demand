# Comparisons with other published methods under the conditions of this study (frozen BEFORE any run)

Written: 2026-10-22, before any of the runs and scores below were made. Committed and time-stamped (`PROTOCOL_V5.timestamp`) with the plan.

## Why

Request of the first author: more comparisons with published methods, especially Chi, Xu and Chen (2026), under conditions as similar as
possible to those of this study. Most published models need recent observations of the target or training on the target, so they cannot be
run without target data. This protocol covers the two comparisons that can be made with the data that are already local and no download:

1. The allocation of Chi et al. trained on the scale-matched helpers of this study, to test whether scale matching also helps a published
   method (as the XGBoost check of the paper does for a non-neural allocator).
2. Target-history baselines, to show what simple methods that see the target's own past achieve on exactly the same slots. These are not
   strict and are labelled so; they put the strictly zero-shot numbers in context.

Comparisons that need downloads (time-series foundation models, OpenCity, UniST) are not part of this protocol; they need a separate
protocol after the download is approved by the first author.

## Comparison 1: the allocation of Chi et al. on scale-matched helpers (Chi-M)

- Implementation: `chi_baseline.py` of PROTOCOL_CHI_BASELINE.md (frozen on 2026-10-18), unchanged except that the re-aggregated helper
  units of `data/agg` are registered as in `run_x.py`, so that run r uses partition r of every merged helper with model seed r - 1 (ten runs,
  as the final method). The method itself (ReLU output with bias 1.0, 5,000 full-batch epochs, 8 OpenStreetMap groups and LODES resident
  workers as categories, KL loss on the weekly 48-slot template with the weights 5 and 2) is not changed. Output tag `_M`:
  `out_cal/chi_<target>_M_seed<s>.npy`. The native-helper runs (Chi-N) are the existing ten seeds.
- Targets and helpers: those of the final method where at least one helper is merged: Chicago <- nyc_for_chicago_p_r, sf_for_chicago_p_r;
  NYC <- chicago, sf_for_nyc_p_r; DC <- chicago, nyc, sf_for_dc_p_r; sf_zcta <- chicago, nyc_for_sf_zcta_p_r; dc_cluster <- chicago,
  nyc_for_dc_cluster_p_r, sf_for_dc_cluster_p_r; Austin <- chicago, nyc, sf_for_austin_p_r. San Francisco has no merged helper, so Chi-M
  would equal Chi-N and is not run.
- Scoring (`chi_m_eval.py`): fully strict trip MAE and RMSE with the same city totals, slots and hourly scoring as the final method, Chi-M
  against Chi-N with the 7-day block bootstrap (2,000 draws) of the mean daily MAE difference, and next to the final method (v1) as
  reported in the paper. The target demand is read by the evaluation script only.
- Prediction, fixed now: Chi-M has a lower MAE than Chi-N in the targets where helpers are merged. Reading rule: scale matching is called
  to GENERALISE to the allocation of Chi et al. if Chi-M has a lower MAE than Chi-N with an interval that excludes zero in at least 4 of the 6
  targets and is not significantly higher in any; it is called NOT GENERALISING if Chi-M is significantly higher in at least 3 targets;
  otherwise MIXED. All results are reported as obtained.
- Disclosure: the Chi-N results were known; the idea was formed after seeing that Chi-N has a lower error than the final method at the
  coarse targets (sf_zcta, dc_cluster).

## Comparison 2: target-history baselines (not strict, context only; `history_baselines.py`)

- Last week: the prediction for day d, slot s and region r is the observed count of day d - 7. Historical average: the mean of the observed
  counts of days d - 7, d - 14, d - 21 and d - 28 at the same slot and region (the same weekday). Evaluated on the days from day 28 on of
  each target period, so that both have the same history; the final method (v1: runs 1 to 10, condition M, equal average of eh and eo, with
  the strict city totals, `out_cal`) and the zero forecast are scored on exactly the same days and slots. Washington, DC and its clusters at
  hourly resolution as in the paper; Austin on its 30-minute cells.
- Targets: Chicago, NYC, SF, DC, sf_zcta, dc_cluster, Austin. Metrics: mean-forecast MAE and RMSE, as in the main tables.
- No prediction is made; the numbers are descriptive and reported as obtained. They use the observed demand of the previous weeks of the
  target, so they are not comparable in information with the strictly zero-shot method and are always labelled "uses target history".
