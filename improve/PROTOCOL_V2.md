# Version 2 of the method: loss-consistent point forecasts and a people-present allocator (frozen BEFORE any v2 run)

Written: 2026-10-20, before any v2 model was trained, before any v2 prediction was scored, and before the existing
predictions were re-scored with the rule in I1. Committed and time-stamped (`PROTOCOL_V2.timestamp`) before any of
these steps.

## Why

v1 (the paper's final method) is the best published result for Chicago (MAE 1.83) but not better than simple public-data
baselines for New York City, San Francisco, Washington, DC or Austin. Three measured facts, all from files already in
`results/` or from read-only checks listed under "Disclosure", motivate the three changes below:

1. The error in NYC, SF and DC is almost entirely spatial allocation, not the city total (`results/diag_*.txt`: NYC MAE
   7.45 with predicted totals, 7.45 with true totals, 2.31 with a perfect allocation).
2. A forecast of zero for every region and slot has a lower MAE than every method in NYC (6.63), SF (0.40), DC (0.66,
   hourly) and Austin (0.110). MAE is minimised by the median of the predictive distribution, and most region-slots
   are usually zero, while every method, including all baselines, reports the mean.
3. The learned allocator transfers the concentration of its helpers: at the coarse targets its RMSE is 6.26 (sf_zcta)
   and 8.00 (dc_cluster) against 3.93 and 6.00 for the best simple baselines. Its inputs describe activity (places,
   jobs, residents, rail, airports) but not who uses taxis, and its area offset (head eo) moves mass to large outer
   zones in NYC (eo 8.17 against eh 6.96, 10-run results in the paper).

## I1. Loss-consistent point forecasts (all methods, no retraining)

- MAE is consistent for the median and RMSE for the mean (Gneiting, 2011). From v2 on, every method, ours and every
  baseline alike, is scored with MAE on the median of a negative binomial predictive distribution NB(lambda, k(lambda))
  and with RMSE on its mean lambda, where lambda is the method's predicted trips per region and slot exactly as in v1.
  The mean-forecast MAE of v1 is reported alongside, so nothing is replaced silently.
- k(lambda) is estimated on the share helpers of the target only (condition M, partition 1), never on the target:
  for each helper, lambda_hat[d, s, r] = total[d, s] x m[weekday(d), s, r], where m is the helper's mean share of region
  r in slot s on that weekday; slots with a zero total are skipped. Cells of all helpers are pooled into 40 bins of
  lambda_hat, log-spaced from 0.001 to the maximum; in every bin with at least 1,000 cells, k = mean(lambda_hat)^2 /
  (mean((Y - lambda_hat)^2) - mean(lambda_hat)), and k = 10^6 (Poisson) when the excess variance is not positive.
  log k is interpolated linearly in log lambda between bin centres and held constant beyond the end bins.
- Targets scored hourly (Washington, DC and dc_cluster) use the same estimator on helper data summed to hours, and the
  hourly lambda is the sum of the two half-hour lambdas.
- The median is the 0.5 quantile of the NB distribution (0 when lambda = 0). Also reported: the coverage of the
  central 80% predictive interval (0.1 to 0.9 quantiles) and the MAE of the zero forecast.

## I2. People-present exposure and taxi-propensity covariates (new allocator heads)

- Flag `p` in `run_x.py` (people-present exposure): logits = f(x, t) + log(alpha_t * J_r + (1 - alpha_t) * P_r + 1e-6),
  where J_r and P_r are the region's shares of the city's LODES 2021 jobs and ACS population, alpha_t =
  sigmoid(w . c_t + b), c_t are the nine calendar features already used, and w = 0, b = 0 at initialisation.
  Trips are made by people, so the exposure is the people present (workers by day, residents by night) instead of
  the area used by head eo. The learned log-relative-area input is removed, as for eo.
- Flag `a` (taxi propensity): three region inputs from the ACS 2020-2024 five-year tract tables (Census Reporter API,
  the same release as the level step): the share of households without a vehicle (B25044), the share of households
  with an income of at least $150,000 (B19001) and log1p(population per km2) (B01003). Tract counts are spread to
  regions by tract-area overlap, as for LODES in `build_extra.py`; blocks of re-aggregated helpers and coarse targets
  are exact sums of their member regions, and the membership is reconstructed with the original k-means seeds and
  checked against the stored jobs sums. Shares are smoothed towards the city share with 50 pseudo-households
  (share_r = (n_r + 50 * city share) / (households_r + 50)), so regions without households get the city share.
- New parameters are created only when a flag is used, so all other variants keep their initialisation.
- Variants screened: `ep` and `epa` against `eo`, and `eha` against `eh`. The baselines `eo` and `eh` are re-run under
  identical conditions.

## I3. Worst-helper training (only if a variant passes I2)

Group-DRO weights over helper cities (weights proportional to exp(eta x mean loss of the helper), eta = 1, updated
every epoch) and early stopping on the worst helper's validation loss. Screened like I2 on top of the selected heads.

## Screening and selection (development targets Chicago and NYC only)

- Helpers as in the paper's final method: Chicago <- nyc_for_chicago_p_r + sf_for_chicago_p_r; NYC <- chicago +
  sf_for_nyc_p_r. These partitions were lost with the local working data and are rebuilt with `aggregate.py`,
  `aggregate_multi.py` and `aggregate_multi2.py` unchanged; the rebuild must give the block counts of the paper's
  re-aggregation table, `tab:k` (NYC 263 -> 101 and SF 243 -> 21 for Chicago, SF 243 -> 60 for NYC), from input files whose SHA-256
  matches `features/SHA256SUMS.txt`.
- Screen: runs r = 1, 2, 3 (partition r, seed r - 1), demand-reading mode for the development targets only, metric:
  MAE_share of the 3-run ensemble. Rule as pre-declared in `run_x.py`: a variant passes if it is at least 3% better
  in one development target and not more than 3% worse in the other.
- Confirmation: runs r = 1..10 of passing variants and their baselines; a variant is adopted if its 10-run ensemble
  is better in the mean daily MAE_share over both targets and a 7-day block bootstrap (2,000 draws) of the daily
  difference excludes zero in at least one target and is not significantly worse in the other.
- If both `ep` and `epa` are adopted, the one with the lower mean MAE_share over the two targets replaces `eo`; if
  `eha` is adopted it replaces `eh`. The final v2 prediction is the equal average of the two selected heads (the v1
  rule). If nothing is adopted, v2 = v1 point predictions with the I1 scoring only, and this is reported.

## Evaluation of v2 (run once, after selection is frozen and committed)

- Fully strict calendar predictions for Chicago, NYC, SF, DC, sf_zcta, dc_cluster and Austin, ten runs per head, with
  the same helpers, partitions, seeds, city totals and scoring code as v1. v1 predictions lost with the local data
  (Chicago, NYC, SF) are recomputed with the v1 code to score them under I1; their mean-forecast MAE must reproduce
  `results/eval_*.txt` / the paper within 0.02, otherwise this is reported.
- Reported for every target: v1 and v2, mean-forecast MAE, median-forecast MAE, RMSE, 80% interval coverage, the zero
  forecast and all baselines (uniform, jobs, jobs + residents, Chi et al.), with the block bootstrap of v2 against v1
  and against the best baseline. SF and DC are reported as a second look at held-out cities, because their v1 results
  were known when v2 was designed. Porto, never used as a target, is evaluated with v2 without flag `a` (no ACS), and
  this is stated.
- Success criteria, fixed now: NYC MAE below 6.63 (zero forecast) and 6.83 (UrbanGPT); NYC RMSE below 17.51
  (jobs-proportional) with an interval excluding zero; SF MAE at most 0.40 and RMSE below 0.98; DC hourly MAE below
  0.66 and RMSE below 2.40; Chicago no more than 3% worse than 1.83 / 5.21. Every number is reported whether or not
  the criteria are met.

## Disclosure (read-only checks run on 2026-10-19 and 2026-10-20, before this protocol)

1. Poisson-median forecasts were computed on the existing v1 and baseline predictions for DC (native), dc_cluster and
   sf_zcta. They lowered every method's MAE (DC: v1 0.856 -> 0.757, jobs + residents 0.804 -> 0.689) but the zero
   forecast (0.662) stayed lowest. The choice of a negative binomial instead of a Poisson predictive distribution in I1
   was made after this check.
2. ACS tract covariates (car-free households, households >= $150k, taxi commuters, population) were compared with the
   true time-averaged shares of Chicago and NYC only (development targets). No SF, DC or Austin demand was used to
   design I2.
3. The ten busiest regions of Chicago and NYC were inspected (development targets only).
4. A one-week query showed that only 43% of Chicago's 2021 taxi trips carry a pickup census tract, so a fine-scale
   Chicago helper was not pursued.
