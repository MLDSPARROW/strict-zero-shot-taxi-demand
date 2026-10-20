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

## Addendum 1 (2026-10-20, after building the ACS covariates, before any v2 run or re-score)

The verification step of `build_acs.py` (population in regions within 2% of the ACS city total) failed for NYC: -4.0%.
The loss is concentrated in Manhattan (240,000 of 340,000 people): the generalised cartographic tract outlines
(`cb_2021_36_tract_500k`) leave waterfront tracts up to 74% outside every taxi zone, so plain tract-area weights drop
their counts. The same method in `build_extra.py` drops 5.6% of NYC's LODES jobs and 3.9% of its resident workers in the
v1 files (SF and DC lose at most 0.2%; Chicago's regions cover only part of Cook County, so its gap is expected).
Rule: where the regions cover the listed counties completely (NYC, SF, DC), each tract's overlap weights are normalised
to sum to 1; elsewhere (Chicago, Austin) plain area weights are kept, and the re-interpolated LODES values must equal the
v1 files there. This applies to the ACS counts and to LODES jobs and resident workers re-interpolated for v2 (`jobs_n`,
`res_n`), which are used by flag `p` and by the v2 rows of the jobs and jobs + residents baselines. The v1 files and
v1 rows are not changed, and both versions of the baselines are reported for NYC.

## Addendum 2 (2026-10-20, after the I2 screen, before confirmation runs 4-10 and before any held-out v2 run)

Screen result (`results/v2_screen.txt`, runs 1-3, MAE_share of 3-run ensembles): ep vs eo Chicago -32.4%, NYC +18.0%;
epa vs eo Chicago -26.6%, NYC +14.6%; eha vs eh Chicago -18.6%, NYC -13.6%. All three fail the pre-declared rule and are
not adopted as replacement heads. The ACS propensity inputs (flag `a`) are dropped. I3 is not run (it was tied to a
passing variant).

Exploration after the screen, development data only (no held-out target data): (1) on every helper data set and scale
(NYC and SF blocks at Chicago scale, Chicago, SF blocks at NYC scale, NYC, SF, SF blocks at DC scale), the yearly shares
are much closer to the people prior (LODES jobs + ACS population) than to the area prior (L1 0.47-1.30 vs 1.26-1.77; the
best geometric mix is 100% people in every case); the people head fails in Chicago because the learned correction on top
of it overshoots the busiest areas (peak ratio 1.34). (2) Six combinations of the screened heads were compared on the
development targets (runs 1-3, change in MAE_share vs v1 = (eh + eo)/2, Chicago / NYC): (eh + ep)/2 -7.3% / +10.8%;
(eh + eo + ep)/3 -1.8% / +4.7%; eh/2 + eo/4 + ep/4 +0.3% / +5.9%; v1 shrunk 25% to the people prior -1.2% / -1.8%;
people prior alone -40.9% / -11.6%.

New rule, declared now: the v2 final share is eh/2 + eo/4 + ep/4, that is, v1's equal weighting of the hub head and the
exposure head, with the exposure head replaced by an equal average of the area-exposure (eo) and people-exposure (ep)
heads, treating the two exposure models as equally plausible. It was chosen after seeing the runs 1-3 numbers above, and
this is disclosed. Confirmation on the development targets: runs r = 1..10 of eh, eo and ep (runs 4-10 are new). v2 is
adopted if (a) its 10-run MAE_share averaged over Chicago and NYC is lower than that of v1, (b) the 7-day block bootstrap
(2,000 draws) of the daily MAE_share difference v1 - v2 excludes zero on the positive side in at least one development
target, and (c) in the other target v2 is not more than 3% worse and its interval does not lie entirely on the worse side.
If adopted, v2 is evaluated once on all targets with the calendar-only predictions, as specified above, and scored with I1.

## Addendum 3 (2026-10-20, after the 10-run confirmation, before any further run)

Result (`results/v2_confirm.txt`, 10-run ensembles on the development targets, MAE_share):
Chicago v1 0.00953, v2 0.00960 (v2 0.7% worse; 7-day block bootstrap of v1 - v2 [-0.000101, -0.000039], so v2 is
significantly worse, better on 46% of days); NYC v1 0.00444, v2 0.00421 (v2 5.2% better, interval [+0.000225, +0.000234],
better on 100% of days). Rule (a) holds, rule (b) holds in NYC, rule (c) fails: in the other target (Chicago) the interval
of v1 - v2 lies entirely on the side where v2 is worse.

DECISION, by the rule fixed in Addendum 2: v2 is NOT adopted and v1 stays the final method. No v2 prediction is run on
SF, DC, the coarse targets or Austin. Per the protocol, the only v2 element that remains is the I1 scoring (median of
the negative binomial predictive distribution for MAE, mean for RMSE), reported next to the v1 mean-forecast numbers for
every target. For this, the v1 calendar-only predictions lost with the local data (Chicago, NYC, SF) are recomputed with
the unchanged v1 code (heads eh and eo, ten runs each, same helpers, partitions and seeds as the paper); their
mean-forecast MAE must reproduce the paper within 0.02, otherwise this is reported. Reported as a finding, not tuned
further: the people-exposure head alone is the best head in NYC (MAE_share 0.00407 vs 0.00444 for v1) and the worst in
Chicago (0.01197 vs 0.00953), so its effect depends on the city.

## Addendum 4 (2026-10-21, result of the I1 scoring of the final v1 method, `results/v1_i1_eval.txt`)

Reproduction: the v1 predictions recomputed with the unchanged code match the archived final method within 0.005 MAE on every
target (Chicago 1.829 / 5.207, NYC 7.454 / 17.590, SF 0.485 / 1.082, DC 0.856 / 2.658, sf_zcta 3.443 / 6.262, dc_cluster
3.181 / 7.998, Austin 0.121 / 1.096), inside the declared 0.02 tolerance.
Success criteria fixed on 2026-10-20 (median-forecast MAE, mean-forecast RMSE): Chicago not more than 3% worse than 1.83 / 5.21:
MET (median-forecast MAE 1.715, the lowest of all methods, better than the zero forecast on 91% of days). NYC MAE below 6.63 and
6.83: NOT MET (7.244; zero forecast 6.627; jobs-proportional 7.300, ours better by 0.055 [0.010, 0.094]); NYC RMSE below 17.51:
NOT MET (17.59). SF MAE at most 0.40 and RMSE below 0.98: NOT MET (0.430 / 1.082; zero forecast 0.400, Chi et al. 0.386 / 0.979).
DC hourly MAE below 0.66 and RMSE below 2.40: NOT MET (0.738 / 2.658; zero 0.662, jobs + residents 0.673 / 2.404).
Austin: all methods tie at 0.110 to 0.111 (median-forecast MAE). Coarse targets: sf_zcta 3.442 and dc_cluster 3.096, behind
Chi et al. (2.348) and jobs + residents (2.508). The median-forecast scoring lowers the MAE of every method but does not change
which method is best in NYC, SF, DC or Austin; it is reported as an additional column, and v1 remains the final method.
