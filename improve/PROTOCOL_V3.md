# Region-size summary (mean rule) and the block-size sweep for Washington, DC (frozen BEFORE any partition is built and any run)

Written: 2026-10-21, before the helper blocks of the conditions below were built and before any model was trained or scored with them.

## Questions

1. A reviewer-style request (Dr Hasan, 2026-10): extend the block-size sweep of the development cities to a held-out city. Washington, DC
   (tracts, median 0.519 km2) is the only held-out city with a helper that can be merged (San Francisco, 0.353 km2).
2. Is the median region area the best summary of the target's scale? An external suggestion to match the whole distribution of areas was
   checked with development-only diagnostics (`results/explore_b1.txt`: the mismatch between the area quantiles of a partition and those of the
   target did not predict the quality of the run trained on that partition, Spearman -0.19 for Chicago and -0.37 for NYC, n = 10, and the
   choice of the best 10 of many partitions would reduce the partition diversity to which the paper attributes the ensemble gain;
   `results/explore_scale_stats.txt`: the geometric mean area is within 0.89 to 1.17 times the median for every target, inside the flat part
   of the sweep). The same diagnostics showed one thing worth testing, below.

## Disclosure (before this protocol)

The ratio of the arithmetic mean to the median area of the regions is 1.05 for Chicago and 1.41 for New York City, and the best block sizes of
the paper's sweeps were q = 1.0 (Chicago) and q = 1.5 (New York City). The mean area is the total area divided by the number of regions, so
matching it means matching the number of regions per km2. The hypothesis that the MEAN is the better summary of scale was formed after seeing
both facts and is therefore post hoc for Chicago and New York City. It has not been tested on any other case. Washington, DC tracts (mean/median
1.656) and the ZIP-code areas of San Francisco (mean/median 2.882) are the first tests; their median-rule results (the paper's final method)
are known, their mean-rule and sweep results are not. No other summary statistic (trimmed mean, quantile signature, area-weighted mean,
compactness) will be tried after the results are seen.

## Definitions (computed from public polygons only)

- Median rule (the frozen method of the paper): goal block area = median area of the target regions, q = 1.
- Mean rule: goal block area = sum of the target region areas / number of target regions. Washington, DC tracts: 0.859 km2 (q = 1.656).
  San Francisco ZIP-code areas: 13.557 km2 (q = 2.882).
- In both rules a share helper whose native median area is below the goal is re-aggregated by area-weighted k-means with K chosen as in
  `aggregate.py` (median block area closest to the goal on the logarithmic scale), partitions r = 1..10 with random_state r; a helper whose
  native median area is at least the goal stays native. Regions are never split.

## Conditions and runs (final method: heads eh and eo, `run_x.py` hyper-parameters unchanged, calendar-slot mode, final = equal average of heads)

- Washington, DC (share helpers Chicago, NYC, San Francisco; scored hourly): sweep q in {0.75, 1.5, 2, 4} and the mean rule (q = 1.656); San
  Francisco is re-aggregated, Chicago (7.39 km2) and NYC (2.14 km2) stay native at every goal up to q = 4 (goal 2.08 km2). q = 0.25 and
  0.5 give goals (0.13 and 0.26 km2) below San Francisco's native median (0.353 km2), so they equal native helpers, which exist already.
  Run r uses partition r with seed r - 1, r = 1..5, heads eh and eo, so 10 runs per new condition (50 runs). The comparators q = 1 (median
  rule) and native helpers use the existing runs r = 1..5 (seeds 0..4), so that every condition is a 5-run-per-head ensemble.
- San Francisco ZIP-code areas (share helpers Chicago and NYC, both re-aggregated to 13.557 km2): the mean rule, r = 1..10, heads eh and eo
  (20 runs), against the median rule runs that exist (r = 1..10).
- Run order by priority, so that partial results are usable: SF ZIP-code areas mean rule, DC mean rule, DC q = 1.5, DC q = 2, DC q = 0.75, DC q = 4.

## Scoring

Fully strict trip MAE and RMSE, hourly for Washington, DC, scored only by the evaluation script `scale_rule_eval.py`; mean daily MAE
difference to the median rule with a 7-day block bootstrap (2,000 draws) 95% interval, as in the sweep of the paper.

## Predictions and reading rule (fixed now)

- H1: at Washington, DC tracts the mean rule has a lower MAE than the median rule, with an interval that excludes zero.
- H2: at the San Francisco ZIP-code areas the mean rule has a lower MAE than the median rule, with an interval that excludes zero.
- H3: for the DC sweep the lowest MAE lies at a q between 0.5 and 2, as in both development cities, and q = 4 is worse than q = 1.
- The mean rule is called SUPPORTED if H1 and H2 both hold, NOT SUPPORTED if either case shows a higher MAE with an interval that excludes
  zero, and INCONCLUSIVE otherwise. All results are reported as obtained. In every case the headline results of the paper remain those of the
  frozen median rule; a supported mean rule would be reported as a refinement that is closer to the sweep optima, with its post hoc origin
  stated, and a recommendation to adopt it in future work, not as a replacement of the reported results.

## Compute

About 70 runs; a Washington, DC run took 5.3 minutes on average when two jobs shared the laptop (mean over the 40 existing runs), so the
whole experiment needs about 3 hours with two runs in parallel.

## Not done in this round

Scale-conditioned allocator, constrained downscaling, origin-destination gravity features, hierarchical allocation and the level diagnostic
of the same external list are not run; the reasons are in the assessment recorded in the repository history of 2026-10-21.

## Result (2026-10-21, `results/scale_rule_eval.txt`, `results/dc_sweep_table.txt`; recorded here as obtained)

All 70 runs finished as specified (block counts and median block areas in `results/scale_rules_build.txt`; DC mean rule: goal 0.859 km2, 124 blocks).
- H1 (DC, mean rule better than median rule, interval excludes zero): FALSE. Hourly MAE 0.8556 against 0.8555 for the median rule, mean daily
  difference -0.0002 [-0.0010, +0.0005]; the RMSE is lower, 2.639 against 2.673.
- H2 (San Francisco ZIP-code areas, same test): FALSE, and in the other direction. MAE 3.556 against 3.443 for the median rule, a difference of
  -0.114 [-0.134, -0.092] (3.3% worse), better on 12% of days; RMSE 6.116 against 6.262.
- Reading rule: the mean rule is NOT SUPPORTED. The median rule remains the frozen method; no further summary statistic is tried.
- H3 (DC sweep): TRUE. Lowest MAE at q = 1 (0.855); q = 0.75, 1.5, 2 and 4 are higher by 0.003, 0.002, 0.004 and 0.006 (intervals exclude zero),
  native helpers by 0.017; the RMSE is slightly lower for the larger blocks (2.63 at q = 4 against 2.67 at q = 1).
The block-size sweep for Washington, DC (H3 and the table of `dc_sweep_table.py`) is reported in the paper as the extension of the sweep that was
requested; the mean-rule hypothesis (H1, H2) is a negative result that is kept in this repository record only.
