# Coarse versions of the held-out targets (frozen BEFORE any coarse-target data were built or any run)

Written: 2026-10-18, before any taxi demand was aggregated to the coarse geographies below and before any model run.

## Question

The paper's claim is conditional: scale matching helps when the target's regions are coarser than those of the helpers.
San Francisco (median tract 0.35 km2) is finer than every helper, so scale matching cannot act there, and Washington, DC
(0.52 km2) has only one finer helper. Both held-out targets therefore performed at the level of simple baselines. This
test asks whether the gain appears in the SAME two cities, with the SAME trips, when only the target's regions are made
coarser. If it does, the region-size mismatch, not the city, is what drives the gain.

## Coarse geographies (chosen from public boundaries only, no demand data looked at)

Each target tract is assigned to the official coarse unit that it overlaps most, so every coarse region is a union of the
city's own tracts. All trips and all tract features are kept, and region features, trips, jobs, residents, rail stations
and airport passengers of a coarse region are exact sums over its tracts, computed with the same block-feature formula
(`feats` in `aggregate.py`) that is used for re-aggregated helpers.

- San Francisco: 2020 Census ZIP Code Tabulation Areas (Census TIGERweb), clipped to the city. SF's 41 official Analysis
  Neighborhoods were considered first and rejected from geometry alone: their median area (2.14 km2) equals NYC's, so no
  helper would be finer and the test would be void. ZCTAs (28 units inside the city, median 4.17 km2 before tract
  assignment) are coarser than NYC, so NYC can be merged.
- Washington, DC: the 46 official Neighborhood Clusters (Open Data DC), median 3.43 km2 before tract assignment; NYC and
  SF are both finer and can be merged.

Target names: `sf_zcta`, `dc_cluster`. Their median areas after tract assignment are recorded by the build script and
used as the matching target, exactly as for every other target.

## Frozen method (unchanged)

Share helpers by the existing rule (all other taxi cities with 30-minute or finer timestamps; DC is never a helper):
`sf_zcta` <- Chicago, NYC; `dc_cluster` <- Chicago, NYC, SF. Level and rhythm are exactly those of `predict_final.py`
for SF and DC; they do not depend on the target geography, since the coarse regions cover the same tracts and trips.
Heads eh and eo, `run_x.py` hyper-parameters unchanged, final = equal average of the two 10-run ensembles.

## Conditions

| Target | N, native helpers (seeds 0-9) | M, scale-matched helpers (run r = partition r, seed r-1) |
|---|---|---|
| `sf_zcta` | chicago + nyc | chicago + nyc_for_sf_zcta_p{r} (Chicago is coarser, left native) |
| `dc_cluster` | chicago + nyc + sf | chicago + nyc_for_dc_cluster_p{r} + sf_for_dc_cluster_p{r} |
| `dc` (native contrast) | chicago + nyc + sf | chicago + nyc + sf_for_dc_p{r} (NYC is coarser at native scale) |

Native SF needs no runs: no helper is finer, so N and M are identical by construction (gain zero).

Partitions: area-weighted k-means as in `aggregate.py`, K chosen so the median block area is closest to the target's
median area, partition r uses random_state r.

## Scoring and reading rule (fixed now)

Fully strict, full year, trip MAE and RMSE, scored only by the evaluation script. Primary inference: mean daily MAE
difference N minus M with a 7-day block bootstrap (2,000 draws) 95% interval, as in `control_eval.py`. Baselines with
the same level and rhythm (uniform, jobs-proportional, jobs + residents) are reported for each coarse target.

Prediction written now: M has lower MAE than N at both coarse targets, with bootstrap intervals that exclude zero, and the
relative gain (N - M) / N at `dc_cluster` is larger than at native `dc`. Any other outcome, including no gain or a loss,
is reported as obtained. Results are not compared across geographies in absolute terms, since coarser regions carry more
trips each; only within-geography differences and relative gains are compared.

## Disclosure

SF and DC demand were already used once as held-out targets at native scale (PROTOCOL_HELDOUT.md) and as helper data
during development. The coarse versions reuse the same trips, so they are not new cities; they are a controlled test of
the scale mechanism. Austin and Porto (PROTOCOL_MORE_CITIES.md) remain the cities never seen in any role as a target.
