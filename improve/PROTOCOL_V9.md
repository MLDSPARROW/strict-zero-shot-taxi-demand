# Bike-share cities as extra teachers of the allocation (frozen BEFORE any allocator is trained with them)

Written: 2026-10-24, before any allocator run with bike-share data. Committed and time-stamped (`PROTOCOL_V9.timestamp`) with the plan and the scripts.

## Why

Idea of the first author: use more cities, but not their taxi demand, only other urban data, to learn the structure of urban demand, and then use the
trained model to estimate taxi demand in a target city with no taxi data. The test is about the strictly zero-shot estimation of demand only; no
data of the target city enter the pipeline, and nothing with target history is involved. Of the possible data types, trips of bike-share systems are the
closest to taxi pickups that is public for many cities (origins per region and time), so they are used as extra helper cities of the allocation (step 3 of the
method). Speed data and maps alone were not chosen: speeds carry no information about where trips start, and map-only pre-training (PROTOCOL_V4, Way 1) hurt Chicago.

Prior evidence, stated before the runs: the taxi cities already used as helpers help; extra helpers of another kind did not help before (Washington, DC as a
share helper degraded Chicago, and the ride-hailing cells of Austin were screened as a helper and not used). No gain is predicted; the rule below decides.

## Data (fixed now, built by `multicity/prep_bike.py`; map data and bike-share trip origins only)

- Five systems, trips of 2021 (start time, start coordinates): Boston (Bluebikes), Philadelphia (Indego), Los Angeles (Metro Bike Share), Minneapolis (Nice Ride; the
  system runs from April to November) and Columbus (CoGo). Public files of the operators: `s3.amazonaws.com/hubway-data`, `rideindego.com`, `bikeshare.metro.net`,
  `s3.amazonaws.com/niceride-data`, `s3.amazonaws.com/cogo-sys-data`; 148 MB in total. Portland (BIKETOWN) was dropped before any run because its public files end in
  August 2020. The bike-share systems of the development targets (Divvy, Citi Bike) and of the held-out targets (Capital Bikeshare, Bay Wheels, MetroBike) are not used.
  No city is added or dropped after results.
- Regions: the 2021 cartographic-boundary census tracts that contain at least one trip origin of 2021 and whose centroid lies within 12 km of the geocoded downtown
  point of the city. Counts per day and 30-minute slot of the trip origins. Map features exactly as for the taxi cities (OpenStreetMap places on a 40 x 50 grid, area-weighted
  to the tracts; the 19 columns of `bench77/region_features.py`; the airport input is 0).
- Scale matching exactly as for the taxi helpers (`build_bike_partitions.py`): K from k-means with random state 0 so that the median block area is closest to the median
  region area of the target; ten partitions with the random states 1 to 10; a city whose median tract area is already larger than the target's is left native.
- Code checks before this protocol, without any result of the allocator with bike data: the feature formula of the re-aggregation reproduces the stored tract features
  (asserted in `build_bike_partitions.py`); `run_x.py` without the new switch reproduces a stored run of the final method (`results/v9_repro_check.log`).

## Method (fixed now; switch k of `run_x.py`; everything else is the final method, heads eh and eo)

- Helpers: Chicago <- nyc_for_chicago_p_r, sf_for_chicago_p_r and the five bike cities re-aggregated for Chicago (partition r); NYC <- chicago, sf_for_nyc_p_r and the five bike
  cities re-aggregated for NYC (partition r). Run r uses partition r and model seed r - 1 (`v9_common.py`).
- The changes made by the switch, and nothing else: (1) bike-share slots with fewer than 10 trips in the city are not used; (2) an input column tells the type of the helper
  (1 for bike-share, 0 for taxi helpers and for the target); (3) in every training step the type is drawn with probability 1/2 (taxi or bike-share), then a helper of that type
  uniformly, so that the taxi helpers keep half of the steps; (4) early stopping uses the held-out days of the taxi helpers only; (5) the feature scaling (median and
  inter-quartile range) is pooled over all helper regions, as before, now including the bike-share regions.
- The target's own demand is used only by `v9_eval.py` for scoring (demand-reading mode, as in the screen of PROTOCOL_V4).

## Rules (fixed now; the rules of PROTOCOL_V4)

- Comparator v1: the runs r = 1..10 of the final method (heads eh and eo) in `out/`.
- Screen (`runner_v9.py`, runs r = 1..3, `python v9_eval.py`): k passes if the MAE_share of its 3-run ensemble (equal average of the heads ehk and eok) is at least 3% better than
  v1 in one development target (Chicago, NYC) and not more than 3% worse in the other.
- Confirmation (runs r = 4..10, `python v9_eval.py --confirm`): k is adopted if (a) its mean MAE_share over the two targets is lower than v1's, (b) the 7-day block bootstrap
  (2,000 draws) interval of the daily MAE_share difference v1 - k lies above zero in at least one target, and (c) in the other target it is not more than 3% worse and its
  interval does not lie entirely below zero.
- Held-out evaluation, only if k is adopted: strict calendar-slot predictions, ten runs per head, for San Francisco, Washington, DC, the two coarse versions and Austin, with the
  bike-share cities re-aggregated for each target, scored like the other held-out runs; the plan is written as an addendum before any such run. Whatever the outcome, it is reported.
- If k is not adopted, v1 stays the final method, the result is recorded in this file, nothing is added to the paper, and no variant is tried without a new time-stamped protocol.

## Disclosure

- The idea, the choice of bike-share as the data type, the five cities (all that could be downloaded for 2021 outside the targets), the minimum of 10 trips, the type probability of
  1/2 and the 12 km radius were fixed here without looking at any result of the allocator with bike data. The development targets were used in the design of the method, so
  a pass on them is not a held-out result.
- The first author's wish was to compare only with strict methods; this test does not use any target data at any stage.

## Data as built (2026-10-24, before any allocator run with bike-share data)

| City | Tracts | Trips 2021 | Slots with >= 10 trips | Median tract area, km2 | Blocks for Chicago (7.39 km2) | Blocks for NYC (2.14 km2) |
|---|---|---|---|---|---|---|
| Boston | 218 | 2,928,839 | 15,601 | 0.46 | 20 | 64 |
| Philadelphia | 94 | 851,828 | 13,471 | 0.39 | 8 | 23 |
| Los Angeles | 82 | 118,280 | 5,341 | 0.34 | 6 | 19 |
| Minneapolis | 142 | 267,653 | 6,935 | 1.17 | 29 | 90 |
| Columbus | 78 | 43,523 | 1,004 | 1.59 | 23 | 63 |

All five cities are used (the rule above: no city is added or dropped after results; Columbus has few usable slots and is kept). The median block areas over the ten
partitions are within 6% of the target's in every case (6.84 to 7.38 km2 for Chicago, 2.01 to 2.19 km2 for NYC). The map features are finite, the three missing-data flags are 0,
and the grid cover of every tract is 1.00. Reproduction check: `run_x.py` without the switch reproduces the stored run of the final method (Chicago, r = 1, head eh) exactly
(largest absolute share difference 0.0; MAE 1.776 and 19 epochs). Smoke tests of the switch (1 epoch of 5 steps, outputs deleted) ran on both targets.
For reference, the comparator v1 has MAE_share 0.00937 (Chicago) and 0.00460 (NYC) for runs r = 1..3 (`python v9_eval.py`).
The Way 2 held-out test (PROTOCOL_V4, addendum 2) was still running when this was written; the runs of this protocol start after it has finished, so that the two do not compete
for the laptop. Compute: the screen is 12 runs of about 5 to 8 minutes, on two workers.
