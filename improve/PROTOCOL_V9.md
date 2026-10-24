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

## Addendum 1 (2026-10-24, variant dk: Way 2 together with the bike-share cities; written before any run of dk)

Request of the first author, made while the screen above was running: combine Way 2 (masked demand modelling on the helper cities, switch d of PROTOCOL_V4) with the bike-share
cities, and test again. Way 2 alone, on the taxi helpers only, gave a gain of 1% or less at the held-out targets in both directions (PROTOCOL_V4, to be recorded in its
addendum 3), so the question is whether the extra cities change that.
- Variant dk: the switches d and k together (heads ehdk and eodk), with the helpers, partitions, seeds, bike-share data and all other rules of this protocol. It is screened with
  the same screen rule against v1 (`runner_v9.py <worker> <n> screen dk`, then `python v9_eval.py`), and confirmed with the same rules (a), (b), (c) if it passes. k and dk are
  judged separately against v1; if both are adopted, the one with the lower mean MAE_share is selected. d alone (the existing PROTOCOL_V4 runs) is printed for reference only.
- Code: `runner_v9.py` takes the variant as a fourth argument (default k, unchanged); `v9_eval.py` prints d, k and dk. Smoke tests of ehdk and eodk (1 epoch of 5 steps, outputs deleted)
  ran on both targets.
- Disclosure: this addendum was written after worker 0 of the screen of k had finished its six runs of head ehk (their single-run log lines were visible, none was scored) and before any
  run of dk. The request itself was made before those lines were seen.

## Result of the screen of k (2026-10-24, `results/v9_screen_k.txt`; recorded here as obtained)

Runs r = 1..3, heads ehk and eok, ensembles of 3, scored on the observed slots (MAE_share of the equal average of the two heads; allocation-only trip errors use the true slot totals).
Chicago: v1 0.00937, k 0.01128 (20.3% worse); daily difference v1 - k -0.001906 [-0.002169, -0.001651], k better on 2% of days; allocation-only MAE 1.494 -> 2.128 and RMSE 4.035 -> 6.762;
peak ratio 1.01 -> 0.64 (the predicted shares are flatter than the observed ones). NYC: v1 0.00460, k 0.00423 (8.2% better); difference +0.000378 [+0.000372, +0.000383], k better on 100% of days;
allocation-only MAE 7.735 -> 7.151 and RMSE 17.643 -> 17.105; peak ratio 0.54 -> 0.64. For reference, Way 2 alone (d, PROTOCOL_V4 runs): Chicago 0.00931, NYC 0.00447.
DECISION by the rule fixed above: k FAILS the screen, because it is more than 3% worse in Chicago although it is more than 3% better in NYC. No confirmation and no held-out run of k are made.
Reading: the bike-share cities carry information for the allocation in NYC (the largest single gain in the allocation of any variant so far) and spoil it in Chicago, where the true shares are far
more concentrated than the bike-share shares. The screen of dk (Way 2 together with the bike-share cities, addendum 1) is still running and is judged separately.

## Result of the screen of dk (2026-10-24, `results/v9_screen_k_dk.txt`; recorded here as obtained)

Runs r = 1..3, heads ehdk and eodk. Chicago: v1 0.00937, dk 0.01058 (12.9% worse; difference v1 - dk -0.001205 [-0.001431, -0.000991], dk better on 10% of days; allocation-only MAE 1.494 -> 1.950,
peak ratio 1.01 -> 0.67). NYC: v1 0.00460, dk 0.00430 (6.6% better; +0.000304 [+0.000299, +0.000307], dk better on 100% of days; allocation-only MAE 7.735 -> 7.302, peak ratio 0.54 -> 0.67).
DECISION by the rule fixed above: dk FAILS the screen (more than 3% worse in Chicago). Against k, dk is less bad in Chicago (-12.9% against -20.3%) and gains less in NYC (+6.6% against +8.2%); Way 2 alone
(d) was +0.6% and +2.8% there, so the combination is not better than its parts. Neither k nor dk is confirmed or run at the held-out targets. No variant is tried without a new time-stamped addendum.

## Addendum 2 (2026-10-24, variants q and t; written before any run of them)

After the screens of k and dk, the first author asked for a way to keep the gain in NYC without the loss in Chicago. A diagnosis of the Chicago predictions of k from the existing runs (no new run): the annual
share of the Near North Side falls from 28.2% in v1 (true 24.5%) to 15.7%, that of the Loop from 18.7% (true 14.3%) to 8.5%, and that of O'Hare from 11.6% to 8.2% (true 14.6%), while Lincoln Park rises
from 5.3% (true 2.4%) to 10.2% and the Near West Side from 9.2% to 15.0% (true 9.2%). The annual share error (half the sum of the absolute differences) is 18.2% for v1 and 32.2% for k; in NYC it falls from
55.6% to 51.0%. Bike-share trips are concentrated in residential and leisure areas where taxi trips are not, and the shared output layer mixes the two. Two variants that separate the two types are fixed now:
- q: stage 1 trains on the bike-share helpers only (early stopping on their held-out days, patience 3, at most 12 epochs, best state restored); stage 2 is the training of the final method on the taxi helpers
  only (fresh optimiser, at most 25 epochs, early stopping on the held-out days of the taxi helpers as before). Heads ehq and eoq.
- t: as k (type input, minimum of 10 trips, stopping on the taxi helpers, type probability 1/2), but the output layer of the allocator has a separate copy for the bike-share helpers while the encoder is shared;
  the taxi helpers and the target use the original output layer. Heads eht and eot.
Rules: the screen and confirmation rules of this protocol, each variant judged separately against v1 (`runner_v9.py <worker> 2 screen q` and `... t`, then `python v9_eval.py --cand q,t`); if both are adopted at the
confirmation, the one with the lower mean MAE_share is selected; the held-out evaluation is made only for an adopted variant, under the plan of this protocol. Code: switches q and t of `run_x.py`, inert without them;
smoke tests ran on both targets (1 epoch of 5 steps, outputs deleted); the reproduction check of the unchanged path is repeated after the start of the screen and recorded here.
Disclosure: the two variants are a reaction to the screens of k and dk on the development targets, so a pass on them is a development result and not a held-out one. Since PROTOCOL_V4, seven variants (m, d, md, k, dk, q, t)
have been screened on the same two development targets; the held-out evaluation decides.

## Result of the screen of q and t (2026-10-24, `results/v9_screen_q_t.txt`; recorded here as obtained)

Runs r = 1..3, heads ehq, eoq, eht, eot, ensembles of 3, scored on the observed slots. Stage 1 of q stopped after 6 to 12 epochs (the cap is 12).
q: Chicago v1 0.00937, q 0.01036 (10.5% worse; difference v1 - q -0.000985 [-0.001157, -0.000826], q better on 7% of days; allocation-only MAE 1.494 -> 1.817, RMSE 4.035 -> 5.359; peak ratio 1.01 -> 0.74).
NYC v1 0.00460, q 0.00445 (3.4% better; +0.000157 [+0.000154, +0.000160], q better on 100% of days; allocation-only MAE 7.735 -> 7.570, RMSE 17.643 -> 17.669; peak ratio 0.54 -> 0.52).
t: Chicago t 0.01044 (11.4% worse; -0.001070 [-0.001246, -0.000900], t better on 7% of days; allocation-only MAE 1.842, RMSE 5.453; peak ratio 0.75). NYC t 0.00438 (4.8% better; +0.000221 [+0.000215, +0.000225],
t better on 100% of days; allocation-only MAE 7.438, RMSE 17.450; peak ratio 0.68).
DECISION by the rule fixed above: q and t both FAIL the screen (more than 3% worse in Chicago). No confirmation and no held-out run of q or t are made, and no bike-share variant (k, dk, q, t) is adopted.
Reading: separating the two types does not keep the bike-share shares out of the Chicago allocation (peak ratio 0.74 and 0.75 against 1.01 for v1). q and t are less bad than k in Chicago (-10.5% and -11.4% against
-20.3%) but gain less in NYC than k (+3.4% and +4.8% against +8.2%). The extra data help NYC and hurt Chicago in all four variants.
Fully strict trip errors, MAE / RMSE in trips per region per 30 minutes, 3-run ensembles (descriptive, `results/v10_curve_q_t.txt`): Chicago v1 1.828 / 5.186, q 2.070 / 5.912, t 2.102 / 5.985; NYC v1 7.723 / 17.930,
q 7.604 / 18.016, t 7.470 / 17.764.
Reproduction check 2 (`results/v9_repro_check2.log`): after the code of addendum 2 and during the screen, the stored v1 run of Chicago (r = 1, head eh) was repeated with unchanged flags; largest absolute
difference of the shares 0.0.

Descriptive note on weights (2026-10-24, after the four screens; `v10_curve.py`, `results/v10_curve_k_dk.txt`, `results/v10_curve_q_t.txt`). The first author suggested that the weight of the bike-share model should
depend on the target. From the existing runs (no new run) the mix P = (1 - w) v1 + w V was computed for V = k, dk, q, t. The weight that is best for each development target, read off that target's own demand, is 0.2 to 0.3
in Chicago and 1.0 in NYC for all four variants. This is an oracle: a weight chosen on the demand of the target uses target data and is never reported as a method. Any rule for the weight must be computable for a city
without its taxi demand and is fixed in a new addendum before any run. For orientation, the equal mix (w = 1/2) gives MAE / RMSE in trips (Chicago; NYC): k 1.908 / 5.288; 7.444 / 17.598, dk 1.859 / 5.098; 7.512 / 17.710,
q 1.851 / 5.050; 7.647 / 17.911, t 1.864 / 5.023; 7.566 / 17.755, against v1 1.828 / 5.186; 7.723 / 17.930. Nothing is adopted from these curves.
