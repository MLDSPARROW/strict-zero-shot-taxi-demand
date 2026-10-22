# Version 4: masked pre-training inside the allocation step (frozen BEFORE any run)

Written: 2026-10-22, before any model with the new switches was trained for this protocol, before any of their predictions was scored,
and before any map pre-training output that is used here was produced. Committed and time-stamped (`PROTOCOL_V4.timestamp`) with the
code (`mapmae.py`, `run_x.py`, `runner_v4.py`, `v4_eval.py`) before the runs.

## Why

Scale matching, the contribution of the paper, is not changed: helpers are re-aggregated to the target's median region area in ten
k-means partitions exactly as before. Both new parts sit inside the allocation step (step 3) and use the same target scale.

- The error of New York City is almost entirely allocation (`results/diag_dev.txt`: MAE 7.45 with predicted or true city totals,
  2.31 with a perfect allocation). The allocator sees each region's own map profile, a hand-made local contrast (busyness minus the
  mean of the six nearest regions) and calendar inputs, and it is trained on only two share helpers per target.
- Way 1 (switch `m`) learns how cities are built from public maps alone by masked reconstruction (masked autoencoders, He et al.
  2022), with hidden patches the size of the target's regions, and gives every region or block a context latent: what its
  surroundings imply at the target's scale. It generalises the hand-made local contrast.
- Way 2 (switch `d`) is masked demand modelling on the scale-matched helper blocks. Masked pre-training of spatio-temporal models
  (GPT-ST, UniST, both cited in the paper) hides parts of the demand of the city that is later predicted; here only helper demand is
  ever hidden or shown, and the target is always fully hidden.

## Way 1: scale-matched masked map pre-training (`mapmae.py`, map data only, no taxi data)

For a target T with median region area a_T (public polygons) and a seed s:
- Cells: the 40 x 50 map grids of Chicago, New York City, San Francisco, Washington, DC and Austin (5 x 2,000 cells), 13 features per
  cell (8 log place densities, 3 log distances, log busyness, place-mix entropy; the 3 missing-data flags are zero in every city and
  are dropped), standardised with the median and inter-quartile range pooled over all cells, clipped to [-8, 8]. The archived grids of
  Chicago, NYC, SF and DC are the reference features of the paper (they reproduce the stored region features exactly, `aggregate.py`
  formula check). The public map of the target is part of the pre-training data; no taxi data of any city is read.
- Cell size: the archived grids do not store their edges, and the edge files rebuilt after the loss of the local data have slightly
  different extents (Chicago: 0.725 against 0.621 km2 per cell). The cell height and width are therefore recovered from each archived
  grid by regressing the region centroids (km) on their overlap-weighted positions in grid units (R2 >= 0.9998 in every city) and
  rescaled to the exact cell area of the grid file (Chicago 0.99 x 0.64 km, NYC 1.04 x 0.89, SF 0.62 x 0.58, DC 0.57 x 0.36; the
  fitted area differs from the stored one by 1.8%, 0.7%, 0.0% and 0.0%). Austin: from the edges stored with its grid.
- Masks, the scale-matching rule applied to map cells: per city, 20 partitions of its cells into superpixels by k-means on the cell
  centres in km, K = round(grid area / a_T) (at most one per cell), random_state 1000 s + j. A training mask hides a random half of
  the superpixels of one partition, so every hidden patch has about the size of a target region.
- Encoder: masked normalised convolution in physical units, 3 residual layers with hidden size 64; in each layer every cell receives
  the Gaussian-weighted mean of the states of the visible cells at the bandwidths 0.5, 1 and 2 x sqrt(a_T) km and the visible
  fraction of the kernel mass at each bandwidth. A hidden cell starts from a learned mask token and is never a source, so its state
  depends only on its visible surroundings. Bottleneck: 8 numbers per cell; decoder: 2-layer perceptron to the 13 features.
- Training: mean squared error on the hidden cells, averaged over the five cities (one mask per city and step), AdamW (learning rate
  1e-3, weight decay 1e-4), 2,000 steps, gradient norm clipped at 1, torch and numpy seeds s.
- Output: the context latent of every cell is its bottleneck while it is hidden, averaged over the 20 partitions, each used once with
  a random half of its superpixels hidden and once with the other half. The latent of a region or block is the mean over its cells
  weighted by overlap area (the region-feature formula). Blocks of re-aggregated helpers are rebuilt with the original k-means seeds and
  checked against the stored jobs sums (as in `build_acs.py`).
- Check, reported and not used for any choice: reconstruction error on the hidden cells of 5 further partitions that are not used in
  training, against the city mean of the visible cells and the Gaussian-weighted mean of the visible cells at the bandwidth sqrt(a_T).
- In the allocator (`run_x.py`, switch `m`), the 8 latents are appended to the region inputs and standardised with the helper-pooled
  median and inter-quartile range like all other inputs. Run r uses the map pre-training of seed r - 1.

## Way 2: masked demand modelling on the helper blocks (`run_x.py`, switch `d`)

- In every batch of 64 helper slots, half are the usual task (all regions hidden, the usual loss). In the other half, a spatially
  contiguous random subset of the regions is visible: its observed shares enter as an input (visible indicator and visible x
  log(1 + R x share), through a new linear layer added to the first hidden state), and the model predicts how the remaining trips of
  the slot are shared among the hidden regions (cross-entropy with the predicted and observed shares renormalised over the hidden
  regions). The hidden subset is the lowest-ranked fraction rho of a Gaussian-smoothed white-noise field over the region centroids
  (bandwidth sqrt of the median region area of that helper), with rho drawn from U(min(1, 0.3 + 0.07 x epoch), 1), so the hidden
  fraction rises from 30% to 100% and from epoch 10 on every slot is the usual task.
- The masks use their own random generator, so the helper and batch draws are those of the run without `d`. Validation, early
  stopping and prediction are unchanged and never see any demand: the target is always fully hidden.

## Variants, runs and comparator

- Heads eh and eo with the switches m, d and md (ehm, eom, ehd, eod, ehmd, eomd); the final allocation of a variant is the equal
  average of its two heads, as in v1.
- Development targets only, helpers of the paper (Chicago <- nyc_for_chicago_p_r + sf_for_chicago_p_r; NYC <- chicago +
  sf_for_nyc_p_r). Screen: runs r = 1, 2, 3 (helper partition r, model seed r - 1, map pre-training seed r - 1). Demand-reading mode
  as in the PROTOCOL_V2 screen: the target demand is used for scoring only, by `v4_eval.py`.
- Comparator: v1 = eh and eo, runs r = 1..10 that exist in `out/` from the PROTOCOL_V2 screen and confirmation. Reproduction check
  done before this protocol: `run_x.py` with the new code and without the new switches reproduces the stored v1 run (Chicago, r = 1,
  head eh) exactly (largest absolute share difference 0.0; MAE 1.776 and 19 epochs as in `runner_v2_screen.log`).

## Rules (fixed now)

- Screen (`python v4_eval.py`): a variant passes if the MAE_share of its 3-run ensemble is at least 3% better than v1 in one
  development target and not more than 3% worse in the other (the rule pre-declared in `run_x.py`).
- Confirmation (`python v4_eval.py --confirm`, runs r = 1..10 of every passing variant): a variant is adopted if (a) its mean MAE_share
  over the two targets is lower than v1's, (b) the 7-day block bootstrap (2,000 draws) interval of the daily MAE_share difference
  v1 - variant lies above zero in at least one target, and (c) in the other target it is not more than 3% worse and its interval does
  not lie entirely below zero. If several variants are adopted, the one with the lowest mean MAE_share over the two targets is selected.
- Held-out evaluation, only if a variant is adopted: fully strict calendar-slot predictions, ten runs per head, for SF, DC, sf_zcta,
  dc_cluster and Austin, with the same helpers, partitions, seeds and city totals as v1, scored by the paper's scripts against v1 and
  the baselines with the 7-day block bootstrap, and reported as obtained (SF and DC as a second look, because their v1 results are
  known). Before that, `mapmae.py` is extended to the unit sets of these targets (coarse regions, Austin cells and their helper blocks)
  without any change to the method, and this is recorded in an addendum before any held-out run.
- If no variant is adopted, v1 stays the final method, the result is recorded in this file, and the paper is not changed.

## Compute

Six map pre-trainings (about 8 to 12 minutes each) and 36 allocator runs (about 4 to 6 minutes each) for the screen, on two workers
of the laptop, about 2.5 hours. Confirmation: up to 28 further runs per passing variant (and 14 pre-trainings if m is involved).

## Disclosure

- No new taxi data of any target was read to design this version. Its motivation uses diagnostics that are already in `results/` and
  in the disclosure of PROTOCOL_V2.md.
- Code tests before this protocol: map pre-training with 20 and 100 steps for the Chicago target and 20 steps for the NYC target, into
  a separate folder that was deleted (the 100-step check: model error below the Gaussian reference in all five cities, ratio 0.70 to
  0.97), and 1-epoch allocator runs of 5 steps with m and d in calendar-slot mode, which never opens the target demand; their outputs
  were deleted. The only change made after these tests was the cell-size recovery described above (a correctness fix: the first test
  stopped at the cell-area check). The design values (masked fraction, partitions, layers, bandwidths, latent size, steps, batch split,
  hidden-fraction schedule) were fixed before any test and were not tuned.

## Result (2026-10-22, `results/v4_screen.txt`; recorded here as obtained)

All 6 map pre-trainings and 36 allocator runs finished as specified (`runner_v4_w0.log`, `runner_v4_w1.log`; mean 176 s and 17.6
epochs per run). Map pre-training check (map data only): in all 30 city-seed-target cases the reconstruction error on the hidden cells
of unused partitions was below the Gaussian reference, ratio 0.46 to 0.87.
Screen, MAE_share of the 3-run ensembles of the two-head average, change against v1 (positive = better):
- m: Chicago -8.3% (0.01015 against 0.00937; peak ratio 0.77 against 1.01, so the busiest areas are under-predicted), NYC +1.7%
  (0.00452 against 0.00460). FAIL.
- d: Chicago +0.6% (0.00931; interval of the daily difference v1 - d [+0.000045, +0.000080], better on 65% of days), NYC +2.8% (0.00447;
  [+0.000123, +0.000135], better on 100% of days). Better in both targets, but below the 3% threshold in both. FAIL.
- md: Chicago -3.9% (0.00973), NYC +1.6% (0.00453). FAIL.
Allocation-only trip MAE / RMSE (true slot totals): Chicago v1 1.494 / 4.035, m 1.632 / 4.575, d 1.487 / 4.063, md 1.531 / 4.211; NYC v1
7.735 / 17.643, m 7.526 / 17.154, d 7.517 / 17.260, md 7.528 / 17.170.
DECISION by the rule fixed above: no variant passes the screen, so no confirmation and no held-out run is made, v1 remains the final
method and the paper is not changed. No variant of m or d will be tried on the basis of this result without a new time-stamped
protocol that discloses it.

## Addendum 1 (2026-10-22, after the screen result above, before any further run)

Way 2 (switch d) did not pass the screen: Chicago +0.6% and NYC +2.8%, so the 3% threshold was not reached in either development target.
Way 2 is carried to the confirmation stage anyway, by the decision of the first author, taken after seeing the screen result and the
allocation-only trip errors above. The screen threshold is therefore waived for d by decision and not by the rule, and this is
disclosed here. The reasons given for the decision: d is the only variant that was better than v1 in both development targets (m and md
are worse in Chicago, -8.3% and -3.9%), and in NYC it was better on every day of the year. m and md are not carried forward.
The confirmation rule above is unchanged: runs r = 4..10 of ehd and eod are run for Chicago and NYC (28 runs), the 10-run ensemble
(r = 1..10; runs 1 to 3 are those of the screen) is compared with the existing 10-run v1, and d is adopted only if rules (a), (b) and (c)
hold (`python v4_eval.py --confirm`). Rule (b) asks for an interval above zero, which a small effect can satisfy, so an adoption would be
reported as a small refinement. If d is adopted, it is evaluated once on the held-out targets as specified above (it needs no map
pre-training), fully strict, and reported as obtained, with its post-screen origin stated. If it is not adopted, v1 stays and the paper is
not changed.
