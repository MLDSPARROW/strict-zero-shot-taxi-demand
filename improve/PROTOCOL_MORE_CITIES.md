# Austin and Porto as new targets (frozen BEFORE any Austin or Porto target run)

Written: 2026-10-17, before any model was trained or scored with Austin or Porto as the TARGET city, and before
the raw data for either city was re-downloaded this round.

Purpose: Austin and Porto have only ever been screened as candidate helper cities (see `UrbanMind_PROGRESS.md`,
section 6h and section 4, and the paper's "Approaches that did not help"), and both were rejected as helpers.
Neither has ever been predicted as a target. This protocol freezes the method and the evaluation plan before
either city is used as a target, the same way `PROTOCOL_HELDOUT.md` did for San Francisco and Washington, DC.

## Disclosed gap (added before any rebuild work, not after)

The scripts that originally built Austin's region grid (`multicity/data/austin_regions.geojson`) and Porto's
region polygons (feeding `porto_region_30min.npy`) were never committed to this repository, only their output
files were, and those output files were lost along with the rest of the local working data between 3 and 17
October 2026. Rebuilding Austin and Porto as targets therefore requires rebuilding these region definitions
from the raw public sources, not only re-running an existing script. This round, the rebuild script itself
will be saved to the repo, so this gap does not recur.

## Target data (public, as previously identified)

- Austin: RideAustin ride-hailing trips, June 2016 to February 2017 (~911,057 rides), aggregated to 1 km grid
  cells (687 cells in the earlier run).
- Porto: the public Porto taxi trip dataset, July 2013 to June 2014, aggregated to its existing small-area
  partition (485 areas in the earlier run).

## Frozen method (identical to the paper's final method)

1. LEVEL: unit chosen by leave-one-city-out among the level helpers only (candidates: jobs, residents,
   population, taxi commuters, transit commuters, car-free households); level helpers = Chicago, NYC, SF, DC.
   Austin has the same US Census covariates as the other targets. Porto does not: it is not a US city, so its
   LODES/ACS/FAA-equivalent covariates do not exist in the same form. Porto's LEVEL step will use the closest
   available Portuguese or Eurostat public statistic for each helper-selected unit, substituted one-for-one;
   if no defensible substitute exists for the unit chosen by the helper cities, this will be reported as a
   limitation rather than papered over with an invented number.
2. RHYTHM: mean day-of-week x slot profile of the share helpers; month factor from share helpers with the same
   calendar year as the target (none for Austin 2016-17 or Porto 2013-14, factor 1, same convention as SF 2023).
3. SHARE helpers rule: all other taxi cities whose pickup timestamps have native resolution of 30 minutes or
   finer (excludes DC, as in every other protocol). -> Austin and Porto target: share helpers Chicago, NYC, SF.
4. Scale matching: a share helper whose median region area is smaller than the target's is re-aggregated
   (area-weighted k-means, K chosen so the median block area is closest to the target median, partitions with
   random states 1..10); others are used at native scale. Which helpers qualify depends on Austin's and
   Porto's own median region area, which is not yet known until the region definitions are rebuilt, so this
   will be computed and reported, not assumed in advance.
5. Heads eh and eo, each a 10-run ensemble; final = equal-weight average. Same hyper-parameters as `run_x.py`.
6. Airport rule: FAA CY2021 enplanements (Austin only; Porto has no FAA data, substitute or omit, reported
   either way) for large airports whose reference point lies inside the region and whose municipality is the
   city itself.

## Evaluation (target data used only here)

- MAE and RMSE per region and slot, full available year, same metric definitions as every other target.
- Baselines with the same LEVEL and RHYTHM: uniform, jobs-proportional, jobs+residents, base joint allocator
  (10 seeds).
- No re-running or changing of the method after seeing these results; every number is reported as obtained,
  including if either city performs at or below baseline level.

## Disclosure

Both cities' demand data were previously used only as rejected helper candidates during development (see
`UrbanMind_PROGRESS.md` section 6h for Austin, section 4 for Porto); no component of the frozen method above
was chosen using them as targets, since they have never been used in that role before.

## Addendum (added 2026-10-17, before building the Austin region grid)

The RideAustin CSV's own coordinates are already rounded to 0.01 degrees (confirmed: every value has exactly
two decimal places), so each distinct (lat, lon) pair is a natural cell, matching the earlier note that the
687 one-km squares came from this same rounding. The script that chose which of these cells counted as a
region, and the exact threshold it used, was not saved, only the resulting count (687) was recorded. Rather
than search for a threshold that reproduces 687 exactly, which would be tuning a design choice to a known
answer after the fact, the rule declared here is: keep a cell only if it received at least one trip per week
on average over the dataset's active span (2016-06-04 to 2017-02-07, 35.4 weeks, so a threshold of 36 trips).
This rule is motivated independently of the target count, by wanting enough data per cell for a meaningful
weekly rhythm. Applied to the downloaded data, it gives 684 cells, close to but not identical to the earlier
687; the small difference is reported as is, not adjusted to match.
