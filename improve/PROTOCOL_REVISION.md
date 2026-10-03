# Revision protocol (written 3 October 2026, before any run described here)

This file fixes, before running, two changes made in response to an internal review. Nothing in the method changes.

## A. Prediction is separated from evaluation

1. `predict` code (run_x.py and multicity/run_multi.py, flag `--calendar`) builds the target time slots from the calendar
   only: every day of the target year and every 30-minute slot, except the slots that do not exist in the time
   convention of the dataset. Chicago, New York City and San Francisco record local clock time, so 02:00 and 02:30 on
   the day clocks move forward in spring (second Sunday of March) do not exist. The Washington, DC data are hourly
   counts with no gap at that hour, so all slots are kept. Target demand is not read by the prediction step for the
   target city. Helper cities are unchanged.
2. `evaluate_strict.py` loads the saved predictions and the target demand and scores every calendar slot from point 1.
   Slots where the data contain no trip at all, which are data gaps in San Francisco (2 slots) and Washington, DC
   (14 slots), are scored as observed, that is with zero trips. No slot is removed by looking at target demand.
3. For Chicago and New York City the calendar slot set is identical to the slot set used so far (the only empty slots
   are the two non-existent spring slots), so their numbers cannot change. This is checked in code.
4. San Francisco and Washington, DC are retrained with the frozen held-out method of PROTOCOL_HELDOUT.md (heads eh and
   eo, 10 runs each, and the base allocator, 10 runs) with the new slot rule, on CPU, and rescored. Check: for every
   slot that existed before, the new prediction must equal the old one up to floating-point noise
   (max abs share difference below 1e-5); otherwise the difference is reported.

## B. Aggregation control for scale matching (development targets only)

Question: does the gain come from matching the median area of the target, or from merging (smoothing) helper regions?

Targets and helpers: Chicago (helpers New York City and San Francisco, both merged) and New York City (helpers Chicago,
native, and San Francisco, merged). Every condition uses the final method: heads eh and eo, 10 runs each, run r uses
partition r (r = 1..10) and model seed r-1, final = equal average of the two 10-run ensembles.

Conditions, all built with the same block-feature formula as aggregate.py:

| Code | Helper representation |
|---|---|
| N | native helper regions (existing runs, seeds 0-9) |
| M | proposed: area-weighted k-means, K chosen so the median block area matches the target median area (existing runs) |
| R | random grouping with the same K as M: regions are assigned to K groups at random (balanced counts), so the amount of merging is the same but blocks are not spatially coherent |
| H | area-weighted k-means with K chosen so the median block area is about half of the target median area |
| D | area-weighted k-means with K chosen so the median block area is about twice the target median area |

K for H and D is chosen with the same search as aggregate.py (random_state 0), with the goal area 0.5 a_T or 2 a_T.

Scoring: fully strict, full year, trip MAE and RMSE. Primary inference: mean daily difference to M with a 7-day block
bootstrap (2,000 draws) 95% interval. Run-to-run variability is reported as mean and standard deviation over the 10
single runs, where single run r is the equal average of eh run r and eo run r.

Reading rule, fixed now: area matching is supported if M has lower MAE than R, H and D in both targets with bootstrap
intervals that exclude zero. Any other outcome is reported as it is.
