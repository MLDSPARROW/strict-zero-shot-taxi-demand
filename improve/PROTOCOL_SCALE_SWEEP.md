# Block-size sweep protocol (written 3 October 2026, before any run described here)

Question: how does the error of the final method depend on the block size of the re-aggregated helpers?

Targets and helpers as in PROTOCOL_REVISION.md part B: Chicago (helpers New York City and San Francisco, both merged)
and New York City (helpers Chicago, native, and San Francisco, merged).

Scale ratio q = goal median block area / target median region area. Values: 0.25, 0.5, 0.75, 1, 1.5, 2, 4.
q = 0.5 (H), 1 (M) and 2 (D) are the runs of PROTOCOL_REVISION.md part B and are reused unchanged. New runs: q = 0.25,
0.75, 1.5 and 4.

Blocks: area-weighted k-means on region centroids, K chosen with the search of aggregate.py (random_state 0) for the goal
area q times the target median, then 10 partitions (random_state 1..10). A helper whose native median region area is
already at least the goal area is used at its native scale for that q (this happens only for New York City as a helper
of Chicago at q = 0.25). Block features use exactly the formula of aggregate.py.

Runs: final method, heads eh and eo, run r on partition r with model seed r-1, 10 runs per head, `--calendar`, on the
same type of Kaggle CPU workers as part B. Scoring: fully strict, full year, trip MAE and RMSE of the final method
(average of the two 10-run ensembles), mean daily MAE difference to q = 1 with a 7-day block-bootstrap 95% interval
(2,000 draws), and the mean and standard deviation over the 10 single runs.

Reporting rule, fixed now: all seven values of q are reported for both targets in one table and one figure (MAE and
RMSE against q on a log scale), whatever the outcome. No value of q is chosen after the runs; the method keeps q = 1.
