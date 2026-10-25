# Ridge tilt of the final method, fitted and selected on the share helpers only (frozen BEFORE the tilt is fitted or scored)

Written: 2026-10-25, after PROTOCOL_V12 (fixed mixes with the jobs-and-residents prior failed the development rule, `results/v12_dev.txt`) and before any number of the method
below was computed. Committed and time-stamped (`PROTOCOL_V13.timestamp`) with `v13_tilt.py`. The idea comes from an external review round (a language model, asked
with the numbers of this project, 2026-10-25) and was adapted to the pseudo-target files that exist since PROTOCOL_V2, addendum 6.

## Method

Let s[t, i] be the shares of the final method v1 for the target T (equal average of the ten-run eh and eo ensembles, unchanged). For every region i that is not an airport region
(enplanements = 0 in `data/enpl_<tag>.npy`) the tilt multiplies the share by exp(clip(z_i beta, -2, 2)); the shares of the airport regions keep their mass, and the non-airport shares are
renormalised to the remaining mass in every slot. The four features z are standardised with the mean and standard deviation of the training regions: log(1 + LODES jobs),
log(1 + LODES resident workers), the log busyness map feature (column 12 of the static features) and log(area in km2). beta is the ridge solution
of: minimise over beta and an unpenalised intercept the sum over non-airport regions of p_i (y_i - z_i beta - b)^2 + lambda |beta|^2, with y_i = log((q_i + 1e-6) / (p_i + 1e-6)), where p_i is the predicted
annual share of a share helper c (pseudo-target prediction of v1, true slot totals as weights, as in `calibrate.py`) and q_i its observed annual share. Each helper counts equally in the fit.

## What is fitted on what (no taxi datum of T enters)

For T = Chicago the share helpers are NYC and SF, for T = NYC they are Chicago and SF, each scale-matched to T exactly as in the paper. The pseudo-target predictions of v1 for each helper,
trained on the OTHER helper only (5 runs per head, `out/multi_<tag>_from_<helpers>_v{eh,eo}_seed*.npy`, `calibrate.py` PSEUDO), already exist; they were produced in PROTOCOL_V2 addendum 6
and no new training is needed. lambda is chosen from {0.1, 1, 10, 100} by two folds: fit on one helper, apply to the pseudo-prediction of the other, and take the mean over the two folds of
the relative gain of the allocation MAE (true slot totals of the helper); the largest lambda wins ties. The final beta is fitted on both helpers with the chosen lambda and applied to v1 of T.

## Kill check (computed before T is scored)

Stop for T, score nothing and keep v1, if the fitted coefficients of jobs or residents change sign between the two folds at the chosen lambda, or if the tilt does not lower the held-out-helper
MAE in either fold.

## Development rule (Chicago and NYC, comparator v1, strict trips with the city totals of the paper)

Pass only if the mean over the two cities of the relative MAE change is at least +3%, neither city's MAE change is below -2%, and neither city's RMSE change is below -5% (as PROTOCOL_V12).
If it passes, the held-out evaluation needs pseudo-target runs for each held-out target and is specified in an addendum before it is run; it is a second look (the held-out numbers of the
jobs-and-residents allocation are known). If it does not pass, v1 stays.

## Not run: explicit airport component (the second experiment of the review)

It would fit trips per enplanement on one helper city. Each target has exactly one share helper with airports (Chicago has NYC; NYC has Chicago; SF has none), so no helper-only
validation is possible and the coefficient could only be checked on the target itself, which is not allowed. It is not run, and the review's check ("predicted airport shares of the helpers under
leave-one-out") cannot be computed.

## Disclosure

The v1 numbers of Chicago and NYC, the jobs-and-residents numbers and the failure of PROTOCOL_V12 and of PROTOCOL_V2 addendum 6 (shrinkage towards a prior fitted on pseudo-targets) were known when
this protocol was written. No number of the tilt was known.
