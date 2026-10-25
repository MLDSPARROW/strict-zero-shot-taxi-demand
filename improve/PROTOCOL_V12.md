# Fixed combinations of the final method with the jobs-and-residents prior (frozen BEFORE any rule below is scored)

Written: 2026-10-25, before any of the rules below was computed, scored or looked at, on any target. Committed and time-stamped
(`PROTOCOL_V12.timestamp`) with the scripts `v12_rules.py`, `v12_dev.py` and `v12_heldout.py`.

## Why

A review round (an external language model, asked with the numbers of the paper and the rules of this project, 2026-10-25) and the results of
PROTOCOL_V10 agree on one point. The learned allocators do not transfer reliably, and a transparent allocation proportional to public jobs and
residents is as accurate or more accurate in SF, DC, the coarse targets and Austin (`results/v10_heldout.txt`), while it is clearly worse in Chicago and
NYC. The idea is to keep the final method v1 and anchor it to this prior. Only fixed rules with equal weights are tested; no weight is tuned on any
target. This is a different question from PROTOCOL_V10 (which mixed v1 with XGBoost).

## Rules (fixed now; shares per slot, same city totals as the paper, no target data)

Let v1 be the shares of the final method (equal average of the ten-run eh and eo ensembles), g the shares of the scale-matched XGBoost baseline
(ten runs, `out_xgb/`), and jr the jobs-plus-residents share of the target (LODES, public, the same in every slot).
- R1 = 1/2 v1 + 1/2 jr (arithmetic).
- R2 = 1/3 (v1 + g + jr) (arithmetic).
- R3 = normalised square root of v1 x jr per slot (geometric mean, equal weights).
No other rule, weight, temperature or feature is tested in this protocol. Any further idea needs a new protocol.

## Honest limits (stated before the runs)

- Chicago and NYC are the development targets. SF, DC, sf_zcta, dc_cluster and Austin are NOT untouched: the jobs-plus-residents numbers of all five were read
  in PROTOCOL_V10, so their results here are a second look and not an independent test. No untouched target with taxi demand exists.
- The rules have no fitted parameter, so they cannot be overfitted by a search, but they were chosen after the V10 table was known. This is disclosed in the paper if
  any rule is used.

## Development rule (Chicago and NYC, `v12_dev.py`; comparator is v1)

A rule passes development only if the mean over the two cities of the relative MAE change (v1 - rule) / v1 is at least +3%, neither city's MAE
change is below -2%, and neither city's RMSE change is below -5%.

## Held-out rule (`v12_heldout.py`, run once, only for rules that pass development)

Adopted only if the mean MAE change and the mean RMSE change over the five held-out targets are both above zero and the rule is better than v1 in MAE
in at least three of the five and in RMSE in at least three of the five (the rule of PROTOCOL_V10). Every number is reported whether or not the rule holds.
If a rule is adopted, the paper reports it as a second look, with v1 and the prior beside it on every target.

## Correction recorded before any result

`v12_dev.py` crashed on its first run (KeyError: the helper lists of Chicago and NYC, which `v10_dev.py` sets with `SHARE.update`, were missing). The line was added. No number of any rule had been computed or seen. The hash of `v12_dev.py` in `PROTOCOL_V12.timestamp` is that of the first version.

## Result of the development screen (2026-10-25, `results/v12_dev.txt`; recorded as obtained)

MAE / RMSE of the final method v1: Chicago 1.829 / 5.207, NYC 7.454 / 17.590. Change against v1 (positive = rule better), MAE / RMSE:
R1: Chicago -6.71% / -2.03%, NYC -6.25% / -0.32%. R2: Chicago -9.17% / -0.60%, NYC +4.07% / +4.43%. R3: Chicago -9.14% / -10.08%, NYC -6.36% / -0.88%.
No rule passes the development rule (mean MAE change at least +3% and no city below -2%): mean MAE changes -6.48%, -2.55% and -7.75%. By the protocol the rules are not run on the held-out
targets, and v1 stays the final method. Reading: the jobs-plus-residents prior is clearly worse than v1 in Chicago (2.595 against 1.829) and in NYC (8.645 against 7.454), so a fixed equal mix
with it loses in both development cities. Only R2 helps NYC, through the XGBoost share, and it hurts Chicago, as in PROTOCOL_V10.
