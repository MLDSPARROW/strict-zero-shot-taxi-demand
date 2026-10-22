# Extra: the strict forecast plus a week of target data, on equal terms with a foundation model (frozen BEFORE any run)

Written: 2026-10-22, before any of the forecasts below were made. Committed and time-stamped (`PROTOCOL_V7.timestamp`) with the plan.
The Chronos-Bolt run of PROTOCOL_V6.md was started before this protocol, and none of its results has been looked at.

## Why

Request of the first author: compare the method with models that use one week of target data, or any amount of data, and also state the
result without any target data. The strictly zero-shot result of the paper is unchanged. This extra asks what the strictly zero-shot
forecast adds when a week of target data is available, and how this compares with the foundation model of PROTOCOL_V6 given the same week.
The extra is not part of the strictly zero-shot claims of the paper and is labelled "uses the previous seven days of the target".

## Method (fixed now; no training; it uses the strict forecast that already exists and the observed counts of the previous seven days)

Inputs at a forecast origin t0 (a half-hour slot; origins every 6 slots): the observed counts of all regions before t0, and the strict
forecast of the final method (v1: ten-run ensemble of the shares of heads eh and eo, condition M, with the strict city totals, `out_cal`),
which uses no target data. For the target slot t = t0 + j (j = 0, ..., 5) and region r:
- Observation set I(t): the days i = 1, ..., 7 before t whose type (Monday to Friday, or Saturday and Sunday) is that of t;
  n(t) = |I(t)|; S(r,t) = sum over i in I(t) of the observed count y(r, t - 48 i). (All are before t0.)
- Level of the week: rho_week(t0) = sum of the observed city totals over the slots [t0 - 336, t0) divided by the sum of the strict city
  totals over the same slots (slots without a strict forecast skipped).
- Level of today: rho(t0) = (sum of the observed city totals over [t0 - 6, t0) + 1) / (sum of E(u) over the same slots + 1), where E(u) is the
  mean of the observed city totals at u - 48 i over the days of I(u); clipped to [0.5, 2].
- Prior: lambda0(r,t) = rho_week(t0) x (strict forecast of r at t), at least 1e-6.
- Forecast, the posterior mean of a Gamma-Poisson model in which the prior is worth a trips:
  lambda(r,t) = (a + rho(t0) S(r,t)) / (a / lambda0(r,t) + n(t)).
  a = 0 is the data-only estimate (the mean of the same slot on the days of the same type, scaled by the level of today);
  a very large is the strict forecast alone, scaled to the level of the week.
- One parameter, a. It is chosen once, on October and November 2021 of Chicago and New York City (origins every 6 slots, horizons 1 to 6),
  from the grid {0, 0.5, 1, 2, 4, 8, 16, 32, 64}: the value that minimises the mean over the two cities of MAE(a) / MAE(0); ties go to the
  smaller a. It is then frozen and not changed.

## Evaluation (`fewshot.py`)

- December 2021, Chicago and New York City, the same origins, horizons and slots as PROTOCOL_V6 (each slot forecast once), mean-forecast MAE and
  RMSE at 30-minute resolution. Rows: the strict forecast (version A, same slots), the data-only estimate (a = 0), the forecast with the chosen
  a, and next to them the result of Chronos-Bolt-small (PROTOCOL_V6) and the values of the cost-of-strictness table.
- Hypotheses, fixed now. H1: the forecast with the chosen a has a lower MAE and RMSE than the strict forecast in both cities (expected). H2: it has
  a lower MAE than the data-only estimate (a = 0) in both cities, that is, the strict forecast adds information to a week of data. H3 (uncertain,
  no expectation): it has a lower MAE than Chronos-Bolt-small in both cities. Everything is reported as obtained, whether better or worse.
- Held-out cities (San Francisco, Washington, DC, the two coarse versions, Austin) are not touched by this protocol. If the extra is carried
  to them, it will be under a new addendum, once, with the frozen a.

## Disclosure

- The idea was formed after seeing that simple methods that copy the previous weeks beat the strict forecast by a wide margin in most
  targets (`results/history_baselines.txt`); the form of the estimator is the standard shrinkage of a Gamma-Poisson model, and a is chosen on
  October and November, not on December.
- The strict forecast used as the prior is that of the final method (v1); the strict method of Way 2 (PROTOCOL_V4) is not used here.
