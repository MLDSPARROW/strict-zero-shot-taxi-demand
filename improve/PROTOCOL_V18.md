# Goal 4, step 3: a neural forecaster of our own on the 336-slot context, and an equal average with C5 (frozen BEFORE any run of the network)

Written: 2026-10-28 after the failed screen of C5 (PROTOCOL_V17). No Chronos in any form. Training on helper cities only (as PROTOCOL_V8: Chicago <- NYC, SF; NYC <- Chicago, SF; scale-matched
partition r; 2021 helpers cut before 1 October 2021; SF whole; held-out helper days = day of the year mod 12 = 5 for early stopping).

## Network N1 (`nn18.py`)
Input per (origin, region): the 336 counts divided by s = mean of the 336 counts + 0.1 (clipped to 30); log s; the seven same-slot counts of the previous days are inside the window;
calendar of the origin and of the forecast slots (sin/cos of slot of day, weekday, weekend, federal holiday flags for the day of the origin). No region or city identifier, no map features.
Net: 336+extra -> 256 -> 128 -> 6, GELU, dropout 0.1. Output: forecast_j = max(b_j + s * g_j, 0), where b_j = mean of the same slot over the previous 7 days (the weekly average)
and g_j the network output. Loss: mean of Huber(delta 2) and L1 on the raw counts (so the loss is in the units of the metric). AdamW, learning rate 3e-4, weight decay 1e-4,
batch 512, at most 15 epochs of 300,000 sampled windows, early stopping on helper validation (patience 3). Training origins: every 12 slots from day 8, as C2.
Runs: partitions r = 1, 2, seed r - 1; forecast = mean of the two runs.
Kill check: after epoch 1 the helper validation MAE must be below that of the weekly average b; otherwise stop and debug the normalisation (recorded).

## Candidates scored on October and November 2021 (same origins as V8)
N1 alone; B1 = equal average of N1 and C5 (our own models, fixed now, no fitted weights).
Screen (against C2): mean MAE ratio <= 0.97, no city above 1.01, no RMSE ratio above 1.02. Goal rule (real goal 4): MAE and RMSE below Chronos-Bolt-small in both cities.
December 2021 only for a candidate that reaches the goal rule; ten partitions then; once.
