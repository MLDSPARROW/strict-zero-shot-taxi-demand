# Goal 4, step 4: scale-aware neural forecaster N2 (frozen BEFORE any run of N2)

Written: 2026-10-28, after the V18 result (B1 = N1 + C5 passed the screen against C2 but is 1.5% (Chicago) and 4.1% (NYC) above Chronos-Bolt in MAE) and a second question to ChatGPT
(scratchpad q6/a6). Own models only; helper cities only; no Chronos. (ChatGPT called October-November "the untouched test"; it is our development period; December stays the test.)

## N2 (`nn19.py`; the net of PROTOCOL_V18 with these changes only)
- Scale s = max(1, mean of the 336 counts). Inputs: counts / s (clipped at 30), log s, mean of the last 6 slots / s, (mean of the last 6 - mean of the 6 before) / s, calendar as N1. Output: max(b_j + s * g_j, 0), b_j the 7-day same-slot mean.
- Multiplicative augmentation in training: every window's counts (context, baseline and label) are multiplied by a, log-uniform on [0.5, 2]; s is computed after the multiplication.
- Loss on the normalised residual (p - y) / s: 0.5 Huber(delta 1) + 0.5 * 0.05... precisely: Huber(delta 1) + 0.05 * L1.
- 1,000,000 sampled windows per epoch, batch 2048, AdamW learning rate 1e-3, weight decay 1e-4, at most 12 epochs, patience 3 on helper-validation raw-count MAE (unaugmented).
- Partitions r = 1, 2 (seeds 0, 1), forecast = mean of the two runs.
## Candidates
N2 alone; B2 = equal average of N2 and C5; B3 = equal average of N1, N2 and C5 (fixed now, no fitted weights).
Kill check of N2 (against N1): NYC MAE at least 3% below N1 and Chicago not more than 1% above N1; otherwise N2 is dropped (result recorded).
Screen (against C2): mean MAE ratio <= 0.97, no city above 1.01, no RMSE ratio above 1.02. Goal rule: MAE and RMSE below Chronos-Bolt-small in both cities. December only for a candidate that reaches the goal rule.

## Result on October and November 2021 (recorded 2026-10-28; `../results/v19_dev.txt`)
N2: NYC 1.812 / 5.757 (N1 1.773, C2 1.738, Chronos 1.577): kill check against N1 FAILED (NYC not 3% better), N2 dropped as a stand-alone model. Chicago N2 is in the file.
Averages: B2 (N2 + C5) Chicago 1.173 / 3.352, NYC 1.658 / 5.178; B3 (N1 + N2 + C5) Chicago 1.163 / 3.303 (0.5% above Chronos MAE, 2.9% RMSE; daily difference +0.0062 [-0.0182, +0.0221]), NYC 1.659 / 5.250 (5.2% above).
Both pass the screen against C2; neither reaches the goal rule (below Chronos in both cities). Best NYC so far remains B1 (1.642 / 5.190). December is NOT run.
