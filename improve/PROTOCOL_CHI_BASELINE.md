# Baseline: the spatiotemporal allocation of Chi, Xu and Chen (2026) (frozen BEFORE any run)

Written: 2026-10-18, before the baseline was trained or scored on any target.

## Why now

An earlier POI-activity baseline modelled on this paper was removed because only the abstract was available and the
re-implementation could not be checked. The full paper (Transportation Research Part A 213, 105233) is now available, so
its second stage can be implemented as published. Section and equation numbers below refer to that paper.

## What is implemented (Section 3.4, Eqs. 2 to 7, Algorithm 1)

- Regional temporal heat h_{r,t} = sum_c n_{r,c} a_{c,t} (Eq. 2), allocation delta_{r,t} = alpha_t h_{r,t} / sum (Eq. 5),
  alpha_t = 5 on weekday slots and 2 on weekend slots.
- Weekly template of T = 48 hourly slots (24 weekday, 24 weekend). Time encoding e_t = [I_weekend, cos(2 pi h/24),
  sin(2 pi h/24)] (Eq. 3).
- a_t = ReLU(MLP(e_t)), three hidden layers of 48, 64 and 48 units with LeakyReLU (Eq. 4).
- Loss: KL(delta_obs || delta) over all regions and template slots of each helper city (Eq. 6), helpers pooled; Adam,
  learning rate 0.001, full batch, 5,000 epochs.
- delta_obs of a helper: its mean weekday and mean weekend demand per region and hour, weighted 5 and 2, normalised over
  all regions and slots.

## Adaptations, stated now

1. Stage 1 (city-level linear regression on population, GDP and POI count) is NOT used: with two or three helper cities
   per target it cannot be estimated. Like every other baseline in the paper, the Chi allocation is combined with the
   paper's own level and rhythm, so only the spatial allocation differs.
2. POI categories: the paper uses 14 Amap categories plus population. Here, the 8 OpenStreetMap groups already in the
   region features (counts = expm1(log density) x area) plus LODES resident workers as the population category.
3. Units: the paper uses square grid cells; here each city's own regions, since the method only needs per-region counts.
4. The hourly template is applied to both half-hours of each hour. For each 30-minute calendar slot of the target, the
   spatial share is delta normalised over regions for that slot's hour and day type.
5. Helpers at native scale (the paper does not re-aggregate). Ten seeds (0 to 9), ensemble = mean of normalised shares,
   for parity with the other learned allocators; single-run mean and SD also reported.

## Targets and scoring

All targets of the paper and of PROTOCOL_COARSE_TARGETS.md and PROTOCOL_MORE_CITIES.md, with the same share helpers as
the final method. Fully strict trip MAE and RMSE with the same city totals, scored by the evaluation scripts only; the
prediction script never opens the target demand. Reported as obtained, whether better or worse than the final method.
