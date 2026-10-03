# Held-out target evaluation protocol (frozen BEFORE any San Francisco or DC target run)

Written: 2026-10-02, before any model was trained or scored with San Francisco or Washington DC as the TARGET city.
Purpose: Chicago and NYC were used as development targets (design decisions were screened on them). San Francisco (2023)
and Washington DC (2021) have never been predicted as targets. The method below is frozen and applied to them once.

## Frozen method (identical to the paper's final method)
1. LEVEL: unit chosen by leave-one-city-out among the level helpers only (candidates: jobs, residents, population,
   taxi commuters, transit commuters, car-free households); level helpers = all other taxi cities.
2. RHYTHM: mean day-of-week x slot profile of the share helpers; month factor from share helpers with the same calendar
   year as the target (none for SF 2023 -> factor 1).
3. SHARE helpers rule (data-quality rule, stated before the run): all other taxi cities whose pickup timestamps have
   native resolution of 30 minutes or finer. DC timestamps are hourly, so DC is never a share helper.
   -> SF target: share helpers Chicago, NYC.  DC target: share helpers Chicago, NYC, SF.
4. Scale matching: a share helper whose median region area is smaller than the target's is re-aggregated (area-weighted
   k-means, K chosen so the median block area is closest to the target median, partitions with random states 1..10);
   others are used at native scale.
   -> SF target (median 0.35 km2): no helper is finer, no re-aggregation; 10 runs = seeds 0..9.
   -> DC target (median 0.52 km2): SF (0.35 km2) re-aggregated, 10 partitions; run r uses partition r and seed r-1.
5. Heads eh and eo, each a 10-run ensemble; final = equal-weight average. Same hyper-parameters as run_x.py.
6. Airport rule: FAA CY2021 enplanements of large airports whose reference point lies inside the region and whose
   municipality is the city itself. SF and DC have no such airport region.

## Evaluation (target data used only here)
- MAE and RMSE per region and slot, full year. SF: 30-min slots of 2023. DC: hourly slots of 2021 (DC data are hourly;
  half-hour values are equal splits), 30-min reported as secondary.
- Baselines with the same LEVEL and RHYTHM: uniform, jobs-proportional, jobs+residents, base joint allocator (10 seeds).
- No re-running or changing of the method after seeing these results; every number is reported as obtained.

## Disclosure (added before the run)
San Francisco was previously predicted as a pseudo-target (sf <- nyc, sf <- chicago; 3 seeds, heads eh) only to calibrate a
post-hoc sharpening step that was rejected; no component of the frozen method above was chosen using those runs.
Washington DC has never been predicted as a target. Both cities' demand data were used as helper data during development.

## Addendum (added 2026-10-02 while the held-out runs were in progress and BEFORE any held-out result was inspected)
1. Additional baseline "POI-activity allocation (adapted from the design described by Chi et al., 2026)": same LEVEL and
   RHYTHM as the final method; allocation pi[r,t] proportional to sum_k n[r,k] * a_k(t), where n[r,k] are OpenStreetMap
   counts of the 8 point-of-interest groups in region r and a_k(t) = softplus(MLP_k(calendar features)) are non-negative
   activity profiles learned on the share helpers (cross-entropy on shares, same optimiser, 10 seeds). Applied to all four targets.
2. Airport rule reformulated as a purely geographic rule: an airport is assigned if its reference point lies inside the
   city's 2021 Census place boundary. Checked to give identical assignments to the municipality rule in all four cities.
