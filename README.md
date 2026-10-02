# Strictly zero-shot taxi demand prediction via scale-matched multi-source transfer — code

Code accompanying the paper *"Strictly zero-shot taxi demand prediction for unseen cities via scale-matched
multi-source transfer"* (M. Ghalejughi). Every number in the paper's tables is produced by the scripts listed below,
from saved model predictions.

**Strictness rule enforced throughout:** no taxi data of the target city (Chicago or NYC) is used for training,
scaling, early stopping, model/variant selection, the city-wide total, or as input. Target data is read only by the
evaluation code.

## Directory layout

The scripts use relative paths and expect this layout (the same as the original working directory):

```
bench77/       Chicago and NYC 2021 demand arrays + region features
multicity/     San Francisco and DC demand + features; base joint allocator (run_multi.py)
grid/          fine-grid features and cell-to-region area fractions (grid_prep.py)
improve/       extra covariates, scale matching, final models, strict evaluation, paper tables
kaggle-fix/    notebook.ipynb: OpenStreetMap feature builder (build_static_features), reused unchanged
```

## Pipeline (in order)

| Step | Script | Output |
|---|---|---|
| 1. Chicago & NYC demand (2021, 30-min, community areas / taxi zones) | `bench77/prep_data.py` | `bench77/data/*_30min.npy` |
| 2. Region map features (18) from the 40×50 OpenStreetMap grid | `bench77/region_features.py`, `multicity/prep_sf.py`, `multicity/prep_dc.py` | `region_static_<city>.npz` |
| 3. Public covariates: LODES jobs/residents, OSM rail stations, airport flag | `improve/build_extra.py` | `improve/data/extra_<city>.npz` |
| 4. FAA CY2021 enplanements per region (ORD 26,350,976; MDW 7,680,617; JFK 15,273,342; LGA 7,827,307) | `improve/make_enpl.py` | `improve/data/enpl_<city>.npy` |
| 5. Fine-grid cell-to-region fractions | `grid/grid_prep.py` | `grid/data/grid_<city>.npz` |
| 6. Scale-matched helpers (partition 0; median block area = target median region area) | `improve/aggregate.py`, `improve/aggregate_dc.py` | `improve/data/agg/` |
| 7. Ten random re-aggregations per helper | `improve/aggregate_multi.py` (partitions 1–5), `improve/aggregate_multi2.py` (6–10) | `improve/data/agg/*_p<k>_*` |
| 8. Base joint allocator (10 seeds) | `multicity/run_multi.py` | `multicity/multi_*_seed*.npy` |
| 9. Allocator heads eh / eo on native, scale-matched and partition-diverse helpers | `improve/run_x.py` (flags: `e` enplanements, `h` airport branch, `o` area offset), runners `runner15.py`, `runner17.py` | `improve/out/multi_*.npy` |
| 10. Same allocator trained on target Jan–Nov (cost-of-strictness only, NOT zero-shot) | `improve/runner18.py` | `improve/out/multi_*_hist_*` |

## Strict evaluation and paper tables

| Paper element | Script |
|---|---|
| Level estimator (transit commuters, helper-only leave-one-city-out) and rhythm (month factor) | `improve/idea_avg_rhythm.py` (`strict_T`), `improve/strict_v2.py`, `improve/strict_total.py`, `improve/tests_round9.py` (ACS/LODES units) |
| Table 1 (data), Table 2 (main results), Table 4 (level ablation), rhythm ablation, true-total diagnostic | `improve/paper_numbers.py` |
| Table 3 (level predictor selection), day-level significance tests vs baselines | `improve/paper_extra.py` |
| Per-run paired tests (10 runs), partition-diversity tests | `improve/confirm10.py`, `improve/ens_test.py` |
| Table 6 (cost of strictness) | `improve/cost_final.py` |
| Poisson noise floor and true-pattern ceilings (diagnostic) | `improve/ceiling.py` |
| Simple baselines (uniform, jobs, jobs+residents, gravity, XGBoost) | `improve/baselines.py` (functions reused by `paper_numbers.py`) |

## Data sources (all public)

- Chicago Taxi Trips (City of Chicago Data Portal, dataset `wrvz-psew`), 2021.
- NYC TLC yellow-taxi trip records, 2021.
- SFMTA taxi trips (DataSF `m8hk-2ipk`), 2023.
- Open Data DC taxi trips, 2021.
- OpenStreetMap (Overpass API), LODES8 2021 (WAC/RAC), ACS 2020–2024 5-year (Census Reporter API; tables B08301, B25044, B01003),
  FAA CY2021 enplanements, OurAirports, Census cartographic tract boundaries (2021) and TIGER 2020 tracts.

## Requirements

Python 3.10, numpy, pandas, geopandas, shapely, scikit-learn, scipy, torch (CPU), xgboost, osmnx, requests.

## Notes

- `run_x.py` is the experiment driver used for all allocator variants; its docstring lists every variant and the date its
  decision rule was declared, before the corresponding runs.
- Variants that were tested and not adopted (see the paper, Section "Diagnostics and approaches that did not help") are also
  selectable by flags in `run_x.py`.
