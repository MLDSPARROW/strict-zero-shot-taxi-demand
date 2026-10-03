# Scale-matched cross-city transfer for strictly zero-shot taxi demand prediction: code

Code accompanying the paper *"Scale-matched cross-city transfer for strictly zero-shot taxi demand prediction"* (M. Ghalejughi). Every number in the paper's tables is produced by the scripts listed below,
from saved model predictions.

**Strictness rule enforced throughout:** no taxi data of the target city is used for training, scaling, early stopping,
the city-wide total, or as input. Target data is read only by the evaluation code.

**Prediction and evaluation are separate steps** (`improve/PROTOCOL_REVISION.md`):

| Step | Script | Reads target demand? |
|---|---|---|
| Target time slots from the calendar | `improve/slots.py` | no |
| Allocator runs (`--calendar` flag) | `improve/run_x.py`, `multicity/run_multi.py` | no, the target demand file is not opened |
| Level, rhythm, shares, trip predictions of every method | `improve/predict_final.py` -> `improve/out_cal/pred_<city>.npz` | no |
| Metrics, 95% block-bootstrap intervals, paired daily differences, single-run mean and SD | `improve/evaluate_strict.py` | yes, only here |
| Step-by-step tests (native, scale-matched, ten partitions, average of heads), trip errors | `improve/scale_test.py` | yes (evaluation) |

For Chicago and NYC the calendar slots are identical to the slots of the earlier runs, and a rerun with `--calendar`
reproduces an earlier prediction exactly (maximum absolute difference 0.0). San Francisco and Washington, DC were
retrained with `--calendar` (`improve/runner_revision.py`).

## Derived features

`features/` archives the derived OpenStreetMap, LODES, FAA and ACS feature arrays with SHA-256 digests, so that the
models can be retrained without a new OpenStreetMap download (see `features/README.md`).

## Aggregation control

`improve/aggregate_control.py` builds the control helpers of `PROTOCOL_REVISION.md` part B (random grouping with the same
number of blocks, blocks of half and of twice the target median area, 10 partitions each). The runs were done on Kaggle
CPU workers (`kaggle/control_worker.py`); the NYC demand file was not part of the uploaded data.

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
| Figures 2–6 (maps, scale matching, week of demand, component bars, cost of strictness) | `improve/figures.py` |
| Poisson reference level and true-pattern ceilings (diagnostic) | `improve/ceiling.py` |
| Simple baselines (uniform, jobs, jobs+residents, gravity, XGBoost) | `improve/baselines.py` (functions reused by `paper_numbers.py`) |

## Held-out evaluation (San Francisco and Washington, DC)

`improve/PROTOCOL_HELDOUT.md` fixes the frozen method, helper rules and evaluation plan; it was written before any held-out run.
`improve/PROTOCOL_HELDOUT.timestamp` records the date and SHA-256 digest of each version of the protocol (the last addendum was
added before any held-out result was inspected). Runs: `improve/aggregate_heldout.py` (San Francisco re-aggregated to DC scale),
`improve/runner_heldout.py`; scoring for all four targets: `improve/score_all.py`, `improve/v4_numbers.py`,
`improve/heldout_ceiling.py`; four-city figure: `improve/fig_four.py`; sensitivity analyses: `improve/revision_extras.py`.

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
