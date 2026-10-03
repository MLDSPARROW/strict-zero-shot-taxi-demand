# Archived derived features

OpenStreetMap changes over time, so the derived feature arrays used in the paper are archived here. With these files the
models can be retrained without a new OpenStreetMap download. `SHA256SUMS.txt` lists the digest of every file.

| File | Content | File written on |
|---|---|---|
| `region_static_<city>.npz` | 18 region features from OpenStreetMap (`X`), region ids, centroids in km, areas in km² | chicago, nyc 2026-09-27; sf 2026-09-29; dc 2026-09-30 |
| `grid_<city>.npz` | fine-grid map features and region overlap fractions, used to recompute block features after re-aggregation | 2026-10-01 |
| `extra_<city>.npz` | LODES 2021 jobs and resident workers, rail stations from OpenStreetMap, per region | |
| `enpl_<city>.npy` | FAA CY2021 enplanements of the large airport inside each region (0 elsewhere) | |
| `acs_city.json` | ACS 2020 to 2024 five-year city estimates (Census Reporter API) used for the annual level | |

The OpenStreetMap data were retrieved through the Overpass API between 27 September and 1 October 2026. The raw Overpass
responses were not kept, so these derived arrays are the reference snapshot. To use them, copy `region_static_*` to
`bench77/data` (Chicago, NYC) or `multicity/data` (SF, DC), `grid_*` to `grid/data`, `extra_*` and `enpl_*` to
`improve/data` and `acs_city.json` to `improve/raw`.

The taxi demand arrays are not included. They are rebuilt from the public trip records by `bench77/prep_data.py` and
`multicity/prep_sf.py`, `multicity/prep_dc.py`.
