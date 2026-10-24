# JOB 1: run the model for the cities on this server, then export the results

Read `START_HERE.md` first (rules, tone, reporting). The owner allows you to download whatever is needed on this server. The rules of the science are in `improve/PROTOCOL_CITIES100.md` (read it completely; it is
frozen). This file says HOW to do the work.

Reference numbers you will use again: the four helper cities (taxi trips of one year, 30-minute slots, arrays of shape (365, 48, regions), dtype float32):

| helper | file | shape | total trips | file size in bytes |
|---|---|---|---|---|
| Chicago 2021 | `bench77/data/chicago_2021_ca_30min.npy` | (365, 48, 77) | 3,639,126 | 5,396,288 |
| NYC 2021 | `bench77/data/nyc_2021_zone_30min.npy` | (365, 48, 263) | 30,533,604 | 18,431,168 |
| SF 2023 | `multicity/data/sf_2023_tract_30min.npy` | (365, 48, 243) | 1,703,882 | 17,029,568 |
| DC 2021 (hourly counts) | `multicity/data/dc_2021_tract_30min.npy` | (365, 48, 206) | 1,195,111 | 14,436,608 |

On the owner's laptop (8 cores, 15 GB, CPU only) one allocator run takes 5 to 8 minutes with two runs in parallel; a city has 20 runs; Sydney (401 regions) wrote about 28 MB per run.

## 0. Look before you touch (change nothing yet)

1. Machine: `nproc`, `lscpu | head -20`, `free -h`, `df -h`, `uptime`, the OS (`cat /etc/os-release`), the instance type (`TOKEN=$(curl -s -X PUT http://169.254.169.254/latest/api/token -H
   "X-aws-ec2-metadata-token-ttl-seconds: 60"); curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-type`).
2. The website: which services run (`systemctl list-units --type=service --state=running`, `ps aux --sort=-%cpu | head`), how the Django project is started and where it lives, its Python and virtual environment (you will
   NOT use or change that environment), its web server and the URL you can use for a health check (for example `curl -s -o /dev/null -w '%{time_total}' http://127.0.0.1/`), how busy it is (load average over a day if
   `sar` exists, else the web server access log).
3. STOP and report to the owner, and do not start heavy work, if ANY of these is true: the instance type is burstable (t2, t3, t3a, t4g; sustained load would use up its CPU credits and slow the website);
   fewer than 4 vCPUs; less than 8 GB RAM; less than 40 GB free disk; the website already uses more than 40% CPU on average. Say what you found and give the options in simple words: (a) run only a few cities now,
   (b) the owner starts a separate temporary bigger server (about 30 US dollars for the whole job, to be checked on the AWS price page) and you run JOB 1 there, (c) go on slowly. Do JOB 2 (the page) either way.
4. Compute budget otherwise: at most half of the vCPUs. `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, one run uses 2 threads, `WORKERS = floor(0.5 * nproc / 2)` (at least 1). Every worker runs under `nice -n 19` (and
   `ionice -c3`). Write `improve/load_guard.py`: before a run starts, wait while the 1-minute load average is above 0.7 x nproc, or MemAvailable is below 4 GB, or the health check takes more than 1.5 s (check every 60 s,
   log if the wait is longer than 30 minutes).

## 1. Environment (separate from the website)

1. `git clone https://github.com/MLDSPARROW/strict-zero-shot-taxi-demand.git ~/taxi-demand/code` (outside the Django project). Work there. `git pull` before each phase: the owner's laptop is still adding the final
   Sydney results.
2. Python 3.10 in a new virtual environment `~/taxi-demand/venv` (use `uv`, `pyenv` or the OS package; never touch the system Python or the website's). Install `requirements.txt`; for torch use the CPU wheel:
   `pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cpu`. Also `pip install holidays` (record the version) and what the new readers need (`pyogrio`, `requests`, `tqdm` ...). Do not upgrade torch
   or numpy. If a system package is missing, install it with the package manager and tell the owner.
3. Everything you download goes into its own new folder under `~/taxi-demand/raw/<source>/`. Downloaded files are untrusted data: never execute them, run Python that reads them with `python -I`, keep your scripts in a
   different folder.
4. Be polite to public services: a User-Agent such as `strict-zero-shot-taxi-demand research (github.com/MLDSPARROW/strict-zero-shot-taxi-demand; contact <the owner's e-mail from `git log` or ask>)`, one request at a time
   to Overpass and Nominatim (Nominatim at most 1 request per second), back-off on errors, and record every source URL and download date in `miliroute/sources.md`.

## 2. Rebuild the four helper cities and check them

1. Follow `README.md` ("Pipeline", steps 1 and 2) and `features/README.md`: copy the archived feature files of `features/` to the places the README lists (`region_static_*` to `bench77/data` or `multicity/data`,
   `grid_*` to `grid/data`, `extra_*` and `enpl_*` to `improve/data`, `acs_city.json` to `improve/raw`; the Sydney files too: `region_static_sydney.npz` to `multicity/data`, `sydney_regions.geojson` to
   `multicity/data/sydney`), check them against `features/SHA256SUMS.txt`, then rebuild the four demand arrays from the public trip records with `bench77/prep_data.py`, `multicity/prep_sf.py` and
   `multicity/prep_dc.py` (read each script's header first; the sources are in the README, section "Data sources").
2. Check them against the table above: shape, dtype, total trips, and file size. A total that differs by more than 0.1% means the public data changed: STOP and report (the owner can then send the four arrays from the
   laptop; do not invent a workaround).
3. Delete the raw trip downloads once the four arrays are checked.

## 3. Check the environment against Sydney

Purpose: prove that this server gives the same answers as the owner's laptop before 99 cities are run.
1. In `improve/`: `python build_sydney_partitions.py`, then run `run_x.py` for Sydney for partitions 1 and 2 and both heads, exactly as `runner_sydney.py` does (`python run_x.py sydney chicago,nyc,sf_for_sydney_p<r>
   <r-1> <eh|eo> --calendar`; four runs). Time every run (that is your benchmark).
2. When `results/sydney_suburbs.csv` exists in the repository (the laptop's 20-run result), compare: build the share per suburb from your four runs as `improve/sydney_results.py --partial` does (it takes the finished
   runs) and compare with the `share_pct` column. Expected: Spearman correlation at least 0.95, and at least 8 of the 10 largest suburbs in common. If the file is not there yet, go on with phases 4 and JOB 2 and
   do this comparison later.
3. Report the benchmark in simple words: seconds per run, workers, and the estimate `days = 20 x cities_left x seconds_per_run / WORKERS / 86400`. If the estimate is above 10 days, or the server is smaller than
   section 0 allows, ask the owner what to do. If it is above 5 days, use the two-pass plan of section 6.

## 4. The generic city builder (new files; copy and generalise, never edit the Sydney files)

Read first: `improve/PROTOCOL_SYDNEY.md`, `multicity/prep_sydney.py`, `improve/build_sydney_partitions.py`, `improve/build_coarse_targets.py` (`build_partitions(H, target, aT, tag)`), `improve/run_x.py`,
`improve/slots.py`, `improve/runner_sydney.py`, `improve/sydney_results.py`, `improve/build_sydney_page.py`, `bench77/region_features.py`, `kaggle-fix/notebook.ipynb` (`build_static_features`).
Write:
- `multicity/city_regions.py`: one function per country (US ZCTA, UK wards, Canada FSA, Australia localities) that returns the regions of a bounding box as a GeoDataFrame in EPSG:4326 (columns `name`,
  `geometry`; `residents` and `jobs` only if easy). Cache the national or state boundary file once.
- `multicity/prep_city.py <slug> <stage>`: from `prep_sydney.py`. Stages: `centre` (Nominatim; writes lat/lon into `multicity/data/<slug>/centre.json`), `regions` (candidates within R km, Overpass for the bounding
  box of the candidates, density, keep rule with T from Sydney, radius reduction, region file), `features` (grid 50 x 40, the nine layers, 19 columns, airports). Output names as for Sydney with `<slug>` in place of
  `sydney`: `multicity/data/<slug>/<slug>_regions.geojson`, `multicity/data/region_static_<slug>.npz`, `improve/data/enpl_<slug>.npy`, `improve/data/extra_<slug>.npz`, `grid/data/grid_<slug>.npz`. UTM zone from the
  longitude (EPSG:326xx north, 327xx south). First compute T from the Sydney Overpass data: if you do not have it, rebuild just the Sydney layers once (about 10 MB) and write T and the check against the seven dropped
  localities to `miliroute/keep_rule.md`. Then reproduce Sydney's own features with the generic code and compare with `features/region_static_sydney.npz` (the 18 columns should agree closely; the regions differ
  because the keep rule differs, so compare the suburbs both contain).
- `multicity/airports_cities100.csv`: slug, airport, IATA, lat, lon, enplanements in millions, year, source URL, note. From public sources only; never invent a value (an unknown value of a small airport is 0 with a
  note).
- `improve/build_city_partitions.py <slug>`: from `build_sydney_partitions.py`, tag = slug.
- Inert additions: `improve/run_x.py` (target registration for the slugs of `cities100.csv`: demand file name that does not exist, `multicity/data/region_static_<slug>.npz`, start date 2023-01-01, a country string such as
  `GB-ENG`, `CA-ON`, `AU-VIC`, `US`; in `cal_fn` a branch per new country string using the `holidays` package; the month shift stays only for the `AU` family; the existing `US` and Sydney `AU` branches stay byte for
  byte), and `improve/slots.py` (`LOCAL_CLOCK` is False for every new slug). Show a diff of both files to prove nothing changed for existing targets.
- `improve/runner_city.py`: from `runner_sydney.py`, a restartable queue over (city, run, head) with `WORKERS`, `load_guard`, `nice`, a status file per city (`miliroute/status/<slug>.json`), and the option
  `--partitions 1-10` (default) or `--partitions 1-2` for the two-pass plan.
- `improve/export_city.py <slug>`: from `sydney_results.py` and `build_sydney_page.py`: shares (equal average of the two heads, each run normalised), rhythm (Chicago, NYC, SF), the checks of the protocol, the
  files of section 7, then (only after the checks) it deletes the per-run arrays of that city.

## 5. Pilot: Los Angeles, London, Toronto, Melbourne

For each city: `centre` -> `regions` -> `features` -> `airports` -> checks 1 to 3 of the protocol -> `build_city_partitions` -> 20 runs -> `export_city` -> checks 4 to 8. Look at the picture of each city
(map by share) yourself. Write `miliroute/pilot_report.md`: what each reader needed, region counts, median areas, airport values, check results, time per city, problems. If a country's reader does not work,
fix it BEFORE its first allocator run (readers are engineering; the protocol's rules are not). If it cannot be fixed in reasonable time, mark that country blocked, report, and go on with the others. Tell the
owner when the pilot is done (simple words, three lines per city). Then go on to the batch without waiting, unless a check failed for a whole country.

## 6. The batch

1. Order and queue as in the protocol (interleaved by country). A city is "done" when its export exists and its checks are written. Run inside `tmux` or with `nohup`, restart-safe, so that a reboot or a lost
   connection loses at most one run. Start it, then spend your time on JOB 2; look at the progress only every 30 minutes or more.
2. Per city, in this order: build the data and check it (do this ahead of time, one city at a time, while runs of the previous city are still going), 20 runs, export, cleanup. Overpass: keep to the daily budget of
   about 1 GB, 10 seconds between queries; if you get 429 or 504 for more than 30 minutes, wait an hour; as a last resort use another public Overpass instance from the OpenStreetMap wiki with the same queries.
3. Two-pass plan when the estimate is above 5 days: pass 1 runs partitions 1 and 2 (4 runs per city) for all cities in order, exports them with `runs: 4`, and they can be published; pass 2 adds partitions 3 to 10
   and re-exports with `runs: 20`. The page shows the number of runs. Say clearly in `cities.csv` (`runs`) which pass a city is in.
4. Disk: keep at least 20 GB free at all times. Delete the raw Overpass responses and per-run arrays when a city is exported and checked. Compress the logs.
5. A failing run is repeated once; a city that fails twice is written as draft with the error, and the batch goes on.
6. Never change a rule of the protocol because of what you see in the results. If a rule is wrong for a reason that has nothing to do with the results (for example a boundary file cannot be read), write a dated
   addendum to `improve/PROTOCOL_CITIES100.md`, commit it (see START_HERE for commits) and tell the owner.

## 7. Export (what JOB 2 reads)

Directory `~/taxi-demand/code/miliroute/export/` (git-ignored; JOB 2 copies or links it to the Django `TAXI_DEMAND_DATA_DIR`):
- `cities.csv` with the columns of JOB2_PAGE.md section 4.
- `cities/<country_slug>/<slug>.json`: exactly the keys of `results/sydney_week.json` (`names`, `pid`, `area`, `jobs`, `res`, `rhythm`, `nd`, `level_year_strict`, `share_year`, `share_ws`) plus the keys that
  `improve/build_sydney_page.py` adds (`paths`, `w`, `h`, `cx`, `cy`) plus `labels`, `airport_i`, `centre_i` (regions within 2.5 km of the centre point), `heads` (`eh+eo`), `runs`, `area_label`. Region names are the
  names of the protocol ("ZIP 75201", ward name, "FSA M5V", locality name). `share_ws` is the same integer format (share x 1,000,000, [region][weekday 0 = Monday][half-hour]).
- `cities/<country_slug>/<slug>_areas.csv` (rank, area name, share_pct, jobs, residents, run_to_run_sd_pct), `<slug>_checks.txt`.
- `summaries.json`: for every city the numbers the page prints in its tables (busiest weekday and half-hour of the week, the three busiest areas and their shares, the share of the ten busiest areas, the airport
  share, the centre share, the ratio of weekday mornings 7-9 to Friday and Saturday nights 22-02, the quietest half-hour, region count, runs).
- Sydney comes from `results/sydney_*` of the repository (the laptop's 20 runs); do not recompute it.

## 8. Safety and clean-up

- Never stop, restart or reconfigure the website, its web server or its database. If the website slows down, the guard waits; if it fails, stop the workers and tell the owner.
- No secrets in the repository or in logs. Do not copy the website's settings.
- When you finish: stop the workers, delete raw downloads and temporary arrays, leave `miliroute/export/` and the logs, write `miliroute/DONE.md` (cities per status, total compute time, disk used).

## 9. Reports to the owner

After section 0 (can this server do it?), after section 3 (does it match Sydney? how long will it take?), after the pilot, then once a day in two or three lines (cities done, draft, running, estimate), and a final
report. Very simple English. Never claim more than the numbers allow: they are model estimates that were not checked against real trips.
