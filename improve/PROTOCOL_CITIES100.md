# Taxi demand estimates for 100 cities of the USA, the UK, Canada and Australia, with no taxi data of any of them (an application of the final method v1; frozen BEFORE any download, code or allocator run for these cities)

Written: 2026-10-24. Sydney (PROTOCOL_SYDNEY.md) was still running its 20 allocator runs when this was written. Sydney is the first of the 100 cities, is covered by its own protocol and is NOT rerun here.
Committed and time-stamped (`PROTOCOL_CITIES100.timestamp`, with the digests of the method files as they are today) before any download, any new code or any run for the other 99 cities.

## Purpose and limits

The first author asked for the Sydney-type estimate (expected taxi pickups per area, weekday and half-hour, no local taxi data) for 100 cities, to be shown on ONE dynamic page of the website miliroute.com
(`miliroute/JOB2_PAGE.md`), to bring visits to the website and to the paper. This is an application of the final method v1. It is not evidence of the paper and no accuracy is claimed:
- no taxi, rideshare or trip data of any of the 100 cities is used, opened or looked at, not even to check an estimate;
- everywhere the numbers are shown they are labelled "model estimate, not checked against real trips";
- the method was developed and tested on cities with public taxi data; a transfer to the other countries is untested, and the yearly level of demand is its weakest part (see Level);
- if any of this is used in the paper, it is a labelled illustration that cites this protocol, never a result.

## Cities

`multicity/cities100.csv`: 50 US, 20 UK, 15 Canadian and 15 Australian cities, the largest by population of the metropolitan area, one per metropolitan area (a city that lies inside the study area of a bigger one is
covered there). New York City, Chicago, San Francisco, Washington DC and Austin are left out: real taxi data exist for them and they are the helper, development and held-out cities of the paper. The list was written
from general knowledge of city sizes before any data were looked at. The exact membership matters to nothing below; it can be extended by an addendum.

## Method (the final method v1, unchanged)

- Heads eh (airport hub mixture) and eo (log-area offset) with the hyper-parameters of `improve/run_x.py` as frozen in PROTOCOL_SYDNEY.md, ten runs each. Run r uses partition r (r = 1..10) and model seed r - 1.
- Share helpers: Chicago, NYC and SF, each re-aggregated to the median region area of the target with `build_partitions` of `improve/build_coarse_targets.py`, unchanged (K from k-means with random state 0, ten partitions
  with the random states 1 to 10; a helper whose median region area is already at least the target's stays native; a region is never split).
- Calendar mode (`--calendar`): the target has no demand file and none is opened. Final share = equal average of the eh and eo ten-run ensembles (each run normalised per slot).
- Rhythm: the mean weekday x half-hour profile of Chicago, NYC and SF (`improve/predict_final.py`), no month factor, as for Sydney.
- No line of the method is changed. New files only (a generic city builder, runner and exporter). `run_x.py` and `slots.py` may receive inert additions (new target names, holiday lists); a diff review must show
  that nothing changes for the existing targets. The files of PROTOCOL_SYDNEY.md (`prep_sydney.py`, `build_sydney_partitions.py`, `runner_sydney.py`, `sydney_results.py`, `build_sydney_page.py`) stay as they are; the
  generic versions are copies that are then generalised.

## Study area and regions

- Centre of a city: the OpenStreetMap (Nominatim) place centre of "<city>, <region>, <country>"; latitude and longitude go into `cities.csv`.
- Regions by country (name of a region in brackets): US ZIP code tabulation areas 2020 ("ZIP 12345", US Census Bureau cartographic boundary file); UK wards (ward name; ONS Open Geography UK-wide boundaries);
  Canada forward sortation areas 2021 ("FSA M5V", Statistics Canada cartographic boundary file); Australia gazetted localities (locality name; Geoscape Administrative Boundaries on data.gov.au, as for Sydney).
- Candidate regions: centroid within R km of the centre. R = 25 km, reduced in steps of 1 km while the bounding box of the kept regions is larger than 1,950 km2 (New York City 1,900 km2, Sydney 1,924 km2), so that
  the 50 x 40 grid of map features has cells of at most 0.95 km2, inside the range of the helpers (0.07 to 0.92 km2).
- Keep rule: a candidate region is kept if its density of OpenStreetMap places (the eight layers other than water, per km2, counted from the Overpass data of the city) is at least T. T is the first percentile of that
  density over the 401 Sydney suburbs of PROTOCOL_SYDNEY.md, computed once from the Sydney Overpass data and then used for every city. A region that contains the reference point of an airport is always kept.
  How many of the seven localities that PROTOCOL_SYDNEY.md dropped lie below T is reported (descriptive).
- A city with fewer than 60 or more than 1,500 kept regions is not published (status draft) and is reported.
- Substitution: if a named boundary set cannot be obtained or is unusable (for example most regions are slivers below 0.05 km2), an equivalent official set of similar scale may be used. It is written as a dated
  addendum below and committed before the first run of any city of that country.
- Residents and jobs of a region are display-only and optional (null when not quickly available from the same official source). They are used neither by the method nor by the keep rule.

## Map features and airports

- Exactly as `multicity/prep_sydney.py`, generalised (its Sydney values must be reproduced): the nine Overpass layers with the same tag sets, the notebook's `build_static_features` (`kaggle-fix/notebook.ipynb`) on a grid of
  50 x 40 cells over the kept regions (50 cells along the longer side), area-weighted to the regions (`bench77/region_features.py`), 19 columns. Overpass: one query at a time, a User-Agent that names the project and
  gives a contact address, back-off on 429 and 504, the main instance overpass-api.de first.
- Airports: every airport with scheduled passenger service whose reference point lies in a kept region. Value = enplanements of a typical year (FAA CY2019 in the US; half of the annual passengers of 2019 where only
  passengers are published), from the airport's authority or national statistics, with the source written in `multicity/airports_cities100.csv`. An airport below 5 million enplanements is set to 0: the smallest helper
  airport has 7.7 million, and the learned airport mass of the hub head belongs to large airports. If no region has a value above 0 the code skips the hub branch (`run_x.py`, `if HUB and (self.hub > 0).any()`); nothing else changes.

## Calendar

Year 2023, local clock time, all 48 slots on all 365 days (no daylight-saving gap, as Sydney). US: the federal holidays, as the helpers. UK: the public holidays of England, Scotland, Wales or Northern Ireland as the
city requires. Canada: those of the province. Australia: those of the state (New South Wales stays as hard-coded for Sydney). Source: the Python package `holidays` (version written to the logs), `subdiv` by nation,
province or state. Australia: the month feature is shifted by six months as for Sydney; no shift elsewhere.

## Level

The yearly level is NOT part of what the page promises: `total_reliable` is false for every city and no default total is shown. If the strict rule of the paper (public-transport commuters x trips per commuter) can be
applied quickly from official data of a year before 2020, its value goes into the export as `level_year_strict` for the optional "type your own total" control; otherwise it is null. No time is spent on it.

## Checks (descriptive; none is an accuracy test; none may be used to change a rule, a number or the model)

1. 60 to 1,500 kept regions; the bounding box at most 1,950 km2; the grid cell between 0.07 and 0.95 km2.
2. No map layer empty; no NaN; at most 5% of the values of the 18 map columns outside the minimum and maximum of the pooled helper regions (Sydney: 0.2%).
3. Every airport with a value above 0 found in a kept region.
4. All 20 runs present, every slot's shares sum to 1 within 1e-6, no NaN.
5. Run-to-run stability: the mean standard deviation of a region's share over the runs divided by the mean share is at most 0.35 (Sydney: about 0.16).
6. Sanity: the Spearman correlation of the yearly share with the OpenStreetMap place density is at least 0.3 (this guards against misaligned features; it says nothing about accuracy).
7. The exported map renders (paths not empty, all regions inside the box, north up); one picture per city is looked at.
8. A city that fails a check is written as status draft with the reason, is not published, and is not tuned until it passes.

## Outputs

Per city (in `miliroute/export/`, not committed to this repository unless the owner asks): `<country>/<slug>.json` (the format of `results/sydney_week.json` plus the map keys of `build_sydney_page.py`, plus `labels`,
`airport_i`, `centre_i`, `heads`, `runs`, `area_label`), `<country>/<slug>_areas.csv`, `<slug>_checks.txt`; one row per city in `cities.csv`; summaries for the page. The per-run prediction arrays are deleted after
the export has been validated. The logs of each city are kept.

## Freezing and disclosure

- Everything above is fixed before any run. For a city the data are built and checked before its first allocator run; if the check shows a defect (broken geometry, a wrong coordinate system) the defect is fixed and the
  city rebuilt. No decision may depend on a share, a rhythm or any model output of any of the 99 cities. A change of a rule is a dated addendum below, committed before the runs that use it.
- Run order: the order of `cities100.csv`, interleaved by country (US 1, UK 1, Canada 1, Australia 2, US 2, ...), after a pilot of Los Angeles, London, Toronto and Melbourne.
- Disclosure: the radius and box rule come from the Sydney and New York City boxes; T comes from the Sydney Overpass data; the airport threshold of 5 million comes from the range of the helpers; the boundary sets were
  named from general knowledge of what the four countries publish. No taxi, rideshare or trip data of any of the 100 cities was used or looked at. The earlier negative results (PROTOCOL_V9, PROTOCOL_V10) do not enter:
  the final method is v1.

## Sources and licences (the attribution text goes on the page)

OpenStreetMap contributors (ODbL); US Census Bureau (public domain); Office for National Statistics and Ordnance Survey data (Open Government Licence v3.0, Crown copyright); Statistics Canada (Open Government
Licence - Canada); Geoscape Australia and the Australian Bureau of Statistics (CC BY 4.0); FAA, UK Civil Aviation Authority, Statistics Canada and the airports, BITRE for airport traffic (public statistics, source and year
per row). The agent writes the exact attribution string per country into `cities.csv`.

## Addenda (dated; each committed before the runs that use it)

**Addendum 1 (2026-10-24, after the final Sydney result, before any download, code or run for the 99 cities).** Check 5 gave Sydney "about 0.16". That figure came from a preview with two runs. With all ten runs
per head the Sydney value is 0.28 (standard deviation of a suburb's single-run share 0.0695 percentage points, mean share 0.249%; see PROTOCOL_SYDNEY.md, Result). Check 5 now reads: the mean standard deviation
of a region's share between single runs (average of the two heads), divided by the mean share, is at most 0.56, twice the Sydney value. No other rule changes.

**Addendum 2 (2026-10-24, on the server, before any Overpass download, region selection or allocator run for the 99 cities; no output of any of them exists).**
Engineering details that the text above leaves open or states inconsistently. None depends on a result.
1. Box limit. 1,950 km2 contradicts check 1 (grid cells of at most 0.95 km2): 1,950 km2 / 2,000 cells = 0.975 km2. The purpose stated above is cells of
   at most 0.95 km2, so the radius is reduced while the box is larger than **1,900 km2** (= 0.95 x 2,000). The box is measured as the notebook measures
   its grid (`grid_geometry`: haversine height x width of the latitude/longitude box of the kept regions). Measured this way the box of the 401 Sydney
   suburbs is about 1,861 km2 (the text above says 1,924; descriptive only).
2. Grid orientation: 50 cells along the longer side of the box measured in metres (UTM), 40 along the shorter side (Sydney: 50 rows along latitude).
3. Keep rule counting: a place is an element of one of the eight notebook layers (Overpass, `out center`) whose point lies inside the region polygon;
   each layer is counted separately (as the features count them). T = numpy's default (linear) first percentile over the 401 Sydney suburbs.
   The Overpass layers of a city are downloaded once for the box of its candidate regions (centroid within 25 km); the grid features use the
   elements inside the grid box from the same download.
4. Downtown point of the distance feature: the city centre of the protocol (Nominatim place centre of "<city>, <region>, <country>"), as Sydney used
   Sydney Town Hall.
5. Region names that occur twice within a city: UK ward name + " (" + local authority name + ")"; Australian locality name + " (" + LOC_PID + ")".
6. Boundary versions: US ZCTA 2020 cartographic 1:500k (cb_2020_us_zcta520_500k); UK Wards (December 2024) Boundaries UK BGC (ONS); Australia
   Geoscape localities GDA2020, release August 2026 (all states, the neighbouring state's file added when the box crosses a border); Canada FSA 2021
   cartographic (Statistics Canada). GDA2020 is used as WGS84 (difference about 1.5 m), as for Sydney.
7. Jobs, residents and rail stations of the new cities are not built (the final method, heads eh and eo, does not use them); `extra_<slug>.npz` holds
   zeros, as the rail column of Sydney.
