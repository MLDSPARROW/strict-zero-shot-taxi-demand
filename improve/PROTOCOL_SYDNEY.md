# Sydney: a demand estimate per suburb, with no Sydney taxi data (an application of the final method; frozen BEFORE any allocator run for Sydney)

Written: 2026-10-24, after the suburb set and the map-data stage were built and before any allocator was trained or any prediction made for Sydney.
Committed and time-stamped (`PROTOCOL_SYDNEY.timestamp`) before those runs.

## Purpose and limits

The first author asked for the average taxi demand of every suburb of Sydney, by suburb and time, without Sydney taxi data. This is an application of the
final method v1 (strict zero-shot: Chicago, NYC and SF as helper cities, public map data of the target, the calendar). There is no Sydney demand file
anywhere in the pipeline, so there is nothing to score and no accuracy is claimed for Sydney. What can be said is limited to this:
- the allocation (which suburb gets which share, and when) is the part of the method that was tested on seven cities of the paper; Sydney is an eighth, untested city;
- the taxi system differs (rideshare and regulation in New South Wales, six-hour-reversed seasons, a different housing and transit structure), so a transfer from
  Chicago, NYC and SF can be wrong in ways the seven tested cities do not show;
- the yearly level is the weakest part (see "Level and its limits"); the display therefore lets the total be set by the user.
It is not part of the evidence of the paper. If it is used there, it is a labelled illustration, not a result.

## Public data (downloaded 2026-10-24; untrusted files kept in `multicity/data/sydney_raw/` and read with `python -I`)

| data | source | size | licence |
|---|---|---|---|
| suburb outlines (gazetted localities, GDA2020), box 150.20 to 151.45 E, -34.20 to -33.20 N | Geoscape Administrative Boundaries on data.gov.au, WFS | 18 MB | CC BY 4.0 |
| jobs by place of work and residents (ERP) in 2021, per travel zone (TZP24) | Transport for NSW Open Data Hub | 5 MB | CC BY |
| travel-zone outlines (TZ21) | Transport for NSW Open Data Hub | 26 MB | CC BY |
| shops, offices, transit stops, food, education, health, leisure, civic places, water | OpenStreetMap, Overpass API (one query at a time) | about 10 MB | ODbL, (c) OpenStreetMap contributors |
| public-transport commuters of Greater Sydney, Census 2021 (136,614 employed people, 5.6%) | ABS QuickStats | web page | CC BY 4.0 |

## Suburbs (regions)

All localities whose centroid lies within 25 km of Sydney Town Hall (geocoded by OpenStreetMap, -33.8732, 151.2062) and that have at least 200 jobs plus residents per km2
(`prep_sydney.py regions`). The radius makes the bounding box 37 km by 52 km (1,900 km2), the size of the New York City box, the largest area the allocator was trained on:
the map features divide distances by the box diagonal, so a much larger box would shift them out of the trained range. Outer Sydney (Penrith, Campbelltown, the Hawkesbury,
the northern beaches beyond 25 km) is therefore NOT covered. The density rule removes water and bushland localities (Botany Bay, Middle Harbour, North Harbour, Jibbon,
Duffys Forest, Rookwood) and one real suburb of low density, Kurnell (188 per km2). Result: 401 suburbs, median area 1.83 km2. Jobs and residents of 2021 are moved from the travel zones to the
suburbs by area overlap, each zone's weights normalised to one (as PROTOCOL_V2 addendum 1); they are used only for the density rule, for the checks and for the
display, not by the final method (heads eh and eo use only the 19 map columns and the airport value).

## Map features and the airport

Exactly as `prep_bike.py` / `prep_dc.py`: the notebook's `build_static_features` on a grid over the suburbs (here 50 rows along latitude by 40 columns, cells of about 1.0 by 0.9 km,
inside the range of the helpers, 0.07 to 0.92 km2), place layers from Overpass, suburb features area-weighted from the grid cells (`bench77/region_features.py`),
19 columns. Airport: the suburb that contains the reference point of Sydney Kingsford Smith (-33.9461, 151.1772); its value is 22.2 million enplanements, half of about 44 million
passengers in 2019 (from the airport's published traffic, not downloaded). With one airport suburb the hub head puts its learned airport mass there whatever the value, so the value only
enters as one input column; it is a typical-year value because the helpers' 2021 values are COVID-reduced and Sydney's 2021 borders were closed.

## Helpers and scale matching

Share helpers: Chicago, NYC and SF (taxi data of 30-minute resolution; DC is hourly and excluded, as for Austin and Porto). Each is re-aggregated to Sydney's median suburb area
(1.83 km2) with `build_sydney_partitions.py`, which calls the function `build_partitions` of `build_coarse_targets.py` unchanged (area-weighted k-means, K chosen by random state 0, ten
partitions with random states 1 to 10; a helper whose median region is already at least that large stays native).

## Method (the final method, unchanged)

Heads eh (airport hub mixture) and eo (log-area offset) with the hyper-parameters of `run_x.py`, ten runs each (run r uses partition r and seed r - 1), final share = equal average of
the two ten-run ensembles. `runner_sydney.py` runs it in calendar mode (`--calendar`): the target's demand file is never opened (none exists). Changes to `run_x.py`, inert for every other city:
a Sydney entry; for country "AU" the 12 New South Wales public holidays of 2023 (1 and 2 January, 26 January, Good Friday to Easter Monday, 25 April, 12 June, 2 October, 25 and 26
December; the bank-only holiday of 7 August left out) and the month shifted by six (`((m + 5) mod 12) + 1`), so that the season of the calendar features matches the helpers' (the
helper cities are in the northern hemisphere). `slots.py`: all 48 slots of the 365 days of 2023, no daylight-saving gap.

## Rhythm and level

Rhythm: the mean weekday-by-slot profile of Chicago, NYC and SF (`predict_final.py`); no month factor (the only same-year helper, SF 2023, has northern seasons).
Level: the strict rule of the paper: the unit is chosen by leave-one-helper-out among the helper cities, then the log-pooled ratio of trips per unit over Chicago, NYC, SF and DC is applied to
the Sydney value of that unit. The unit chosen by the helpers is public-transport commuters, and its Sydney value is 136,614 for Greater Sydney (ABS, Census 2021). Census day, 10 August
2021, was in the Sydney lockdown, when only 5.6% of employed people went to work by public transport, so the strict yearly total (about 2.3 million, 6,300 a day) is almost certainly far
too low: the NSW Point to Point Transport Commissioner reports 48.5 million point-to-point trips (taxi and rideshare) for 2021-22 and 68 million for 2022-23 (from a search summary of the
annual review; the report itself was not opened). The strict value is therefore shown only as one preset; the display lets the user type the daily total. Shares and rhythm do not depend on it.

## Outputs (`sydney_results.py`)

`results/sydney_suburbs.csv` (rank, share of the demand, jobs, residents, trips per day at the strict level), `results/sydney_week.json` (shares of every suburb by weekday and half-hour, the
rhythm and the strict level, for an interactive page), `results/sydney_checks.txt` (descriptive checks against jobs and residents; they cannot validate anything). The estimate is of taxi-like
pickups as recorded for the helper cities (taxi trips), not of rideshare.

## Disclosure

All choices above were made before any model output for Sydney: the radius (25 km, from the New York City box), the density threshold (200 per km2, set after seeing that the water localities
had 87 to 172 per km2 and every inhabited suburb except Kurnell had more), the grid, the airport value, the holiday list, the season shift, no month factor, the year 2023. No Sydney taxi,
rideshare or trip data of any kind was used or looked at. The earlier negative results on other ideas (PROTOCOL_V9, PROTOCOL_V10) do not enter: the final method is v1.

## Data as built (2026-10-24, before any allocator run for Sydney)

- Suburbs: 401 (`multicity/data/sydney/summary_regions.txt`), median area 1.830 km2, 1,072 km2 in total; jobs 1,985,015 and residents 3,206,965 of 2021 (zone mass allocated to the localities of the box: jobs 98.5%, residents 97.4%).
  Most jobs: Sydney, Parramatta, North Sydney, Macquarie Park, Mascot; most residents: Castle Hill, Auburn, Baulkham Hills, Bankstown, Merrylands.
- OpenStreetMap layers fetched (features): food 8,451; retail 10,287; office 1,752; transit 16,091; education 1,354; health 1,136; leisure 44,793; civic 640; water 1,306; no layer empty, no missing-data flag set.
  Grid 50 x 40, cell area 0.930 km2, box diagonal 62.29 km (New York City: 0.916 km2); airport suburb found: Mascot (22.2 million enplanements).
- Helpers (`results/sydney_partitions.log`): Chicago and NYC are left native, their median region areas being at least 1.83 km2 (a region is never split); SF is re-aggregated from 243 tracts to 67 blocks of median area 1.85 km2,
  ten partitions. Sources of run r: `chicago, nyc, sf_for_sydney_p<r>`. The rebuilt trip totals of the helper cities reproduce Table 1 of the paper exactly.
- Range check of the 18 map columns of Sydney against the pooled helpers (median and interquartile range of the helper regions, as the allocator scales them): no Sydney value lies outside the helpers' minimum and maximum
  except 0.2% of the "mix" column; the place-density columns sit about half an interquartile range below the helper median (fewer mapped places per km2), the other columns around the median. Nothing was changed after this check.
