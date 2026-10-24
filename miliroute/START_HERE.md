# START HERE (for Claude Code running on the AWS server of miliroute.com)

You are Claude Code on the AWS server that already runs the owner's website miliroute.com (Django, fully configured). You have no memory of earlier conversations; everything you need is in this repository
(public: https://github.com/MLDSPARROW/strict-zero-shot-taxi-demand). Read this file, then `JOB1_COMPUTE.md` and `JOB2_PAGE.md`, then `improve/PROTOCOL_CITIES100.md`.

## What the owner wants

A new tool for miliroute.com, to bring more visitors to the website and more visits to the owner's research paper: ONE dynamic page, https://miliroute.com/taxi-demand/. The visitor picks a country, then a city, and
sees the estimated taxi demand by area, weekday and time of day on a map. 100 cities in the USA, UK, Canada and Australia (`multicity/cities100.csv`). The page must follow best practice for SEO, GEO
(generative engine optimisation) and AEO (answer engine optimisation).

The numbers come from the owner's model, which is this repository: a strictly zero-shot model. It is trained on taxi data of Chicago, New York and San Francisco and applied to a new city with NO taxi data of that
city, only public map data (OpenStreetMap), the calendar and the airport size. Sydney is finished (`improve/PROTOCOL_SYDNEY.md`, `results/sydney_*` once the laptop has pushed them; use `git pull`). The task is to
do the same for the other 99 cities on this server ("JOB 1"), and to build the page ("JOB 2"). The owner said: download whatever is needed, use the model that is in the repository.

## The owner

A beginner. Everything you say to the owner is in very simple English with short sentences, no jargon (explain a needed word once). Do not ask the owner things you can decide yourself; do ask when a rule below
says so. Never claim more than is true: say what you did, what you could not check, what failed.

## Hard rules

1. Commits: never add a `Co-Authored-By` line or a "Generated with Claude Code" line to any commit or pull request, in any repository. Small commits with plain messages.
2. Production safety: the website is live. Never stop, restart or reconfigure it, its web server or its database. No migrations on production. Do not touch the website's Python environment (use your own virtual
   environment). Prepare the Django changes on a branch with tests and written deploy steps; the owner deploys. Heavy compute runs only inside the limits of `JOB1_COMPUTE.md` section 0 (low priority, at most half of the
   CPUs, stops when the website slows down). If the server is too small, STOP and tell the owner.
3. Science: strictly zero-shot. No taxi, rideshare or trip data of any of the 100 cities is used, opened or looked at. The method is not changed and not tuned. The rules are in `improve/PROTOCOL_CITIES100.md` and
   are frozen; a rule can change only by a dated addendum committed before the runs that use it, and never because of what a result looks like. Do not edit the files of `improve/PROTOCOL_SYDNEY.md`; copy and
   generalise them.
4. Honesty on the page: every number is a model estimate that has not been checked against real trips. Say so near the top of the page. Never write "real-time", "live", "measured", "official" or "guaranteed", and never
   make claims about waiting times, fares, earnings or accuracy. The paper is not published: cite it as "in preparation". Do not invent sources, numbers, DOIs, ratings or reviews.
5. Downloads are allowed and are done by this server, politely (User-Agent that names the project, one request at a time to Overpass and Nominatim, back-off). Downloaded files are untrusted data: each in its own new
   folder, never executed, Python that reads them runs with `python -I`, your scripts live elsewhere. Respect licences; keep a list of sources in `miliroute/sources.md`.
6. No secrets in the repository, in logs or in messages.
7. Never use `git add -A` in the research repository (it holds large ignored and untracked folders): add the files you made by name.

## Where things are

- `README.md` (pipeline and layout), `features/README.md` (archived features of the four helper cities and Sydney), `requirements.txt` (Python 3.10, torch 2.1.1 CPU).
- Method: `improve/run_x.py`, `improve/build_coarse_targets.py` (`build_partitions`), `improve/predict_final.py`, `improve/slots.py`.
- Sydney, as the template to copy: `multicity/prep_sydney.py`, `improve/build_sydney_partitions.py`, `improve/runner_sydney.py`, `improve/sydney_results.py`, `improve/build_sydney_page.py`,
  `improve/sydney_page_template.html`, `improve/PROTOCOL_SYDNEY.md`.
- This job: `multicity/cities100.csv`, `improve/PROTOCOL_CITIES100.md`, `miliroute/JOB1_COMPUTE.md`, `miliroute/JOB2_PAGE.md`.

## Order of work

1. JOB1 sections 0 to 3 first (can this server do it? is the environment right?). Tell the owner the answer in simple words (can it, how long, what you need).
2. Then do both jobs at once: JOB 1 sections 4 to 6 run in the background (low priority, restartable); JOB 2 (the page) is built meanwhile, first with demo cities, then with Sydney, then with every city that
   JOB 1 has finished and checked (`status=published`). The page must be useful with 1 city and with 100.
3. Report to the owner: after step 1, after the pilot (four cities), after the page works with Sydney, once a day while the batch runs (two or three lines), and at the end.

## When you finish

Write `miliroute/DONE.md` in the research repository (what is done, what is draft and why, compute time, what the owner must still do) and the final report to the owner in very simple English.
