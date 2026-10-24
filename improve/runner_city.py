"""PROTOCOL_CITIES100.md + miliroute/JOB1_COMPUTE.md sections 4 to 6: a restartable queue over the 99 cities (copied from runner_sydney.py and generalised).
Per city, in the order of the protocol (rank in the country, then US, UK, Canada, Australia; the first four are the pilot Los Angeles, London, Toronto, Melbourne):
  prepare : prep_city.py (centre, regions, features; Overpass one query at a time), checks 1 to 3 (a city that fails is written as draft, no runs),
            build_city_partitions.py (helpers scaled to the city's median region area)
  runs    : 10 partitions x heads eh and eo = 20 runs of run_x.py in calendar mode, one at a time, below-normal priority, after load_guard says the server is quiet
  export  : export_city.py (checks 4 to 8, files for the website, per-run arrays deleted), then deploy_to_site.sh (copy, import, image)
The data of the next city are prepared in a background thread while the runs of the current one go. Everything is restart-safe: finished runs are reused by run_x.py,
finished cities (published or draft in miliroute/status/<slug>.json) are skipped. Create ~/taxi-demand/STOP to pause after the current run.
Usage (from improve/): python -I runner_city.py [--partitions 1-10] [--only slug,slug] [--limit N]"""
import json
import os
import subprocess
import sys
import threading
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.join(HERE, "..")
STATUS = f"{ROOT}/miliroute/status"
PY = sys.executable
LOGS = os.path.expanduser("~/taxi-demand/logs/cities")
STOP = os.path.expanduser("~/taxi-demand/STOP")
DEPLOY = os.path.expanduser("~/taxi-demand/scripts/deploy_to_site.sh")
NPROC = os.cpu_count() or 1
WORKERS = max(1, (NPROC // 2) // 2)                      # floor(0.5 x nproc / 2), at least 1; this server: 1
ENV = dict(os.environ, OMP_NUM_THREADS="2" if NPROC >= 8 else "1", MKL_NUM_THREADS="2" if NPROC >= 8 else "1", OPENBLAS_NUM_THREADS="1", LOKY_MAX_CPU_COUNT="1")
NICE = ["nice", "-n", "19", "ionice", "-c3"]
COUNTRY_ORDER = {"United States": 0, "United Kingdom": 1, "Canada": 2, "Australia": 3}


def city_srcs(slug, r):
    """chicago, nyc, sf: the re-aggregated version of partition r when build_city_partitions.py made one, else native."""
    return [f"{H}_for_{slug}_p{r}" if os.path.exists(f"{HERE}/data/agg/{H}_for_{slug}_p{r}_Y.npy") else H for H in ("chicago", "nyc", "sf")]


def queue():
    df = pd.read_csv(f"{ROOT}/multicity/cities100.csv")
    df = df[df["slug"] != "sydney-nsw"].copy()
    df["c"] = df["country"].map(COUNTRY_ORDER)
    return df.sort_values(["rank_in_country", "c"])["slug"].tolist()


def log(slug, msg):
    os.makedirs(LOGS, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}Z {msg}"
    print(f"[{slug}] {line}", flush=True)
    with open(f"{LOGS}/{slug}.log", "a") as f:
        f.write(line + "\n")


def status(slug, **kw):
    os.makedirs(STATUS, exist_ok=True)
    p = f"{STATUS}/{slug}.json"
    st = json.load(open(p)) if os.path.exists(p) else {"slug": slug}
    st.update(kw, updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    json.dump(st, open(p, "w"), indent=1)
    return st


def get_status(slug):
    p = f"{STATUS}/{slug}.json"
    return json.load(open(p)) if os.path.exists(p) else {}


def run(cmd, slug, what, cwd=HERE, tries=2, env=None):
    for k in range(tries):
        t0 = time.time()
        with open(f"{LOGS}/{slug}.log", "a") as lf:
            lf.write(f"--- {what} (attempt {k + 1}) {time.strftime('%FT%TZ', time.gmtime())}\n")
            lf.flush()
            r = subprocess.run(cmd, cwd=cwd, env=env or ENV, stdout=lf, stderr=subprocess.STDOUT)
        if r.returncode == 0:
            return time.time() - t0
        log(slug, f"{what} failed (exit {r.returncode}), attempt {k + 1} of {tries}")
        time.sleep(30)
    raise RuntimeError(f"{what} failed {tries} times; see {LOGS}/{slug}.log")


def draft(slug, reason):
    """Write the city as draft (CSV row for the page) and stop working on it."""
    import export_city as ec
    r = ec.city_row(slug)
    meta = {"slug": slug, "name": r["city"], "country": r["country"], "country_slug": ec.COUNTRY_SLUG[r["country"]], "region": ec.region_display(r),
            "area_label": ec.AREA_LABEL[r["country"]], "lat": "", "lon": "", "timezone": ec.timezone(r), "locale": ec.LOCALE[r["country"]], "calendar_year": 2023, "areas": "",
            "median_area_km2": "", "runs": 0, "heads": "eh+eo", "model_version": ec.MODEL_VERSION, "built_on": time.strftime("%Y-%m-%d", time.gmtime()), "status": "draft",
            "reason": reason[:600], "total_reliable": "false", "default_total_per_day": "", "attribution": ec.ATTRIBUTION[r["country"]], "wikidata_id": ""}
    try:
        c = json.load(open(f"{ROOT}/multicity/data/{slug}/centre.json"))
        meta["lat"], meta["lon"] = round(c["lat"], 5), round(c["lon"], 5)
    except OSError:
        pass
    ec.write_meta(meta)
    status(slug, state="draft", reason=reason)
    log(slug, f"DRAFT: {reason}")


def wait(slug, reason, hours=6):
    """A problem that is not the city's fault (a download, a missing source file, a bug of ours): not a draft, tried again later."""
    status(slug, state="waiting", reason=reason, retry_after=time.time() + 3600 * hours)
    log(slug, f"WAITING (retry in {hours} h): {reason}")


def prepare(slug):
    """Data of one city, checks 1 to 3, helper partitions. Returns None when ready, or a reason string when the city is a draft."""
    st = get_status(slug)
    if st.get("state") in ("prepared", "running", "exporting", "published", "draft"):
        return None if st.get("state") != "draft" else st.get("reason", "draft")
    status(slug, state="preparing")
    log(slug, "preparing (centre, regions, features)")
    mc = f"{ROOT}/multicity"
    c = pd.read_csv(f"{mc}/cities100.csv").set_index("slug").loc[slug]
    if c["country"] == "Canada" and not __import__("glob").glob(os.path.expanduser("~/taxi-demand/raw/ca_fsa/lfsa000*21a_e.zip")):
        return "WAIT: the Statistics Canada FSA boundary file is not on the server yet (~/taxi-demand/raw/ca_fsa/)"
    try:
        run(NICE + [PY, "-I", "prep_city.py", slug, "all"], slug, "prep_city", cwd=mc)
    except RuntimeError as e:
        return f"WAIT: {e}"
    import export_city as ec
    import geopandas as gpd
    reg = gpd.read_file(f"{mc}/data/{slug}/{slug}_regions.geojson").set_crs(4326, allow_override=True)
    info = json.load(open(f"{mc}/data/{slug}/summary_regions.json"))
    feats = json.load(open(f"{mc}/data/{slug}/summary_features.json"))
    lines, fails = ec.common_checks(slug, reg, info, feats, None, ec.airport_rows(slug, reg))
    for ln in lines:
        log(slug, ln)
    if fails:
        return "; ".join(fails)
    log(slug, "building helper partitions")
    run(NICE + [PY, "-I", "build_city_partitions.py", slug], slug, "build_city_partitions")
    status(slug, state="prepared", regions=int(len(reg)), median_area_km2=info["median_area_km2"], radius_km=info["radius_km"])
    return None


def main():
    argv = sys.argv[1:]
    a, b = (1, 10)
    if "--partitions" in argv:
        a, b = map(int, argv[argv.index("--partitions") + 1].split("-"))
    only = argv[argv.index("--only") + 1].split(",") if "--only" in argv else None
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else None
    runs = list(range(a, b + 1))
    from load_guard import wait_until_quiet
    todo = [s for s in (only or queue()) if get_status(s).get("state") not in ("published", "draft")
            and get_status(s).get("retry_after", 0) < time.time()]
    if limit:
        todo = todo[:limit]
    log("queue", f"{len(todo)} cities to do; partitions {a}-{b}; workers {WORKERS}")
    pending = {}                                         # slug -> (thread, result list) of the background preparation

    def start_prepare(slug):
        res = []
        t = threading.Thread(target=lambda: res.append(_safe_prepare(slug)), daemon=True)
        t.start()
        pending[slug] = (t, res)

    def _safe_prepare(slug):
        try:
            return prepare(slug)
        except Exception as e:                           # a broken city must never stop the batch
            return f"WAIT: {type(e).__name__}: {e}"[:300]

    for k, slug in enumerate(todo):
        if os.path.exists(STOP):
            log("queue", f"{STOP} exists: pausing before {slug}")
            return
        if slug not in pending:
            start_prepare(slug)
        t, res = pending[slug]
        t.join()
        if k + 1 < len(todo):
            start_prepare(todo[k + 1])                   # the next city is prepared while this one runs
        reason = res[0] if res else "WAIT: preparation produced no result"
        if reason and reason.startswith("WAIT:"):
            wait(slug, reason[5:].strip())
            continue
        if reason:
            draft(slug, reason)
            continue
        status(slug, state="running", run_seconds=get_status(slug).get("run_seconds", 0))
        jobs = [(r, h) for r in runs for h in ("eh", "eo")]
        fail = None
        for i, (r, h) in enumerate(jobs, 1):
            outp = f"{HERE}/out_cal/multi_{slug}_from_{'+'.join(city_srcs(slug, r))}_v{h}_seed{r - 1}.npy"
            if os.path.exists(outp):
                continue
            while os.path.exists(STOP):
                time.sleep(60)
            wait_until_quiet(lambda m: log(slug, m))
            srcs = ",".join(city_srcs(slug, r))
            try:
                sec = run(NICE + [PY, "-I", "run_x.py", slug, srcs, str(r - 1), h, "--calendar"], slug, f"run {i}/{len(jobs)} r={r} {h}")
            except RuntimeError as e:
                fail = str(e)
                break
            st = get_status(slug)
            status(slug, state="running", runs_done=i, runs_total=len(jobs), run_seconds=round(st.get("run_seconds", 0) + sec))
        if fail:
            draft(slug, f"allocator run failed twice: {fail}")
            continue
        status(slug, state="exporting")
        try:
            run(NICE + [PY, "-I", "export_city.py", slug, "--runs", str(b)], slug, "export_city", tries=1)
        except RuntimeError as e:
            wait(slug, f"export failed (the runs are kept): {e}", hours=1)
            continue
        meta = json.load(open(f"{ROOT}/miliroute/export/_meta/{slug}.json"))
        status(slug, state=meta["status"], reason=meta.get("reason", ""))
        log(slug, f"exported as {meta['status']}" + (f": {meta['reason']}" if meta["reason"] else ""))
        try:
            subprocess.run(["bash", DEPLOY], check=True, stdout=open(f"{LOGS}/deploy.log", "a"), stderr=subprocess.STDOUT, timeout=900)
        except Exception as e:
            log(slug, f"deploy to the website failed (will be retried with the next city): {e}")
    write_done()


def write_done():
    """miliroute/DONE.md when the whole queue has a final state."""
    st = {s: get_status(s) for s in queue()}
    if any(v.get("state") not in ("published", "draft") for v in st.values()):
        return
    pub = [s for s, v in st.items() if v["state"] == "published"]
    dr = [(s, v.get("reason", "")) for s, v in st.items() if v["state"] == "draft"]
    sec = sum(v.get("run_seconds", 0) for v in st.values())
    lines = ["# DONE: taxi demand estimates for the 100 cities", "", f"Written automatically {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} when the last city finished.", "",
             f"- Sydney: converted from the laptop's 20 runs (results/sydney_*).", f"- Published: {len(pub)} of 99 other cities.", f"- Draft: {len(dr)}.",
             f"- Compute time of the allocator runs: {sec / 3600:.1f} hours on one thread of a 2-CPU server.", ""]
    if dr:
        lines += ["## Drafts (not on the page) and why", ""] + [f"- {s}: {r}" for s, r in dr] + [""]
    lines += ["## What the owner must still do", "", "See docs/taxi_demand.md of the website repository (Search Console, IndexNow, Zenodo DOI, co-authors, licence)."]
    open(f"{ROOT}/miliroute/DONE.md", "w").write("\n".join(lines) + "\n")
    log("queue", "all cities have a final state; miliroute/DONE.md written")


if __name__ == "__main__":
    main()
