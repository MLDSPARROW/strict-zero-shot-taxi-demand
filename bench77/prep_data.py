"""Protocol-matched benchmark data :
  Chicago 2021 taxi pickups per COMMUNITY AREA (77) per 30-min slot   -> target (CHI-TAXI-like)
  NYC 2021 yellow-taxi pickups per TAXI ZONE (263) per 30-min slot      -> source
Chicago trip timestamps are rounded to 15 min by the city, so 30-min slots are exact.
Chicago counts are aggregated server-side (Socrata GROUP BY), month by month.
"""
import os, io, json, time
import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
D = f"{HERE}/data"
os.makedirs(D, exist_ok=True)
DAYS = pd.date_range("2021-01-01", "2021-12-31", freq="D")
N_DAYS, SLOTS = len(DAYS), 48


def slot_index(ts):
    ts = pd.to_datetime(ts)
    day = (ts.dt.normalize() - pd.Timestamp("2021-01-01")).dt.days.values
    slot = (ts.dt.hour.values * 2 + (ts.dt.minute.values >= 30)).astype(int)
    return day, slot


# ---------------- Chicago community areas ----------------
ca_path = f"{D}/chicago_community_areas.geojson"
if not os.path.exists(ca_path):
    r = requests.get("https://data.cityofchicago.org/resource/igwz-8jzy.geojson", timeout=300)
    r.raise_for_status()
    open(ca_path, "wb").write(r.content)
print("community areas geojson ok", flush=True)

# ---------------- Chicago 2021 demand ----------------
chi_path = f"{D}/chicago_2021_ca_30min.npy"
if not os.path.exists(chi_path):
    rows = []
    for m in range(1, 13):
        start = f"2021-{m:02d}-01T00:00:00"
        end = f"2022-01-01T00:00:00" if m == 12 else f"2021-{m+1:02d}-01T00:00:00"
        off = 0
        while True:
            params = {"$select": "trip_start_timestamp, pickup_community_area, count(*) AS n",
                      "$where": f"trip_start_timestamp >= '{start}' AND trip_start_timestamp < '{end}' AND pickup_community_area IS NOT NULL",
                      "$group": "trip_start_timestamp, pickup_community_area",
                      "$order": "trip_start_timestamp, pickup_community_area",
                      "$limit": 50000, "$offset": off}
            for attempt in range(5):
                try:
                    r = requests.get("https://data.cityofchicago.org/resource/wrvz-psew.json", params=params, timeout=600)
                    r.raise_for_status(); break
                except Exception as e:
                    print("  retry", m, off, e, flush=True); time.sleep(10 * (attempt + 1))
            else:
                raise RuntimeError(f"Chicago month {m} failed")
            part = r.json()
            rows.extend(part)
            off += len(part)
            if len(part) < 50000:
                break
        print(f"chicago month {m}: cumulative grouped rows {len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    df["n"] = df["n"].astype(int)
    df["ca"] = df["pickup_community_area"].astype(float).astype(int)
    day, slot = slot_index(df["trip_start_timestamp"])
    arr = np.zeros((N_DAYS, SLOTS, 77), dtype=np.float32)
    ok = (day >= 0) & (day < N_DAYS) & (df["ca"].values >= 1) & (df["ca"].values <= 77)
    np.add.at(arr, (day[ok], slot[ok], df["ca"].values[ok] - 1), df["n"].values[ok])
    np.save(chi_path, arr)
    print("chicago trips kept:", int(arr.sum()), "of", int(df["n"].sum()), flush=True)

# ---------------- NYC 2021 demand by taxi zone ----------------
nyc_path = f"{D}/nyc_2021_zone_30min.npy"
if not os.path.exists(nyc_path):
    arr = np.zeros((N_DAYS, SLOTS, 263), dtype=np.float32)
    for m in range(1, 13):
        url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2021-{m:02d}.parquet"
        for attempt in range(5):
            try:
                r = requests.get(url, timeout=900); r.raise_for_status(); break
            except Exception as e:
                print("  retry", m, e, flush=True); time.sleep(15 * (attempt + 1))
        else:
            raise RuntimeError(f"NYC month {m} failed")
        df = pd.read_parquet(io.BytesIO(r.content), columns=["tpep_pickup_datetime", "PULocationID"])
        day, slot = slot_index(df["tpep_pickup_datetime"])
        z = df["PULocationID"].values.astype(int)
        ok = (day >= 0) & (day < N_DAYS) & (z >= 1) & (z <= 263)
        np.add.at(arr, (day[ok], slot[ok], z[ok] - 1), 1)
        print(f"nyc month {m}: {int(ok.sum())} trips kept of {len(df)}", flush=True)
    np.save(nyc_path, arr)
print("done", flush=True)
