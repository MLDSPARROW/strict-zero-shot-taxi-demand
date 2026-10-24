"""PROTOCOL_CITIES100.md, "Map features and airports": writes multicity/airports_cities100.csv. Public sources only; no value is invented (an airport whose value is not
looked up because it is far below the threshold has 0 and a note). prep_city.py puts an airport into the region that contains its reference point and applies the rule of the
protocol (below 5 million enplanements = 0).
  US : FAA calendar-year 2019 enplanements of every commercial service airport (cy19-commercial-service-enplanements.xlsx), reference point from OurAirports (public domain)
  UK : UK Civil Aviation Authority 2019 annual airport data, Table 01 (terminal passengers); enplanements = half of the passengers
  CA/AU : airport authorities' 2019 reports (URLs below); enplanements = half of the passengers
Columns: slug, airport, iata, lat, lon, enplanements_m, year, source_url, note.   Usage: python -I make_airports_cities100.py"""
import math
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.expanduser("~/taxi-demand/raw")
CAA = "https://www.caa.co.uk/data-and-analysis/uk-aviation-market/airports/uk-airport-data/uk-airport-data-2019/annual-2019"
FAA = "https://www.faa.gov/sites/faa.gov/files/airports/planning_capacity/passenger_allcargo_stats/passenger/cy19-commercial-service-enplanements.xlsx"
RADIUS_KM = 45.0                         # an airport is listed for a city when its reference point is within this distance of the city centre; the keep rule decides the rest


def hav(a1, o1, a2, o2):
    p1, p2 = math.radians(a1), math.radians(a2)
    x = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(o2 - o1) / 2) ** 2
    return 12742 * math.asin(math.sqrt(x))


oa = pd.read_csv(f"{RAW}/ourairports/airports.csv", low_memory=False)
oa = oa[oa["iata_code"].notna()].drop_duplicates("iata_code").set_index("iata_code")


def ref(iata):
    r = oa.loc[iata]
    return float(r["latitude_deg"]), float(r["longitude_deg"]), r["name"]


rows = []      # (airport, iata, lat, lon, enplanements_m, year, source, note)
faa = pd.read_excel(f"{RAW}/faa/cy19-commercial-service-enplanements.xlsx")
for _, r in faa.iterrows():
    iata = str(r["Locid"]).strip()
    if iata in oa.index:
        lat, lon, name = ref(iata)
        rows.append((str(r["Airport Name"]).strip().title(), iata, lat, lon, round(float(r["CY 19 Enplanements"]) / 1e6, 4), 2019, FAA, ""))
caa = pd.read_csv(f"{RAW}/uk_caa/caa_2019_table_01_size_of_uk_airports.csv", encoding="utf-8-sig")
UK = {"HEATHROW": "LHR", "GATWICK": "LGW", "MANCHESTER": "MAN", "STANSTED": "STN", "LUTON": "LTN", "EDINBURGH": "EDI", "BIRMINGHAM": "BHX", "BRISTOL": "BRS", "GLASGOW": "GLA",
      "BELFAST INTERNATIONAL": "BFS", "NEWCASTLE": "NCL", "LONDON CITY": "LCY", "LIVERPOOL (JOHN LENNON)": "LPL", "EAST MIDLANDS INTERNATIONAL": "EMA", "LEEDS BRADFORD": "LBA",
      "ABERDEEN": "ABZ", "BELFAST CITY (GEORGE BEST)": "BHD", "SOUTHEND": "SEN", "SOUTHAMPTON": "SOU", "CARDIFF WALES": "CWL", "DONCASTER SHEFFIELD": "DSA"}
for _, r in caa.iterrows():
    if r["airport_name"] in UK and UK[r["airport_name"]] in oa.index:
        iata = UK[r["airport_name"]]
        lat, lon, name = ref(iata)
        rows.append((str(r["airport_name"]).title(), iata, lat, lon, round(int(r["this_year_pax"]) / 2e6, 4), 2019, CAA,
                     f"UK CAA 2019 terminal passengers {int(r['this_year_pax']):,}; enplanements = half"))
OTHER = [  # airport, iata, passengers 2019 (millions), source, note
    ("Toronto Pearson International", "YYZ", 50.5, "https://www.torontopearson.com/ar2019/_downloads/GTAA-MDA-Financial-2019.pdf", "GTAA annual report 2019, management's discussion: 50.5 million passengers"),
    ("Vancouver International", "YVR", 26.3, "https://news.yvr.ca/yvr-marks-second-highest-passenger-count-in-airport-history-breaks-cargo-record-in-2024/", "Vancouver Airport Authority statement: 26.3 million travellers in 2019"),
    ("Montreal-Trudeau International", "YUL", 20.3, "https://www.wallstreetitalia.com/?p=666146", "20.3 million in 2019 from secondary reporting (the full-year 2019 release of Aeroports de Montreal was not located); 2018: 19.4 million per ADM"),
    ("Calgary International", "YYC", 18.0, "https://www.yyc.com/News/tabid/91/articleType/ArticleView/articleId/377/A-record-18-million-guests-travelled-through-YYC-in-2019-YYC-a-accueilli-un-nombre-record-de-18-millions-de-voyageurs-en-2019.aspx", "Calgary Airport Authority: 18 million in 2019"),
    ("Melbourne", "MEL", 37.4, "https://www.melbourneairport.com.au/corporate/melbourne-airport-passenger-performance-fy19-20", "financial year 2018-19 (July to June), 37.4 million passengers excluding transits"),
    ("Brisbane", "BNE", 23.78, "https://www.bne.com.au/sites/default/files/docs/BAC-Annual-Report_2019.pdf", "financial year 2018-19: 23,782,623 passengers (BAC annual report 2019)"),
    ("Perth", "PER", 14.5, "https://www.perthairport.com.au/-/media/Files/CORPORATE/About-us/Reports-and-publications/Perth-Airport-Annual-Report-2019.pdf", "financial year 2018-19: 14.5 million passengers (Perth Airport annual report)"),
]
for name, iata, pax, src, note in OTHER:
    lat, lon, _ = ref(iata)
    rows.append((name, iata, lat, lon, round(pax / 2, 4), 2019, src, note + "; enplanements = half"))
SMALL = [("Edmonton International", "YEG"), ("Ottawa Macdonald-Cartier International", "YOW"), ("Winnipeg James Armstrong Richardson International", "YWG"), ("Halifax Stanfield International", "YHZ"),
         ("Adelaide", "ADL"), ("Gold Coast", "OOL"), ("Canberra", "CBR"), ("Sunshine Coast", "MCY"), ("Newcastle", "NTL"), ("Hobart", "HBA"), ("Cairns", "CNS"), ("Townsville", "TSV"), ("Darwin", "DRW"),
         ("Victoria International", "YYJ"), ("Saskatoon John G. Diefenbaker International", "YXE"), ("Quebec City Jean Lesage International", "YQB"), ("Hamilton John C. Munro International", "YHM"),
         ("Region of Waterloo International", "YKF"), ("London International", "YXU"), ("Windsor International", "YQG")]
for name, iata in SMALL:
    lat, lon, _ = ref(iata)
    rows.append((name, iata, lat, lon, 0.0, 2019, "", "scheduled service; far below 10 million passengers a year, so below the 5 million enplanements of the protocol: set to 0 (value not looked up)"))

cities = pd.read_csv(f"{HERE}/cities100.csv")
cen = {}
for _, c in cities.iterrows():
    p = f"{HERE}/data/{c['slug']}/centre.json"
    if os.path.exists(p):
        import json
        j = json.load(open(p)); cen[c["slug"]] = (j["lat"], j["lon"])
cen["sydney-nsw"] = (-33.8731, 151.2065)
out = []
for slug, (clat, clon) in cen.items():
    if slug == "sydney-nsw":
        continue
    country = cities.set_index("slug").loc[slug, "country"]
    for a in rows:
        iata_us = (a[6] == FAA)
        if (country == "United States") != iata_us:
            continue
        if country == "United Kingdom" and a[6] != CAA:
            continue
        if country == "Canada" and not (a[1].startswith("Y")):
            continue
        if country == "Australia" and a[1] not in ("MEL", "BNE", "PER", "ADL", "OOL", "CBR", "MCY", "NTL", "HBA", "CNS", "TSV", "DRW", "SYD"):
            continue
        if hav(clat, clon, a[2], a[3]) <= RADIUS_KM:
            out.append((slug,) + a)
df = pd.DataFrame(out, columns=["slug", "airport", "iata", "lat", "lon", "enplanements_m", "year", "source_url", "note"]).sort_values(["slug", "enplanements_m"], ascending=[True, False])
df.to_csv(f"{HERE}/airports_cities100.csv", index=False)
big = df[df["enplanements_m"] >= 5.0]
print(f"{len(df)} rows, {df['slug'].nunique()} cities; {len(big)} rows at or above 5 million enplanements in {big['slug'].nunique()} cities")
print(big[["slug", "iata", "enplanements_m"]].to_string(index=False))
