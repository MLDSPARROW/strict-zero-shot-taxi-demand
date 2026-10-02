"""FAA calendar-year 2021 enplanements per region (in millions), saved as improve/data/enpl_<city>.npy.
Airports are assigned to the region containing the OurAirports reference point, for large commercial airports whose
listed municipality is the city itself (build_extra.py, column 'airport'). Source: FAA CY2021 passenger boarding data
(raw/cy21.xlsx). The assignment is checked against the airport flag produced by build_extra.py."""
import numpy as np

ENPL = {"chicago": {76: 26.350976, 56: 7.680617},      # O'Hare (ORD) in community area 76, Midway (MDW) in 56
        "nyc": {132: 15.273342, 138: 7.827307},        # JFK in taxi zone 132, LaGuardia (LGA) in zone 138
        "sf": {}, "dc": {}}                            # main airports lie outside the city boundaries
B = "../bench77/data"
for c, m in ENPL.items():
    rid = (np.load(f"{B}/region_static_{c}.npz")["rid"] if c in ("chicago", "nyc")
           else np.arange(len(np.load(f"../multicity/data/region_static_{c}.npz")["X"])))
    e = np.array([m.get(int(r), 0.0) for r in rid], np.float32)
    flag = np.load(f"data/extra_{c}.npz")["E"][:, 3] > 0
    assert ((e > 0) == flag).all(), c
    np.save(f"data/enpl_{c}.npy", e)
    print(c, e[e > 0])
