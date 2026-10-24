"""PROTOCOL_V9.md: the helper lists of the allocator with bike-share cities as extra teachers (flag k)."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
BIKE = ["boston", "philadelphia", "la", "minneapolis", "columbus"]
TAXI = {"chicago": lambda r: [f"nyc_for_chicago_p{r}", f"sf_for_chicago_p{r}"], "nyc": lambda r: ["chicago", f"sf_for_nyc_p{r}"]}   # helpers of the final method (v1)


def srcs(T, r):
    """v1 helpers of the target T for run r, then the five bike-share cities: partition r of the re-aggregated city if its tracts are smaller than the
    regions of the target, otherwise the city at its native scale"""
    bike = [f"bike_{b}_for_{T}_p{r}" if os.path.exists(f"{HERE}/data/agg/bike_{b}_for_{T}_p{r}_Y.npy") else f"bike_{b}" for b in BIKE]
    return TAXI[T](r) + bike
