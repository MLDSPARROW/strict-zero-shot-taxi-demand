"""Target time slots from the calendar only (PROTOCOL_REVISION.md, part A). No demand data are read here.
Chicago, New York City and San Francisco record local clock time, so 02:00 and 02:30 on the spring daylight-saving day
(second Sunday of March) do not exist. The Washington, DC data are hourly counts without that gap, so all slots are kept."""
from datetime import date, timedelta
LOCAL_CLOCK = {"chicago": True, "nyc": True, "sf": True, "dc": False}


def spring_forward(year):
    d = date(year, 3, 1); d += timedelta(days=(6 - d.weekday()) % 7)      # first Sunday of March
    return d + timedelta(days=7)


def calendar_slots(city, start):
    n = (date(start.year + 1, 1, 1) - start).days
    gap = spring_forward(start.year) if LOCAL_CLOCK[city] else None
    return [(d, k) for d in range(n) for k in range(48) if not (gap is not None and start + timedelta(days=d) == gap and k in (4, 5))]
