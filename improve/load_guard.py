"""JOB1_COMPUTE.md section 0.4: wait before an allocator run while the server is busy.
Waits while the 1-minute load average is above 0.7 x nproc, or MemAvailable is below 4 GB, or the
website health check takes more than 1.5 s (checked every 60 s; a wait longer than 30 minutes is logged).
Usage: from load_guard import wait_until_quiet; wait_until_quiet(log)   or   python load_guard.py (one check)"""
import os
import time
import urllib.request

HEALTH_URL = os.environ.get("LOAD_GUARD_URL", "http://127.0.0.1:8001/")   # the website's gunicorn, read only
NPROC = os.cpu_count() or 1


def mem_available_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    return 0.0


def health_seconds():
    t = time.time()
    try:
        req = urllib.request.Request(HEALTH_URL, headers={"Host": "miliroute.com", "X-Forwarded-Proto": "https",
                                                          "User-Agent": "taxi-demand-load-guard"})
        urllib.request.urlopen(req, timeout=10).read(1)
    except Exception:
        return 99.0
    return time.time() - t


def busy():
    """Reason the server is busy, or '' when a run may start."""
    load1 = os.getloadavg()[0]
    if load1 > 0.7 * NPROC:
        return f"load {load1:.2f} > {0.7 * NPROC:.2f}"
    mem = mem_available_gb()
    if mem < 4.0:
        return f"MemAvailable {mem:.1f} GB < 4 GB"
    h = health_seconds()
    if h > 1.5:
        return f"website health check {h:.2f} s > 1.5 s"
    return ""


def wait_until_quiet(log=print):
    start, logged = time.time(), False
    while True:
        why = busy()
        if not why:
            return time.time() - start
        if time.time() - start > 1800 and not logged:
            log(f"load_guard: waiting for more than 30 minutes ({why})")
            logged = True
        time.sleep(60)


if __name__ == "__main__":
    why = busy()
    print(why or f"quiet (load {os.getloadavg()[0]:.2f}, MemAvailable {mem_available_gb():.1f} GB, health {health_seconds():.2f} s)")
