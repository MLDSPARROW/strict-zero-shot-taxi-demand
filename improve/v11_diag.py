"""DESCRIPTIVE diagnostics on the development targets Chicago and NYC only (reads their demand, so nothing here is a result or a rule).
(1) the yearly allocation error of the final method v1 and its heads, and how it relates to region area and job density;
(2) arithmetic against logarithmic (geometric) pooling of the heads and runs of v1 (10-run sets in out/, demand-reading mode);
(3) whether the spread of the ensemble members (a quantity that needs no target demand) tracks the size of the transfer error.
Strict trip errors use the city totals of the paper (coarse_eval.py totals). Usage (from improve/): python v11_diag.py"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(f"{HERE}/coarse_eval.py").read(); ns = {"__file__": f"{HERE}/coarse_eval.py"}
exec(src[:src.index("norm = lambda")], ns); totals, YP, SHARE = ns["totals"], ns["YP"], ns["SHARE"]
SHARE.update({"chicago": ["nyc", "sf"], "nyc": ["chicago", "sf"]})                  # as predict_final.py
SRC = {"chicago": lambda r: f"nyc_for_chicago_p{r}+sf_for_chicago_p{r}", "nyc": lambda r: f"chicago+sf_for_nyc_p{r}"}
B = f"{HERE}/../bench77/data"; norm = lambda p: p / p.sum(1, keepdims=True)
for T in ("chicago", "nyc"):
    Y = np.load(YP[T]).astype(np.float64); tot = Y.sum(2)
    di, si, Tt, unit, lev = totals(T, T); y = Y[di, si]; Ts = tot[di, si]
    st = np.load(f"{B}/region_static_{T}.npz"); area = st["area_km2"].astype(np.float64); jobs = np.load(f"{HERE}/data/extra_{T}.npz")["jobs"].astype(np.float64)
    M = {h: np.array([norm(np.load(f"{HERE}/out/multi_{T}_from_{SRC[T](r)}_v{h}_seed{r - 1}.npy").astype(np.float64)).astype(np.float32) for r in range(1, 11)]) for h in ("eh", "eo", "ep")}
    H = {h: M[h].mean(0).astype(np.float64) for h in M}
    lg = lambda p: np.log(np.maximum(p, 1e-12))
    C = {"v1 = (eh + eo)/2 [arithmetic]": 0.5 * H["eh"] + 0.5 * H["eo"],
         "geometric mean of the two heads": norm(np.exp(0.5 * lg(H["eh"]) + 0.5 * lg(H["eo"]))),
         "geometric mean of all 20 members": norm(np.exp(np.mean([lg(m) for m in np.concatenate([M["eh"], M["eo"]])], 0))),
         "eh alone": H["eh"], "eo alone": H["eo"], "ep alone (people head, V2)": H["ep"],
         "(eh + eo + ep)/3 arithmetic": (H["eh"] + H["eo"] + H["ep"]) / 3,
         "geometric mean of eh, eo, ep": norm(np.exp((lg(H["eh"]) + lg(H["eo"]) + lg(H["ep"])) / 3))}
    ty = Y.sum((0, 1)) / Y.sum(); R = len(ty)
    print(f"\n======== {T}: {R} regions, {len(di)} slots, strict level {lev:,.0f} vs observed {Y.sum():,.0f} ({100 * (lev / Y.sum() - 1):+.1f}%) ========")
    print("   combination                              strict MAE  RMSE | alloc-only MAE  RMSE | yearly TV  top-10 share (true %.2f)  peak ratio" % np.sort(ty)[-10:].sum())
    for k, P in C.items():
        e = P * Tt[:, None] - y; ea = P * Ts[:, None] - y
        py = (P * Tt[:, None]).sum(0); py /= py.sum()
        print(f"   {k:40s} {np.abs(e).mean():6.3f} {np.sqrt((e ** 2).mean()):7.3f} | {np.abs(ea).mean():6.3f} {np.sqrt((ea ** 2).mean()):7.3f} | "
              f"{0.5 * np.abs(py - ty).sum():.3f}     {np.sort(py)[-10:].sum():.2f}                    {(P.max(1) / (y / Ts[:, None]).max(1)).mean():.2f}")
    v1 = C["v1 = (eh + eo)/2 [arithmetic]"]; py = (v1 * Tt[:, None]).sum(0); py /= py.sum()
    lr = np.log(np.maximum(py, 1e-9) / np.maximum(ty, 1e-9)); w = ty                                     # log over-prediction of the yearly share, weighted by true share
    dens = np.log1p(jobs / area)
    print(f"   yearly log ratio pred/true of v1 vs log area: corr {np.corrcoef(lr, np.log(area))[0, 1]:+.2f}; vs log job density: corr {np.corrcoef(lr, dens)[0, 1]:+.2f}")
    over = py - ty; o = np.argsort(over)
    print("   most under-predicted regions (rid, true %, v1 %, eh %, eo %, area km2, jobs/km2):")
    hy = {h: (lambda q: q / q.sum())((H[h] * Tt[:, None]).sum(0)) for h in ("eh", "eo")}
    for i in o[:6]: print(f"      {int(st['rid'][i]):4d}  {100 * ty[i]:5.2f} {100 * py[i]:5.2f} {100 * hy['eh'][i]:5.2f} {100 * hy['eo'][i]:5.2f}  {area[i]:6.2f} {jobs[i] / area[i]:9.0f}")
    print("   most over-predicted regions:")
    for i in o[::-1][:6]: print(f"      {int(st['rid'][i]):4d}  {100 * ty[i]:5.2f} {100 * py[i]:5.2f} {100 * hy['eh'][i]:5.2f} {100 * hy['eo'][i]:5.2f}  {area[i]:6.2f} {jobs[i] / area[i]:9.0f}")
    # (3) spread of the 20 members of the yearly log share against the size of the yearly error, by region
    ly = np.array([lg((m * Tt[:, None]).sum(0) / (m * Tt[:, None]).sum()) for m in np.concatenate([M["eh"], M["eo"]])])
    sd = ly.std(0); ae = np.abs(lr)
    print(f"   member spread of the yearly log share: share-weighted mean sd {np.sum(w * sd):.3f}; share-weighted mean |log error| {np.sum(w * ae):.3f}; "
          f"rank corr(sd, |error|) {np.corrcoef(np.argsort(np.argsort(sd)), np.argsort(np.argsort(ae)))[0, 1]:+.2f}")
    del M
