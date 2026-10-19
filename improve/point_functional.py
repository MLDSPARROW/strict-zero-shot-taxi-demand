"""Loss-consistent point forecasts (PROTOCOL_V2.md, I1). MAE is scored on the median of a negative binomial predictive
distribution NB(lambda, k(lambda)) and RMSE on its mean lambda (Gneiting, 2011). k(lambda) is estimated on the target's
share helpers only: lambda_hat = slot total x the helper's own weekday x slot mean share of the region; cells pooled into
40 log-spaced bins of lambda_hat from 0.001 to the maximum; per bin with >= 1,000 cells k = mean(lam)^2 / (mean((y - lam)^2)
- mean(lam)), 10^6 when the excess variance is not positive; log k interpolated in log lambda, constant beyond the ends.
No target data are read here."""
import numpy as np
from datetime import date, timedelta
from scipy.stats import nbinom


def dow(start, n_days):
    return np.array([(start + timedelta(days=d)).weekday() for d in range(n_days)])


def dispersion(helpers, hourly=False, nbins=40, min_cells=1000):
    """helpers: list of (Y days x 48 x R, start date). Returns (k_of(lam) function, table of bins)."""
    L, V = [], []
    for Y, start in helpers:
        Y = Y.astype(np.float64)
        if hourly: Y = Y.reshape(Y.shape[0], 24, 2, -1).sum(2)
        w = dow(start, Y.shape[0]); tot = Y.sum(2)
        num = np.stack([Y[w == k].sum(0) for k in range(7)]); den = np.stack([tot[w == k].sum(0) for k in range(7)])
        m = num / np.maximum(den[..., None], 1e-12); lam = tot[:, :, None] * m[w]
        keep = np.broadcast_to((tot > 0)[:, :, None], Y.shape)
        L.append(lam[keep]); V.append(Y[keep])
    L, V = np.concatenate(L), np.concatenate(V)
    edges = np.logspace(-3, np.log10(L.max()) + 1e-9, nbins + 1); b = np.digitize(L, edges) - 1
    rows = []
    for i in range(nbins):
        s = b == i
        if s.sum() < min_cells: continue
        lb, vb = L[s].mean(), ((V[s] - L[s]) ** 2).mean()
        rows.append((lb, vb, lb ** 2 / (vb - lb) if vb > lb else 1e6, int(s.sum())))
    T = np.array(rows); ll, lk = np.log(T[:, 0]), np.log(T[:, 2])
    return (lambda lam: np.exp(np.interp(np.log(np.maximum(lam, 1e-12)), ll, lk))), T


class Quantiles:
    """NB quantiles of lambda on a fine log grid (median is a step function of lambda; grid ratio 1.00005)."""
    def __init__(self, k_of, lam_max, qs=(0.1, 0.5, 0.9)):
        self.g = np.logspace(-6, np.log10(max(lam_max, 1e-6)) + 0.01, 400_000)
        k = k_of(self.g); p = k / (k + self.g)
        self.q = {q: nbinom.ppf(q, k, p) for q in qs}
    def __call__(self, lam, q):
        lam = np.asarray(lam, dtype=np.float64); out = np.zeros_like(lam)
        pos = lam > 0
        i = np.clip(np.searchsorted(self.g, lam[pos], side="right") - 1, 0, len(self.g) - 1)
        out[pos] = self.q[q][i]; return out


def check(qf, k_of, lam, n=20000, seed=0):
    """Fraction of cells where the grid median differs from the exact NB median (printed by the callers)."""
    rng = np.random.default_rng(seed); x = rng.choice(lam[lam > 0], size=min(n, int((lam > 0).sum())), replace=False)
    k = k_of(x); exact = nbinom.ppf(0.5, k, k / (k + x)); return float(np.mean(exact != qf(x, 0.5)))
