"""PROTOCOL_V12.md: the three fixed combination rules (no parameter is tuned; the weights are the equal weights stated in the protocol).
v1 = shares of the final method, g = shares of the scale-matched XGBoost baseline, jr = jobs + residents share (public data, same for every slot).
All shares are (slots x regions) and rows sum to one."""
import numpy as np


def norm(p): return p / p.sum(1, keepdims=True)


def rules(v1, g, jr):
    J = np.repeat(jr[None], len(v1), 0)
    geo = norm(np.sqrt(np.clip(v1, 1e-12, None) * np.clip(J, 1e-12, None)))
    return {"R1 arithmetic v1 + jobs-residents": 0.5 * v1 + 0.5 * J,
            "R2 arithmetic v1 + XGBoost + jobs-residents": (v1 + g + J) / 3.0,
            "R3 geometric v1 x jobs-residents": geo}
