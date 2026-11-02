"""Small statistics helpers shared by the collector, probe and alert rules."""

from __future__ import annotations

import math
from collections.abc import Sequence


def percentile(values: Sequence[float], q: float) -> float | None:
    """Return the q-quantile (0..1) using linear interpolation, or None if empty."""
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo))


def fisher_one_sided(fail_recent: int, n_recent: int, fail_base: int, n_base: int) -> float:
    """P(>= fail_recent failures in the recent group) under the hypergeometric null.

    The null pools both groups: with F = fail_recent + fail_base failures among
    N = n_recent + n_base trials, the recent group is a random draw of n_recent.
    """
    total_fail = fail_recent + fail_base
    total = n_recent + n_base
    denom = math.comb(total, n_recent)
    upper = min(n_recent, total_fail)
    p = (
        sum(
            math.comb(total_fail, k) * math.comb(total - total_fail, n_recent - k)
            for k in range(fail_recent, upper + 1)
        )
        / denom
    )
    return min(1.0, p)
