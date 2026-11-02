"""Small statistics helpers shared by the collector, probe and alert rules."""

from __future__ import annotations

import math
from collections.abc import Sequence


def percentile(values: Sequence[float], q: float) -> float | None:
    """Return the q-quantile (0..1) using linear interpolation, or None if empty.

    Raises ValueError if q is outside [0, 1] (or not a number).
    """
    if not 0 <= q <= 1:
        raise ValueError(f"q must be between 0 and 1, got {q}")
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

    Raises ValueError for a negative count or a failure count above its group size.
    """
    if not (0 <= fail_recent <= n_recent and 0 <= fail_base <= n_base):
        raise ValueError(
            "failure counts must satisfy 0 <= failures <= group size, got "
            f"{fail_recent}/{n_recent} recent and {fail_base}/{n_base} baseline"
        )
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
