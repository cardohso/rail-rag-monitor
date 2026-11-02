import math

import pytest

from rail_rag_monitor.stats import fisher_one_sided, percentile


def test_percentile():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([1, 2, 3, 4], 0.95) == pytest.approx(3.85)
    assert percentile([], 0.5) is None


def test_fisher():
    assert fisher_one_sided(0, 24, 0, 112) == 1.0
    assert fisher_one_sided(4, 24, 0, 112) == pytest.approx(
        math.comb(24, 4) / math.comb(136, 4), abs=1e-12
    )
    assert fisher_one_sided(1, 24, 5, 112) > 0.5


def test_fisher_hand_checked_value():
    # 3 failures among 6 runs; P(X >= 2 in the recent 3) = (C(3,2)*C(3,1) + C(3,3)*C(3,0)) / C(6,3)
    assert fisher_one_sided(2, 3, 1, 3) == pytest.approx((9 + 1) / 20)
    assert fisher_one_sided(2, 3, 1, 3) == 0.5


@pytest.mark.parametrize("q", [-0.01, 1.01, 2, -1, math.inf, math.nan])
def test_percentile_rejects_q_outside_unit_interval(q):
    with pytest.raises(ValueError, match="q"):
        percentile([1, 2, 3], q)
    with pytest.raises(ValueError, match="q"):
        percentile([], q)


def test_percentile_accepts_the_endpoints():
    assert percentile([3, 1, 2], 0) == 1
    assert percentile([3, 1, 2], 1) == 3


@pytest.mark.parametrize(
    "args",
    [
        (-1, 3, 1, 3),  # negative recent failures
        (1, 3, -1, 3),  # negative baseline failures
        (4, 3, 1, 3),  # more recent failures than recent runs
        (1, 3, 4, 3),  # more baseline failures than baseline runs
        (1, -3, 0, 3),  # negative group size
        (0, 3, 0, -3),
    ],
)
def test_fisher_rejects_impossible_counts(args):
    with pytest.raises(ValueError):
        fisher_one_sided(*args)
