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
