import numpy as np
import pytest

from soccer_betting.odds.devig import (
    devig, odds_to_prob, prob_to_odds, overround,
)


def test_odds_prob_roundtrip():
    odds = np.array([2.0, 3.5, 4.0])
    p = odds_to_prob(odds)
    np.testing.assert_allclose(prob_to_odds(p), odds)


def test_overround_positive_for_real_book():
    odds = np.array([2.0, 3.4, 3.6])  # implied sum > 1
    assert overround(odds) > 0


@pytest.mark.parametrize("method", ["multiplicative", "additive", "power", "shin"])
def test_devig_sums_to_one(method):
    odds = np.array([1.8, 3.6, 4.5])
    p = devig(odds, method=method)
    assert pytest.approx(p.sum(), abs=1e-6) == 1.0
    assert np.all(p > 0)


def test_devig_2d_batch():
    odds = np.array([[2.0, 3.4, 3.6], [1.5, 4.0, 7.0]])
    p = devig(odds, method="multiplicative")
    assert p.shape == (2, 3)
    np.testing.assert_allclose(p.sum(axis=1), [1, 1], atol=1e-9)


def test_invalid_odds_raise():
    with pytest.raises(ValueError):
        odds_to_prob([1.0, 2.0, 3.0])
