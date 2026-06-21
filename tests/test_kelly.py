import pytest

from soccer_betting.backtest.kelly import kelly_fraction, stake_size


def test_kelly_zero_when_no_edge():
    # Fair odds (no edge): f* should be ~0.
    assert kelly_fraction(0.5, 2.0) == pytest.approx(0.0, abs=1e-9)


def test_kelly_positive_with_edge():
    # p=0.6 at evens is a strong edge.
    assert kelly_fraction(0.6, 2.0) > 0


def test_kelly_negative_edge_clamped():
    assert kelly_fraction(0.4, 2.0) == 0.0


def test_fractional_kelly_scales():
    full = stake_size(0.6, 2.0, 1000, method="kelly", max_stake_fraction=1.0)
    half = stake_size(0.6, 2.0, 1000, method="fractional_kelly",
                      kelly_fraction_mult=0.5, max_stake_fraction=1.0)
    assert half == pytest.approx(full * 0.5, rel=1e-6)


def test_stake_respects_max_fraction():
    stake = stake_size(0.99, 2.0, 1000, method="kelly", max_stake_fraction=0.05)
    assert stake <= 50.0 + 1e-9


def test_stake_zero_below_min_edge():
    assert stake_size(0.51, 2.0, 1000, min_edge=0.05) == 0.0


def test_flat_stake():
    assert stake_size(0.6, 2.0, 1000, method="flat", flat_stake=25) == 25
