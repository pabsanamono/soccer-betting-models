"""Kelly criterion staking.

The Kelly criterion maximises the long-term logarithmic growth of the bankroll.
For a single binary bet at decimal odds ``o`` with win probability ``p``:

    b = o - 1                      (net fractional winnings)
    f* = (p * b - (1 - p)) / b     = p - (1 - p) / b

Because full Kelly assumes the *true* probability is known — rarely the case in
sports — and produces extreme volatility, the field standard is **fractional
Kelly** (half or quarter). This module also caps any single stake at a maximum
bankroll fraction for safety.
"""
from __future__ import annotations


def kelly_fraction(prob: float, odds: float) -> float:
    """Full-Kelly optimal fraction of bankroll for one bet.

    Returns 0 when the bet has no positive edge (never stake on -EV bets).
    """
    if odds <= 1.0:
        return 0.0
    b = odds - 1.0
    f = (prob * b - (1.0 - prob)) / b
    return max(0.0, f)


def stake_size(
    prob: float,
    odds: float,
    bankroll: float,
    method: str = "fractional_kelly",
    kelly_fraction_mult: float = 0.25,
    flat_stake: float = 10.0,
    max_stake_fraction: float = 0.05,
    min_edge: float = 0.0,
) -> float:
    """Compute the monetary stake for a single bet.

    Parameters
    ----------
    prob:
        Model probability of the selection winning.
    odds:
        Decimal odds offered.
    bankroll:
        Current bankroll.
    method:
        ``"flat"``, ``"kelly"`` (full) or ``"fractional_kelly"``.
    kelly_fraction_mult:
        Multiplier applied to full Kelly (e.g. 0.25 = quarter Kelly).
    flat_stake:
        Stake used by the flat method.
    max_stake_fraction:
        Hard cap on stake as a fraction of bankroll.
    min_edge:
        Only stake when ``prob * odds - 1 >= min_edge``.
    """
    if bankroll <= 0:
        return 0.0
    edge = prob * odds - 1.0
    if edge < min_edge:
        return 0.0

    if method == "flat":
        stake = min(flat_stake, bankroll)
    elif method in ("kelly", "fractional_kelly"):
        f = kelly_fraction(prob, odds)
        if method == "fractional_kelly":
            f *= kelly_fraction_mult
        stake = f * bankroll
    else:
        raise ValueError(f"Unknown staking method '{method}'")

    stake = min(stake, max_stake_fraction * bankroll)
    return max(0.0, float(stake))
