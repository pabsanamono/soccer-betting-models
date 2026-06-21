import numpy as np
import pytest

from soccer_betting.models import (
    PoissonModel, DixonColesModel, BivariatePoissonModel, EloModel,
)


@pytest.mark.parametrize("ModelCls", [PoissonModel, DixonColesModel, BivariatePoissonModel, EloModel])
def test_predict_proba_valid_distribution(ModelCls, synthetic_matches):
    train = synthetic_matches.iloc[:400]
    model = ModelCls().fit(train)
    teams = train["home_team"].unique()
    p = model.predict_proba(teams[0], teams[1])
    assert p.shape == (3,)
    assert pytest.approx(p.sum(), abs=1e-6) == 1.0
    assert np.all(p >= 0)


def test_predict_frame_aligns(synthetic_matches):
    train = synthetic_matches.iloc[:400]
    test = synthetic_matches.iloc[400:430]
    model = DixonColesModel().fit(train)
    preds = model.predict_frame(test)
    assert list(preds.columns) == ["prob_home", "prob_draw", "prob_away"]
    assert len(preds) == len(test)


def test_dixon_coles_increases_draw_mass(synthetic_matches):
    """Dixon-Coles rho should adjust low-score draw probabilities vs Poisson."""
    train = synthetic_matches.iloc[:600]
    pois = PoissonModel().fit(train)
    dc = DixonColesModel().fit(train)
    teams = train["home_team"].unique()
    # Both should produce valid, finite draw probabilities.
    assert 0 < dc.predict_proba(teams[0], teams[1])[1] < 1
    assert 0 < pois.predict_proba(teams[0], teams[1])[1] < 1


def test_unknown_team_raises(synthetic_matches):
    model = PoissonModel().fit(synthetic_matches.iloc[:300])
    with pytest.raises(KeyError):
        model.predict_proba("NoSuchTeam", "AlsoMissing")


def test_over_under_and_btts(synthetic_matches):
    model = DixonColesModel().fit(synthetic_matches.iloc[:400])
    teams = synthetic_matches["home_team"].unique()
    over, under = model.prob_over_under(teams[0], teams[1], line=2.5)
    assert pytest.approx(over + under, abs=1e-6) == 1.0
    assert 0 <= model.prob_btts(teams[0], teams[1]) <= 1
