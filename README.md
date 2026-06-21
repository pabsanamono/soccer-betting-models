# ⚽ Soccer Betting Modeling Toolkit

A production-oriented Python framework for modeling soccer (association
football) match probabilities and identifying **value** against bookmaker
prices. It implements the full, disciplined pipeline distilled from the research
report in [`research/`](research/): leak-free data → theory-grounded features →
market-integrated & devigged probabilities → calibration → proper-scoring
evaluation → honest walk-forward backtesting → fractional-Kelly staking.

> **Responsible-use disclaimer.** This software is for research and educational
> purposes only. Sports betting markets are highly efficient; the research is
> clear that sustained edges are thin, intermittent, and that winning accounts
> get limited. Nothing here is financial advice. Never stake money you cannot
> afford to lose, and obey the laws of your jurisdiction.

---

## ✨ What's inside

| Area | Module | Highlights |
|------|--------|-----------|
| **Data collection** | `soccer_betting.data.collectors` | football-data.co.uk (free results + closing odds), The Odds API client, offline synthetic generator |
| **Preprocessing** | `soccer_betting.data.preprocess` | canonical match schema, closing-odds selection |
| **Feature engineering** | `soccer_betting.data.features` | leakage-free rolling form, Elo ratings, devigged market probs |
| **Statistical models** | `soccer_betting.models` | Independent **Poisson** (Maher 1982), **Dixon-Coles** (rho + time decay), **Bivariate Poisson** (Karlis-Ntzoufras), **Elo** |
| **Machine learning** | `soccer_betting.models.ml_models` / `nn` | gradient-boosted trees (XGBoost), optional Torch MLP, soft-voting **ensemble** |
| **Calibration** | `soccer_betting.calibration` | Platt / isotonic, ECE, reliability curves |
| **Odds / devig** | `soccer_betting.odds.devig` | multiplicative, additive, power, **Shin** |
| **Backtesting** | `soccer_betting.backtest` | walk-forward engine, **Kelly** staking, ROI/drawdown/CLV |
| **Evaluation** | `soccer_betting.evaluation` | **RPS**, log-loss, **Brier**, value-bet identification |

---

## 📦 Installation

```bash
# 1. (Recommended) create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install the package itself (editable)
pip install -e .

# Optional extras
pip install -e ".[nn]"        # PyTorch neural network
pip install -e ".[catboost]"  # CatBoost backend
pip install -e ".[dev]"       # pytest
```

Python 3.9+ is required.

---

## 🚀 Quickstart (no internet needed)

Run the full pipeline on a built-in **synthetic** dataset:

```bash
python examples/quickstart.py
```

This generates data, compares every model with proper scoring rules, finds value
bets, and runs a quarter-Kelly walk-forward backtest — all in one script.

---

## 🛠️ Typical workflow

The four console commands (installed by `pip install -e .`) mirror the scripts
in `scripts/`:

```bash
# 1. Collect & standardise data (uses config/config.yaml). --synthetic for offline.
python scripts/collect_data.py            # or: sb-collect
python scripts/collect_data.py --synthetic

# 2. Fit a model and report forecast quality (RPS / log-loss / Brier)
python scripts/train_models.py --model dixon_coles     # or: sb-train
python scripts/train_models.py --model ml

# 3. Walk-forward backtest with Kelly staking
python scripts/run_backtest.py --model dixon_coles     # or: sb-backtest

# 4. List value bets from a predictions CSV (prob_* + odds_* columns)
python scripts/find_value_bets.py data/processed/predictions.csv --min-edge 0.03
```

All behaviour is driven by [`config/config.yaml`](config/config.yaml) —
leagues/seasons, feature windows, model hyper-parameters, devig method, and
staking rules.

---

## 🧑‍💻 Library usage

```python
import pandas as pd
from soccer_betting.data.preprocess import make_synthetic_matches
from soccer_betting.models import DixonColesModel
from soccer_betting.evaluation.metrics import evaluate_predictions
from soccer_betting.evaluation.value import find_value_bets

matches = make_synthetic_matches(n_teams=16, n_seasons=4)
train, test = matches.iloc[:700], matches.iloc[700:].reset_index(drop=True)

# Fit a Dixon-Coles model with time decay (xi per day).
model = DixonColesModel(xi=0.0018).fit(train)

# Probabilistic predictions [P(home), P(draw), P(away)]
preds = model.predict_frame(test).reset_index(drop=True)
print(evaluate_predictions(preds.to_numpy(), test["result"].to_numpy()))

# Identify value vs the market (Shin devig)
frame = pd.concat([test[["home_team", "away_team",
                         "odds_home", "odds_draw", "odds_away"]], preds], axis=1)
print(find_value_bets(frame, min_edge=0.05, devig_method="shin").head())
```

### Using real data
`scripts/collect_data.py` downloads free CSVs from football-data.co.uk for the
divisions/seasons set in `config.yaml`. For live, forward-looking odds set
`data.odds_api.enabled: true` and export an API key:

```bash
export ODDS_API_KEY="your_key_here"   # never hard-code secrets
```

---

## 📐 Modeling principles (from the research)

1. **Avoid leakage** — every feature is computed walk-forward, using only
   information available before kick-off.
2. **Calibration > accuracy** — miscalibrated, overconfident probabilities
   inflate the perceived edge and lead to over-staking. Calibrate on a held-out
   set and pick the method with the lowest Brier/log-loss.
3. **Devig the market** — the bookmaker price is the strongest single
   predictor; strip the margin before comparing.
4. **Score properly** — use RPS, log-loss and Brier, never raw accuracy.
5. **Backtest honestly** — out-of-sample, 250–500+ bets, benchmark to the
   closing line (CLV), and check maximum drawdown.
6. **Stake with fractional Kelly** — full Kelly assumes a known true
   probability and risks ruin; quarter-Kelly is the field standard.

See [`research/soccer_betting_models_and_bookmaker_pricing_report.md`](research/)
for the full literature review behind these choices.

---

## 🗂️ Project structure

```
soccer_betting_project/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   └── config.yaml                 # central configuration
├── research/                       # the source research report (PDF + MD)
├── data/                           # raw / processed / external (gitignored)
├── src/soccer_betting/
│   ├── config.py                   # YAML config loader
│   ├── cli.py                      # console entry points
│   ├── data/                       # collectors, preprocess, features
│   ├── models/                     # poisson, dixon_coles, bivariate_poisson, elo, ml_models, nn
│   ├── odds/                       # devigging
│   ├── calibration/                # probability calibration + diagnostics
│   ├── backtest/                   # engine, kelly, metrics
│   ├── evaluation/                 # scoring rules, value identification
│   └── utils/                      # logging
├── scripts/                        # collect / train / backtest / value CLIs
├── examples/                       # quickstart.py
├── notebooks/                      # (your exploratory notebooks)
└── tests/                          # pytest suite
```

---

## ✅ Testing

```bash
pytest -q
```

The suite covers odds/devig math, Kelly staking, proper scoring rules, every
statistical model, the feature pipeline, ML models, value identification and an
end-to-end backtest.

---

## 📄 License

MIT — see headers. Provided "as is" without warranty of any kind.
