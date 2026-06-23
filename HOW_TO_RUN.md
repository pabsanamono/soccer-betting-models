# 🟢 HOW TO RUN — Easy Beginner Guide

This is the **plain-English, copy-paste guide** to run the Soccer Betting
Toolkit on your own computer (Windows, Mac or Linux) and to connect the
**live odds API**. No prior Python experience needed — just follow the steps
in order.

> 💡 You only do **Part 1 (Setup)** once. After that you jump straight to
> **Part 3 (Run things)** whenever you want.

---

## Part 0 — What you need first

1. **Python 3.9 or newer** installed.
   - Check by opening a terminal (see below) and typing: `python --version`
   - If you get an error, install it from <https://www.python.org/downloads/>
     (on Windows, tick **“Add Python to PATH”** during install).
2. **The project folder** on your computer (this folder).

**How to open a terminal:**
- **Windows:** press `Win`, type **PowerShell**, hit Enter.
- **Mac:** press `Cmd+Space`, type **Terminal**, hit Enter.
- **Linux:** `Ctrl+Alt+T`.

---

## Part 1 — One-time setup

Copy these commands **one block at a time** into your terminal.

### Step 1.1 — Go into the project folder
```bash
cd path/to/soccer-betting-models
```
> Replace `path/to/` with the real location. Tip: type `cd ` (with a space)
> then drag the folder into the terminal window and press Enter.

### Step 1.2 — Create a private workspace (“virtual environment”)
This keeps the project’s libraries separate from the rest of your computer.

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Mac / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```
> ✅ You’ll know it worked when you see `(.venv)` at the start of your terminal line.

### Step 1.3 — Install the project and its libraries
```bash
pip install -r requirements.txt
pip install -e .
```
> This downloads the math/ML libraries (numpy, pandas, scikit-learn, xgboost…).
> It can take a few minutes the first time. ☕

**Optional add-ons** (only if you want them):
```bash
pip install -e ".[nn]"        # neural-network model (PyTorch)
pip install -e ".[dev]"       # to run the test suite
```

✅ **Setup done!**

---

## Part 2 — Quick test (works offline, no API needed)

Make sure everything is wired up correctly by running the built-in demo on
**fake (synthetic) data**:

```bash
python examples/quickstart.py
```

You should see it:
1. generate sample matches,
2. compare the models (Poisson, Dixon-Coles, etc.),
3. find “value bets”,
4. run a backtest and print a summary.

If you see a **Backtest summary** table at the end, 🎉 it’s working.

---

## Part 3 — Running the real workflow

There are 4 simple commands. Run them in order. They all read their settings
from `config/config.yaml` (you can edit that file to change leagues, seasons,
etc.).

```bash
# 1) Download real historical results + odds (free, no key needed)
python scripts/collect_data.py
#    …or test offline with fake data:
python scripts/collect_data.py --synthetic

# 2) Train a model and see how good its forecasts are
python scripts/train_models.py --model dixon_coles
#    other options: --model poisson | elo | ml

# 3) Backtest it (simulate betting on history with Kelly staking)
python scripts/run_backtest.py --model dixon_coles

# 4) List the value bets from a predictions file
python scripts/find_value_bets.py data/processed/predictions.csv --min-edge 0.03
```

> The free historical data in step 1 comes from **football-data.co.uk** and
> needs **no API key**. You only need the API for *upcoming/live* matches —
> see Part 4.

---

## Part 4 — 🔌 Connecting the live Odds API

The free data above is **historical** (past matches). To get **upcoming
matches and live odds**, the project uses **The Odds API**.

### Step 4.1 — Get a free API key
1. Go to <https://the-odds-api.com/>
2. Click **Get API Key** and sign up (the free tier gives 500 requests/month).
3. Copy your key — it looks like `a1b2c3d4e5f6...`

### Step 4.2 — Give the key to the project
**Never type your key inside the code.** Instead, set it as an
“environment variable” in your terminal:

**Windows (PowerShell):**
```powershell
$env:ODDS_API_KEY = "paste_your_key_here"
```

**Mac / Linux:**
```bash
export ODDS_API_KEY="paste_your_key_here"
```
> ⚠️ This lasts only for the current terminal window. To make it permanent,
> add that line to your shell profile (`~/.bashrc`, `~/.zshrc`) or use a
> `.env` file. The `.gitignore` already prevents keys from being uploaded to GitHub.

### Step 4.3 — Turn the API on in the config
Open `config/config.yaml` and find the `odds_api` section. Change
`enabled: false` to `enabled: true`:

```yaml
data:
  odds_api:
    enabled: true                 # ← was false
    base_url: https://api.the-odds-api.com/v4
    sport: soccer_epl             # league: soccer_epl, soccer_spain_la_liga, ...
    regions: eu                   # eu | uk | us | au
    markets: h2h                  # h2h = home/draw/away
```

### Step 4.4 — Test the API connection in Python
Paste this into a file called `test_api.py` (or run `python` and paste line by line):

```python
from soccer_betting.data.collectors import TheOddsAPICollector

# Reads your ODDS_API_KEY automatically from the environment.
collector = TheOddsAPICollector(sport="soccer_epl", regions="eu", markets="h2h")
odds = collector.fetch_odds()

print(f"Got {len(odds)} odds rows")
print(odds.head())
```

Run it:
```bash
python test_api.py
```
✅ If you see a table with `home_team`, `away_team`, `odds_home`, `odds_draw`,
`odds_away`, your API is connected!

**Common errors:**
| Message | Fix |
|---|---|
| `No Odds API key. Set ODDS_API_KEY...` | You forgot Step 4.2 — set the key in the **same** terminal. |
| `401 Unauthorized` | The key is wrong or has extra spaces/quotes. |
| `429 Too Many Requests` | You hit the free monthly limit — wait or upgrade. |
| Empty table | No matches scheduled right now for that `sport`/`regions`. |

---

## Part 5 — Use it from your own Python code

```python
import pandas as pd
from soccer_betting.data.preprocess import make_synthetic_matches
from soccer_betting.models import DixonColesModel
from soccer_betting.evaluation.value import find_value_bets

# 1. Get some matches (here: synthetic; or load your collected CSV with pandas)
matches = make_synthetic_matches(n_teams=16, n_seasons=4)
train, test = matches.iloc[:700], matches.iloc[700:].reset_index(drop=True)

# 2. Train a model
model = DixonColesModel(xi=0.0018).fit(train)

# 3. Predict probabilities [home, draw, away]
preds = model.predict_frame(test).reset_index(drop=True)

# 4. Find value vs the bookmaker
frame = pd.concat(
    [test[["home_team", "away_team", "odds_home", "odds_draw", "odds_away"]], preds],
    axis=1,
)
print(find_value_bets(frame, min_edge=0.05, devig_method="shin").head())
```

---

## Part 6 — Running the tests (optional, for peace of mind)

```bash
pip install -e ".[dev]"
pytest -q
```
All tests passing = the math (odds, Kelly, scoring, models) is behaving.

---

## 🆘 Quick troubleshooting

| Problem | Likely fix |
|---|---|
| `python` not found | Try `python3` instead, or reinstall Python with “Add to PATH”. |
| `(.venv)` not showing | Re-run the activate command from Step 1.2. |
| `ModuleNotFoundError: soccer_betting` | Activate the venv, then re-run `pip install -e .` |
| `pip` is slow / fails | Check your internet; re-run the command (it resumes). |
| Permission error on Windows activate | Run PowerShell as Admin once: `Set-ExecutionPolicy RemoteSigned` |

---

## 📌 Cheat sheet (after setup)

```bash
# every new session:
cd path/to/soccer-betting-models
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1

# then do what you need:
python examples/quickstart.py                       # offline demo
python scripts/collect_data.py                      # get historical data
python scripts/train_models.py --model dixon_coles  # train
python scripts/run_backtest.py --model dixon_coles  # backtest

# for live odds, once per session:
export ODDS_API_KEY="your_key"     # Windows: $env:ODDS_API_KEY="your_key"
```

> ⚠️ **Reminder:** This toolkit is for research/education. Betting markets are
> efficient and edges are thin. Never stake money you can’t afford to lose.
```
