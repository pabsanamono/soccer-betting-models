# World Cup 2026 — Predictions Guide

This guide shows you, step by step, how to download World Cup 2026 data and get
match predictions and value bets. **No programming knowledge required** — you
just run two commands.

> **Good news:** you can try it right now *without any API key*. The scripts
> include a realistic built-in "demo" dataset so you can see exactly how
> everything works before paying for anything. When you're ready for live data,
> just paste in an API key (Step 2).

---

## What you'll get

* A table of **win / draw / loss probabilities** for every upcoming World Cup
  fixture.
* The model's **prediction** and **confidence** for each match.
* A list of **value bets** — matches where the model thinks the bookmaker odds
  are too generous (a potential edge).
* Everything saved to **CSV files** you can open in Excel.

---

## Step 0 — One-time setup (install the requirements)

Open **PowerShell** (or Command Prompt), go into the project folder, and
install the requirements. You only do this once.

```powershell
cd C:\Users\Pablo\soccer-betting-models
pip install -r requirements.txt
```

---

## Step 1 — Try it now in demo mode (optional but recommended)

You can run the whole thing immediately, before getting any API key:

```powershell
python scripts/fetch_worldcup_data.py
python scripts/predict_worldcup.py
```

This uses the built-in demo dataset (48 real national teams, all 72 group-stage
matches). It's a great way to confirm everything works on your computer. When
you're ready for real data, continue to Step 2.

---

## Step 2 — Get your API key (for live data)

We recommend **TheStatsAPI** — it provides historical results, fixtures, **and**
bookmaker odds in one subscription.

1. Go to **<https://www.thestatsapi.com>**
2. Click **Get Started / Sign Up** and create an account.
3. Choose a plan (the **Starter** plan covers everything these scripts need).
4. Open your **dashboard** and **copy your API key**.

### Paste the key into the config file

Open the file **`config/api_config.yaml`** in any text editor (e.g. Notepad).
Find this line near the top:

```yaml
thestatsapi:
  api_key: ""                              # <-- PASTE YOUR KEY HERE
```

Put your key between the quotes, for example:

```yaml
thestatsapi:
  api_key: "abc123yourkeyhere456"
```

Save the file. **That's the only thing you have to change.**

> Prefer not to edit the file? You can instead set an environment variable in
> PowerShell before running the scripts:
> ```powershell
> $env:THESTATSAPI_KEY = "abc123yourkeyhere456"
> ```

---

## Step 3 — Download the data

```powershell
python scripts/fetch_worldcup_data.py
```

This downloads everything and saves three files into **`data/worldcup/`**:

| File | What it contains |
|------|------------------|
| `worldcup_historical.csv` | Past results used to train the models |
| `worldcup_fixtures.csv`   | Upcoming matches to predict |
| `worldcup_odds.csv`       | Bookmaker odds for those matches |

It also prints the data source it used and a preview of the next few fixtures.

---

## Step 4 — Get predictions

```powershell
python scripts/predict_worldcup.py
```

This trains the models, prints the predictions and value bets on screen, and
saves two more files:

| File | What it contains |
|------|------------------|
| `worldcup_predictions.csv` | Probabilities + prediction for every fixture |
| `worldcup_value_bets.csv`  | The matches with a betting edge |

---

## Example output

**Predictions table:**

```
       date   stage     home_team    away_team P(Home) P(Draw) P(Away) prediction confidence
2026-06-11 Group A        Mexico       Canada   47.7%   27.8%   24.5%   Home win      47.7%
2026-06-17 Group D        France      Denmark   56.7%   23.1%   20.2%   Home win      56.7%
2026-06-20 Group G       England        Qatar   69.2%   16.4%   14.4%   Home win      69.2%
```

**Value bets table:**

```
       date  home_team  away_team selection  odds model_prob market_prob  edge
2026-06-13 United States   Wales  Home win 1.799      65.1%       53.3% 17.1%
2026-06-18      Germany  Morocco  Away win 4.491      29.9%       20.5% 34.5%
```

* **P(Home) / P(Draw) / P(Away)** — the model's estimated chance of each result.
* **model_prob** — the model's probability for that specific selection.
* **market_prob** — the bookmaker's implied probability (margin removed).
* **edge** — how much value the model sees. A positive edge means the model
  rates the bet better than the bookmaker's price implies.

---

## Settings you can tweak (optional)

All settings live in **`config/api_config.yaml`**. The most useful ones:

| Setting | What it does |
|---------|--------------|
| `prediction.model` | Which model makes the headline predictions. `dixon_coles` is recommended (best results in our tests). |
| `value_bets.min_edge` | Minimum edge to flag a bet. `0.05` = 5%. Lower it to see more bets, raise it to be stricter. |
| `value_bets.devig_method` | How the bookmaker margin is removed (`shin` is a solid default). |
| `demo.allow_demo_fallback` | Set to `false` once your live key works, to make the script error out instead of silently using demo data. |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `python` is not recognized | Install Python from <https://www.python.org/downloads/> and tick *"Add Python to PATH"*. |
| `Could not find ... Run fetch first` | Run `python scripts/fetch_worldcup_data.py` before `predict_worldcup.py`. |
| It says "DEMO data" but I added a key | Make sure you saved `config/api_config.yaml`, and that the key is inside the quotes. If you still get a 401/403, try changing `auth_style` to `header_x_api_key` or `query`. |
| Authentication error (401/403) | Your key may be wrong or your plan inactive. Double-check it in your TheStatsAPI dashboard. |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again. |

---

## How it works (in one paragraph)

`fetch_worldcup_data.py` calls TheStatsAPI (or the free Bzzoiro backup, or the
built-in demo data) and saves the results in the standard format the models
expect. `predict_worldcup.py` then trains the **Dixon-Coles** model — a proven
soccer-scoreline model — on the historical results, predicts every upcoming
fixture, compares the predictions to the bookmaker odds, and reports any value
bets. All the heavy lifting reuses the existing, tested code in
`src/soccer_betting/`.

> **Disclaimer:** These predictions are for informational and research purposes
> only. Betting carries financial risk — never bet more than you can afford to
> lose.
