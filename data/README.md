# Data directory

This directory holds datasets. Bulk contents are **gitignored** (only the folder
structure is tracked via `.gitkeep`).

```
data/
├── raw/         # original downloaded CSVs (e.g. football-data.co.uk: E0_2324.csv)
├── processed/   # standardised, model-ready data (matches.parquet, bet logs)
└── external/    # third-party / manually-added datasets (xG, player stats, ...)
```

## Canonical match schema (`processed/matches.parquet`)

Produced by `soccer_betting.data.preprocess.standardise_matches`:

| column | type | description |
|--------|------|-------------|
| `date` | datetime | match date |
| `season` / `division` | str | optional identifiers |
| `home_team` / `away_team` | str | team names |
| `home_goals` / `away_goals` | int | full-time goals |
| `result` | str | `H` / `D` / `A` |
| `odds_home` / `odds_draw` / `odds_away` | float | closing decimal odds (best available source) |

## Populating the data

```bash
# Real data (per config/config.yaml)
python scripts/collect_data.py

# Offline synthetic data for development/testing
python scripts/collect_data.py --synthetic
```

### Recommended free / paid sources
- **football-data.co.uk** — free historical results + opening/closing odds (built-in collector).
- **The Odds API** (the-odds-api.com) — live forward odds (set `ODDS_API_KEY`).
- **FBref / StatsBomb / Understat** — xG and event data for richer features.

> Respect each provider's terms of use and rate limits.
