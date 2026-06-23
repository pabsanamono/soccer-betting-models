# 🪟 YOUR EXACT WINDOWS COMMANDS — Copy & Paste Ready

**For: Pablo's Windows PC (Acer C: drive)**

This file has EVERY command you'll ever need, already filled in for YOUR computer.
No guessing, no tech knowledge needed — just **copy the blue boxes** into PowerShell.

---

## 📌 FIRST TIME: Where to put the project

**Best location:** `C:\Users\Pablo\Documents\soccer-betting-models`

This puts it in your Documents folder where it's easy to find and safe.

---

## 🚀 How to open PowerShell (do this every time)

1. Press the **Windows key** (⊞ on your keyboard)
2. Type: **PowerShell**
3. Click **Windows PowerShell** (the blue one)
4. A blue window opens — that's your terminal ✅

---

## 📥 STEP 0: Download the project from GitHub (one time only)

### Option A: Download as ZIP (easiest)
1. Go to: https://github.com/pabsanamono/soccer-betting-models
2. Click the green **Code** button
3. Click **Download ZIP**
4. The file `soccer-betting-models-main.zip` downloads
5. **Right-click** the ZIP → **Extract All** → Choose `C:\Users\Pablo\Documents`
6. Rename the folder from `soccer-betting-models-main` to `soccer-betting-models`

**OR**

### Option B: Use Git (if you have it installed)
Open PowerShell and paste this:

```powershell
cd C:\Users\Pablo\Documents
git clone https://github.com/pabsanamono/soccer-betting-models.git
```

---

## 🎯 STEP 1.1: Go into the project folder

**Every time you open PowerShell, paste this FIRST:**

```powershell
cd C:\Users\Pablo\Documents\soccer-betting-models
```

**How to check it worked:**
- The line should now say: `C:\Users\Pablo\Documents\soccer-betting-models>`
- If you see that, ✅ you're in the right place!

**Pro tip:** After pasting, press **Enter**.

---

## 🛠️ STEP 1.2: Create the virtual environment (one time only)

**Copy and paste this:**

```powershell
python -m venv .venv
```

Press Enter. It takes 10–30 seconds. You'll see nothing happen — that's normal!

**If you get "python not found":**
Try this instead:
```powershell
py -m venv .venv
```

---

## ▶️ STEP 1.3: Activate the virtual environment

**YOU MUST DO THIS EVERY TIME** you open a new PowerShell window.

```powershell
.venv\Scripts\Activate.ps1
```

✅ **Success looks like this:**
```
(.venv) C:\Users\Pablo\Documents\soccer-betting-models>
```
See the `(.venv)` at the start? That means it's working!

**If you get a red error about "execution policy":**
Run this ONCE (as Administrator):
```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then try the activate command again.

---

## 📦 STEP 1.4: Install everything (one time only)

**After activating, paste these two commands:**

```powershell
pip install -r requirements.txt
```
(Wait 2–5 minutes... lots of text will scroll by)

Then:
```powershell
pip install -e .
```

✅ Done! You never have to do this again (unless you delete the `.venv` folder).

---

## 🧪 STEP 2: Test it works (offline, no internet needed)

```powershell
python examples/quickstart.py
```

You should see:
- `Generating synthetic data...`
- `Comparing models...`
- `Backtest summary` at the end

If you see that, 🎉 **EVERYTHING IS WORKING!**

---

## 📥 STEP 3: Download real historical data (free, no API needed)

```powershell
python scripts/collect_data.py
```

This downloads free match results + odds from the last few years.

---

## 🏋️ STEP 4: Train a model

**Pick one:**

```powershell
python scripts/train_models.py --model poisson
```
```powershell
python scripts/train_models.py --model dixon_coles
```
```powershell
python scripts/train_models.py --model elo
```
```powershell
python scripts/train_models.py --model ml
```

---

## 📊 STEP 5: Run a backtest

```powershell
python scripts/run_backtest.py --model dixon_coles
```

This shows you how the model would have performed betting on past matches.

---

## 💰 STEP 6: Find value bets

```powershell
python scripts/find_value_bets.py data/processed/predictions.csv --min-edge 0.03
```

---

## 🔌 BONUS: Connect the live Odds API

### Get your API key
1. Go to: https://the-odds-api.com/
2. Click **Get API Key**
3. Sign up (free)
4. Copy your key (looks like `a1b2c3d4e5f6...`)

### Set the key in PowerShell
**Paste this, but replace `YOUR_KEY_HERE` with your real key:**

```powershell
$env:ODDS_API_KEY = "YOUR_KEY_HERE"
```

Example (fake key):
```powershell
$env:ODDS_API_KEY = "a1b2c3d4e5f6g7h8i9j0"
```

⚠️ **Important:** You must do this EVERY TIME you open a new PowerShell window.
(Or put it in a file — see "Future: Save the key permanently" below)

### Turn on the API in the config
1. Open `C:\Users\Pablo\Documents\soccer-betting-models\config\config.yaml`
2. Find the line that says `enabled: false` (under `odds_api`)
3. Change it to `enabled: true`
4. Save the file

### Test the API
Create a file `test_api.py` in the project folder with this:

```python
from soccer_betting.data.collectors import TheOddsAPICollector

collector = TheOddsAPICollector(sport="soccer_epl", regions="eu", markets="h2h")
odds = collector.fetch_odds()

print(f"Got {len(odds)} odds rows")
print(odds.head())
```

Then run:
```powershell
python test_api.py
```

If you see a table with team names and odds, ✅ **IT WORKS!**

---

## 🔁 CHEAT SHEET: Every future session

**Every time you want to use the project:**

1. Open PowerShell
2. Paste these 2 commands:

```powershell
cd C:\Users\Pablo\Documents\soccer-betting-models
.venv\Scripts\Activate.ps1
```

3. (Optional, if using the API) Set your key:
```powershell
$env:ODDS_API_KEY = "your_key_here"
```

4. Then run whatever you need:
```powershell
python examples/quickstart.py
python scripts/train_models.py --model dixon_coles
python scripts/run_backtest.py --model dixon_coles
```

---

## 🔐 FUTURE: Save the API key permanently (optional)

**So you don't have to type it every time:**

1. In the project folder, create a file called `.env` (note the dot at the start)
2. Inside, put:
   ```
   ODDS_API_KEY=your_actual_key_here
   ```
3. Install the python-dotenv package:
   ```powershell
   pip install python-dotenv
   ```
4. At the top of any script where you use the API, add:
   ```python
   from dotenv import load_dotenv
   load_dotenv()
   ```

Now it will read the key automatically! (And `.env` is in `.gitignore` so it won't upload to GitHub.)

---

## 🆘 TROUBLESHOOTING

| Problem | Fix |
|---------|-----|
| `python` not found | Use `py` instead of `python` in all commands |
| `(.venv)` not showing after activate | Re-run: `.venv\Scripts\Activate.ps1` |
| Red error about "execution policy" | Run as Admin: `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `pip` not found | Make sure `(.venv)` is showing. If not, activate first. |
| Can't find the folder | Check the path: `C:\Users\Pablo\Documents\soccer-betting-models` |
| API says "No key" | Re-run: `$env:ODDS_API_KEY = "your_key"` in the SAME PowerShell window |

---

## 📞 REMEMBER

- Always open PowerShell in the project folder: `cd C:\Users\Pablo\Documents\soccer-betting-models`
- Always activate first: `.venv\Scripts\Activate.ps1`
- Look for `(.venv)` at the start of the line — if you don't see it, activate again!

**Bookmark this file!** 🔖
```
