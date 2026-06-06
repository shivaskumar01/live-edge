# live-edge

Per-sport **live win-probability models** (NFL / NBA / MLB) compared against the **live
sportsbook moneyline**, flagging positive-EV bets with a fractional-Kelly stake. It trains a
calibrated model on historical play-by-play, then at game time reads ESPN for the live game
state and The Odds API for the live line and prints rows like:

```
BOS @ LAL: bet LAL @ -190 | EV +40.4%/$1 | Kelly 19.2%
```

This is a **personal decision-support / EV tool for legal sports betting. It is not a
guaranteed-profit system.** A flagged "edge" is only real if the model is genuinely calibrated
on *live* games — see [Responsible use](#responsible-use) and [Limits & honesty](#limits--honesty).

---

## Responsible use

- **Bet only where it's legal for you, and only what you can afford to lose.** This is a tool
  for thinking about value, not a money machine.
- **Decision support, not a guarantee.** The model can be wrong, the data feeds can be stale or
  wrong, and the market is sharp.
- **An edge is only real if the model is calibrated on live games.** The included training runs
  on *synthetic* data prove the pipeline works — they say **nothing** about real-world accuracy.
  Before trusting a single live number, you must train on real historical data and verify
  calibration (predicted ≈ actual) on out-of-sample, real games.
- Fractional Kelly (default ¼) is used to blunt the damage from an over-confident or
  mis-calibrated model. It does not make a bad model safe.

---

## Quick start (zero setup, no API key)

```bash
python3 -m venv .venv
source .venv/bin/activate          # or use .venv/bin/python directly
pip install -r requirements.txt

pytest -q                          # ~50 tests: odds math, edge engine, Elo, feature contract, e2e

# Train on synthetic data — proves the train -> calibrate -> evaluate machinery with no network:
python -m liveedge.train --sport nba --synthetic 30000 --epochs 15
```

> ⚠️ **Synthetic numbers only prove the machinery.** The synthetic generator
> (`tools/simulate.py`) builds games that are calibrated *by construction*, so a good
> reliability table there confirms the code works — it is **not** a measure of real predictive
> power. Do not read anything into them beyond "the pipeline trains and calibrates."

A synthetic run prints per-epoch validation loss, final metrics, and a reliability table. What
you're checking is that **`log_loss` lands well below `baseline_logloss`** (the base-rate
predictor) and that **predicted ≈ actual** down the reliability table, e.g. (from a
`--synthetic 30000` run — *synthetic data, machinery check only, not real performance*):

```
          bin        n    pred  actual
  0.0-0.1     6196   0.031   0.030
  0.4-0.5     2607   0.450   0.450
  0.9-1.0     6220   0.968   0.972
```

---

## Train on real historical data

Each loader hits an external source (no API key needed for these three, but see the caveats):

```bash
# NFL — nflfastR play-by-play via nflreadpy (successor to the deprecated nfl_data_py)
python -m liveedge.train --sport nfl --seasons 2016 2017 2018 2019 2020 2021 2022 2023

# NBA — reconstructed from nba_api PlayByPlayV2 (rate-limited; capped to ~200 games/season)
python -m liveedge.train --sport nba --seasons 2021 2022

# MLB — pybaseball Statcast, one row per plate appearance (defaults to a ~3-week window/season)
python -m liveedge.train --sport mlb --seasons 2022 2023
```

Data-source caveats:

- **NFL (`nflreadpy`)** — cleanest source; down/distance is reliable from ~2001 on.
- **NBA (`nba_api` / stats.nba.com)** — rate-limits aggressively and often **blocks cloud /
  datacenter IPs**. There's a built-in sleep and a `max_games` cap so a first run completes; add
  a local cache and raise the cap for serious training.
- **MLB (`pybaseball` Statcast)** — pulls are heavy; the loader uses a short date window per
  season by default. Widen it and cache to parquet for real training.

Models are saved as a bundle: `models/<sport>.pt` (weights) + `models/<sport>.json` (feature
names, scaler stats, temperature, hidden sizes).

---

## Live monitor

Needs a free key from [the-odds-api.com](https://the-odds-api.com/):

```bash
cp .env.example .env       # then put your key in it, or:
export ODDS_API_KEY=...

python -m liveedge.monitor --sport nba --model models/nba
python -m liveedge.monitor --sport mlb --model models/mlb --min-ev 0.02   # only flag >2c/$1 edges
```

Each cycle it pulls in-progress games (ESPN) and live moneylines (The Odds API), matches them by
team-name overlap, runs the calibrated model, de-vigs the market, and renders a live table:

| Matchup | State | Model | Market | Best price | EV/$1 | Bet |
|---|---|---|---|---|---|---|
| LAA @ LAD | Bot 7th | 97% | 68% | LAD -250 / LAA +200 | +35.5% | LAD (22.2%) |

(That row is from a **synthetic-trained** model on a fabricated line — illustrating the format
and exactly why you must not trust an un-validated model: 97% vs a 68% market is the model being
naive about real baseball variance, not free money.)

The first time a game is seen, its pregame prior is anchored to the de-vigged opening line; as
the game moves, the in-game features pull the model away from that anchor and we compare to the
live market. Best prices are **line-shopped** (max across books) — the price you'd actually bet.

---

## How it works

```
HISTORICAL                                          LIVE (each cycle)
----------                                          -----------------
play-by-play (nflreadpy / nba_api / pybaseball)     ESPN scoreboard ──► GameState ─┐
        │                                           The Odds API ──► moneyline      │
        ▼                                                   │                       ▼
  data_*.py  ──► feature frame  (+ Elo pregame prior)       │                features.py
        │                                                   │                  (same path!)
        ▼                                                   ▼                       │
  train.py: WinProbNet (MLP) ─► temperature-scale ─► bundle │                       ▼
        │                                                   │            model ─► P(home win)
        ▼                                                   ▼                       │
   models/<sport>.{pt,json}  ───────────────────►  engine.evaluate(model_p, devig(market)) ◄─┘
                                                            │
                                                            ▼
                                              EdgeRead ─► live table / summary
```

Key design choices:

- **One feature contract** (`features.py`). Historical rows and live `GameState`s produce
  feature vectors through the *exact same* code and ordering — no train/serve skew.
- **Calibration first.** The model is judged on log loss / Brier / ECE / a reliability table and
  is temperature-scaled on a held-out split. "70%" should win ~70% of the time.
- **Elo pregame prior** built only from past results (no betting lines needed), computed
  strictly *before* each game's result is applied so a game never leaks into its own feature.
- **Odds math is isolated and exhaustively tested** (`oddsmath.py`) — a sign error there is the
  difference between +EV and a guaranteed loss.

---

## Project layout

```
liveedge/
  oddsmath.py     odds conversions, de-vig, EV, Kelly (pure, heavily tested)
  features.py     GameState + per-sport FeatureSpec + state_to_features (the feature contract)
  elo.py          pregame win-prob prior
  model.py        WinProbNet, scaler, temperature scaler, metrics, save/load
  train.py        training + calibration + CLI (python -m liveedge.train)
  engine.py       model prob + live line -> EdgeRead
  data_nfl/nba/mlb.py   historical loaders
  live_state.py   ESPN -> live GameState (all 3 sports)
  live_odds.py    The Odds API -> live moneylines
  monitor.py      live CLI table (python -m liveedge.monitor)
tools/
  simulate.py     synthetic game generator — TESTING/DEMO ONLY, never imported by the monitor
  xgb_baseline.py XGBoost baseline vs. the MLP on the same features/metrics (experiment)
tests/            oddsmath, engine, pipeline (elo + contract + synthetic e2e)
```

---

## Limits & honesty

- **The synthetic results prove the pipeline, not predictive power.** Real calibration on real,
  out-of-sample games is on you to verify before trusting anything live.
- **ESPN's site API is unofficial** and can change shape or rate-limit without notice.
- **`nba_api` rate-limits and often blocks cloud IPs.** Expect to run NBA pulls from a
  residential IP and to cache.
- **Statcast pulls are heavy** — window + cache them.
- **MLB top/bottom-of-inning is parsed from ESPN status text** and is approximate; the
  MLB-StatsAPI live feed gives it cleanly.
- **Team-name matching is token-overlap**, not a maintained alias map; an exotic name could mis-
  or fail-to-match.
- **MLP vs. XGBoost.** Gradient-boosted trees are the usual strong baseline for tabular
  win-probability (and are what nflfastR uses), so `tools/xgb_baseline.py` trains one on the
  *same* features and metrics for an apples-to-apples comparison. In one out-of-sample NFL test
  (train 2010–22, score 2023) the temperature-calibrated MLP actually *edged* an **untuned,
  uncalibrated** XGBoost on both log loss (0.474 vs 0.493) and ECE (0.019 vs 0.024) — a reminder
  that calibration can matter as much as the model family. That is **not** the final word:
  tuning XGBoost's hyperparameters and adding isotonic/Platt calibration is the natural next step
  before concluding anything. Run it yourself:
  `python -m tools.xgb_baseline --sport nfl --seasons 2010 … 2022 --test-season 2023`.

---

Personal project. No warranty. Bet responsibly, and only where legal.
