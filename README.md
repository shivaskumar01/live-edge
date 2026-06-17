# live-edge

I train per-sport win-probability models (NFL, NBA, MLB) and compare them against
the live sportsbook moneyline to flag positive-EV bets with a fractional-Kelly stake.
Each model learns from historical play-by-play. At game time it reads ESPN for the
live game state and The Odds API for the live line, then prints rows like:

```
BOS @ LAL: bet LAL @ -190 | EV +40.4%/$1 | Kelly 19.2%
```

This is a personal decision-support tool for legal sports betting, not a
guaranteed-profit system. A flagged "edge" is only real if the model is genuinely
calibrated on *live* games. See [Responsible use](#responsible-use) and
[Limits & honesty](#limits--honesty).

---

## Responsible use

- Bet only where it's legal for you, and only what you can afford to lose. This is a tool
  for thinking about value, not a money machine.
- Decision support, not a guarantee. The model can be wrong, the data feeds can be stale or
  wrong, and the market is sharp.
- An edge is only real if the model is calibrated on live games. The included training runs
  on *synthetic* data prove the pipeline works, they say nothing about real-world accuracy.
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

# Train on synthetic data, proves the train -> calibrate -> evaluate machinery with no network:
python -m liveedge.train --sport nba --synthetic 30000 --epochs 15

# Then open the multi-sport dashboard in your browser (no API key needed):
python -m liveedge.dashboard                                   # http://localhost:8080
```

> Synthetic numbers only prove the machinery. The synthetic generator
> (`tools/simulate.py`) builds games that are calibrated *by construction*, so a good
> reliability table there confirms the code works, it is not a measure of real predictive
> power. Do not read anything into them beyond "the pipeline trains and calibrates."

A synthetic run prints per-epoch validation loss, final metrics, and a reliability table. What
you're checking is that `log_loss` lands well below `baseline_logloss` (the base-rate
predictor) and that predicted ≈ actual down the reliability table, e.g. (from a
`--synthetic 30000` run, *synthetic data, machinery check only, not real performance*):

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
# NFL, nflfastR play-by-play via nflreadpy (successor to the deprecated nfl_data_py)
python -m liveedge.train --sport nfl --seasons 2016 2017 2018 2019 2020 2021 2022 2023

# NBA, reconstructed from nba_api PlayByPlayV3 (rate-limited; cached to .cache/, resumable)
python -m liveedge.train --sport nba --seasons 2023                    # ~200 games (quick)
python -m liveedge.train --sport nba --seasons 2023 --max-games 2000   # full season (slow 1st run, then cached)

# MLB, pybaseball Statcast, one row per plate appearance (defaults to a ~3-week window/season)
python -m liveedge.train --sport mlb --seasons 2022 2023
```

Data-source caveats:

- NFL (`nflreadpy`), cleanest source; down/distance is reliable from ~2001 on.
- NBA (`nba_api` / stats.nba.com), rate-limits aggressively and may block datacenter
  IPs. Uses PlayByPlayV3 (V2 is defunct), a built-in sleep, and a `max_games` cap. Every pull
  is cached to `.cache/` (parquet), so a full-season pull (`--max-games 2000`) is a one-time
  cost and resumes if throttled mid-way.
- MLB (`pybaseball` Statcast), pulls are heavy; the loader uses a short date window per
  season by default and enables pybaseball's on-disk cache. Widen the window for real training.

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

(That row is from a synthetic-trained model on a fabricated line, illustrating the format
and exactly why you must not trust an un-validated model: 97% vs a 68% market is the model being
naive about real baseball variance, not free money.)

The first time a game is seen, its pregame prior is anchored to the de-vigged opening line; as
the game moves, the in-game features pull the model away from that anchor and we compare to the
live market. Best prices are line-shopped (max across books), the price you'd actually bet.

---

## Dashboard (web)

A local multi-sport web view: browse real games (today's schedule, in-progress, and finals)
for every sport that has a trained model, by tab, and click through them. In-progress games show
the model's live win probability; each tab also shows that model's calibration curve.

```bash
python -m liveedge.dashboard            # http://localhost:8080 (no key needed)
python -m liveedge.dashboard --port 8090
```

Games come from ESPN (free). With an `ODDS_API_KEY` (env var or a local `.env`), each game pulls
every US book and computes line-shopping value: the best price per side, the no-vig market
consensus (fair prob), and the EV of the best price vs that consensus, which flags soft / off-
market books. A per-book price table shows where the best number is. For in-progress games the
win-probability model's edge vs the best line is also shown. Spreads & totals show best-price line
shopping (the model doesn't price them, no edge shown). Player props are omitted (a win-prob
model can't price them). Note: against efficient/closing lines the value EV is usually ~0 or
slightly negative, positive EV (a real edge) is the exception you're hunting, and games are sorted
by value so it floats to the top. Odds cost API credits, so they're fetched only on tab-click /
manual refresh and cached ~3 min; the 30s auto-refresh updates ESPN only. Pure stdlib `http.server`.

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

- One feature contract (`features.py`). Historical rows and live `GameState`s produce
  feature vectors through the *exact same* code and ordering, no train/serve skew.
- Calibration first. The model is judged on log loss / Brier / ECE / a reliability table and
  is temperature-scaled on a held-out split. "70%" should win ~70% of the time.
- Elo pregame prior built only from past results (no betting lines needed), computed
  strictly *before* each game's result is applied so a game never leaks into its own feature.
- Odds math is isolated and tested hard (`oddsmath.py`). A sign error there is the
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
  dashboard.py    local web dashboard (python -m liveedge.dashboard)
tools/
  simulate.py     synthetic game generator, TESTING/DEMO ONLY, never imported by the monitor
  xgb_baseline.py XGBoost baseline vs. the MLP on the same features/metrics (experiment)
tests/            oddsmath, engine, pipeline (elo + contract + synthetic e2e)
```

---

## Limits & honesty

- The synthetic results prove the pipeline, not predictive power. Real calibration on real,
  out-of-sample games is on you to verify before trusting anything live.
- ESPN's site API is unofficial and can change shape or rate-limit without notice.
- `nba_api` rate-limits and may block datacenter IPs. Pulls are cached to `.cache/`, so a
  throttled run resumes rather than restarting; a residential IP helps for big pulls.
- Statcast pulls are heavy, window + cache them.
- MLB top/bottom-of-inning is parsed from ESPN status text and is approximate; the
  MLB-StatsAPI live feed gives it cleanly.
- Team-name matching is token-overlap, not a maintained alias map; an exotic name could mis-
  or fail-to-match.
- MLP vs. XGBoost (measured). Gradient-boosted trees are the usual strong baseline for
  tabular win-prob (it's what nflfastR uses), so `tools/xgb_baseline.py` trains one on the *same*
  features/metrics and compares out-of-sample by season. On the NFL test (train 2010-22, score
  2023) the temperature-calibrated MLP won across the board and XGBoost didn't catch it even
  after a depth bump and the *same* temperature calibration, OOS log loss / ECE: MLP 0.474 /
  0.019, XGB-depth4 0.485 / 0.025, XGB-depth6 0.493 / 0.025, XGB-depth6+temp 0.500 / 0.029. Two
  honest wrinkles: deeper trees *overfit* the training seasons (worse OOS), and temperature
  scaling *helped the MLP but hurt XGBoost*, its log-loss-trained probabilities were already
  roughly calibrated, so rescaling (T≈0.82) just transferred a wrong adjustment into a new
  season. Caveats: one small, smooth 9-feature problem and a single test season, XGBoost
  typically shines with richer features, and shape calibration (isotonic; needs scikit-learn, not
  installed here) could help it more than temperature. Reproduce:
  `python -m tools.xgb_baseline --sport nfl --seasons 2010 … 2022 --test-season 2023`.

---

Personal project. No warranty. Bet responsibly, and only where legal.
