"""Live monitor: poll ESPN + The Odds API, run the model, and print +EV bets.

Each cycle it pulls in-progress games (ESPN) and live moneylines (The Odds API), matches them
by team-name token overlap, runs the calibrated model, and compares P(home win) to the
de-vigged market. Output is a live rich table (or plain summary lines if rich isn't installed).

    python -m liveedge.monitor --sport nba --model models/nba
    python -m liveedge.monitor --sport mlb --model models/mlb --min-ev 0.02   # only >2c/$1 edges
"""

from __future__ import annotations

import argparse
import time

from liveedge.engine import EdgeRead, evaluate
from liveedge.features import GameState, get_spec
from liveedge.live_odds import GameOdds, OddsClient
from liveedge.live_state import ESPNProvider
from liveedge.model import load_bundle, predict_prob
from liveedge.oddsmath import decimal_to_american, devig_two_way

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table

    _HAS_RICH = True
except ImportError:  # rich is recommended but optional
    _HAS_RICH = False

_STOPWORDS = {"the", "fc", "sc"}


def _tokenize(*names: str) -> set[str]:
    """Lowercase token set for a set of names, dropping filler words and periods."""
    toks: set[str] = set()
    for nm in names:
        for raw in str(nm).lower().replace(".", "").split():
            if raw and raw not in _STOPWORDS:
                toks.add(raw)
    return toks


def _match(state: GameState, odds_games: list[GameOdds]) -> GameOdds | None:
    """Match a live game to an odds game by largest team-token overlap (require >= 1)."""
    want = _tokenize(state.home_full or state.home_team, state.away_full or state.away_team)
    best: GameOdds | None = None
    best_n = 0
    for og in odds_games:
        n = len(want & _tokenize(og.home_team, og.away_team))
        if n > best_n:
            best_n, best = n, og
    return best if best_n >= 1 else None


def _collect(
    provider: ESPNProvider,
    client: OddsClient,
    spec,
    model,
    scaler,
    calibrator,
    anchors: dict[str, float],
    kelly: float,
    min_ev: float,
) -> list[tuple[GameState, GameOdds, EdgeRead]]:
    """One polling cycle: build (state, odds, read) triples for every matched live game."""
    live = provider.live_games()
    odds = client.moneylines()
    out: list[tuple[GameState, GameOdds, EdgeRead]] = []
    for state in live:
        og = _match(state, odds)
        if og is None:
            continue
        # Freeze the pregame prior the first time we see a game: de-vig its current market line
        # and keep that fair home prob. As the game moves, in-game features pull the model away
        # from this anchor. (Cleaner alternative: load Elo ratings trained alongside the model.)
        if state.game_id not in anchors:
            market_home, _ = devig_two_way(og.home_decimal, og.away_decimal)
            anchors[state.game_id] = market_home
        state.pregame_home_prob = anchors[state.game_id]

        model_home = float(predict_prob(model, scaler, [spec.vector(state)], calibrator)[0])
        read = evaluate(
            state.home_team,
            state.away_team,
            model_home,
            og.home_decimal,
            og.away_decimal,
            kelly_multiplier=kelly,
            min_ev=min_ev,
        )
        out.append((state, og, read))
    return out


def _build_table(sport: str, reads, last_remaining: str | None) -> "Table":
    quota = f"  |  odds quota left: {last_remaining}" if last_remaining else ""
    table = Table(title=f"live-edge · {sport.upper()}{quota}", expand=True)
    for col in ("Matchup", "State", "Model", "Market", "Best price", "EV/$1", "Bet"):
        table.add_column(col)
    if not reads:
        table.add_row("—", "waiting for matched in-progress games…", "", "", "", "", "")
        return table
    for state, og, read in reads:
        ah = int(decimal_to_american(og.home_decimal))
        aa = int(decimal_to_american(og.away_decimal))
        if read.best_side is None:
            bet, ev_style = "—", "dim"
        else:
            team = read.home_team if read.best_side == "home" else read.away_team
            bet = f"{team} ({read.kelly_fraction:.1%})"
            ev_style = "bold green"
        table.add_row(
            f"{read.away_team} @ {read.home_team}",
            state.description or "",
            f"{read.model_home_prob:.0%}",
            f"{read.market_home_prob:.0%}",
            f"{read.home_team} {ah:+d} / {read.away_team} {aa:+d}",
            f"[{ev_style}]{read.best_ev:+.1%}[/{ev_style}]",
            f"[{ev_style}]{bet}[/{ev_style}]",
        )
    return table


def run(
    sport: str,
    model_path: str,
    interval: float = 25.0,
    kelly: float = 0.25,
    min_ev: float = 0.0,
    regions: str = "us",
) -> None:
    """Poll forever, rendering a live +EV table. Transient API errors don't kill the loop."""
    model, scaler, calibrator, _meta = load_bundle(model_path)
    spec = get_spec(sport)
    provider = ESPNProvider(sport)
    client = OddsClient(sport, regions=regions)
    anchors: dict[str, float] = {}

    if _HAS_RICH:
        console = Console()
        with Live(_build_table(sport, [], None), console=console, refresh_per_second=4) as live:
            while True:
                try:
                    reads = _collect(
                        provider, client, spec, model, scaler, calibrator, anchors, kelly, min_ev
                    )
                    live.update(_build_table(sport, reads, client.last_remaining))
                except Exception as exc:  # noqa: BLE001, keep the loop alive on transient errors
                    console.log(f"[red]cycle error:[/red] {exc}")
                time.sleep(interval)
    else:
        while True:
            try:
                reads = _collect(
                    provider, client, spec, model, scaler, calibrator, anchors, kelly, min_ev
                )
                if not reads:
                    print("(no matched in-progress games)")
                for _state, _og, read in reads:
                    print(read.summary())
                if client.last_remaining:
                    print(f"odds quota left: {client.last_remaining}")
            except Exception as exc:  # noqa: BLE001
                print(f"cycle error: {exc}")
            time.sleep(interval)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Live +EV monitor (ESPN + The Odds API).")
    parser.add_argument("--sport", choices=["nfl", "nba", "mlb"], required=True)
    parser.add_argument("--model", required=True, help="bundle path, e.g. models/nba")
    parser.add_argument("--interval", type=float, default=25.0, help="seconds between polls")
    parser.add_argument("--kelly", type=float, default=0.25, help="fractional-Kelly multiplier")
    parser.add_argument(
        "--min-ev", type=float, default=0.0, dest="min_ev", help="e.g. 0.02 = only flag >2c/$1"
    )
    parser.add_argument("--regions", default="us")
    args = parser.parse_args(argv)
    run(
        args.sport,
        args.model,
        interval=args.interval,
        kelly=args.kelly,
        min_ev=args.min_ev,
        regions=args.regions,
    )


if __name__ == "__main__":
    main()
