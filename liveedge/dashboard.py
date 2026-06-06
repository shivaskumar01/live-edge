"""Local web dashboard for live-edge.

    python -m liveedge.dashboard --sport nba --model models/nba          # DEMO (no key)
    python -m liveedge.dashboard --sport mlb --model models/mlb --live   # LIVE (needs ODDS_API_KEY)

Serves a single page on http://localhost:PORT showing current +EV reads as cards plus the
model's reliability (calibration) curve read from the bundle. Pure stdlib http.server — no web
framework. DEMO mode (the default) runs the real model + engine over a handful of synthetic
in-game states so you can see exactly what the tool produces with zero setup; --live uses real
ESPN + The Odds API.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from liveedge.engine import EdgeRead, evaluate
from liveedge.features import GameState, get_spec
from liveedge.model import load_bundle, predict_prob
from liveedge.oddsmath import american_to_decimal, decimal_to_american, devig_two_way

# A few plausible in-game situations per sport for DEMO mode:
# (home, away, score_diff, fraction_elapsed, home_american, away_american)
_DEMO = {
    "nfl": [
        ("KC", "BUF", 4, 0.55, -160, 140),
        ("DAL", "PHI", -7, 0.78, 230, -280),
        ("SF", "SEA", 10, 0.40, -350, 280),
        ("GB", "CHI", -1, 0.20, 110, -130),
    ],
    "nba": [
        ("LAL", "BOS", 7, 0.62, -150, 130),
        ("DEN", "MIA", -4, 0.80, 175, -210),
        ("GSW", "PHX", 2, 0.45, -120, 100),
        ("MIL", "NYK", -9, 0.30, 240, -300),
    ],
    "mlb": [
        ("LAD", "SF", 2, 0.66, -170, 145),
        ("NYY", "BOS", -1, 0.80, 115, -135),
        ("HOU", "SEA", 3, 0.50, -150, 130),
        ("ATL", "NYM", -2, 0.40, 130, -150),
    ],
}

_CFG: dict = {}  # populated by run() before the server starts


def _state_label(sport: str, gs: GameState) -> str:
    if sport in ("nba", "nfl"):
        plen = 720.0 if sport == "nba" else 900.0
        within = max(0.0, gs.seconds_remaining - (4 - gs.period) * plen)
        return f"Q{gs.period} {int(within // 60)}:{int(within % 60):02d}"
    return f"{'Bot' if gs.is_bottom else 'Top'} {gs.inning}"


def _demo_state(sport: str, home: str, away: str, score_diff: int, frac: float) -> GameState:
    gs = GameState(sport=sport, home_team=home, away_team=away, score_diff=score_diff)
    if sport == "nba":
        gs.period = min(4, int(frac * 4) + 1)
        gs.seconds_remaining = (1 - frac) * 2880.0
    elif sport == "nfl":
        gs.period = min(4, int(frac * 4) + 1)
        gs.seconds_remaining = (1 - frac) * 3600.0
        gs.posteam_is_home = 1.0 if score_diff >= 0 else 0.0
        gs.down, gs.ydstogo, gs.yardline_100 = 2, 7, 55
        gs.home_timeouts, gs.away_timeouts = 2, 3
    elif sport == "mlb":
        gs.inning = min(9, int(frac * 9) + 1)
        gs.is_bottom, gs.outs, gs.on_first = 1.0, 1, 1.0
    gs.description = _state_label(sport, gs)
    return gs


def _card(state: GameState, read: EdgeRead) -> dict:
    bet = None
    if read.best_side is not None:
        team = read.home_team if read.best_side == "home" else read.away_team
        bet = {"team": team, "kelly_pct": round(read.kelly_fraction * 100, 1)}
    return {
        "matchup": f"{read.away_team} @ {read.home_team}",
        "state": state.description,
        "model_pct": round(read.model_home_prob * 100),
        "market_pct": round(read.market_home_prob * 100),
        "home_team": read.home_team,
        "away_team": read.away_team,
        "home_american": int(decimal_to_american(read.home_decimal)),
        "away_american": int(decimal_to_american(read.away_decimal)),
        "ev_pct": round(read.best_ev * 100, 1),
        "best_side": read.best_side,
        "bet": bet,
    }


def _demo_reads(c: dict) -> list[dict]:
    reads = []
    for home, away, sd, frac, ham, aam in _DEMO.get(c["sport"], []):
        hd, ad = american_to_decimal(ham), american_to_decimal(aam)
        gs = _demo_state(c["sport"], home, away, sd, frac)
        market_home, _ = devig_two_way(hd, ad)
        gs.pregame_home_prob = market_home  # anchor the prior to the opening line (like the monitor)
        model_home = float(predict_prob(c["model"], c["scaler"], [c["spec"].vector(gs)], c["calibrator"])[0])
        read = evaluate(home, away, model_home, hd, ad, kelly_multiplier=c["kelly"], min_ev=c["min_ev"])
        reads.append(_card(gs, read))
    return reads


def _live_reads(c: dict) -> tuple[list[dict], str | None]:
    from liveedge.monitor import _collect

    triples = _collect(
        c["provider"], c["client"], c["spec"], c["model"], c["scaler"],
        c["calibrator"], c["anchors"], c["kelly"], c["min_ev"],
    )
    return [_card(state, read) for state, _og, read in triples], c["client"].last_remaining


def _state_payload() -> dict:
    c = _CFG
    payload = {
        "sport": c["sport"],
        "mode": "LIVE" if c["live"] else "DEMO",
        "reads": [],
        "quota": None,
        "reliability": c["meta"].get("reliability"),
        "metrics": c["meta"].get("metrics"),
        "error": None,
    }
    try:
        if c["live"]:
            payload["reads"], payload["quota"] = _live_reads(c)
        else:
            payload["reads"] = _demo_reads(c)
    except Exception as exc:  # noqa: BLE001 — never 500 the page on a transient API error
        payload["error"] = f"{type(exc).__name__}: {exc}"
    return payload


_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>live-edge dashboard</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{color-scheme:dark}
body{margin:0;background:#0e1320;color:#cdd6f4;font:15px/1.4 -apple-system,Segoe UI,Roboto,sans-serif}
header{display:flex;align-items:center;gap:12px;padding:16px 22px;border-bottom:1px solid #1e2740}
h1{font-size:18px;margin:0;font-weight:650}
.badge{font-size:11px;font-weight:700;padding:3px 8px;border-radius:6px;letter-spacing:.5px}
.badge.demo{background:#3a2f12;color:#e0af68}.badge.live{background:#12331f;color:#9ece6a}
.muted{color:#8892b0;font-size:12px;margin-left:auto}
main{max-width:1000px;margin:0 auto;padding:22px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.6px;color:#8892b0;margin:26px 0 10px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:14px}
.card{background:#151c2e;border:1px solid #1e2740;border-radius:12px;padding:14px}
.match{font-weight:650;font-size:15px}.state{color:#8892b0;font-size:12px;margin-bottom:10px}
.row{display:flex;align-items:center;gap:6px;font-size:13px;margin:5px 0}
.row .vs{color:#8892b0;margin-left:6px}.row b{font-variant-numeric:tabular-nums}
.price{color:#8892b0;font-size:12px;margin:6px 0;font-variant-numeric:tabular-nums}
.good{color:#9ece6a}
.bet{margin-top:10px;padding:7px 9px;border-radius:8px;font-size:13px;font-weight:600}
.bet.good{background:#12331f;color:#9ece6a}.bet.none{background:#1a2030;color:#8892b0;font-weight:500}
.panel{display:flex;gap:24px;flex-wrap:wrap;align-items:center;background:#151c2e;border:1px solid #1e2740;border-radius:12px;padding:16px}
#metrics{color:#8892b0;font-size:13px;font-variant-numeric:tabular-nums}
.empty,.err{color:#8892b0;padding:14px;background:#151c2e;border:1px solid #1e2740;border-radius:12px}
.err{color:#f7768e}
</style></head><body>
<header><h1 id="title">live-edge</h1><span id="mode" class="badge demo">DEMO</span>
<span class="muted" id="quota"></span></header>
<main>
<h2>current reads</h2><div id="err"></div><div id="cards" class="grid"></div>
<h2>model calibration (reliability)</h2>
<div class="panel"><div id="relwrap"></div><div id="metrics"></div></div>
<p class="muted">predicted win-prob (x) vs. actual win rate (y); the dashed line is perfect
calibration. From the loaded model's held-out validation split.</p>
</main>
<script>
const fmtA=a=>a>0?'+'+a:''+a, fmtN=x=>x==null?'\\u2014':(+x).toFixed(3);
function card(c){
  const bet=c.bet?`<div class="bet good">\\u25b8 BET ${c.bet.team} \\u00b7 Kelly ${c.bet.kelly_pct}%</div>`
                 :`<div class="bet none">no +EV side</div>`;
  return `<div class="card"><div class="match">${c.matchup}</div>
    <div class="state">${c.state||''}</div>
    <div class="row"><span>model</span><b>${c.model_pct}%</b><span class="vs">vs market</span><b>${c.market_pct}%</b></div>
    <div class="price">${c.home_team} ${fmtA(c.home_american)} / ${c.away_team} ${fmtA(c.away_american)}</div>
    <div class="row"><span>EV/$1</span><b class="${c.best_side?'good':''}">${c.ev_pct>0?'+':''}${c.ev_pct}%</b></div>
    ${bet}</div>`;
}
function relSVG(rel){
  const S=240,pad=30,pl=S-2*pad,x=p=>pad+p*pl,y=p=>S-pad-p*pl;
  let s=`<svg viewBox="0 0 ${S} ${S}" width="240" height="240">`;
  s+=`<rect x="${pad}" y="${pad}" width="${pl}" height="${pl}" fill="none" stroke="#2a3550"/>`;
  s+=`<line x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}" stroke="#4a5578" stroke-dasharray="4 4"/>`;
  (rel||[]).forEach(r=>{const pred=r[3],act=r[4],n=r[2];
    if(pred==null||act==null||!n)return;
    s+=`<circle cx="${x(pred)}" cy="${y(act)}" r="${Math.max(2.5,Math.min(8,Math.sqrt(n)/9))}" fill="#7aa2f7" opacity="0.85"/>`;});
  s+=`<text x="${x(0.5)}" y="${S-6}" fill="#8892b0" font-size="10" text-anchor="middle">predicted</text>`;
  s+=`<text x="12" y="${y(0.5)}" fill="#8892b0" font-size="10" text-anchor="middle" transform="rotate(-90 12 ${y(0.5)})">actual</text></svg>`;
  return s;
}
async function refresh(){
  let s; try{s=await (await fetch('/api/state')).json();}catch(e){return;}
  title.textContent='live-edge \\u00b7 '+s.sport.toUpperCase();
  mode.textContent=s.mode; mode.className='badge '+(s.mode==='LIVE'?'live':'demo');
  quota.textContent=s.quota?('odds quota left: '+s.quota):'';
  err.innerHTML=s.error?`<div class="err">${s.error}</div>`:'';
  cards.innerHTML=(s.reads&&s.reads.length)?s.reads.map(card).join(''):'<div class="empty">no games to show right now</div>';
  if(s.reliability){relwrap.innerHTML=relSVG(s.reliability);const m=s.metrics||{};
    metrics.innerHTML=`n = ${m.n_rows??'?'}<br>log loss ${fmtN(m.log_loss)} (baseline ${fmtN(m.baseline_log_loss)})<br>ECE ${fmtN(m.ece)} \\u00b7 Brier ${fmtN(m.brier)}<br>temperature ${fmtN(m.temperature)}`;
  }else{relwrap.innerHTML='';metrics.textContent='No calibration data embedded in this bundle \\u2014 retrain to populate it.';}
}
refresh(); setInterval(refresh,20000);
</script></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        try:
            if self.path == "/" or self.path.startswith("/index"):
                self._send(200, "text/html; charset=utf-8", _PAGE.encode())
            elif self.path.startswith("/api/state"):
                self._send(200, "application/json", json.dumps(_state_payload()).encode())
            else:
                self._send(404, "text/plain", b"not found")
        except Exception as exc:  # noqa: BLE001
            self._send(500, "text/plain", str(exc).encode())

    def _send(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence per-request logging
        pass


def run(sport: str, model_path: str, *, live: bool = False, port: int = 8080,
        kelly: float = 0.25, min_ev: float = 0.0, regions: str = "us") -> None:
    model, scaler, calibrator, meta = load_bundle(model_path)
    global _CFG
    _CFG = {
        "sport": sport, "model": model, "scaler": scaler, "calibrator": calibrator,
        "meta": meta, "spec": get_spec(sport), "live": live, "kelly": kelly,
        "min_ev": min_ev, "anchors": {},
    }
    if live:
        from liveedge.live_odds import OddsClient
        from liveedge.live_state import ESPNProvider

        _CFG["provider"] = ESPNProvider(sport)
        _CFG["client"] = OddsClient(sport, regions=regions)

    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    mode = "LIVE" if live else "DEMO"
    print(f"live-edge dashboard [{mode}] for {sport} -> http://localhost:{port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Local web dashboard for live-edge.")
    p.add_argument("--sport", choices=["nfl", "nba", "mlb"], required=True)
    p.add_argument("--model", required=True, help="bundle path, e.g. models/nba")
    p.add_argument("--live", action="store_true", help="use real ESPN + Odds API (needs ODDS_API_KEY)")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--kelly", type=float, default=0.25)
    p.add_argument("--min-ev", type=float, default=0.0, dest="min_ev")
    p.add_argument("--regions", default="us")
    args = p.parse_args(argv)
    run(args.sport, args.model, live=args.live, port=args.port, kelly=args.kelly,
        min_ev=args.min_ev, regions=args.regions)


if __name__ == "__main__":
    main()
