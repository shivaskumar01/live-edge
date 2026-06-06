"""Local multi-sport web dashboard for live-edge.

    python -m liveedge.dashboard                 # http://localhost:8080
    python -m liveedge.dashboard --port 8090

Browse REAL games (today's schedule + in-progress + finals) for every sport with a trained model
(NFL / NBA / MLB), by tab. Games come from ESPN (free).

The headline feature is LINE-SHOPPING VALUE: with an ODDS_API_KEY set (env or local .env), each
game pulls every US book, finds the best price per side, computes the no-vig market consensus as
fair value, and flags where the best book beats consensus (a soft line / "leak"). That's a
model-free, market-vs-market edge and works pregame. Separately, for IN-PROGRESS games, the
win-probability model's edge vs the best line is shown. Spreads & totals show best-price line
shopping. Player props are omitted (a win-prob model can't price them).

Odds cost API credits (3 per fetch), so they're fetched only on tab-click / manual refresh and
cached ~3 min; the 30s auto-refresh updates ESPN only. Pure stdlib http.server.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from liveedge.engine import evaluate
from liveedge.features import get_spec
from liveedge.live_state import ESPNProvider
from liveedge.model import load_bundle, predict_prob
from liveedge.oddsmath import decimal_to_american, devig_two_way

_SPORTS = ("nfl", "nba", "mlb")
_STOPWORDS = {"the", "fc", "sc"}
_ODDS_TTL = 180.0
_STATE_ORDER = {"in": 0, "pre": 1, "post": 2}
_MODELS: dict = {}
_ODDS_CACHE: dict = {}


def _load_dotenv(path: str = ".env") -> None:
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def _load_models(models_dir: str) -> None:
    for sport in _SPORTS:
        base = os.path.join(models_dir, sport)
        if os.path.exists(base + ".json"):
            model, scaler, calibrator, meta = load_bundle(base)
            _MODELS[sport] = {"model": model, "scaler": scaler, "calibrator": calibrator,
                              "meta": meta, "spec": get_spec(sport), "provider": ESPNProvider(sport)}


def _has_key() -> bool:
    return bool(os.environ.get("ODDS_API_KEY"))


def _sport_games(sport: str) -> list[dict]:
    m = _MODELS[sport]
    prov = m["provider"]
    out = []
    for event in prov._fetch().get("events", []):
        comps = event.get("competitions") or []
        if not comps:
            continue
        comp = comps[0]
        status = comp.get("status") or {}
        state = (status.get("type") or {}).get("state") or "pre"
        competitors = comp.get("competitors") or []
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        ht, at = home.get("team") or {}, away.get("team") or {}
        g = {
            "id": str(event.get("id", "")), "state": state,
            "detail": (status.get("type") or {}).get("shortDetail", "") or "",
            "home": ht.get("abbreviation", "HOME"), "away": at.get("abbreviation", "AWAY"),
            "home_full": ht.get("displayName", ""), "away_full": at.get("displayName", ""),
            "home_logo": ht.get("logo", ""), "away_logo": at.get("logo", ""),
            "home_score": int(home.get("score") or 0), "away_score": int(away.get("score") or 0),
            "model_home_prob": None, "_mp_f": None, "odds": None,
        }
        if state == "in":
            gs = prov._parse(event, comp)
            if gs is not None:
                p = float(predict_prob(m["model"], m["scaler"], [m["spec"].vector(gs)], m["calibrator"])[0])
                g["model_home_prob"], g["_mp_f"] = round(p * 100), p
        out.append(g)
    return out


# --------------------------------------------------------------------------------------
# Odds + line-shopping value
# --------------------------------------------------------------------------------------


def _tok(*names: str) -> set[str]:
    t: set[str] = set()
    for n in names:
        for w in str(n).lower().replace(".", "").split():
            if w and w not in _STOPWORDS:
                t.add(w)
    return t


def _match_odds(g: dict, odds_list: list[dict]) -> dict | None:
    want = _tok(g.get("home_full") or g["home"], g.get("away_full") or g["away"])
    best, best_n = None, 0
    for od in odds_list:
        n = len(want & _tok(od["home_team"], od["away_team"]))
        if n > best_n:
            best_n, best = n, od
    return best if best_n >= 1 else None


def _am(dec) -> int | None:
    return int(decimal_to_american(dec)) if dec and dec > 1 else None


def _moneyline_value(books: list[dict]) -> dict | None:
    """Best price per side + no-vig consensus fair prob + EV of best price vs consensus."""
    valid = [b for b in books if b.get("home_dec", 0) > 1 and b.get("away_dec", 0) > 1]
    if not valid:
        return None
    fair_home = [devig_two_way(b["home_dec"], b["away_dec"])[0] for b in valid]
    ch = sum(fair_home) / len(fair_home)
    ca = 1 - ch
    bh = max(valid, key=lambda b: b["home_dec"])
    ba = max(valid, key=lambda b: b["away_dec"])
    ev_h, ev_a = ch * bh["home_dec"] - 1, ca * ba["away_dec"] - 1
    home_best = ev_h >= ev_a
    return {
        "n_books": len(valid),
        "cons_home": round(ch * 100, 1), "cons_away": round(ca * 100, 1),
        "ev_home": round(ev_h * 100, 1), "ev_away": round(ev_a * 100, 1),
        "best_side": "home" if home_best else "away",
        "best_ev": round((ev_h if home_best else ev_a) * 100, 1),
        "best_book": bh["book"] if home_best else ba["book"],
        "best_am": _am(bh["home_dec"]) if home_best else _am(ba["away_dec"]),
        "best_home_am": _am(bh["home_dec"]), "best_home_book": bh["book"],
        "best_away_am": _am(ba["away_dec"]), "best_away_book": ba["book"],
        "books": [
            {"book": b["book"], "home_am": _am(b["home_dec"]), "away_am": _am(b["away_dec"]),
             "best_home": b is bh, "best_away": b is ba}
            for b in sorted(valid, key=lambda b: b["book"].lower())
        ],
    }


def _best_spread(spreads: list[dict]) -> dict | None:
    valid = [s for s in spreads if s.get("home_dec", 0) > 1 and s.get("away_dec", 0) > 1]
    if not valid:
        return None
    bh = max(valid, key=lambda s: s["home_dec"])
    ba = max(valid, key=lambda s: s["away_dec"])
    return {"home_point": bh["home_point"], "home_am": _am(bh["home_dec"]), "home_book": bh["book"],
            "away_point": ba["away_point"], "away_am": _am(ba["away_dec"]), "away_book": ba["book"]}


def _best_total(totals: list[dict]) -> dict | None:
    valid = [t for t in totals if t.get("over_dec", 0) > 1 and t.get("under_dec", 0) > 1]
    if not valid:
        return None
    bo = max(valid, key=lambda t: t["over_dec"])
    bu = max(valid, key=lambda t: t["under_dec"])
    return {"over_point": bo["point"], "over_am": _am(bo["over_dec"]), "over_book": bo["book"],
            "under_point": bu["point"], "under_am": _am(bu["under_dec"]), "under_book": bu["book"]}


def _attach_odds(g: dict, od: dict) -> dict:
    ml = _moneyline_value(od["h2h"])
    model_ev = None
    if g["state"] == "in" and g.get("_mp_f") is not None and ml:
        valid = [b for b in od["h2h"] if b.get("home_dec", 0) > 1 and b.get("away_dec", 0) > 1]
        hd = max(b["home_dec"] for b in valid)
        ad = max(b["away_dec"] for b in valid)
        r = evaluate(g["home"], g["away"], g["_mp_f"], hd, ad, kelly_multiplier=0.25, min_ev=0.0)
        model_ev = {"best_side": r.best_side, "ev_pct": round(r.best_ev * 100, 1),
                    "kelly_pct": round(r.kelly_fraction * 100, 1),
                    "bet_team": (g["home"] if r.best_side == "home" else g["away"]) if r.best_side else None}
    return {"is_live": od["is_live"], "moneyline": ml,
            "spread": _best_spread(od["spreads"]), "total": _best_total(od["totals"]),
            "model_ev": model_ev}


def _get_odds(sport: str, force: bool):
    now = time.time()
    cached = _ODDS_CACHE.get(sport)
    if cached and not force and (now - cached["ts"]) < _ODDS_TTL:
        return cached["games"], cached["quota"], None
    try:
        from liveedge.live_odds import OddsClient

        client = OddsClient(sport)
        games = client.full_markets()
        _ODDS_CACHE[sport] = {"ts": now, "games": games, "quota": client.last_remaining}
        return games, client.last_remaining, None
    except Exception as exc:  # noqa: BLE001
        if cached:
            return cached["games"], cached["quota"], f"{type(exc).__name__}: {exc} (cached)"
        return None, None, f"{type(exc).__name__}: {exc}"


def _ev_sort_key(g: dict) -> float:
    ml = (g.get("odds") or {}).get("moneyline") if g.get("odds") else None
    return ml["best_ev"] if ml else -99.0


def _sport_payload(sport: str, odds_mode: str) -> dict:
    m = _MODELS[sport]
    quota = odds_err = None
    try:
        games, err = _sport_games(sport), None
    except Exception as exc:  # noqa: BLE001
        games, err = [], f"{type(exc).__name__}: {exc}"

    if _has_key() and games:
        if odds_mode in ("1", "force"):
            odds_list, quota, odds_err = _get_odds(sport, force=(odds_mode == "force"))
        else:
            cached = _ODDS_CACHE.get(sport)
            odds_list, quota = (cached["games"], cached["quota"]) if cached else (None, None)
        if odds_list:
            for g in games:
                od = _match_odds(g, odds_list)
                if od:
                    g["odds"] = _attach_odds(g, od)

    games.sort(key=lambda g: (_STATE_ORDER.get(g["state"], 3), -_ev_sort_key(g)))
    for g in games:
        g.pop("_mp_f", None)
    return {"sport": sport, "games": games, "error": err, "odds_err": odds_err,
            "has_key": _has_key(), "quota": quota,
            "reliability": m["meta"].get("reliability"), "metrics": m["meta"].get("metrics")}


_PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>live-edge</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{color-scheme:dark}*{box-sizing:border-box}
body{margin:0;color:#e6ebf5;font:15px/1.45 -apple-system,Segoe UI,Roboto,sans-serif;
  background:radial-gradient(1200px 600px at 80% -10%,#16213f 0%,#0a0e1a 55%) fixed,#0a0e1a}
header{position:sticky;top:0;z-index:5;padding:14px 22px;display:flex;align-items:center;gap:14px;flex-wrap:wrap;
  background:rgba(10,14,26,.82);backdrop-filter:blur(10px);border-bottom:1px solid #1c2742}
h1{margin:0;font-size:20px;font-weight:800;letter-spacing:.4px;background:linear-gradient(90deg,#7aa2f7,#9ece6a);-webkit-background-clip:text;background-clip:text;color:transparent}
.tabs{display:flex;gap:8px}
.tab{padding:7px 16px;border-radius:999px;background:#121a2e;border:1px solid #1e2b49;cursor:pointer;font-weight:700;font-size:13px;color:#9fb0d4;transition:.15s}
.tab:hover{border-color:#3a5da0}.tab.on{background:linear-gradient(180deg,#244680,#1a3160);border-color:#3f6fc0;color:#fff;box-shadow:0 4px 16px -6px #2d5aa0}
.right{margin-left:auto;display:flex;align-items:center;gap:10px;font-size:12px;color:#8c98b8}
.chip{padding:5px 10px;border-radius:999px;background:#121a2e;border:1px solid #1e2b49}
button{background:#15203a;border:1px solid #2d416e;color:#bccbe8;border-radius:999px;padding:6px 13px;cursor:pointer;font-size:12px;font-weight:600}
button:hover{border-color:#7aa2f7;color:#fff}
main{display:grid;grid-template-columns:minmax(300px,380px) 1fr;gap:20px;max-width:1180px;margin:0 auto;padding:22px}
@media(max-width:820px){main{grid-template-columns:1fr}}
h2{font-size:11px;text-transform:uppercase;letter-spacing:.7px;color:#7e8bad;margin:0 0 11px;font-weight:700}
.list{display:flex;flex-direction:column;gap:10px}
.game{background:linear-gradient(180deg,#131c31,#101728);border:1px solid #1d2942;border-radius:14px;padding:12px 14px;cursor:pointer;transition:.14s}
.game:hover{border-color:#3a5da0;transform:translateY(-1px)}.game.sel{border-color:#7aa2f7;box-shadow:0 0 0 1px #7aa2f7 inset,0 8px 24px -12px #2d5aa0}
.row{display:flex;align-items:center;gap:9px}
.logo{width:22px;height:22px;object-fit:contain;filter:drop-shadow(0 1px 2px #000)}
.mu{font-weight:700;font-size:14px}.spacer{flex:1}
.badge{font-size:9.5px;font-weight:800;letter-spacing:.5px;padding:2px 7px;border-radius:5px}
.b-in{background:#10331e;color:#9ece6a;box-shadow:0 0 0 1px #1d5c33 inset}.b-pre{background:#1b2747;color:#9fb3e0}.b-post{background:#241a24;color:#a98aa0}
.sub{color:#8c98b8;font-size:12px;margin-top:6px;display:flex;justify-content:space-between}
.vchip{font-weight:800;font-size:12px;padding:2px 8px;border-radius:6px;font-variant-numeric:tabular-nums}
.bar{height:6px;border-radius:4px;background:#1c2742;margin-top:9px;overflow:hidden}.bar>i{display:block;height:100%;background:linear-gradient(90deg,#7aa2f7,#9ece6a)}
.detail{background:linear-gradient(180deg,#131c31,#0f1626);border:1px solid #1d2942;border-radius:18px;padding:22px;min-height:180px}
.dhead{display:flex;align-items:center;gap:14px}.dhead .logo{width:42px;height:42px}
.dmu{font-size:21px;font-weight:800}.st{color:#8c98b8;margin-top:3px;font-size:13px}
.hero{margin-top:18px;border-radius:16px;padding:18px;background:linear-gradient(135deg,#15233f,#101a30);border:1px solid #243a66;position:relative;overflow:hidden}
.hero .lab{font-size:10.5px;letter-spacing:.7px;text-transform:uppercase;color:#8c98b8;font-weight:700}
.hero .ev{font-size:40px;font-weight:900;line-height:1;font-variant-numeric:tabular-nums;margin:6px 0 2px}
.hero .bet{font-size:15px;font-weight:700}.hero .bet b{color:#fff}
.meter{height:8px;border-radius:6px;background:#1c2742;margin-top:12px;overflow:hidden}.meter>i{display:block;height:100%}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:14px}
.tile{background:#101a30;border:1px solid #1d2942;border-radius:12px;padding:11px 13px}
.tile .t{font-size:10.5px;text-transform:uppercase;letter-spacing:.5px;color:#7e8bad;font-weight:700;margin-bottom:5px}
.tile .v{font-size:13.5px;font-variant-numeric:tabular-nums}.tile .bk{color:#8c98b8;font-size:11px}
.books{margin-top:14px;border:1px solid #1d2942;border-radius:12px;overflow:hidden}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:7px 12px;text-align:left;font-variant-numeric:tabular-nums}
th{font-size:10px;text-transform:uppercase;letter-spacing:.5px;color:#7e8bad;background:#101a30}
tbody tr{border-top:1px solid #161f36}td.bk{color:#aeb8d4}
td.best{color:#0c0f18;background:linear-gradient(90deg,#f0c674,#e0af68);font-weight:800;border-radius:4px}
.model{margin-top:14px;padding:12px 14px;border-radius:12px;background:#101a30;border:1px solid #1d2942;font-size:13px}
.note{color:#7e8bad;font-size:11px;margin-top:10px;line-height:1.5}
.panel{margin-top:18px;background:linear-gradient(180deg,#131c31,#0f1626);border:1px solid #1d2942;border-radius:16px;padding:18px;display:flex;gap:24px;flex-wrap:wrap;align-items:center}
.muted{color:#7e8bad}.empty{color:#7e8bad;padding:16px}
.calhead{display:flex;align-items:center;gap:12px;margin:22px 0 8px}
.verdict{font-size:12px;font-weight:800;padding:4px 11px;border-radius:999px;border:1px solid}
.calwrap{display:grid;grid-template-columns:300px 1fr;gap:20px;background:linear-gradient(180deg,#131c31,#0f1626);border:1px solid #1d2942;border-radius:16px;padding:18px}
@media(max-width:820px){.calwrap{grid-template-columns:1fr}}
.calplot{display:flex;flex-direction:column;gap:8px;min-width:0}
.legend{display:flex;gap:13px;flex-wrap:wrap;font-size:10.5px;color:#8c98b8;align-items:center}
.legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px;vertical-align:middle}
.calgrid{display:grid;grid-template-columns:1fr 1fr;gap:10px;align-content:start}
@media(max-width:520px){.calgrid{grid-template-columns:1fr}}
.mcard{background:#101a30;border:1px solid #1d2942;border-radius:12px;padding:11px 13px}
.mcard .mt{font-size:10.5px;text-transform:uppercase;letter-spacing:.5px;color:#7e8bad;font-weight:700}
.mcard .mv{font-size:23px;font-weight:800;font-variant-numeric:tabular-nums;margin:3px 0 5px}
.mcard .mx{font-size:11.5px;color:#8c98b8;line-height:1.45}
</style></head><body>
<header><h1>live·edge</h1><div class="tabs" id="tabs"></div>
<div class="right"><span class="chip" id="quota">…</span><button onclick="if(SPORT)load(SPORT,'force')">↻ refresh odds</button></div></header>
<main>
  <div><h2 id="listhdr">games</h2><div id="err"></div><div class="list" id="list"></div></div>
  <div><h2>game detail</h2><div class="detail" id="detail"><div class="empty">pick a game on the left.</div></div>
    <div id="calib" style="margin-top:22px"></div>
  </div>
</main>
<script>
let DATA=null,SEL=null,SPORT=null;
const fmtA=a=>a==null?'—':(a>0?'+'+a:''+a), f3=x=>x==null?'—':(+x).toFixed(3), sgn=p=>p==null?'':(p>0?'+'+p:''+p);
const evCol=e=>e>=2?'#f0c674':e>=0.5?'#9ece6a':e>0?'#7aa2f7':'#6b7394';
const evBg=e=>e>=2?'rgba(240,198,116,.16)':e>=0.5?'rgba(158,206,106,.15)':e>0?'rgba(122,162,247,.14)':'rgba(120,130,160,.13)';
function badge(s){return s==='in'?'<span class="badge b-in">● LIVE</span>':s==='post'?'<span class="badge b-post">FINAL</span>':'<span class="badge b-pre">UPCOMING</span>';}
function logo(u){return u?`<img class="logo" src="${u}" onerror="this.style.visibility='hidden'">`:'<span class="logo"></span>';}
function gcard(g,i){
  const ml=g.odds&&g.odds.moneyline;
  const score=(g.state!=='pre')?`${g.away} ${g.away_score} · ${g.home} ${g.home_score}`:'';
  const bar=(g.model_home_prob!=null)?`<div class="bar"><i style="width:${g.model_home_prob}%"></i></div>`:'';
  const v=ml?`<span class="vchip" style="color:${evCol(ml.best_ev)};background:${evBg(ml.best_ev)}">${sgn(ml.best_ev)}%</span>`:'';
  return `<div class="game ${i===SEL?'sel':''}" onclick="sel(${i})">
    <div class="row">${logo(g.away_logo)}${logo(g.home_logo)}<span class="mu">${g.away} @ ${g.home}</span><span class="spacer"></span>${badge(g.state)}</div>
    <div class="sub"><span>${g.detail||''}</span><span>${v||score}</span></div>${bar}</div>`;
}
function bookTable(ml){
  let rows=ml.books.map(b=>`<tr><td class="bk">${b.book}</td><td class="${b.best_away?'best':''}">${fmtA(b.away_am)}</td><td class="${b.best_home?'best':''}">${fmtA(b.home_am)}</td></tr>`).join('');
  return `<div class="books"><table><thead><tr><th>book (${ml.n_books})</th><th>away ML</th><th>home ML</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}
function valueHero(g){
  const o=g.odds;
  if(!DATA.has_key) return `<div class="hero"><div class="lab">line-shopping value</div><div class="bet" style="margin-top:8px">set <b>ODDS_API_KEY</b> to pull live odds across books.</div></div>`;
  if(!o||!o.moneyline) return `<div class="hero"><div class="lab">line-shopping value</div><div class="bet" style="margin-top:8px">no sportsbook line matched for this game.</div></div>`;
  const ml=o.moneyline, ev=ml.best_ev, team=ml.best_side==='home'?g.home:g.away, col=evCol(ev), w=Math.max(3,Math.min(100,ev*18));
  const verdict=ev>=2?'strong value — soft line':ev>=0.5?'shop the best price here':ev>0?'slight edge from shopping':'no standout value';
  return `<div class="hero" style="border-color:${col}55">
    <div class="lab">best moneyline value · line shopping</div>
    <div class="ev" style="color:${col}">${sgn(ev)}%<span style="font-size:14px;color:#8c98b8;font-weight:600"> EV vs consensus</span></div>
    <div class="bet">bet <b>${team}</b> ${fmtA(ml.best_am)} at <b style="color:${col}">${ml.best_book}</b> · ${verdict}</div>
    <div class="meter"><i style="width:${w}%;background:${col}"></i></div>
    <div class="grid2">
      <div class="tile"><div class="t">home best</div><div class="v">${g.home} ${fmtA(ml.best_home_am)}</div><div class="bk">${ml.best_home_book} · fair ${ml.cons_home}% · EV ${sgn(ml.ev_home)}%</div></div>
      <div class="tile"><div class="t">away best</div><div class="v">${g.away} ${fmtA(ml.best_away_am)}</div><div class="bk">${ml.best_away_book} · fair ${ml.cons_away}% · EV ${sgn(ml.ev_away)}%</div></div>
    </div>
    ${bookTable(ml)}
    <div class="grid2">
      ${o.spread?`<div class="tile"><div class="t">spread (best price)</div><div class="v">${g.home} ${sgn(o.spread.home_point)} ${fmtA(o.spread.home_am)} / ${g.away} ${sgn(o.spread.away_point)} ${fmtA(o.spread.away_am)}</div><div class="bk">market · no model edge</div></div>`:''}
      ${o.total?`<div class="tile"><div class="t">total (best price)</div><div class="v">O ${o.total.over_point} ${fmtA(o.total.over_am)} / U ${o.total.under_point} ${fmtA(o.total.under_am)}</div><div class="bk">market · no model edge</div></div>`:''}
    </div>
    ${o.model_ev?modelEdge(g,o.model_ev):''}
    <div class="note">value = best available price vs the no-vig consensus of ${ml.n_books} books (a model-free, market-vs-market edge — line shopping). spreads/totals are best-price only. player props aren't modeled.</div>
  </div>`;
}
function modelEdge(g,m){
  if(!m.best_side) return `<div class="model"><b>model edge (live):</b> model ${g.model_home_prob}% home — no +EV side vs the best line.</div>`;
  return `<div class="model"><b style="color:#9ece6a">model edge (live):</b> model says ${g.model_home_prob}% home → bet <b>${m.bet_team}</b> · EV ${sgn(m.ev_pct)}%/$1 · Kelly ${m.kelly_pct}%</div>`;
}
function detail(g){
  let body;
  if(g.state==='in'&&g.model_home_prob!=null){const mh=g.model_home_prob,fav=mh>=50?g.home:g.away,fp=mh>=50?mh:100-mh;
    body=`<div class="st" style="margin-top:12px"><span style="font-size:30px;font-weight:800;color:#e6ebf5">${mh}%</span> model home win · favors ${fav} (${fp}%) · ${g.away} ${g.away_score}–${g.home} ${g.home_score}</div>`;
  }else if(g.state==='post'){body=`<div class="st" style="margin-top:12px">FINAL · ${g.away} ${g.away_score} – ${g.home} ${g.home_score} · the model reads live games only</div>`;}
  else{body=`<div class="st" style="margin-top:12px">${g.detail||'upcoming'} · live model read appears once it tips off</div>`;}
  return `<div class="dhead">${logo(g.away_logo)}${logo(g.home_logo)}<div><div class="dmu">${g.away_full||g.away} @ ${g.home_full||g.home}</div><div class="st">${badge(g.state)} &nbsp; ${g.detail||''}</div></div></div>${body}${valueHero(g)}`;
}
function relSVG(rel){
  const W=300,H=300,L=42,R=14,T=14,B=34,pw=W-L-R,ph=H-T-B,X=p=>L+p*pw,Y=p=>T+(1-p)*ph;
  let s=`<svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:320px">`;
  for(const t of [0,.25,.5,.75,1]) s+=`<line x1="${X(t)}" y1="${T}" x2="${X(t)}" y2="${T+ph}" stroke="#18233d"/><line x1="${L}" y1="${Y(t)}" x2="${L+pw}" y2="${Y(t)}" stroke="#18233d"/>`;
  s+=`<polygon points="${X(0)},${Y(.05)} ${X(.95)},${Y(1)} ${X(1)},${Y(1)} ${X(1)},${Y(.95)} ${X(.05)},${Y(0)} ${X(0)},${Y(0)}" fill="#7aa2f7" opacity=".06"/>`;
  s+=`<line x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(1)}" stroke="#5a6ea0" stroke-dasharray="5 4"/>`;
  s+=`<text x="${X(.8)}" y="${Y(.8)-6}" fill="#6b7aa5" font-size="9" transform="rotate(-45 ${X(.8)} ${Y(.8)})">perfect</text>`;
  const pts=(rel||[]).filter(r=>r[3]!=null&&r[4]!=null&&r[2]>0).sort((a,b)=>a[3]-b[3]);
  if(pts.length>1) s+=`<polyline fill="none" stroke="#7aa2f7" stroke-width="1.5" opacity=".45" points="${pts.map(r=>X(r[3])+','+Y(r[4])).join(' ')}"/>`;
  pts.forEach(r=>{const p=r[3],a=r[4],n=r[2],d=Math.abs(p-a),c=d<=.03?'#9ece6a':d<=.07?'#e0af68':'#f7768e';
    s+=`<circle cx="${X(p)}" cy="${Y(a)}" r="${Math.max(3,Math.min(9,Math.sqrt(n)/11))}" fill="${c}" opacity=".92"><title>${(r[0]*100).toFixed(0)}–${(r[1]*100).toFixed(0)}% calls: predicted ${(p*100).toFixed(1)}%, actually won ${(a*100).toFixed(1)}%  (n=${n})</title></circle>`;});
  for(const t of [0,.5,1]) s+=`<text x="${X(t)}" y="${H-12}" fill="#6b7aa5" font-size="9" text-anchor="middle">${(t*100)|0}%</text><text x="${L-6}" y="${Y(t)+3}" fill="#6b7aa5" font-size="9" text-anchor="end">${(t*100)|0}%</text>`;
  s+=`<text x="${L+pw/2}" y="${H-1}" fill="#7e8bad" font-size="9.5" text-anchor="middle">model predicted win %</text><text x="11" y="${T+ph/2}" fill="#7e8bad" font-size="9.5" text-anchor="middle" transform="rotate(-90 11 ${T+ph/2})">actual win %</text></svg>`;
  return s;}
function calibHTML(rel,m){
  if(!rel) return `<div class="calhead"><h2 style="margin:0">model calibration</h2></div><div class="calwrap"><div class="empty">no calibration data embedded in this bundle — retrain the model to populate it.</div></div>`;
  m=m||{};
  const ece=m.ece, v = ece==null?['unknown','#6b7aa5','?'] : ece<=.02?['Well calibrated','#9ece6a','✓'] : ece<=.05?['Reasonably calibrated','#e0af68','≈'] : ['Miscalibrated','#f7768e','⚠'];
  const skill=(m.log_loss!=null&&m.baseline_log_loss!=null)?(1-m.log_loss/m.baseline_log_loss)*100:null;
  const cards=[
    ['Calibration error (ECE)', ece==null?'—':(ece*100).toFixed(1)+'%', `Average gap between the model's stated win % and what actually happens. Under 2% is sharp, under 5% is solid.`],
    ['Skill vs. guessing', skill==null?'—':(skill>0?'+':'')+skill.toFixed(0)+'%', `Log loss ${f3(m.log_loss)} vs ${f3(m.baseline_log_loss)} for just guessing the base rate. Positive means the model adds real signal.`],
    ['Brier score', f3(m.brier), `Mean squared error of the probabilities: 0 is perfect, 0.25 is a coin-flip guess. Lower is better.`],
    ['Temperature', f3(m.temperature), `Post-hoc scaling used to fix over/under-confidence. ≈1.0 means the raw model was already honest.`],
  ];
  const cardHTML=cards.map(c=>`<div class="mcard"><div class="mt">${c[0]}</div><div class="mv">${c[1]}</div><div class="mx">${c[2]}</div></div>`).join('');
  return `<div class="calhead"><h2 style="margin:0">model calibration</h2><span class="verdict" style="color:${v[1]};border-color:${v[1]}55;background:${v[1]}1a">${v[2]} ${v[0]}</span></div>
    <p class="note" style="margin:0 0 14px">Does the model tell the truth about its own confidence? When it says <b>70%</b>, do those teams really win about <b>70%</b> of the time? That's calibration — and it's what makes the value/EV numbers above trustworthy (a model can be "accurate" yet lie about probabilities).</p>
    <div class="calwrap">
      <div class="calplot">${relSVG(rel)}
        <div class="legend"><span><i style="background:#9ece6a"></i>on the money</span><span><i style="background:#e0af68"></i>a bit off</span><span><i style="background:#f7768e"></i>off</span><span class="muted">• dot size = games in that bucket</span></div></div>
      <div class="calgrid">${cardHTML}</div>
    </div>
    <p class="note">How to read the chart: each dot is a bucket of predictions (e.g. every "60%" call); landing on the dashed line means the predicted % matched how often it actually happened. Hover a dot for its numbers. Measured on ${m.n_rows!=null?(+m.n_rows).toLocaleString():'?'} held-out game-states the model never trained on.</p>`;}
function render(){const gs=DATA.games||[];
  document.getElementById('listhdr').textContent=`${SPORT.toUpperCase()} · ${gs.length} games`;
  document.getElementById('err').innerHTML=(DATA.error||DATA.odds_err)?`<div class="empty" style="color:#f7768e">${DATA.error||DATA.odds_err}</div>`:'';
  document.getElementById('list').innerHTML=gs.length?gs.map(gcard).join(''):'<div class="empty">no games on the board today.</div>';
  document.getElementById('detail').innerHTML=(SEL!=null&&gs[SEL])?detail(gs[SEL]):'<div class="empty">pick a game on the left.</div>';
  document.getElementById('quota').textContent=DATA.has_key?(DATA.quota?('odds quota: '+DATA.quota):'odds ready'):'no odds key';
  document.getElementById('calib').innerHTML=calibHTML(DATA.reliability,DATA.metrics);
}
function sel(i){SEL=i;render();}
async function load(s,mode){SPORT=s;document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('on',t.dataset.s===s));
  const prev=SEL;DATA=await (await fetch(`/api/sport?s=${s}&odds=${mode}`)).json();const gs=DATA.games||[];
  SEL=(prev==null)?(gs.length?0:null):Math.min(prev,gs.length-1);render();}
async function init(){const cfg=await (await fetch('/api/config')).json();
  document.getElementById('tabs').innerHTML=(cfg.sports||[]).map(s=>`<div class="tab" data-s="${s}" onclick="load('${s}','1')">${s.toUpperCase()}</div>`).join('');
  if(cfg.sports&&cfg.sports.length)load(cfg.sports[0],'1');}
init();setInterval(()=>{if(SPORT)load(SPORT,'0');},30000);
</script></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        try:
            if self.path == "/" or self.path.startswith("/index"):
                self._send(200, "text/html; charset=utf-8", _PAGE.encode())
            elif self.path.startswith("/api/config"):
                self._send(200, "application/json",
                           json.dumps({"sports": list(_MODELS), "has_key": _has_key()}).encode())
            elif self.path.startswith("/api/sport"):
                q = parse_qs(urlparse(self.path).query)
                s = (q.get("s") or [""])[0]
                mode = (q.get("odds") or ["0"])[0]
                if s not in _MODELS:
                    self._send(404, "application/json", b'{"error":"no model for sport"}')
                    return
                self._send(200, "application/json", json.dumps(_sport_payload(s, mode)).encode())
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

    def log_message(self, *args):
        pass


def run(models_dir: str = "models", port: int = 8080) -> None:
    _load_dotenv()
    _load_models(models_dir)
    if not _MODELS:
        raise SystemExit(f"no model bundles found in {models_dir}/ — train one first")
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    odds = "odds ON" if _has_key() else "odds OFF (no ODDS_API_KEY)"
    print(f"live-edge dashboard [{', '.join(_MODELS)}] {odds} -> http://localhost:{port}  (Ctrl-C)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Local multi-sport dashboard for live-edge.")
    p.add_argument("--models-dir", default="models", dest="models_dir")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args(argv)
    run(models_dir=args.models_dir, port=args.port)


if __name__ == "__main__":
    main()
