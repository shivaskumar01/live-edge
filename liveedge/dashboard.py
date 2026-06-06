"""Local multi-sport web dashboard for live-edge.

    python -m liveedge.dashboard                 # http://localhost:8080
    python -m liveedge.dashboard --port 8090

Browse REAL games (today's schedule + in-progress + finals) for every sport with a trained model
(NFL / NBA / MLB), by tab, and click through them. In-progress games show the model's live win
probability; each tab shows that model's calibration curve. Games come from ESPN (free).

If an ODDS_API_KEY is set (read from the environment or a local .env), the game detail also shows
live sportsbook odds: the MONEYLINE with the model's EV/Kelly on in-progress games, plus SPREADS
and TOTALS as live market lines (the win-prob model does not price those, so no edge is shown).
Odds cost API credits, so they're fetched only on tab-click / manual refresh (cached ~3 min); the
30s auto-refresh updates ESPN only. Player props are intentionally omitted (can't be priced by a
win-prob model). Pure stdlib http.server.
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
from liveedge.oddsmath import decimal_to_american

_SPORTS = ("nfl", "nba", "mlb")
_STOPWORDS = {"the", "fc", "sc"}
_ODDS_TTL = 180.0  # seconds; odds are cached this long to protect the API quota
_MODELS: dict = {}
_ODDS_CACHE: dict = {}  # sport -> {"ts", "games", "quota"}


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
            _MODELS[sport] = {
                "model": model, "scaler": scaler, "calibrator": calibrator, "meta": meta,
                "spec": get_spec(sport), "provider": ESPNProvider(sport),
            }


def _has_key() -> bool:
    return bool(os.environ.get("ODDS_API_KEY"))


# --------------------------------------------------------------------------------------
# ESPN real games
# --------------------------------------------------------------------------------------


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
            "home_score": int(home.get("score") or 0), "away_score": int(away.get("score") or 0),
            "model_home_prob": None, "_mp_f": None, "odds": None,
        }
        if state == "in":
            gs = prov._parse(event, comp)
            if gs is not None:
                p = float(predict_prob(m["model"], m["scaler"], [m["spec"].vector(gs)], m["calibrator"])[0])
                g["model_home_prob"] = round(p * 100)
                g["_mp_f"] = p
        out.append(g)
    order = {"in": 0, "pre": 1, "post": 2}
    out.sort(key=lambda g: (order.get(g["state"], 3), g["detail"]))
    return out


# --------------------------------------------------------------------------------------
# Odds (quota-protected)
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


def _attach_odds(g: dict, od: dict) -> dict:
    out: dict = {"is_live": od["is_live"], "moneyline": None, "spread": None, "total": None, "ev": None}
    if od["h2h"]:
        hd, ad = od["h2h"]["home_dec"], od["h2h"]["away_dec"]
        out["moneyline"] = {"home_am": _am(hd), "away_am": _am(ad), "book": od["h2h"]["home_book"]}
        if g["state"] == "in" and g.get("_mp_f") is not None:
            r = evaluate(g["home"], g["away"], g["_mp_f"], hd, ad, kelly_multiplier=0.25, min_ev=0.0)
            out["ev"] = {
                "best_side": r.best_side, "ev_pct": round(r.best_ev * 100, 1),
                "kelly_pct": round(r.kelly_fraction * 100, 1),
                "bet_team": (g["home"] if r.best_side == "home" else g["away"]) if r.best_side else None,
            }
    if od["spread"]:
        s = od["spread"]
        out["spread"] = {"home_point": s["home_point"], "home_am": _am(s["home_dec"]),
                         "away_point": s["away_point"], "away_am": _am(s["away_dec"]), "book": s["book"]}
    if od["total"]:
        t = od["total"]
        out["total"] = {"point": t["point"], "over_am": _am(t["over_dec"]),
                        "under_am": _am(t["under_dec"]), "book": t["book"]}
    return out


def _get_odds(sport: str, force: bool) -> tuple[list[dict] | None, str | None, str | None]:
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
            return cached["games"], cached["quota"], f"{type(exc).__name__}: {exc} (showing cached)"
        return None, None, f"{type(exc).__name__}: {exc}"


def _sport_payload(sport: str, odds_mode: str) -> dict:
    m = _MODELS[sport]
    quota, odds_err = None, None
    try:
        games, err = _sport_games(sport), None
    except Exception as exc:  # noqa: BLE001
        games, err = [], f"{type(exc).__name__}: {exc}"

    if _has_key() and games:
        if odds_mode in ("1", "force"):
            odds_list, quota, odds_err = _get_odds(sport, force=(odds_mode == "force"))
        else:  # cache-only (auto-refresh) — never spends a credit
            cached = _ODDS_CACHE.get(sport)
            odds_list, quota = (cached["games"], cached["quota"]) if cached else (None, None)
        if odds_list:
            for g in games:
                od = _match_odds(g, odds_list)
                if od:
                    g["odds"] = _attach_odds(g, od)

    for g in games:
        g.pop("_mp_f", None)
    return {
        "sport": sport, "games": games, "error": err, "odds_err": odds_err,
        "has_key": _has_key(), "quota": quota,
        "reliability": m["meta"].get("reliability"), "metrics": m["meta"].get("metrics"),
    }


_PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>live-edge</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{color-scheme:dark}*{box-sizing:border-box}
body{margin:0;background:#0b1020;color:#cdd6f4;font:15px/1.45 -apple-system,Segoe UI,Roboto,sans-serif}
header{padding:14px 22px;border-bottom:1px solid #1e2740;display:flex;align-items:center;gap:14px;flex-wrap:wrap}
h1{margin:0;font-size:19px;font-weight:700}h1 .e{color:#7aa2f7}
.tabs{display:flex;gap:8px}.tab{padding:6px 14px;border-radius:999px;background:#151c2e;border:1px solid #1e2740;cursor:pointer;font-weight:600;font-size:13px;color:#aeb8d4}
.tab.on{background:#1b3a6b;border-color:#2d5aa0;color:#dbe6ff}
.right{margin-left:auto;display:flex;align-items:center;gap:10px;font-size:12px;color:#8892b0}
button{background:#151c2e;border:1px solid #2d3a5c;color:#aeb8d4;border-radius:8px;padding:5px 10px;cursor:pointer;font-size:12px}
button:hover{border-color:#7aa2f7}
main{display:grid;grid-template-columns:minmax(280px,360px) 1fr;gap:18px;max-width:1120px;margin:0 auto;padding:20px}
@media(max-width:780px){main{grid-template-columns:1fr}}
h2{font-size:12px;text-transform:uppercase;letter-spacing:.6px;color:#8892b0;margin:0 0 10px}
.list{display:flex;flex-direction:column;gap:9px}
.game{background:#141b2d;border:1px solid #1e2740;border-radius:11px;padding:11px 13px;cursor:pointer;transition:.12s}
.game:hover{border-color:#2d5aa0}.game.sel{border-color:#7aa2f7;background:#16203a}
.game .top{display:flex;justify-content:space-between;align-items:center;gap:8px}.game .mu{font-weight:650}
.badge{font-size:10px;font-weight:800;letter-spacing:.5px;padding:2px 7px;border-radius:5px}
.b-in{background:#12331f;color:#9ece6a}.b-pre{background:#23304d;color:#9fb3e0}.b-post{background:#241a24;color:#a98aa0}
.game .sub{color:#8892b0;font-size:12px;margin-top:3px;display:flex;justify-content:space-between}
.tag{font-size:10px;color:#9ece6a;margin-left:6px}
.bar{height:6px;border-radius:4px;background:#23304d;margin-top:9px;overflow:hidden}.bar>i{display:block;height:100%;background:linear-gradient(90deg,#7aa2f7,#9ece6a)}
.detail{background:#141b2d;border:1px solid #1e2740;border-radius:14px;padding:20px;min-height:160px}
.detail .mu{font-size:20px;font-weight:700}.detail .st{color:#8892b0;margin:4px 0 14px}
.big{font-size:44px;font-weight:800;font-variant-numeric:tabular-nums;line-height:1}.fav{color:#9ece6a;font-weight:600;margin-top:6px}
.kv{color:#8892b0;font-size:13px;margin-top:12px;line-height:1.6}.kv code{color:#aeb8d4}
.odds{margin-top:16px;border-top:1px solid #1e2740;padding-top:14px}
.oddrow{display:flex;gap:10px;align-items:center;padding:7px 0;border-bottom:1px solid #161d30;font-size:13px;flex-wrap:wrap}
.oddrow .lab{width:74px;color:#8892b0;font-size:11px;text-transform:uppercase;letter-spacing:.4px}
.oddrow .val{font-variant-numeric:tabular-nums}
.ev{margin-left:auto;background:#12331f;color:#9ece6a;font-weight:700;padding:3px 9px;border-radius:6px;font-size:12px}
.ev.none{background:#1a2030;color:#8892b0;font-weight:500}
.panel{margin-top:18px;background:#141b2d;border:1px solid #1e2740;border-radius:14px;padding:18px;display:flex;gap:22px;flex-wrap:wrap;align-items:center}
#metrics{color:#8892b0;font-size:13px;font-variant-numeric:tabular-nums;line-height:1.7}
.muted{color:#8892b0}.empty{color:#8892b0;padding:14px}.locked{margin-top:14px;padding:10px 12px;border-radius:9px;background:#1a2030;color:#9fb3e0;font-size:12.5px;border:1px dashed #2d3a5c}
</style></head><body>
<header><h1>live<span class="e">·</span>edge</h1><div class="tabs" id="tabs"></div>
<div class="right"><span id="quota"></span><button onclick="if(SPORT)load(SPORT,'force')">↻ odds</button></div></header>
<main>
  <div><h2 id="listhdr">games</h2><div id="err"></div><div class="list" id="list"></div></div>
  <div><h2>game</h2><div class="detail" id="detail"><div class="empty">pick a game on the left.</div></div>
    <h2 style="margin-top:22px">model calibration</h2>
    <div class="panel"><div id="relwrap"></div><div id="metrics"></div></div>
    <p class="muted" style="font-size:12px">predicted win-prob (x) vs actual win rate (y); dashed = perfect. From the model's held-out validation split.</p>
  </div>
</main>
<script>
let DATA=null,SEL=null,SPORT=null;
const fmtA=a=>a==null?'—':(a>0?'+'+a:''+a), f3=x=>x==null?'—':(+x).toFixed(3);
const sgn=p=>p==null?'':(p>0?'+'+p:''+p);
function badge(s){return s==='in'?'<span class="badge b-in">LIVE</span>':s==='post'?'<span class="badge b-post">FINAL</span>':'<span class="badge b-pre">SCHEDULED</span>';}
function gcard(g,i){
  const score=(g.state!=='pre')?`${g.away} ${g.away_score} · ${g.home} ${g.home_score}`:'';
  const bar=(g.model_home_prob!=null)?`<div class="bar"><i style="width:${g.model_home_prob}%"></i></div>`:'';
  const tag=(g.odds&&g.odds.ev&&g.odds.ev.best_side)?'<span class="tag">▸ +EV</span>':(g.odds?'<span class="tag" style="color:#9fb3e0">💵</span>':'');
  return `<div class="game ${i===SEL?'sel':''}" onclick="sel(${i})"><div class="top"><span class="mu">${g.away} @ ${g.home}${tag}</span>${badge(g.state)}</div>
    <div class="sub"><span>${g.detail||''}</span><span>${score}</span></div>${bar}</div>`;
}
function oddsBlock(g){
  if(!DATA.has_key) return `<div class="locked">💵 set <code>ODDS_API_KEY</code> to show live moneyline EV + market spreads/totals here.</div>`;
  if(!g.odds) return `<div class="locked">no sportsbook line matched for this game.</div>`;
  const o=g.odds; let h='';
  if(o.moneyline){
    let ev;
    if(o.ev&&o.ev.best_side) ev=`<span class="ev">▸ ${o.ev.bet_team} EV ${sgn(o.ev.ev_pct)}% · Kelly ${o.ev.kelly_pct}%</span>`;
    else if(g.state==='in') ev=`<span class="ev none">no +EV side</span>`;
    else ev=`<span class="ev none">edge shows live</span>`;
    h+=`<div class="oddrow"><span class="lab">Moneyline</span><span class="val">${g.home} ${fmtA(o.moneyline.home_am)} / ${g.away} ${fmtA(o.moneyline.away_am)}</span>${ev}</div>`;
  }
  if(o.spread) h+=`<div class="oddrow"><span class="lab">Spread</span><span class="val">${g.home} ${sgn(o.spread.home_point)} (${fmtA(o.spread.home_am)}) / ${g.away} ${sgn(o.spread.away_point)} (${fmtA(o.spread.away_am)})</span><span class="ev none">market</span></div>`;
  if(o.total) h+=`<div class="oddrow"><span class="lab">Total</span><span class="val">O/U ${o.total.point} · O ${fmtA(o.total.over_am)} / U ${fmtA(o.total.under_am)}</span><span class="ev none">market</span></div>`;
  h+=`<div class="muted" style="font-size:11px;margin-top:9px">moneyline EV is the model vs the line; spread &amp; total are live market numbers (the win-prob model doesn't price them — no edge shown). player props not modeled.</div>`;
  return `<div class="odds">${h}</div>`;
}
function detail(g){
  let body;
  if(g.state==='in'&&g.model_home_prob!=null){const mh=g.model_home_prob,fav=mh>=50?g.home:g.away,fp=mh>=50?mh:100-mh;
    body=`<div class="big">${mh}%</div><div class="fav">model favors ${fav} (${fp}% to win)</div>
      <div class="kv"><code>${g.away} ${g.away_score} — ${g.home} ${g.home_score}</code> · ${g.detail}</div>`;
  }else if(g.state==='post'){body=`<div class="big" style="font-size:30px">${g.away} ${g.away_score} — ${g.home} ${g.home_score}</div><div class="kv">final · the model reads live games only.</div>`;
  }else{body=`<div class="big" style="font-size:26px;color:#9fb3e0">scheduled</div><div class="kv">${g.detail||'not started'} · the live model read appears once it tips off.</div>`;}
  return `<div class="mu">${g.away_full||g.away} @ ${g.home_full||g.home}</div><div class="st">${badge(g.state)} &nbsp; ${g.detail||''}</div>${body}${oddsBlock(g)}`;
}
function relSVG(rel){const S=230,pad=30,pl=S-2*pad,x=p=>pad+p*pl,y=p=>S-pad-p*pl;
  let s=`<svg viewBox="0 0 ${S} ${S}" width="230" height="230"><rect x="${pad}" y="${pad}" width="${pl}" height="${pl}" fill="none" stroke="#2a3550"/>`;
  s+=`<line x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}" stroke="#4a5578" stroke-dasharray="4 4"/>`;
  (rel||[]).forEach(r=>{const p=r[3],a=r[4],n=r[2];if(p==null||a==null||!n)return;s+=`<circle cx="${x(p)}" cy="${y(a)}" r="${Math.max(2.5,Math.min(8,Math.sqrt(n)/12))}" fill="#7aa2f7" opacity=".85"/>`;});
  s+=`<text x="${x(.5)}" y="${S-6}" fill="#8892b0" font-size="10" text-anchor="middle">predicted</text><text x="12" y="${y(.5)}" fill="#8892b0" font-size="10" text-anchor="middle" transform="rotate(-90 12 ${y(.5)})">actual</text></svg>`;
  return s;}
function render(){const gs=DATA.games||[];
  document.getElementById('listhdr').textContent=`${SPORT.toUpperCase()} · ${gs.length} games today`;
  document.getElementById('err').innerHTML=(DATA.error||DATA.odds_err)?`<div class="empty" style="color:#f7768e">${DATA.error||DATA.odds_err}</div>`:'';
  document.getElementById('list').innerHTML=gs.length?gs.map(gcard).join(''):'<div class="empty">no games on the board today.</div>';
  document.getElementById('detail').innerHTML=(SEL!=null&&gs[SEL])?detail(gs[SEL]):'<div class="empty">pick a game on the left.</div>';
  document.getElementById('quota').textContent=DATA.has_key?(DATA.quota?('odds quota: '+DATA.quota):'odds: ready'):'no odds key';
  if(DATA.reliability){document.getElementById('relwrap').innerHTML=relSVG(DATA.reliability);const m=DATA.metrics||{};
    document.getElementById('metrics').innerHTML=`n = ${m.n_rows??'?'}<br>log loss ${f3(m.log_loss)} (baseline ${f3(m.baseline_log_loss)})<br>ECE ${f3(m.ece)} · Brier ${f3(m.brier)}<br>temperature ${f3(m.temperature)}`;
  }else{document.getElementById('relwrap').innerHTML='';document.getElementById('metrics').textContent='no calibration data in this bundle.';}
}
function sel(i){SEL=i;render();}
async function load(s,mode){SPORT=s;document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('on',t.dataset.s===s));
  const prev=SEL; DATA=await (await fetch(`/api/sport?s=${s}&odds=${mode}`)).json();
  const gs=DATA.games||[]; if(prev==null){const live=gs.findIndex(g=>g.state==='in');SEL=live>=0?live:(gs.length?0:null);}else SEL=Math.min(prev,gs.length-1);
  render();}
async function init(){const cfg=await (await fetch('/api/config')).json();
  document.getElementById('tabs').innerHTML=(cfg.sports||[]).map(s=>`<div class="tab" data-s="${s}" onclick="load('${s}','1')">${s.toUpperCase()}</div>`).join('');
  if(cfg.sports&&cfg.sports.length)load(cfg.sports[0],'1');}
init(); setInterval(()=>{if(SPORT)load(SPORT,'0');},30000);  // auto-refresh = ESPN only (no odds credits)
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
