"""Local multi-sport web dashboard for live-edge.

    python -m liveedge.dashboard                 # serves http://localhost:8080
    python -m liveedge.dashboard --port 8090

Browse REAL games (today's schedule + in-progress + finals) for every sport that has a trained
model (NFL / NBA / MLB), by tab. For in-progress games it shows the model's live win
probability; each sport tab also shows that model's calibration curve (read from the bundle).
Games come from ESPN (no key needed). Pure stdlib http.server — no web framework.

Scope / honesty
---------------
This is a win-probability model. It produces a read only for IN-PROGRESS games, and it edges
the MONEYLINE. Live sportsbook odds (moneyline EV + market spreads/totals) are a separate layer
that needs an ODDS_API_KEY — wired in once a key is configured. Player props are intentionally
NOT included: a win-prob model can't price them, so showing them would be analysis-free.
"""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from liveedge.features import get_spec
from liveedge.live_state import ESPNProvider
from liveedge.model import load_bundle, predict_prob

_SPORTS = ("nfl", "nba", "mlb")
_MODELS: dict = {}  # sport -> {model, scaler, calibrator, meta, spec, provider}


def _load_models(models_dir: str) -> None:
    for sport in _SPORTS:
        base = os.path.join(models_dir, sport)
        if os.path.exists(base + ".json"):
            model, scaler, calibrator, meta = load_bundle(base)
            _MODELS[sport] = {
                "model": model, "scaler": scaler, "calibrator": calibrator, "meta": meta,
                "spec": get_spec(sport), "provider": ESPNProvider(sport),
            }


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
            "id": str(event.get("id", "")),
            "state": state,
            "detail": (status.get("type") or {}).get("shortDetail", "") or "",
            "home": ht.get("abbreviation", "HOME"),
            "away": at.get("abbreviation", "AWAY"),
            "home_full": ht.get("displayName", ""),
            "away_full": at.get("displayName", ""),
            "home_score": int(home.get("score") or 0),
            "away_score": int(away.get("score") or 0),
            "model_home_prob": None,
        }
        if state == "in":
            gs = prov._parse(event, comp)
            if gs is not None:
                p = predict_prob(m["model"], m["scaler"], [m["spec"].vector(gs)], m["calibrator"])
                g["model_home_prob"] = round(float(p[0]) * 100)
        out.append(g)
    order = {"in": 0, "pre": 1, "post": 2}
    out.sort(key=lambda g: (order.get(g["state"], 3), g["detail"]))
    return out


def _sport_payload(sport: str) -> dict:
    m = _MODELS[sport]
    try:
        games, err = _sport_games(sport), None
    except Exception as exc:  # noqa: BLE001 — never 500 the page on a transient ESPN error
        games, err = [], f"{type(exc).__name__}: {exc}"
    return {
        "sport": sport, "games": games, "error": err,
        "reliability": m["meta"].get("reliability"), "metrics": m["meta"].get("metrics"),
    }


_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>live-edge</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:#0b1020;color:#cdd6f4;font:15px/1.45 -apple-system,Segoe UI,Roboto,sans-serif}
header{padding:16px 22px;border-bottom:1px solid #1e2740;display:flex;align-items:center;gap:14px;flex-wrap:wrap}
h1{margin:0;font-size:19px;font-weight:700;letter-spacing:.3px}
h1 .e{color:#7aa2f7}
.tabs{display:flex;gap:8px;margin-left:6px}
.tab{padding:6px 14px;border-radius:999px;background:#151c2e;border:1px solid #1e2740;cursor:pointer;font-weight:600;font-size:13px;color:#aeb8d4}
.tab.on{background:#1b3a6b;border-color:#2d5aa0;color:#dbe6ff}
.note{margin-left:auto;color:#8892b0;font-size:12px;max-width:420px;text-align:right}
main{display:grid;grid-template-columns:minmax(280px,360px) 1fr;gap:18px;max-width:1100px;margin:0 auto;padding:20px}
@media(max-width:760px){main{grid-template-columns:1fr}}
h2{font-size:12px;text-transform:uppercase;letter-spacing:.6px;color:#8892b0;margin:0 0 10px}
.list{display:flex;flex-direction:column;gap:9px}
.game{background:#141b2d;border:1px solid #1e2740;border-radius:11px;padding:11px 13px;cursor:pointer;transition:.12s}
.game:hover{border-color:#2d5aa0}.game.sel{border-color:#7aa2f7;background:#16203a}
.game .top{display:flex;justify-content:space-between;align-items:center;gap:8px}
.game .mu{font-weight:650}
.badge{font-size:10px;font-weight:800;letter-spacing:.5px;padding:2px 7px;border-radius:5px}
.b-in{background:#12331f;color:#9ece6a}.b-pre{background:#23304d;color:#9fb3e0}.b-post{background:#241a24;color:#a98aa0}
.game .sub{color:#8892b0;font-size:12px;margin-top:3px;display:flex;justify-content:space-between}
.bar{height:6px;border-radius:4px;background:#23304d;margin-top:9px;overflow:hidden}
.bar>i{display:block;height:100%;background:linear-gradient(90deg,#7aa2f7,#9ece6a)}
.detail{background:#141b2d;border:1px solid #1e2740;border-radius:14px;padding:20px;min-height:160px}
.detail .mu{font-size:20px;font-weight:700}.detail .st{color:#8892b0;margin:4px 0 16px}
.big{font-size:46px;font-weight:800;font-variant-numeric:tabular-nums;line-height:1}
.fav{color:#9ece6a;font-weight:600;margin-top:6px}
.kv{color:#8892b0;font-size:13px;margin-top:14px;line-height:1.7}
.kv code{color:#aeb8d4}
.panel{margin-top:18px;background:#141b2d;border:1px solid #1e2740;border-radius:14px;padding:18px;display:flex;gap:22px;flex-wrap:wrap;align-items:center}
#metrics{color:#8892b0;font-size:13px;font-variant-numeric:tabular-nums;line-height:1.7}
.muted{color:#8892b0}.empty{color:#8892b0;padding:14px}
.locked{margin-top:16px;padding:10px 12px;border-radius:9px;background:#1a2030;color:#9fb3e0;font-size:12.5px;border:1px dashed #2d3a5c}
</style></head><body>
<header>
  <h1>live<span class="e">·</span>edge</h1>
  <div class="tabs" id="tabs"></div>
  <div class="note">real games via ESPN. live model read shows for <b>in-progress</b> games.
  odds (moneyline EV + market spreads/totals) need an ODDS_API_KEY. player props: not modeled.</div>
</header>
<main>
  <div><h2 id="listhdr">games</h2><div id="err"></div><div class="list" id="list"></div></div>
  <div>
    <h2>game</h2><div class="detail" id="detail"><div class="empty">pick a game on the left.</div></div>
    <h2 style="margin-top:22px">model calibration</h2>
    <div class="panel"><div id="relwrap"></div><div id="metrics"></div></div>
    <p class="muted" style="font-size:12px">predicted win-prob (x) vs actual win rate (y); dashed = perfect. From the model's held-out validation split.</p>
  </div>
</main>
<script>
let DATA=null, SEL=null, SPORT=null;
const pct=x=>x==null?'—':x+'%';
function badge(s){return s==='in'?'<span class="badge b-in">LIVE</span>':s==='post'?'<span class="badge b-post">FINAL</span>':'<span class="badge b-pre">SCHEDULED</span>';}
function gameCard(g,i){
  const score=(g.state!=='pre')?`${g.away} ${g.away_score} · ${g.home} ${g.home_score}`:'';
  const bar=(g.model_home_prob!=null)?`<div class="bar"><i style="width:${g.model_home_prob}%"></i></div>`:'';
  return `<div class="game ${i===SEL?'sel':''}" onclick="sel(${i})">
    <div class="top"><span class="mu">${g.away} @ ${g.home}</span>${badge(g.state)}</div>
    <div class="sub"><span>${g.detail||''}</span><span>${score}</span></div>${bar}</div>`;
}
function detail(g){
  let body;
  if(g.state==='in'&&g.model_home_prob!=null){
    const mh=g.model_home_prob, fav=mh>=50?g.home:g.away, favp=mh>=50?mh:100-mh;
    body=`<div class="big">${mh}%</div><div class="fav">model favors ${fav} (${favp}% to win)</div>
      <div class="kv"><code>${g.away} ${g.away_score} — ${g.home} ${g.home_score}</code> · ${g.detail}<br>
      home win probability from the in-game model (flat pregame prior until odds are connected).</div>`;
  }else if(g.state==='post'){
    body=`<div class="big" style="font-size:30px">${g.away} ${g.away_score} — ${g.home} ${g.home_score}</div>
      <div class="kv">final. the model reads live games only.</div>`;
  }else{
    body=`<div class="big" style="font-size:26px;color:#9fb3e0">scheduled</div>
      <div class="kv">${g.detail||'not started'} · the live model read appears once the game tips off.</div>`;
  }
  return `<div class="mu">${g.away_full||g.away} @ ${g.home_full||g.home}</div>
    <div class="st">${badge(g.state)} &nbsp; ${g.detail||''}</div>${body}
    <div class="locked">💵 odds for this game (moneyline EV + market spread/total) appear here once an
    <code>ODDS_API_KEY</code> is set. player props aren't modeled, so they're omitted by design.</div>`;
}
function relSVG(rel){
  const S=230,pad=30,pl=S-2*pad,x=p=>pad+p*pl,y=p=>S-pad-p*pl;
  let s=`<svg viewBox="0 0 ${S} ${S}" width="230" height="230">`;
  s+=`<rect x="${pad}" y="${pad}" width="${pl}" height="${pl}" fill="none" stroke="#2a3550"/>`;
  s+=`<line x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}" stroke="#4a5578" stroke-dasharray="4 4"/>`;
  (rel||[]).forEach(r=>{const p=r[3],a=r[4],n=r[2];if(p==null||a==null||!n)return;
    s+=`<circle cx="${x(p)}" cy="${y(a)}" r="${Math.max(2.5,Math.min(8,Math.sqrt(n)/12))}" fill="#7aa2f7" opacity=".85"/>`;});
  s+=`<text x="${x(.5)}" y="${S-6}" fill="#8892b0" font-size="10" text-anchor="middle">predicted</text>`;
  s+=`<text x="12" y="${y(.5)}" fill="#8892b0" font-size="10" text-anchor="middle" transform="rotate(-90 12 ${y(.5)})">actual</text></svg>`;
  return s;
}
function render(){
  const gs=DATA.games||[];
  document.getElementById('listhdr').textContent=`${SPORT.toUpperCase()} · ${gs.length} games today`;
  document.getElementById('err').innerHTML=DATA.error?`<div class="empty" style="color:#f7768e">${DATA.error}</div>`:'';
  document.getElementById('list').innerHTML=gs.length?gs.map(gameCard).join(''):'<div class="empty">no games on the board today.</div>';
  document.getElementById('detail').innerHTML=(SEL!=null&&gs[SEL])?detail(gs[SEL]):'<div class="empty">pick a game on the left.</div>';
  if(DATA.reliability){document.getElementById('relwrap').innerHTML=relSVG(DATA.reliability);
    const m=DATA.metrics||{},f=x=>x==null?'—':(+x).toFixed(3);
    document.getElementById('metrics').innerHTML=`n = ${m.n_rows??'?'}<br>log loss ${f(m.log_loss)} (baseline ${f(m.baseline_log_loss)})<br>ECE ${f(m.ece)} · Brier ${f(m.brier)}<br>temperature ${f(m.temperature)}`;
  }else{document.getElementById('relwrap').innerHTML='';document.getElementById('metrics').textContent='no calibration data in this bundle.';}
}
function sel(i){SEL=i;render();}
async function loadSport(s){SPORT=s;SEL=null;
  document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('on',t.dataset.s===s));
  DATA=await (await fetch('/api/sport?s='+s)).json();
  // auto-select the first in-progress game if any
  const gs=DATA.games||[]; const live=gs.findIndex(g=>g.state==='in'); SEL=live>=0?live:(gs.length?0:null);
  render();
}
async function init(){
  const sp=(await (await fetch('/api/sports')).json()).sports||[];
  document.getElementById('tabs').innerHTML=sp.map(s=>`<div class="tab" data-s="${s}" onclick="loadSport('${s}')">${s.toUpperCase()}</div>`).join('');
  if(sp.length)loadSport(sp[0]);
}
init(); setInterval(()=>{if(SPORT)loadSport(SPORT);},30000);
</script></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        try:
            if self.path == "/" or self.path.startswith("/index"):
                self._send(200, "text/html; charset=utf-8", _PAGE.encode())
            elif self.path.startswith("/api/sports"):
                self._send(200, "application/json", json.dumps({"sports": list(_MODELS)}).encode())
            elif self.path.startswith("/api/sport"):
                s = (parse_qs(urlparse(self.path).query).get("s") or [""])[0]
                if s not in _MODELS:
                    self._send(404, "application/json", b'{"error":"no model for sport"}')
                    return
                self._send(200, "application/json", json.dumps(_sport_payload(s)).encode())
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
    _load_models(models_dir)
    if not _MODELS:
        raise SystemExit(f"no model bundles found in {models_dir}/ — train one first")
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    print(f"live-edge dashboard [{', '.join(_MODELS)}] -> http://localhost:{port}  (Ctrl-C to stop)")
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
