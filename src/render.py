"""
render.py - GlobalPulse dashboard uretici

data/latest.json + config/thresholds.yaml  ->  docs/index.html

Tek dosya, bagimliliksiz HTML uretir. GitHub Pages dogrudan servis eder.
Build adimi, JS framework, harici veri cagrisi yoktur.

Tasarim tezi: bir gostergenin SEVIYESI tek basina bilgi tasimaz;
2 yillik aralikta NEREDE durdugu tasir. Bu yuzden sayfanin imza
ogesi persentil konum seridi.

Kullanim:
    python src/render.py
    python src/render.py --out docs/index.html
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime
from pathlib import Path
from string import Template

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "thresholds.yaml"
LATEST_PATH = ROOT / "data" / "latest.json"
DEFAULT_OUT = ROOT / "docs" / "index.html"

TR_MONTHS = [
    "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
    "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık",
]

# Gosterge kisa adlari - teknik seri isimleri okunabilir hale gelsin
LABELS = {
    "vix": "VIX",
    "move": "MOVE",
    "hy_oas": "Yüksek getirili spread",
    "ig_oas": "Yatırım derecesi spread",
    "real_10y": "10Y reel faiz",
    "fwd_5y5y": "5y5y enflasyon beklentisi",
    "breakeven_10y": "10Y başabaş enflasyon",
    "turkey_cds_5y": "Türkiye 5Y CDS",
    "ust_2y": "ABD 2Y tahvil",
    "ust_10y": "ABD 10Y tahvil",
    "curve_2s10s": "2s10s eğri",
    "dxy": "Dolar endeksi",
    "dxy_broad": "Geniş dolar endeksi",
    "gold": "Altın",
    "brent": "Brent",
    "copper": "Bakır",
    "usdjpy": "USD/JPY",
    "usdtry": "USD/TRY",
    "spx": "S&P 500",
    "em_equity": "Gelişen piyasa hissesi",
    "xu100": "BIST 100",
    "bist_usd": "BIST 100 (dolar bazlı)",
    "tr_rel_5d": "BIST vs EM — 5 gün",
}

BAND_LABELS = {
    "calm": "sakin",
    "normal": "normal",
    "warning": "uyarı",
    "stress": "stres",
    "deflation_risk": "deflasyon riski",
    "inflation_risk": "enflasyon riski",
    "outperform": "görece güçlü",
    "veri_yok": "veri yok",
    "tanimsiz": "tanımsız",
}

BAND_CLASS = {
    "calm": "calm", "normal": "normal",
    "warning": "warn", "stress": "stress",
    "deflation_risk": "warn", "inflation_risk": "warn",
    "outperform": "calm",
    "veri_yok": "void", "tanimsiz": "void",
}

SEVERITY_LABELS = {
    "none": "önem düşük",
    "low": "önem düşük",
    "medium": "önem orta",
    "high": "önem yüksek",
}


def tr_date(iso: str) -> str:
    try:
        d = datetime.fromisoformat(iso)
        return f"{d.day} {TR_MONTHS[d.month - 1]} {d.year}"
    except Exception:
        return iso


def label_of(key: str) -> str:
    return LABELS.get(key, key)


def esc(s) -> str:
    return html.escape(str(s), quote=True)


# ---------------------------------------------------------------
# Imza ogesi: persentil konum seridi
# ---------------------------------------------------------------
def strip(pct: float | None, band: str) -> str:
    """
    2 yillik aralikta bugunku konumu gosteren cetvel.
    Persentil yoksa bos durum acikca gosterilir - sahte cizgi cizilmez.
    """
    cls = BAND_CLASS.get(band, "void")

    if pct is None:
        return (
            '<div class="strip strip--void" role="img" '
            'aria-label="persentil hesaplanamadı">'
            '<div class="strip__rail strip__rail--hatched"></div>'
            '<span class="strip__none">persentil yok</span>'
            "</div>"
        )

    p = max(0.0, min(100.0, float(pct)))
    return f"""<div class="strip" role="img" aria-label="2 yıllık aralıkta yüzde {p:.0f} konumunda">
  <div class="strip__rail">
    <span class="strip__tick" style="left:25%"></span>
    <span class="strip__tick strip__tick--mid" style="left:50%"></span>
    <span class="strip__tick" style="left:75%"></span>
    <span class="strip__marker strip__marker--{cls}" style="--pos:{p:.2f}%"></span>
  </div>
  <span class="strip__val">p{p:.0f}</span>
</div>"""


def fmt_val(v, unit: str = "") -> str:
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        s = f"{v:,.2f}".rstrip("0").rstrip(".")
        s = s.replace(",", "\u2009")  # ince bosluk binlik ayraci
        return f"{s}{unit}"
    return str(v)


def fmt_chg(v, unit: str) -> str:
    if v is None:
        return ""
    sign = "+" if v > 0 else ""
    return f"{sign}{v:g} {unit}".strip()


# ---------------------------------------------------------------
# Bolumler
# ---------------------------------------------------------------
def render_highlights(highlights: list[dict]) -> str:
    if not highlights:
        return (
            '<p class="empty">Hiçbir gösterge 2 yıllık aralığın uçlarında '
            "değil. Bugün olağandışı bir konumlanma yok.</p>"
        )

    rows = []
    for h in highlights:
        status = h.get("status", "yeni")
        streak = h.get("streak_days", 1)
        unit = h.get("chg_unit", "")
        chg = fmt_chg(h.get("chg_20d"), unit)

        if status == "yeni":
            tag = '<span class="tag tag--new">yeni</span>'
            meta = "bugün listeye girdi" if streak == 1 else f"{streak} gündür"
        else:
            tag = '<span class="tag tag--cont">devam</span>'
            meta = f"{streak} gündür"

        rows.append(f"""<li class="hl hl--{esc(status)}">
  <div class="hl__head">
    {tag}
    <span class="hl__name">{esc(label_of(h['indicator']))}</span>
    <span class="hl__num">{fmt_val(h.get('value'))}</span>
  </div>
  {strip(h.get('pct_rank'), 'normal')}
  <div class="hl__meta">{esc(meta)}{' · 20 iş günü ' + esc(chg) if chg else ''}</div>
</li>""")

    return f'<ul class="hl-list">{"".join(rows)}</ul>'


def render_levels(levels: dict) -> str:
    rows = []
    for key, v in levels.items():
        band = v.get("band", "veri_yok")
        cls = BAND_CLASS.get(band, "void")
        obs = v.get("real_obs", 0)

        note = ""
        if v.get("value") is None:
            note = '<span class="note">veri gelmedi</span>'
        elif v.get("pct_rank") is None:
            note = f'<span class="note">{obs} gözlem — persentil için yetersiz</span>'

        rows.append(f"""<tr>
  <th scope="row">{esc(label_of(key))}</th>
  <td class="num">{fmt_val(v.get('value'))}</td>
  <td><span class="band band--{cls}">{esc(BAND_LABELS.get(band, band))}</span></td>
  <td class="strip-cell">{strip(v.get('pct_rank'), band)}</td>
  <td class="note-cell">{note}</td>
</tr>""")

    return f"""<table class="levels">
  <caption class="sr-only">Gösterge seviyeleri ve 2 yıllık persentil konumları</caption>
  <thead><tr>
    <th scope="col">Gösterge</th><th scope="col">Değer</th>
    <th scope="col">Bant</th><th scope="col">2 yıllık konum</th><th scope="col"></th>
  </tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>"""


def render_alerts(alerts: list[dict]) -> str:
    if not alerts:
        return '<p class="empty">Hız eşiğini aşan hareket yok.</p>'

    items = []
    for a in alerts:
        arrow = "↑" if a["direction"] == "yukari" else "↓"
        items.append(f"""<li class="alert">
  <span class="alert__arrow">{arrow}</span>
  <div>
    <strong>{esc(a['label'])}</strong>
    <span class="alert__detail">{esc(label_of(a['indicator']))} ·
      {a['window_days']} iş gününde {a['change']:+g}{esc(a['unit'])}
      (eşik {esc(a['threshold'])}{esc(a['unit'])})</span>
  </div>
</li>""")
    return f'<ul class="alerts">{"".join(items)}</ul>'


def render_quality(c: dict) -> str:
    items = []
    if c.get("missing_series"):
        names = ", ".join(label_of(s) for s in c["missing_series"])
        items.append(f"Veri gelmeyen seriler: {names}. Bu göstergeleri "
                     "kullanan kurallar değerlendirilmedi.")
    if c.get("no_percentile"):
        names = ", ".join(label_of(s) for s in c["no_percentile"])
        items.append(f"Persentili hesaplanamayanlar: {names}. Gerçek gözlem "
                     "sayısı yetersiz — konum şeridi boş gösteriliyor.")
    if c.get("skipped_regimes"):
        names = ", ".join(s["label"] for s in c["skipped_regimes"])
        items.append(f"Veri eksikliğinden atlanan kurallar: {names}.")

    if not items:
        return ('<p class="empty">Tüm seriler geldi, tüm kurallar '
                "değerlendirildi.</p>")
    return "<ul class=\"quality\">" + "".join(f"<li>{esc(i)}</li>" for i in items) + "</ul>"


def render_narrative(text: str) -> str:
    paras = [p.strip() for p in (text or "").split("\n") if p.strip()]
    if not paras:
        return '<p class="empty">Yorum üretilemedi.</p>'
    return "".join(f"<p>{esc(p)}</p>" for p in paras)


# ---------------------------------------------------------------
# Sablon
# ---------------------------------------------------------------
TEMPLATE = Template("""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GlobalPulse — $date_tr</title>
<meta name="description" content="Küresel piyasa rejimi izleme panosu. $regime_label.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@400;500;600;700&family=IBM+Plex+Serif:ital,wght@0,400;0,500;1,400&display=swap" rel="stylesheet">
<style>
:root{
  --paper:#E9EBEE; --paper-2:#F4F5F7; --ink:#14181F; --ink-soft:#5A6472;
  --rule:#C9CFD6; --rule-soft:#DDE1E6;
  --calm:#1F6F6B; --normal:#3D5A73; --warn:#B4761A; --stress:#A32B2B;
  --new:#2D4EA8; --void:#9AA3AE;
  --sans:"IBM Plex Sans Condensed",system-ui,-apple-system,sans-serif;
  --serif:"IBM Plex Serif",Georgia,serif;
  --mono:"IBM Plex Mono",ui-monospace,monospace;
}
@media (prefers-color-scheme:dark){
  :root{
    --paper:#14181F; --paper-2:#1C2128; --ink:#E4E8ED; --ink-soft:#98A3B0;
    --rule:#333B47; --rule-soft:#262D37;
    --calm:#4FB3AC; --normal:#7FA3C4; --warn:#D89B3F; --stress:#D96666;
    --new:#7A9BE8; --void:#5A6472;
  }
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{
  margin:0; background:var(--paper); color:var(--ink);
  font-family:var(--serif); font-size:17px; line-height:1.6;
  font-feature-settings:"kern" 1;
}
.wrap{max-width:78ch; margin:0 auto; padding:0 1.25rem 5rem}
.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}

/* --- ust bant --- */
.masthead{
  display:flex; justify-content:space-between; align-items:baseline;
  gap:1rem; flex-wrap:wrap;
  padding:1.5rem 0 .75rem; border-bottom:1px solid var(--ink);
  font-family:var(--sans);
}
.masthead__name{
  font-weight:700; font-size:1.05rem; letter-spacing:.14em;
  text-transform:uppercase; margin:0;
}
.masthead__date{font-size:.95rem; color:var(--ink-soft); letter-spacing:.02em}

/* --- rejim: sayfanin tezi --- */
.regime{padding:2.5rem 0 2rem; border-bottom:1px solid var(--rule)}
.regime__eyebrow{
  font-family:var(--sans); font-size:.75rem; letter-spacing:.18em;
  text-transform:uppercase; color:var(--ink-soft); margin:0 0 .6rem;
}
.regime__name{
  font-family:var(--sans); font-weight:600; font-size:clamp(2rem,7vw,3.4rem);
  line-height:1.02; letter-spacing:-.015em; margin:0;
}
.regime__name--none{color:var(--ink)}
.regime__name--low{color:var(--ink)}
.regime__name--medium{color:var(--warn)}
.regime__name--high{color:var(--stress)}
.regime__sub{
  font-family:var(--sans); font-size:.95rem; color:var(--ink-soft);
  margin:.8rem 0 0; display:flex; gap:.5rem; flex-wrap:wrap; align-items:center;
}
.regime__sub span+span::before{content:"·"; margin-right:.5rem; color:var(--rule)}
.regime__why{
  margin:1.4rem 0 0; padding-left:1rem; border-left:2px solid var(--rule);
  color:var(--ink-soft); font-size:.95rem; font-style:italic;
}

/* --- bolumler --- */
section{padding:2.25rem 0; border-bottom:1px solid var(--rule-soft)}
h2{
  font-family:var(--sans); font-size:.78rem; font-weight:600;
  letter-spacing:.16em; text-transform:uppercase; color:var(--ink-soft);
  margin:0 0 1.25rem;
}
.narrative p{margin:0 0 1.1rem}
.narrative p:last-child{margin-bottom:0}
.empty{color:var(--ink-soft); font-style:italic; margin:0}

/* --- IMZA: persentil konum seridi --- */
.strip{display:flex; align-items:center; gap:.6rem; margin:.5rem 0 0}
.strip__rail{
  position:relative; flex:1; height:16px;
  border-left:1px solid var(--rule); border-right:1px solid var(--rule);
}
.strip__rail::before{
  content:""; position:absolute; left:0; right:0; top:50%;
  height:1px; background:var(--rule);
}
.strip__rail--hatched{
  background:repeating-linear-gradient(45deg,transparent,transparent 4px,
    var(--rule-soft) 4px,var(--rule-soft) 5px);
}
.strip__tick{position:absolute; top:5px; width:1px; height:6px; background:var(--rule)}
.strip__tick--mid{top:3px; height:10px; background:var(--rule)}
.strip__marker{
  position:absolute; top:50%; left:var(--pos); width:9px; height:9px;
  border-radius:50%; transform:translate(-50%,-50%);
  background:var(--normal); box-shadow:0 0 0 3px var(--paper);
}
.strip__marker--calm{background:var(--calm)}
.strip__marker--normal{background:var(--normal)}
.strip__marker--warn{background:var(--warn)}
.strip__marker--stress{background:var(--stress)}
.strip__marker--void{background:var(--void)}
.strip__val{
  font-family:var(--mono); font-size:.72rem; color:var(--ink-soft);
  min-width:3.2em; text-align:right; font-variant-numeric:tabular-nums;
}
.strip__none{
  font-family:var(--sans); font-size:.72rem; color:var(--void);
  min-width:7em; text-align:right;
}

/* --- one cikanlar --- */
.hl-list{list-style:none; margin:0; padding:0; display:grid; gap:1.5rem}
.hl__head{display:flex; align-items:baseline; gap:.6rem; flex-wrap:wrap}
.hl__name{font-family:var(--sans); font-weight:600; font-size:1.05rem}
.hl__num{
  font-family:var(--mono); font-size:1.05rem; margin-left:auto;
  font-variant-numeric:tabular-nums;
}
.hl__meta{font-family:var(--sans); font-size:.8rem; color:var(--ink-soft); margin-top:.35rem}
.tag{
  font-family:var(--sans); font-size:.65rem; font-weight:600;
  letter-spacing:.12em; text-transform:uppercase;
  padding:.15rem .45rem; border:1px solid currentColor;
}
.tag--new{color:var(--new)}
.tag--cont{color:var(--void)}
.hl--devam .hl__name{font-weight:400; color:var(--ink-soft)}

/* --- tablo --- */
.levels{width:100%; border-collapse:collapse; font-family:var(--sans); font-size:.9rem}
.levels th[scope=col]{
  text-align:left; font-size:.68rem; letter-spacing:.1em; text-transform:uppercase;
  color:var(--ink-soft); font-weight:600; padding:0 .6rem .5rem 0;
  border-bottom:1px solid var(--rule);
}
.levels tbody th{text-align:left; font-weight:500; padding:.7rem .6rem .7rem 0}
.levels td{padding:.7rem .6rem .7rem 0; border-bottom:1px solid var(--rule-soft); vertical-align:middle}
.levels tbody th{border-bottom:1px solid var(--rule-soft)}
.num{font-family:var(--mono); font-variant-numeric:tabular-nums; white-space:nowrap}
.strip-cell{min-width:11rem}
.band{font-size:.72rem; letter-spacing:.06em; text-transform:uppercase; font-weight:600}
.band--calm{color:var(--calm)} .band--normal{color:var(--normal)}
.band--warn{color:var(--warn)} .band--stress{color:var(--stress)}
.band--void{color:var(--void)}
.note{font-size:.75rem; color:var(--void)}
.note-cell{max-width:12rem}

/* --- uyarilar --- */
.alerts{list-style:none; margin:0; padding:0; display:grid; gap:.9rem}
.alert{display:flex; gap:.75rem; align-items:flex-start; font-family:var(--sans); font-size:.92rem}
.alert__arrow{font-family:var(--mono); font-size:1.1rem; color:var(--warn); line-height:1.2}
.alert__detail{display:block; font-size:.82rem; color:var(--ink-soft)}
.quality{margin:0; padding-left:1.1rem; font-size:.9rem; color:var(--ink-soft)}
.quality li{margin-bottom:.5rem}

/* --- alt bilgi --- */
.colophon{
  padding-top:2rem; font-family:var(--sans); font-size:.78rem;
  color:var(--ink-soft); display:grid; gap:.5rem;
}
.colophon dl{display:flex; flex-wrap:wrap; gap:.4rem 1.5rem; margin:0}
.colophon dt{font-weight:600}
.colophon dd{margin:0; font-family:var(--mono)}
.disclaimer{
  margin-top:1.25rem; padding-top:1rem; border-top:1px solid var(--rule);
  font-style:italic;
}
.badge{
  display:inline-block; font-family:var(--sans); font-size:.68rem;
  font-weight:600; letter-spacing:.1em; text-transform:uppercase;
  padding:.2rem .5rem; border:1px solid var(--warn); color:var(--warn);
  margin-bottom:1rem;
}

/* --- tek orkestre edilmis an: isaretciler yerine kayar --- */
@media (prefers-reduced-motion:no-preference){
  .strip__marker{animation:slide .7s cubic-bezier(.2,.8,.2,1) both}
  .hl-list .hl:nth-child(1) .strip__marker{animation-delay:.05s}
  .hl-list .hl:nth-child(2) .strip__marker{animation-delay:.12s}
  .hl-list .hl:nth-child(3) .strip__marker{animation-delay:.19s}
  .hl-list .hl:nth-child(4) .strip__marker{animation-delay:.26s}
  .hl-list .hl:nth-child(n+5) .strip__marker{animation-delay:.33s}
  @keyframes slide{from{left:0;opacity:0}to{left:var(--pos);opacity:1}}
}
:focus-visible{outline:2px solid var(--new); outline-offset:2px}
@media (max-width:620px){
  body{font-size:16px}
  .levels{font-size:.82rem}
  .note-cell{display:none}
  .strip-cell{min-width:7rem}
}
</style>
</head>
<body>
<main class="wrap">

  <header class="masthead">
    <h1 class="masthead__name">GlobalPulse</h1>
    <span class="masthead__date">$date_tr</span>
  </header>

  <div class="regime">
    <p class="regime__eyebrow">Bugünün rejimi</p>
    <p class="regime__name regime__name--$severity">$regime_label</p>
    <p class="regime__sub">
      <span>$severity_label</span>
      <span>$change_note</span>
    </p>
    $regime_why
  </div>

  <section class="narrative">
    <h2>Yorum</h2>
    $fallback_badge
    $narrative
  </section>

  <section>
    <h2>Öne çıkanlar — 2 yıllık aralığın uçları</h2>
    $highlights
  </section>

  <section>
    <h2>Hız uyarıları</h2>
    $alerts
  </section>

  <section>
    <h2>Göstergeler</h2>
    $levels
  </section>

  <section>
    <h2>Veri kalitesi</h2>
    $quality
  </section>

  <footer class="colophon">
    <dl>
      <dt>Üretim</dt><dd>$generated_at</dd>
      <dt>Kaynak</dt><dd>$narrative_source</dd>
      <dt>Bu ay</dt><dd>$month_calls çağrı · $month_cost</dd>
      <dt>Toplam</dt><dd>$total_calls çağrı · $total_cost</dd>
    </dl>
    <p class="disclaimer">$disclaimer</p>
  </footer>

</main>
</body>
</html>
""")


def build_html(payload: dict, cfg: dict) -> str:
    c = payload["classification"]
    r = c["regime"]
    usage = payload.get("usage_summary") or {}
    month = usage.get("this_month") or {}

    fallback_badge = ""
    source = "Claude (claude-haiku-4-5)"
    if payload.get("generated_by") != "llm":
        fallback_badge = ('<p class="badge">otomatik özet — dil modeli '
                          "yorumu üretilemedi</p>")
        source = "kural tabanlı yedek metin"

    why = ""
    if r.get("note"):
        why = f'<p class="regime__why">{esc(r["note"])}</p>'

    nh = sum(1 for h in c.get("highlights", []) if h.get("status") == "yeni")
    change_note = (f"{nh} yeni öne çıkan" if nh
                   else "yeni öne çıkan yok")

    return TEMPLATE.substitute(
        date_tr=esc(tr_date(c["as_of"])),
        regime_label=esc(r["label"]),
        severity=esc(r.get("severity", "none")),
        severity_label=esc(SEVERITY_LABELS.get(r.get("severity", "none"), "")),
        change_note=esc(change_note),
        regime_why=why,
        fallback_badge=fallback_badge,
        narrative=render_narrative(payload.get("narrative", "")),
        highlights=render_highlights(c.get("highlights", [])),
        alerts=render_alerts(c.get("velocity_alerts", [])),
        levels=render_levels(c.get("levels", {})),
        quality=render_quality(c),
        generated_at=esc((payload.get("generated_at") or "")[:16].replace("T", " ") + " UTC"),
        narrative_source=esc(source),
        month_calls=month.get("calls", 0),
        month_cost=f"${month.get('cost_usd', 0):.4f}",
        total_calls=usage.get("total_calls", 0),
        total_cost=f"${usage.get('total_cost_usd', 0):.4f}",
        disclaimer=esc(payload.get("disclaimer", "")),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--latest", default=str(LATEST_PATH))
    args = ap.parse_args()

    latest = Path(args.latest)
    if not latest.exists():
        print(f"[HATA] {latest} yok. Önce narrate.py çalıştır.", file=sys.stderr)
        return 1

    payload = json.loads(latest.read_text(encoding="utf-8"))
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(payload, cfg), encoding="utf-8")

    kb = out.stat().st_size / 1024
    print(f"Yazıldı: {out.relative_to(ROOT)}  ({kb:.1f} KB)")
    print(f"  Tarih  : {payload['classification']['as_of']}")
    print(f"  Rejim  : {payload['classification']['regime']['label']}")
    print(f"  Kaynak : {payload.get('generated_by')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
