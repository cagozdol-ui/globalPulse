"""
narrate.py - GlobalPulse anlati katmani

classify.py'nin ciktisini alir, Claude API ile Turkce yoruma cevirir.

Tasarim ilkeleri:
  1. LLM'e HAM SAYI YIGINI verilmez. Sadece siniflandirilmis ozet
     verilir (rejim, bantlar, persentiller, one cikanlar). Boylece
     LLM sinif atamaz, sadece aciklar - halusinasyon yuzeyi kucuk.
  2. Highlight sureklilik takibi: bir gosterge 40 gundur ucta ise
     her gun ayni paragrafi yazmasin diye "yeni" / "devam (N gun)"
     ayrimi yapilir.
  3. Her cagrinin token ve maliyeti data/usage.json'a yazilir.
  4. API anahtari yoksa veya cagri basarisizsa is DURMAZ: kural
     tabanli bir yedek metin uretilir, dashboard yine calisir.

Kullanim:
    python src/narrate.py                # uret ve kaydet
    python src/narrate.py --dry-run      # API cagirma, promptu goster
    python src/narrate.py --show-usage   # maliyet ozetini bas

Ortam degiskeni:
    ANTHROPIC_API_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify import classify  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "thresholds.yaml"
HISTORY_PATH = ROOT / "data" / "history.parquet"
LATEST_PATH = ROOT / "data" / "latest.json"


# ===============================================================
# Hafiza
# ===============================================================
def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[UYARI] {path.name} okunamadi: {e}", file=sys.stderr)
        return default


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ===============================================================
# Highlight sureklilik takibi
# ===============================================================
def annotate_highlights(
    highlights: list[dict],
    history: list[dict],
    new_threshold: int = 3,
) -> list[dict]:
    """
    Her highlight icin kac ARDISIK gundur listede oldugunu hesaplar.

    streak = 1        -> bugun ilk kez
    streak < esik     -> "yeni"
    streak >= esik    -> "devam"

    Bu ayrim olmadan, aylardir trend halindeki bir gosterge
    (ornegin surekli yukselen reel faiz) her gun p100 cikar ve
    her gun ayni paragrafi uretir. Dashboard okunmaz hale gelir.
    """
    # Gecmisi yeniden eskiye sirala
    past = sorted(history, key=lambda h: h.get("as_of", ""), reverse=True)

    out = []
    for h in highlights:
        name = h["indicator"]
        streak = 1
        for entry in past:
            if name in (entry.get("highlight_ids") or []):
                streak += 1
            else:
                break

        h = dict(h)
        h["streak_days"] = streak
        h["status"] = "yeni" if streak < new_threshold else "devam"
        out.append(h)

    # Yeniler once
    out.sort(key=lambda x: (x["status"] != "yeni", -abs(x["pct_rank"] - 50)))
    return out


def regime_changed(result: dict, history: list[dict]) -> tuple[bool, str | None]:
    """Rejim dunkune gore degisti mi?"""
    if not history:
        return True, None
    past = sorted(history, key=lambda h: h.get("as_of", ""), reverse=True)
    prev = past[0].get("regime_id")
    return prev != result["regime"]["id"], prev


# ===============================================================
# Prompt insasi
# ===============================================================
def build_prompt(result: dict, history: list[dict], cfg: dict) -> tuple[str, str]:
    """LLM'e giden sistem ve kullanici mesajlarini uretir."""
    n = cfg["narrative"]

    system = f"""Sen bir finansal piyasa analistisin. Turkce yaziyorsun.

Sana ONCEDEN SINIFLANDIRILMIS bir piyasa durumu verilecek. Rejim
tespiti ve esik degerlendirmesi zaten kural tabanli bir sistem
tarafindan yapildi. Senin isin siniflandirmak DEGIL, aciklamak.

KURALLAR:
- En fazla {n['max_paragraphs']} paragraf yaz. Baslik kullanma.
- SADECE sana verilen verilerdeki sayilari kullan. Verilmemis
  hicbir gosterge, seviye veya rakam uydurma.
- "yeni" isaretli gostergelere agirlik ver. "devam" isaretli
  olanlar zaten gunlerdir boyle; onlara en fazla bir cumle ayir.
- Rejim etiketini oldugu gibi kabul et, sorgulama.
- Neden-sonuc kurarken temkinli ol: "olabilir", "isaret ediyor"
  gibi ifadeler kullan.

YASAK:
{chr(10).join('- ' + f for f in n['forbidden'])}

ZORUNLU OLARAK DEGIN:
{chr(10).join('- ' + m for m in n['must_cover'])}"""

    # --- Kullanici mesaji: sadece kurate edilmis ozet ---
    changed, prev_regime = regime_changed(result, history)

    lines = [f"TARIH: {result['as_of']}", ""]

    r = result["regime"]
    lines += [
        f"REJIM: {r['label']}  (onem: {r['severity']})",
        f"Rejim gerekcesi: {r['note'] or '-'}",
        f"Eslesen kosullar: {', '.join(r['matched_conditions']) or '-'}",
        f"Dune gore rejim degisti mi: {'EVET, onceki: ' + str(prev_regime) if changed and prev_regime else ('EVET (ilk kayit)' if changed else 'HAYIR, ayni')}",
        "",
        "GOSTERGE SEVIYELERI:",
    ]
    for k, v in result["levels"].items():
        if v["value"] is None:
            lines.append(f"  {k}: veri yok")
            continue
        pct = f", 2 yillik persentil {v['pct_rank']}" if v["pct_rank"] is not None else ", persentil hesaplanamadi (yetersiz veri)"
        lines.append(f"  {k}: {v['value']} [{v['band']}{pct}]")

    lines += ["", "ONE CIKANLAR (persentil ucunda olanlar):"]
    if result["highlights"]:
        for h in result["highlights"]:
            chg = f", 20 is gunu degisim {h['chg_20d']}" if h["chg_20d"] is not None else ""
            lines.append(
                f"  [{h['status'].upper()}] {h['indicator']}: {h['value']} "
                f"(persentil {h['pct_rank']}{chg}) - {h['streak_days']} gundur listede"
            )
    else:
        lines.append("  yok")

    lines += ["", "HIZ UYARILARI:"]
    if result["velocity_alerts"]:
        for a in result["velocity_alerts"]:
            lines.append(
                f"  {a['label']}: {a['indicator']} {a['window_days']} is gununde "
                f"{a['change']}{a['unit']} ({a['direction']})"
            )
    else:
        lines.append("  yok")

    # Veri kalitesi - LLM bunu bilmezse eksik veriyi "sakin" sanir
    quality = []
    if result["missing_series"]:
        quality.append(f"Veri gelmeyen seriler: {', '.join(result['missing_series'])}")
    if result["no_percentile"]:
        quality.append(
            f"Persentili hesaplanamayanlar (yetersiz gercek gozlem): "
            f"{', '.join(result['no_percentile'])}"
        )
    if result["skipped_regimes"]:
        quality.append(
            f"Veri eksikliginden degerlendirilemeyen kurallar: "
            f"{', '.join(s['label'] for s in result['skipped_regimes'])}"
        )
    if quality:
        lines += ["", "VERI KALITESI UYARILARI:"] + ["  " + q for q in quality]

    # Onceki yorumlar
    past = sorted(history, key=lambda h: h.get("as_of", ""), reverse=True)
    past = past[: n.get("history_lookback", 3)]
    if past:
        lines += ["", "ONCEKI GUNLERIN YORUMLARI (tekrar etme, uzerine ekle):"]
        for e in past:
            txt = (e.get("narrative") or "").replace("\n", " ")
            lines.append(f"  [{e.get('as_of')}] ({e.get('regime_label')}) {txt[:400]}")

    return system, "\n".join(lines)


# ===============================================================
# Yedek metin (API yoksa)
# ===============================================================
def fallback_narrative(result: dict) -> str:
    r = result["regime"]
    parts = [f"Bugunku rejim: {r['label']}."]
    if r["note"]:
        parts.append(r["note"])

    new_h = [h for h in result["highlights"] if h.get("status") == "yeni"]
    if new_h:
        parts.append(
            "Yeni one cikanlar: "
            + "; ".join(f"{h['indicator']} {h['value']} (p{h['pct_rank']})" for h in new_h)
            + "."
        )
    if result["velocity_alerts"]:
        parts.append(
            "Hiz uyarilari: "
            + "; ".join(f"{a['label']} ({a['change']}{a['unit']})" for a in result["velocity_alerts"])
            + "."
        )
    parts.append("(Otomatik ozet - LLM yorumu uretilemedi.)")
    return " ".join(parts)


# ===============================================================
# Maliyet takibi
# ===============================================================
def record_usage(cfg: dict, model: str, usage, as_of: str) -> dict:
    """Token sayimini ve tahmini maliyeti usage.json'a ekler."""
    n = cfg["narrative"]
    path = ROOT / n["usage_file"]

    prices = n.get("pricing_usd_per_mtok", {}).get(model)
    inp = int(getattr(usage, "input_tokens", 0) or 0)
    out = int(getattr(usage, "output_tokens", 0) or 0)

    if prices:
        cost = inp / 1e6 * prices["input"] + out / 1e6 * prices["output"]
        priced = True
    else:
        cost, priced = 0.0, False
        print(f"[UYARI] '{model}' icin fiyat tanimli degil, maliyet 0 kaydedildi.",
              file=sys.stderr)

    entry = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "as_of": as_of,
        "model": model,
        "input_tokens": inp,
        "output_tokens": out,
        "cost_usd": round(cost, 6),
        "priced": priced,
    }

    data = load_json(path, {"entries": []})
    data.setdefault("entries", []).append(entry)
    save_json(path, data)
    return entry


def summarize_usage(cfg: dict) -> dict:
    """Aylik toplamlari cikarir. render.py bunu dashboard'a basar."""
    path = ROOT / cfg["narrative"]["usage_file"]
    data = load_json(path, {"entries": []})
    entries = data.get("entries", [])

    by_month: dict[str, dict] = defaultdict(
        lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    )
    for e in entries:
        month = (e.get("as_of") or e.get("timestamp", ""))[:7]
        m = by_month[month]
        m["calls"] += 1
        m["input_tokens"] += e.get("input_tokens", 0)
        m["output_tokens"] += e.get("output_tokens", 0)
        m["cost_usd"] += e.get("cost_usd", 0.0)

    for m in by_month.values():
        m["cost_usd"] = round(m["cost_usd"], 4)

    total = round(sum(e.get("cost_usd", 0.0) for e in entries), 4)
    months = dict(sorted(by_month.items()))
    this_month = months.get(str(date.today())[:7], {})

    return {
        "total_calls": len(entries),
        "total_cost_usd": total,
        "by_month": months,
        "this_month": this_month,
    }


# ===============================================================
# API cagrisi
# ===============================================================
def call_claude(system: str, user: str, cfg: dict) -> tuple[str, object | None]:
    n = cfg["narrative"]
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()

    if not api_key:
        print("[UYARI] ANTHROPIC_API_KEY yok, yedek metin kullanilacak.",
              file=sys.stderr)
        return "", None

    try:
        from anthropic import Anthropic
    except ImportError:
        print("[HATA] 'anthropic' paketi kurulu degil.", file=sys.stderr)
        return "", None

    try:
        client = Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=n["model"],
            max_tokens=n.get("max_tokens", 1500),
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return text.strip(), resp.usage
    except Exception as e:
        print(f"[HATA] API cagrisi basarisiz: {e}", file=sys.stderr)
        return "", None


# ===============================================================
# Ana akis
# ===============================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="API cagirma, promptu ekrana bas")
    ap.add_argument("--show-usage", action="store_true",
                    help="sadece maliyet ozetini goster")
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    n = cfg["narrative"]

    if args.show_usage:
        print(json.dumps(summarize_usage(cfg), indent=2, ensure_ascii=False))
        return 0

    if not HISTORY_PATH.exists():
        print(f"[HATA] {HISTORY_PATH} yok. Once fetch.py calistir.", file=sys.stderr)
        return 1

    df = pd.read_parquet(HISTORY_PATH)
    result = classify(df, cfg)

    history_path = ROOT / n["history_file"]
    history = load_json(history_path, [])

    result["highlights"] = annotate_highlights(
        result["highlights"], history,
        new_threshold=n.get("highlight_streak_new", 3),
    )

    system, user = build_prompt(result, history, cfg)

    if args.dry_run:
        print("=" * 64 + "\nSISTEM\n" + "=" * 64)
        print(system)
        print("\n" + "=" * 64 + "\nKULLANICI\n" + "=" * 64)
        print(user)
        print("\n" + "=" * 64)
        print(f"Yaklasik giris boyutu: {len(system) + len(user)} karakter "
              f"(~{(len(system) + len(user)) // 4} token)")
        return 0

    text, usage = call_claude(system, user, cfg)

    used_llm = bool(text)
    if not used_llm:
        text = fallback_narrative(result)

    usage_entry = None
    if usage is not None:
        usage_entry = record_usage(cfg, n["model"], usage, result["as_of"])
        print(f"Token: {usage_entry['input_tokens']} giris / "
              f"{usage_entry['output_tokens']} cikis  ->  "
              f"${usage_entry['cost_usd']:.5f}")

    # Hafizaya yaz
    history = [h for h in history if h.get("as_of") != result["as_of"]]
    history.append({
        "as_of": result["as_of"],
        "regime_id": result["regime"]["id"],
        "regime_label": result["regime"]["label"],
        "highlight_ids": [h["indicator"] for h in result["highlights"]],
        "narrative": text,
        "generated_by": "llm" if used_llm else "fallback",
    })
    history = sorted(history, key=lambda h: h["as_of"])[-400:]
    save_json(history_path, history)

    # render.py icin tek dosya
    payload = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "classification": result,
        "narrative": text,
        "generated_by": "llm" if used_llm else "fallback",
        "disclaimer": n["disclaimer"].strip(),
        "usage_summary": summarize_usage(cfg),
        "last_call": usage_entry,
    }
    save_json(LATEST_PATH, payload)

    print("\n" + "=" * 64)
    print(f"{result['as_of']}  |  {result['regime']['label']}  "
          f"|  {'LLM' if used_llm else 'YEDEK METIN'}")
    print("=" * 64)
    print(text)

    s = payload["usage_summary"]
    tm = s.get("this_month") or {}
    print("\n" + "-" * 64)
    print(f"Bu ay: {tm.get('calls', 0)} cagri, ~${tm.get('cost_usd', 0):.4f}   |   "
          f"Toplam: {s['total_calls']} cagri, ~${s['total_cost_usd']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
