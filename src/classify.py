"""
classify.py - GlobalPulse deterministik rejim siniflandirici

Bu modulde LLM YOKTUR. Girdi -> kural -> cikti. Ayni girdi her zaman
ayni ciktiyi verir. LLM sadece bu modulun ciktisini Turkce anlatiya
cevirir (narrate.py).

v0.2 degisiklikleri:
  - Takvim gunu yerine IS GUNU indeksi. _chg_20d artik gercekten
    20 is gunu (~1 ay). Onceki surumde 20 takvim gunu = 14 is gunuydu.
  - Persentil yalnizca GERCEK gozlemlerden hesaplanir. ffill ile
    uretilen satirlar sayilmaz; yetersiz gozlemde persentil None doner.
  - "highlights": rejim etiketinden bagimsiz olarak ucta olan
    gostergeleri one cikarir.

Kullanim:
    import pandas as pd, yaml
    from classify import classify

    df  = pd.read_parquet("data/history.parquet")
    cfg = yaml.safe_load(open("config/thresholds.yaml", encoding="utf-8"))
    result = classify(df, cfg)
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

# ---------------------------------------------------------------
# Seri tipleri: degisim hesabinin birimini belirler
# ---------------------------------------------------------------
# rate   -> FRED'den yuzde olarak gelir (4.25 = %4.25), degisim bp
# spread -> FRED'den yuzde olarak gelir (3.10 = 310bp), SEVIYE bp'ye cevrilir
# cds    -> zaten bp, degisim bp
# price  -> endeks/fiyat, degisim %
SERIES_KIND = {
    "ust_2y":        "rate",
    "ust_10y":       "rate",
    "real_10y":      "rate",
    "breakeven_10y": "rate",
    "fwd_5y5y":      "rate",
    "hy_oas":        "spread",
    "ig_oas":        "spread",
    "turkey_cds_5y": "cds",
    "curve_2s10s":   "cds",
    "vix":           "price",
    "move":          "price",
    "dxy":           "price",
    "dxy_broad":     "price",
    "gold":          "price",
    "brent":         "price",
    "copper":        "price",
    "usdjpy":        "price",
    "spx":           "price",
    "em_equity":     "price",
    "usdtry":        "price",
    "xu100":         "price",
    # Turetilmis seriler
    "bist_usd":      "price",   # xu100 / usdtry
    "tr_rel_5d":     "cds",     # yuzde puan; degisim basit fark
}

# Degisim pencereleri - IS GUNU cinsinden
#   5  ~ 1 hafta | 20 ~ 1 ay | 30 ~ 6 hafta | 60 ~ 3 ay
CHANGE_WINDOWS = [1, 5, 20, 30, 60]

# Bir gozlem en fazla kac IS GUNU ileri tasinir.
# FRED gecikmesi 1-3 gun; haftalik manuel seriler icin daha genis gerekir.
FFILL_LIMIT = 10

# Persentil icin gereken minimum GERCEK gozlem sayisi.
MIN_REAL_OBS = 60

# Persentilin "ucta" sayildigi esikler
EXTREME_HIGH = 95.0
EXTREME_LOW = 5.0

# One cikanlar bolumunde gosterilecek yorunge uzunlugu (is gunu)
SPARK_DAYS = 60


# ---------------------------------------------------------------
# 1. Normalizasyon
# ---------------------------------------------------------------
def normalize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Spread serilerini bp'ye cevirir, IS GUNU indeksine oturtur.

    Doner:
        (normalized_df, real_mask)
        real_mask: bool DataFrame. True = gercek gozlem,
                   False = ffill ile uretilmis satir.
    """
    out = df.copy()
    out.index = pd.to_datetime(out.index).normalize()
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]

    # Spread'leri yuzdeden bp'ye cevir (idempotent)
    for col in out.columns:
        if SERIES_KIND.get(col) == "spread":
            median = out[col].dropna().median()
            if pd.notna(median) and median < 50:
                out[col] = out[col] * 100.0

    if out.empty:
        return out, out.notna()

    # IS GUNU takvimi: orijinal + is gunu birlesimine yay, ffill,
    # sonra sadece is gunlerini tut.
    bidx = pd.bdate_range(out.index.min(), out.index.max())
    full = out.reindex(out.index.union(bidx))

    real_mask_full = full.notna()
    full = full.ffill(limit=FFILL_LIMIT)

    normalized = full.reindex(bidx)
    real_mask = real_mask_full.reindex(bidx, fill_value=False)

    return normalized, real_mask


# ---------------------------------------------------------------
# 2. Turetilmis metrikler
# ---------------------------------------------------------------
def build_metrics(
    df: pd.DataFrame,
    real_mask: pd.DataFrame,
    percentile_window: int = 504,
    out_series: dict | None = None,
) -> tuple[dict[str, float], dict[str, int]]:
    """
    Uretilen isimler:
        <seri>                -> son seviye
        <seri>_chg_<N>d       -> N IS GUNU degisim (rate/spread/cds: bp, price: %)
        <seri>_chg_<N>d_pct   -> N IS GUNU yuzde degisim
        <seri>_pct_rank       -> GERCEK gozlemlere gore persentil (0-100)
        curve_2s10s           -> 10Y - 2Y (bp)

    Doner: (metrics, real_obs_counts)
    """
    m: dict[str, float] = {}
    real_counts: dict[str, int] = {}

    work = df.copy()
    mask = real_mask.copy()

    # Egri seviyesi (bp) - iki bacak da gercekse gercek sayilir
    if {"ust_2y", "ust_10y"}.issubset(work.columns):
        work["curve_2s10s"] = (work["ust_10y"] - work["ust_2y"]) * 100.0
        mask["curve_2s10s"] = mask["ust_2y"] & mask["ust_10y"]

    # BIST'in dolar bazli degeri
    if {"xu100", "usdtry"}.issubset(work.columns):
        denom = work["usdtry"].where(work["usdtry"] > 0)
        work["bist_usd"] = work["xu100"] / denom
        mask["bist_usd"] = mask["xu100"] & mask["usdtry"]

    # Turkiye'ye ozgu goreli stres:
    #   BIST(USD) 5 gunluk getirisi  -  EM hisse 5 gunluk getirisi
    # Negatif = Turkiye EM'den ayrisarak geride kaliyor.
    # CDS verisi olmadigi gunlerde idiyosinkratik riski yakalar.
    if {"bist_usd", "em_equity"}.issubset(work.columns):
        bist_ret = work["bist_usd"].pct_change(5) * 100.0
        em_ret = work["em_equity"].pct_change(5) * 100.0
        work["tr_rel_5d"] = bist_ret - em_ret
        mask["tr_rel_5d"] = mask["bist_usd"] & mask["em_equity"]

    for col in work.columns:
        s = work[col].dropna()
        if s.empty:
            continue

        kind = SERIES_KIND.get(col, "price")
        last = float(s.iloc[-1])
        if not math.isfinite(last):
            continue
        m[col] = last

        # --- Persentil: SADECE gercek gozlemler ---
        window_slice = s.tail(percentile_window)
        col_mask = mask[col].reindex(window_slice.index, fill_value=False)
        real_vals = window_slice[col_mask].dropna()
        real_counts[col] = int(len(real_vals))

        if len(real_vals) >= MIN_REAL_OBS:
            m[f"{col}_pct_rank"] = float((real_vals <= last).mean() * 100.0)

        # --- Degisimler (is gunu) ---
        for n in CHANGE_WINDOWS:
            if len(s) <= n:
                continue
            prev = float(s.iloc[-(n + 1)])
            if not math.isfinite(prev):
                continue

            if prev != 0:
                m[f"{col}_chg_{n}d_pct"] = (last / prev - 1.0) * 100.0

            if kind == "rate":
                m[f"{col}_chg_{n}d"] = (last - prev) * 100.0      # bp
            elif kind in ("spread", "cds"):
                m[f"{col}_chg_{n}d"] = last - prev                # zaten bp
            elif prev != 0:
                m[f"{col}_chg_{n}d"] = (last / prev - 1.0) * 100.0

    if out_series is not None:
        out_series["df"] = work      # turetilmis serileri de icerir

    clean = {k: v for k, v in m.items() if v is not None and math.isfinite(v)}
    return clean, real_counts


# ---------------------------------------------------------------
# 3. Seviye siniflandirmasi
# ---------------------------------------------------------------
def classify_levels(
    metrics: dict[str, float],
    real_counts: dict[str, int],
    cfg: dict,
) -> dict[str, dict]:
    out: dict[str, dict] = {}

    for name, bands in cfg.get("levels", {}).items():
        if name not in metrics:
            out[name] = {
                "value": None, "band": "veri_yok",
                "pct_rank": None, "real_obs": real_counts.get(name, 0),
            }
            continue

        val = metrics[name]
        band = "tanimsiz"
        for band_name, rng in bands.items():
            lo, hi = rng
            if lo <= val < hi:
                band = band_name
                break

        pct = metrics.get(f"{name}_pct_rank")
        out[name] = {
            "value": round(val, 2),
            "band": band,
            "pct_rank": round(pct, 1) if pct is not None else None,
            "real_obs": real_counts.get(name, 0),
        }

    return out


# ---------------------------------------------------------------
# 4. Hiz / momentum uyarilari
# ---------------------------------------------------------------
def check_velocity(metrics: dict[str, float], cfg: dict) -> list[dict]:
    alerts: list[dict] = []

    for name, spec in cfg.get("velocity", {}).items():
        n = spec["window_days"]

        if "threshold_bp" in spec:
            key, thr, unit = f"{name}_chg_{n}d", spec["threshold_bp"], "bp"
        else:
            key, thr, unit = f"{name}_chg_{n}d_pct", spec["threshold_pct"], "%"

        if key not in metrics:
            continue

        val = metrics[key]
        if abs(val) >= thr:
            alerts.append({
                "indicator": name,
                "label": spec["label"],
                "window_days": n,
                "change": round(val, 1),
                "unit": unit,
                "threshold": thr,
                "direction": "yukari" if val > 0 else "asagi",
            })

    return alerts


# ---------------------------------------------------------------
# 5. One cikanlar (rejimden bagimsiz)
# ---------------------------------------------------------------
def find_highlights(
    metrics: dict[str, float],
    cfg: dict,
    series: pd.DataFrame | None = None,
    window: int = 504,
) -> list[dict]:
    """
    Persentil ucunda olan gostergeleri toplar. Rejim "Sakin seyir"
    ciktigi gunlerde bile dikkat cekilmesi gerekenler kaybolmasin diye.
    """
    highlights: list[dict] = []
    tracked = set(cfg.get("levels", {}).keys()) | {
        "ust_2y", "ust_10y", "curve_2s10s", "dxy", "gold",
        "brent", "copper", "usdjpy", "spx", "em_equity",
        "bist_usd", "tr_rel_5d",
    }

    for name in sorted(tracked):
        key = f"{name}_pct_rank"
        if key not in metrics or name not in metrics:
            continue

        pct = metrics[key]
        if pct >= EXTREME_HIGH:
            note = "2 yillik araligin tepesine yakin"
        elif pct <= EXTREME_LOW:
            note = "2 yillik araligin dibine yakin"
        else:
            continue

        kind = SERIES_KIND.get(name, "price")
        unit = "bp" if kind in ("rate", "spread", "cds") else "%"

        item = {
            "indicator": name,
            "value": round(metrics[name], 2),
            "pct_rank": round(pct, 1),
            "note": note,
            "chg_20d": round(metrics[f"{name}_chg_20d"], 1)
            if f"{name}_chg_20d" in metrics else None,
            "chg_unit": unit,
        }

        # Yorunge: son SPARK_DAYS is gunu + 2 yillik aralik.
        # "p100'de" olmak tek basina yetersiz; oraya nasil geldigi
        # (aylardir orada mi, dun mu firladi) farkli bir bilgi.
        if series is not None and name in series.columns:
            s = series[name].dropna()
            ref = s.tail(window)
            tail = s.tail(SPARK_DAYS)
            if len(tail) >= 10 and len(ref) >= 10:
                lo, hi = float(ref.min()), float(ref.max())
                if hi > lo:
                    item["spark"] = [round(float(v), 4) for v in tail]
                    item["spark_range"] = [round(lo, 4), round(hi, 4)]

        highlights.append(item)

    highlights.sort(key=lambda h: abs(h["pct_rank"] - 50), reverse=True)
    return highlights


# ---------------------------------------------------------------
# 6. Bilesik rejim kurallari
# ---------------------------------------------------------------
_SAFE_GLOBALS = {"__builtins__": {}, "abs": abs, "min": min, "max": max}


def _eval_condition(expr: str, metrics: dict[str, float]) -> bool | None:
    """
    True/False -> degerlendirildi
    None       -> gerekli metrik yok, kural atlanmali (False DEGIL)
    """
    try:
        return bool(eval(expr, _SAFE_GLOBALS, metrics))  # noqa: S307
    except NameError:
        return None
    except Exception:
        return None


def match_regime(metrics: dict[str, float], cfg: dict) -> tuple[dict, list[dict]]:
    regimes = sorted(cfg.get("regimes", []), key=lambda r: r.get("priority", 9999))
    skipped: list[dict] = []

    for reg in regimes:
        conds = reg.get("conditions") or []

        if not conds:  # catch-all
            return {
                "id": reg["id"],
                "label": reg["label"],
                "severity": reg.get("severity", "none"),
                "note": reg.get("note", "").strip(),
                "matched_conditions": [],
            }, skipped

        results = [(c, _eval_condition(c, metrics)) for c in conds]

        if any(r is None for _, r in results):
            skipped.append({
                "id": reg["id"],
                "label": reg["label"],
                "missing": [c for c, r in results if r is None],
            })
            continue

        if all(r for _, r in results):
            return {
                "id": reg["id"],
                "label": reg["label"],
                "severity": reg.get("severity", "none"),
                "note": reg.get("note", "").strip(),
                "matched_conditions": conds,
            }, skipped

    return {
        "id": "unknown",
        "label": "Siniflandirilamadi",
        "severity": "none",
        "note": "Hicbir kural eslesmedi ve catch-all tanimli degil.",
        "matched_conditions": [],
    }, skipped


# ---------------------------------------------------------------
# 7. Ana giris noktasi
# ---------------------------------------------------------------
def classify(df: pd.DataFrame, cfg: dict) -> dict[str, Any]:
    window = cfg.get("meta", {}).get("percentile_window_days", 504)

    norm, real_mask = normalize(df)
    holder: dict = {}
    metrics, real_counts = build_metrics(
        norm, real_mask, percentile_window=window, out_series=holder
    )
    full = holder.get("df", norm)

    regime, skipped = match_regime(metrics, cfg)

    no_pct = sorted(
        name for name in cfg.get("levels", {})
        if name in metrics and f"{name}_pct_rank" not in metrics
    )

    return {
        "as_of": str(norm.index[-1].date()),
        "regime": regime,
        "levels": classify_levels(metrics, real_counts, cfg),
        "highlights": find_highlights(metrics, cfg, series=full, window=window),
        "velocity_alerts": check_velocity(metrics, cfg),
        "skipped_regimes": skipped,
        "missing_series": [s for s in cfg.get("levels", {}) if s not in metrics],
        "no_percentile": no_pct,
        "metrics": {k: round(v, 3) for k, v in sorted(metrics.items())},
    }


# ---------------------------------------------------------------
# CLI: python src/classify.py data/history.parquet config/thresholds.yaml
# ---------------------------------------------------------------
if __name__ == "__main__":
    import json
    import sys

    import yaml

    path = sys.argv[1] if len(sys.argv) > 1 else "data/history.parquet"
    cfg_path = sys.argv[2] if len(sys.argv) > 2 else "config/thresholds.yaml"

    df = pd.read_parquet(path)
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    result = classify(df, cfg)

    summary = {k: v for k, v in result.items() if k != "metrics"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n--- metrics ---")
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))
