"""
classify.py - GlobalPulse deterministik rejim siniflandirici

Bu modulde LLM YOKTUR. Girdi -> kural -> cikti. Ayni girdi her zaman
ayni ciktiyi verir. LLM sadece bu modulun ciktisini Turkce anlatiya
cevirir (narrate.py).

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

import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# Seri tipleri: degisim hesabinin birimini belirler
# ---------------------------------------------------------------
# rate   -> FRED'den yuzde olarak gelir (4.25 = %4.25), degisim bp
# spread -> FRED'den yuzde olarak gelir (3.10 = 310bp), SEVIYE bp'ye cevrilir
# cds    -> zaten bp (elle girilir), degisim hem bp hem %
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
}

# Hangi seri icin hangi pencerelerde degisim hesaplansin
CHANGE_WINDOWS = [1, 5, 20, 30, 60]


# ---------------------------------------------------------------
# 1. Normalizasyon
# ---------------------------------------------------------------
def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Spread serilerini yuzdeden bp'ye cevirir, gunluk frekansa oturtur."""
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()

    for col in out.columns:
        if SERIES_KIND.get(col) == "spread":
            # FRED OAS yuzde cinsinden gelir; 3.10 -> 310 bp
            # Zaten bp gibi gorunuyorsa (>50) dokunma - idempotent olsun
            median = out[col].dropna().median()
            if pd.notna(median) and median < 50:
                out[col] = out[col] * 100.0

    # Takvim gunlerine yay ve ileri doldur (tatil/veri gecikmesi icin)
    out = out.asfreq("D") if out.index.inferred_freq else out.resample("D").last()
    out = out.ffill(limit=7)
    return out


# ---------------------------------------------------------------
# 2. Turetilmis metrikler
# ---------------------------------------------------------------
def build_metrics(df: pd.DataFrame, percentile_window: int = 504) -> dict[str, float]:
    """
    Kurallarda kullanilacak tum metrikleri uretir.

    Uretilen isimler:
        <seri>                    -> son seviye
        <seri>_chg_<N>d           -> N gunluk degisim (rate/spread/cds: bp, price: %)
        <seri>_chg_<N>d_pct       -> N gunluk yuzde degisim (her seri icin)
        <seri>_pct_rank           -> son <percentile_window> gun icindeki persentil (0-100)
        curve_2s10s               -> 10Y - 2Y (bp)
        curve_2s10s_chg_<N>d      -> egri degisimi (bp)
    """
    m: dict[str, float] = {}

    work = df.copy()

    # Egri seviyesi (bp)
    if {"ust_2y", "ust_10y"}.issubset(work.columns):
        work["curve_2s10s"] = (work["ust_10y"] - work["ust_2y"]) * 100.0
        SERIES_KIND.setdefault("curve_2s10s", "cds")  # zaten bp

    for col in work.columns:
        s = work[col].dropna()
        if s.empty:
            continue

        kind = SERIES_KIND.get(col, "price")
        last = float(s.iloc[-1])
        m[col] = last

        # Persentil (rejim kaymasina karsi koruma)
        window = s.tail(percentile_window)
        if len(window) >= 30:
            m[f"{col}_pct_rank"] = float((window <= last).mean() * 100.0)

        # Degisimler
        for n in CHANGE_WINDOWS:
            if len(s) <= n:
                continue
            prev = float(s.iloc[-(n + 1)])
            if not math.isfinite(prev):
                continue

            # Yuzde degisim her seri icin uretilir
            if prev != 0:
                m[f"{col}_chg_{n}d_pct"] = (last / prev - 1.0) * 100.0

            # Ana degisim metrigi seri tipine gore
            if kind in ("rate",):
                m[f"{col}_chg_{n}d"] = (last - prev) * 100.0      # bp
            elif kind in ("spread", "cds"):
                m[f"{col}_chg_{n}d"] = last - prev                # zaten bp
            else:
                m[f"{col}_chg_{n}d"] = (last / prev - 1.0) * 100.0 if prev else float("nan")

    return {k: v for k, v in m.items() if v is not None and math.isfinite(v)}


# ---------------------------------------------------------------
# 3. Seviye siniflandirmasi
# ---------------------------------------------------------------
def classify_levels(metrics: dict[str, float], cfg: dict) -> dict[str, dict]:
    """Her gosterge icin sakin/normal/uyari/stres etiketi uretir."""
    out: dict[str, dict] = {}

    for name, bands in cfg.get("levels", {}).items():
        if name not in metrics:
            out[name] = {"value": None, "band": "veri_yok", "pct_rank": None}
            continue

        val = metrics[name]
        band = "tanimsiz"
        for band_name, rng in bands.items():
            lo, hi = rng
            if lo <= val < hi:
                band = band_name
                break

        out[name] = {
            "value": round(val, 2),
            "band": band,
            "pct_rank": round(metrics[f"{name}_pct_rank"], 1)
            if f"{name}_pct_rank" in metrics else None,
        }

    return out


# ---------------------------------------------------------------
# 4. Hiz / momentum uyarilari
# ---------------------------------------------------------------
def check_velocity(metrics: dict[str, float], cfg: dict) -> list[dict]:
    """Esik asan hizli hareketleri dondurur."""
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
# 5. Bilesik rejim kurallari
# ---------------------------------------------------------------
_SAFE_GLOBALS = {"__builtins__": {}, "abs": abs, "min": min, "max": max}


def _eval_condition(expr: str, metrics: dict[str, float]) -> bool | None:
    """
    Kosulu degerlendirir.
    True/False -> degerlendirildi
    None       -> gerekli metrik yok, kural atlanmali (False DEGIL)
    """
    try:
        result = eval(expr, _SAFE_GLOBALS, metrics)  # noqa: S307
        return bool(result)
    except NameError:
        return None
    except Exception:
        return None


def match_regime(metrics: dict[str, float], cfg: dict) -> tuple[dict, list[dict]]:
    """
    Oncelik sirasina gore ilk eslesen rejimi dondurur.
    Ikinci donen deger: degerlendirilemeyen kurallarin listesi (seffaflik icin).
    """
    regimes = sorted(cfg.get("regimes", []), key=lambda r: r.get("priority", 999))
    skipped: list[dict] = []

    for reg in regimes:
        conds = reg.get("conditions") or []

        if not conds:  # catch-all (neutral)
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
# 6. Ana giris noktasi
# ---------------------------------------------------------------
def classify(df: pd.DataFrame, cfg: dict) -> dict[str, Any]:
    """
    Tam siniflandirma ciktisi. narrate.py bu sozlugu alip Turkce
    anlatiya cevirir - ham sayilari degil.
    """
    window = cfg.get("meta", {}).get("percentile_window_days", 504)

    norm = normalize(df)
    metrics = build_metrics(norm, percentile_window=window)

    regime, skipped = match_regime(metrics, cfg)

    return {
        "as_of": str(norm.index[-1].date()),
        "regime": regime,
        "levels": classify_levels(metrics, cfg),
        "velocity_alerts": check_velocity(metrics, cfg),
        "skipped_regimes": skipped,
        "missing_series": [
            s for s in cfg.get("levels", {}) if s not in metrics
        ],
        "metrics": {k: round(v, 3) for k, v in sorted(metrics.items())},
    }


# ---------------------------------------------------------------
# CLI: python classify.py data/history.parquet
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

    print(json.dumps(classify(df, cfg), indent=2, ensure_ascii=False))
