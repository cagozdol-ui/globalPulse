"""
fetch.py - GlobalPulse veri toplama katmani

Sorumlulugu SADECE veri indirmek ve birlestirmek. Yorum, esik,
siniflandirma yok - onlar classify.py'nin isi.

Cikti: data/history.parquet
    index   : tarih (DatetimeIndex)
    columns : thresholds.yaml'daki kanonik seri isimleri

Kullanim:
    python src/fetch.py                  # artimli guncelleme
    python src/fetch.py --full           # sifirdan tam cekim
    python src/fetch.py --dry-run        # kaydetmeden dene

Ortam degiskeni:
    FRED_API_KEY  (zorunlu, ucretsiz: fred.stlouisfed.org/docs/api/api_key.html)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
import yaml

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
DEFAULT_START = "2015-01-01"

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "thresholds.yaml"
DATA_DIR = ROOT / "data"
HISTORY_PATH = DATA_DIR / "history.parquet"
MANUAL_DIR = DATA_DIR / "manual"


# ---------------------------------------------------------------
# FRED
# ---------------------------------------------------------------
def fetch_fred_series(series_id: str, api_key: str, start: str) -> pd.Series:
    """Tek bir FRED serisini ceker. Basarisizsa bos seri doner."""
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
    }
    try:
        r = requests.get(FRED_URL, params=params, timeout=30)
        r.raise_for_status()
        obs = r.json().get("observations", [])
    except Exception as e:
        print(f"    [HATA] FRED {series_id}: {e}", file=sys.stderr)
        return pd.Series(dtype="float64")

    if not obs:
        return pd.Series(dtype="float64")

    df = pd.DataFrame(obs)
    # FRED eksik gunleri "." ile isaretler
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    s = df.set_index("date")["value"].dropna()
    s.name = series_id
    return s


def fetch_fred_all(mapping: dict[str, str], api_key: str, start: str) -> pd.DataFrame:
    frames = {}
    for canonical, series_id in mapping.items():
        print(f"  FRED  {canonical:<16} <- {series_id}")
        s = fetch_fred_series(series_id, api_key, start)
        if s.empty:
            print(f"    [UYARI] {canonical} bos dondu, atlaniyor")
            continue
        frames[canonical] = s
        time.sleep(0.15)  # nazik ol
    return pd.DataFrame(frames) if frames else pd.DataFrame()


# ---------------------------------------------------------------
# yfinance
# ---------------------------------------------------------------
def fetch_yf_all(mapping: dict[str, str], start: str) -> pd.DataFrame:
    import yfinance as yf

    frames = {}
    for canonical, ticker in mapping.items():
        if not ticker:
            continue
        print(f"  YF    {canonical:<16} <- {ticker}")
        try:
            data = yf.download(
                ticker, start=start, progress=False,
                auto_adjust=True, threads=False,
            )
        except Exception as e:
            print(f"    [HATA] {ticker}: {e}", file=sys.stderr)
            continue

        if data is None or data.empty:
            print(f"    [UYARI] {canonical} ({ticker}) bos dondu, atlaniyor")
            continue

        close = data["Close"]
        if isinstance(close, pd.DataFrame):       # MultiIndex kolon durumu
            close = close.iloc[:, 0]
        close = close.dropna()
        close.index = pd.to_datetime(close.index).tz_localize(None)
        frames[canonical] = close

    return pd.DataFrame(frames) if frames else pd.DataFrame()


# ---------------------------------------------------------------
# Manuel seriler (TR CDS gibi ucretsiz kaynagi olmayanlar)
# ---------------------------------------------------------------
def load_manual() -> pd.DataFrame:
    """
    data/manual/<seri_adi>.csv dosyalarini okur.
    Format:
        date,value
        2026-07-01,285
        2026-07-08,291
    Haftalik/duzensiz olabilir; classify.py ffill uygular.
    """
    if not MANUAL_DIR.exists():
        return pd.DataFrame()

    frames = {}
    for path in sorted(MANUAL_DIR.glob("*.csv")):
        name = path.stem
        try:
            df = pd.read_csv(path)
            df["date"] = pd.to_datetime(df["date"])
            s = df.set_index("date")["value"].astype(float).dropna()
            frames[name] = s
            print(f"  MANUEL {name:<15} <- {path.name} ({len(s)} kayit)")
        except Exception as e:
            print(f"    [HATA] manuel {path.name}: {e}", file=sys.stderr)

    return pd.DataFrame(frames) if frames else pd.DataFrame()


# ---------------------------------------------------------------
# Fallback: bir seri bosssa alternatifini kullan
# ---------------------------------------------------------------
FALLBACKS = {
    # ^MOVE cogu zaman bos doner; MOVE yoksa rejim kurallari onu atlar
    "dxy": "dxy_broad",       # klasik DXY yoksa genis dolar endeksi
}


def apply_fallbacks(df: pd.DataFrame) -> pd.DataFrame:
    for primary, backup in FALLBACKS.items():
        if primary not in df.columns and backup in df.columns:
            print(f"  [FALLBACK] {primary} yok -> {backup} kullanildi")
            df[primary] = df[backup]
        elif primary in df.columns and backup in df.columns:
            # primary'de bosluk varsa backup ile doldurma YAPMA
            # (farkli olcekte endeksler, karistirmak yaniltir)
            pass
    return df


# ---------------------------------------------------------------
# Birlestirme
# ---------------------------------------------------------------
def merge_history(new: pd.DataFrame, existing: pd.DataFrame | None) -> pd.DataFrame:
    if existing is None or existing.empty:
        out = new
    else:
        out = existing.combine_first(new)
        # yeni veri onceligi: ayni tarih+kolon icin new kazansin
        out.update(new)
        for col in new.columns:
            if col not in out.columns:
                out[col] = new[col]

    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


# ---------------------------------------------------------------
# Ana akis
# ---------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="sifirdan tam cekim")
    ap.add_argument("--dry-run", action="store_true", help="kaydetme")
    ap.add_argument("--start", default=DEFAULT_START)
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    sources = cfg.get("sources", {})

    api_key = os.environ.get("FRED_API_KEY", "").strip()
    if not api_key:
        print("[HATA] FRED_API_KEY ortam degiskeni tanimli degil.", file=sys.stderr)
        print("       Ucretsiz anahtar: fred.stlouisfed.org/docs/api/api_key.html",
              file=sys.stderr)
        return 1

    existing = None
    start = args.start

    if not args.full and HISTORY_PATH.exists():
        existing = pd.read_parquet(HISTORY_PATH)
        existing.index = pd.to_datetime(existing.index)
        # Son 90 gunu yeniden cek: FRED gecmise donuk revizyon yapar
        last = existing.index.max().date()
        start = str(max(pd.Timestamp(last).date() - timedelta(days=90),
                        pd.Timestamp(DEFAULT_START).date()))
        print(f"Artimli guncelleme. Mevcut son tarih: {last}, cekim baslangici: {start}")
    else:
        print(f"Tam cekim. Baslangic: {start}")

    print("\nVeri cekiliyor...")
    fred_df = fetch_fred_all(sources.get("fred", {}), api_key, start)
    yf_df = fetch_yf_all(sources.get("yfinance", {}), start)
    manual_df = load_manual()

    parts = [d for d in (fred_df, yf_df, manual_df) if not d.empty]
    if not parts:
        print("[HATA] Hicbir kaynaktan veri gelmedi.", file=sys.stderr)
        return 1

    new = pd.concat(parts, axis=1, sort=False)
    new.index = pd.to_datetime(new.index)
    new = new.sort_index()
    new = apply_fallbacks(new)

    combined = merge_history(new, existing)

    # Ozet
    print("\n" + "=" * 58)
    print(f"Toplam {combined.shape[0]} gun x {combined.shape[1]} seri")
    print(f"Aralik: {combined.index.min().date()} -> {combined.index.max().date()}")
    print("=" * 58)

    stale = []
    cutoff = pd.Timestamp(date.today()) - pd.Timedelta(days=10)
    for col in sorted(combined.columns):
        s = combined[col].dropna()
        if s.empty:
            print(f"  {col:<16} BOS")
            continue
        last_date = s.index.max()
        flag = ""
        if last_date < cutoff:
            flag = "  <-- BAYAT"
            stale.append(col)
        print(f"  {col:<16} son: {last_date.date()}  deger: {s.iloc[-1]:.2f}{flag}")

    if stale:
        print(f"\n[UYARI] {len(stale)} seri 10 gunden eski: {', '.join(stale)}")

    expected = set(cfg.get("levels", {}).keys())
    missing = sorted(expected - set(combined.columns))
    if missing:
        print(f"[UYARI] Esik tanimli ama veri yok: {', '.join(missing)}")

    if args.dry_run:
        print("\n--dry-run: kaydedilmedi.")
        return 0

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(HISTORY_PATH)
    print(f"\nKaydedildi: {HISTORY_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
