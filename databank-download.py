# databank-download.py – download US equity data from EODHD
# Tom Schoen, University of Konstanz

import re
import sys
import time
import json

import argparse
from datetime import datetime
from pathlib import Path

for _pkg in ("requests", "pandas"):
    try:
        __import__(_pkg)
    except ImportError:
        sys.exit(f"Missing package: {_pkg}  (pip install {_pkg})")

import requests
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


API_TOKEN = "api.token"
BASE_URL = "https://eodhd.com/api"

# Sample period. EODHD coverage starts ~1985 broadly, ~1972 for blue chips.
SAMPLE_START_DATE = "1985-01-01"
SAMPLE_END_DATE = "2025-12-31"

# Splits/dividends: request from 1970 to catch pre-sample corporate actions.
HISTORY_START = "1970-01-01"

# Exchanges mirroring CRSP: NYSE, AMEX (= NYSE MKT since 2012), NASDAQ.
VALID_EXCHANGES = {"NYSE", "NASDAQ", "AMEX", "NYSE MKT", "NYSE ARCA"}

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Rate limiting: EODHD allows 1,000 req/min. 4/sec is conservative.
REQUEST_DELAY = 0.25

# Retry settings
MAX_RETRIES = 3
RETRY_BACKOFF = 5  # seconds, doubles each retry



def _api_get(url, params):
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=30)

            if resp.status_code == 404:
                return None
            if resp.status_code == 429:
                wait = RETRY_BACKOFF * (2 ** attempt)
                print(f"WARNING:   Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            return resp

        except requests.exceptions.Timeout:
            wait = RETRY_BACKOFF * (2 ** attempt)
            print(f"WARNING:   Timeout (attempt {attempt+1}/{MAX_RETRIES}), "
                  f"retry in {wait}s")
            time.sleep(wait)

        except requests.exceptions.RequestException as e:
            print(f"WARNING:   Request error: {e}")
            return None

    return None


def _csv_safe(val):  # escape commas/quotes for manual CSV writes
    if val is None or (isinstance(val, float) and val != val):
        return ""
    s = str(val)
    if "," in s or '"' in s or "\n" in s:
        return '"' + s.replace('"', '""') + '"'
    return s


def fetch_exchange_symbols(include_delisted=False):
    params = {"api_token": API_TOKEN, "fmt": "json"}
    if include_delisted:
        params["delisted"] = "1"

    label = "delisted" if include_delisted else "active"
    print(f"  Fetching {label} US tickers...")

    resp = _api_get(f"{BASE_URL}/exchange-symbol-list/US", params)
    if resp is None:
        print("ERROR: Failed to fetch exchange symbols")
        sys.exit(1)

    df = pd.DataFrame(resp.json())
    print(f"    -> {len(df):,} {label} tickers (all types)")
    return df


def is_non_equity(code, name):
    c = str(code).upper()
    n = str(name).upper() if pd.notna(name) else ""

    if re.search(r'[-+/.]W[ST]?$', c):
        return True, "warrant:ticker_suffix"
    if any(w in n for w in ["WARRANT", "WARRANTS"]):
        return True, "warrant:name"

    # units (exclude "UNITED" false positives)
    if re.search(r'[-.]U[N]?$', c):
        return True, "unit:ticker_suffix"
    if " UNIT" in n and "UNITED" not in n:
        return True, "unit:name"

    if re.search(r'[-.]R[T]?$', c):
        return True, "rights:ticker_suffix"
    if " RIGHT" in n:
        return True, "rights:name"

    if re.search(r'[-.]N[T]?$', c) and len(c) > 3:
        return True, "notes:ticker_suffix"

    if re.search(r'[-.]P[A-Z]?$', c) and "PREFERRED" in n:
        return True, "preferred:ticker_suffix"

    return False, ""


def classify_domicile(isin):
    if isin and isinstance(isin, str) and len(isin) >= 2:
        return "domestic" if isin[:2].upper() == "US" else "foreign"
    return "unclassified"


def build_universe():
    print("Building universe...")

    # Fetch active
    active = fetch_exchange_symbols(include_delisted=False)
    active["is_delisted"] = False
    active.to_csv(DATA_DIR / "universe_active_raw.csv", index=False,
                   encoding="utf-8")

    # Fetch delisted (returns active+delisted, so remove active)
    delisted = fetch_exchange_symbols(include_delisted=True)
    delisted = delisted[~delisted["Code"].isin(set(active["Code"]))]
    delisted["is_delisted"] = True
    delisted.to_csv(DATA_DIR / "universe_delisted_raw.csv", index=False,
                     encoding="utf-8")

    # Combine
    combined = pd.concat([active, delisted], ignore_index=True)
    print(f"\n  Raw universe: {len(active):,} active + "
                f"{len(delisted):,} delisted = {len(combined):,} total")

    # --- Filter cascade ---
    excluded = []

    # 1. Exchange
    mask = combined["Exchange"].isin(VALID_EXCHANGES)
    for _, r in combined[~mask].iterrows():
        excluded.append({"Code": r["Code"], "reason": f"exchange:{r['Exchange']}"})
    combined = combined[mask].copy()
    print(f"  After exchange filter:      {len(combined):,}")

    # 2. Common Stock only
    mask = combined["Type"] == "Common Stock"
    for _, r in combined[~mask].iterrows():
        excluded.append({"Code": r["Code"], "reason": f"type:{r['Type']}"})
    combined = combined[mask].copy()
    print(f"  After Common Stock filter:  {len(combined):,}")

    # 3. Suspect ticker filter
    keep = []
    for _, r in combined.iterrows():
        bad, reason = is_non_equity(r["Code"], r.get("Name", ""))
        if bad:
            excluded.append({"Code": r["Code"], "reason": reason})
        keep.append(not bad)
    combined = combined[keep].copy()
    print(f"  After non-equity filter:    {len(combined):,}")

    # 4. Currency filter (USD only)
    mask = combined["Currency"] == "USD"
    for _, r in combined[~mask].iterrows():
        excluded.append({"Code": r["Code"], "reason": f"currency:{r['Currency']}"})
    combined = combined[mask].copy()
    print(f"  After currency filter:      {len(combined):,}")

    # 5. Domicile classification from ISIN
    if "Isin" in combined.columns:
        combined["home_category"] = combined["Isin"].apply(classify_domicile)
        dom = combined["home_category"].value_counts()
        print("\n  Domicile classification (ISIN-based):")
        for cat, n in dom.items():
            print(f"    {cat:<20s}: {n:>7,}")

    # API ticker column
    combined["ticker_api"] = combined["Code"] + ".US"

    # Save
    combined.to_csv(DATA_DIR / "universe_combined.csv", index=False,
                     encoding="utf-8")
    if excluded:
        pd.DataFrame(excluded).to_csv(
            DATA_DIR / "universe_excluded.csv", index=False,
            encoding="utf-8"
        )

    print(f"\n  Final universe: {len(combined):,} common stocks")
    print(f"    Active:   {(~combined['is_delisted']).sum():,}")
    print(f"    Delisted: {combined['is_delisted'].sum():,}")

    # Verification report
    _write_universe_report(combined, active, delisted, excluded)

    return combined


def _write_universe_report(universe, active_raw, delisted_raw, excluded):
    lines = [
        "Universe report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Raw data",
        f"  Active tickers (all types):     {len(active_raw):>10,}",
        f"  Delisted tickers (all types):   {len(delisted_raw):>10,}",
        "",
        "Exclusions",
    ]

    if excluded:
        exc_df = pd.DataFrame(excluded)
        for reason, count in exc_df["reason"].value_counts().head(20).items():
            lines.append(f"  {reason:<40s}: {count:>7,}")
        lines.append(f"  {'total excluded':<40s}: {len(excluded):>7,}")

    lines += [
        "",
        "Final universe",
        f"  Total:     {len(universe):>10,}",
        f"  Active:    {(~universe['is_delisted']).sum():>10,}",
        f"  Delisted:  {universe['is_delisted'].sum():>10,}",
        "",
        "  By exchange:",
    ]
    for exc, count in universe["Exchange"].value_counts().items():
        n_act = ((universe["Exchange"] == exc) & (~universe["is_delisted"])).sum()
        n_del = ((universe["Exchange"] == exc) & (universe["is_delisted"])).sum()
        lines.append(f"    {exc:<15s}: {count:>7,}  "
                     f"(active: {n_act:,}, delisted: {n_del:,})")

    if "home_category" in universe.columns:
        lines += ["", "  Domicile (ISIN-based):"]
        for cat, n in universe["home_category"].value_counts().items():
            pct = 100 * n / len(universe)
            lines.append(f"    {cat:<20s}: {n:>7,}  ({pct:.1f}%)")

    lines += ["", "Sanity check"]
    for t in ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA",
              "JPM", "JNJ", "V", "WMT", "BRK-A", "NVDA"]:
        found = t in universe["Code"].values
        lines.append(f"  {t:<10s}: {'ok' if found else 'MISSING'}")

    report = "\n".join(lines)
    (DATA_DIR / "universe_report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)


PRICES_FILE = DATA_DIR / "daily_prices.csv"
SPLITS_FILE = DATA_DIR / "metadata_splits.csv"
DIVS_FILE = DATA_DIR / "metadata_dividends.csv"
LOG_FILE = DATA_DIR / "download_log.csv"


def fetch_prices(ticker_api):
    resp = _api_get(f"{BASE_URL}/eod/{ticker_api}", {
        "from": SAMPLE_START_DATE, "to": SAMPLE_END_DATE,
        "period": "d", "api_token": API_TOKEN, "fmt": "json",
    })
    if resp is None:
        return None
    try:
        data = resp.json()
        return data if data else None
    except json.JSONDecodeError:
        return None


def fetch_splits(ticker_api):
    resp = _api_get(f"{BASE_URL}/splits/{ticker_api}", {
        "from": HISTORY_START, "to": SAMPLE_END_DATE,
        "api_token": API_TOKEN, "fmt": "json",
    })
    if resp is None:
        return None
    try:
        parsed = resp.json()
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    return parsed or None


def fetch_dividends(ticker_api):
    resp = _api_get(f"{BASE_URL}/div/{ticker_api}", {
        "from": HISTORY_START, "to": SAMPLE_END_DATE,
        "api_token": API_TOKEN, "fmt": "json",
    })
    if resp is None:
        return None
    try:
        data = resp.json()
    except (json.JSONDecodeError, ValueError):
        print(f"  WARNING: bad JSON for dividends {ticker_api}")
        return None
    return data if isinstance(data, list) else None


def download_all(universe):
    print("Downloading data...")

    # Resume logic: check what's already done
    downloaded = set()
    if LOG_FILE.exists():
        log_df = pd.read_csv(LOG_FILE)
        downloaded = set(log_df[log_df["status"] == "ok"]["ticker_api"])
        print(f"  Resuming: {len(downloaded):,} tickers already done")

    remaining = universe[~universe["ticker_api"].isin(downloaded)]
    print(f"  To download: {len(remaining):,} of {len(universe):,}")
    print(f"  Estimated API calls: ~{len(remaining) * 3:,}")

    if len(remaining) == 0:
        print("  All tickers already downloaded.")
        _write_download_summary()
        return

    # Open files for append -- check if headers needed before opening
    need_prices_hdr = not PRICES_FILE.exists() or PRICES_FILE.stat().st_size == 0
    need_splits_hdr = not SPLITS_FILE.exists() or SPLITS_FILE.stat().st_size == 0
    need_divs_hdr = not DIVS_FILE.exists() or DIVS_FILE.stat().st_size == 0
    need_log_hdr = not LOG_FILE.exists() or LOG_FILE.stat().st_size == 0

    ok = empty = fail = 0

    fp = open(PRICES_FILE, "a", encoding="utf-8")
    fs = open(SPLITS_FILE, "a", encoding="utf-8")
    fd = open(DIVS_FILE, "a", encoding="utf-8")
    fl = open(LOG_FILE, "a", encoding="utf-8")

    try:
        if need_prices_hdr:
            fp.write("ticker,date,close,adjusted_close,volume\n")
        if need_splits_hdr:
            fs.write("ticker,date,split_ratio\n")
        if need_divs_hdr:
            fd.write("ticker,date,value,currency,"
                     "declaration_date,record_date,payment_date\n")
        if need_log_hdr:
            fl.write("ticker_api,status,n_prices,n_splits,n_divs,timestamp\n")
        for _, row in tqdm(remaining.iterrows(), total=len(remaining),
                           desc="Downloading", unit="ticker"):
            ticker = row["Code"]
            ticker_api = row["ticker_api"]
            ts = datetime.now().isoformat()

            # --- Prices ---
            prices = fetch_prices(ticker_api)
            n_prices = 0
            if prices:
                for p in prices:
                    if all(k in p for k in ("date", "close", "adjusted_close", "volume")):
                        fp.write(f"{ticker},{p['date']},{p['close']},"
                                 f"{p['adjusted_close']},{p['volume']}\n")
                        n_prices += 1
            time.sleep(REQUEST_DELAY)

            # --- Splits ---
            splits = fetch_splits(ticker_api)
            n_splits = 0
            if splits:
                for s in splits:
                    fs.write(f"{ticker},{_csv_safe(s.get('date',''))},"
                             f"{_csv_safe(s.get('split',''))}\n")
                    n_splits += 1
            time.sleep(REQUEST_DELAY)

            # --- Dividends ---
            divs = fetch_dividends(ticker_api)
            n_divs = 0
            if divs:
                for d in divs:
                    fd.write(",".join([
                        _csv_safe(ticker),
                        _csv_safe(d.get("date", "")),
                        _csv_safe(d.get("value", "")),
                        _csv_safe(d.get("currency", "")),
                        _csv_safe(d.get("declarationDate", "")),
                        _csv_safe(d.get("recordDate", "")),
                        _csv_safe(d.get("paymentDate", "")),
                    ]) + "\n")
                    n_divs += 1
            time.sleep(REQUEST_DELAY)

            # Log result
            if n_prices > 0:
                fl.write(f"{ticker_api},ok,{n_prices},{n_splits},{n_divs},{ts}\n")
                ok += 1
            elif prices is not None:
                fl.write(f"{ticker_api},no_data,0,{n_splits},{n_divs},{ts}\n")
                empty += 1
            else:
                fl.write(f"{ticker_api},failed,0,{n_splits},{n_divs},{ts}\n")
                fail += 1

            # Flush periodically
            total_done = ok + empty + fail
            if total_done % 100 == 0:
                fp.flush(); fs.flush(); fd.flush(); fl.flush()

    except KeyboardInterrupt:
        print("\n  Interrupted. Progress saved. Re-run to resume.")
    finally:
        fp.close(); fs.close(); fd.close(); fl.close()

    print(f"\n  Phase 2 done: {ok:,} ok, {empty:,} no data, {fail:,} failed")

    _write_download_summary()


def _write_download_summary():
    buf = "Download summary\n"
    buf += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

    uni_file = DATA_DIR / "universe_combined.csv"
    if uni_file.exists():
        uni = pd.read_csv(uni_file)
        buf += (
            f"\nUniverse\n"
            f"  Total tickers:   {len(uni):>10,}\n"
            f"  Active:          {(~uni['is_delisted']).sum():>10,}\n"
            f"  Delisted:        {uni['is_delisted'].sum():>10,}\n"
        )
        if "home_category" in uni.columns:
            buf += "  Domicile:\n"
            for cat, n in uni["home_category"].value_counts().items():
                buf += f"    {cat:<20s}: {n:>7,}\n"

    if LOG_FILE.exists():
        log = pd.read_csv(LOG_FILE)
        buf += "\nDownload status\n"
        for status, count in log["status"].value_counts().items():
            buf += f"  {status:<15s}: {count:>8,}\n"

        if "n_prices" in log.columns:
            total_prices = log["n_prices"].sum()
            buf += f"  Total price rows: {total_prices:>8,.0f}\n"
        if "n_splits" in log.columns:
            total_splits = log["n_splits"].sum()
            total_divs = log["n_divs"].sum()
            buf += f"  Total splits:     {total_splits:>8,.0f}\n"
            buf += f"  Total dividends:  {total_divs:>8,.0f}\n"

    if SPLITS_FILE.exists() and SPLITS_FILE.stat().st_size > 100:
        try:
            df_s = pd.read_csv(SPLITS_FILE)
            buf += (
                f"\nSplit history\n"
                f"  Total events:    {len(df_s):>10,}\n"
                f"  Unique tickers:  {df_s['ticker'].nunique():>10,}\n"
                f"  Top ratios:\n"
            )
            for r, n in df_s["split_ratio"].value_counts().head(10).items():
                buf += f"    {str(r):<20s}: {n:>7,}\n"
        except Exception:
            pass

    if DIVS_FILE.exists() and DIVS_FILE.stat().st_size > 100:
        try:
            df_d = pd.read_csv(DIVS_FILE, low_memory=False)
            buf += (
                f"\nDividend history\n"
                f"  Total events:    {len(df_d):>10,}\n"
                f"  Unique tickers:  {df_d['ticker'].nunique():>10,}\n"
            )
        except Exception:
            pass

    buf += "\nFile sizes\n"
    for f in sorted(DATA_DIR.glob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            buf += f"  {f.name:<35s}: {size_mb:>8.1f} MB\n"

    txt = buf.rstrip("\n")
    (DATA_DIR / "download_summary.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download US equity data from EODHD"
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Run Phase 2: download prices, splits, and dividends"
    )
    args = parser.parse_args()

    t0 = datetime.now()

    # Phase 1 -- Universe (loads from disk if already built)
    universe_file = DATA_DIR / "universe_combined.csv"
    if universe_file.exists():
        print("\n  Universe already exists. Loading from disk.")
        universe = pd.read_csv(universe_file)
        print(f"  {len(universe):,} tickers loaded")
    else:
        universe = build_universe()

    # Phase 2 -- Download
    if args.download:
        download_all(universe)
    else:
        print("\n  Universe ready. To download all data:")
        print("    python databank-download.py --download")

    dt = (datetime.now() - t0).total_seconds()
    print(f"\nDone in {dt:.0f}s.")
