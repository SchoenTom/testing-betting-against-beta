# databank-prepare.py – daily prices to monthly panel with beta estimation
# Tom Schoen, University of Konstanz

import io
import re
import sys
import zipfile
from pathlib import Path
from datetime import datetime

_missing = []
for _pkg in ["numpy", "pandas", "requests"]:
    try:
        __import__(_pkg)
    except ImportError:
        _missing.append(_pkg)
if _missing:
    print(f"Missing packages: {', '.join(_missing)}")
    print(f"  pip install {' '.join(_missing)}")
    sys.exit(1)

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# F&P 2014 estimation params
VOL_WINDOW = 252          # 1-year daily window for volatility
VOL_MIN_OBS = 120         # minimum daily observations for σ
CORR_WINDOW = 1260        # 5-year daily window for correlation
CORR_MIN_OBS = 750        # minimum 3-day returns for ρ
SHRINKAGE_W = 0.6         # Vasicek shrinkage weight
BETA_PRIOR = 1.0          # Vasicek prior (cross-sectional mean)


def _download_ff_zip(url, label):
    print(f"  Downloading {label}...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = [n for n in zf.namelist()
                    if n.lower().endswith(".csv")][0]
        with zf.open(csv_name) as f:
            raw = f.read().decode("utf-8")
    return raw


def _parse_ff_csv(raw, ncols, freq="daily"):
    """Extract numeric rows from a Ken French CSV string, returning
    date + ncols value columns with percentages converted to decimals."""
    lines = raw.splitlines()
    records = []
    header_found = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if header_found and len(records) > 10:
                break  # end of main data section
            continue
        if "Mkt-RF" in stripped or "Mkt" in stripped or "Mom" in stripped:
            header_found = True
            continue
        if header_found and stripped[0].isdigit():
            parts = [p.strip() for p in stripped.split(",")]
            if len(parts) < ncols + 1:
                continue
            date_str = parts[0]
            try:
                if freq == "daily" and len(date_str) == 8:
                    dt = pd.to_datetime(date_str, format="%Y%m%d")
                elif freq == "monthly" and len(date_str) == 6:
                    dt = pd.to_datetime(date_str, format="%Y%m")
                else:
                    continue
                vals = [float(p) / 100 for p in parts[1:ncols + 1]]
                records.append([dt] + vals)
            except (ValueError, IndexError):
                continue

    return records


def download_ff_daily():
    path = DATA_DIR / "ff_factors_daily.csv"
    if path.exists():
        print("  FF daily factors: cached")
        return pd.read_csv(path, parse_dates=["date"])

    try:
        raw = _download_ff_zip(
            "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
            "ftp/F-F_Research_Data_Factors_daily_CSV.zip",
            "FF 3-factor daily"
        )
    except Exception as e:
        print(f"ERROR: could not retrieve FF daily factors: {e}")
        sys.exit(1)
    records = _parse_ff_csv(raw, ncols=4, freq="daily")
    ff = pd.DataFrame(records, columns=["date", "mkt_rf", "smb", "hml", "rf"])
    ff = ff.sort_values("date").reset_index(drop=True)
    ff.to_csv(path, index=False, encoding="utf-8")
    print(f"  Saved: {path.name} ({len(ff):,} days)")
    return ff


def download_ff_monthly():
    path = DATA_DIR / "ff_factors_monthly.csv"
    if path.exists():
        print("  FF monthly factors: cached")
        return

    try:
        raw = _download_ff_zip(
            "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
            "ftp/F-F_Research_Data_Factors_CSV.zip",
            "FF 3-factor monthly"
        )
    except Exception as e:
        print(f"ERROR: FF monthly factors unavailable: {e}")
        sys.exit(1)
    records = _parse_ff_csv(raw, ncols=4, freq="monthly")
    result = pd.DataFrame(records, columns=["month", "mkt_rf", "smb", "hml", "rf"])
    result["month"] = result["month"].dt.strftime("%Y-%m")
    result.to_csv(path, index=False, encoding="utf-8")
    print(f"  Saved: {path.name} ({len(result):,} months)")


def download_ff5_monthly():
    path = DATA_DIR / "ff5_factors_monthly.csv"
    if path.exists():
        print("  FF5 monthly factors: cached")
        return

    try:
        raw = _download_ff_zip(
            "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
            "ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip",
            "FF 5-factor monthly"
        )
    except Exception as e:
        sys.exit(f"ERROR: FF5 download failed ({e})")
    records = _parse_ff_csv(raw, ncols=6, freq="monthly")
    df = pd.DataFrame(records,
                      columns=["month", "mkt_rf", "smb", "hml", "rmw", "cma", "rf"])
    df["month"] = df["month"].dt.strftime("%Y-%m")
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"  Saved: {path.name} ({len(df):,} months)")


def download_q_factors():
    path = DATA_DIR / "q_factors_monthly.csv"
    if path.exists():
        print("  q-factors monthly: cached")
        return

    print("  Downloading q-factors...")
    url = "https://global-q.org/uploads/1/2/2/6/122636014/q5_factors_monthly_2024.csv"
    try:
        df = pd.read_csv(url)
        rename = {}
        for col in df.columns:
            cl = col.strip().lower()
            if "mkt" in cl or "market" in cl:
                rename[col] = "q_mkt"
            elif cl in ("me", "r_me"):
                rename[col] = "q_me"
            elif cl in ("ia", "r_ia"):
                rename[col] = "q_ia"
            elif cl in ("roe", "r_roe"):
                rename[col] = "q_roe"
            elif cl == "year":
                rename[col] = "year"
            elif cl == "month":
                rename[col] = "month_num"

        df = df.rename(columns=rename)

        if "year" in df.columns and "month_num" in df.columns:
            df["month"] = (df["year"].astype(str) + "-"
                           + df["month_num"].astype(str).str.zfill(2))
            for col in ["q_mkt", "q_me", "q_ia", "q_roe"]:  # pct → decimal
                if col in df.columns and df[col].abs().mean() > 1:
                    df[col] = df[col] / 100
            df = df[["month", "q_mkt", "q_me", "q_ia", "q_roe"]].dropna()
            df.to_csv(path, index=False, encoding="utf-8")
            print(f"  Saved: {path.name} ({len(df):,} months)")
        else:
            print("WARNING: q-factor column layout not recognized, skipping")
    except Exception as e:
        print(f"WARNING: q-factor download failed: {e}")


def download_ted_spread():
    path = DATA_DIR / "ted_spread.csv"
    if path.exists():
        print("  TED spread: cached")
        return

    print("  Downloading TED spread from FRED...")
    try:
        url = ("https://fred.stlouisfed.org/graph/fredgraph.csv"
               "?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans"
               "&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on"
               "&txtcolor=%23444444&ts=12&tts=12&width=1168&nt=0&thu=0"
               "&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes"
               "&id=TEDRATE&scale=left&cosd=1986-01-02&coed=2025-12-31"
               "&line_color=%234572a7&link_values=false&line_style=solid"
               "&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999"
               "&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2020-02-01"
               "&line_index=1&transformation=lin&vintage_date=2026-02-14"
               "&revision_date=2026-02-14&nd=1986-01-02")
        df = pd.read_csv(url, parse_dates=["DATE"])
        df = df.rename(columns={"DATE": "date", "TEDRATE": "ted"})
        df = df.dropna()
        df["month"] = df["date"].dt.to_period("M").astype(str)
        monthly = df.groupby("month")["ted"].mean().reset_index()
        monthly = monthly.sort_values("month")
        monthly["ted_lag"] = monthly["ted"].shift(1)
        monthly["ted_change"] = monthly["ted"].diff()
        monthly = monthly.dropna()
        monthly.to_csv(path, index=False, encoding="utf-8")
        print(f"  Saved: {path.name} ({len(monthly):,} months)")
    except Exception as e:
        print(f"WARNING: TED spread unavailable ({e})")


def download_momentum():
    path = DATA_DIR / "momentum_monthly.csv"
    if path.exists():
        print("  Momentum monthly: cached")
        return

    print("  Downloading Momentum factor...")
    try:
        raw = _download_ff_zip(
            "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
            "ftp/F-F_Momentum_Factor_CSV.zip",
            "Momentum monthly"
        )
        records = _parse_ff_csv(raw, ncols=1, freq="monthly")
        if not records:
            print("WARNING: momentum CSV empty or unrecognized format")
            return
        df = pd.DataFrame(records, columns=["month", "umd"])
        df["month"] = df["month"].dt.strftime("%Y-%m")
        df.to_csv(path, index=False, encoding="utf-8")
        print(f"  Saved: {path.name} ({len(df):,} months)")
    except Exception as e:
        print(f"WARNING: momentum factor download failed: {e}")


def download_all_factors():
    print("Step 1: Downloading factor data...")
    ff_daily = download_ff_daily()
    download_ff_monthly()
    download_ff5_monthly()
    download_q_factors()
    download_momentum()
    download_ted_spread()
    return ff_daily


def build_exclusion_set():
    """Flag preferred shares, rights, unit trusts, ETF sponsors, and warrants."""
    universe = pd.read_csv(DATA_DIR / "universe_combined.csv")
    all_codes = set(universe["Code"])
    exclude = set()

    for _, row in universe.iterrows():
        code = str(row["Code"])
        name = str(row.get("Name", ""))

        if re.search(r'-P-[A-Z]|-PR-[A-Z]', code):
            exclude.add(code)
        elif re.search(r'-U$', code):
            exclude.add(code)
        elif code.endswith("UU") and len(code) > 3:
            exclude.add(code)
        elif (code.endswith("R")
              and re.search(r'\bRights?\b', name, re.IGNORECASE)):
            exclude.add(code)
        elif re.search(
            r'^Tidal Trust|^ProShares Trust|^First Trust Exchange-Traded Fund'
            r'|^J\.P\. Morgan Exchange-Traded Fund|^Bitwise Funds|^Impax Funds',
            name
        ):
            exclude.add(code)
        elif code.endswith("W") and len(code) > 3 and code[:-1] in all_codes:
            exclude.add(code)

    print(f"  Exclusion set: {len(exclude)} tickers")
    return exclude


def load_daily_prices(exclude):
    print("Step 2: Loading daily_prices.csv...")

    chunks = []
    total_raw = 0
    for chunk in pd.read_csv(DATA_DIR / "daily_prices.csv",
                             chunksize=2_000_000):
        total_raw += len(chunk)
        chunk = chunk[~chunk["ticker"].isin(exclude)]
        chunk = chunk[chunk["adjusted_close"] > 0]
        chunk = chunk.dropna(subset=["close", "adjusted_close", "volume"])
        chunks.append(chunk)

    if not chunks:
        print("ERROR:", "daily_prices.csv contains no valid data rows.")
        sys.exit(1)
    prices = pd.concat(chunks, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    print(f"  Raw rows:      {total_raw:,}")
    print(f"  After cleaning: {len(prices):,} "
             f"({len(prices) / total_raw * 100:.1f}%)")
    print(f"  Tickers:        {prices['ticker'].nunique():,}")
    print(f"  Date range:     {prices['date'].min().date()} to "
             f"{prices['date'].max().date()}")
    return prices


def compute_returns(prices, ff_daily):
    print("Step 3: Computing returns...")

    prices = prices.merge(
        ff_daily[["date", "rf", "mkt_rf"]],
        on="date", how="left"
    )
    n_before = len(prices)
    prices = prices.dropna(subset=["rf"])
    print(f"  Rows with FF match: {len(prices):,} "
             f"(dropped {n_before - len(prices):,})")

    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    prices["ret"] = prices.groupby("ticker")["adjusted_close"].pct_change()
    prices["ret_excess"] = prices["ret"] - prices["rf"]

    # 3-day overlapping log returns (per ticker)
    prices["log_ret"] = np.log1p(prices["ret"])
    prices["r3_stock"] = (
        prices.groupby("ticker")["log_ret"]
        .transform(lambda x: x.rolling(3, min_periods=3).sum())
    )

    # Market: total return and 3-day log return (computed on UNIQUE dates)
    mkt = (prices.drop_duplicates("date")[["date", "mkt_rf", "rf"]]
           .set_index("date").sort_index())
    mkt["mkt_total"] = mkt["mkt_rf"] + mkt["rf"]
    mkt["log_ret_mkt"] = np.log1p(mkt["mkt_total"])
    mkt["r3_market"] = mkt["log_ret_mkt"].rolling(3, min_periods=3).sum()

    # Market rolling 1-year volatility (of excess returns = mkt_rf)
    mkt["sigma_m"] = mkt["mkt_rf"].rolling(VOL_WINDOW,
                                            min_periods=VOL_MIN_OBS).std()

    # Merge market stats back to prices (by date)
    prices = prices.merge(
        mkt[["r3_market", "sigma_m"]].reset_index(),
        on="date", how="left", suffixes=("", "_mkt")
    )

    # Add month column
    prices["month"] = prices["date"].dt.to_period("M")

    print(f"  Returns computed. Valid r3_stock: "
             f"{prices['r3_stock'].notna().sum():,}")

    return prices, mkt


def estimate_betas(prices):
    print("Step 4: Estimating rolling betas...")

    # Exchange map for panel output
    uni_path = DATA_DIR / "universe_combined.csv"
    if uni_path.exists():
        universe = pd.read_csv(uni_path)
        exchange_map = dict(zip(universe["Code"], universe["Exchange"]))
    else:
        exchange_map = {}

    # Group by ticker
    print("  Grouping by ticker...")
    ticker_groups = dict(list(prices.groupby("ticker")))
    total_tickers = len(ticker_groups)
    print(f"  Processing {total_tickers:,} tickers...")

    all_results = []
    processed = 0

    for ticker, tdf in ticker_groups.items():
        processed += 1
        if processed % 2000 == 0:
            print(f"  Progress: {processed:,}/{total_tickers:,} "
                     f"({processed / total_tickers * 100:.0f}%)")

        tdf = tdf.sort_values("date").set_index("date")

        if len(tdf) < VOL_MIN_OBS:
            continue

        # rolling 1-year volatility of daily excess returns
        sigma_i = tdf["ret_excess"].rolling(
            VOL_WINDOW, min_periods=VOL_MIN_OBS
        ).std()
        n_daily = tdf["ret_excess"].rolling(
            VOL_WINDOW, min_periods=VOL_MIN_OBS
        ).count()

        # rolling 5-year correlation of 3-day log returns
        r3_s = tdf["r3_stock"]
        r3_m = tdf["r3_market"]

        rho_im = r3_s.rolling(
            CORR_WINDOW, min_periods=CORR_MIN_OBS
        ).corr(r3_m)
        n_3day = r3_s.rolling(
            CORR_WINDOW, min_periods=CORR_MIN_OBS
        ).count()

        sigma_m = tdf["sigma_m"]
        beta_ts = rho_im * (sigma_i / sigma_m)
        beta = SHRINKAGE_W * beta_ts + (1 - SHRINKAGE_W) * BETA_PRIOR

        tdf = tdf.copy()
        tdf["sigma_i_val"] = sigma_i
        tdf["sigma_m_val"] = sigma_m
        tdf["rho_im_val"] = rho_im
        tdf["beta_ts_val"] = beta_ts
        tdf["beta_val"] = beta
        tdf["n_daily_val"] = n_daily
        tdf["n_3day_val"] = n_3day

        # aggregate to monthly: sample end-of-month values
        month = tdf["month"]

        # Monthly return: first adj_close to last adj_close within month
        monthly_agg = tdf.groupby(month).agg(
            first_adj=("adjusted_close", "first"),
            last_adj=("adjusted_close", "last"),
            median_price=("close", "median"),
            mean_volume=("volume", "mean"),
            rf_sum=("rf", "sum"),
            # End-of-month beta estimates
            beta_ts=("beta_ts_val", "last"),
            beta=("beta_val", "last"),
            sigma_i=("sigma_i_val", "last"),
            sigma_m=("sigma_m_val", "last"),
            rho_im=("rho_im_val", "last"),
            n_daily=("n_daily_val", "last"),
            n_3day=("n_3day_val", "last"),
        )

        # Monthly return
        monthly_agg["monthly_return"] = (
            monthly_agg["last_adj"] / monthly_agg["first_adj"] - 1
        )
        monthly_agg["monthly_excess_return"] = (
            monthly_agg["monthly_return"] - monthly_agg["rf_sum"]
        )

        # Filter: require valid beta and positive starting price
        valid = (
            monthly_agg["beta"].notna()
            & (monthly_agg["first_adj"] > 0)
            & monthly_agg["sigma_i"].notna()
        )
        monthly_agg = monthly_agg[valid]

        if len(monthly_agg) == 0:
            continue

        # Build output rows
        exchange = exchange_map.get(ticker, "")
        for m, row in monthly_agg.iterrows():
            all_results.append({
                "ticker": ticker,
                "month": str(m),
                "beta_ts": round(row["beta_ts"], 6),
                "beta": round(row["beta"], 6),
                "monthly_return": round(row["monthly_return"], 6),
                "monthly_excess_return": round(
                    row["monthly_excess_return"], 6),
                "sigma_i": round(row["sigma_i"], 6),
                "sigma_m": round(row["sigma_m"], 6),
                "rho_im": round(row["rho_im"], 6),
                "n_daily": int(row["n_daily"]),
                "n_3day": int(row["n_3day"]),
                "median_price": round(row["median_price"], 2),
                "mean_volume": round(row["mean_volume"], 0),
                "exchange": exchange,
            })

    panel = pd.DataFrame(all_results)
    print(f"  Beta estimation complete: {len(panel):,} stock-months, "
             f"{panel['ticker'].nunique():,} tickers")
    return panel


def save_panel(panel):
    print("Step 5: Saving panel...")

    path = DATA_DIR / "monthly_panel_unfiltered.csv"
    panel.to_csv(path, index=False, encoding="utf-8")
    print(f"  Saved: {path.name} ({len(panel):,} rows)")

    # Summary
    print(f"\n  Unfiltered panel summary")
    print(f"  Stock-months:   {len(panel):>12,}")
    print(f"  Unique tickers: {panel['ticker'].nunique():>12,}")
    print(f"  Month range:    {panel['month'].min()} to "
             f"{panel['month'].max()}")
    print(f"  Beta — mean:    {panel['beta'].mean():>12.4f}")
    print(f"  Beta — median:  {panel['beta'].median():>12.4f}")
    print(f"  Beta — std:     {panel['beta'].std():>12.4f}")
    print(f"  Beta — [P1,P99]:"
             f" [{panel['beta'].quantile(0.01):.3f},"
             f" {panel['beta'].quantile(0.99):.3f}]")

    mc = panel.groupby("month").size()
    print(f"  Stocks/month — mean:   {mc.mean():>8.0f}")
    print(f"  Stocks/month — median: {mc.median():>8.0f}")
    print(f"  Stocks/month — min:    {mc.min():>8,}")

    print(f"\n  Next: python databank-filter.py")


def main():
    t0 = datetime.now()
    print("Preparing monthly panel...")

    # Check prerequisites
    if not (DATA_DIR / "daily_prices.csv").exists():
        print("ERROR:", "daily_prices.csv not found. Run databank-download.py first.")
        sys.exit(1)
    if not (DATA_DIR / "universe_combined.csv").exists():
        print("ERROR:", "universe_combined.csv not found. "
              "Run databank-download.py first.")
        sys.exit(1)

    ff_daily = download_all_factors()

    exclude = build_exclusion_set()
    prices = load_daily_prices(exclude)

    prices, mkt = compute_returns(prices, ff_daily)

    panel = estimate_betas(prices)

    save_panel(panel)

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\nDone in {elapsed / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
