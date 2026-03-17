# databank-audit.py – data quality checks for the equity panel
# Tom Schoen, University of Konstanz

import sys
import warnings
from datetime import datetime
from pathlib import Path
from collections import defaultdict

for _pkg in ["numpy", "pandas"]:
    try:
        __import__(_pkg)
    except ImportError:
        sys.exit(f"Missing: {_pkg}. Install with: pip install {_pkg}")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

DATA_DIR = Path("data")
CHUNK_SIZE = 1_000_000

def iter_prices(cols=None):
    return pd.read_csv(
        DATA_DIR / "daily_prices.csv", usecols=cols, chunksize=CHUNK_SIZE,
        encoding="utf-8",
    )


def load_ticker(ticker):
    frames = []
    for chunk in iter_prices():
        match = chunk[chunk["ticker"] == ticker]
        if len(match) > 0:
            frames.append(match)
    if frames:
        df = pd.concat(frames).sort_values("date").reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame()


def check_persistence(report):
    """Flag tickers with large date gaps in the daily price file."""
    report.append("\nCheck 1: persistence")

    ticker_dates = defaultdict(list)
    for chunk in iter_prices(cols=["ticker", "date"]):
        for ticker, date in zip(chunk["ticker"], chunk["date"]):
            ticker_dates[ticker].append(date)

    all_dates = set()
    for dates in ticker_dates.values():
        all_dates.update(dates)
    all_dates_sorted = sorted(all_dates)

    ticker_stats = {}
    gap_distribution = []
    long_gap_tickers = []

    for ticker, dates in ticker_dates.items():
        dates_sorted = sorted(set(dates))
        n = len(dates_sorted)
        if n < 2:
            continue

        dt_dates = pd.to_datetime(dates_sorted)
        diffs = np.diff(dt_dates).astype("timedelta64[D]").astype(int)
        max_gap = int(diffs.max())
        gaps_over_20 = int((diffs > 20).sum())

        gap_distribution.append(max_gap)
        if gaps_over_20 > 0:
            long_gap_tickers.append(
                (ticker, max_gap, gaps_over_20, n,
                 dates_sorted[0], dates_sorted[-1])
            )

        ticker_stats[ticker] = {
            "min_date": dates_sorted[0],
            "max_date": dates_sorted[-1],
            "n_obs": n,
            "max_gap": max_gap,
            "gaps_over_20": gaps_over_20,
        }

    gap_arr = np.array(gap_distribution)

    report.append(f"\n  Tickers analyzed: {len(ticker_stats):,}")
    report.append(f"  Trading dates:    {len(all_dates_sorted):,}")
    if all_dates_sorted:
        report.append(f"  Date range:       {all_dates_sorted[0]} → "
                      f"{all_dates_sorted[-1]}")
    if len(gap_arr) > 0:
        report.append(f"\n  Max gap (calendar days):")
        report.append(f"    Median: {np.median(gap_arr):>6.0f}    "
                      f"P90: {np.percentile(gap_arr, 90):>6.0f}")
        report.append(f"    P99:    {np.percentile(gap_arr, 99):>6.0f}    "
                      f"Max: {gap_arr.max():>6.0f}")
        report.append(f"\n  Tickers with gaps > 20 days: {(gap_arr > 20).sum():>6,}")
        report.append(f"  Tickers with gaps > 60 days: {(gap_arr > 60).sum():>6,}")

    long_gap_tickers.sort(key=lambda x: x[1], reverse=True)
    if long_gap_tickers:
        report.append(f"\n  Top 10 by longest gap:")
        for t, mg, g20, n, mind, maxd in long_gap_tickers[:10]:
            report.append(f"    {t:12s}: {mg:>5d}d  obs={n:>5d}  "
                          f"{mind}→{maxd}")

    return ticker_stats



def check_price_sanity(report):
    report.append("\nCheck 2: price sanity")

    total = 0
    null_close = null_adj = null_vol = 0
    zero_close = zero_adj = neg_close = neg_adj = 0

    # Collect last row per ticker at each chunk boundary to compute
    # cross-chunk returns correctly.
    last_row = {}
    extreme_count = 0

    for chunk in iter_prices():
        total += len(chunk)
        null_close += chunk["close"].isna().sum()
        null_adj += chunk["adjusted_close"].isna().sum()
        null_vol += chunk["volume"].isna().sum()
        zero_close += (chunk["close"] == 0).sum()
        zero_adj += (chunk["adjusted_close"] == 0).sum()
        neg_close += (chunk["close"] < 0).sum()
        neg_adj += (chunk["adjusted_close"] < 0).sum()

        # Prepend carryover rows from previous chunk for correct returns
        if last_row:
            carryover = pd.DataFrame(last_row.values())
            chunk = pd.concat([carryover, chunk], ignore_index=True)

        chunk = chunk.sort_values(["ticker", "date"])
        chunk["ret"] = chunk.groupby("ticker")["adjusted_close"].pct_change()
        extreme_count += (chunk["ret"].abs() > 1.0).sum()

        # Save last row per ticker for next chunk
        last_row = {}
        for ticker, group in chunk.groupby("ticker"):
            last_row[ticker] = group.iloc[-1].to_dict()

    report.append(f"\n  Observations: {total:,}")
    report.append(f"\n  Nulls:  close={null_close:,}  adj_close={null_adj:,}  "
                  f"volume={null_vol:,}")
    report.append(f"  Zeros:  close={zero_close:,}  adj_close={zero_adj:,}")
    report.append(f"  Negative: close={neg_close:,}  adj_close={neg_adj:,}")
    report.append(f"  Extreme returns (|r| > 100%/day): {extreme_count:,}")

    return {
        "total": total,
        "null_adj": null_adj,
        "zero_adj": zero_adj,
        "extreme": extreme_count,
    }


# splits and dividends

def check_splits(report):
    report.append("\nCheck 3: splits and dividends")

    splits_file = DATA_DIR / "metadata_splits.csv"
    if not splits_file.exists():
        report.append("\n  metadata_splits.csv not found — skipping.")
        return

    splits = pd.read_csv(splits_file, encoding="utf-8")
    report.append(f"\n  Splits: {len(splits):,} events across "
                  f"{splits['ticker'].nunique():,} tickers")

    # Verify 2:1 splits: adjusted_close should be smooth across split date
    ratio_mask = splits["split_ratio"].astype(str).str.contains(
        "2.000000/1.000000", na=False
    )
    sample = splits[ratio_mask]["ticker"].unique()[:10]
    verified = issues = 0

    for ticker in sample:
        prices = load_ticker(ticker)
        if prices.empty:
            continue
        for _, row in splits[splits["ticker"] == ticker].iterrows():
            split_date = pd.to_datetime(row["date"])
            before = prices[prices["date"] < split_date].tail(1)
            after = prices[prices["date"] >= split_date].head(1)
            if len(before) == 0 or len(after) == 0:
                continue
            change = abs(
                after["adjusted_close"].iloc[0]
                / before["adjusted_close"].iloc[-1] - 1
            )
            if change > 0.10:
                issues += 1
            else:
                verified += 1

    report.append(f"  2:1 split check: {verified} smooth, {issues} suspicious")

    div_file = DATA_DIR / "metadata_dividends.csv"
    if div_file.exists():
        divs = pd.read_csv(div_file, encoding="utf-8")
        report.append(f"  Dividends: {len(divs):,} events across "
                      f"{divs['ticker'].nunique():,} tickers")

        # Flag EODHD sentinel values (99999.99999 = data error, not real dividend)
        sentinel_mask = divs["value"] > 99_000
        n_sentinel = sentinel_mask.sum()
        clean = divs[~sentinel_mask]
        report.append(f"  Dividend range (clean): ${clean['value'].min():.4f} — "
                      f"${clean['value'].max():.2f}")
        if n_sentinel > 0:
            bad_tickers = sorted(divs.loc[sentinel_mask, "ticker"].unique())
            report.append(f"  ✗ EODHD data error: {n_sentinel} entries with "
                          f"sentinel value $99,999.99 across {len(bad_tickers)} "
                          f"tickers: {', '.join(bad_tickers)}")
            report.append(f"    These are API placeholder values, not real "
                          f"dividends. Ignored (dividends not used in pipeline).")

def check_coverage(report, ticker_stats):  # universe vs actually downloaded
    report.append("\nCheck 4: coverage")

    universe = pd.read_csv(DATA_DIR / "universe_combined.csv", encoding="utf-8")
    log = pd.read_csv(DATA_DIR / "download_log.csv", encoding="utf-8")

    year_active = defaultdict(int)
    for stats in ticker_stats.values():
        min_yr = int(str(stats["min_date"])[:4])
        max_yr = int(str(stats["max_date"])[:4])
        for yr in range(min_yr, max_yr + 1):
            year_active[yr] += 1

    report.append(f"\n  Tickers active per year:")
    for yr in sorted(year_active.keys()):
        report.append(f"    {yr}: {year_active[yr]:>6,}")

    successful = set(
        log[log["status"] == "ok"]["ticker_api"].str.replace(".US", "",
                                                              regex=False)
    )

    total_uni = len(universe)
    total_covered = len(universe[universe["Code"].isin(successful)])
    pct_total = total_covered / total_uni * 100 if total_uni > 0 else 0

    report.append(f"\n  Universe: {total_covered:,} / {total_uni:,} "
                  f"downloaded ({pct_total:.1f}%)")
    report.append(f"    Active + delisted across "
                  f"{sorted(universe['Exchange'].unique())}")


def check_spot_values(report):
    report.append("\nCheck 5: spot checks")

    checks = [
        ("AAPL", "2020-01-02", 300.35, 3.0, "AAPL 2020-01-02 (pre-split)"),
        ("AAPL", "2024-01-02", 185.64, 2.0, "AAPL 2024-01-02"),
        ("MSFT", "2020-01-02", 160.62, 2.0, "MSFT 2020-01-02"),
        ("JPM",  "2020-01-02", 139.40, 2.0, "JPM 2020-01-02"),
        ("TSLA", "2020-01-02", 430.26, 5.0, "TSLA 2020-01-02 (pre-split)"),
    ]

    for ticker, date, expected, tol, desc in checks:
        prices = load_ticker(ticker)
        if prices.empty:
            report.append(f"  ✗ {desc}: ticker not found")
            continue
        match = prices[prices["date"].dt.strftime("%Y-%m-%d") == date]
        if match.empty:
            target = pd.to_datetime(date)
            nearest = prices.loc[(prices["date"] - target).abs().idxmin()]
            report.append(
                f"  ?     {desc}: date missing, nearest "
                f"{nearest['date'].strftime('%Y-%m-%d')}: "
                f"${nearest['close']:.2f}"
            )
        else:
            actual = match.iloc[0]["close"]
            diff = abs(actual - expected)
            status = "✓" if diff <= tol else "✗"
            report.append(
                f"  {status} {desc}: ${actual:.2f} (expected ~${expected:.2f})"
            )

    aapl = load_ticker("AAPL")
    if not aapl.empty:
        report.append(f"\n  AAPL: {aapl['date'].min().date()} → "
                      f"{aapl['date'].max().date()} ({len(aapl):,} obs)")

    leh = load_ticker("LEH")
    if not leh.empty:
        report.append(f"  LEH:  {leh['date'].min().date()} → "
                      f"{leh['date'].max().date()} ({len(leh):,} obs, "
                      f"last=${leh['close'].iloc[-1]:.2f})")


# monthly panel

def check_panel(report):
    report.append("\nCheck 6: monthly panel")

    path = DATA_DIR / "monthly_panel.csv"
    if not path.exists():
        report.append("\n  monthly_panel.csv not found — "
                      "run prepare.py + filter.py first.")
        return

    panel = pd.read_csv(path, encoding="utf-8")
    report.append(f"\n  Rows: {len(panel):,}   "
                  f"Tickers: {panel['ticker'].nunique():,}")
    report.append(f"  Months: {panel['month'].min()} → {panel['month'].max()}")

    # Nulls
    nulls = panel.isnull().sum()
    null_cols = nulls[nulls > 0]
    if len(null_cols) > 0:
        report.append(f"\n  Null values:")
        for col, cnt in null_cols.items():
            report.append(f"    {col}: {cnt:,}")
    else:
        report.append(f"  Nulls: none")

    # Beta
    b = panel["beta"]
    report.append(f"\n  Beta:  mean={b.mean():.4f}  std={b.std():.4f}")
    report.append(f"    [min={b.min():.3f}, P1={b.quantile(0.01):.3f}, "
                  f"P99={b.quantile(0.99):.3f}, max={b.max():.3f}]")

    infinite_beta = np.isinf(b).sum()
    if infinite_beta > 0:
        report.append(f"  ✗ {infinite_beta} infinite betas")

    # Returns
    r = panel["monthly_return"]
    report.append(f"\n  Monthly return:  mean={r.mean():.4f}  std={r.std():.4f}")
    report.append(f"    [min={r.min():.3f}, P1={r.quantile(0.01):.3f}, "
                  f"P99={r.quantile(0.99):.3f}, max={r.max():.3f}]")

    extreme_ret = (r.abs() > 2.0).sum()
    if extreme_ret > 0:
        report.append(f"  Note: {extreme_ret:,} observations with "
                      f"|return| > 200%")

    # Cross-section
    mc = panel.groupby("month").size()
    report.append(f"\n  Stocks per month:  mean={mc.mean():.0f}  "
                  f"median={mc.median():.0f}  min={mc.min()}")

    if mc.min() < 100:
        report.append(f"  ✗ Months with < 100 stocks — "
                      f"decile sorts may be unreliable")
    else:
        report.append(f"  Decile sorts: feasible "
                      f"(min {mc.min()} stocks/month)")

    # Duplicates
    dupes = panel.duplicated(subset=["ticker", "month"]).sum()
    if dupes > 0:
        report.append(f"  ✗ {dupes:,} duplicate ticker-month pairs")
    else:
        report.append(f"  Duplicates: none")

    # Exchange
    report.append(f"\n  Exchange breakdown:")
    for exc, cnt in panel["exchange"].value_counts().items():
        report.append(f"    {exc:<20s} {cnt:>9,}  "
                      f"({cnt / len(panel) * 100:.1f}%)")

def main():
    print("Running data audit...")

    _needed = ["daily_prices.csv", "universe_combined.csv", "download_log.csv"]
    missing = [f for f in _needed if not (DATA_DIR / f).exists()]
    if missing:
        print(f"ERROR: missing in {DATA_DIR}/: {', '.join(missing)}. "
              "Run databank-download.py first.")
        sys.exit(1)

    report = [
        "Data quality audit",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]

    print("  persistence...")
    ticker_stats = check_persistence(report)

    print("  price sanity...")
    price_stats = check_price_sanity(report)

    print("  splits/dividends...")
    check_splits(report)

    print("  coverage...")
    check_coverage(report, ticker_stats)

    print("  spot checks...")
    check_spot_values(report)

    print("  panel...")
    check_panel(report)

    # Summary
    report.append("\nSummary")
    report.append(f"  Observations:    {price_stats['total']:>12,}")
    report.append(f"  Tickers:         {len(ticker_stats):>12,}")
    report.append(f"  Null adj_close:  {price_stats['null_adj']:>12,}")
    report.append(f"  Zero adj_close:  {price_stats['zero_adj']:>12,}")
    report.append(f"  Extreme returns: {price_stats['extreme']:>12,}")

    issues = []
    if price_stats["null_adj"] > 0:
        issues.append(f"{price_stats['null_adj']} null adjusted_close")
    if price_stats["zero_adj"] > 100:
        issues.append(f"{price_stats['zero_adj']} zero adjusted_close")

    if not issues:
        report.append(f"\n  Data quality: good")
    else:
        report.append(f"\n  Issues:")
        for iss in issues:
            report.append(f"    {iss}")
        report.append(f"  Handled in the preparation phase.")

    full_report = "\n".join(report)
    (DATA_DIR / "audit_report.txt").write_text(full_report, encoding="utf-8")

    print(f"\n{full_report}")
    print(f"\nSaved: {DATA_DIR / 'audit_report.txt'}")


if __name__ == "__main__":
    main()