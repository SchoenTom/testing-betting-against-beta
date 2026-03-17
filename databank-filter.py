# databank-filter.py – apply sample filters to monthly panel
# Tom Schoen, University of Konstanz

import sys
from pathlib import Path
from datetime import datetime
from io import StringIO

for _pkg in ["numpy", "pandas"]:
    try:
        __import__(_pkg)
    except ImportError:
        sys.exit(f"Missing package: {_pkg}. Run: pip install {_pkg}")

import numpy as np
import pandas as pd

DATA_DIR = Path("data")


# exchange classification (CRSP EXCHCD 1/2/3 equivalent)

MAJOR_EXCHANGES = {"NYSE", "NYSE MKT", "NYSE ARCA", "AMEX", "NASDAQ"}


def _log_step(name: str, n_before: int, n_after: int):
    # standardised progress line so filter cascade is easy to scan in terminal
    dropped = n_before - n_after
    pct = (n_after / n_before * 100) if n_before > 0 else 0
    print(f"  {name:<42s} {n_after:>9,}  (−{dropped:>7,}, {pct:>5.1f}% kept)")


def _apply_seasoning(panel: pd.DataFrame, min_months: int) -> pd.DataFrame:
    df = panel.copy()
    df["_m"] = pd.to_datetime(df["month"])
    first = df.groupby("ticker")["_m"].transform("min")
    elapsed = (df["_m"].dt.year - first.dt.year) * 12 + (df["_m"].dt.month - first.dt.month)
    return df[elapsed >= min_months].drop(columns=["_m"])


def apply_filters(panel: pd.DataFrame) -> pd.DataFrame:
    print("Applying sample filters...")
    n0 = len(panel)
    df = panel.copy()

    n = len(df)
    df = df[df["median_price"] >= 5.0]
    _log_step("F1  median_price ≥ $5", n, len(df))

    n = len(df)
    df = df[df["beta"].abs() <= 5.0]
    _log_step("F2  |β| ≤ 5", n, len(df))

    n = len(df)
    df = df[df["exchange"].isin(MAJOR_EXCHANGES)]
    _log_step("F3  Major exchange (NYSE/AMEX/NASDAQ)", n, len(df))

    n = len(df)
    df = df[df["mean_volume"] >= 1_000]
    _log_step("F4  mean_volume ≥ 1,000/day", n, len(df))

    n = len(df)
    df = _apply_seasoning(df, min_months=36)
    _log_step("F5  Seasoning ≥ 36 months", n, len(df))

    print()
    print(f"  Result: {len(df):,} / {n0:,} stock-months ({len(df)/n0*100:.1f}% retained)")
    return df


def _panel_summary(panel: pd.DataFrame, label: str) -> str:
    # build multi-line summary block and return as single string
    buf = StringIO()
    buf.write(f"\n{label}\n")
    buf.write(f"  Stock-months:    {len(panel):>10,}\n")
    buf.write(f"  Unique tickers:  {panel['ticker'].nunique():>10,}\n")

    if len(panel) == 0:
        return buf.getvalue()

    buf.write(f"  Month range:     {panel['month'].min()} → {panel['month'].max()}\n")

    mc = panel.groupby("month").size()
    buf.write(f"\n  Cross-section per month:\n")
    buf.write(f"    Mean:   {mc.mean():>8.0f}    Median: {mc.median():>8.0f}\n")
    buf.write(f"    Min:    {mc.min():>8,}  ({mc.idxmin()})\n")
    buf.write(f"    Max:    {mc.max():>8,}  ({mc.idxmax()})\n")

    b = panel["beta"]
    buf.write(f"\n  Beta:  mean={b.mean():.4f}  std={b.std():.4f}\n")
    buf.write(f"    [P1, P99]: [{b.quantile(0.01):.3f}, {b.quantile(0.99):.3f}]\n")

    p = panel["median_price"]
    buf.write(f"\n  Price:  mean=${p.mean():.2f}  median=${p.median():.2f}\n")

    v = panel["mean_volume"]
    buf.write(f"  Volume: mean={v.mean():>12,.0f}  median={v.median():>12,.0f}\n")

    if "exchange" in panel.columns:
        buf.write(f"\n  Exchange breakdown:\n")
        for exc, cnt in panel["exchange"].value_counts().items():
            buf.write(f"    {exc:<30s} {cnt:>9,}  ({cnt/len(panel)*100:.1f}%)\n")

    buf.write(f"\n  Decile feasibility: min stocks/month = {mc.min()}")
    buf.write(" ✓\n" if mc.min() >= 100 else " — thin\n")
    return buf.getvalue()


def main():
    print("Applying sample filters...")

    path = DATA_DIR / "monthly_panel_unfiltered.csv"
    if not path.exists():
        print("ERROR:",f"{path} not found. Run prepare.py first.")
        sys.exit(1)

    panel = pd.read_csv(path, encoding="utf-8")
    print(f"Loaded {len(panel):,} rows from {path.name}")
    print(f"Exchange values: {sorted(panel['exchange'].unique())}")

    filtered = apply_filters(panel)

    filtered.to_csv(DATA_DIR / "monthly_panel.csv", index=False, encoding="utf-8")
    print(f"Saved: monthly_panel.csv ({len(filtered):,} rows)")

    summary_lines = [
        f"Filter report, {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "Filters:",
        "  F1  median_price ≥ $5           penny stock exclusion",
        "  F2  |β| ≤ 5                     estimation error screen",
        "  F3  exchange ∈ {NYSE,AMEX,NQ}   major US exchanges",
        "  F4  mean_volume ≥ 1,000/day     liquidity floor",
        "  F5  ≥ 36 months panel history   seasoning requirement",
        _panel_summary(panel, "Before filters"),
        _panel_summary(filtered, "After filters"),
        f"\nComparison",
        f"  {'':30s} {'Before':>10s} {'After':>10s}",
        f"  {'Stock-months':30s} {len(panel):>10,} {len(filtered):>10,}",
        f"  {'Tickers':30s} {panel['ticker'].nunique():>10,} {filtered['ticker'].nunique():>10,}",
        f"  {'Beta mean':30s} {panel['beta'].mean():>10.4f} {filtered['beta'].mean():>10.4f}",
        f"  {'Beta std':30s} {panel['beta'].std():>10.4f} {filtered['beta'].std():>10.4f}",
    ]

    if len(filtered) > 0:
        mc_b = panel.groupby("month").size()
        mc_a = filtered.groupby("month").size()
        summary_lines.append(f"  {'Avg stocks/month':30s} {mc_b.mean():>10.0f} {mc_a.mean():>10.0f}")
        summary_lines.append(f"  {'Min stocks/month':30s} {mc_b.min():>10,} {mc_a.min():>10,}")

    summary_text = "\n".join(summary_lines)
    (DATA_DIR / "filter_report.txt").write_text(summary_text, encoding="utf-8")
    print(f"Saved: filter_report.txt")
    print("\n" + summary_text)


if __name__ == "__main__":
    main()