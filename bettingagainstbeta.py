# bettingagainstbeta.py – BAB replication (Frazzini & Pedersen 2014)
# Tom Schoen, University of Konstanz

import sys
import warnings
from pathlib import Path
from datetime import datetime

_missing = []
for _pkg in ["numpy", "pandas", "statsmodels", "scipy", "matplotlib"]:
    try:
        __import__(_pkg)
    except ImportError:
        _missing.append(_pkg)
if _missing:
    print(f"Missing required packages: {', '.join(_missing)}")
    print(f"  pip install {' '.join(_missing)}")
    sys.exit(1)

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import openpyxl  # needed for AQR .xlsx benchmark
    _HAS_OPENPYXL = True
except ImportError:
    _HAS_OPENPYXL = False

DATA_DIR = Path("data")
FIG_DIR = DATA_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_START = "2000-01"
SAMPLE_END = "2020-12"
NW_LAGS = 6                       # Newey-West lag truncation
WINSORIZE_PCT = 0.5                # 0.5th and 99.5th percentile (Bali, Engle & Murray 2016)

plt.rcParams["font.family"] = "serif"
plt.rcParams["figure.dpi"] = 300
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["savefig.bbox"] = "tight"

COLORS = {
    "main":      "#1a3a4a",
    "accent":    "#4a8db7",
    "negative":  "#c0392b",
    "positive":  "#2e8b57",
    "muted":     "#8b9daa",
    "highlight": "#d4793a",
    "secondary": "#6b5b8a",
    "teal":      "#2a9d8f",
    "warm":      "#c9963b",
}


def _stars(t):
    a = abs(t)
    if a > 2.576: return "***"
    if a > 1.960: return "**"
    if a > 1.645: return "*"
    return ""


def nw_reg(y, X, lags=NW_LAGS):
    return sm.OLS(y, sm.add_constant(X)).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags})


def _save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.pdf", dpi=300)
    plt.close(fig)
    print(f"  Saved: figures/{name}.pdf")


def winsorize_cross_section(values, pct=WINSORIZE_PCT):
    lo = np.nanpercentile(values, pct)
    hi = np.nanpercentile(values, 100 - pct)
    return np.clip(values, lo, hi)


def sortino_ratio(returns, target=0.0):
    excess = returns.mean() - target
    downside = returns[returns < target] - target
    if len(downside) == 0 or downside.std() == 0:
        return np.inf
    downside_std = np.sqrt(np.mean(downside ** 2))
    return excess / downside_std * np.sqrt(12)


def calmar_ratio(returns):
    cum = (1 + returns).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    ann_return = returns.mean() * 12
    return ann_return / abs(max_dd) if max_dd != 0 else np.inf


def load_panel():
    """Main stock panel with betas and returns."""
    path = DATA_DIR / "monthly_panel.csv"
    if not path.exists():
        print(f"ERROR: {path} not found. Run databank-prepare.py and "
              "databank-filter.py first.")
        sys.exit(1)
    panel = pd.read_csv(path, encoding="utf-8")
    panel["month"] = pd.to_datetime(panel["month"]).dt.to_period("M")

    # quick sanity check
    required = {"ticker", "month", "beta", "monthly_return"}
    if not required.issubset(panel.columns):
        print(f"ERROR: panel missing columns: {required - set(panel.columns)}")
        sys.exit(1)

    has_excess = "monthly_excess_return" in panel.columns
    print(f"Panel: {len(panel):,} stock-months, "
             f"{panel['ticker'].nunique():,} tickers, "
             f"{panel['month'].min()} to {panel['month'].max()}")
    print(f"  Columns: monthly_return"
             f"{', monthly_excess_return' if has_excess else ''}")
    return panel


def load_ff3():
    path = DATA_DIR / "ff_factors_monthly.csv"
    if not path.exists():
        sys.exit(f"ERROR: {path} not found. Run databank-prepare.py first.")
    ff = pd.read_csv(path, encoding="utf-8")
    ff["month"] = pd.to_datetime(ff["month"]).dt.to_period("M")
    return ff


def load_ff5():
    path = DATA_DIR / "ff5_factors_monthly.csv"
    if not path.exists():
        sys.exit(f"ERROR: {path} not found. Run databank-prepare.py first.")
    ff5 = pd.read_csv(path, encoding="utf-8")
    ff5["month"] = pd.to_datetime(ff5["month"]).dt.to_period("M")
    return ff5


# Optional factor sets — return None when unavailable
def load_q():
    path = DATA_DIR / "q_factors_monthly.csv"
    if not path.exists():
        print("  q_factors_monthly.csv not found, Eq. 26 will be skipped")
        return None
    q = pd.read_csv(path, encoding="utf-8")
    q["month"] = pd.to_datetime(q["month"]).dt.to_period("M")
    return q


def load_ted():
    path = DATA_DIR / "ted_spread.csv"
    if not path.exists():
        print("  ted_spread.csv not found, P3 will be skipped")
        return None
    ted = pd.read_csv(path, encoding="utf-8")
    ted["month"] = pd.to_datetime(ted["month"]).dt.to_period("M")
    if "ted_lag" not in ted.columns:
        ted = ted.sort_values("month")
        ted["ted_lag"] = ted["ted"].shift(1)
        ted["ted_change"] = ted["ted"].diff()
    return ted.dropna(subset=["ted_lag", "ted_change"])


def load_aqr():
    if not _HAS_OPENPYXL:
        print("  openpyxl not installed, AQR benchmark skipped")
        return None
    path = DATA_DIR / "Betting_Against_Beta_Equity_Factors_Monthly.xlsx"
    if not path.exists():
        print("  AQR BAB file not found, benchmark comparison skipped")
        return None
    try:
        raw = pd.read_excel(path, sheet_name="BAB Factors", header=18)
        col = [c for c in raw.columns if c == "USA"]
        if not col:
            return None
        df = pd.DataFrame({
            "month": pd.to_datetime(raw.iloc[:, 0]).dt.to_period("M"),
            "bab_aqr": pd.to_numeric(raw[col[0]], errors="coerce"),
        }).dropna()
        return df
    except Exception as e:
        print(f"  AQR load failed: {e}")
        return None


def load_momentum():
    path = DATA_DIR / "momentum_monthly.csv"
    if not path.exists():
        print("  momentum_monthly.csv missing, Carhart model skipped")
        return None
    mom = pd.read_csv(path, encoding="utf-8")
    mom["month"] = pd.to_datetime(mom["month"]).dt.to_period("M")
    return mom


def construct_bab(panel, ff3):
    print("Building BAB...")
    has_excess = "monthly_excess_return" in panel.columns
    ret_col = "monthly_excess_return" if has_excess else "monthly_return"
    print(f"  Return column: {ret_col}")

    months = sorted(panel["month"].unique())
    rf_map = ff3.set_index("month")["rf"].to_dict()
    bab_rows, dec_rows = [], []

    for month in months:
        cross = panel[panel["month"] == month].copy()
        if len(cross) < 20:
            continue
        rf = rf_map.get(month, 0.0)

        cross = cross.sort_values("beta").reset_index(drop=True)
        n = len(cross)
        ranks = np.arange(1, n + 1, dtype=float)
        z_bar = ranks.mean()
        k = 2.0 / np.sum(np.abs(ranks - z_bar))  # normalizing constant (Eq. 11)

        betas = cross["beta"].values
        median_beta = np.median(betas)
        low_mask = betas < median_beta
        high_mask = betas >= median_beta

        if has_excess:
            excess = cross["monthly_excess_return"].values
        else:
            excess = cross["monthly_return"].values - rf

        raw_for_winsorize = winsorize_cross_section(
            cross["monthly_return"].values.copy())
        excess = winsorize_cross_section(excess)

        w_low = k * np.maximum(z_bar - ranks, 0)   # Eq. 9
        w_high = k * np.maximum(ranks - z_bar, 0)   # Eq. 10
        wls, whs = w_low[low_mask].sum(), w_high[high_mask].sum()
        if wls <= 0 or whs <= 0:
            continue
        w_low = np.where(low_mask, w_low / wls, 0)
        w_high = np.where(high_mask, w_high / whs, 0)

        beta_L = np.sum(w_low * betas)
        beta_H = np.sum(w_high * betas)
        if beta_L <= 0.01 or beta_H <= 0.01:
            continue

        re_L = np.sum(w_low * excess)
        re_H = np.sum(w_high * excess)
        r_bab = (1.0 / beta_L) * re_L - (1.0 / beta_H) * re_H  # Eq. 13

        raw = raw_for_winsorize
        bab_rows.append({
            "month": month, "bab_return": r_bab,
            "ret_L_excess": re_L, "ret_H_excess": re_H,
            "ret_L_raw": np.sum(w_low * raw),
            "ret_H_raw": np.sum(w_high * raw),
            "beta_L": beta_L, "beta_H": beta_H,
            "n_low": int(low_mask.sum()), "n_high": int(high_mask.sum()),
            "n_total": n, "rf": rf, "median_beta": median_beta,
        })

        # Decile portfolios (using winsorized returns)
        cross["decile"] = pd.qcut(cross["beta"], 10, labels=False,
                                  duplicates="drop") + 1
        cross["_ret_w"] = raw_for_winsorize
        cross["_exc_w"] = excess
        for d in range(1, 11):
            dm = cross["decile"] == d
            if dm.sum() < 3:
                continue
            de = cross.loc[dm, "_exc_w"].mean()
            dec_rows.append({
                "month": month, "decile": d,
                "return": cross.loc[dm, "_ret_w"].mean(),
                "excess_return": de,
                "mean_beta": cross.loc[dm, "beta"].mean(),
                "median_beta": cross.loc[dm, "beta"].median(),
                "n_stocks": int(dm.sum()),
            })

    bab_df = pd.DataFrame(bab_rows)
    dec_df = pd.DataFrame(dec_rows)

    m, s = bab_df["bab_return"].mean(), bab_df["bab_return"].std()
    sr = m / s * np.sqrt(12)
    print(f"  BAB months:    {len(bab_df)}")
    print(f"  BAB mean:      {m:.4f} ({m * 1200:.2f}% p.a.)")
    print(f"  BAB std:       {s:.4f} ({s * np.sqrt(12) * 100:.2f}% p.a.)")
    print(f"  BAB Sharpe:    {sr:.3f}")
    print(f"  Avg beta_L={bab_df['beta_L'].mean():.3f}, "
             f"beta_H={bab_df['beta_H'].mean():.3f}, "
             f"N/mo={bab_df['n_total'].mean():.0f}")
    return bab_df, dec_df


def test_p1(dec_df, ff3, ff5=None, q_fac=None, mom=None):
    print("Running P1 regressions...")
    ff3i = ff3.set_index("month")
    ff5i = ff5.set_index("month") if ff5 is not None else None
    qi = q_fac.set_index("month") if q_fac is not None else None
    mi = mom.set_index("month") if mom is not None else None

    results = []
    for d in range(1, 11):
        dec = dec_df[dec_df["decile"] == d].set_index("month")
        merged = dec.join(ff3i, how="inner")
        if len(merged) < 36:
            continue

        r = {"decile": d}
        er = merged["excess_return"]

        # Mean excess return with NW t-stat
        r["mean_excess_return"] = er.mean()
        nw_mean = sm.OLS(er, np.ones(len(er))).fit(
            cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})
        r["mean_excess_t"] = nw_mean.tvalues.iloc[0]

        # CAPM alpha (Eq. 22)
        mdl = nw_reg(er, merged[["mkt_rf"]])
        r["alpha"] = mdl.params.iloc[0]
        r["alpha_t"] = mdl.tvalues.iloc[0]
        r["alpha_p"] = mdl.pvalues.iloc[0]
        r["capm_beta"] = mdl.params.iloc[1]
        r["r_squared"] = mdl.rsquared
        r["r_squared_adj"] = mdl.rsquared_adj

        # FF3
        mdl3 = nw_reg(er, merged[["mkt_rf", "smb", "hml"]])
        r["alpha_ff3"] = mdl3.params.iloc[0]
        r["alpha_ff3_t"] = mdl3.tvalues.iloc[0]

        # Carhart (FF3 + UMD)
        if mi is not None:
            mc = dec.join(mi[["umd"]], how="inner")
            mc = mc.join(ff3i[["mkt_rf", "smb", "hml"]], how="inner").dropna()
            if len(mc) >= 36:
                mdl_c = nw_reg(mc["excess_return"],
                               mc[["mkt_rf", "smb", "hml", "umd"]])
                r["alpha_carhart"] = mdl_c.params.iloc[0]
                r["alpha_carhart_t"] = mdl_c.tvalues.iloc[0]

        # q-factor
        if qi is not None:
            qcols = [c for c in ["q_mkt", "q_me", "q_ia", "q_roe"]
                     if c in qi.columns]
            if qcols:
                mq = dec.join(qi[qcols], how="inner").dropna()
                if len(mq) >= 36:
                    mdl_q = nw_reg(mq["excess_return"], mq[qcols])
                    r["alpha_q"] = mdl_q.params.iloc[0]
                    r["alpha_q_t"] = mdl_q.tvalues.iloc[0]

        # FF5
        if ff5i is not None:
            ff5cols = ["mkt_rf", "smb", "hml", "rmw", "cma"]
            avail = [c for c in ff5cols if c in ff5i.columns]
            m5 = dec.join(ff5i[avail], how="inner").dropna()
            if len(m5) >= 36:
                mdl5 = nw_reg(m5["excess_return"], m5[avail])
                r["alpha_ff5"] = mdl5.params.iloc[0]
                r["alpha_ff5_t"] = mdl5.tvalues.iloc[0]

        # Additional statistics
        r["mean_assigned_beta"] = merged["mean_beta"].mean()
        r["volatility"] = er.std() * np.sqrt(12) * 100  # ann %
        r["sharpe"] = er.mean() / er.std() * np.sqrt(12) if er.std() > 0 else 0
        r["n_months"] = len(merged)
        r["mean_stocks"] = merged["n_stocks"].mean()

        results.append(r)

    p1 = pd.DataFrame(results)

    if len(p1) >= 10:
        sp = p1.iloc[0]["alpha"] - p1.iloc[-1]["alpha"]
        print(f"  {len(p1)} deciles, D1-D10 alpha spread: "
                 f"{sp:.4f} ({sp * 1200:.2f}% p.a.)")
        sp9 = p1.iloc[0]["alpha"] - p1.iloc[8]["alpha"]
        print(f"  D1-D9 alpha spread:  "
                 f"{sp9:.4f} ({sp9 * 1200:.2f}% p.a.)")
    return p1


def test_p2(bab_df, ff3, ff5, q_fac, mom=None):
    print("Running P2 regressions...")
    bab = bab_df.set_index("month")[["bab_return"]]
    ff3i, ff5i = ff3.set_index("month"), ff5.set_index("month")
    results = []

    def _run(label, cols, df):
        m = bab.join(df[cols], how="inner").dropna()
        if len(m) < 36:
            return
        mdl = nw_reg(m["bab_return"], m[cols])
        r = {"model": label, "alpha": mdl.params.iloc[0],
             "alpha_t": mdl.tvalues.iloc[0], "alpha_p": mdl.pvalues.iloc[0],
             "r_squared": mdl.rsquared, "r_squared_adj": mdl.rsquared_adj,
             "n_obs": int(mdl.nobs)}
        for i, f in enumerate(cols):
            r[f"beta_{f}"] = mdl.params.iloc[i + 1]
            r[f"t_{f}"] = mdl.tvalues.iloc[i + 1]
        results.append(r)

    _run("CAPM (23)", ["mkt_rf"], ff3i)
    _run("FF3 (24)", ["mkt_rf", "smb", "hml"], ff3i)
    if mom is not None:
        mi = mom.set_index("month")
        ff3_umd = ff3i.join(mi[["umd"]], how="inner")
        _run("Carhart (25)", ["mkt_rf", "smb", "hml", "umd"], ff3_umd)
    if q_fac is not None:
        qi = q_fac.set_index("month")
        qc = [c for c in ["q_mkt", "q_me", "q_ia", "q_roe"] if c in qi.columns]
        if qc:
            _run("q-fac (26)", qc, qi)
    _run("FF5 (27)", ["mkt_rf", "smb", "hml", "rmw", "cma"], ff5i)

    p2 = pd.DataFrame(results)
    for _, r in p2.iterrows():
        print(f"  {r['model']:>12}: alpha={r['alpha'] * 1200:+.2f}% p.a."
                 f"{_stars(r['alpha_t'])} (t={r['alpha_t']:.2f})")
    return p2


def test_p3(bab_df, ted_df):
    if ted_df is None:
        return pd.DataFrame()
    print("Running P3 regression...")
    bab = bab_df.set_index("month")[["bab_return"]]
    ted = ted_df.set_index("month")[["ted_change", "ted_lag"]]
    m = bab.join(ted, how="inner").dropna()
    if len(m) < 36:
        print("WARNING:", "  Insufficient TED overlap")
        return pd.DataFrame()

    # main regression (Eq. 28)
    mdl = nw_reg(m["bab_return"], m[["ted_change", "ted_lag"]])
    r = {"model": "TED (Eq.28)",
         "alpha": mdl.params.iloc[0], "alpha_t": mdl.tvalues.iloc[0],
         "beta_ted_change": mdl.params.iloc[1], "t_ted_change": mdl.tvalues.iloc[1],
         "p_ted_change": mdl.pvalues.iloc[1],
         "beta_ted_lag": mdl.params.iloc[2], "t_ted_lag": mdl.tvalues.iloc[2],
         "p_ted_lag": mdl.pvalues.iloc[2],
         "r_squared": mdl.rsquared, "r_squared_adj": mdl.rsquared_adj,
         "n_obs": int(mdl.nobs)}

    # univariate regressions
    mdl_change = nw_reg(m["bab_return"], m[["ted_change"]])
    mdl_level = nw_reg(m["bab_return"], m[["ted_lag"]])

    r["beta_change_univ"] = mdl_change.params.iloc[1]
    r["t_change_univ"] = mdl_change.tvalues.iloc[1]
    r["r2_change_univ"] = mdl_change.rsquared
    r["beta_level_univ"] = mdl_level.params.iloc[1]
    r["t_level_univ"] = mdl_level.tvalues.iloc[1]
    r["r2_level_univ"] = mdl_level.rsquared

    # regressor correlation
    corr_regressors = m["ted_change"].corr(m["ted_lag"])
    r["corr_change_lag"] = corr_regressors

    # print results
    print(f"  r^BAB = a + b1*dTED + b2*TED(t-1) + e")
    print(f"  b1(dTED) = {r['beta_ted_change']:+.4f}"
             f"{_stars(r['t_ted_change'])}  (t={r['t_ted_change']:.2f})")
    print(f"  b2(TED)  = {r['beta_ted_lag']:+.4f}"
             f"{_stars(r['t_ted_lag'])}  (t={r['t_ted_lag']:.2f})")
    print(f"  R2={r['r_squared']:.3f}, N={r['n_obs']}")

    # Univariate robustness
    print(f"  ΔTED only:  b={r['beta_change_univ']:+.4f}"
             f"{_stars(r['t_change_univ'])} (t={r['t_change_univ']:.2f}), "
             f"R2={r['r2_change_univ']:.3f}")
    print(f"  TED level:  b={r['beta_level_univ']:+.4f}"
             f"{_stars(r['t_level_univ'])} (t={r['t_level_univ']:.2f}), "
             f"R2={r['r2_level_univ']:.3f}")
    print(f"  corr(ΔTED, TED_{{t-1}}) = {corr_regressors:.3f}")

    return pd.DataFrame([r])


def compare_aqr(bab_df, aqr_df):
    if aqr_df is None:
        return
    m = bab_df[["month", "bab_return"]].merge(aqr_df, on="month", how="inner")
    corr = m["bab_return"].corr(m["bab_aqr"])
    print(f"AQR benchmark: {len(m)} months, corr={corr:.3f}")
    print(f"  Replication: {m['bab_return'].mean() * 1200:.2f}% ann, "
             f"std={m['bab_return'].std() * np.sqrt(12) * 100:.2f}%")
    print(f"  AQR:         {m['bab_aqr'].mean() * 1200:.2f}% ann, "
             f"std={m['bab_aqr'].std() * np.sqrt(12) * 100:.2f}%")


def fig1_sml(p1, ff3):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))

    b = p1["mean_assigned_beta"].values
    al = p1["alpha"].values * 100  # monthly pct
    er = p1["mean_excess_return"].values * 100
    ds = p1["decile"].values.astype(int)

    # market premium from data
    mkt_prem = ff3["mkt_rf"].mean() * 100  # % per month

    # panel (a)
    a1.scatter(b, er, c=COLORS["main"], s=60, zorder=5,
               edgecolors="white", linewidths=0.6)
    for i, d in enumerate(ds):
        offset = (0, 9) if d != 10 else (8, -5)
        a1.annotate(f"D{d}", (b[i], er[i]), textcoords="offset points",
                    xytext=offset, fontsize=7.5, ha="center", color="#555")

    beta_range = np.linspace(0, max(b) * 1.15, 100)

    # Theoretical SML: passes through (0,0) with slope = E[MKT-RF]
    a1.plot(beta_range, mkt_prem * beta_range, "--",
            color=COLORS["negative"], lw=1.2, alpha=0.8,
            label=f"Theoretical SML (slope={mkt_prem:.3f})")

    # Empirical SML: OLS fit to decile data
    slope, intercept = np.polyfit(b, er, 1)
    a1.plot(beta_range, slope * beta_range + intercept,
            color=COLORS["accent"], lw=1.2, alpha=0.85,
            label=f"Empirical SML (slope={slope:.3f})")

    a1.set_xlabel(r"Assigned $\bar{\beta}$", fontsize=9)
    a1.set_ylabel("Mean excess return (% / month)", fontsize=9)
    a1.set_title("(a) Beta vs. Expected Excess Return", fontsize=10)
    a1.legend(fontsize=7.5, framealpha=0.9, loc="upper left")
    a1.set_xlim(0, max(b) * 1.12)
    a1.set_ylim(0, None)

    # Mark origin where theoretical SML starts
    a1.plot(0, 0, "o", color=COLORS["negative"], ms=5, alpha=0.6, zorder=4)
    a1.annotate("(0, 0)", (0, 0), textcoords="offset points",
                xytext=(6, 6), fontsize=7, color=COLORS["negative"], alpha=0.7)

    # panel (b)
    bar_colors = []
    for i, x in enumerate(al):
        if ds[i] == 10:
            bar_colors.append(COLORS["warm"])  # lottery decile
        elif x > 0:
            bar_colors.append(COLORS["positive"])
        else:
            bar_colors.append(COLORS["negative"])

    a2.bar(range(1, 11), al, color=bar_colors, edgecolor="white",
                  linewidth=0.5, width=0.72, alpha=0.88)
    for i, row in p1.iterrows():
        sig = _stars(row["alpha_t"])
        if sig:
            a2.text(row["decile"], al[i] + 0.025, sig,
                    ha="center", fontsize=7, color="#333", fontweight="bold")

    a2.axhline(y=0, color="#333", linewidth=0.5)
    a2.set_xlabel("Beta Decile", fontsize=9)
    a2.set_ylabel("CAPM alpha (% / month)", fontsize=9)
    a2.set_title("(b) CAPM Alpha by Beta Decile", fontsize=10)
    a2.set_xticks(range(1, 11))
    a2.set_xticklabels([f"D{d}" for d in range(1, 11)], fontsize=8)

    fig.suptitle(f"The Empirical Security Market Line  ({SAMPLE_START} – {SAMPLE_END})",
                 fontsize=11, y=1.0)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    _save(fig, "figure1_sml")


def fig_table3(p1, p2, bab_df, dec_df=None):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis("off")

    # BAB stats for table
    bab_er = bab_df["bab_return"]
    bab_mean = bab_er.mean()
    nw_bab = sm.OLS(bab_er, np.ones(len(bab_er))).fit(
        cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})
    bab_mean_t = nw_bab.tvalues.iloc[0]
    bab_vol = bab_er.std() * np.sqrt(12) * 100
    bab_sr = bab_er.mean() / bab_er.std() * np.sqrt(12) if bab_er.std() > 0 else 0

    # BAB alphas from p2
    bab_stats = {}
    for _, r in p2.iterrows():
        if "CAPM" in r["model"]:
            bab_stats["capm_alpha"] = r["alpha"]
            bab_stats["capm_alpha_t"] = r["alpha_t"]
            bab_stats["capm_beta"] = r.get("beta_mkt_rf", np.nan)
        elif "FF3" in r["model"]:
            bab_stats["ff3_alpha"] = r["alpha"]
            bab_stats["ff3_alpha_t"] = r["alpha_t"]
        elif "Carhart" in r["model"]:
            bab_stats["carhart_alpha"] = r["alpha"]
            bab_stats["carhart_alpha_t"] = r["alpha_t"]
        elif "q-fac" in r["model"]:
            bab_stats["q_alpha"] = r["alpha"]
            bab_stats["q_alpha_t"] = r["alpha_t"]
        elif "FF5" in r["model"]:
            bab_stats["ff5_alpha"] = r["alpha"]
            bab_stats["ff5_alpha_t"] = r["alpha_t"]

    # BAB ex ante beta is ~0 by construction; realized from CAPM regression
    bab_beta_ante = 0.00
    bab_beta_real = bab_stats.get("capm_beta", np.nan)

    # table data
    col_labels = ([f"P{d}" for d in range(1, 11)] + ["BAB"])
    col_labels[0] = "P1\n(low β)"
    col_labels[9] = "P10\n(high β)"

    def _fmt(val, is_t=False):
        if pd.isna(val):
            return ""
        if is_t:
            return f"({val:.2f})"
        return f"{val:.2f}"

    def _bold(val):
        if pd.isna(val):
            return ""
        return f"{val:.2f}"

    rows_data = []
    rows_labels = []

    # Row 1: Excess return
    vals = [p1.iloc[d]["mean_excess_return"] * 100 for d in range(10)]
    vals.append(bab_mean * 100)
    rows_data.append([_bold(v) for v in vals])
    rows_labels.append("Excess return")

    # t-stats
    ts = [p1.iloc[d]["mean_excess_t"] for d in range(10)]
    ts.append(bab_mean_t)
    rows_data.append([_fmt(t, True) for t in ts])
    rows_labels.append("")

    # Row 2: CAPM alpha
    vals = [p1.iloc[d]["alpha"] * 100 for d in range(10)]
    vals.append(bab_stats.get("capm_alpha", np.nan) * 100)
    rows_data.append([_bold(v) for v in vals])
    rows_labels.append("CAPM alpha")

    ts = [p1.iloc[d]["alpha_t"] for d in range(10)]
    ts.append(bab_stats.get("capm_alpha_t", np.nan))
    rows_data.append([_fmt(t, True) for t in ts])
    rows_labels.append("")

    # Row 3: Three-factor alpha
    vals = [p1.iloc[d].get("alpha_ff3", np.nan) * 100
            if pd.notna(p1.iloc[d].get("alpha_ff3", np.nan)) else np.nan
            for d in range(10)]
    vals.append(bab_stats.get("ff3_alpha", np.nan) * 100
                if not pd.isna(bab_stats.get("ff3_alpha", np.nan))
                else np.nan)
    rows_data.append([_bold(v) for v in vals])
    rows_labels.append("Three-factor alpha")

    ts = [p1.iloc[d].get("alpha_ff3_t", np.nan) for d in range(10)]
    ts.append(bab_stats.get("ff3_alpha_t", np.nan))
    rows_data.append([_fmt(t, True) for t in ts])
    rows_labels.append("")

    # Row 3b: Carhart four-factor alpha
    has_carhart = "alpha_carhart" in p1.columns and p1["alpha_carhart"].notna().any()
    if has_carhart:
        vals = [p1.iloc[d].get("alpha_carhart", np.nan) * 100
                if pd.notna(p1.iloc[d].get("alpha_carhart", np.nan)) else np.nan
                for d in range(10)]
        vals.append(bab_stats.get("carhart_alpha", np.nan) * 100
                    if not pd.isna(bab_stats.get("carhart_alpha", np.nan))
                    else np.nan)
        rows_data.append([_bold(v) for v in vals])
        rows_labels.append("Four-factor alpha")

        ts = [p1.iloc[d].get("alpha_carhart_t", np.nan) for d in range(10)]
        ts.append(bab_stats.get("carhart_alpha_t", np.nan))
        rows_data.append([_fmt(t, True) for t in ts])
        rows_labels.append("")

    # Row 4: q-factor alpha (if available)
    has_q = "alpha_q" in p1.columns and p1["alpha_q"].notna().any()
    if has_q:
        vals = [p1.iloc[d].get("alpha_q", np.nan) * 100
                if pd.notna(p1.iloc[d].get("alpha_q", np.nan)) else np.nan
                for d in range(10)]
        vals.append(bab_stats.get("q_alpha", np.nan) * 100
                    if not pd.isna(bab_stats.get("q_alpha", np.nan))
                    else np.nan)
        rows_data.append([_bold(v) for v in vals])
        rows_labels.append("q-factor alpha")

        ts = [p1.iloc[d].get("alpha_q_t", np.nan) for d in range(10)]
        ts.append(bab_stats.get("q_alpha_t", np.nan))
        rows_data.append([_fmt(t, True) for t in ts])
        rows_labels.append("")

    # Row 5: Five-factor alpha
    has_ff5 = "alpha_ff5" in p1.columns and p1["alpha_ff5"].notna().any()
    if has_ff5:
        vals = [p1.iloc[d].get("alpha_ff5", np.nan) * 100
                if pd.notna(p1.iloc[d].get("alpha_ff5", np.nan)) else np.nan
                for d in range(10)]
        vals.append(bab_stats.get("ff5_alpha", np.nan) * 100
                    if not pd.isna(bab_stats.get("ff5_alpha", np.nan))
                    else np.nan)
        rows_data.append([_bold(v) for v in vals])
        rows_labels.append("Five-factor alpha")

        ts = [p1.iloc[d].get("alpha_ff5_t", np.nan) for d in range(10)]
        ts.append(bab_stats.get("ff5_alpha_t", np.nan))
        rows_data.append([_fmt(t, True) for t in ts])
        rows_labels.append("")

    # Row 6: Beta (ex ante)
    vals = [p1.iloc[d]["mean_assigned_beta"] for d in range(10)]
    vals.append(bab_beta_ante)
    rows_data.append([f"{v:.2f}" for v in vals])
    rows_labels.append("Beta (ex ante)")

    # Row 7: Beta (realized)
    vals = [p1.iloc[d]["capm_beta"] for d in range(10)]
    vals.append(bab_beta_real if not pd.isna(bab_beta_real) else 0)
    rows_data.append([f"{v:.2f}" for v in vals])
    rows_labels.append("Beta (realized)")

    # Row 8: Volatility (annualized %)
    vals = [p1.iloc[d]["volatility"] for d in range(10)]
    vals.append(bab_vol)
    rows_data.append([f"{v:.2f}" for v in vals])
    rows_labels.append("Volatility")

    # Row 9: Sharpe ratio
    vals = [p1.iloc[d]["sharpe"] for d in range(10)]
    vals.append(bab_sr)
    rows_data.append([f"{v:.2f}" for v in vals])
    rows_labels.append("Sharpe ratio")

    # Row 10: Sortino ratio
    bab_sortino = sortino_ratio(bab_er)
    dec_sortinos = []
    dec_calmars = []
    for d in range(1, 11):
        if dec_df is not None and "excess_return" in dec_df.columns:
            dr = dec_df[dec_df["decile"] == d]["excess_return"]
            dec_sortinos.append(sortino_ratio(dr))
            dec_calmars.append(calmar_ratio(dr))
        else:
            dec_sortinos.append(np.nan)
            dec_calmars.append(np.nan)
    vals = dec_sortinos + [bab_sortino]
    rows_data.append([f"{v:.2f}" if np.isfinite(v) else "—" for v in vals])
    rows_labels.append("Sortino ratio")

    # Row 11: Calmar ratio
    bab_calmar = calmar_ratio(bab_er)
    vals = dec_calmars + [bab_calmar]
    rows_data.append([f"{v:.2f}" if np.isfinite(v) else "—" for v in vals])
    rows_labels.append("Calmar ratio")

    # render
    tbl = ax.table(
        cellText=rows_data,
        rowLabels=rows_labels,
        colLabels=col_labels,
        cellLoc="center",
        rowLoc="right",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1.0, 1.25)

    n_rows = len(rows_data)
    n_cols = len(col_labels)

    # Style header row
    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_facecolor(COLORS["main"])
        cell.set_text_props(color="white", fontweight="bold", fontsize=7.5)
        cell.set_edgecolor(COLORS["main"])
        cell.set_height(0.07)

    # Style row labels
    for i in range(n_rows):
        cell = tbl[i + 1, -1]  # row label cell
        cell.set_text_props(fontsize=7.5)
        if rows_labels[i]:  # main row (not t-stat)
            cell.set_text_props(fontweight="bold")

    # Style data cells
    for i in range(n_rows):
        is_tstat = rows_labels[i] == ""
        for j in range(n_cols):
            cell = tbl[i + 1, j]
            cell.set_edgecolor("#DEE2E6")
            if is_tstat:
                cell.set_text_props(fontsize=7, color="#666")
                cell.set_facecolor("#FAFAFA")
            else:
                cell.set_text_props(fontsize=7.5, fontweight="bold")
            # Alternate background for main rows
            main_row_idx = sum(1 for r in rows_labels[:i + 1] if r != "")
            if not is_tstat and main_row_idx % 2 == 0:
                cell.set_facecolor("#F5F7FA")
            # Highlight BAB column
            if j == n_cols - 1:
                if not is_tstat:
                    cell.set_facecolor("#EBF5FB")
                else:
                    cell.set_facecolor("#F0F8FF")

    fig.suptitle(
        f"Decile Portfolio Results — U.S. Equities, {SAMPLE_START} – {SAMPLE_END}\n"
        f"(cf. Frazzini & Pedersen, 2014, Table 3)",
        fontsize=11, y=0.97, fontweight="bold"
    )
    fig.text(
        0.5, 0.02,
        "Newey-West HAC t-statistics (L = 6) in parentheses.  "
        f"Returns winsorized at {WINSORIZE_PCT}/{100 - WINSORIZE_PCT} "
        "percentiles.  "
        "Values in % per month except beta, volatility (ann. %), "
        "and ratios (ann.).",
        ha="center", fontsize=7, color="#777", style="italic"
    )

    fig.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.08)
    _save(fig, "table3_fp_style")


def fig2_cumulative(bab, ff3, aqr=None):
    fig, ax = plt.subplots(figsize=(8, 5))

    merged = bab.merge(ff3[["month", "mkt_rf"]], on="month", how="inner")
    if aqr is not None:
        merged = merged.merge(aqr[["month", "bab_aqr"]], on="month", how="inner")
    merged = merged.sort_values("month").reset_index(drop=True)
    dates = merged["month"].apply(lambda p: p.to_timestamp()).values

    cum_bab = (1 + merged["bab_return"]).cumprod()
    cum_mkt = (1 + merged["mkt_rf"]).cumprod()

    ax.plot(dates, cum_bab, color=COLORS["main"], lw=1.5, label="BAB (Replication)")
    if aqr is not None and "bab_aqr" in merged.columns:
        cum_aqr = (1 + merged["bab_aqr"]).cumprod()
        ax.plot(dates, cum_aqr, color=COLORS["accent"], lw=1.2, ls="--",
                alpha=0.85, label="BAB (AQR)")
    ax.plot(dates, cum_mkt, color=COLORS["muted"], lw=1.0, alpha=0.6,
            ls="-.", label="Market (MKT − RF)")
    ax.set_yscale("log")
    ax.set_ylabel("Growth of $1 (log scale)", fontsize=10)
    ax.set_title(f"Cumulative BAB Factor Returns  ({SAMPLE_START} – {SAMPLE_END})",
                 fontsize=11)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)

    ax.yaxis.set_major_locator(mticker.FixedLocator([1, 2, 3, 5, 7, 10]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:.0f}"))

    # NBER recessions (only those within sample)
    for s, e in [("2001-03", "2001-11"),
                 ("2007-12", "2009-06"), ("2020-02", "2020-04")]:
        try:
            ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                       alpha=0.08, color="#333")
        except Exception:
            pass

    # Stats box
    bab_ann = merged["bab_return"].mean() * 1200
    bab_sr = merged["bab_return"].mean() / merged["bab_return"].std() * np.sqrt(12)
    mkt_ann = merged["mkt_rf"].mean() * 1200
    mkt_sr = merged["mkt_rf"].mean() / merged["mkt_rf"].std() * np.sqrt(12)

    stats_text = f"Replication: {bab_ann:+.2f}% p.a.,  SR = {bab_sr:.2f}\n"
    if aqr is not None and "bab_aqr" in merged.columns:
        aqr_ann = merged["bab_aqr"].mean() * 1200
        aqr_sr = merged["bab_aqr"].mean() / merged["bab_aqr"].std() * np.sqrt(12)
        corr_aqr = merged["bab_return"].corr(merged["bab_aqr"])
        stats_text += f"AQR:         {aqr_ann:+.2f}% p.a.,  SR = {aqr_sr:.2f}\n"
        stats_text += f"ρ(Repl., AQR) = {corr_aqr:.3f}\n"
    stats_text += f"Market:      {mkt_ann:+.2f}% p.a.,  SR = {mkt_sr:.2f}"
    ax.text(0.98, 0.05, stats_text, transform=ax.transAxes,
            fontsize=8, va="bottom", ha="right", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec="#ccc", alpha=0.92))

    # End-of-line price tags
    val_bab = cum_bab.iloc[-1]
    ax.annotate(f"${val_bab:.1f}", xy=(dates[-1], val_bab),
                xytext=(6, 8), textcoords="offset points",
                fontsize=8, color=COLORS["main"], fontweight="bold", va="center")
    val_mkt = cum_mkt.iloc[-1]
    ax.annotate(f"${val_mkt:.1f}", xy=(dates[-1], val_mkt),
                xytext=(6, -8), textcoords="offset points",
                fontsize=8, color=COLORS["muted"], fontweight="bold", va="center")
    if aqr is not None and "bab_aqr" in merged.columns:
        val_aqr = cum_aqr.iloc[-1]
        ax.annotate(f"${val_aqr:.1f}", xy=(dates[-1], val_aqr),
                    xytext=(6, -8), textcoords="offset points",
                    fontsize=8, color=COLORS["accent"], fontweight="bold", va="center")

    plt.tight_layout()
    _save(fig, "figure2_cumulative")

def fig3_distribution(bab):
    returns = bab["bab_return"].values * 100  # to percent

    mu, sigma = returns.mean(), returns.std()
    skew_val = stats.skew(returns)
    kurt_val = stats.kurtosis(returns)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.hist(returns, bins=50, density=True, color=COLORS["accent"], alpha=0.6,
            edgecolor="white", linewidth=0.3, label="BAB monthly returns")

    x = np.linspace(returns.min(), returns.max(), 200)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), "--", color=COLORS["negative"],
            lw=1.0, label=f"Normal ({mu:.2f}, {sigma:.2f})")

    ax.text(0.97, 0.95,
            f"Skewness: {skew_val:.2f}\n"
            f"Excess kurtosis: {kurt_val:.2f}\n"
            f"Sharpe (ann.): {mu / sigma * np.sqrt(12):.3f}",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc", alpha=0.9))

    ax.set_xlabel("Monthly return (%)")
    ax.set_ylabel("Density")
    fig.suptitle("Distribution of BAB Monthly Returns", fontsize=11, y=0.98)
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    _save(fig, "figure3_distribution")


def fig4_ted(bab, ted):
    if ted is None:
        return
    fig, ax1 = plt.subplots(figsize=(7.5, 4.3))

    merged = bab.merge(ted[["month", "ted"]], on="month", how="inner")
    merged = merged.sort_values("month").reset_index(drop=True)
    dates = merged["month"].apply(lambda p: p.to_timestamp()).values
    merged["bab_12m"] = (merged["bab_return"]
                         .rolling(12, min_periods=6).mean() * 1200)

    ax1.fill_between(dates, 0, merged["ted"], color=COLORS["negative"],
                     alpha=0.15, label="TED spread (left)")
    ax1.plot(dates, merged["ted"], color=COLORS["negative"], lw=0.7, alpha=0.6)
    ax1.set_ylabel("TED spread (pp)", color=COLORS["negative"])
    ax1.tick_params(axis="y", labelcolor=COLORS["negative"])
    ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx()
    ax2.plot(dates, merged["bab_12m"], color=COLORS["main"], lw=1.0,
             label="BAB 12m rolling (right)")
    ax2.set_ylabel("BAB rolling 12m (% p.a.)", color=COLORS["main"])
    ax2.tick_params(axis="y", labelcolor=COLORS["main"])
    ax2.axhline(y=0, color=COLORS["main"], lw=0.4, ls=":")

    ax1.spines["right"].set_visible(True)
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, fontsize=8, loc="upper right", framealpha=0.9)
    ax1.set_title("Funding Conditions and BAB Performance")
    plt.tight_layout()
    _save(fig, "figure4_ted")



def fig5_beta(panel):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))

    betas = panel["beta"].values
    betas_clean = betas[(betas > -2) & (betas < 5)]

    # Panel (a): Full distribution
    a1.hist(betas_clean, bins=100, density=True, color=COLORS["accent"],
            alpha=0.6, edgecolor="white", linewidth=0.2)
    a1.axvline(x=1.0, color=COLORS["negative"], lw=1.0, ls="--", label=r"$\beta$ = 1")
    med_b = np.median(betas_clean)
    a1.axvline(x=med_b, color=COLORS["positive"], lw=1.0, ls=":",
               label=f"Median = {med_b:.2f}")
    a1.set_xlabel(r"$\beta$ (Vasicek shrinkage)")
    a1.set_ylabel("Density")
    a1.set_title("(a) Cross-Sectional Beta Distribution")
    a1.legend(fontsize=8)
    a1.text(0.97, 0.85,
            f"N = {len(betas_clean):,}\n"
            f"Mean = {betas_clean.mean():.3f}\n"
            f"Std = {betas_clean.std():.3f}",
            transform=a1.transAxes, fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc"))

    # Panel (b): Time series of cross-sectional dispersion
    monthly = panel.groupby("month")["beta"].agg(["mean", "std", "count"])
    monthly.index = monthly.index.to_timestamp()
    a2.fill_between(monthly.index,
                    monthly["mean"] - monthly["std"],
                    monthly["mean"] + monthly["std"],
                    alpha=0.2, color=COLORS["accent"], label=r"$\pm$1 std")
    a2.plot(monthly.index, monthly["mean"], color=COLORS["main"], lw=1.0,
            label=r"Mean $\beta$")
    a2.axhline(y=1.0, color=COLORS["negative"], lw=0.7, ls="--")
    a2.set_ylabel(r"$\beta$")
    a2.set_title("(b) Cross-Sectional Beta Over Time")
    a2.legend(fontsize=8)

    fig.suptitle("Cross-Sectional Beta Distribution", fontsize=12, y=1.02)
    plt.tight_layout()
    _save(fig, "figure5_beta_distribution")


def fig6_risk(bab):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    bab_s = bab.sort_values("month").reset_index(drop=True)
    dates = bab_s["month"].apply(lambda p: p.to_timestamp()).values
    r = bab_s["bab_return"].values

    # (a) Rolling 36-month Sharpe
    window = 36
    rm = pd.Series(r).rolling(window, min_periods=24).mean()
    rs = pd.Series(r).rolling(window, min_periods=24).std()
    sharpe = (rm / rs) * np.sqrt(12)

    a1.plot(dates, sharpe, color=COLORS["main"], lw=1.0)
    a1.axhline(y=0, color="#333", lw=0.5, ls=":")
    a1.fill_between(dates, 0, sharpe,
                    where=sharpe > 0, alpha=0.15, color=COLORS["positive"])
    a1.fill_between(dates, 0, sharpe,
                    where=sharpe < 0, alpha=0.15, color=COLORS["negative"])
    a1.set_ylabel("Annualized Sharpe Ratio")
    a1.set_title("(a) Rolling 36-Month Sharpe Ratio")

    # (b) Drawdown
    cum = (1 + pd.Series(r)).cumprod()
    running_max = cum.cummax()
    dd = (cum / running_max - 1) * 100

    a2.fill_between(dates, dd, 0, color=COLORS["negative"], alpha=0.4)
    a2.plot(dates, dd, color=COLORS["negative"], lw=0.6)
    a2.set_ylabel("Drawdown (%)")
    a2.set_title("(b) BAB Factor Drawdown")
    a2.set_ylim(top=5)

    # Annotate max drawdown
    max_dd = dd.min()
    max_dd_idx = dd.idxmin()
    if max_dd_idx < len(dates):
        a2.annotate(f"Max: {max_dd:.1f}%",
                    xy=(dates[max_dd_idx], max_dd),
                    xytext=(20, -15), textcoords="offset points",
                    fontsize=8, color=COLORS["negative"],
                    arrowprops=dict(arrowstyle="->", color=COLORS["negative"], lw=0.8))

    fig.suptitle("BAB Risk Profile", fontsize=12, y=1.01)
    plt.tight_layout()
    _save(fig, "figure6_risk_profile")


def fig7_factors(p2):
    if p2.empty:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))

    models = p2["model"].values
    alphas_ann = p2["alpha"].values * 1200
    alpha_t = p2["alpha_t"].values
    nm = len(models)

    # (a) Alpha bar chart
    cols = [COLORS["positive"] if a > 0 else COLORS["negative"] for a in alphas_ann]
    a1.barh(range(nm), alphas_ann, color=cols, edgecolor="white",
            height=0.6, alpha=0.85)
    a1.set_yticks(range(nm))
    a1.set_yticklabels(models)
    a1.set_xlabel("Alpha (% p.a.)")
    a1.set_title("(a) BAB Alpha Across Factor Models")
    a1.axvline(x=0, color="#333", lw=0.5)
    for i, (a, t) in enumerate(zip(alphas_ann, alpha_t)):
        sig = _stars(t)
        offset = max(abs(a) * 0.1, 0.3) * np.sign(a)
        a1.text(a + offset, i, f"{a:+.2f}%{sig}",
                va="center", fontsize=8, color="#333")

    # (b) Factor loadings heatmap
    factor_names = ["mkt_rf", "smb", "hml", "umd", "rmw", "cma",
                    "q_mkt", "q_me", "q_ia", "q_roe"]
    available = [f for f in factor_names if f"beta_{f}" in p2.columns]

    if available:
        data = np.full((nm, len(available)), np.nan)
        for i, (_, row) in enumerate(p2.iterrows()):
            for j, f in enumerate(available):
                col = f"beta_{f}"
                if col in row.index and pd.notna(row[col]):
                    data[i, j] = row[col]

        vmax = max(1.5, np.nanmax(np.abs(data)) * 1.1)
        im = a2.imshow(data, aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax)
        a2.set_xticks(range(len(available)))
        labels_map = {"mkt_rf": "MKT", "smb": "SMB", "hml": "HML",
                      "umd": "UMD", "rmw": "RMW", "cma": "CMA",
                      "q_mkt": "q_MKT", "q_me": "q_ME",
                      "q_ia": "q_IA", "q_roe": "q_ROE"}
        a2.set_xticklabels([labels_map.get(f, f) for f in available],
                           rotation=45, ha="right", fontsize=8)
        a2.set_yticks(range(nm))
        a2.set_yticklabels(models)
        a2.set_title("(b) Factor Loadings")

        for i in range(nm):
            for j in range(len(available)):
                v = data[i, j]
                if not np.isnan(v):
                    t_col = f"t_{available[j]}"
                    tv = p2.iloc[i].get(t_col, 0)
                    sig = _stars(tv) if pd.notna(tv) else ""
                    color = "white" if abs(v) > vmax * 0.5 else "black"
                    a2.text(j, i, f"{v:.2f}{sig}", ha="center",
                            va="center", fontsize=7, color=color)

        plt.colorbar(im, ax=a2, shrink=0.8, label="Loading")

    fig.suptitle("Factor Model Comparison", fontsize=12, y=1.02)
    plt.tight_layout()
    _save(fig, "figure7_factor_models")


def fig8_transitions(panel):
    # compute transition matrix first, then plot
    p = panel[["ticker", "month", "beta"]].copy()
    p = p.sort_values(["ticker", "month"])

    fig, ax = plt.subplots(figsize=(5.8, 5.2))

    # Quintile assignment per month
    p["quintile"] = p.groupby("month")["beta"].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop") + 1
    )
    p["q_next"] = p.groupby("ticker")["quintile"].shift(-1)
    p = p.dropna(subset=["q_next"])
    p["q_next"] = p["q_next"].astype(int)

    trans = pd.crosstab(p["quintile"], p["q_next"], normalize="index")

    im = ax.imshow(trans.values * 100, cmap="YlOrRd", vmin=0, vmax=100)
    for i in range(5):
        for j in range(5):
            v = trans.values[i, j] * 100
            color = "white" if v > 50 else "black"
            ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                    fontsize=10, fontweight="bold" if i == j else "normal",
                    color=color)

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels([f"Q{q}" for q in range(1, 6)])
    ax.set_yticklabels([f"Q{q}" for q in range(1, 6)])
    ax.set_xlabel(r"$\beta$ Quintile (month $t$+1)")
    ax.set_ylabel(r"$\beta$ Quintile (month $t$)")
    ax.set_title("Beta Quintile Transition Probabilities")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Probability (%)")

    plt.tight_layout()
    _save(fig, "figure8_beta_transitions")



def fig9_rolling_alpha(bab, ff3):
    fig, ax = plt.subplots(figsize=(7.2, 4.5))

    merged = bab[["month", "bab_return"]].merge(
        ff3[["month", "mkt_rf"]], on="month", how="inner"
    ).sort_values("month")

    window = 60
    alphas, alpha_se, dates_out = [], [], []

    for end in range(window, len(merged)):
        chunk = merged.iloc[end - window:end]
        y = chunk["bab_return"].values
        X = sm.add_constant(chunk["mkt_rf"].values)
        try:
            model = sm.OLS(y, X).fit(cov_type="HAC",
                                     cov_kwds={"maxlags": NW_LAGS})
            alphas.append(model.params[0] * 1200)
            alpha_se.append(model.bse[0] * 1200)
            dates_out.append(chunk["month"].iloc[-1].to_timestamp())
        except Exception:
            continue

    alphas = np.array(alphas)
    alpha_se = np.array(alpha_se)
    dates_out = np.array(dates_out)

    ax.plot(dates_out, alphas, color=COLORS["main"], lw=1.0,
            label=r"Rolling 60m $\alpha$ (% p.a.)")
    ax.fill_between(dates_out,
                    alphas - 1.96 * alpha_se,
                    alphas + 1.96 * alpha_se,
                    alpha=0.15, color=COLORS["accent"], label="95% CI")
    ax.axhline(y=0, color="#333", lw=0.5, ls=":")
    ax.set_ylabel(r"CAPM $\alpha$ (% p.a.)")
    ax.set_title("Time-Varying BAB Alpha (60-Month Rolling)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, "figure9_rolling_alpha")


def fig10_decomposition(bab, ff3):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    bab_s = bab.sort_values("month").reset_index(drop=True)
    dates = bab_s["month"].apply(lambda p: p.to_timestamp()).values

    # Levered long and de-levered short legs
    long_leg = (1.0 / bab_s["beta_L"]) * bab_s["ret_L_excess"]
    short_leg = (1.0 / bab_s["beta_H"]) * bab_s["ret_H_excess"]

    # (a) Cumulative returns
    cum_long = (1 + long_leg).cumprod()
    cum_short = (1 + short_leg).cumprod()
    cum_bab = (1 + bab_s["bab_return"]).cumprod()

    a1.plot(dates, cum_long, color=COLORS["positive"], lw=1.0,
            label=r"Long: $(1/\beta_L) \times r^e_L$")
    a1.plot(dates, cum_short, color=COLORS["negative"], lw=1.0,
            label=r"Short: $(1/\beta_H) \times r^e_H$")
    a1.plot(dates, cum_bab, color=COLORS["main"], lw=1.2, ls="--",
            label="BAB = Long - Short")
    a1.set_yscale("log")
    a1.set_ylabel("Cumulative return (log)")
    a1.set_title("(a) Cumulative Return Decomposition")
    a1.legend(fontsize=8, loc="upper left")
    a1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:.1f}"))

    # (b) Leverage ratios
    a2.plot(dates, 1.0 / bab_s["beta_L"], color=COLORS["positive"],
            lw=0.8, label=r"$1/\beta_L$ (long leverage)")
    a2.plot(dates, 1.0 / bab_s["beta_H"], color=COLORS["negative"],
            lw=0.8, label=r"$1/\beta_H$ (short leverage)")
    a2.axhline(y=1, color="#333", lw=0.5, ls=":")
    a2.set_ylabel("Leverage factor")
    a2.set_title("(b) Leverage Ratios Over Time")
    a2.legend(fontsize=8)

    fig.suptitle("BAB Long-Short Decomposition",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    _save(fig, "figure10_decomposition")


def fig11_market(bab, aqr, aqr_path=None):
    if aqr is None or not _HAS_OPENPYXL:
        return

    mkt_df = None
    if aqr_path is None:
        aqr_path = DATA_DIR / "Betting_Against_Beta_Equity_Factors_Monthly.xlsx"
    try:
        mkt_raw = pd.read_excel(aqr_path, sheet_name="MKT", header=18)
        mkt_df = pd.DataFrame({
            "month": pd.to_datetime(mkt_raw["DATE"]).dt.to_period("M"),
            "mkt_aqr": pd.to_numeric(mkt_raw["USA"], errors="coerce"),
        }).dropna()
    except Exception as e:
        print("WARNING:", f"  Could not load AQR MKT sheet: {e}")
        return

    merged = bab[["month", "bab_return", "rf"]].merge(aqr, on="month", how="inner")
    merged = merged.merge(mkt_df, on="month", how="inner")
    merged["mkt_total"] = merged["mkt_aqr"] + merged["rf"]

    m2 = merged.sort_values("month").reset_index(drop=True)
    if len(m2) < 12:
        return
    dates = m2["month"].apply(lambda p: p.to_timestamp()).values

    fig, ax = plt.subplots(figsize=(8, 5.2))
    cum_ours = (1 + m2["bab_return"]).cumprod()
    cum_aqr = (1 + m2["bab_aqr"]).cumprod()
    cum_mkt = (1 + m2["mkt_total"]).cumprod()

    ax.plot(dates, cum_ours, color=COLORS["main"], lw=1.3, label="BAB (Replication)")
    ax.plot(dates, cum_aqr, color=COLORS["negative"], lw=1.1, ls="--", label="BAB (AQR)")
    ax.plot(dates, cum_mkt, color=COLORS["accent"], lw=1.0, ls="-.", alpha=0.8,
            label="U.S. Market")
    ax.set_yscale("log")

    ax.yaxis.set_major_locator(mticker.FixedLocator([1, 2, 3, 5, 7]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:.0f}"))

    ax.set_ylabel("Growth of $1 (log scale)")
    ax.set_title(f"BAB vs. Market Performance\n"
                 f"{SAMPLE_START} – {SAMPLE_END}", fontsize=11)

    our_ann = m2["bab_return"].mean() * 1200
    aqr_ann = m2["bab_aqr"].mean() * 1200
    mkt_ann = m2["mkt_total"].mean() * 1200
    our_sr = m2["bab_return"].mean() / m2["bab_return"].std() * np.sqrt(12)
    aqr_sr = m2["bab_aqr"].mean() / m2["bab_aqr"].std() * np.sqrt(12)
    mkt_sr = m2["mkt_total"].mean() / m2["mkt_total"].std() * np.sqrt(12)

    ax.text(0.97, 0.05,
            f"Replication:  {our_ann:+.2f}% p.a.,  SR = {our_sr:.2f}\n"
            f"AQR:          {aqr_ann:+.2f}% p.a.,  SR = {aqr_sr:.2f}\n"
            f"U.S. Market:  {mkt_ann:+.2f}% p.a.,  SR = {mkt_sr:.2f}",
            transform=ax.transAxes, fontsize=8.5, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc", alpha=0.9),
            fontfamily="monospace")

    for s, e in [("2001-03", "2001-11"), ("2007-12", "2009-06"),
                 ("2020-02", "2020-04")]:
        try:
            ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.08, color="#333")
        except Exception:
            pass

    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    _save(fig, "figure11_bab_vs_market")



def fig12_alpha_heatmap(p1):
    fig, ax = plt.subplots(figsize=(10, 4.5))

    models = []
    model_keys = []
    if "alpha" in p1.columns:
        models.append("CAPM")
        model_keys.append(("alpha", "alpha_t"))
    if "alpha_ff3" in p1.columns and p1["alpha_ff3"].notna().any():
        models.append("FF3")
        model_keys.append(("alpha_ff3", "alpha_ff3_t"))
    if "alpha_carhart" in p1.columns and p1["alpha_carhart"].notna().any():
        models.append("Carhart")
        model_keys.append(("alpha_carhart", "alpha_carhart_t"))
    if "alpha_q" in p1.columns and p1["alpha_q"].notna().any():
        models.append("q-factor")
        model_keys.append(("alpha_q", "alpha_q_t"))
    if "alpha_ff5" in p1.columns and p1["alpha_ff5"].notna().any():
        models.append("FF5")
        model_keys.append(("alpha_ff5", "alpha_ff5_t"))

    if not models:
        plt.close(fig)
        return

    data = np.full((len(models), 10), np.nan)
    t_data = np.full((len(models), 10), np.nan)
    for i, (a_col, t_col) in enumerate(model_keys):
        for d in range(10):
            if d < len(p1):
                val = p1.iloc[d].get(a_col, np.nan)
                if pd.notna(val):
                    data[i, d] = val * 100  # monthly %
                tval = p1.iloc[d].get(t_col, np.nan)
                if pd.notna(tval):
                    t_data[i, d] = tval

    vmax = max(0.5, np.nanmax(np.abs(data)) * 1.05)
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn",
                   vmin=-vmax, vmax=vmax)

    # Annotate with alpha value and significance
    for i in range(len(models)):
        for j in range(10):
            v = data[i, j]
            t = t_data[i, j]
            if np.isnan(v):
                continue
            sig = _stars(t) if not np.isnan(t) else ""
            color = "white" if abs(v) > vmax * 0.6 else "black"
            weight = "bold" if sig else "normal"
            ax.text(j, i, f"{v:.2f}{sig}", ha="center", va="center",
                    fontsize=7.5, color=color, fontweight=weight)

    ax.set_xticks(range(10))
    ax.set_xticklabels([f"D{d}" for d in range(1, 11)])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel("Beta Decile")
    ax.set_title(f"Decile Alphas Across Factor Models  ({SAMPLE_START} – {SAMPLE_END})\n"
                 f"(% per month, Newey-West t-statistics)", fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Alpha (% / month)")

    plt.tight_layout()
    _save(fig, "figure12_alpha_heatmap")


def fig13_subperiod(bab):
    periods = [
        ("Pre-Crisis", "2000-01", "2006-12"),
        ("Financial Crisis", "2007-01", "2009-06"),
        ("Recovery", "2009-07", "2015-12"),
        ("Late Sample", "2016-01", "2020-12"),
    ]

    results = []
    for label, start, end in periods:
        mask = (bab["month"] >= start) & (bab["month"] <= end)
        sub = bab.loc[mask, "bab_return"]
        if len(sub) < 6:
            continue
        ann_ret = sub.mean() * 12 * 100
        ann_vol = sub.std() * np.sqrt(12) * 100
        sr = sub.mean() / sub.std() * np.sqrt(12) if sub.std() > 0 else 0
        cum = (1 + sub).cumprod()
        max_dd = (cum / cum.cummax() - 1).min() * 100
        results.append({
            "period": label, "n_months": len(sub),
            "ann_return": ann_ret, "ann_vol": ann_vol,
            "sharpe": sr, "max_dd": max_dd,
            "sortino": sortino_ratio(sub),
            "calmar": calmar_ratio(sub),
        })

    if not results:
        return
    df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    x = range(len(df))
    labels = [f"{r['period']}\n({r['n_months']}m)" for _, r in df.iterrows()]

    # (a) Annualized return
    cols = [COLORS["positive"] if v > 0 else COLORS["negative"] for v in df["ann_return"]]
    axes[0].bar(x, df["ann_return"], color=cols, edgecolor="white",
                width=0.65, alpha=0.85)
    axes[0].axhline(y=0, color="#333", lw=0.5, ls=":")
    for i, v in enumerate(df["ann_return"]):
        axes[0].text(i, v + (0.5 if v > 0 else -1.5), f"{v:+.1f}%",
                     ha="center", fontsize=8, fontweight="bold")
    axes[0].set_ylabel("% per annum")
    axes[0].set_title("(a) Annualized Return")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=7.5)

    # (b) Sharpe ratio
    cols = [COLORS["accent"] if v > 0 else COLORS["negative"] for v in df["sharpe"]]
    axes[1].bar(x, df["sharpe"], color=cols, edgecolor="white",
                width=0.65, alpha=0.85)
    axes[1].axhline(y=0, color="#333", lw=0.5, ls=":")
    for i, v in enumerate(df["sharpe"]):
        axes[1].text(i, v + (0.05 if v > 0 else -0.1), f"{v:.2f}",
                     ha="center", fontsize=8, fontweight="bold")
    axes[1].set_ylabel("Annualized Sharpe")
    axes[1].set_title("(b) Sharpe Ratio")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=7.5)

    # (c) Maximum drawdown
    axes[2].bar(x, df["max_dd"], color=COLORS["negative"], edgecolor="white",
                width=0.65, alpha=0.7)
    for i, v in enumerate(df["max_dd"]):
        axes[2].text(i, v - 1.0, f"{v:.1f}%",
                     ha="center", fontsize=8, fontweight="bold", color="white")
    axes[2].set_ylabel("Max Drawdown (%)")
    axes[2].set_title("(c) Maximum Drawdown")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, fontsize=7.5)
    axes[2].set_ylim(top=2)

    fig.suptitle(f"BAB Performance by Sub-Period  ({SAMPLE_START} – {SAMPLE_END})",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    _save(fig, "figure13_subperiod")


def write_summary(bab, p1, p2, p3):
    lines = []
    lines.append("Results Summary")
    lines.append(f"Sample: {SAMPLE_START} to {SAMPLE_END}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    m = bab["bab_return"].mean()
    s = bab["bab_return"].std()

    lines.append("\nBAB factor statistics")
    lines.append(f"  Months:             {len(bab)}")
    lines.append(f"  Period:             {bab['month'].min()} to "
                 f"{bab['month'].max()}")
    lines.append(f"  Mean (monthly):     {m:+.4f}")
    lines.append(f"  Mean (annual):      {m * 1200:+.2f}%")
    lines.append(f"  Std (monthly):      {s:.4f}")
    lines.append(f"  Std (annual):       {s * np.sqrt(12) * 100:.2f}%")
    sr = m / s * np.sqrt(12)
    so = sortino_ratio(bab["bab_return"])
    ca = calmar_ratio(bab["bab_return"])
    cum = (1 + bab["bab_return"]).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()

    lines.append(f"  Sharpe (ann.):      {sr:.3f}")
    lines.append(f"  Sortino (ann.):     {so:.3f}")
    lines.append(f"  Calmar:             {ca:.3f}")
    lines.append(f"  Max drawdown:       {max_dd * 100:.2f}%")
    lines.append(f"  Skewness:           {bab['bab_return'].skew():.3f}")
    lines.append(f"  Kurtosis:           {bab['bab_return'].kurtosis():.3f}")
    lines.append(f"  Min:                {bab['bab_return'].min():.4f}")
    lines.append(f"  Max:                {bab['bab_return'].max():.4f}")
    lines.append(f"  % positive months:  {(bab['bab_return'] > 0).mean() * 100:.1f}%")
    lines.append(f"  Avg beta_L:         {bab['beta_L'].mean():.3f}")
    lines.append(f"  Avg beta_H:         {bab['beta_H'].mean():.3f}")
    lines.append(f"  Avg stocks/month:   {bab['n_total'].mean():.0f}")

    if not p1.empty:
        lines.append(f"\nProposition 1: flat SML (Eq. 22)")
        lines.append(f"  {'D':>3} {'alpha(mo)':>10} {'t(a)':>7} "
                     f"{'beta_CAPM':>10} {'E[re]':>8} {'R2':>5} {'R2adj':>6}")
        lines.append("")
        for _, r in p1.iterrows():
            sig = _stars(r["alpha_t"])
            lines.append(
                f"  D{r['decile']:>1.0f}  {r['alpha']:>9.4f}{sig:<3s}"
                f" {r['alpha_t']:>6.2f}  {r['capm_beta']:>9.3f}"
                f"  {r['mean_excess_return']:>7.4f}"
                f"  {r['r_squared']:>4.3f} {r['r_squared_adj']:>5.3f}")
        if len(p1) >= 10:
            sp = p1.iloc[0]["alpha"] - p1.iloc[-1]["alpha"]
            lines.append(f"\n  D1-D10 spread: {sp:.4f} ({sp * 1200:.2f}% p.a.)")

    if not p2.empty:
        lines.append(f"\nProposition 2: BAB regressions (Eq. 23-27)")
        for _, r in p2.iterrows():
            sig = _stars(r["alpha_t"])
            lines.append(f"\n  {r['model']}:")
            lines.append(f"    alpha = {r['alpha']:.4f}{sig}  "
                         f"(t = {r['alpha_t']:.2f})")
            lines.append(f"    alpha annualized = {r['alpha'] * 1200:+.2f}%")
            lines.append(f"    R2 = {r['r_squared']:.3f}, "
                         f"R2adj = {r['r_squared_adj']:.3f}, "
                         f"N = {r['n_obs']}")

    if not p3.empty:
        lines.append(f"\nProposition 3: funding conditions (Eq. 28)")
        r = p3.iloc[0]
        lines.append(f"  r^BAB = a + b1*dTED + b2*TED(t-1) + e")
        lines.append(f"  b1(dTED) = {r['beta_ted_change']:+.4f}"
                     f"{_stars(r['t_ted_change'])}  "
                     f"(t = {r['t_ted_change']:.2f})")
        lines.append(f"  b2(TED)  = {r['beta_ted_lag']:+.4f}"
                     f"{_stars(r['t_ted_lag'])}  "
                     f"(t = {r['t_ted_lag']:.2f})")
        lines.append(f"  R2 = {r['r_squared']:.3f}, N = {r['n_obs']}")

        # Univariate robustness (if available)
        if "beta_change_univ" in r:
            lines.append(f"\n  Univariate robustness:")
            lines.append(f"    ΔTED only:  b={r['beta_change_univ']:+.4f}"
                         f"{_stars(r['t_change_univ'])}  "
                         f"(t = {r['t_change_univ']:.2f}), "
                         f"R2 = {r['r2_change_univ']:.3f}")
            lines.append(f"    TED level:  b={r['beta_level_univ']:+.4f}"
                         f"{_stars(r['t_level_univ'])}  "
                         f"(t = {r['t_level_univ']:.2f}), "
                         f"R2 = {r['r2_level_univ']:.3f}")
        if "corr_change_lag" in r:
            lines.append(f"    corr(ΔTED, TED_{{t-1}}) = "
                         f"{r['corr_change_lag']:.3f}")


    lines.append("\nNewey-West HAC standard errors (L=6) throughout.")
    lines.append("* p<0.10  ** p<0.05  *** p<0.01")

    text = "\n".join(lines)
    (DATA_DIR / "results_summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)



def main():
    t0 = datetime.now()
    print(f"BAB replication – {SAMPLE_START} to {SAMPLE_END}")

    # -- Load data --
    panel = load_panel()
    ff3 = load_ff3()
    ff5 = load_ff5()
    q_fac = load_q()
    ted = load_ted()
    aqr = load_aqr()
    mom = load_momentum()

    # -- Filter to sample period --
    print(f"Filtering to: {SAMPLE_START} – {SAMPLE_END}")
    panel = panel[(panel["month"] >= SAMPLE_START) & (panel["month"] <= SAMPLE_END)]
    ff3 = ff3[(ff3["month"] >= SAMPLE_START) & (ff3["month"] <= SAMPLE_END)]
    ff5 = ff5[(ff5["month"] >= SAMPLE_START) & (ff5["month"] <= SAMPLE_END)]
    if q_fac is not None:
        q_fac = q_fac[(q_fac["month"] >= SAMPLE_START) & (q_fac["month"] <= SAMPLE_END)]
    if ted is not None:
        ted = ted[(ted["month"] >= SAMPLE_START) & (ted["month"] <= SAMPLE_END)]
    if aqr is not None:
        aqr = aqr[(aqr["month"] >= SAMPLE_START) & (aqr["month"] <= SAMPLE_END)]
    if mom is not None:
        mom = mom[(mom["month"] >= SAMPLE_START) & (mom["month"] <= SAMPLE_END)]
    print(f"  Panel: {len(panel):,} stock-months, "
             f"{panel['month'].min()} to {panel['month'].max()}")

    # -- BAB factor --
    bab, deciles = construct_bab(panel, ff3)

    bab_save = bab.copy()
    bab_save["month"] = bab_save["month"].astype(str)
    bab_save.to_csv(DATA_DIR / "bab_returns.csv", index=False)

    dec_save = deciles.copy()
    dec_save["month"] = dec_save["month"].astype(str)
    dec_save.to_csv(DATA_DIR / "decile_portfolios.csv", index=False)

    # -- AQR benchmark --
    if aqr is not None:
        compare_aqr(bab, aqr)

    # -- P1 --
    prop1 = test_p1(deciles, ff3, ff5, q_fac, mom)
    prop1.to_csv(DATA_DIR / "results_proposition1.csv", index=False)

    # -- P2 --
    prop2 = test_p2(bab, ff3, ff5, q_fac, mom)
    prop2.to_csv(DATA_DIR / "results_proposition2.csv", index=False)

    # -- P3 --
    prop3 = test_p3(bab, ted)
    if not prop3.empty:
        prop3.to_csv(DATA_DIR / "results_proposition3.csv", index=False)

    # -- Summary --
    write_summary(bab, prop1, prop2, prop3)

    # -- Figures --
    print("\nGenerating figures...")
    fig1_sml(prop1, ff3)
    fig_table3(prop1, prop2, bab, deciles)
    fig2_cumulative(bab, ff3, aqr)
    fig3_distribution(bab)
    fig4_ted(bab, ted)
    fig5_beta(panel)
    fig6_risk(bab)
    fig7_factors(prop2)
    fig8_transitions(panel)
    fig9_rolling_alpha(bab, ff3)
    fig10_decomposition(bab, ff3)
    aqr_xlsx = DATA_DIR / "Betting_Against_Beta_Equity_Factors_Monthly.xlsx"
    fig11_market(bab, aqr, aqr_path=aqr_xlsx)
    fig12_alpha_heatmap(prop1)
    fig13_subperiod(bab)
    print(f"All figures saved to {FIG_DIR}/")

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\nDone in {elapsed:.0f}s.")


if __name__ == "__main__":
    main()