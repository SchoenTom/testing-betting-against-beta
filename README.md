# Testing Betting Against Beta

B.Sc. thesis, University of Konstanz. Replication of the BAB factor (Frazzini & Pedersen, 2014) for U.S. equities, January 2000 to December 2020.

## Research Question

Why is the empirical Security Market Line flatter than the CAPM implies, which market frictions generate this flattening, and can these distortions be systematically exploited through a Betting Against Beta framework?

The analysis tests three propositions derived from the Margin-CAPM:

1. CAPM alphas decline from low-beta to high-beta decile portfolios (flat SML).
2. A beta-neutral BAB factor earns significant risk-adjusted returns across factor models.
3. BAB returns deteriorate when funding liquidity conditions tighten.

## How to Run

Make sure you're inside the project folder, then:

```bash
pip install numpy pandas statsmodels scipy matplotlib openpyxl requests tqdm
python bettingagainstbeta.py
```

That's it. Factor data (Fama-French, momentum, q-factors, TED spread) downloads automatically on first run. Results and figures go into `data/` and `data/figures/`.

If you want to rebuild the panel from scratch (not needed for the analysis):

```bash
python databank-download.py --download   # needs EODHD API key, takes hours
python databank-prepare.py               # ~15 min
python databank-filter.py
```

## Data

| Source | What | Required |
|--------|------|----------|
| EODHD | Daily U.S. equity prices | Yes (via download script) |
| Kenneth French Data Library | FF3, FF5, momentum, risk-free rate | Yes (auto-downloaded) |
| global-q.org | q-factor model | Optional (auto-downloaded) |
| FRED | TED spread | Optional (auto-downloaded) |
| AQR | Published BAB returns (benchmark) | Optional (manual download) |

The AQR file goes into `data/` as `Betting_Against_Beta_Equity_Factors_Monthly.xlsx`. If missing, the benchmark comparison is skipped.

## Project Structure

```
├── bettingagainstbeta.py       Main analysis, all results and figures
├── databank-download.py        EODHD download (optional, resumable)
├── databank-prepare.py         Daily prices → monthly panel + betas
├── databank-filter.py          Sample filters
├── databank-audit.py           Data quality checks
└── data/                       All inputs, outputs, and figures
```

## AI Usage

I used AI tools for coding assistance (syntax, debugging). The research design, statistical specifications, and interpretation of results are entirely my own. No AI tools were used in writing the thesis.
