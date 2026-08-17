# Market Regime Detection and Adaptive Portfolio Construction
### A Detect–Explain–Allocate Framework for NIFTY 50



## What this is

An end-to-end pipeline that:

1. **Detects** market regimes using a 3-state Gaussian HMM on NIFTY 50 daily log-returns
2. **Explains** regime structure via per-regime GARCH(1,1) and curated event mapping
3. **Allocates** capital across NIFTY 50 constituents using regime-conditioned Hierarchical Risk Parity with Ledoit-Wolf shrunk covariance
4. **Backtests** the regime-adaptive strategy against buy-and-hold, equal-weight, and single-regime HRP baselines over Jan 2020 – Dec 2024 (5 years out-of-sample)

## Project layout

```
project/
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # NIFTY 50 index + constituent ingestion from yfinance
│   ├── hmm_regime.py      # 3-state Gaussian HMM fit, Viterbi decode, regime labelling
│   ├── garch_regime.py    # Global GARCH baseline + per-regime GARCH(1,1) fits
│   ├── event_mapping.py   # Curated event list and transition-event alignment
│   ├── allocation.py      # Ledoit-Wolf shrinkage + HRP recursive bisection
│   ├── backtest.py        # Walk-forward engine with regime-trigger + monthly rebalancing
│   └── metrics.py         # CAGR, Sharpe, Sortino, Calmar, max drawdown
├── notebooks/
│   └── endsem_analysis.ipynb   # Main analysis notebook
├── figures/               # Generated plots (created on notebook run)
├── data/                  # Cached parquet files (created on first run)
├── requirements.txt
└── README.md
```

## Running

```bash
pip install -r requirements.txt
cd notebooks
jupyter notebook endsem_analysis.ipynb
```

The notebook is self-contained: every cell can be run top-to-bottom on a fresh environment with internet access (yfinance fetches the data). The first run takes ~2 minutes for data download; subsequent runs use the parquet cache.

## Key design decisions

- **In-sample HMM fit, out-of-sample backtest.** Parameters are estimated on 2004–2019 only. The Viterbi path is decoded on the full sample for descriptive analysis but the test backtest (2020–2024) uses pre-computed regime-specific weights from training, eliminating look-ahead bias.

- **Hybrid rebalancing rule.** Monthly rebalancing combined with a regime-trigger (rebalance when the regime changes and persists for 5+ days). This balances responsiveness against transaction-cost realism.

- **Ledoit-Wolf shrinkage per regime.** Regime sub-samples are smaller than the full sample, making the sample covariance ill-conditioned. Shrinkage is essential before running HRP.

- **HRP over mean-variance.** HRP avoids the expected-return input that makes mean-variance optimization fragile under noisy data. Replicates the Assignment 3 evidence that HRP gives lower turnover and better crisis stability.

## Limitations (also stated in the notebook conclusion)

- Two-stage HMM-then-GARCH estimation ignores decoding uncertainty
- Current NIFTY 50 composition used as proxy across the full 20-year window (survivorship bias)
- Curated event list is non-exhaustive; live GDELT integration would be a future extension
- Single seed (42) used for HMM EM; results should be checked under multiple inits in a production study
