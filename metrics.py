"""
metrics.py
----------
Standard portfolio performance metrics: CAGR, Sharpe, Sortino, Calmar,
maximum drawdown, and drawdown-period summaries.

All metrics assume daily frequency and 252 trading days per year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def returns_from_wealth(wealth: pd.Series) -> pd.Series:
    """Daily simple returns from a wealth path."""
    return wealth.pct_change().dropna()


def cagr(wealth: pd.Series) -> float:
    """Compound annual growth rate."""
    if len(wealth) < 2:
        return np.nan
    n_years = (wealth.index[-1] - wealth.index[0]).days / 365.25
    if n_years <= 0:
        return np.nan
    return float((wealth.iloc[-1] / wealth.iloc[0]) ** (1.0 / n_years) - 1.0)


def annualized_vol(wealth: pd.Series) -> float:
    """Annualized volatility from a wealth path."""
    r = returns_from_wealth(wealth)
    if r.empty:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(wealth: pd.Series, rf_annual: float = 0.065) -> float:
    """Annualized Sharpe ratio (default risk-free rate: 6.5% for India)."""
    r = returns_from_wealth(wealth)
    if r.empty or r.std() == 0:
        return np.nan
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    excess = r - rf_daily
    return float(excess.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sortino_ratio(wealth: pd.Series, rf_annual: float = 0.065) -> float:
    """Sortino: like Sharpe but using downside deviation."""
    r = returns_from_wealth(wealth)
    if r.empty:
        return np.nan
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    excess = r - rf_daily
    downside = r[r < 0]
    if downside.empty or downside.std() == 0:
        return np.nan
    return float(excess.mean() / downside.std(ddof=1) * np.sqrt(TRADING_DAYS))


def max_drawdown(wealth: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (negative number)."""
    if wealth.empty:
        return np.nan
    cum_max = wealth.cummax()
    dd = wealth / cum_max - 1.0
    return float(dd.min())


def calmar_ratio(wealth: pd.Series) -> float:
    """CAGR divided by |max drawdown|."""
    mdd = max_drawdown(wealth)
    if mdd is None or mdd >= 0:
        return np.nan
    return float(cagr(wealth) / abs(mdd))


def drawdown_series(wealth: pd.Series) -> pd.Series:
    """Drawdown at every point in time (<= 0)."""
    return wealth / wealth.cummax() - 1.0


def crisis_drawdown(wealth: pd.Series, start: str, end: str) -> float:
    """Drawdown sustained within a specific window."""
    sub = wealth.loc[start:end]
    if sub.empty:
        return np.nan
    return max_drawdown(sub)


def performance_summary(
    wealth_dict: dict[str, pd.Series],
    rf_annual: float = 0.065,
    crisis_window: tuple[str, str] | None = None,
) -> pd.DataFrame:
    """
    Side-by-side performance table for multiple strategies.

    Parameters
    ----------
    wealth_dict : dict
        Mapping from strategy name to wealth path.
    rf_annual : float
        Risk-free rate for Sharpe / Sortino.
    crisis_window : (start, end), optional
        If given, includes a 'Crisis MDD' column for the window.

    Returns
    -------
    DataFrame with one row per strategy.
    """
    rows = []
    for name, w in wealth_dict.items():
        row = {
            "Strategy": name,
            "CAGR (%)": cagr(w) * 100,
            "Ann. Vol (%)": annualized_vol(w) * 100,
            "Sharpe": sharpe_ratio(w, rf_annual),
            "Sortino": sortino_ratio(w, rf_annual),
            "Max DD (%)": max_drawdown(w) * 100,
            "Calmar": calmar_ratio(w),
            "Final Wealth": float(w.iloc[-1] / w.iloc[0]),
        }
        if crisis_window is not None:
            row[f"Crisis MDD (%) [{crisis_window[0]}:{crisis_window[1]}]"] = (
                crisis_drawdown(w, *crisis_window) * 100
            )
        rows.append(row)
    return pd.DataFrame(rows).set_index("Strategy").round(4)
