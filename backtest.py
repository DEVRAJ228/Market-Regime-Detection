"""
backtest.py
-----------
Regime-adaptive portfolio backtest engine.

Walks daily through the test sample, maintains the current portfolio,
and rebalances under a hybrid rule:
    - At the start of every calendar month, OR
    - Immediately after the HMM regime has *changed and remained stable*
      for `regime_confirm_days` consecutive days.

The dual rule balances responsiveness (regime trigger) against transaction
costs (monthly cap). Per-regime weights are computed in-sample on the
training window only, eliminating look-ahead.

Transaction costs are applied as a fixed bps charge per unit of one-way
turnover at each rebalance.

Benchmarks computed alongside the regime-adaptive strategy:
    - buy_and_hold_nifty : NIFTY 50 index return
    - equal_weight       : equal-weighted constituent portfolio (rebalanced monthly)
    - hrp_single         : HRP on full-sample covariance (no regime conditioning)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .allocation import hrp_weights, ledoit_wolf_cov, regime_weights


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """Container for backtest parameters."""
    transaction_cost_bps: float = 15.0   # one-way, in bps
    regime_confirm_days: int = 5         # rebalance trigger after N stable days
    monthly_rebalance: bool = True       # also rebalance at month start
    initial_wealth: float = 1.0


# ---------------------------------------------------------------------------
# Core backtest
# ---------------------------------------------------------------------------

def _next_month_start(idx: pd.DatetimeIndex) -> pd.Series:
    """
    For each date in idx, mark True if it is the first trading day of a
    calendar month (i.e. month changed vs. the previous day).
    """
    months = idx.to_series().dt.to_period("M")
    return months != months.shift(1)


def _compute_rebalance_flags(
    dates: pd.DatetimeIndex,
    states: pd.Series,
    cfg: BacktestConfig,
) -> pd.Series:
    """
    Boolean series marking days on which the portfolio should rebalance.

    Rule: rebalance on
        (a) the first trading day of a month, OR
        (b) the regime has changed and remained stable for
            regime_confirm_days consecutive days.
    """
    month_flag = _next_month_start(dates).reindex(dates).fillna(False)
    month_flag.iloc[0] = True  # always set the initial weights on day 1

    # Regime confirmation flag
    s = states.reindex(dates).ffill()
    change = (s != s.shift(1)).astype(int)
    # Days since the most recent regime change
    days_since_change = change.groupby(change.cumsum()).cumcount()
    confirm_flag = days_since_change == cfg.regime_confirm_days

    return (month_flag | confirm_flag).fillna(False)


def _apply_turnover_cost(
    w_old: pd.Series,
    w_new: pd.Series,
    cost_bps: float,
) -> tuple[float, float]:
    """
    Return (one_way_turnover, cost_drag).
    cost_drag is a multiplicative factor in (0, 1] applied to wealth.
    """
    turnover = 0.5 * (w_new.sub(w_old, fill_value=0.0).abs().sum())
    cost = (cost_bps / 1e4) * 2.0 * turnover  # bps -> fraction, two-way
    return float(turnover), float(1.0 - cost)


def run_backtest(
    constituent_returns: pd.DataFrame,
    states: pd.Series,
    label_map: dict[int, str],
    regime_weight_dict: dict[str, pd.Series],
    cfg: BacktestConfig | None = None,
) -> dict:
    """
    Walk-forward backtest of the regime-adaptive HRP strategy.

    Parameters
    ----------
    constituent_returns : DataFrame
        Daily log returns of constituents over the *test* sample.
    states : Series
        HMM-decoded regime labels over the test sample.
    label_map : dict
        State -> label mapping.
    regime_weight_dict : dict
        Pre-computed HRP weights per regime label (computed from training data).
    cfg : BacktestConfig
        Backtest configuration. Defaults applied if None.

    Returns
    -------
    Dict with keys:
        wealth : pd.Series of portfolio wealth (starting at 1.0)
        weights: pd.DataFrame of weights at each rebalance
        log    : pd.DataFrame with one row per rebalance event
        turnover_series : pd.Series of one-way turnover per rebalance
    """
    if cfg is None:
        cfg = BacktestConfig()

    R = constituent_returns.copy()
    # Convert log to simple returns for compounding
    simple = np.expm1(R.fillna(0.0))

    dates = R.index
    rebalance_flags = _compute_rebalance_flags(dates, states, cfg)

    state_aligned = states.reindex(dates).ffill().astype(int)

    wealth = cfg.initial_wealth
    wealth_path = pd.Series(index=dates, dtype=float)

    current_weights = pd.Series(0.0, index=R.columns)
    rebalance_log = []
    weights_history = []

    for i, date in enumerate(dates):
        if rebalance_flags.loc[date]:
            state_id = int(state_aligned.loc[date])
            label = label_map.get(state_id, "bull")
            new_weights = regime_weight_dict.get(
                label, pd.Series(1.0 / R.shape[1], index=R.columns)
            ).reindex(R.columns).fillna(0.0)

            turnover, cost_mult = _apply_turnover_cost(
                current_weights, new_weights, cfg.transaction_cost_bps
            )
            wealth *= cost_mult
            current_weights = new_weights.copy()

            rebalance_log.append({
                "date": date,
                "regime": label,
                "turnover": turnover,
                "wealth_after_cost": wealth,
            })
            weights_history.append(current_weights.rename(date))

        # Apply the day's return
        day_ret = (current_weights * simple.loc[date]).sum()
        wealth *= (1.0 + day_ret)
        wealth_path.loc[date] = wealth

    return {
        "wealth": wealth_path,
        "weights": pd.DataFrame(weights_history) if weights_history else pd.DataFrame(),
        "log": pd.DataFrame(rebalance_log).set_index("date") if rebalance_log else pd.DataFrame(),
        "turnover_series": (
            pd.DataFrame(rebalance_log).set_index("date")["turnover"]
            if rebalance_log else pd.Series(dtype=float)
        ),
    }


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def buy_and_hold(index_returns: pd.Series, cfg: BacktestConfig | None = None) -> pd.Series:
    """Buy-and-hold the NIFTY 50 index. No costs (you buy once)."""
    if cfg is None:
        cfg = BacktestConfig()
    simple = np.expm1(index_returns.fillna(0.0))
    return cfg.initial_wealth * (1.0 + simple).cumprod()


def equal_weight_backtest(
    constituent_returns: pd.DataFrame,
    cfg: BacktestConfig | None = None,
) -> pd.Series:
    """Equal-weighted portfolio rebalanced monthly with transaction costs."""
    if cfg is None:
        cfg = BacktestConfig()
    R = constituent_returns.copy()
    simple = np.expm1(R.fillna(0.0))
    dates = R.index
    rebalance_flags = _next_month_start(dates)
    rebalance_flags.iloc[0] = True

    wealth = cfg.initial_wealth
    path = pd.Series(index=dates, dtype=float)
    current = pd.Series(0.0, index=R.columns)
    target = pd.Series(1.0 / R.shape[1], index=R.columns)

    for date in dates:
        if rebalance_flags.loc[date]:
            _, cost_mult = _apply_turnover_cost(
                current, target, cfg.transaction_cost_bps
            )
            wealth *= cost_mult
            current = target.copy()
        day_ret = (current * simple.loc[date]).sum()
        wealth *= (1.0 + day_ret)
        path.loc[date] = wealth
    return path


def hrp_single_backtest(
    constituent_returns_train: pd.DataFrame,
    constituent_returns_test: pd.DataFrame,
    cfg: BacktestConfig | None = None,
) -> pd.Series:

    if cfg is None:
        cfg = BacktestConfig()

    # -----------------------------
    # Clean training matrix
    # -----------------------------
    clean_train = constituent_returns_train.copy()

    # Drop assets with too many NaNs
    clean_train = clean_train.dropna(
        thresh=int(0.9 * len(clean_train)),
        axis=1
    )

    # Fill remaining NaNs
    clean_train = clean_train.fillna(0.0)

    print("Train shape:", clean_train.shape)

    if clean_train.shape[1] < 2:
        raise ValueError(
            f"Need at least 2 assets for HRP, got {clean_train.shape[1]}"
        )

    # -----------------------------
    # Covariance + HRP
    # -----------------------------
    cov_lw, _ = ledoit_wolf_cov(clean_train)

    print("Cov shape:", cov_lw.shape)

    weights = hrp_weights(cov_lw)

    # Align to test universe
    w_full = pd.Series(0.0, index=constituent_returns_test.columns)
    w_full.loc[weights.index] = weights.values

    if w_full.sum() > 0:
        w_full /= w_full.sum()

    # -----------------------------
    # Backtest
    # -----------------------------
    R = constituent_returns_test.copy()
    simple = np.expm1(R.fillna(0.0))

    dates = R.index
    rebalance_flags = _next_month_start(dates)
    rebalance_flags.iloc[0] = True

    wealth = cfg.initial_wealth
    path = pd.Series(index=dates, dtype=float)

    current = pd.Series(0.0, index=R.columns)

    for date in dates:

        if rebalance_flags.loc[date]:
            _, cost_mult = _apply_turnover_cost(
                current,
                w_full,
                cfg.transaction_cost_bps
            )

            wealth *= cost_mult
            current = w_full.copy()

        day_ret = (current * simple.loc[date]).sum()

        wealth *= (1.0 + day_ret)

        path.loc[date] = wealth

    return path