"""
allocation.py
-------------
Regime-conditioned portfolio construction: Ledoit-Wolf covariance shrinkage
followed by Hierarchical Risk Parity (HRP) allocation.

For each detected regime, we:
    1. Compute the sample covariance matrix of constituent returns on days
       belonging to that regime.
    2. Apply Ledoit-Wolf shrinkage toward a constant-correlation target.
       This regularizes the matrix and dramatically reduces the condition
       number, which is critical because regime sub-samples are smaller
       than the full sample (Markowitz error-maximizer problem).
    3. Run HRP (Lopez de Prado, 2016) on the shrunk covariance:
        a. Convert covariance to a correlation-based distance metric.
        b. Hierarchically cluster assets (single linkage).
        c. Quasi-diagonalize the covariance via tree ordering.
        d. Recursively bisect, allocating inversely proportional to the
           cluster variance at each split.

HRP eschews expected returns entirely, which makes it well-suited to noisy
regime-conditioned data. It also produces strictly positive weights for
every asset (no corner solutions of standard mean-variance optimization).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf


# ---------------------------------------------------------------------------
# Covariance estimation
# ---------------------------------------------------------------------------

def ledoit_wolf_cov(returns: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    Estimate Ledoit-Wolf shrunk covariance.

    Returns
    -------
    cov_lw : pd.DataFrame
        Shrunk covariance matrix.
    shrinkage : float
        Optimal shrinkage intensity in [0, 1].
    """
    X = returns.dropna(how="any").values
    if X.shape[0] < 2:
        raise ValueError("Not enough observations for covariance estimation.")
    lw = LedoitWolf().fit(X)
    cov_lw = pd.DataFrame(
        lw.covariance_, index=returns.columns, columns=returns.columns
    )
    return cov_lw, float(lw.shrinkage_)


def condition_number(cov: pd.DataFrame | np.ndarray) -> float:
    """Spectral condition number lambda_max / lambda_min."""
    M = cov.values if hasattr(cov, "values") else cov
    eigs = np.linalg.eigvalsh(M)
    eigs = eigs[eigs > 0]
    if len(eigs) == 0:
        return np.inf
    return float(eigs.max() / eigs.min())


# ---------------------------------------------------------------------------
# Hierarchical Risk Parity
# ---------------------------------------------------------------------------

def _correlation_distance(corr: np.ndarray) -> np.ndarray:
    """
    Convert a correlation matrix to a proper distance matrix:
        d_{ij} = sqrt(0.5 * (1 - rho_{ij}))
    """
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)
    return dist


def _quasi_diag(link: np.ndarray) -> list[int]:
    """
    Recover the leaf ordering induced by hierarchical clustering.
    Implementation follows Lopez de Prado (2016).
    """
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]

    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])

    return sort_ix.tolist()


def _ivp_weights(cov: np.ndarray) -> np.ndarray:
    """Inverse-variance portfolio weights for a small cluster."""
    ivp = 1.0 / np.diag(cov)
    return ivp / ivp.sum()


def _cluster_var(cov: np.ndarray, items: list[int]) -> float:
    """Variance of an inverse-variance-weighted sub-cluster."""
    sub = cov[np.ix_(items, items)]
    w = _ivp_weights(sub).reshape(-1, 1)
    return float((w.T @ sub @ w).item())


def _recursive_bisection(cov: np.ndarray, sort_ix: list[int]) -> pd.Series:
    """
    Top-down recursive bisection: split the sorted asset list in half,
    allocate inversely proportional to cluster variance.
    """
    w = pd.Series(1.0, index=sort_ix)
    clusters = [sort_ix]
    while clusters:
        # Split each cluster into two halves
        clusters = [
            c[start:stop]
            for c in clusters
            for start, stop in ((0, len(c) // 2), (len(c) // 2, len(c)))
            if len(c) > 1
        ]
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            v_left = _cluster_var(cov, left)
            v_right = _cluster_var(cov, right)
            alpha = 1.0 - v_left / (v_left + v_right)
            w[left] *= alpha
            w[right] *= 1.0 - alpha
    return w


def hrp_weights(cov: pd.DataFrame) -> pd.Series:
    """
    Compute HRP weights from a (shrunk) covariance matrix.

    Returns a pd.Series of weights summing to 1, indexed by asset.
    """
    cols = cov.columns
    cov_arr = cov.values
    # Correlation from covariance
    std = np.sqrt(np.diag(cov_arr))
    corr = cov_arr / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    dist = _correlation_distance(corr)
    # squareform expects strict upper-triangle; ensure symmetric
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="single")
    sort_ix = _quasi_diag(link)

    w = _recursive_bisection(cov_arr, sort_ix)
    # Re-map integer positions back to asset names
    w.index = [cols[i] for i in w.index]
    # Reorder to original asset order for consistency
    return w.reindex(cols).fillna(0.0)


# ---------------------------------------------------------------------------
# Regime-conditioned allocation
# ---------------------------------------------------------------------------

def regime_weights(
    constituent_returns: pd.DataFrame,
    states: pd.Series,
    label_map: dict[int, str],
    min_obs: int = 60,
) -> dict[str, pd.Series]:
    """
    For each regime label, compute HRP weights using Ledoit-Wolf shrunk
    covariance estimated on that regime's sub-sample.

    Parameters
    ----------
    constituent_returns : DataFrame
        Wide DataFrame of asset log returns (rows: dates, cols: assets).
    states : Series
        Decoded HMM states aligned with constituent_returns.
    label_map : dict
        Integer state -> economic label (bull/bear/high_vol).
    min_obs : int
        Minimum observations required to run HRP for a regime.

    Returns
    -------
    Dict mapping label -> pd.Series of HRP weights.
    """
    # Align indices
    common = constituent_returns.index.intersection(states.index)
    R = constituent_returns.loc[common].dropna(how="all")
    s = states.loc[common].loc[R.index]

    weights: dict[str, pd.Series] = {}
    for state_id, label in label_map.items():
        mask = s == state_id
        sub = R.loc[mask].dropna(how="any", axis=1)
        sub = sub.dropna(how="any", axis=0)
        if len(sub) < min_obs or sub.shape[1] < 2:
            # Fall back to equal weights if insufficient data
            cols = R.dropna(how="any", axis=1).columns
            weights[label] = pd.Series(1.0 / len(cols), index=cols)
            continue
        cov_lw, _ = ledoit_wolf_cov(sub)
        w = hrp_weights(cov_lw)
        # Re-expand to full asset universe with zeros for assets not present
        full = pd.Series(0.0, index=R.columns)
        full.loc[w.index] = w.values
        # Re-normalize (in case any assets were dropped due to missing data)
        if full.sum() > 0:
            full = full / full.sum()
        weights[label] = full
    return weights
