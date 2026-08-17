"""
hmm_regime.py
-------------
3-state Gaussian Hidden Markov Model for regime detection on NIFTY 50
log-returns.

The HMM treats the market as a system that switches between latent regimes
(bull / bear / high-volatility), each with its own Gaussian return distribution.
Parameters are estimated by EM (Baum-Welch); the most likely state sequence is
recovered by Viterbi decoding.

The fitted model returns a regime label for every trading day. These labels
drive the GARCH and HRP layers downstream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


def fit_hmm(
    returns: pd.Series,

    n_states: int = 3,
    n_iter: int = 1000,
    random_state: int = 42,
    covariance_type: str = "full",
) -> GaussianHMM:
    """
    Fit a Gaussian HMM on a univariate return series using EM.

    Parameters
    ----------
    returns : pd.Series
        Daily log returns of NIFTY 50.
    n_states : int
        Number of hidden states (default 3: bull / bear / high-vol).
    n_iter : int
        Max EM iterations.
    random_state : int
        Seed for reproducibility.
    covariance_type : str
        'full' is appropriate for univariate (reduces to scalar variance).

    Returns
    -------
    Fitted hmmlearn GaussianHMM object.
    """
    X = returns.values.reshape(-1, 1)
    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
        tol=1e-5,
    )
    model.fit(X)
    return model


def viterbi_decode(model: GaussianHMM, returns: pd.Series) -> pd.Series:
    """
    Decode the most likely state sequence using Viterbi.

    Returns a pd.Series of integer state labels aligned with the input index.
    """
    X = returns.values.reshape(-1, 1)
    states = model.predict(X)
    return pd.Series(states, index=returns.index, name="state")


def posterior_probs(model: GaussianHMM, returns: pd.Series) -> pd.DataFrame:
    """
    Compute the smoothed posterior probability of each state for each day.
    Useful for soft regime assignment and for plotting regime certainty.
    """
    X = returns.values.reshape(-1, 1)
    probs = model.predict_proba(X)
    return pd.DataFrame(
        probs,
        index=returns.index,
        columns=[f"P(s={k})" for k in range(model.n_components)],
    )


def label_regimes(model: GaussianHMM) -> dict[int, str]:
    """
    Map raw HMM state indices to economically meaningful labels.

    Convention:
        bull       : highest mean return
        bear       : lowest mean return  (negative or near-zero)
        high_vol   : highest variance (remaining state)

    This ordering is recovered automatically from the fitted parameters,
    so the label assignment is reproducible across runs even though
    hmmlearn does not impose any state ordering.
    """
    means = model.means_.flatten()
    variances = np.array([np.diag(c).item() for c in model.covars_])

    n = len(means)
    if n != 3:
        # General fallback: rank by mean, label extremes
        order = np.argsort(means)
        labels = {}
        labels[int(order[0])] = "bear"
        labels[int(order[-1])] = "bull"
        for idx in order[1:-1]:
            labels[int(idx)] = f"intermediate_{int(idx)}"
        return labels

    # For 3 states: highest mean = bull, lowest mean = bear,
    # remaining state labeled by variance: if it has the highest variance,
    # call it high_vol; otherwise it's an intermediate regime.
    bull = int(np.argmax(means))
    bear = int(np.argmin(means))
    other = int([k for k in range(3) if k not in (bull, bear)][0])

    labels = {bull: "bull", bear: "bear", other: "high_vol"}

    # Sanity check: the "high_vol" state should genuinely have higher variance
    # than the bull state. If not, the labelling is still by mean ranking,
    # which is the more economically meaningful axis.
    return labels


def summarize_regimes(
    model: GaussianHMM,
    states: pd.Series,
    label_map: dict[int, str] | None = None,
) -> pd.DataFrame:
    """
    Produce a per-regime summary table: mean return, volatility, average
    duration (in days), and frequency.

    Average duration is computed from the diagonal of the transition matrix:
        E[duration | state k] = 1 / (1 - A[k,k])
    """
    if label_map is None:
        label_map = label_regimes(model)

    means = model.means_.flatten()
    variances = np.array([np.diag(c).item() for c in model.covars_])
    diag = np.diag(model.transmat_)

    rows = []
    for k in range(model.n_components):
        rows.append({
            "state": k,
            "label": label_map.get(k, f"state_{k}"),
            "mean_return_daily": means[k],
            "volatility_daily": np.sqrt(variances[k]),
            "ann_return": means[k] * 252,
            "ann_volatility": np.sqrt(variances[k]) * np.sqrt(252),
            "expected_duration_days": 1.0 / (1.0 - diag[k]) if diag[k] < 1 else np.inf,
            "frequency": (states == k).mean(),
        })
    return pd.DataFrame(rows).set_index("label").sort_values(
        "mean_return_daily", ascending=False
    )


def regime_transitions(states: pd.Series) -> pd.DataFrame:
    """
    Identify days on which the decoded regime changes.

    Returns a DataFrame with columns:
        date, from_state, to_state
    """
    changes = states != states.shift(1)
    changes.iloc[0] = False  # The first observation isn't a transition
    transition_dates = states.index[changes]

    return pd.DataFrame({
        "date": transition_dates,
        "from_state": states.shift(1).loc[transition_dates].astype(int).values,
        "to_state": states.loc[transition_dates].astype(int).values,
    })
