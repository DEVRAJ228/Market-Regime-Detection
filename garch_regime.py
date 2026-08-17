"""
garch_regime.py
---------------
Regime-conditioned GARCH(1,1) volatility modeling.

After the HMM decodes a regime sequence, this module fits an independent
GARCH(1,1) within each regime's sub-series. The motivation (Bauwens et al.,
2006; Mikosch & Starica, 2004): a single GARCH fitted globally on
regime-switching data produces inflated persistence (beta + gamma -> 1),
the so-called IGARCH-as-misspecification artifact. Conditioning on regime
removes this bias and reveals genuinely different volatility dynamics
across market states.

We do not jointly estimate HMM and GARCH (this would be path-dependent and
intractable by maximum likelihood); we use the Viterbi-decoded state
sequence as fixed and fit GARCH within each regime.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model


def fit_garch_global(returns: pd.Series, rescale: bool = True) -> dict:
    """
    Fit a single GARCH(1,1) on the full return series. Used to demonstrate
    the IGARCH-style bias when regimes are ignored.

    Returns a dict with the fitted result and key parameters.
    """
    # arch_model prefers percent returns for numerical stability
    y = returns.dropna() * (100 if rescale else 1)
    am = arch_model(y, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
    res = am.fit(disp="off", show_warning=False)

    params = res.params
    omega = float(params.get("omega", np.nan))
    alpha = float(params.get("alpha[1]", np.nan))  # ARCH term (γ in our notation)
    beta = float(params.get("beta[1]", np.nan))    # GARCH term

    return {
        "result": res,
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "persistence": alpha + beta,
        "loglik": float(res.loglikelihood),
        "aic": float(res.aic),
        "bic": float(res.bic),
        "n_obs": int(len(y)),
    }


def fit_garch_per_regime(
    returns: pd.Series,
    states: pd.Series,
    label_map: dict[int, str],
    rescale: bool = True,
    min_obs: int = 100,
) -> pd.DataFrame:
    """
    Fit GARCH(1,1) independently within each regime's sub-series.

    Parameters
    ----------
    returns : pd.Series
        Daily log returns.
    states : pd.Series
        Decoded HMM state labels, integer-valued, aligned with returns.
    label_map : dict
        Mapping from integer state to economic label (bull/bear/high_vol).
    rescale : bool
        Multiply returns by 100 before fitting (numerical stability).
    min_obs : int
        Minimum observations required to fit GARCH in a regime.

    Returns
    -------
    DataFrame with one row per regime and columns:
        label, n_obs, omega, alpha, beta, persistence, ann_vol, loglik
    """
    rows = []
    for state_id, label in label_map.items():
        mask = states == state_id
        sub = returns.loc[mask].dropna()
        if len(sub) < min_obs:
            rows.append({
                "state": state_id,
                "label": label,
                "n_obs": len(sub),
                "omega": np.nan,
                "alpha": np.nan,
                "beta": np.nan,
                "persistence": np.nan,
                "ann_vol": np.nan,
                "loglik": np.nan,
                "note": "insufficient observations",
            })
            continue

        try:
            fit = fit_garch_global(sub, rescale=rescale)
            # Compute unconditional volatility implied by the fit
            uncond_var_pct = (
                fit["omega"] / (1 - fit["persistence"])
                if fit["persistence"] < 1 else np.nan
            )
            # Convert back to log-return scale and annualize
            scale = 100 if rescale else 1
            ann_vol = np.sqrt(uncond_var_pct / (scale ** 2)) * np.sqrt(252)

            rows.append({
                "state": state_id,
                "label": label,
                "n_obs": len(sub),
                "omega": fit["omega"],
                "alpha": fit["alpha"],
                "beta": fit["beta"],
                "persistence": fit["persistence"],
                "ann_vol": ann_vol,
                "loglik": fit["loglik"],
                "note": "ok",
            })
        except Exception as exc:
            rows.append({
                "state": state_id,
                "label": label,
                "n_obs": len(sub),
                "omega": np.nan,
                "alpha": np.nan,
                "beta": np.nan,
                "persistence": np.nan,
                "ann_vol": np.nan,
                "loglik": np.nan,
                "note": f"failed: {exc}",
            })

    df = pd.DataFrame(rows).set_index("label")
    # Order rows: bull, high_vol, bear
    order_pref = ["bull", "high_vol", "bear"]
    available = [o for o in order_pref if o in df.index]
    other = [o for o in df.index if o not in order_pref]
    return df.loc[available + other]
