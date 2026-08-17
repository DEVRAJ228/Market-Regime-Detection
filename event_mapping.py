"""
event_mapping.py
----------------
Map detected HMM regime transitions to known macro / financial events.

This is the "explain" layer of the detect-explain-allocate pipeline. The HMM
tells us *when* the market changed; the events tell us *why*. We curate a
small set of major Indian and global events spanning 2004-2024 and align
them to regime transitions within a small date window.

Sources (for the curated event list):
    - RBI monetary policy timeline (rate changes)
    - NSE / press archives for major shocks (2008 GFC, 2013 taper tantrum,
      2016 demonetization, 2020 COVID, 2022 rate-hike cycle)
    - GDELT-derived geopolitical / pandemic events

The list is intentionally curated rather than fetched live so the
analysis is fully reproducible without API dependencies.
"""

from __future__ import annotations

import pandas as pd


# Curated event list: (date, event_type, description)
# These are the dominant macro / market events that any regime detector
# should be able to align with on the Indian market over 2004-2024.
CURATED_EVENTS: list[tuple[str, str, str]] = [
    # --- 2008 Global Financial Crisis ---
    ("2008-01-21", "global_shock",     "Global selloff begins; NIFTY -7.4% intraday"),
    ("2008-09-15", "global_shock",     "Lehman Brothers files for bankruptcy"),
    ("2008-10-24", "global_shock",     "Global panic; NIFTY hits 2008 low"),
    ("2008-10-20", "rbi_policy",       "RBI emergency repo cut: 9.0% -> 8.0%"),

    # --- 2009 recovery ---
    ("2009-05-18", "domestic_political", "UPA wins general election; NIFTY +17% (upper circuit)"),

    # --- 2011-2013 high-vol period ---
    ("2011-08-05", "global_shock",     "S&P downgrades US sovereign rating"),
    ("2013-05-22", "global_shock",     "Bernanke taper-tantrum speech; EM selloff"),
    ("2013-08-28", "domestic_macro",   "INR hits all-time low vs USD"),

    # --- 2014 BJP rally ---
    ("2014-05-16", "domestic_political", "BJP wins majority; NIFTY rallies on reform hopes"),

    # --- 2015-2016 China / oil / demonetization ---
    ("2015-08-24", "global_shock",     "China devaluation; global selloff"),
    ("2016-02-11", "global_shock",     "Oil collapses below $30; equity selloff"),
    ("2016-11-08", "domestic_political", "Demonetization announced"),
    ("2016-12-07", "rbi_policy",       "RBI surprise pause despite demonetization shock"),

    # --- 2018-2019 NBFC / trade-war ---
    ("2018-09-21", "domestic_macro",   "IL&FS default; NBFC liquidity crisis"),
    ("2018-10-04", "rbi_policy",       "RBI unexpected pause; INR weakens further"),

    # --- 2020 COVID ---
    ("2020-03-12", "global_shock",     "WHO declares COVID-19 a pandemic"),
    ("2020-03-23", "global_shock",     "Indian national lockdown begins; NIFTY 2020 low"),
    ("2020-03-27", "rbi_policy",       "RBI emergency repo cut: 5.15% -> 4.40%"),
    ("2020-05-22", "rbi_policy",       "RBI further cut: 4.40% -> 4.00%"),

    # --- 2022 rate hike cycle / Russia-Ukraine ---
    ("2022-02-24", "global_shock",     "Russia invades Ukraine; commodity / equity shock"),
    ("2022-05-04", "rbi_policy",       "RBI off-cycle hike: 4.00% -> 4.40%"),
    ("2022-06-08", "rbi_policy",       "RBI hike: 4.40% -> 4.90%"),
    ("2022-08-05", "rbi_policy",       "RBI hike: 4.90% -> 5.40%"),
    ("2022-09-30", "rbi_policy",       "RBI hike: 5.40% -> 5.90%"),
    ("2022-12-07", "rbi_policy",       "RBI hike: 5.90% -> 6.25%"),

    # --- 2023 ---
    ("2023-02-08", "rbi_policy",       "RBI hike: 6.25% -> 6.50%"),
    ("2023-03-10", "global_shock",     "SVB collapse; banking-sector stress"),
    ("2023-10-07", "global_shock",     "Israel-Hamas conflict escalates"),

    # --- 2024 ---
    ("2024-06-04", "domestic_political", "Election results; BJP loses majority; NIFTY -5.9% intraday"),
]


def events_df() -> pd.DataFrame:
    """Return curated events as a DataFrame indexed by date."""
    df = pd.DataFrame(CURATED_EVENTS, columns=["date", "type", "description"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def map_transitions_to_events(
    transitions: pd.DataFrame,
    window_days: int = 5,
    events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    For each detected regime transition, attach any curated events within
    a +/- window_days window. A transition may be matched to zero, one, or
    several events.

    Parameters
    ----------
    transitions : DataFrame
        Output of hmm_regime.regime_transitions(): columns
        date, from_state, to_state.
    window_days : int
        Symmetric window around the transition date to search for events.
    events : DataFrame, optional
        Override the curated event list. Must be indexed by datetime.

    Returns
    -------
    DataFrame with columns:
        transition_date, from_state, to_state, event_date,
        event_type, event_description, lag_days
    """
    if events is None:
        events = events_df()

    rows = []
    for _, row in transitions.iterrows():
        td = pd.Timestamp(row["date"])
        lo = td - pd.Timedelta(days=window_days)
        hi = td + pd.Timedelta(days=window_days)
        matches = events.loc[lo:hi]
        if matches.empty:
            rows.append({
                "transition_date": td,
                "from_state": int(row["from_state"]),
                "to_state": int(row["to_state"]),
                "event_date": pd.NaT,
                "event_type": "",
                "event_description": "",
                "lag_days": pd.NA,
            })
        else:
            for ed, e in matches.iterrows():
                rows.append({
                    "transition_date": td,
                    "from_state": int(row["from_state"]),
                    "to_state": int(row["to_state"]),
                    "event_date": ed,
                    "event_type": e["type"],
                    "event_description": e["description"],
                    "lag_days": (ed - td).days,
                })
    return pd.DataFrame(rows)


def match_rate(transitions: pd.DataFrame, window_days: int = 5) -> dict:
    """
    Compute summary statistics on how well detected transitions align with
    curated events:
        n_transitions    : total HMM-detected transitions
        n_matched        : transitions with at least one event in the window
        match_rate       : ratio of matched to total
    """
    mapping = map_transitions_to_events(transitions, window_days=window_days)
    n_total = mapping["transition_date"].nunique()
    n_matched = mapping.loc[mapping["event_date"].notna(), "transition_date"].nunique()
    return {
        "window_days": window_days,
        "n_transitions": n_total,
        "n_matched": n_matched,
        "match_rate": n_matched / n_total if n_total else 0.0,
    }
