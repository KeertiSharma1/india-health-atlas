"""
india_health_atlas / src / scoring.py
──────────────────────────────────────────────────────────────────
Core scoring logic — District Health Vulnerability Score (DHVS).

This module is imported by every other script that needs scores.
Nothing is hard-coded here; all indicator config lives in
config/indicators.yaml so you can adjust weights without touching
this file.
"""

import pandas as pd
import numpy as np
import yaml
import os

# Path to config (works regardless of where you call this from)
_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(_HERE, '..', 'config', 'indicators.yaml')


def load_config(config_path: str = CONFIG_PATH) -> dict:
    """Load indicator weights and column mappings from YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Guard: weights must sum to 1.0
    total_w = sum(m['weight'] for m in config['indicators'].values())
    assert abs(total_w - 1.0) < 1e-6, (
        f"Indicator weights sum to {total_w:.4f}, not 1.0. "
        f"Edit config/indicators.yaml to fix this."
    )
    return config


def load_raw_data(csv_path: str) -> pd.DataFrame:
    """
    Load the NFHS-5 district CSV.
    Strips whitespace from column names and key text columns.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()
    df['District Names'] = df['District Names'].str.strip()
    df['State/UT']       = df['State/UT'].str.strip()
    return df


def extract_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Pull the 6 indicator columns out of the raw wide-format CSV,
    clean messy values, and fill missing data with state medians.

    NFHS data quirks this handles:
      *         → missing (sample too small to report)
      (76.3)    → 76.3   (low sample, but still usable — we keep it)
    """
    clean = df[['District Names', 'State/UT']].copy()
    clean.columns = ['district', 'state']

    for key, meta in config['indicators'].items():
        col = meta['column']
        raw = df[col].astype(str).str.strip()
        raw = raw.str.replace(r'^\((.+)\)$', r'\1', regex=True)  # remove brackets
        raw = raw.replace('*', np.nan)
        clean[key] = pd.to_numeric(raw, errors='coerce')

    # Fill missing values — use state median first, national median as fallback
    for key in config['indicators']:
        clean[key] = clean.groupby('state')[key].transform(
            lambda x: x.fillna(x.median())
        )
        clean[key] = clean[key].fillna(clean[key].median())

    return clean


def _minmax_normalize(series: pd.Series) -> pd.Series:
    """
    Scale a series to 0–100.
    After normalization: 0 = best outcome, 100 = worst outcome.
    For inverted indicators, the flip happens BEFORE this function is called,
    so this function always maps the as-passed minimum to 0 and maximum to 100.
    """
    mn, mx = series.min(), series.max()
    if mn == mx:
        # All districts identical on this indicator — assign neutral midpoint
        return pd.Series(50.0, index=series.index)
    return (series - mn) / (mx - mn) * 100


def compute_dhvs(clean: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Normalize each indicator and compute the composite DHVS score.

    Inversion logic (for indicators where HIGH value = GOOD outcome,
    e.g. high immunisation rate):
        flipped = max(vals) - vals
    This mirrors the range so that a high raw value becomes a low
    normalized value (less vulnerable), without shifting the scale.
    The flipped series is then passed to _minmax_normalize just like
    any other indicator.

    NOTE: the previously used formula  max - vals + min  was incorrect;
    adding min shifted the values unnecessarily and caused the flipped
    series to span [min, max+min] instead of [0, max-min] before
    normalization. The correct inversion is simply  max - vals.
    """
    df = clean.copy()

    for key, meta in config['indicators'].items():
        vals = df[key].copy()
        if meta.get('invert', False):
            # Flip: highest raw value → lowest vulnerability
            vals = vals.max() - vals
        df[f'{key}_norm'] = _minmax_normalize(vals)

    # Weighted sum — weights already validated to sum to 1.0 in load_config()
    df['dhvs'] = sum(
        df[f'{key}_norm'] * meta['weight']
        for key, meta in config['indicators'].items()
    ).round(2)

    # Rank (1 = most vulnerable)
    df['dhvs_rank'] = df['dhvs'].rank(ascending=False, method='min').astype(int)

    # Vulnerability band
    bands      = config['bands']
    bin_edges  = [b['min'] for b in bands] + [bands[-1]['max']]
    bin_labels = [b['label'] for b in bands]
    df['vulnerability_band'] = pd.cut(
        df['dhvs'], bins=bin_edges, labels=bin_labels, include_lowest=True
    )

    return df


def build_score_table(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Return a clean, sorted output table — one row per district."""
    indicator_keys = list(config['indicators'].keys())
    cols = ['dhvs_rank', 'district', 'state', 'dhvs', 'vulnerability_band'] + indicator_keys
    return df[cols].sort_values('dhvs_rank').reset_index(drop=True)


def run_pipeline(raw_csv: str, config_path: str = CONFIG_PATH) -> pd.DataFrame:
    """
    End-to-end convenience function.
    Returns the scored, ranked table ready for analysis or saving.
    """
    config = load_config(config_path)
    raw    = load_raw_data(raw_csv)
    clean  = extract_indicators(raw, config)
    scored = compute_dhvs(clean, config)
    return build_score_table(scored, config)
