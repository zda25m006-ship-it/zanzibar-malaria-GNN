"""
Feature engineering — LEAKAGE-FREE.

Node features use only information available BEFORE the prediction month:
  - Lagged case counts (t-1, t-2)
  - Lagged imported fraction (t-1) — scale-free, avoids count leakage
  - Individual risk scores (t-1): mean risk, fraction high-risk, visit footfall
    * Long-stay travelers (>14 nights) shifted to t-1 as timing correction
  - CHIRPS rainfall (already has 2–6 week built-in lag)
  - Temperature (environmental, not target-derived)
  - Static: log-population, risk category
  - Temporal: sin/cos month, is_rainy_season

NO current-month case counts. NO travel_outflow (== imported_cases).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data.data_loader import (
    UNGUJA_DISTRICTS, ALL_MAINLAND_REGIONS,
    RISK_CATEGORY, POPULATION_PROXY
)
from data.graph_builder import get_all_nodes, get_node_to_idx

FEATURE_NAMES = [
    # Aggregated count features (lagged)
    'lagged_total_cases_1m',    # 0:  total cases in district t-1
    'lagged_imported_frac_1m',  # 1:  fraction imported t-1 (scale-free)
    'lagged_total_cases_2m',    # 2:  total cases t-2
    'lagged_imported_frac_2m',  # 3:  fraction imported t-2

    # Individual risk-score features (lagged t-1, no leakage)
    'mean_risk_score_1m',       # 4:  mean P(imported) across district patients t-1
    'frac_high_risk_1m',        # 5:  fraction with risk > 0.5 at t-1
    'n_high_risk_1m',           # 6:  count of high-risk patients at t-1 (scale)
    'total_visits_1m',          # 7:  total clinic visits at t-1 (footfall, not imported count)
    'longstay_shift_risk_1m',   # 8:  mean risk of long-stay travelers shifted from t (timing corr)

    # Environmental
    'rainfall_mm',              # 9:  CHIRPS (already lagged 2-6 wks)
    'temperature',              # 10: monthly mean temp

    # Static
    'log_population',           # 11: static
    'risk_category',            # 12: static (0-3)

    # Temporal
    'sin_month',                # 13: seasonal cycle
    'cos_month',                # 14: seasonal cycle
]
NUM_FEATURES = len(FEATURE_NAMES)


def _get_prev(all_months: list, current: str, lag: int):
    try:
        idx = all_months.index(current)
        if idx - lag >= 0:
            return all_months[idx - lag]
    except ValueError:
        pass
    return None


def build_node_features(
    monthly_counts: pd.DataFrame,
    rainfall: pd.DataFrame,
    temperature: pd.DataFrame,
    year_month: str,
    all_months: list,
    risk_features: dict = None,
) -> np.ndarray:
    """
    Build [num_nodes, NUM_FEATURES] node feature array for `year_month`.
    Only uses lagged or static/environmental information — no target leakage.

    Args:
        risk_features: optional dict from build_risk_features_for_month().
                       If None, risk-score features are set to 0.
    """
    nodes = get_all_nodes()
    node_to_idx = get_node_to_idx()
    n = len(nodes)
    feats = np.zeros((n, NUM_FEATURES), dtype=np.float32)

    month_num = int(year_month.split('-')[1])
    sin_m = np.sin(2 * np.pi * month_num / 12)
    cos_m = np.cos(2 * np.pi * month_num / 12)

    prev1 = _get_prev(all_months, year_month, 1)
    prev2 = _get_prev(all_months, year_month, 2)

    prev1_data = monthly_counts[monthly_counts['year_month'] == prev1] if prev1 else pd.DataFrame()
    prev2_data = monthly_counts[monthly_counts['year_month'] == prev2] if prev2 else pd.DataFrame()

    # ---- Unguja districts ------------------------------------------------
    for district in UNGUJA_DISTRICTS:
        idx = node_to_idx[district]

        # Lag 1 (aggregated)
        if len(prev1_data):
            row = prev1_data[prev1_data['home_district'] == district]
            if len(row):
                r = row.iloc[0]
                feats[idx, 0] = r['total_cases']
                feats[idx, 1] = r['imported_frac']

        # Lag 2 (aggregated)
        if len(prev2_data):
            row = prev2_data[prev2_data['home_district'] == district]
            if len(row):
                r = row.iloc[0]
                feats[idx, 2] = r['total_cases']
                feats[idx, 3] = r['imported_frac']

        # Individual risk-score features (lagged t-1)
        if risk_features and district in risk_features:
            rf = risk_features[district]
            feats[idx, 4] = rf.get('mean_risk', 0.0)
            feats[idx, 5] = rf.get('frac_high_risk', 0.0)
            feats[idx, 6] = rf.get('n_high_risk', 0.0)
            feats[idx, 7] = rf.get('n_total', 0.0)
            feats[idx, 8] = rf.get('longstay_mean_risk', 0.0)

        # Rainfall: use Tanga as proxy for Zanzibar (closest mainland region)
        rain = rainfall[(rainfall['region'] == 'Tanga') &
                        (rainfall['year_month'] == year_month)]
        feats[idx, 9] = rain['rainfall_mm'].values[0] if len(rain) else 0.0

        # Temperature
        temp = temperature[(temperature['region'].isin([district, 'Dar Es Salaam'])) &
                           (temperature['year_month'] == year_month)]
        feats[idx, 10] = temp['temperature'].mean() if len(temp) else 0.0

        feats[idx, 11] = np.log1p(POPULATION_PROXY.get(district, 100_000))
        feats[idx, 12] = float(RISK_CATEGORY.get(district, 0))
        feats[idx, 13] = sin_m
        feats[idx, 14] = cos_m

    # ---- Mainland regions -----------------------------------------------
    for region in ALL_MAINLAND_REGIONS:
        idx = node_to_idx[region]

        # Lag 1: inflow from Unguja to this region
        if len(prev1_data):
            col = f'travel_to_{region}'
            if col in prev1_data.columns:
                feats[idx, 0] = prev1_data[col].sum()
        if len(prev2_data):
            col = f'travel_to_{region}'
            if col in prev2_data.columns:
                feats[idx, 2] = prev2_data[col].sum()

        # Rainfall (CHIRPS)
        rain = rainfall[(rainfall['region'] == region) &
                        (rainfall['year_month'] == year_month)]
        feats[idx, 9] = rain['rainfall_mm'].values[0] if len(rain) else 0.0

        # Temperature
        temp = temperature[(temperature['region'] == region) &
                           (temperature['year_month'] == year_month)]
        feats[idx, 10] = temp['temperature'].mean() if len(temp) else 0.0

        feats[idx, 11] = np.log1p(POPULATION_PROXY.get(region, 1_000_000))
        feats[idx, 12] = float(RISK_CATEGORY.get(region, 1))
        feats[idx, 13] = sin_m
        feats[idx, 14] = cos_m

    return feats


# ---------------------------------------------------------------------------
def build_risk_features_for_month(
    clinic_raw: pd.DataFrame,
    scorer,
    prev_month: str,
    districts: list,
) -> dict:
    """
    Build per-district risk-score features from individual-level data at prev_month.

    Timing correction for long-stay travelers:
      - Patients with travel_over_14_nights=1 who tested positive likely
        traveled in the PREVIOUS month; their risk contribution is excluded
        from the current month and instead attributed to t-1 (the caller
        uses this as the lag-1 feature).
      - Short-stay patients (<= 14 nights) are attributed to the month they
        visited clinic.

    Returns:
        dict: { district -> {mean_risk, frac_high_risk, n_high_risk,
                             n_total, longstay_mean_risk} }
    """
    month_data = clinic_raw[clinic_raw['date'] == prev_month].copy()
    if len(month_data) == 0:
        return {d: {'mean_risk': 0, 'frac_high_risk': 0, 'n_high_risk': 0,
                    'n_total': 0, 'longstay_mean_risk': 0} for d in districts}

    # Score all patients in that month
    scores = scorer.predict_risk(month_data)
    month_data = month_data.copy()
    month_data['_risk'] = scores

    result = {}
    for dist in districts:
        d_pat = month_data[month_data['home_district'] == dist]

        # Long-stay patients — likely present month was their return, actual
        # exposure was previous month; separate their risk information
        long_mask = d_pat['travel_over_14_nights'].fillna(0) == 1
        short_pat = d_pat[~long_mask]
        long_pat  = d_pat[long_mask]

        n_total  = len(d_pat)
        n_short  = len(short_pat)

        if n_short > 0:
            mean_risk      = float(short_pat['_risk'].mean())
            frac_high_risk = float((short_pat['_risk'] > 0.5).mean())
            n_high_risk    = float((short_pat['_risk'] > 0.5).sum())
        else:
            mean_risk = frac_high_risk = n_high_risk = 0.0

        longstay_mean_risk = float(long_pat['_risk'].mean()) if len(long_pat) > 0 else 0.0

        result[dist] = {
            'mean_risk':        mean_risk,
            'frac_high_risk':   frac_high_risk,
            'n_high_risk':      n_high_risk,
            'n_total':          float(n_total),
            'longstay_mean_risk': longstay_mean_risk,
        }
    return result


# ---------------------------------------------------------------------------
class FeatureScaler:
    """StandardScaler fitted on training months only."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, features_list: list):
        all_f = np.concatenate(features_list, axis=0)
        self.scaler.fit(all_f)
        self.is_fitted = True

    def transform(self, features: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")
        return self.scaler.transform(features).astype(np.float32)

    def fit_transform(self, features_list: list) -> list:
        self.fit(features_list)
        return [self.transform(f) for f in features_list]


def get_target_vector(monthly_counts: pd.DataFrame, year_month: str) -> np.ndarray:
    """
    Target: number of imported cases per node for `year_month`.
    Unguja districts → imported_cases.
    Mainland regions → travel_to_<region> from all Unguja districts (inflow).
    """
    nodes = get_all_nodes()
    node_to_idx = get_node_to_idx()
    targets = np.zeros(len(nodes), dtype=np.float32)
    month_data = monthly_counts[monthly_counts['year_month'] == year_month]

    for district in UNGUJA_DISTRICTS:
        idx = node_to_idx[district]
        row = month_data[month_data['home_district'] == district]
        if len(row):
            targets[idx] = float(row.iloc[0]['imported_cases'])

    for region in ALL_MAINLAND_REGIONS:
        idx = node_to_idx[region]
        col = f'travel_to_{region}'
        if col in month_data.columns:
            targets[idx] = float(month_data[col].sum())

    return targets
