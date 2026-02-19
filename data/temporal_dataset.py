"""
Temporal dataset: creates PyG graph snapshots and LOOCV splits.

Paper scope: May 2022 – Dec 2023 (20 months).
LOOCV: for each test month in Jul-Dec 2023, train on all prior months.

Key change vs previous version:
    Per-fold IndividualRiskScorer is fitted on training patients only,
    then individual risk-score features are computed for each training
    and test month using the fold's scorer — no leakage.
"""

import numpy as np
import torch
from torch_geometric.data import Data

from data.data_loader import build_master_dataset, UNGUJA_DISTRICTS
from data.graph_builder import build_monthly_edge_features, get_all_nodes
from data.feature_engineering import (
    build_node_features, build_risk_features_for_month,
    get_target_vector, FeatureScaler, FEATURE_NAMES, NUM_FEATURES
)
from models.risk_scorer import IndividualRiskScorer

# Paper scope: May 2022 – Dec 2023
ALL_MONTHS = [
    '2022-05', '2022-06', '2022-07', '2022-08', '2022-09', '2022-10',
    '2022-11', '2022-12', '2023-01', '2023-02', '2023-03', '2023-04',
    '2023-05', '2023-06', '2023-07', '2023-08', '2023-09',
    '2023-10', '2023-11', '2023-12'
]

# LOOCV test folds: last 6 months
LOOCV_TEST_MONTHS = ALL_MONTHS[14:]  # Jul 2023 to Dec 2023


def _make_graph(monthly_counts, rainfall, temperature, year_month,
                all_months, edge_index, edge_attr, risk_features=None):
    """Build a single PyG Data object for one month."""
    x = build_node_features(
        monthly_counts, rainfall, temperature,
        year_month, all_months,
        risk_features=risk_features,
    )
    y = get_target_vector(monthly_counts, year_month)
    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        year_month=year_month,
    )


def _build_risk_features_for_all_months(
    clinic_raw, scorer, months, prev_month_map
):
    """
    For each month in `months`, build risk features using the fitted scorer
    applied to the PREVIOUS month's patients (lag-1, no leakage).

    prev_month_map: dict {month -> previous_month_str or None}
    Returns: dict {month -> {district -> risk_feature_dict}}
    """
    all_risk = {}
    for ym in months:
        prev = prev_month_map.get(ym)
        if prev is None:
            # No previous month available — zero risk features
            all_risk[ym] = {d: {'mean_risk': 0, 'frac_high_risk': 0,
                                 'n_high_risk': 0, 'n_total': 0,
                                 'longstay_mean_risk': 0}
                             for d in UNGUJA_DISTRICTS}
        else:
            # Use patients from prev month to score (already has date as 'YYYY-MM')
            all_risk[ym] = build_risk_features_for_month(
                clinic_raw, scorer, prev, UNGUJA_DISTRICTS
            )
    return all_risk


def create_loocv_folds(data_dir: str = 'c:/malaria', lookback: int = 2):
    """
    Leave-one-month-out cross-validation on the last 6 months (Jul–Dec 2023).

    For each fold:
      1. Identify training months (all months before test month)
      2. Fit IndividualRiskScorer on training patients only
      3. Build risk features for ALL months using that scorer (lag-1)
      4. Build PyG graphs with risk features
      5. Fit FeatureScaler on training graphs only
      6. Scale all graphs

    Returns list of fold dicts:
        {test_month, train_graphs, test_graph, scaler, raw_data, scorer}
    """
    raw = build_master_dataset(data_dir)
    monthly_counts = raw['monthly_counts']
    rainfall       = raw['rainfall']
    temperature    = raw['temperature']
    clinic_raw     = raw['clinic_raw']

    # clinic_raw has 'date' as datetime — convert to 'YYYY-MM' string for matching
    clinic_raw = clinic_raw.copy()
    clinic_raw['date'] = clinic_raw['date'].dt.strftime('%Y-%m')

    months_in_data = sorted(monthly_counts['year_month'].unique())
    available = [m for m in ALL_MONTHS if m in months_in_data]

    # Static edge structure — reference month for consistent shapes
    ref_month = available[0]
    edge_index, edge_attr = build_monthly_edge_features(monthly_counts, ref_month)

    # Previous-month lookup
    prev_map = {m: available[available.index(m) - 1] if available.index(m) > 0 else None
                for m in available}

    print(f"  Graph: {len(get_all_nodes())} nodes, {NUM_FEATURES} features")
    print(f"  LOOCV folds: {len(LOOCV_TEST_MONTHS)}")

    folds = []

    for test_month in LOOCV_TEST_MONTHS:
        if test_month not in available:
            continue

        test_idx          = available.index(test_month)
        train_months_fold = available[:test_idx]

        if len(train_months_fold) < lookback + 1:
            continue

        print(f"    {test_month}: train={len(train_months_fold)} months")

        # ---- 1. Fit risk scorer on training patients only ----------------
        train_patients = clinic_raw[clinic_raw['date'].isin(train_months_fold)]
        scorer = IndividualRiskScorer(C=0.1)
        scorer.fit(train_patients)

        # ---- 2. Build risk features for training + test months -----------
        all_fold_months = train_months_fold + [test_month]
        risk_feats = _build_risk_features_for_all_months(
            clinic_raw, scorer, all_fold_months, prev_map
        )

        # ---- 3. Build raw graphs with risk features ----------------------
        raw_graphs = {}
        for ym in all_fold_months:
            raw_graphs[ym] = _make_graph(
                monthly_counts, rainfall, temperature,
                ym, available, edge_index, edge_attr,
                risk_features=risk_feats.get(ym),
            )

        train_gs = [raw_graphs[m] for m in train_months_fold]
        test_g   = raw_graphs[test_month]

        # ---- 4. Fit scaler on training features only ---------------------
        scaler = FeatureScaler()
        scaler.fit([g.x.numpy() for g in train_gs])

        def _scale(gs):
            out = []
            for g in gs:
                g2 = g.clone()
                g2.x = torch.tensor(scaler.transform(g.x.numpy()), dtype=torch.float32)
                out.append(g2)
            return out

        train_scaled = _scale(train_gs)
        test_scaled  = _scale([test_g])[0]

        folds.append({
            'test_month':   test_month,
            'train_graphs': train_scaled,
            'test_graph':   test_scaled,
            'scaler':       scaler,
            'scorer':       scorer,
            'raw_data':     raw,
        })

    return folds
