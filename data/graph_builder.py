"""
Graph construction for the malaria mobility network.
Builds spatial graphs where nodes = districts/regions and edges = travel connections.
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from data.data_loader import (
    UNGUJA_DISTRICTS, ALL_MAINLAND_REGIONS, RISK_CATEGORY,
    POPULATION_PROXY, REGION_COORDS
)


def get_all_nodes():
    """Return ordered list of all graph nodes."""
    return UNGUJA_DISTRICTS + ALL_MAINLAND_REGIONS


def get_node_to_idx():
    """Return dict mapping node name to index."""
    nodes = get_all_nodes()
    return {name: idx for idx, name in enumerate(nodes)}


def build_static_adjacency(monthly_counts: pd.DataFrame, threshold: int = 1):
    """
    Build a static adjacency matrix from aggregated travel data.
    Edges connect Unguja districts to mainland regions they have travel connections with.

    Args:
        monthly_counts: aggregated monthly data with travel_to_<region> columns
        threshold: minimum total travelers to create an edge

    Returns:
        edge_index: [2, E] tensor
        edge_weights: [E] tensor (total travel volume)
    """
    nodes = get_all_nodes()
    node_to_idx = get_node_to_idx()
    n = len(nodes)

    # Accumulate total travel between each Unguja district and mainland region
    edges = []
    weights = []

    for district in UNGUJA_DISTRICTS:
        d_idx = node_to_idx[district]
        district_data = monthly_counts[monthly_counts['home_district'] == district]

        for region in ALL_MAINLAND_REGIONS:
            col = f'travel_to_{region}'
            if col not in district_data.columns:
                continue
            total_travelers = district_data[col].sum()
            if total_travelers >= threshold:
                r_idx = node_to_idx[region]
                # Bidirectional edges
                edges.append([d_idx, r_idx])
                edges.append([r_idx, d_idx])
                weights.append(total_travelers)
                weights.append(total_travelers)

    # Add self-loops for all nodes
    for i in range(n):
        edges.append([i, i])
        weights.append(1.0)

    # Also add edges between neighboring Unguja districts
    unguja_neighbors = [
        ('Mjini', 'Magharibi A'), ('Mjini', 'Magharibi B'),
        ('Mjini', 'Kati'), ('Magharibi A', 'Magharibi B'),
        ('Magharibi A', 'Kaskazini A'), ('Kati', 'Kaskazini B'),
        ('Kati', 'Kusini'), ('Kaskazini A', 'Kaskazini B'),
    ]
    for d1, d2 in unguja_neighbors:
        i, j = node_to_idx[d1], node_to_idx[d2]
        edges.append([i, j])
        edges.append([j, i])
        weights.append(1.0)
        weights.append(1.0)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float32)

    return edge_index, edge_weight


def build_monthly_edge_features(monthly_counts: pd.DataFrame, year_month: str):
    """
    Build edge features for a specific month.

    Returns:
        edge_index: [2, E]
        edge_attr: [E, num_edge_features]
            Features: [travel_volume, prop_long_term, is_wet_season, risk_multiplier]
    """
    nodes = get_all_nodes()
    node_to_idx = get_node_to_idx()

    month_data = monthly_counts[monthly_counts['year_month'] == year_month]
    month_num = int(year_month.split('-')[1])
    is_wet = 1.0 if month_num in [3, 4, 5, 10, 11, 12] else 0.0

    edges = []
    edge_attrs = []

    for district in UNGUJA_DISTRICTS:
        d_idx = node_to_idx[district]
        district_data = month_data[month_data['home_district'] == district]

        if len(district_data) == 0:
            continue

        row = district_data.iloc[0]

        for region in ALL_MAINLAND_REGIONS:
            col = f'travel_to_{region}'
            if col not in district_data.columns:
                continue
            vol = row.get(col, 0)
            if vol > 0 or True:  # Include all potential edges
                r_idx = node_to_idx[region]
                risk = RISK_CATEGORY.get(region, 1)

                feat = [
                    float(vol),
                    0.0,         # prop_long_term removed from data
                    is_wet,
                    float(risk),
                ]

                # Bidirectional
                edges.append([d_idx, r_idx])
                edge_attrs.append(feat)
                edges.append([r_idx, d_idx])
                edge_attrs.append(feat)

    # Self-loops
    for i in range(len(nodes)):
        edges.append([i, i])
        edge_attrs.append([0.0, 0.0, is_wet, 0.0])

    # Unguja neighbor edges
    unguja_neighbors = [
        ('Mjini', 'Magharibi A'), ('Mjini', 'Magharibi B'),
        ('Mjini', 'Kati'), ('Magharibi A', 'Magharibi B'),
        ('Magharibi A', 'Kaskazini A'), ('Kati', 'Kaskazini B'),
        ('Kati', 'Kusini'), ('Kaskazini A', 'Kaskazini B'),
    ]
    for d1, d2 in unguja_neighbors:
        i, j = node_to_idx[d1], node_to_idx[d2]
        edges.append([i, j])
        edge_attrs.append([1.0, 0.0, is_wet, 0.0])
        edges.append([j, i])
        edge_attrs.append([1.0, 0.0, is_wet, 0.0])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)

    return edge_index, edge_attr


def compute_geographic_distance(region1: str, region2: str) -> float:
    """Compute approximate distance in degrees between two regions."""
    if region1 not in REGION_COORDS or region2 not in REGION_COORDS:
        return 10.0  # default large distance
    lat1, lon1 = REGION_COORDS[region1]
    lat2, lon2 = REGION_COORDS[region2]
    return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)
