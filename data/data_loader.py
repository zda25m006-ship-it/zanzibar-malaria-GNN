"""
Data loader for malaria importation prediction.
Loads clinic, rainfall, and temperature datasets and aggregates to monthly graph-level data.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# Regions that have CHIRPS rainfall data
RAINFALL_REGIONS = [
    'Geita', 'Kagera', 'Mara', 'Morogoro', 'Mwanza',
    'Pwani', 'Shinyanga', 'Simiyu', 'Tanga'
]

# Unguja (Zanzibar) home districts
UNGUJA_DISTRICTS = [
    'Mjini', 'Magharibi A', 'Magharibi B', 'Kati',
    'Kaskazini A', 'Kaskazini B', 'Kusini'
]

# Additional mainland regions from clinic data (beyond the 9 with rainfall)
ADDITIONAL_MAINLAND_REGIONS = [
    'Dar Es Salaam', 'Tabora', 'Dodoma', 'Kigoma', 'Lindi',
    'Mtwara', 'Ruvuma', 'Rukwa', 'Singida', 'Arusha',
    'Songwe', 'Mbeya', 'Kilimanjaro', 'Manyara', 'Katavi'
]

# All mainland regions
ALL_MAINLAND_REGIONS = RAINFALL_REGIONS + ADDITIONAL_MAINLAND_REGIONS

# Approximate lat/lon centroids for mainland regions (for temperature mapping)
REGION_COORDS = {
    'Geita': (-2.87, 32.23), 'Kagera': (-1.55, 31.26), 'Mara': (-1.75, 34.05),
    'Morogoro': (-6.82, 37.66), 'Mwanza': (-2.52, 32.90), 'Pwani': (-7.32, 38.82),
    'Shinyanga': (-3.66, 33.42), 'Simiyu': (-3.03, 34.15), 'Tanga': (-5.07, 39.10),
    'Dar Es Salaam': (-6.79, 39.28), 'Tabora': (-5.08, 32.83),
    'Dodoma': (-6.17, 35.75), 'Kigoma': (-4.88, 29.63), 'Lindi': (-10.00, 39.71),
    'Mtwara': (-10.27, 40.18), 'Ruvuma': (-10.68, 35.68), 'Rukwa': (-7.98, 31.61),
    'Singida': (-4.82, 34.74), 'Arusha': (-3.37, 36.68), 'Songwe': (-8.92, 33.27),
    'Mbeya': (-8.90, 33.46), 'Kilimanjaro': (-3.08, 37.35),
    'Manyara': (-4.58, 35.83), 'Katavi': (-6.35, 31.27),
    # Unguja districts (approximate)
    'Mjini': (-6.16, 39.19), 'Magharibi A': (-6.13, 39.15),
    'Magharibi B': (-6.20, 39.17), 'Kati': (-6.08, 39.25),
    'Kaskazini A': (-5.88, 39.28), 'Kaskazini B': (-5.93, 39.22),
    'Kusini': (-6.35, 39.43),
}

# Malaria risk categories from Tanzania's 2021-2025 National Malaria Strategic Plan
RISK_CATEGORY = {
    'Geita': 3, 'Kagera': 3, 'Mara': 2, 'Morogoro': 2, 'Mwanza': 2,
    'Pwani': 2, 'Shinyanga': 2, 'Simiyu': 2, 'Tanga': 2,
    'Dar Es Salaam': 1, 'Tabora': 2, 'Dodoma': 1, 'Kigoma': 3,
    'Lindi': 2, 'Mtwara': 2, 'Ruvuma': 2, 'Rukwa': 2, 'Singida': 1,
    'Arusha': 1, 'Songwe': 2, 'Mbeya': 2, 'Kilimanjaro': 1,
    'Manyara': 1, 'Katavi': 3,
    # Unguja districts -- low/near-elimination
    'Mjini': 0, 'Magharibi A': 0, 'Magharibi B': 0, 'Kati': 0,
    'Kaskazini A': 0, 'Kaskazini B': 0, 'Kusini': 0,
}

# Approximate population (relative, for node features)
POPULATION_PROXY = {
    'Mjini': 600000, 'Magharibi A': 100000, 'Magharibi B': 180000,
    'Kati': 150000, 'Kaskazini A': 90000, 'Kaskazini B': 80000, 'Kusini': 120000,
    'Dar Es Salaam': 5400000, 'Tanga': 2100000, 'Morogoro': 3000000,
    'Pwani': 1100000, 'Mwanza': 3100000, 'Geita': 1800000, 'Kagera': 2700000,
    'Shinyanga': 1900000, 'Simiyu': 1800000, 'Mara': 2000000,
    'Tabora': 2600000, 'Dodoma': 2200000, 'Kigoma': 2100000,
    'Lindi': 900000, 'Mtwara': 1300000, 'Ruvuma': 1600000, 'Rukwa': 1000000,
    'Singida': 1500000, 'Arusha': 1900000, 'Songwe': 1000000,
    'Mbeya': 2700000, 'Kilimanjaro': 1800000, 'Manyara': 1600000, 'Katavi': 600000,
}


def load_clinic_data(path: str) -> pd.DataFrame:
    """Load and preprocess clinic survey data."""
    df = pd.read_csv(path)

    # Parse date (format: YYYY-MM)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m')
    df['year_month'] = df['date'].dt.to_period('M')

    # === CAP AT DEC 2023 (paper scope: May 2022 – Dec 2023) ===
    df = df[df['date'] <= pd.Timestamp('2023-12-31')].copy()

    # Clean home district - keep only known Unguja districts
    df.loc[~df['home_district'].isin(UNGUJA_DISTRICTS), 'home_district'] = None
    df = df.dropna(subset=['home_district'])

    # Binary travel indicator
    df['travel'] = df['travel'].fillna(0).astype(int)

    # Clean travel region
    df['travel_tz_region_primary'] = df['travel_tz_region_primary'].replace('NA', np.nan)
    df.loc[df['travel_tz_region_primary'] == 'Unknown', 'travel_tz_region_primary'] = np.nan

    # Travel duration flags
    df['travel_over_4_nights'] = pd.to_numeric(df['travel_over_4_nights'], errors='coerce')
    df['travel_over_14_nights'] = pd.to_numeric(df['travel_over_14_nights'], errors='coerce')

    # Mainlander flag (mainland resident visiting Zanzibar — distinct from travel outflow)
    df['mainlander_on_zb'] = pd.to_numeric(df['mainlander_on_zb'], errors='coerce').fillna(0).astype(int)

    # Season classification
    df['month'] = df['date'].dt.month
    df['is_rainy'] = df['month'].isin([3, 4, 5, 10, 11, 12]).astype(int)

    return df


def aggregate_monthly_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate clinic data to monthly counts per Unguja district.

    NOTE: `travel_outflow` is intentionally removed — it equals `imported_cases`
    and using it as a feature would be target leakage.

    Returns: DataFrame with columns:
      year_month, home_district, total_cases, imported_cases, local_cases,
      imported_frac, n_mainlanders, is_rainy, travel_to_<region>...
    """
    records = []

    for (ym, district), group in df.groupby(['year_month', 'home_district']):
        total = len(group)
        imported = group['travel'].sum()
        local = total - imported

        # Travelers and their destinations
        travelers = group[group['travel'] == 1]
        region_counts = travelers['travel_tz_region_primary'].value_counts().to_dict()

        record = {
            'year_month': ym,
            'home_district': district,
            'total_cases': int(total),
            'imported_cases': int(imported),
            'local_cases': int(local),
            # Fraction imported — useful lagged feature (scale-free, no count leakage)
            'imported_frac': imported / max(total, 1),
            'n_mainlanders': int(group['mainlander_on_zb'].sum()),
            'is_rainy': int(group['is_rainy'].iloc[0]),
        }

        # Per-region travel counts (how many from this district went to each region)
        for region in ALL_MAINLAND_REGIONS:
            record[f'travel_to_{region}'] = int(region_counts.get(region, 0))

        records.append(record)

    result = pd.DataFrame(records)
    result['year_month'] = result['year_month'].astype(str)
    return result


def load_rainfall_data(path: str) -> pd.DataFrame:
    """Load CHIRPS rainfall data and aggregate to monthly."""
    df = pd.read_csv(path)
    df['clinic_visit_week'] = pd.to_datetime(df['clinic_visit_week'])
    df['year_month'] = df['clinic_visit_week'].dt.to_period('M').astype(str)

    # Aggregate weekly to monthly (mean of lagged rainfall)
    monthly = df.groupby(['region', 'year_month']).agg(
        rainfall_mm=('rainfall_lagged_mm', 'mean')
    ).reset_index()

    return monthly


def load_temperature_data(path: str) -> pd.DataFrame:
    """
    Load POWER temperature data and map grid points to regions.
    Returns monthly temperature per region.
    """
    # Read file, skip header lines
    with open(path, 'r') as f:
        lines = f.readlines()

    # Find the data start (after -END HEADER-)
    header_end = 0
    for i, line in enumerate(lines):
        if '-END HEADER-' in line:
            header_end = i + 1
            break

    df = pd.read_csv(path, skiprows=header_end)
    df.columns = df.columns.str.strip()

    month_cols = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                  'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

    # Map each grid point to nearest region
    all_regions = list(REGION_COORDS.keys())
    region_temps = []

    for region in all_regions:
        if region not in REGION_COORDS:
            continue
        rlat, rlon = REGION_COORDS[region]

        # Find nearest grid point
        distances = np.sqrt((df['LAT'] - rlat)**2 + (df['LON'] - rlon)**2)
        for year in df['YEAR'].unique():
            year_mask = df['YEAR'] == year
            year_distances = distances[year_mask]
            if len(year_distances) == 0:
                continue
            nearest_idx = year_distances.idxmin()
            row = df.loc[nearest_idx]

            for m_idx, m_col in enumerate(month_cols):
                ym_str = f"{int(row['YEAR'])}-{m_idx+1:02d}"
                temp_val = row[m_col]
                if temp_val == -999:
                    temp_val = np.nan
                region_temps.append({
                    'region': region,
                    'year_month': ym_str,
                    'temperature': temp_val
                })

    return pd.DataFrame(region_temps)


def build_master_dataset(data_dir: str) -> dict:
    """
    Build the complete dataset combining all data sources.

    Returns dict with:
        - 'monthly_counts': monthly aggregated clinic data
        - 'rainfall': monthly rainfall per region
        - 'temperature': monthly temperature per region
        - 'clinic_raw': raw clinic DataFrame
    """
    data_dir = Path(data_dir)

    print("Loading clinic data...")
    clinic = load_clinic_data(data_dir / 'ZIM_clinic_data - ZIM_clinic_data.csv (1).csv')

    print("Aggregating to monthly counts...")
    monthly = aggregate_monthly_counts(clinic)

    print("Loading rainfall data...")
    rainfall = load_rainfall_data(data_dir / 'CHIRPS_rainfall_RAW_ONLY (1).csv')

    print("Loading temperature data...")
    temperature = load_temperature_data(data_dir / 'POWER_Regional_Monthly_2022_2023.csv')

    return {
        'monthly_counts': monthly,
        'rainfall': rainfall,
        'temperature': temperature,
        'clinic_raw': clinic,
    }


if __name__ == '__main__':
    data = build_master_dataset('c:/malaria')
    print(f"\nMonthly counts shape: {data['monthly_counts'].shape}")
    print(f"Rainfall shape: {data['rainfall'].shape}")
    print(f"Temperature shape: {data['temperature'].shape}")
    print(f"Raw clinic shape: {data['clinic_raw'].shape}")
    print(f"\nMonthly counts columns: {data['monthly_counts'].columns.tolist()}")
    print(f"\nSample monthly row:\n{data['monthly_counts'].iloc[0]}")
