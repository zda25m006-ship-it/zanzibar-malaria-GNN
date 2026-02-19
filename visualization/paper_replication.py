
import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from data.data_loader import load_clinic_data

def replicate_paper_figures():
    """
    Generates figures matching Muller et al. (2025) style.
    Fig 3: Monthly Rainfall vs Imported Cases
    Fig 2: Spatial Incidence Heatmap (Actual)
    """
    os.makedirs('results', exist_ok=True)
    
    # --- Load Data ---
    print("Loading data...")
    clinic_df = load_clinic_data('c:/malaria/ZIM_clinic_data - ZIM_clinic_data.csv (1).csv')
    rain_df = pd.read_csv('c:/malaria/CHIRPS_rainfall_RAW_ONLY (1).csv')
    
    # --- Preprocess Cases ---
    clinic_df['month'] = clinic_df['date'].dt.to_period('M')
    # Filter for Imported Cases (Travel=1)
    imported = clinic_df[clinic_df['travel'] == 1].groupby('month').size().reset_index(name='cases')
    
    # --- Preprocess Rainfall ---
    rain_df['date'] = pd.to_datetime(rain_df['clinic_visit_week'])
    rain_df['month'] = rain_df['date'].dt.to_period('M')
    # Mean intensity across regions, then sum by month (to get total monthly rainfall index)
    # Alternatively: sum across regions? Paper says "Mainland rainfall".
    # We'll take the mean across regions (as an index) per month.
    monthly_rain = rain_df.groupby('month')['rainfall_lagged_mm'].mean().reset_index(name='rainfall')
    
    # Merge
    merged = pd.merge(imported, monthly_rain, on='month', how='inner')
    
    # --- Figure 3: Scatter Plot ---
    print("Generating Figure 3 (Rainfall vs Cases)...")
    plt.figure(figsize=(8, 6))
    sns.regplot(data=merged, x='rainfall', y='cases', scatter_kws={'s': 50, 'alpha': 0.7}, line_kws={'color': 'red'})
    plt.title('Monthly Imported Cases vs Lagged Mainland Rainfall', fontsize=14, fontweight='bold')
    plt.xlabel('Mainland Rainfall Index (mm)', fontsize=12)
    plt.ylabel('Imported Cases (Unguja)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/paper_fig3_replication.png', dpi=300)
    plt.close()
    
    # --- Figure 2: Spatial Heatmap ---
    print("Generating Figure 2 (Spatial Heatmap)...")
    # Group by district
    district_cases = clinic_df[clinic_df['travel'] == 1].groupby('home_district').size()
    
    # Approximate grid
    grid_map = {
        'Kaskazini A': (0, 0), 'North A': (0, 0),
        'Kaskazini B': (0, 1), 'North B': (0, 1),
        'Magharibi A': (1, 0), 'West A':  (1, 0),
        'Kati':        (1, 1), 'Central': (1, 1),
        'Magharibi B': (2, 0), 'West B':  (2, 0),
        'Kusini':      (2, 1), 'South':   (2, 1),
        'Mjini':       (3, 0), 'Urban':   (3, 0),
    }
    grid_shape = (4, 2)
    heatmap = np.full(grid_shape, np.nan)
    
    for district, count in district_cases.items():
        # Match fuzzy names if needed
        d_key = None
        for k in grid_map:
            if k in district:     d_key = k; break
        
        if d_key:
            r, c = grid_map[d_key]
            heatmap[r, c] = count
            
    plt.figure(figsize=(5, 8))
    plt.imshow(heatmap, cmap='YlOrRd', vmin=0)
    plt.colorbar(label='Total Imported Cases')
    
    # Add labels
    for district, count in district_cases.items():
        d_key = None
        for k in grid_map:
            if k in district: d_key = k; break
        if d_key:
            r, c = grid_map[d_key]
            plt.text(c, r, f"{district}\n{count}", ha='center', va='center', fontweight='bold')
            
    plt.title('Spatial Distribution of Imported Cases', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('results/paper_fig2_replication.png', dpi=300)
    plt.close()
    
    print("Done.")

if __name__ == "__main__":
    replicate_paper_figures()
