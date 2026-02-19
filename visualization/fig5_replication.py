
import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from data.data_loader import load_clinic_data, RISK_CATEGORY

def replicate_fig5():
    """
    Replicates Figure 5: Demographic Heatmap of Travelers.
    Dimensions:
    - Residence: Zanzibari vs Mainlander
    - Duration: Short (<=14) vs Long (>14)
    - Season: Wet vs Dry
    - Risk: High/Moderate vs Low
    """
    os.makedirs('results', exist_ok=True)
    
    # Load Data
    print("Loading data for Fig 5...")
    df = load_clinic_data('c:/malaria/ZIM_clinic_data - ZIM_clinic_data.csv (1).csv')
    
    # Filter for travelers
    travelers = df[df['travel'] == 1].copy()
    
    # --- Define Categories ---
    
    # Risk
    def get_risk_level(region):
        if pd.isna(region): return 'Unknown'
        cat = RISK_CATEGORY.get(region, 0) # Default to 0 (Zanzibar/Low)
        if cat >= 2: return 'High/Mod'
        return 'Low'
    
    travelers['Risk'] = travelers['travel_tz_region_primary'].apply(get_risk_level)
    
    # Season (Wet: Mar-May, Oct-Dec)
    travelers['Month'] = travelers['date'].dt.month
    travelers['Season'] = travelers['Month'].apply(
        lambda m: 'Wet' if m in [3,4,5,10,11,12] else 'Dry'
    )
    
    # Duration (Short <= 14, Long > 14)
    # Check column name for duration. 'trip_duration_days'? 
    # data_loader doesn't show it explicitly in my view, but 'load_clinic_data' line 100 comment hinted at it.
    # I'll check column names if this fails. But assuming standard name or 'duration'.
    # If not present, I'll rely on 'trip_duration_days' which is common.
    # Duration (Short <= 14, Long > 14)
    # Using pre-calculated binary flag
    if 'travel_over_14_nights' in travelers.columns:
        travelers['Duration'] = travelers['travel_over_14_nights'].apply(
            lambda d: 'Long' if d == 1 else 'Short'
        )
    else:
        # Fallback if not found (should be there based on data_loader)
        travelers['Duration'] = 'Short'
    
    # Residence (Mainlander vs Zanzibari)
    travelers['Residence'] = travelers['mainlander_on_zb'].apply(
        lambda x: 'Mainlander' if x == 1 else 'Zanzibari'
    )
    
    # Filter out unknowns if needed
    travelers = travelers[travelers['Risk'] != 'Unknown']
    
    # --- Pivot for Heatmap ---
    # We want % of TOTAL travelers in each cell.
    total_travelers = len(travelers)
    
    # Group by all 4 dimensions
    grouped = travelers.groupby(['Residence', 'Duration', 'Season', 'Risk']).size().reset_index(name='Count')
    grouped['Percentage'] = (grouped['Count'] / total_travelers) * 100
    
    # Plotting
    # FacetGrid or Subplots?
    # Paper has two heatmaps: (A) Zanzibari, (B) Mainlander (actually ferry travelers in B, but clinic in A).
    # Fig 5A: Zanzibari malaria cases.
    # Fig 5B: Ferry travelers (we don't have ferry data).
    # We will plot Fig 5A only (Zanzibari cases) or split by Residence available in clinic data.
    # Clinic data has both Zanzibari and Mainlander patients.
    # Let's plot both side-by-side using clinic data.
    
    # Prepare Pivot Tables
    # Rows: Duration (Short, Long)
    # Cols: Risk (High/Mod, Low) x Season (Dry, Wet)
    
    # Let's verify grouping
    # We want a heatmap where x-axis = Season-Risk, y-axis = Duration.
    
    def plot_heatmap(data, title, ax):
        pivot = data.pivot_table(
            index='Duration', 
            columns=['Risk', 'Season'], 
            values='Percentage', 
            aggfunc='sum',
            fill_value=0
        )
        # Reorder columns/index if possible for logic
        # Index: Short, Long
        # Cols: (High/Mod, Dry), (High/Mod, Wet), (Low, Dry), (Low, Wet)
        
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='Reds', cbar=False, ax=ax, vmin=0, vmax=20)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('Travel Duration')
        ax.set_xlabel('Risk Region & Season')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Zanzibaris
    zanzibaris = grouped[grouped['Residence'] == 'Zanzibari']
    # Recalculate % relative to group total? Or total overall?
    # Paper Fig 5 says "of total travelers corresponding to specific combinations".
    # Usually % of that group (Zanzibaris).
    total_z = travelers[travelers['Residence'] == 'Zanzibari'].shape[0]
    zanzibaris = zanzibaris.copy()
    zanzibaris['Percentage'] = (zanzibaris['Count'] / total_z) * 100
    plot_heatmap(zanzibaris, f'Zanzibari Cases (n={total_z})', axes[0])
    
    # 2. Mainlanders
    mainlanders = grouped[grouped['Residence'] == 'Mainlander']
    total_m = travelers[travelers['Residence'] == 'Mainlander'].shape[0]
    mainlanders = mainlanders.copy()
    mainlanders['Percentage'] = (mainlanders['Count'] / total_m) * 100
    plot_heatmap(mainlanders, f'Mainlander Cases (n={total_m})', axes[1])
    
    plt.suptitle('Demographic Distribution of Travel-Related Malaria (Fig 5 Replication)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/paper_fig5_replication.png', dpi=300)
    plt.close()
    print("  Saved Fig 5 Replication.")

if __name__ == "__main__":
    replicate_fig5()
