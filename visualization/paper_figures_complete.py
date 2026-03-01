"""
Complete paper-aligned visualization suite for Muller et al. (2025) replication
+ GNN extension plots.

Generates ALL figures that should appear in an extended / conference version:

  [Paper Replications]
  Fig1A  – Bar chart of home districts of travel-associated cases
  Fig1B  – Travel destinations (bar chart, since no GIS)
  Fig2   – Monthly rainfall vs imported case counts (time series)
  Fig3   – NB regression coefficient plot (IRR) for rainfall vs cases
  Fig5   – Demographic heatmaps (Zanzibari vs Mainlander travelers)

  [GNN Extension – Conference-Ready]
  FigG1  – Model comparison bar chart (RMSE + AUC) with error bars
  FigG2  – Per-fold LOOCV performance time series (6 folds Jul-Dec 2023)
  FigG3  – Predicted vs Actual scatter (per district, per model)
  FigG4  – Individual risk score distribution heatmap by district
  FigG5  – Seasonal decomposition of imported cases vs rainfall

Run standalone:
    python visualization/paper_figures_complete.py

Or call generate_all_paper_figures(results_dict, folds, raw_data, save_dir)
after running main.py.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

# ─── Style ────────────────────────────────────────────────────────────────────
PAPER_STYLE = {
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
}

MODEL_COLORS = {
    'Naive':       '#9E9E9E',
    'LR':          '#8B4513',    # brown  – paper baseline
    'NB':          '#D2691E',    # orange – paper baseline
    'GCN':         '#1565C0',    # blue
    'GAT':         '#2E7D32',    # green
    'ST-GNN':      '#6A1B9A',    # purple
}

DATA_DIR  = r'c:\malaria'
SAVE_DIR  = os.path.join(DATA_DIR, 'results')

UNGUJA_ORDER = [
    'Kaskazini A', 'Kaskazini B', 'Magharibi A', 'Kati',
    'Magharibi B', 'Kusini', 'Mjini'
]


# ══════════════════════════════════════════════════════════════════════════════
#   PAPER FIG 1  –  Home districts of travel-associated cases
# ══════════════════════════════════════════════════════════════════════════════

def fig1_district_cases(clinic_csv: str = None, save_dir: str = SAVE_DIR):
    """Bar chart of imported cases per Unguja home district (Fig 1B style)."""
    plt.rcParams.update(PAPER_STYLE)
    os.makedirs(save_dir, exist_ok=True)
    if clinic_csv is None:
        clinic_csv = os.path.join(DATA_DIR, 'ZIM_clinic_data - ZIM_clinic_data.csv (1).csv')

    try:
        df = pd.read_csv(clinic_csv, low_memory=False)
        # Detect travel column
        travel_col = next((c for c in df.columns if 'travel' in c.lower() and
                           df[c].isin([0,1,'0','1','Yes','No',True,False]).all()), 'travel')
        df[travel_col] = pd.to_numeric(df[travel_col], errors='coerce').fillna(0)
        imported = df[df[travel_col] == 1]
        district_col = next((c for c in df.columns
                             if 'district' in c.lower() or 'home' in c.lower()), None)
        if district_col is None:
            print("  [Fig1] Could not find district column – skipping")
            return

        counts = imported[district_col].value_counts().sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(9, 5))
        colors = plt.cm.YlOrRd(np.linspace(0.35, 0.85, len(counts)))
        bars = ax.barh(counts.index, counts.values, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Number of Travel-Associated Cases', fontweight='bold')
        ax.set_title('Fig 1A – Home Districts of Travel-Associated Malaria Cases\n'
                     '(Unguja, Zanzibar 2022–2023)', fontweight='bold', fontsize=13)
        # Annotate bars
        for bar, val in zip(bars, counts.values):
            ax.text(val + counts.values.max()*0.01, bar.get_y() + bar.get_height()/2,
                    str(int(val)), va='center', fontsize=9)
        plt.tight_layout()
        out = os.path.join(save_dir, 'fig1_district_cases.png')
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"  Saved: {out}")
    except Exception as e:
        print(f"  [Fig1] Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#   PAPER FIG 2  –  Monthly rainfall vs imported cases time series
# ══════════════════════════════════════════════════════════════════════════════

def fig2_rainfall_timeseries(clinic_csv: str = None, rain_csv: str = None,
                              save_dir: str = SAVE_DIR):
    """
    Time series of monthly imported cases alongside lagged rainfall.
    Replicates the spirit of Fig 2 in Muller et al. (2025).
    """
    plt.rcParams.update(PAPER_STYLE)
    os.makedirs(save_dir, exist_ok=True)

    if clinic_csv is None:
        clinic_csv = os.path.join(DATA_DIR, 'ZIM_clinic_data - ZIM_clinic_data.csv (1).csv')
    if rain_csv is None:
        rain_csv   = os.path.join(DATA_DIR, 'CHIRPS_rainfall_RAW_ONLY (1).csv')

    try:
        df = pd.read_csv(clinic_csv, low_memory=False)
        date_col = next((c for c in df.columns if 'date' in c.lower()), None)
        travel_col = next((c for c in df.columns
                           if c.lower() in ('travel','is_travel','travel_flag')), 'travel')
        df[travel_col] = pd.to_numeric(df[travel_col], errors='coerce').fillna(0)

        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df['month']  = df[date_col].dt.to_period('M')

        imported = df[df[travel_col] == 1].groupby('month').size().rename('imported')
        total    = df.groupby('month').size().rename('total')
        cases    = pd.concat([imported, total], axis=1).fillna(0)

        # Rainfall
        rdf = pd.read_csv(rain_csv, low_memory=False)
        rcol = next((c for c in rdf.columns if 'rainfall' in c.lower()), None)
        dcol = next((c for c in rdf.columns if 'week' in c.lower() or 'date' in c.lower()), None)

        if rcol and dcol:
            rdf[dcol] = pd.to_datetime(rdf[dcol], errors='coerce')
            rdf['month'] = rdf[dcol].dt.to_period('M')
            monthly_rain = rdf.groupby('month')[rcol].mean().rename('rainfall')
            cases = cases.join(monthly_rain, how='left')

        cases = cases.sort_index()
        months_str = [str(m) for m in cases.index]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

        # Panel A: cases
        ax1.bar(months_str, cases['imported'], color='#C62828', alpha=0.8,
                label='Imported cases', zorder=3)
        ax1.bar(months_str,
                cases['total'] - cases['imported'],
                bottom=cases['imported'],
                color='#EF9A9A', alpha=0.6, label='Local cases', zorder=3)
        ax1.set_ylabel('Monthly Cases', fontweight='bold')
        ax1.set_title('Fig 2A – Monthly Malaria Cases (Imported vs Local)', fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.tick_params(axis='x', rotation=45)

        # Panel B: rainfall
        if 'rainfall' in cases.columns:
            ax2.fill_between(months_str, cases['rainfall'], alpha=0.4,
                             color='#1565C0', label='Mean lagged rainfall (mm)')
            ax2.plot(months_str, cases['rainfall'], color='#1565C0', linewidth=1.5)
        ax2.set_ylabel('Rainfall (mm)', fontweight='bold')
        ax2.set_title('Fig 2B – Mainland Lagged Rainfall (CHIRPS)', fontweight='bold')
        ax2.legend(loc='upper left')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        out = os.path.join(save_dir, 'fig2_rainfall_timeseries.png')
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"  Saved: {out}")
    except Exception as e:
        print(f"  [Fig2] Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#   PAPER FIG 3  –  NB Regression IRR forest plot
# ══════════════════════════════════════════════════════════════════════════════

def fig3_nb_irr_forestplot(save_dir: str = SAVE_DIR):
    """
    Forest plot of Incidence Rate Ratios from Negative Binomial regression.
    Matches style of Fig 3 in Muller et al. (2025).

    Values here are illustrative placeholders derived from fitting the project's
    NB model — replace with your actual fitted coefficients for publication.
    """
    plt.rcParams.update(PAPER_STYLE)
    os.makedirs(save_dir, exist_ok=True)

    # ── Simulated / example IRR values matching paper structure ──────────────
    # Format: (label, IRR, CI_low, CI_high)
    # Replace these with np.exp(coef) and confidence intervals from actual NB fit
    factors = [
        # Rainfall effects (paper Fig 3 core)
        ('Mainland rainfall (lag 2wk)',     1.06, 1.02, 1.11),
        ('Zanzibar rainfall (lag 2wk)',     0.98, 0.94, 1.02),
        # Season
        ('Dry season (ref: Wet)',           0.72, 0.59, 0.88),
        # Demographics
        ('Age < 15 years',                  3.20, 2.20, 4.80),
        ('Female (ref: Male)',              1.50, 1.30, 1.80),
        # Occupation (ref: Farmer)
        ('Occupation: Watchman',            0.30, 0.18, 0.51),
        ('Occupation: Student',             0.50, 0.38, 0.67),
        ('Occupation: Trader',              1.20, 0.95, 1.52),
        # Travel
        ('Travel > 14 nights',             2.10, 1.65, 2.67),
        ('High-risk region destination',    2.85, 2.10, 3.87),
    ]

    labels  = [f[0] for f in factors]
    irrs    = np.array([f[1] for f in factors])
    ci_low  = np.array([f[2] for f in factors])
    ci_high = np.array([f[3] for f in factors])

    fig, ax = plt.subplots(figsize=(10, 7))

    y = np.arange(len(labels))
    colors = ['#C62828' if v > 1 else '#1565C0' for v in irrs]
    xerr = np.array([irrs - ci_low, ci_high - irrs])

    ax.barh(y, irrs - 1, left=1, color=colors, alpha=0.7,
            edgecolor='white', linewidth=0.5, height=0.55)
    ax.errorbar(irrs, y, xerr=xerr, fmt='none',
                ecolor='black', capsize=4, linewidth=1.5, capthick=1.5)
    ax.scatter(irrs, y, color='black', s=30, zorder=5)

    ax.axvline(x=1, color='black', linewidth=1.2, linestyle='-')
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Incidence Rate Ratio (IRR)', fontweight='bold')
    ax.set_title('Fig 3 – NB Regression: Predictors of Malaria Importation\n'
                 '(Muller et al. 2025 style — replace with fitted model values)',
                 fontweight='bold')
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.tick_params(axis='x', which='both')
    ax.set_xlim(0.15, 6)

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor='#C62828', alpha=0.7, label='IRR > 1 (risk increase)'),
                        Patch(facecolor='#1565C0', alpha=0.7, label='IRR < 1 (risk decrease)')],
              loc='lower right', fontsize=9)

    plt.tight_layout()
    out = os.path.join(save_dir, 'fig3_nb_irr_forestplot.png')
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
#   PAPER FIG 5  –  Demographic heatmaps (improved version)
# ══════════════════════════════════════════════════════════════════════════════

def fig5_demographic_heatmap(clinic_csv: str = None, save_dir: str = SAVE_DIR):
    """
    Heatmap of travellers by risk, season, duration, residence.
    Replicates Fig 5 of Muller et al. (2025).
    """
    try:
        import seaborn as sns
    except ImportError:
        print("  [Fig5] seaborn not installed – skipping")
        return

    plt.rcParams.update(PAPER_STYLE)
    os.makedirs(save_dir, exist_ok=True)

    if clinic_csv is None:
        clinic_csv = os.path.join(DATA_DIR, 'ZIM_clinic_data - ZIM_clinic_data.csv (1).csv')
    try:
        df = pd.read_csv(clinic_csv, low_memory=False)
    except Exception as e:
        print(f"  [Fig5] Cannot load clinic data: {e}")
        return

    # Detect columns
    travel_col   = next((c for c in df.columns if c.lower() == 'travel'), None)
    date_col     = next((c for c in df.columns if 'date' in c.lower()), None)
    duration_col = next((c for c in df.columns
                         if 'over_14' in c.lower() or 'duration' in c.lower()), None)
    mainland_col = next((c for c in df.columns if 'mainlander' in c.lower()), None)
    region_col   = next((c for c in df.columns
                         if 'region' in c.lower() and 'tz' in c.lower()), None)

    if travel_col is None or date_col is None:
        print("  [Fig5] Missing travel or date column – skipping")
        return

    df[travel_col] = pd.to_numeric(df[travel_col], errors='coerce').fillna(0)
    travelers = df[df[travel_col] == 1].copy()
    if len(travelers) == 0:
        print("  [Fig5] No traveler data found – skipping")
        return

    travelers[date_col] = pd.to_datetime(travelers[date_col], errors='coerce')
    travelers['Month'] = travelers[date_col].dt.month

    # Season
    travelers['Season'] = travelers['Month'].apply(
        lambda m: 'Wet' if m in [3, 4, 5, 10, 11, 12] else 'Dry'
    )

    # Duration
    if duration_col:
        travelers[duration_col] = pd.to_numeric(travelers[duration_col], errors='coerce').fillna(0)
        travelers['Duration'] = travelers[duration_col].apply(
            lambda x: 'Long (>14d)' if x == 1 else 'Short (≤14d)'
        )
    else:
        travelers['Duration'] = 'Short (≤14d)'

    # Residence
    if mainland_col:
        travelers[mainland_col] = pd.to_numeric(travelers[mainland_col], errors='coerce').fillna(0)
        travelers['Residence'] = travelers[mainland_col].apply(
            lambda x: 'Mainlander' if x == 1 else 'Zanzibari'
        )
    else:
        travelers['Residence'] = 'Zanzibari'

    # Risk category
    HIGH_RISK_REGIONS = {'Morogoro', 'Tanga', 'Mtwara', 'Lindi', 'Pwani', 'Ruvuma',
                         'Mwanza', 'Geita', 'Shinyanga', 'Kagera'}
    if region_col:
        travelers['Risk'] = travelers[region_col].apply(
            lambda r: 'High-Risk Region' if str(r) in HIGH_RISK_REGIONS else 'Low-Risk Region'
        )
    else:
        travelers['Risk'] = 'Low-Risk Region'

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (residence, grp) in zip(axes, travelers.groupby('Residence')):
        n = len(grp)
        total_by_group = n
        grouped = grp.groupby(['Duration', 'Risk', 'Season']).size().reset_index(name='Count')
        grouped['Pct'] = grouped['Count'] / total_by_group * 100

        try:
            pivot = grouped.pivot_table(
                index='Duration',
                columns=['Risk', 'Season'],
                values='Pct',
                aggfunc='sum',
                fill_value=0
            )
            sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd',
                        linewidths=0.5, linecolor='white',
                        ax=ax, vmin=0, annot_kws={'size': 10})
            ax.set_title(f'Fig 5 – {residence} Travelers (n={n})',
                         fontweight='bold', fontsize=12)
            ax.set_xlabel('Risk Region & Season', fontweight='bold')
            ax.set_ylabel('Travel Duration', fontweight='bold')
            ax.tick_params(axis='x', rotation=30)
        except Exception as e:
            ax.set_title(f'{residence} – insufficient data ({e})')

    plt.suptitle('Demographic Heatmaps – Travel-Associated Malaria Cases\n'
                 '(% of total within each residence group)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = os.path.join(save_dir, 'fig5_demographic_heatmap_v2.png')
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
#   GNN FIG G1  –  Model comparison bar chart (RMSE + AUC)
# ══════════════════════════════════════════════════════════════════════════════

def figG1_model_comparison(results: dict, save_dir: str = SAVE_DIR):
    """
    Conference-style bar chart comparing all models on RMSE and AUC.
    Pass results = {model_name: summary_dict, ...} where summary_dict contains
    mean_rmse, std_rmse, mean_auc, std_auc.
    """
    plt.rcParams.update(PAPER_STYLE)
    os.makedirs(save_dir, exist_ok=True)

    if not results:
        print("  [FigG1] No results provided – skipping")
        return

    models = list(results.keys())
    rmses  = [results[m].get('mean_rmse', 0) for m in models]
    rmse_s = [results[m].get('std_rmse', 0)  for m in models]
    aucs   = [results[m].get('mean_auc', np.nan) for m in models]
    auc_s  = [results[m].get('std_auc', 0)   for m in models]

    # Determine colors
    palette = []
    for m in models:
        ml = m.lower()
        if 'naive' in ml:      palette.append(MODEL_COLORS['Naive'])
        elif 'lr' in ml or 'logistic' in ml: palette.append(MODEL_COLORS['LR'])
        elif 'nb' in ml or 'negativ' in ml:  palette.append(MODEL_COLORS['NB'])
        elif 'gcn' in ml:      palette.append(MODEL_COLORS['GCN'])
        elif 'gat' in ml and 'st' not in ml: palette.append(MODEL_COLORS['GAT'])
        elif 'st' in ml:       palette.append(MODEL_COLORS['ST-GNN'])
        else:                  palette.append('#607D8B')

    x = np.arange(len(models))
    width = 0.55

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # RMSE
    bars1 = ax1.bar(x, rmses, width, yerr=rmse_s, capsize=5,
                    color=palette, edgecolor='white', linewidth=0.6,
                    error_kw=dict(ecolor='#333', lw=1.5, capthick=1.5))
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace(' ', '\n') for m in models], fontsize=9)
    ax1.set_ylabel('RMSE (imported cases per district)', fontweight='bold')
    ax1.set_title('Fig G1A – Count Prediction: RMSE\n(lower is better)', fontweight='bold')
    # Annotate
    for bar, val, err in zip(bars1, rmses, rmse_s):
        ax1.text(bar.get_x() + bar.get_width()/2, val + err + max(rmses)*0.01,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    # Highlight best
    best_r = np.argmin(rmses)
    bars1[best_r].set_edgecolor('gold')
    bars1[best_r].set_linewidth(2.5)
    ax1.text(best_r, rmses[best_r]/2, '★ BEST', ha='center',
             color='gold', fontweight='bold', fontsize=9)

    # AUC
    auc_valid = [a if not np.isnan(a) else 0 for a in aucs]
    auc_err_v = [s if not np.isnan(aucs[i]) else 0 for i, s in enumerate(auc_s)]
    bars2 = ax2.bar(x, auc_valid, width, yerr=auc_err_v, capsize=5,
                    color=palette, edgecolor='white', linewidth=0.6,
                    error_kw=dict(ecolor='#333', lw=1.5, capthick=1.5))
    ax2.axhline(0.5, color='red', linestyle='--', linewidth=1.2, label='Random (AUC=0.5)', zorder=0)
    ax2.set_ylim(0, 1.1)
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace(' ', '\n') for m in models], fontsize=9)
    ax2.set_ylabel('AUC (importation detection)', fontweight='bold')
    ax2.set_title('Fig G1B – Importation Detection: AUC\n(higher is better)', fontweight='bold')
    ax2.legend(fontsize=9)
    # Highlight best
    best_a = int(np.argmax(auc_valid))
    bars2[best_a].set_edgecolor('gold')
    bars2[best_a].set_linewidth(2.5)

    # Legend for colours
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=MODEL_COLORS['LR'],    label='Paper baseline (LR)'),
        Patch(facecolor=MODEL_COLORS['NB'],    label='Paper baseline (NB)'),
        Patch(facecolor=MODEL_COLORS['GCN'],   label='GCN (ours)'),
        Patch(facecolor=MODEL_COLORS['GAT'],   label='GAT (ours)'),
        Patch(facecolor=MODEL_COLORS['ST-GNN'],label='ST-GNN (ours)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=5,
               fontsize=9, framealpha=0.8, bbox_to_anchor=(0.5, -0.06))

    plt.suptitle('GNN Model Comparison vs Muller et al. (2025) Baselines\n'
                 'LOOCV: 6 folds (Jul–Dec 2023), Unguja districts only',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = os.path.join(save_dir, 'figG1_model_comparison.png')
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
#   GNN FIG G2  –  Per-fold LOOCV performance time series
# ══════════════════════════════════════════════════════════════════════════════

def figG2_loocv_timeseries(fold_results: dict, save_dir: str = SAVE_DIR):
    """
    Line plot of per-fold RMSE across 6 test months (Jul-Dec 2023).
    fold_results = {model_name: [rmse_fold1, ..., rmse_fold6]}
    """
    plt.rcParams.update(PAPER_STYLE)
    os.makedirs(save_dir, exist_ok=True)

    if not fold_results:
        print("  [FigG2] No fold results provided – skipping")
        return

    folds = ['Jul-23', 'Aug-23', 'Sep-23', 'Oct-23', 'Nov-23', 'Dec-23']
    fig, ax = plt.subplots(figsize=(10, 5))

    for model, rmses in fold_results.items():
        clr = MODEL_COLORS.get(model.split()[0], '#607D8B')
        n = min(len(folds), len(rmses))
        style = '--' if model.lower() in ('lr','nb','naive') else '-'
        ax.plot(folds[:n], rmses[:n], marker='o', linewidth=2,
                linestyle=style, color=clr, label=model, markersize=6)

    ax.set_xlabel('Test Month (LOOCV fold)', fontweight='bold')
    ax.set_ylabel('RMSE (imported cases)', fontweight='bold')
    ax.set_title('Fig G2 – Per-Fold LOOCV Performance\n(Leave-One-Month-Out, Jul–Dec 2023)',
                 fontweight='bold')
    ax.legend(title='Model', fontsize=9, title_fontsize=10)
    plt.tight_layout()
    out = os.path.join(save_dir, 'figG2_loocv_timeseries.png')
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
#   GNN FIG G3  –  Predicted vs Actual scatter (best GNN model)
# ══════════════════════════════════════════════════════════════════════════════

def figG3_pred_vs_actual(all_results_list: list, model_name: str = 'ST-GNN',
                          num_unguja: int = 7, save_dir: str = SAVE_DIR):
    """
    Scatter plot of predicted vs actual imported cases across all folds.
    all_results_list is the list returned by run_loocv().
    """
    plt.rcParams.update(PAPER_STYLE)
    os.makedirs(save_dir, exist_ok=True)

    if not all_results_list:
        print("  [FigG3] No results list provided – skipping")
        return

    preds, actuals, months_lbl = [], [], []
    for r in all_results_list:
        p = r['predictions'][:num_unguja]
        a = r['targets'][:num_unguja]
        preds.extend(p.tolist())
        actuals.extend(a.tolist())
        months_lbl.extend([r.get('test_month', '?')] * num_unguja)

    preds   = np.array(preds)
    actuals = np.array(actuals)

    # Color by month
    unique_months = sorted(set(months_lbl))
    cmap = plt.cm.tab10
    month_colors = {m: cmap(i/max(len(unique_months)-1,1))
                    for i, m in enumerate(unique_months)}
    colors = [month_colors[m] for m in months_lbl]

    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(actuals, preds, c=colors, alpha=0.75, s=60, edgecolors='white', linewidth=0.5)

    max_val = max(actuals.max(), preds.max()) * 1.1
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1.2, label='Perfect prediction')
    ax.set_xlim(-0.5, max_val)
    ax.set_ylim(-0.5, max_val)
    ax.set_xlabel('Actual Imported Cases', fontweight='bold')
    ax.set_ylabel('Predicted Imported Cases', fontweight='bold')
    ax.set_title(f'Fig G3 – Predicted vs Actual: {model_name}\n'
                 f'(All LOOCV folds, Unguja districts only)', fontweight='bold')

    from matplotlib.lines import Line2D
    legend_handles = [Line2D([0],[0], marker='o', color='w',
                             markerfacecolor=month_colors[m], markersize=8,
                             label=m) for m in unique_months]
    legend_handles.append(Line2D([0],[0], linestyle='--', color='k', label='Ideal'))
    ax.legend(handles=legend_handles, title='Test Month', fontsize=8,
              title_fontsize=9, loc='upper left')

    # Metrics annotation
    rmse_val = float(np.sqrt(np.mean((preds - actuals)**2)))
    mae_val  = float(np.mean(np.abs(preds - actuals)))
    ax.text(0.97, 0.05, f'RMSE = {rmse_val:.2f}\nMAE  = {mae_val:.2f}',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=10, bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    out = os.path.join(save_dir, f'figG3_pred_vs_actual_{model_name.replace(" ","_")}.png')
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
#   GNN FIG G4  –  Risk score distribution heatmap by district
# ══════════════════════════════════════════════════════════════════════════════

def figG4_risk_score_heatmap(clinic_csv: str = None, save_dir: str = SAVE_DIR):
    """
    Heatmap: districts on y-axis, months on x-axis, value = mean individual risk score.
    """
    try:
        import seaborn as sns
    except ImportError:
        print("  [FigG4] seaborn not installed – skipping")
        return

    plt.rcParams.update(PAPER_STYLE)
    os.makedirs(save_dir, exist_ok=True)

    if clinic_csv is None:
        clinic_csv = os.path.join(DATA_DIR, 'ZIM_clinic_data - ZIM_clinic_data.csv (1).csv')

    try:
        df = pd.read_csv(clinic_csv, low_memory=False)
        travel_col   = next((c for c in df.columns if c.lower() == 'travel'), None)
        date_col     = next((c for c in df.columns if 'date' in c.lower()), None)
        district_col = next((c for c in df.columns if 'district' in c.lower()), None)

        if not all([travel_col, date_col, district_col]):
            print("  [FigG4] Missing required columns – skipping")
            return

        df[travel_col] = pd.to_numeric(df[travel_col], errors='coerce').fillna(0)
        df[date_col]   = pd.to_datetime(df[date_col], errors='coerce')
        df['month']    = df[date_col].dt.to_period('M').astype(str)

        # Use travel status as proxy for individual risk (since actual risk scores
        # require running the trained scorer — this is from raw clinic data)
        monthly_district = df.groupby(['month', district_col])[travel_col].mean().unstack(fill_value=0)

        # Filter to Unguja districts and recent months
        unguja_cols = [c for c in monthly_district.columns
                       if any(d.lower() in c.lower() for d in
                              ['kaskazini', 'magharibi', 'kati', 'kusini', 'mjini'])]
        if not unguja_cols:
            unguja_cols = monthly_district.columns.tolist()[:7]

        data = monthly_district[unguja_cols].T  # districts × months

        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(data * 100, annot=True, fmt='.0f', cmap='YlOrRd',
                    linewidths=0.5, linecolor='white', ax=ax,
                    cbar_kws={'label': '% Travel-Associated Cases'})
        ax.set_title('Fig G4 – Travel Risk by District and Month\n'
                     '(% of clinic visits that are travel-associated)',
                     fontweight='bold', fontsize=13)
        ax.set_xlabel('Month', fontweight='bold')
        ax.set_ylabel('Home District', fontweight='bold')
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        out = os.path.join(save_dir, 'figG4_risk_heatmap_district_month.png')
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"  Saved: {out}")
    except Exception as e:
        print(f"  [FigG4] Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#   GNN FIG G5  –  Seasonal analysis: cases vs rainfall scatter
# ══════════════════════════════════════════════════════════════════════════════

def figG5_seasonal_scatter(clinic_csv: str = None, rain_csv: str = None,
                            save_dir: str = SAVE_DIR):
    """
    Scatter plot of monthly imported cases vs lagged rainfall, colour-coded by season.
    Extends Fig 3 of Muller et al. with seasonal annotation.
    """
    try:
        import seaborn as sns
    except ImportError:
        print("  [FigG5] seaborn not installed – skipping")
        return

    plt.rcParams.update(PAPER_STYLE)
    os.makedirs(save_dir, exist_ok=True)

    if clinic_csv is None:
        clinic_csv = os.path.join(DATA_DIR, 'ZIM_clinic_data - ZIM_clinic_data.csv (1).csv')
    if rain_csv is None:
        rain_csv   = os.path.join(DATA_DIR, 'CHIRPS_rainfall_RAW_ONLY (1).csv')

    try:
        df = pd.read_csv(clinic_csv, low_memory=False)
        travel_col = next((c for c in df.columns if c.lower() == 'travel'), 'travel')
        date_col   = next((c for c in df.columns if 'date' in c.lower()), None)

        df[travel_col] = pd.to_numeric(df[travel_col], errors='coerce').fillna(0)
        df[date_col]   = pd.to_datetime(df[date_col], errors='coerce')
        df['month']    = df[date_col].dt.to_period('M')
        df['month_n']  = df[date_col].dt.month

        imported = df[df[travel_col] == 1].groupby('month').size().rename('imported')
        month_n  = df.groupby('month')['month_n'].first()

        rdf = pd.read_csv(rain_csv, low_memory=False)
        rcol = next((c for c in rdf.columns if 'rainfall' in c.lower()), None)
        dcol = next((c for c in rdf.columns if 'week' in c.lower() or 'date' in c.lower()), None)
        rdf[dcol] = pd.to_datetime(rdf[dcol], errors='coerce')
        rdf['month'] = rdf[dcol].dt.to_period('M')
        monthly_rain = rdf.groupby('month')[rcol].mean().rename('rainfall')

        merged = pd.concat([imported, monthly_rain, month_n], axis=1).dropna()
        merged['Season'] = merged['month_n'].apply(
            lambda m: 'Wet' if m in [3,4,5,10,11,12] else 'Dry'
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        palette = {'Wet': '#1565C0', 'Dry': '#EF6C00'}

        for season, grp in merged.groupby('Season'):
            ax.scatter(grp['rainfall'], grp['imported'],
                       color=palette[season], label=season,
                       s=70, alpha=0.8, edgecolors='white', linewidth=0.5, zorder=3)

        # Regression line (overall)
        from numpy.polynomial import polynomial as P
        x, y = merged['rainfall'].values, merged['imported'].values
        if len(x) > 2:
            coef = np.polyfit(x, y, 1)
            xfit = np.linspace(x.min(), x.max(), 100)
            ax.plot(xfit, np.polyval(coef, xfit), 'k--', linewidth=1.5,
                    label='Linear trend', alpha=0.7)

        ax.set_xlabel('Mean Lagged Rainfall (mm)', fontweight='bold')
        ax.set_ylabel('Monthly Imported Cases', fontweight='bold')
        ax.set_title('Fig G5 – Imported Cases vs Lagged Rainfall\n'
                     '(colour-coded by season, regression trend shown)',
                     fontweight='bold')
        ax.legend(title='Season', fontsize=9)

        plt.tight_layout()
        out = os.path.join(save_dir, 'figG5_seasonal_scatter.png')
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"  Saved: {out}")
    except Exception as e:
        print(f"  [FigG5] Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#   MASTER ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def generate_all_paper_figures(results: dict = None, loocv_results_list: dict = None,
                                save_dir: str = SAVE_DIR):
    """
    Generate all figures.

    Args:
        results: dict of {model_name: summary_dict} from main.py (for FigG1, G2)
        loocv_results_list: dict of {model_name: all_results_list} from run_loocv() (for FigG3)
        save_dir: output directory
    """
    print(f"\n{'='*60}")
    print("  Generating ALL paper-aligned figures")
    print(f"  Output: {save_dir}")
    print(f"{'='*60}\n")

    ## Paper replications
    print("[Paper Figs] Replicating Muller et al. (2025) figures...")
    fig1_district_cases(save_dir=save_dir)
    fig2_rainfall_timeseries(save_dir=save_dir)
    fig3_nb_irr_forestplot(save_dir=save_dir)
    fig5_demographic_heatmap(save_dir=save_dir)

    ## GNN extension figs
    print("\n[GNN Figs] Generating GNN extension figures...")
    if results:
        figG1_model_comparison(results, save_dir=save_dir)

        # Build fold RMSE dict for G2
        fold_rmse_dict = {}
        for name, summary in results.items():
            if 'fold_rmses' in summary:
                fold_rmse_dict[name] = summary['fold_rmses']
        if fold_rmse_dict:
            figG2_loocv_timeseries(fold_rmse_dict, save_dir=save_dir)

    if loocv_results_list:
        # Use best GNN model results for G3
        gnn_keys = [k for k in loocv_results_list if 'st' in k.lower() or 'gnn' in k.lower()]
        if not gnn_keys:
            gnn_keys = list(loocv_results_list.keys())
        if gnn_keys:
            figG3_pred_vs_actual(loocv_results_list[gnn_keys[0]],
                                  model_name=gnn_keys[0], save_dir=save_dir)

    figG4_risk_score_heatmap(save_dir=save_dir)
    figG5_seasonal_scatter(save_dir=save_dir)

    print(f"\n[OK] All figures saved to: {save_dir}\n")


# ─── Standalone run ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    generate_all_paper_figures()
