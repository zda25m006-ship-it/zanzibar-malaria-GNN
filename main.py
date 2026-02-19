"""
GNN Malaria Importation Prediction — Individual Risk Score pipeline.

Run: python main.py [--mode smoke]

Two separate comparisons (matching paper):
  A) Importation Detection: LR vs GNN on AUC
  B) Count Prediction:      NB  vs GNN on RMSE
Both evaluated with Leave-One-Month-Out CV (6 folds, Jul-Dec 2023).

Key feature: per-fold IndividualRiskScorer trained on training patients only;
risk scores are used as lagged node features (no leakage).
"""

import sys, os, io, argparse, copy
import numpy as np
import torch
import torch.nn as nn
import warnings

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.temporal_dataset  import create_loocv_folds
from data.data_loader       import build_master_dataset, UNGUJA_DISTRICTS
from data.graph_builder     import get_node_to_idx
from data.feature_engineering import NUM_FEATURES
from models.baseline_models import LogisticRegressionBaseline, NegativeBinomialBaseline
from models.gcn_model       import MalariaGCN
from models.gat_model       import MalariaGAT
from models.stgnn_model     import MalariaST_GNN
from training.cv_trainer    import run_loocv, train_one_fold
from evaluation.compare     import compare_models, statistical_significance
from sklearn.metrics        import roc_auc_score

NUM_UNGUJA = 7


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def unguja_metrics(pred: np.ndarray, target: np.ndarray):
    """Compute RMSE, MAE, AUC over Unguja nodes only."""
    p, t = pred[:NUM_UNGUJA], target[:NUM_UNGUJA]
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    mae  = float(np.mean(np.abs(p - t)))
    bt   = (t > 0).astype(int)
    if bt.sum() > 0 and bt.sum() < len(bt):
        try:   auc = float(roc_auc_score(bt, p))
        except: auc = float('nan')
    else:
        auc = float('nan')
    return rmse, mae, auc


def print_table(rows: list, title: str):
    """Print a formatted comparison table."""
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")
    print(f"{'Model':<25} {'RMSE':>8} {'+-':>6} {'MAE':>8} {'AUC':>8} {'+-':>6}")
    print(f"{'-'*72}")
    for name, mean_r, std_r, mean_m, mean_a, std_a in rows:
        auc_s = f"{mean_a:.3f}" if not np.isnan(mean_a) else "  n/a "
        std_s = f"{std_a:.3f}" if not np.isnan(std_a) else "  n/a "
        print(f"{name:<25} {mean_r:>8.3f} {std_r:>6.3f} {mean_m:>8.3f} "
              f"{auc_s:>8} {std_s:>6}")


# ─────────────────────────────────────────────────────────────────────────────
# BASELINES
# ─────────────────────────────────────────────────────────────────────────────

def run_lr_loocv(raw_data: dict, folds: list):
    """
    Logistic Regression: per-patient classification (imported vs local).
    Evaluated correctly on AUC per fold, then aggregated.
    """
    print("\n" + "="*60)
    print("BASELINE A: Logistic Regression (per-patient AUC)")
    print("="*60)

    clinic     = raw_data['clinic_raw']
    node_to_idx = get_node_to_idx()

    lr_model = LogisticRegressionBaseline()
    X_all, y_all = lr_model.prepare_features(clinic)
    clinic_indexed = clinic.reset_index(drop=True)

    fold_aucs, fold_rmses, fold_maes = [], [], []
    rows = []

    for fold in folds:
        test_month = fold['test_month']
        train_months_set = set(fold['test_month'])  # test month is held out

        # Train on all patients from months BEFORE test month
        train_idx = [i for i, row in clinic_indexed.iterrows()
                     if str(row['year_month']) < test_month]
        test_idx  = [i for i, row in clinic_indexed.iterrows()
                     if str(row['year_month']) == test_month]

        if len(train_idx) < 10 or len(test_idx) < 1:
            continue

        lr = LogisticRegressionBaseline()
        Xtr, ytr = X_all[train_idx], y_all[train_idx]
        Xte, yte = X_all[test_idx], y_all[test_idx]
        lr.fit(Xtr, ytr)
        metrics = lr.evaluate(Xte, yte)
        auc = metrics['auc']

        # Node-level count predictions from summed probabilities
        test_clinic_fold = clinic_indexed.iloc[test_idx].copy()
        test_clinic_fold['pred_prob'] = lr.predict_proba(Xte)
        test_clinic_fold['year_month_str'] = test_clinic_fold['year_month'].astype(str)

        actual   = fold['test_graph'].y.numpy()
        pred_arr = np.zeros(len(actual), dtype=np.float32)
        for district in UNGUJA_DISTRICTS:
            d_idx = node_to_idx[district]
            d_data = test_clinic_fold[test_clinic_fold['home_district'] == district]
            if len(d_data):
                pred_arr[d_idx] = d_data['pred_prob'].sum()

        rmse, mae, _ = unguja_metrics(pred_arr, actual)
        fold_aucs.append(auc)
        fold_rmses.append(rmse)
        fold_maes.append(mae)
        print(f"  {test_month}: AUC={auc:.3f}  RMSE={rmse:.2f}  MAE={mae:.2f}")

    summary = {
        'mean_rmse': np.mean(fold_rmses), 'std_rmse': np.std(fold_rmses),
        'mean_mae':  np.mean(fold_maes),  'std_mae':  np.std(fold_maes),
        'mean_auc':  np.mean(fold_aucs),  'std_auc':  np.std(fold_aucs),
        'fold_rmses': fold_rmses, 'fold_aucs': fold_aucs,
    }
    print(f"  => Mean AUC={summary['mean_auc']:.3f}+-{summary['std_auc']:.3f}  "
          f"RMSE={summary['mean_rmse']:.3f}+-{summary['std_rmse']:.3f}")
    return summary


def run_nb_loocv(raw_data: dict, folds: list):
    """
    Negative Binomial Regression: predicts monthly total imported counts,
    distributed by historical district proportions.
    """
    print("\n" + "="*60)
    print("BASELINE B: Negative Binomial Regression (monthly counts)")
    print("="*60)

    monthly_counts = raw_data['monthly_counts']
    rainfall       = raw_data['rainfall']
    node_to_idx    = get_node_to_idx()

    nb = NegativeBinomialBaseline()
    X_all, y_all, merged = nb.prepare_features(monthly_counts, rainfall)
    merged_months = merged['year_month'].values

    fold_rmses, fold_maes, fold_aucs = [], [], []

    for fold in folds:
        test_month = fold['test_month']
        train_mask = [m < test_month for m in merged_months]
        test_mask  = [m == test_month for m in merged_months]

        if sum(train_mask) < 3 or sum(test_mask) < 1:
            continue

        Xtr = X_all[train_mask]; ytr = y_all[train_mask]
        Xte = X_all[test_mask]

        try:
            nb_fold = NegativeBinomialBaseline()
            nb_fold.fit(Xtr, ytr)
            total_pred = float(nb_fold.predict(Xte)[0])
        except Exception as e:
            print(f"  {test_month}: NB failed ({e}), using train mean")
            total_pred = float(ytr.mean())

        # Historical proportions from training months only
        train_data  = monthly_counts[monthly_counts['year_month'] < test_month]
        n_nodes     = fold['test_graph'].y.shape[0]
        node_fracs  = np.zeros(n_nodes, dtype=np.float32)
        total_hist  = 0
        for district in UNGUJA_DISTRICTS:
            d_idx  = node_to_idx[district]
            d_hist = train_data[train_data['home_district'] == district]['imported_cases'].sum()
            node_fracs[d_idx] = float(d_hist)
            total_hist += d_hist
        if total_hist > 0:
            node_fracs = node_fracs / total_hist
        else:
            node_fracs[:NUM_UNGUJA] = 1.0 / NUM_UNGUJA

        pred   = node_fracs * total_pred
        actual = fold['test_graph'].y.numpy()
        rmse, mae, auc = unguja_metrics(pred, actual)
        fold_rmses.append(rmse); fold_maes.append(mae)
        if not np.isnan(auc): fold_aucs.append(auc)
        print(f"  {test_month}: RMSE={rmse:.2f}  MAE={mae:.2f}  AUC={auc:.3f}  "
              f"(pred_total={total_pred:.1f})")

    summary = {
        'mean_rmse': np.mean(fold_rmses), 'std_rmse': np.std(fold_rmses),
        'mean_mae':  np.mean(fold_maes),  'std_mae':  np.std(fold_maes),
        'mean_auc':  np.mean(fold_aucs) if fold_aucs else float('nan'),
        'std_auc':   np.std(fold_aucs)  if fold_aucs else float('nan'),
        'fold_rmses': fold_rmses, 'fold_aucs': fold_aucs,
    }
    print(f"  => RMSE={summary['mean_rmse']:.3f}+-{summary['std_rmse']:.3f}  "
          f"AUC={summary['mean_auc']:.3f}")
    return summary


def run_naive_baseline(folds: list):
    """Predict training-mean per node — the absolute floor to beat."""
    print("\n" + "="*60)
    print("BASELINE C: Naive (predict training mean per district)")
    print("="*60)
    fold_rmses, fold_maes = [], []
    for fold in folds:
        # Training mean of imported cases per Unguja node
        train_targets = np.stack([g.y.numpy()[:NUM_UNGUJA] for g in fold['train_graphs']], axis=0)
        pred_u = train_targets.mean(axis=0)
        actual_u = fold['test_graph'].y.numpy()[:NUM_UNGUJA]
        rmse = float(np.sqrt(np.mean((pred_u - actual_u)**2)))
        mae  = float(np.mean(np.abs(pred_u - actual_u)))
        fold_rmses.append(rmse); fold_maes.append(mae)
        print(f"  {fold['test_month']}: RMSE={rmse:.2f}  MAE={mae:.2f}")
    summary = {'mean_rmse': np.mean(fold_rmses), 'std_rmse': np.std(fold_rmses),
                'mean_mae': np.mean(fold_maes)}
    print(f"  => RMSE={summary['mean_rmse']:.3f}+-{summary['std_rmse']:.3f}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# GNN MODELS via LOOCV
# ─────────────────────────────────────────────────────────────────────────────

def run_gnn_loocv(name: str, model_class, model_kwargs: dict,
                  folds: list, epochs: int = 200, lr: float = 0.005,
                  loss: str = 'combined'):
    print(f"\n{'='*60}")
    print(f"GNN: {name} (Loss: {loss})")
    print(f"{'='*60}")
    n_params = sum(p.numel() for p in model_class(**model_kwargs).parameters())
    print(f"  Parameters: {n_params:,}")

    results, summary = run_loocv(
        model_class, model_kwargs, folds,
        lr=lr, epochs=epochs, patience=25,
        weight_decay=1e-3, num_unguja=NUM_UNGUJA, verbose=True,
        loss=loss,
    )
    print(f"  => RMSE={summary['mean_rmse']:.3f}+-{summary['std_rmse']:.3f}  "
          f"AUC={summary['mean_auc']:.3f}+-{summary['std_auc']:.3f}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='full', choices=['full', 'smoke'])
    parser.add_argument('--data-dir', default='c:/malaria')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--save-dir', default='results')
    args = parser.parse_args()

    if args.mode == 'smoke':
        args.epochs = 15
        print("[SMOKE] Quick test — 15 epochs per fold")

    print("="*70)
    print("  GNN MALARIA IMPORTATION PREDICTION — LEAKAGE-FREE PIPELINE")
    print("  Data: May 2022 - Dec 2023 | LOOCV: Jul-Dec 2023 (6 folds)")
    print("="*70)

    # ── 1. Build LOOCV folds ──────────────────────────────────────────────
    print("\n[1/4] Building LOOCV folds (May 2022 - Dec 2023)...")
    folds = create_loocv_folds(args.data_dir, lookback=2)
    raw   = folds[0]['raw_data']

    # Derive metadata from first fold
    sample_g  = folds[0]['train_graphs'][0]
    in_ch     = sample_g.x.shape[1]
    edge_d    = folds[0]['train_graphs'][0].edge_attr.shape[-1] if folds[0]['train_graphs'][0].edge_attr.dim() > 1 else 1
    n_nodes   = sample_g.x.shape[0]
    print(f"  Graph: {n_nodes} nodes, {in_ch} features")
    print(f"  LOOCV folds: {len(folds)}")
    for f in folds:
        print(f"    {f['test_month']}: train={len(f['train_graphs'])} months")

    # ── Scorer AUC sanity check ───────────────────────────────────────────
    print("\n  [Risk Scorer AUC per fold — sanity check]")
    clinic_raw = raw['clinic_raw'].copy()
    clinic_raw['date'] = clinic_raw['date'].dt.strftime('%Y-%m') if hasattr(clinic_raw['date'].iloc[0], 'strftime') else clinic_raw['date']
    for f in folds:
        scorer = f.get('scorer')
        if scorer is None:
            continue
        test_month = f['test_month']
        test_patients = clinic_raw[clinic_raw['date'] == test_month]
        if len(test_patients) > 0:
            auc = scorer.auc_on(test_patients)
            auc_str = f"{auc:.3f}" if not np.isnan(auc) else "n/a (degenerate)"
            print(f"    {test_month}: risk scorer AUC on test patients = {auc_str}")

    # ── 2. Baselines ─────────────────────────────────────────────────────
    print("\n[2/4] Running baselines...")
    naive_summary = run_naive_baseline(folds)
    lr_summary    = run_lr_loocv(raw, folds)
    nb_summary    = run_nb_loocv(raw, folds)

    in_ch   = in_ch
    edge_d  = edge_d

    # ── 3. GNN models via LOOCV ──────────────────────────────────────────
    print("\n[3/4] Training GNN models (LOOCV)...")

    # Simpler architectures for small dataset (14-19 training months per fold)
    gcn_summary = run_gnn_loocv(
        "GCN (1-layer)", MalariaGCN,
        dict(in_channels=in_ch, hidden_channels=32, out_channels=1,
             num_layers=1, dropout=0.2),
        folds, epochs=args.epochs, lr=0.005,
    )

    gat_summary = run_gnn_loocv(
        "GAT (1-layer, 2-head)", MalariaGAT,
        dict(in_channels=in_ch, hidden_channels=32, out_channels=1,
             num_heads=2, dropout=0.2, edge_dim=edge_d),
        folds, epochs=args.epochs, lr=0.005,
    )

    stgnn_summary = run_gnn_loocv(
        "ST-GNN (GAT+GRU)", MalariaST_GNN,
        dict(in_channels=in_ch, hidden_channels=32, out_channels=1,
             num_spatial_layers=1, num_temporal_layers=1,
             dropout=0.2, edge_dim=edge_d, num_attention_heads=2),
        folds, epochs=args.epochs, lr=0.003,
    )

    # ── 4. Final comparison ──────────────────────────────────────────────
    print("\n[4/4] Results")

    all_rows = [
        ("Naive (train mean)",      naive_summary['mean_rmse'], naive_summary['std_rmse'],
                                    naive_summary['mean_mae'],  float('nan'), float('nan')),
        ("[Paper] LR (Table 2)",    lr_summary['mean_rmse'],    lr_summary['std_rmse'],
                                    lr_summary['mean_mae'],     lr_summary['mean_auc'], lr_summary['std_auc']),
        ("[Paper] NB (Fig.3)",      nb_summary['mean_rmse'],    nb_summary['std_rmse'],
                                    nb_summary['mean_mae'],     nb_summary['mean_auc'], nb_summary['std_auc']),
        ("GCN",                     gcn_summary['mean_rmse'],   gcn_summary['std_rmse'],
                                    gcn_summary['mean_mae'],    gcn_summary['mean_auc'], gcn_summary['std_auc']),
        ("GAT",                     gat_summary['mean_rmse'],   gat_summary['std_rmse'],
                                    gat_summary['mean_mae'],    gat_summary['mean_auc'], gat_summary['std_auc']),
        ("ST-GNN",                  stgnn_summary['mean_rmse'], stgnn_summary['std_rmse'],
                                    stgnn_summary['mean_mae'],  stgnn_summary['mean_auc'], stgnn_summary['std_auc']),
    ]

    print_table(all_rows, "LOOCV RESULTS — Muller et al. 2025 vs GNN (6 folds, Jul-Dec 2023)")
    _print_paper_vs_gnn(lr_summary, nb_summary, gcn_summary, gat_summary, stgnn_summary)

    # Save plots
    save_dir = os.path.join(args.data_dir, args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    _save_cv_summary_plot(all_rows, save_dir)
    print(f"\n  Plots saved to {save_dir}/")


def _print_paper_vs_gnn(lr_s, nb_s, gcn_s, gat_s, stgnn_s):
    """
    Two-task comparison: Paper method (Muller et al. 2025) vs GNNs.
    Task A: AUC  (paper used LR, Table 2)
    Task B: RMSE (paper used NB regression, Fig.3 / Methods)
    """
    W = 72
    print(f"\n{'='*W}")
    print(f"  GNN vs Muller et al. (2025) Malaria Journal 24:373")
    print(f"  doi: 10.1186/s12936-025-05605-1")
    print(f"{'='*W}")

    # Task A: AUC
    print(f"\n  TASK A \u2014 Importation Detection (AUC, higher = better)")
    print(f"  Paper method: Mixed-effects LR, Table 2 (sex/age/occupation/season/district)")
    print(f"  Note: paper reports odds ratios only; we evaluate predictively on held-out months")
    print(f"  {'Model':<32} {'AUC':>8} {'\u00b1':>4}  delta vs paper")
    print(f"  {'-'*60}")
    plr = lr_s['mean_auc']
    print(f"  {'[PAPER] LR (Muller Table 2)':<32} {plr:>8.3f} {lr_s['std_auc']:>6.3f}  (baseline)")
    for nm, s in [("GCN", gcn_s), ("GAT", gat_s), ("ST-GNN", stgnn_s)]:
        auc = s['mean_auc']
        if np.isnan(auc):
            print(f"  {nm:<32} {'n/a':>8} {'n/a':>4}")
            continue
        d = auc - plr
        ds = f"+{d:.3f}" if d >= 0 else f"{d:.3f}"
        beat = "<<< BEATS PAPER" if auc > plr else "    below paper"
        print(f"  {nm:<32} {auc:>8.3f} {s['std_auc']:>6.3f}  {ds:>8}  {beat}")

    # Task B: RMSE
    print(f"\n  TASK B \u2014 Count Prediction (RMSE, lower = better)")
    print(f"  Paper method: NB regression, Methods/Fig.3 (lagged CHIRPS rainfall only)")
    print(f"  Note: paper reports IRR=1.06/inch; we evaluate predictively on held-out months")
    print(f"  {'Model':<32} {'RMSE':>8} {'\u00b1':>4}  delta vs paper")
    print(f"  {'-'*60}")
    pnb = nb_s['mean_rmse']
    print(f"  {'[PAPER] NB (Muller Fig.3)':<32} {pnb:>8.3f} {nb_s['std_rmse']:>6.3f}  (baseline)")
    for nm, s in [("GCN", gcn_s), ("GAT", gat_s), ("ST-GNN", stgnn_s)]:
        rmse = s['mean_rmse']
        d = rmse - pnb
        ds = f"+{d:.3f}" if d >= 0 else f"{d:.3f}"
        beat = "<<< BEATS PAPER" if rmse < pnb else "    above paper"
        print(f"  {nm:<32} {rmse:>8.3f} {s['std_rmse']:>6.3f}  {ds:>8}  {beat}")
    print(f"\n{'='*W}")



def _save_cv_summary_plot(rows, save_dir):
    """Bar chart — red=paper methods, green=GNN."""
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        names   = [r[0] for r in rows]
        rmses   = [r[1] for r in rows]
        stds    = [r[2] for r in rows]
        aucs    = [r[4] for r in rows]
        auc_std = [r[5] for r in rows]

        colors = []
        for n in names:
            if 'Paper' in n:
                colors.append('#EF553B')
            elif 'Naive' in n:
                colors.append('#636EFA')
            else:
                colors.append('#00CC96')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('#0f1117')

        ax1.set_facecolor('#1a1d27')
        ax1.bar(names, rmses, yerr=stds, capsize=4,
                color=colors[:len(names)], edgecolor='white', linewidth=0.5,
                error_kw=dict(ecolor='white', lw=1.5))
        ax1.set_title('RMSE — Count Prediction\nred=Muller2025, green=GNN (ours)',
                      color='white', fontsize=11, fontweight='bold')
        ax1.set_ylabel('RMSE (imported cases)', color='white')
        ax1.tick_params(colors='white'); ax1.xaxis.set_tick_params(rotation=35)
        ax1.spines[['top','right']].set_visible(False)
        for s in ['bottom','left']: ax1.spines[s].set_color('#444')
        [t.set_color('white') for t in ax1.get_xticklabels()+ax1.get_yticklabels()]

        ax2.set_facecolor('#1a1d27')
        auc_vals = [a if not np.isnan(a) else 0 for a in aucs]
        auc_errs = [a if not np.isnan(a) else 0 for a in auc_std]
        ax2.bar(names, auc_vals, yerr=auc_errs, capsize=4,
                color=colors[:len(names)], edgecolor='white', linewidth=0.5,
                error_kw=dict(ecolor='white', lw=1.5))
        ax2.axhline(0.5, color='red', linestyle='--', alpha=0.6, label='Random (AUC=0.5)')
        ax2.set_title('AUC — Importation Detection\nred=Muller2025, green=GNN (ours)',
                      color='white', fontsize=11, fontweight='bold')
        ax2.set_ylabel('AUC', color='white')
        ax2.tick_params(colors='white'); ax2.xaxis.set_tick_params(rotation=35)
        ax2.set_ylim(0, 1.05)
        ax2.spines[['top','right']].set_visible(False)
        for s in ['bottom','left']: ax2.spines[s].set_color('#444')
        [t.set_color('white') for t in ax2.get_xticklabels()+ax2.get_yticklabels()]
        ax2.legend(facecolor='#1a1d27', labelcolor='white')

        plt.tight_layout()
        out = os.path.join(save_dir, 'loocv_comparison.png')
        plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()
        print(f"  Saved: {out}")
    except Exception as e:
        print(f"  Plot failed: {e}")



if __name__ == '__main__':
    main()
