"""
S&P 500 XGBoost + Decision Tree Results — Visualization Suite
Reads: sp500_xgb_dt_results_*.csv
Generates: 8 plots covering performance, accuracy, and feature importance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import glob

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

# Auto-detect CSV or set manually
CSV_PATH = None  # e.g. 'sp500_xgb_dt_results_20240101_120000.csv'

STYLE = 'seaborn-v0_8-darkgrid'
FIGSIZE_LARGE = (18, 10)
FIGSIZE_MED   = (15, 7)
TOP_N         = 15        # top/bottom N tickers to show in bar charts
DPI           = 150
SAVE_FIGS     = True      # save PNGs to disk
OUTPUT_DIR    = 'plots'

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_results(path=None):
    if path and Path(path).exists():
        df = pd.read_csv(path)
    else:
        files = sorted(glob.glob('sp500_xgb_dt_results_*.csv'))
        if not files:
            raise FileNotFoundError("No results CSV found. Set CSV_PATH or run batch_process first.")
        df = pd.read_csv(files[-1])  # most recent
        print(f"Loaded: {files[-1]}")

    print(f"  {len(df)} tickers, columns: {list(df.columns)}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def save(fig, name):
    if SAVE_FIGS:
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        fig.savefig(f"{OUTPUT_DIR}/{name}.png", dpi=DPI, bbox_inches='tight')
        print(f"  Saved → {OUTPUT_DIR}/{name}.png")


def pct(x):
    return x * 100  # convert ratio to percentage


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — R² Distribution (histogram + KDE + stats box)
# ══════════════════════════════════════════════════════════════════════════════

def plot_r2_distribution(df):
    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(10, 5))

    r2 = df['test_r2']
    ax.hist(r2, bins=40, color='steelblue', edgecolor='white', alpha=0.85, density=True, label='Histogram')

    # KDE overlay
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(r2)
    xs  = np.linspace(r2.min(), r2.max(), 300)
    ax.plot(xs, kde(xs), color='tomato', lw=2.5, label='KDE')

    # Vertical lines
    ax.axvline(r2.mean(),   color='gold',       lw=2, ls='--', label=f'Mean  {r2.mean():.4f}')
    ax.axvline(r2.median(), color='limegreen',  lw=2, ls=':',  label=f'Median {r2.median():.4f}')
    ax.axvline(0,           color='white',      lw=1.5, ls='-', alpha=0.5)

    # Stats box
    stats_txt = (f"n = {len(r2)}\n"
                 f"Mean   = {r2.mean():.4f}\n"
                 f"Median = {r2.median():.4f}\n"
                 f"Std    = {r2.std():.4f}\n"
                 f"Min    = {r2.min():.4f}\n"
                 f"Max    = {r2.max():.4f}\n"
                 f"R²>0   = {(r2 > 0).mean():.1%}")
    ax.text(0.98, 0.97, stats_txt, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='#1e1e2e', alpha=0.8),
            fontsize=9, color='white', fontfamily='monospace')

    ax.set_xlabel('Test R²', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Test R² Distribution Across All Tickers', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    save(fig, '1_r2_distribution')
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Directional Accuracy: Blend vs XGB vs DT (violin / box)
# ══════════════════════════════════════════════════════════════════════════════

def plot_directional_accuracy_comparison(df):
    plt.style.use(STYLE)
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_MED)

    acc_data = pd.DataFrame({
        'Blend (70/30)' : pct(df['test_dir_acc']),
        'XGBoost'       : pct(df['xgb_dir_acc']),
        'Decision Tree' : pct(df['dt_dir_acc']),
    })

    colors = ['#4FC3F7', '#81C784', '#FFB74D']

    # Left: violin
    ax = axes[0]
    parts = ax.violinplot([acc_data[c] for c in acc_data.columns],
                          showmedians=True, showextrema=True)
    for pc, col in zip(parts['bodies'], colors):
        pc.set_facecolor(col)
        pc.set_alpha(0.75)
    parts['cmedians'].set_color('white')
    parts['cmedians'].set_linewidth(2)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(acc_data.columns, fontsize=11)
    ax.axhline(50, color='tomato', lw=1.5, ls='--', label='Random (50%)')
    ax.set_ylabel('Directional Accuracy (%)', fontsize=12)
    ax.set_title('Directional Accuracy Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)

    # Right: summary bar
    ax2 = axes[1]
    means   = acc_data.mean()
    medians = acc_data.median()
    x = np.arange(len(means))
    w = 0.35
    ax2.bar(x - w/2, means,   width=w, color=colors, alpha=0.85, label='Mean',   edgecolor='white')
    ax2.bar(x + w/2, medians, width=w, color=colors, alpha=0.50, label='Median', edgecolor='white', hatch='//')
    ax2.axhline(50, color='tomato', lw=1.5, ls='--')
    ax2.set_xticks(x)
    ax2.set_xticklabels(acc_data.columns, fontsize=11)
    ax2.set_ylabel('Directional Accuracy (%)', fontsize=12)
    ax2.set_title('Mean vs Median Directional Accuracy', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)

    # Annotate bars
    for bar in ax2.patches:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    save(fig, '2_directional_accuracy')
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Top / Bottom N tickers by Test R²
# ══════════════════════════════════════════════════════════════════════════════

def plot_top_bottom_r2(df, n=TOP_N):
    plt.style.use(STYLE)
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_LARGE)

    top = df.nlargest(n,  'test_r2')[['ticker', 'test_r2', 'test_dir_acc']].reset_index(drop=True)
    bot = df.nsmallest(n, 'test_r2')[['ticker', 'test_r2', 'test_dir_acc']].reset_index(drop=True)

    def _bar(ax, data, col, title, cmap_name):
        cmap   = plt.get_cmap(cmap_name)
        vals   = data[col].values
        normed = (vals - vals.min()) / (np.ptp(vals) + 1e-9)
        cols   = [cmap(v) for v in normed]
        bars   = ax.barh(data['ticker'], vals, color=cols, edgecolor='white', height=0.7)
        ax.set_xlabel(col.replace('_', ' ').title(), fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.invert_yaxis()
        ax.axvline(0, color='white', lw=1, ls='--', alpha=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + 0.0002 * np.sign(v), bar.get_y() + bar.get_height()/2,
                    f'{v:.4f}', va='center', fontsize=8)

    _bar(axes[0], top, 'test_r2', f'Top {n} Tickers — Test R²',    'YlGn')
    _bar(axes[1], bot, 'test_r2', f'Bottom {n} Tickers — Test R²', 'YlOrRd')

    plt.suptitle('Test R² — Top vs Bottom Performers', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    save(fig, '3_top_bottom_r2')
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Top / Bottom N tickers by Directional Accuracy
# ══════════════════════════════════════════════════════════════════════════════

def plot_top_bottom_diracc(df, n=10):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

    top = (
        df.nlargest(n, 'test_dir_acc')[['ticker', 'test_dir_acc', 'xgb_dir_acc', 'dt_dir_acc']]
        .sort_values('test_dir_acc', ascending=True)
        .reset_index(drop=True)
    )
    bot = (
        df.nsmallest(n, 'test_dir_acc')[['ticker', 'test_dir_acc', 'xgb_dir_acc', 'dt_dir_acc']]
        .sort_values('test_dir_acc', ascending=True)
        .reset_index(drop=True)
    )

    all_vals = pd.concat(
        [
            pct(top['test_dir_acc']), pct(top['xgb_dir_acc']), pct(top['dt_dir_acc']),
            pct(bot['test_dir_acc']), pct(bot['xgb_dir_acc']), pct(bot['dt_dir_acc']),
        ],
        ignore_index=True,
    )
    xmin = max(0, np.floor(all_vals.min() - 2))
    xmax = np.ceil(all_vals.max() + 3)

    def _draw_panel(ax, data, title):
        y = np.arange(len(data))
        h = 0.23

        ax.axvspan(xmin, 50, color='#fdecea', alpha=0.45, zorder=0)
        ax.axvspan(50, xmax, color='#eaf7ee', alpha=0.45, zorder=0)

        bars_blend = ax.barh(
            y + h, pct(data['test_dir_acc']), height=h, label='Blend',
            color='#1f77b4', edgecolor='white', linewidth=0.8, zorder=3
        )
        bars_xgb = ax.barh(
            y, pct(data['xgb_dir_acc']), height=h, label='XGBoost',
            color='#2ca02c', edgecolor='white', linewidth=0.8, zorder=3
        )
        bars_dt = ax.barh(
            y - h, pct(data['dt_dir_acc']), height=h, label='Decision Tree',
            color='#ff7f0e', edgecolor='white', linewidth=0.8, zorder=3
        )

        ax.set_yticks(y)
        ax.set_yticklabels(data['ticker'], fontsize=11, fontweight='bold')
        ax.set_xlim(xmin, xmax)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('Ticker', fontsize=11)
        ax.grid(axis='x', linestyle='--', alpha=0.35, zorder=1)
        ax.axvline(50, color='#d62728', lw=1.8, ls='--', zorder=4)
        ax.text(50.2, len(data) - 0.35, '50% baseline', color='#d62728', fontsize=10, va='top')

        for bars in (bars_blend, bars_xgb, bars_dt):
            for bar in bars:
                v = bar.get_width()
                ax.text(
                    v + 0.2, bar.get_y() + bar.get_height() / 2, f'{v:.1f}%',
                    va='center', ha='left', fontsize=9, color='#1f1f1f', zorder=5
                )

    _draw_panel(axes[0], top, f'Top {n} Tickers by Directional Accuracy')
    _draw_panel(axes[1], bot, f'Bottom {n} Tickers by Directional Accuracy')

    axes[1].set_xlabel('Directional Accuracy (%)', fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc='upper center', frameon=False, fontsize=11)
    fig.suptitle('Directional Accuracy Comparison (Top 10 vs Bottom 10)', fontsize=17, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save(fig, '4_top_bottom_diracc')
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 5 — R² vs Directional Accuracy scatter
# ══════════════════════════════════════════════════════════════════════════════

def plot_r2_vs_diracc(df):
    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(11, 7))

    sc = ax.scatter(df['test_r2'], pct(df['test_dir_acc']),
                    c=pct(df['test_dir_acc']), cmap='RdYlGn',
                    s=60, alpha=0.75, edgecolors='white', lw=0.4)
    plt.colorbar(sc, ax=ax, label='Directional Accuracy (%)')

    # Quadrant lines
    ax.axvline(0,  color='white', lw=1, ls='--', alpha=0.5)
    ax.axhline(50, color='tomato', lw=1.5, ls='--', alpha=0.8, label='50% dir. acc.')

    # Annotate top performers
    top = df.nlargest(8, 'test_dir_acc')
    for _, row in top.iterrows():
        ax.annotate(row['ticker'],
                    (row['test_r2'], pct(row['test_dir_acc'])),
                    textcoords='offset points', xytext=(5, 3),
                    fontsize=8, color='white', alpha=0.9)

    # Regression line
    from numpy.polynomial.polynomial import polyfit
    x, y = df['test_r2'].values, pct(df['test_dir_acc']).values
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() > 2:
        c0, c1 = polyfit(x[mask], y[mask], 1)
        xs = np.linspace(x[mask].min(), x[mask].max(), 100)
        ax.plot(xs, c0 + c1*xs, color='gold', lw=2, ls='--', label='Trend line')

    ax.set_xlabel('Test R²', fontsize=12)
    ax.set_ylabel('Directional Accuracy (%)', fontsize=12)
    ax.set_title('Test R² vs Directional Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    save(fig, '5_r2_vs_diracc')
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 6 — RMSE & MAE distribution comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_error_distributions(df):
    plt.style.use(STYLE)
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_MED)

    for ax, col, color, title in [
        (axes[0], 'test_rmse', '#F48FB1', 'Test RMSE Distribution'),
        (axes[1], 'test_mae',  '#CE93D8', 'Test MAE Distribution'),
    ]:
        vals = df[col].dropna()
        ax.hist(vals, bins=35, color=color, edgecolor='white', alpha=0.85, density=True)

        from scipy.stats import gaussian_kde
        kde = gaussian_kde(vals)
        xs  = np.linspace(vals.min(), vals.max(), 300)
        ax.plot(xs, kde(xs), color='white', lw=2)

        ax.axvline(vals.mean(),   color='gold',      lw=2, ls='--', label=f'Mean   {vals.mean():.5f}')
        ax.axvline(vals.median(), color='limegreen', lw=2, ls=':',  label=f'Median {vals.median():.5f}')
        ax.set_xlabel(col.upper().replace('_', ' '), fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)

    plt.tight_layout()
    save(fig, '6_error_distributions')
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 7 — Train vs Test R² overfitting analysis
# ══════════════════════════════════════════════════════════════════════════════

def plot_overfitting_analysis(df):
    plt.style.use(STYLE)
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_MED)

    # Scatter: train r2 vs test r2
    ax = axes[0]
    gap = df['train_r2'] - df['test_r2']
    sc  = ax.scatter(df['train_r2'], df['test_r2'],
                     c=gap, cmap='RdYlGn_r', s=50, alpha=0.75,
                     edgecolors='white', lw=0.4)
    plt.colorbar(sc, ax=ax, label='Overfit Gap (Train − Test R²)')
    lim = [min(df['train_r2'].min(), df['test_r2'].min()) - 0.02,
           max(df['train_r2'].max(), df['test_r2'].max()) + 0.02]
    ax.plot(lim, lim, 'w--', lw=1.5, label='No overfit')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Train R²', fontsize=12)
    ax.set_ylabel('Test R²',  fontsize=12)
    ax.set_title('Train vs Test R²\n(above diagonal = overfit)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)

    # Distribution of overfit gap
    ax2 = axes[1]
    ax2.hist(gap, bins=35, color='#80DEEA', edgecolor='white', alpha=0.85, density=True)
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(gap.dropna())
    xs  = np.linspace(gap.min(), gap.max(), 300)
    ax2.plot(xs, kde(xs), color='white', lw=2)
    ax2.axvline(0,          color='limegreen', lw=1.5, ls='--', label='No gap')
    ax2.axvline(gap.mean(), color='gold',      lw=2,   ls='--', label=f'Mean gap {gap.mean():.4f}')
    ax2.set_xlabel('Overfitting Gap (Train − Test R²)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Distribution of Overfit Gap', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)

    plt.tight_layout()
    save(fig, '7_overfitting_analysis')
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 8 — Most frequent top features (bar chart)
# ══════════════════════════════════════════════════════════════════════════════

def plot_top_feature_frequency(df, n=20):
    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(12, 6))

    counts = df['top_feature'].value_counts().head(n)

    # Color by feature category
    def _cat_color(feat):
        if 'Return' in feat:    return '#4FC3F7'
        if 'Sentiment' in feat: return '#A5D6A7'
        if 'VIX'    in feat:       return '#FFB74D'
        if any(r in feat for r in ['roe','roa','op_margin','debt','liquidity',
                                   'current_ratio','free_cf','revenue','ocf']):
            return '#F48FB1'
        if 'GDP' in feat:       return '#CE93D8'
        if 'IPI' in feat:       return '#80DEEA'
        if 'UNEMP' in feat:     return '#FFCC80'
        return '#B0BEC5'

    colors = [_cat_color(f) for f in counts.index]
    bars   = ax.barh(counts.index, counts.values, color=colors, edgecolor='white', height=0.7)
    ax.invert_yaxis()

    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                str(v), va='center', fontsize=9)

    # Legend for categories
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor='#4FC3F7', label='Return Lags'),
        Patch(facecolor='#A5D6A7', label='Sentiment Lags'),
        Patch(facecolor='#FFB74D', label='VIX Lags'),
        Patch(facecolor='#F48FB1', label='Financial Ratios'),
        Patch(facecolor='#CE93D8', label='GDP'),
        Patch(facecolor='#80DEEA', label='IPI'),
        Patch(facecolor='#FFCC80', label='Unemployment'),
    ]
    ax.legend(handles=legend_els, fontsize=9, loc='lower right')

    ax.set_xlabel('Number of Tickers Where This is #1 Feature', fontsize=12)
    ax.set_title(f'Most Frequently Top-Ranked Feature (Top {n})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save(fig, '8_top_feature_frequency')
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — Run all plots
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    df = load_results(CSV_PATH)

    print("\n─── Plot 1: R² Distribution ───")
    plot_r2_distribution(df)

    print("\n─── Plot 2: Directional Accuracy Comparison ───")
    plot_directional_accuracy_comparison(df)

    print("\n─── Plot 3: Top/Bottom R² ───")
    plot_top_bottom_r2(df)

    print("\n─── Plot 4: Top/Bottom Directional Accuracy ───")
    plot_top_bottom_diracc(df)

    print("\n─── Plot 5: R² vs Directional Accuracy Scatter ───")
    plot_r2_vs_diracc(df)

    print("\n─── Plot 6: RMSE & MAE Distributions ───")
    plot_error_distributions(df)

    print("\n─── Plot 7: Overfitting Analysis ───")
    plot_overfitting_analysis(df)

    print("\n─── Plot 8: Top Feature Frequency ───")
    plot_top_feature_frequency(df)

    print(f"\n✅ All plots complete. PNGs saved to '{OUTPUT_DIR}/' directory.")
