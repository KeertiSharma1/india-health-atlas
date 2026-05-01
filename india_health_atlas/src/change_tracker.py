"""
india_health_atlas / src / change_tracker.py
──────────────────────────────────────────────────────────────────
WHAT THIS SCRIPT DOES  (plain English)
  Compares health progress across two time periods using the
  NFHS-5 data itself.

  Since we only have NFHS-5 (not NFHS-4 district data yet),
  this script uses a smart proxy approach:
    — We compare each district to its STATE average
    — Then we simulate improvement by quartile-ranking districts
    — We classify every district into 4 quadrants based on
       their score AND how they rank vs their peers

  The 4 categories are:
    ● Bad & Declining  — high vulnerability AND below state avg
    ● Bad but Improving — high vulnerability BUT above state avg
    ● Good & Improving  — low vulnerability AND above state avg
    ● Good but Declining — low vulnerability BUT below state avg (hidden risk)

  Note: When you download NFHS-4 data later, replace
  `data/raw/nfhs4_districts.csv` and re-run this script.
  The code already handles NFHS-4 if present.

HOW TO RUN
  python src/change_tracker.py
"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import load_raw_data, load_config, run_pipeline

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV    = os.path.join(ROOT, 'data', 'raw',       'nfhs5_districts.csv')
NFHS4_CSV  = os.path.join(ROOT, 'data', 'raw',       'nfhs4_districts.csv')
SCORES_CSV = os.path.join(ROOT, 'data', 'processed', 'vulnerability_scores.csv')
OUTPUT_CSV = os.path.join(ROOT, 'data', 'processed', 'change_tracker.csv')
CHART_PATH = os.path.join(ROOT, 'outputs', 'charts',  'change_tracker.png')
CONFIG     = os.path.join(ROOT, 'config', 'indicators.yaml')

QUAD_COLORS = {
    'Bad & Declining':    '#ef4444',
    'Bad but Improving':  '#f97316',
    'Good but Declining': '#facc15',
    'Good & Improving':   '#22c55e',
}
QUAD_DESC = {
    'Bad & Declining':    'Emergency — high vulnerability, falling behind peers',
    'Bad but Improving':  'Needs support — still vulnerable but moving right direction',
    'Good but Declining': 'Hidden risk — healthy now but showing early warning signs',
    'Good & Improving':   'Model districts — low vulnerability, outperforming peers',
}


def classify_districts(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Classify every district into one of 4 quadrants.

    If NFHS-4 data is available → uses real delta (score change over time).
    If not → uses district vs state-average comparison as a proxy.
    """
    df = scores.copy()

    if os.path.exists(NFHS4_CSV):
        print("  NFHS-4 data found — computing real score delta …")
        nfhs4 = run_pipeline(NFHS4_CSV, CONFIG)
        nfhs4 = nfhs4.rename(columns={'dhvs': 'dhvs_nfhs4'})[['district', 'state', 'dhvs_nfhs4']]
        df = df.merge(nfhs4, on=['district', 'state'], how='left')
        df['score_delta'] = df['dhvs'] - df['dhvs_nfhs4']   # positive = worsened
        mode = "NFHS-4 vs NFHS-5 real delta"
    else:
        print("  NFHS-4 file not found — using state-comparison proxy …")
        state_med = df.groupby('state')['dhvs'].transform('median')
        df['score_delta'] = df['dhvs'] - state_med   # positive = worse than state median
        mode = "district vs state median (NFHS-4 not yet available)"

    print(f"  Classification mode: {mode}")

    # Classify into 4 quadrants
    # Thresholds: overall DHVS median splits good/bad; delta sign splits improving/declining
    dhvs_median  = df['dhvs'].median()
    delta_median = df['score_delta'].median()

    def assign_quadrant(row):
        bad       = row['dhvs'] >= dhvs_median
        declining = row['score_delta'] >= delta_median
        if bad and declining:       return 'Bad & Declining'
        if bad and not declining:   return 'Bad but Improving'
        if not bad and declining:   return 'Good but Declining'
        return 'Good & Improving'

    df['quadrant']    = df.apply(assign_quadrant, axis=1)
    df['delta_rank']  = df['score_delta'].rank(ascending=True, method='min').astype(int)

    return df, mode


def main():
    print()
    print("━" * 62)
    print("  India Health Atlas  ·  Change Tracker")
    print("━" * 62)

    if not os.path.exists(SCORES_CSV):
        print("\n  ERROR — vulnerability_scores.csv not found.")
        print("  Run vulnerability_score.py first.")
        sys.exit(1)

    scores = pd.read_csv(SCORES_CSV)
    result, mode = classify_districts(scores)
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Saved → data/processed/change_tracker.csv")

    # ── Print findings ────────────────────────────────────────────────────────
    print()
    print("━" * 62)
    print("  FINDINGS")
    print("━" * 62)
    print(f"\n  Classification mode: {mode}")
    print()

    counts = result['quadrant'].value_counts()
    for quad, count in counts.items():
        pct = count / len(result) * 100
        bar = '█' * (count // 8)
        print(f"  {quad:<25}  {count:>4} ({pct:.0f}%)  {bar}")
        print(f"    → {QUAD_DESC[quad]}")
        print()

    print("  TOP 10 MOST CONCERNING — Bad & Declining:")
    subset = result[result['quadrant'] == 'Bad & Declining'].nlargest(10, 'dhvs')
    for _, r in subset.iterrows():
        print(f"    {r['district']:<26} {r['state']:<22} score {r['dhvs']:.1f}")

    print("\n  TOP 10 HIDDEN RISK — Good but Declining:")
    subset = result[result['quadrant'] == 'Good but Declining'].nlargest(10, 'score_delta')
    for _, r in subset.iterrows():
        print(f"    {r['district']:<26} {r['state']:<22} score {r['dhvs']:.1f}")

    print("\n  TOP 10 MODEL DISTRICTS — Good & Improving:")
    subset = result[result['quadrant'] == 'Good & Improving'].nsmallest(10, 'dhvs')
    for _, r in subset.iterrows():
        print(f"    {r['district']:<26} {r['state']:<22} score {r['dhvs']:.1f}")

    _make_chart(result, CHART_PATH)

    print()
    print("━" * 62)
    print("  Change tracker complete.")
    print("  Next  →  run  app/dashboard.py  (Streamlit dashboard)")
    print("━" * 62)
    print()


def _make_chart(result, save_path):
    BG, CARD, GRID = '#0b0f19', '#111827', '#1f2937'
    TEXT, MUTE     = '#e5e7eb', '#6b7280'

    fig = plt.figure(figsize=(20, 13), facecolor=BG)
    fig.subplots_adjust(left=0.04, right=0.96, top=0.88, bottom=0.07, wspace=0.26, hspace=0.38)

    ax_scatter = fig.add_subplot(2, 2, (1, 3))   # left column, full height
    ax_bad     = fig.add_subplot(2, 2, 2)
    ax_model   = fig.add_subplot(2, 2, 4)

    # ── Scatter ───────────────────────────────────────────────────────────────
    ax = ax_scatter
    ax.set_facecolor(CARD)
    for spine in ax.spines.values(): spine.set_color(GRID)

    for quad, color in QUAD_COLORS.items():
        subset = result[result['quadrant'] == quad]
        ax.scatter(subset['score_delta'], subset['dhvs'],
                   color=color, alpha=0.6, s=28, linewidths=0, label=quad, zorder=3)

    dhvs_med  = result['dhvs'].median()
    delta_med = result['score_delta'].median()
    ax.axhline(dhvs_med,  color='#374151', lw=0.9, linestyle='--', zorder=1)
    ax.axvline(delta_med, color='#374151', lw=0.9, linestyle='--', zorder=1)

    # Quadrant labels
    xmin, xmax = result['score_delta'].min(), result['score_delta'].max()
    ymin, ymax = result['dhvs'].min(),        result['dhvs'].max()
    offx = (xmax - xmin) * 0.03
    offy = (ymax - ymin) * 0.03

    ax.text(xmax - offx, ymax - offy, 'Bad &\nDeclining',
            ha='right', va='top', color='#ef4444', fontsize=9, fontweight='600', alpha=0.8)
    ax.text(xmin + offx, ymax - offy, 'Bad but\nImproving',
            ha='left',  va='top', color='#f97316', fontsize=9, fontweight='600', alpha=0.8)
    ax.text(xmax - offx, ymin + offy, 'Good but\nDeclining',
            ha='right', va='bottom', color='#facc15', fontsize=9, fontweight='600', alpha=0.8)
    ax.text(xmin + offx, ymin + offy, 'Good &\nImproving',
            ha='left',  va='bottom', color='#22c55e', fontsize=9, fontweight='600', alpha=0.8)

    ax.set_xlabel('Score relative to state median  →  positive = worse than peers',
                  color=MUTE, fontsize=9)
    ax.set_ylabel('DHVS Vulnerability Score  →', color=MUTE, fontsize=9)
    ax.set_title('District Classification — 4 Quadrants',
                 color=TEXT, fontsize=11, fontweight='700', pad=10, loc='left')
    ax.tick_params(colors=MUTE, labelsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.5, linestyle=':')
    ax.xaxis.grid(True, color=GRID, lw=0.5, linestyle=':')
    ax.set_axisbelow(True)
    ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=8.5)

    # ── Top 12 Bad & Declining ────────────────────────────────────────────────
    ax = ax_bad
    ax.set_facecolor(CARD)
    for spine in ax.spines.values(): spine.set_color(GRID)
    bad12 = result[result['quadrant'] == 'Bad & Declining'].nlargest(12, 'dhvs').iloc[::-1]
    labels = [f"{r['district'][:18]}  ·  {r['state'][:8]}" for _, r in bad12.iterrows()]
    bars = ax.barh(labels, bad12['dhvs'], color='#ef4444', alpha=0.8, height=0.65)
    for bar, (_, row) in zip(bars, bad12.iterrows()):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{row["dhvs"]:.1f}', va='center', color=TEXT, fontsize=8)
    ax.tick_params(colors=MUTE, labelsize=8.5)
    ax.set_title('Bad & Declining  —  Emergency Priority', color='#ef4444',
                 fontsize=10.5, fontweight='700', pad=10, loc='left')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.xaxis.grid(True, color=GRID, lw=0.5, linestyle=':')
    ax.set_axisbelow(True)

    # ── Top 12 Model Districts ────────────────────────────────────────────────
    ax = ax_model
    ax.set_facecolor(CARD)
    for spine in ax.spines.values(): spine.set_color(GRID)
    model12 = result[result['quadrant'] == 'Good & Improving'].nsmallest(12, 'dhvs').iloc[::-1]
    labels_m = [f"{r['district'][:18]}  ·  {r['state'][:8]}" for _, r in model12.iterrows()]
    bars_m = ax.barh(labels_m, model12['dhvs'], color='#22c55e', alpha=0.8, height=0.65)
    for bar, (_, row) in zip(bars_m, model12.iterrows()):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{row["dhvs"]:.1f}', va='center', color=TEXT, fontsize=8)
    ax.tick_params(colors=MUTE, labelsize=8.5)
    ax.set_title('Good & Improving  —  Model Districts', color='#22c55e',
                 fontsize=10.5, fontweight='700', pad=10, loc='left')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.xaxis.grid(True, color=GRID, lw=0.5, linestyle=':')
    ax.set_axisbelow(True)

    # Titles
    fig.text(0.5, 0.945, 'India Health Atlas  —  District Change Tracker',
             ha='center', color=TEXT, fontsize=17, fontweight='800')
    fig.text(0.5, 0.916,
             'Each district classified into one of 4 quadrants based on vulnerability score and peer comparison  ·  NFHS-5',
             ha='center', color=MUTE, fontsize=9.5)

    # Legend patches at bottom
    handles = [mpatches.Patch(color=c, label=f'{q}  —  {QUAD_DESC[q][:42]}')
               for q, c in QUAD_COLORS.items()]
    fig.legend(handles=handles, loc='lower center', ncol=2, facecolor=CARD,
               edgecolor=GRID, labelcolor=TEXT, fontsize=8, bbox_to_anchor=(0.5, 0.0))

    plt.savefig(save_path, dpi=160, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  Saved → outputs/charts/change_tracker.png")


if __name__ == '__main__':
    main()