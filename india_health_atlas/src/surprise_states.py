"""
india_health_atlas / src / surprise_states.py
──────────────────────────────────────────────────────────────────
WHAT THIS SCRIPT DOES  (plain English)
  Finds "surprise" districts — ones that defy expectations:

  ● Richer but sicker  — district has better-than-average literacy
    and electricity access (proxies for development/wealth) but
    WORSE-than-average health outcomes.  These are policy failures.

  ● Poorer but healthier — district is below average on development
    indicators but BETTER-than-average health outcomes.
    These are model districts — what are they doing right?

  The scatter plot puts every district on a chart:
    X-axis → Development Proxy (literacy + electricity)
    Y-axis → Vulnerability Score (DHVS)
  Then circles the surprises and labels them.

HOW TO RUN
  Run vulnerability_score.py first (it creates the scored CSV).
  Then:
    python src/surprise_states.py

REQUIRES
  pip install pandas numpy matplotlib pyyaml
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import load_raw_data, load_config

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV    = os.path.join(ROOT, 'data', 'raw',       'nfhs5_districts.csv')
SCORES_CSV = os.path.join(ROOT, 'data', 'processed', 'vulnerability_scores.csv')
CHART_PATH = os.path.join(ROOT, 'outputs', 'charts', 'surprise_states.png')
OUTPUT_CSV = os.path.join(ROOT, 'data', 'processed', 'surprise_districts.csv')

# Development proxy columns from NFHS-5
# (We don't have income data, so we use literacy + electricity as proxies)
LITERACY_COL     = 'Women (age 15-49) who are literate4 (%)'
ELECTRICITY_COL  = 'Population living in households with electricity (%)'
SANITATION_COL   = 'Population living in households that use an improved sanitation facility2 (%)'


def main():
    print()
    print("━" * 62)
    print("  India Health Atlas  ·  Surprise States Analysis")
    print("━" * 62)

    # Check prerequisite
    for path, name in [(SCORES_CSV, 'vulnerability_scores.csv'), (RAW_CSV, 'nfhs5_districts.csv')]:
        if not os.path.exists(path):
            print(f"\n  ERROR — {name} not found.")
            print(f"  Run  src/vulnerability_score.py  first.")
            sys.exit(1)

    # Load scored data
    scores = pd.read_csv(SCORES_CSV)
    raw    = load_raw_data(RAW_CSV)
    raw.columns = raw.columns.str.strip()

    # Build development proxy from raw file
    dev = raw[['District Names', 'State/UT']].copy()
    dev.columns = ['district', 'state']

    for col in [LITERACY_COL, ELECTRICITY_COL, SANITATION_COL]:
        key = col[:12].strip().lower().replace(' ', '_')
        raw_vals = raw[col].astype(str).str.strip()
        raw_vals = raw_vals.str.replace(r'^\((.+)\)$', r'\1', regex=True).replace('*', np.nan)
        dev[key] = pd.to_numeric(raw_vals, errors='coerce')

    # Fill missing
    for col in dev.columns[2:]:
        dev[col] = dev.groupby('state')[col].transform(lambda x: x.fillna(x.median()))
        dev[col] = dev[col].fillna(dev[col].median())

    # Composite development proxy (simple average of 3 indicators)
    dev_cols = list(dev.columns[2:])
    dev['development_proxy'] = dev[dev_cols].mean(axis=1).round(2)

    # Merge with DHVS scores
    merged = scores.merge(dev[['district', 'state', 'development_proxy']],
                          on=['district', 'state'], how='left')

    # ── Identify surprises ────────────────────────────────────────────────────
    # Quadrants defined by median split
    med_dev  = merged['development_proxy'].median()
    med_dhvs = merged['dhvs'].median()

    merged['surprise_type'] = 'Expected'
    # Richer but sicker: above median development, above median vulnerability
    mask_rs = (merged['development_proxy'] >= med_dev) & (merged['dhvs'] >= med_dhvs)
    # Poorer but healthier: below median development, below median vulnerability
    mask_ph = (merged['development_proxy'] <  med_dev) & (merged['dhvs'] <  med_dhvs)

    merged.loc[mask_rs, 'surprise_type'] = 'Richer but Sicker'
    merged.loc[mask_ph, 'surprise_type'] = 'Poorer but Healthier'

    # Top surprises to label on chart
    top_rs = merged[mask_rs].nlargest(8,  'dhvs')
    top_ph = merged[mask_ph].nsmallest(8, 'dhvs')

    # Save all classifications
    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Saved → data/processed/surprise_districts.csv")

    # Print findings
    print()
    print("━" * 62)
    print("  FINDINGS")
    print("━" * 62)
    print(f"\n  Development proxy median : {med_dev:.1f}")
    print(f"  Vulnerability score median: {med_dhvs:.1f}")
    print(f"\n  Quadrant counts:")
    for qt, cnt in merged['surprise_type'].value_counts().items():
        print(f"    {qt:<25}  {cnt} districts")

    print(f"\n  TOP 'RICHER BUT SICKER' DISTRICTS (policy failures):")
    print(f"  {'District':<26}{'State':<22}{'Dev Proxy':>10}{'DHVS':>8}")
    print(f"  {'─'*24:<26}{'─'*20:<22}{'─'*8:>10}{'─'*6:>8}")
    for _, r in top_rs.head(8).iterrows():
        print(f"  {r['district']:<26}{r['state']:<22}{r['development_proxy']:>10.1f}{r['dhvs']:>8.1f}")

    print(f"\n  TOP 'POORER BUT HEALTHIER' DISTRICTS (model districts):")
    print(f"  {'District':<26}{'State':<22}{'Dev Proxy':>10}{'DHVS':>8}")
    print(f"  {'─'*24:<26}{'─'*20:<22}{'─'*8:>10}{'─'*6:>8}")
    for _, r in top_ph.head(8).iterrows():
        print(f"  {r['district']:<26}{r['state']:<22}{r['development_proxy']:>10.1f}{r['dhvs']:>8.1f}")

    # Make chart
    _make_chart(merged, top_rs, top_ph, med_dev, med_dhvs, CHART_PATH)

    print()
    print("━" * 62)
    print("  Surprise analysis complete.")
    print("  Next  →  run  src/gender_gap.py")
    print("━" * 62)
    print()


def _make_chart(merged, top_rs, top_ph, med_dev, med_dhvs, save_path):
    BG   = '#0b0f19'
    CARD = '#111827'
    GRID = '#1f2937'
    TEXT = '#e5e7eb'
    MUTE = '#6b7280'

    fig, ax = plt.subplots(figsize=(16, 11), facecolor=BG)
    ax.set_facecolor(CARD)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.1)

    # Quadrant shading
    ax.axhspan(med_dhvs, merged['dhvs'].max() + 5,
               xmin=0, xmax=1, alpha=0.04, color='#ef4444', zorder=0)
    ax.axhspan(merged['dhvs'].min() - 2, med_dhvs,
               xmin=0, xmax=1, alpha=0.04, color='#22c55e', zorder=0)

    # Median lines
    ax.axhline(med_dhvs, color='#374151', linewidth=0.9, linestyle='--', zorder=1)
    ax.axvline(med_dev,  color='#374151', linewidth=0.9, linestyle='--', zorder=1)

    # Quadrant labels
    ax.text(merged['development_proxy'].max() - 1, merged['dhvs'].max() - 1,
            'Richer but Sicker\n(Policy Failure Zone)',
            ha='right', va='top', color='#ef4444', fontsize=9, alpha=0.7,
            fontweight='600')
    ax.text(merged['development_proxy'].min() + 1, merged['dhvs'].min() + 1,
            'Poorer but Healthier\n(Model Districts)',
            ha='left', va='bottom', color='#22c55e', fontsize=9, alpha=0.7,
            fontweight='600')
    ax.text(merged['development_proxy'].max() - 1, merged['dhvs'].min() + 1,
            'Expected: Developed & Healthy',
            ha='right', va='bottom', color=MUTE, fontsize=8.5, alpha=0.6)
    ax.text(merged['development_proxy'].min() + 1, merged['dhvs'].max() - 1,
            'Expected: Underdeveloped & Sick',
            ha='left', va='top', color=MUTE, fontsize=8.5, alpha=0.6)

    # All districts — background dots
    expected = merged[merged['surprise_type'] == 'Expected']
    ax.scatter(expected['development_proxy'], expected['dhvs'],
               color='#4b5563', alpha=0.35, s=22, zorder=2, linewidths=0)

    # Richer but sicker
    rs = merged[merged['surprise_type'] == 'Richer but Sicker']
    ax.scatter(rs['development_proxy'], rs['dhvs'],
               color='#ef4444', alpha=0.55, s=30, zorder=3, linewidths=0)

    # Poorer but healthier
    ph = merged[merged['surprise_type'] == 'Poorer but Healthier']
    ax.scatter(ph['development_proxy'], ph['dhvs'],
               color='#22c55e', alpha=0.55, s=30, zorder=3, linewidths=0)

    # Label top surprises
    def label_points(points, color):
        for _, row in points.iterrows():
            ax.scatter(row['development_proxy'], row['dhvs'],
                       color=color, s=70, zorder=5, linewidths=0)
            ax.annotate(
                f"  {row['district']}\n  {row['state'][:12]}",
                xy=(row['development_proxy'], row['dhvs']),
                fontsize=7.5, color=TEXT, zorder=6,
                arrowprops=dict(arrowstyle='-', color='#374151', lw=0.8),
                xytext=(row['development_proxy'] + np.random.choice([-8, 8]),
                        row['dhvs'] + np.random.choice([-3, 3])),
            )

    label_points(top_rs.head(6), '#ef4444')
    label_points(top_ph.head(6), '#22c55e')

    # Styling
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTE, labelsize=9)
    ax.set_xlabel('Development Proxy  (literacy + electricity + sanitation)  →  higher is better developed',
                  color=MUTE, fontsize=9.5, labelpad=10)
    ax.set_ylabel('Vulnerability Score (DHVS)  →  higher is more vulnerable',
                  color=MUTE, fontsize=9.5, labelpad=10)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5, linestyle=':')
    ax.xaxis.grid(True, color=GRID, linewidth=0.5, linestyle=':')
    ax.set_axisbelow(True)

    # Legend
    handles = [
        mpatches.Patch(color='#ef4444', label='Richer but Sicker  (policy failures)'),
        mpatches.Patch(color='#22c55e', label='Poorer but Healthier  (model districts)'),
        mpatches.Patch(color='#4b5563', label='Expected pattern'),
    ]
    ax.legend(handles=handles, loc='upper left', facecolor=CARD,
              edgecolor=GRID, labelcolor=TEXT, fontsize=9, framealpha=0.9)

    # Titles
    fig.text(0.5, 0.945,
             'India Health Atlas  —  Surprise States Analysis',
             ha='center', color=TEXT, fontsize=17, fontweight='800')
    fig.text(0.5, 0.915,
             'Districts that defy expected development-health relationships  ·  NFHS-5  ·  706 Districts',
             ha='center', color=MUTE, fontsize=9.5)

    plt.savefig(save_path, dpi=160, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  Saved → outputs/charts/surprise_states.png")


if __name__ == '__main__':
    main()