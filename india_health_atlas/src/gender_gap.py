"""
india_health_atlas / src / gender_gap.py
──────────────────────────────────────────────────────────────────
WHAT THIS SCRIPT DOES  (plain English)
  Builds a "Gender Health Gap" score for every district.

  The idea: some districts look okay overall, but when you look
  ONLY at women's health indicators, they're doing terribly.
  That hidden gap is what this script exposes.

  We use 5 women-specific indicators:
    1. Anaemia in all women 15–49
    2. Child marriage rate (married before 18)
    3. Teenage pregnancy rate
    4. Unmet need for family planning
    5. Women with low BMI (undernourished)

  Then we compare each district's Gender Gap Score to its
  overall DHVS — finding districts where women's health is
  significantly worse than the overall picture suggests.

HOW TO RUN
  Run vulnerability_score.py first, then:
    python src/gender_gap.py
"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import load_raw_data

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV    = os.path.join(ROOT, 'data', 'raw',       'nfhs5_districts.csv')
SCORES_CSV = os.path.join(ROOT, 'data', 'processed', 'vulnerability_scores.csv')
OUTPUT_CSV = os.path.join(ROOT, 'data', 'processed', 'gender_gap_scores.csv')
CHART_PATH = os.path.join(ROOT, 'outputs', 'charts', 'gender_gap.png')

# 5 women-specific indicators
# invert=True means higher value = WORSE for women
GENDER_INDICATORS = {
    'anaemia_women': {
        'col':    'All women age 15-49 years who are anaemic22 (%)',
        'label':  'Anaemia in Women (15–49)',
        'weight': 0.25,
        'invert': False,
    },
    'child_marriage': {
        'col':    'Women age 20-24 years married before age 18 years (%)',
        'label':  'Child Marriage Rate',
        'weight': 0.25,
        'invert': False,
    },
    'teen_pregnancy': {
        'col':    'Women age 15-19 years who were already mothers or pregnant at the time of the survey (%)',
        'label':  'Teenage Pregnancy Rate',
        'weight': 0.20,
        'invert': False,
    },
    'unmet_fp': {
        'col':    'Total Unmet need for Family Planning (Currently Married Women Age 15-49  years)7 (%)',
        'label':  'Unmet Family Planning Need',
        'weight': 0.15,
        'invert': False,
    },
    'low_bmi': {
        'col':    'Women (age 15-49 years) whose Body Mass Index (BMI) is below normal (BMI <18.5 kg/m2)21 (%)',
        'label':  'Women with Low BMI (undernourished)',
        'weight': 0.15,
        'invert': False,
    },
}


def _clean_col(series):
    s = series.astype(str).str.strip()
    s = s.str.replace(r'^\((.+)\)$', r'\1', regex=True).replace('*', np.nan)
    return pd.to_numeric(s, errors='coerce')


def _minmax(series):
    mn, mx = series.min(), series.max()
    if mn == mx:
        return pd.Series(50.0, index=series.index)
    return (series - mn) / (mx - mn) * 100


def main():
    print()
    print("━" * 62)
    print("  India Health Atlas  ·  Gender Health Gap Analysis")
    print("━" * 62)

    for p in [RAW_CSV, SCORES_CSV]:
        if not os.path.exists(p):
            print(f"\n  ERROR — {os.path.basename(p)} not found.")
            print(f"  Run vulnerability_score.py first.")
            sys.exit(1)

    raw    = load_raw_data(RAW_CSV)
    scores = pd.read_csv(SCORES_CSV)

    # Build gender gap dataframe
    gdf = raw[['District Names', 'State/UT']].copy()
    gdf.columns = ['district', 'state']
    gdf['district'] = gdf['district'].str.strip()
    gdf['state']    = gdf['state'].str.strip()

    for key, meta in GENDER_INDICATORS.items():
        gdf[key] = _clean_col(raw[meta['col']])
        # Fill missing with state median
        gdf[key] = gdf.groupby('state')[key].transform(lambda x: x.fillna(x.median()))
        gdf[key] = gdf[key].fillna(gdf[key].median())

    # Normalize and score
    for key, meta in GENDER_INDICATORS.items():
        vals = gdf[key].copy()
        if meta.get('invert', False):
            vals = vals.max() - vals + vals.min()
        gdf[f'{key}_norm'] = _minmax(vals)

    gdf['gender_gap_score'] = sum(
        gdf[f'{k}_norm'] * meta['weight']
        for k, meta in GENDER_INDICATORS.items()
    ).round(2)

    gdf['gender_rank'] = gdf['gender_gap_score'].rank(ascending=False, method='min').astype(int)

    # Merge with overall DHVS to find the "hidden gap"
    merged = gdf.merge(
        scores[['district', 'state', 'dhvs', 'dhvs_rank']],
        on=['district', 'state'], how='left'
    )

    # Hidden gap = districts where gender score is much worse than overall DHVS
    # We normalize both to 0–1 and take the difference
    merged['dhvs_norm']   = _minmax(merged['dhvs']) / 100
    merged['gender_norm'] = _minmax(merged['gender_gap_score']) / 100
    merged['hidden_gap']  = (merged['gender_norm'] - merged['dhvs_norm']).round(3)
    merged['hidden_gap_rank'] = merged['hidden_gap'].rank(ascending=False, method='min').astype(int)

    # Classify
    merged['gender_band'] = pd.cut(
        merged['gender_gap_score'],
        bins=[0, 25, 50, 75, 100],
        labels=['Low', 'Moderate', 'High', 'Critical'],
        include_lowest=True
    )

    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Saved → data/processed/gender_gap_scores.csv")

    # ── Print findings ────────────────────────────────────────────────────────
    print()
    print("━" * 62)
    print("  FINDINGS")
    print("━" * 62)

    print("\n  10 Worst Districts — Women's Health:")
    print(f"  {'Rank':<6}{'District':<26}{'State':<22}{'Gender Score':>13}{'DHVS':>7}")
    print(f"  {'─'*4:<6}{'─'*24:<26}{'─'*20:<22}{'─'*11:>13}{'─'*5:>7}")
    for _, r in merged.nsmallest(1, 'gender_rank').iterrows():
        pass
    for _, r in merged.sort_values('gender_rank').head(10).iterrows():
        print(f"  #{r['gender_rank']:<5}{r['district']:<26}{r['state']:<22}"
              f"{r['gender_gap_score']:>13.1f}{r['dhvs']:>7.1f}")

    print("\n  10 Districts with Largest HIDDEN Gender Gap")
    print("  (overall health looks OK, but women's health is much worse):")
    print(f"  {'District':<26}{'State':<22}{'Gap Score':>10}{'Gender':>8}{'DHVS':>7}")
    print(f"  {'─'*24:<26}{'─'*20:<22}{'─'*8:>10}{'─'*6:>8}{'─'*5:>7}")
    for _, r in merged.sort_values('hidden_gap', ascending=False).head(10).iterrows():
        print(f"  {r['district']:<26}{r['state']:<22}"
              f"{r['hidden_gap']:>10.2f}{r['gender_gap_score']:>8.1f}{r['dhvs']:>7.1f}")

    _make_chart(merged, CHART_PATH)

    print()
    print("━" * 62)
    print("  Gender gap analysis complete.")
    print("  Next  →  run  src/change_tracker.py")
    print("━" * 62)
    print()


def _make_chart(merged, save_path):
    BG, CARD, GRID = '#0b0f19', '#111827', '#1f2937'
    TEXT, MUTE     = '#e5e7eb', '#6b7280'
    PINK, BLUE     = '#f472b6', '#60a5fa'

    fig = plt.figure(figsize=(20, 13), facecolor=BG)
    fig.subplots_adjust(left=0.04, right=0.96, top=0.88, bottom=0.07, wspace=0.28, hspace=0.4)
    gs = gridspec.GridSpec(2, 2, figure=fig)

    ax_scatter = fig.add_subplot(gs[:, 0])   # left — full height scatter
    ax_top     = fig.add_subplot(gs[0, 1])   # top right — worst women's health
    ax_hidden  = fig.add_subplot(gs[1, 1])   # bottom right — hidden gap

    # ── Panel 1: Scatter — Gender Score vs DHVS ──────────────────────────────
    ax = ax_scatter
    ax.set_facecolor(CARD)
    for spine in ax.spines.values(): spine.set_color(GRID)

    med_g = merged['gender_gap_score'].median()
    med_d = merged['dhvs'].median()
    ax.axhline(med_g, color=GRID, lw=0.8, linestyle='--')
    ax.axvline(med_d, color=GRID, lw=0.8, linestyle='--')

    # Hidden gap districts (gender much worse than overall)
    hidden = merged[merged['hidden_gap'] > merged['hidden_gap'].quantile(0.85)]
    normal = merged[~merged.index.isin(hidden.index)]

    ax.scatter(normal['dhvs'], normal['gender_gap_score'],
               color='#4b5563', alpha=0.35, s=20, linewidths=0)
    ax.scatter(hidden['dhvs'], hidden['gender_gap_score'],
               color=PINK, alpha=0.7, s=35, linewidths=0, label='Hidden gender gap')

    for _, r in hidden.nlargest(7, 'hidden_gap').iterrows():
        ax.annotate(f"  {r['district'][:14]}", xy=(r['dhvs'], r['gender_gap_score']),
                    color=TEXT, fontsize=7.5, zorder=5)

    ax.set_xlabel('Overall DHVS Score  →', color=MUTE, fontsize=9)
    ax.set_ylabel('Gender Gap Score  →', color=MUTE, fontsize=9)
    ax.set_title("Gender Score vs. Overall Vulnerability\nHighlighted: where women's health is disproportionately worse",
                 color=TEXT, fontsize=10.5, fontweight='700', pad=10, loc='left')
    ax.tick_params(colors=MUTE, labelsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.5, linestyle=':')
    ax.xaxis.grid(True, color=GRID, lw=0.5, linestyle=':')
    ax.set_axisbelow(True)
    ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

    # ── Panel 2: Top 12 worst gender gap districts ───────────────────────────
    ax = ax_top
    ax.set_facecolor(CARD)
    for spine in ax.spines.values(): spine.set_color(GRID)

    top12 = merged.nlargest(12, 'gender_gap_score').iloc[::-1]
    labels = [f"{r['district'][:18]}  ·  {r['state'][:8]}" for _, r in top12.iterrows()]
    bars = ax.barh(labels, top12['gender_gap_score'], color=PINK, alpha=0.8, height=0.65)
    for bar, (_, row) in zip(bars, top12.iterrows()):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{row["gender_gap_score"]:.1f}', va='center', color=TEXT, fontsize=8)

    ax.set_facecolor(CARD)
    ax.tick_params(colors=MUTE, labelsize=8.5)
    ax.set_xlabel('Gender Gap Score', color=MUTE, fontsize=8.5)
    ax.set_title('12 Worst Districts — Women\'s Health', color=TEXT,
                 fontsize=10.5, fontweight='700', pad=10, loc='left')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.xaxis.grid(True, color=GRID, lw=0.5, linestyle=':')
    ax.set_axisbelow(True)

    # ── Panel 3: Hidden gap — top 12 ─────────────────────────────────────────
    ax = ax_hidden
    ax.set_facecolor(CARD)
    for spine in ax.spines.values(): spine.set_color(GRID)

    top_hidden = merged.nlargest(12, 'hidden_gap').iloc[::-1]
    labels_h = [f"{r['district'][:18]}  ·  {r['state'][:8]}" for _, r in top_hidden.iterrows()]
    bars_h = ax.barh(labels_h, top_hidden['hidden_gap'], color=BLUE, alpha=0.8, height=0.65)
    for bar, (_, row) in zip(bars_h, top_hidden.iterrows()):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                f'+{row["hidden_gap"]:.2f}', va='center', color=TEXT, fontsize=8)

    ax.tick_params(colors=MUTE, labelsize=8.5)
    ax.set_xlabel('Hidden Gap Score  (gender score − overall DHVS, normalized)', color=MUTE, fontsize=8)
    ax.set_title('Hidden Gender Gap  —  overall looks OK, women\'s health is not',
                 color=TEXT, fontsize=10.5, fontweight='700', pad=10, loc='left')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.xaxis.grid(True, color=GRID, lw=0.5, linestyle=':')
    ax.set_axisbelow(True)

    fig.text(0.5, 0.945, 'India Health Atlas  —  Gender Health Gap Analysis',
             ha='center', color=TEXT, fontsize=17, fontweight='800')
    fig.text(0.5, 0.916,
             'Districts where women\'s health diverges from the overall picture  ·  NFHS-5  ·  706 Districts',
             ha='center', color=MUTE, fontsize=9.5)

    plt.savefig(save_path, dpi=160, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  Saved → outputs/charts/gender_gap.png")


if __name__ == '__main__':
    main()