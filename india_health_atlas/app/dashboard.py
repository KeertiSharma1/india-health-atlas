"""
india_health_atlas / app / dashboard.py
──────────────────────────────────────────────────────────────────
FULL STREAMLIT DASHBOARD — 5 pages:
  1. Overview          — key headline numbers, India summary
  2. Atlas Rankings    — full district leaderboard with filters
  3. Gender Gap        — where women's health diverges
  4. Change Tracker    — 4-quadrant district classification
  5. What-If Simulator — adjust one indicator, see score change

HOW TO RUN (from india_health_atlas/ folder):
  streamlit run app/dashboard.py
"""

import os, sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ── Allow imports from src/ ───────────────────────────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(APP_DIR)
sys.path.insert(0, os.path.join(ROOT, 'src'))
from scoring import load_config

# ── File paths ────────────────────────────────────────────────────────────────
SCORES_CSV   = os.path.join(ROOT, 'data', 'processed', 'vulnerability_scores.csv')
GENDER_CSV   = os.path.join(ROOT, 'data', 'processed', 'gender_gap_scores.csv')
SURPRISE_CSV = os.path.join(ROOT, 'data', 'processed', 'surprise_districts.csv')
CHANGE_CSV   = os.path.join(ROOT, 'data', 'processed', 'change_tracker.csv')
CONFIG       = os.path.join(ROOT, 'config', 'indicators.yaml')

# ── Colours ───────────────────────────────────────────────────────────────────
BG_DARK  = '#0b0f19'
CARD     = '#111827'
BORDER   = '#1f2937'
GRID_COL = '#1f2937'
TEXT     = '#f9fafb'
MUTED    = '#6b7280'
BAND_COLORS = {
    'Low':      '#22c55e',
    'Moderate': '#f59e0b',
    'High':     '#ef4444',
    'Critical': '#dc2626',
}
QUAD_COLORS = {
    'Bad & Declining':    '#ef4444',
    'Bad but Improving':  '#f97316',
    'Good but Declining': '#facc15',
    'Good & Improving':   '#22c55e',
}


# ── Plotly layout helper ──────────────────────────────────────────────────────
# Returns a dict of layout kwargs.
# xaxis/yaxis overrides are MERGED with the defaults — never duplicated.
def layout(**overrides):
    base_xaxis = dict(gridcolor=GRID_COL, linecolor=BORDER,
                      tickfont=dict(color=MUTED), zerolinecolor=GRID_COL)
    base_yaxis = dict(gridcolor=GRID_COL, linecolor=BORDER,
                      tickfont=dict(color=MUTED), zerolinecolor=GRID_COL)

    if 'xaxis' in overrides:
        base_xaxis = {**base_xaxis, **overrides.pop('xaxis')}
    if 'yaxis' in overrides:
        base_yaxis = {**base_yaxis, **overrides.pop('yaxis')}

    result = dict(
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font=dict(family='DM Sans, sans-serif', color=MUTED, size=12),
        margin=dict(l=20, r=20, t=50, b=30),
        legend=dict(bgcolor=CARD, bordercolor=BORDER, font=dict(color=TEXT)),
        xaxis=base_xaxis,
        yaxis=base_yaxis,
    )
    result.update(overrides)
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG & GLOBAL STYLE
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="India Health Atlas",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main  { background: #0b0f19; }
.block-container { padding: 2rem 2.5rem 3rem; max-width: 1400px; }

.metric-card {
  background: #111827; border: 1px solid #1f2937;
  border-radius: 12px; padding: 18px 22px; height: 100%;
}
.metric-label { color:#6b7280; font-size:11px; font-weight:600;
  letter-spacing:.08em; text-transform:uppercase; margin-bottom:6px; }
.metric-value { color:#f9fafb; font-size:26px; font-weight:800; line-height:1.1; }
.metric-sub   { color:#9ca3af; font-size:12px; margin-top:5px; }

.section-header {
  color:#f9fafb; font-size:17px; font-weight:700;
  margin: 1.8rem 0 .6rem; padding-bottom:8px;
  border-bottom: 1px solid #1f2937;
}
.page-title    { font-size:27px; font-weight:800; color:#f9fafb;
  letter-spacing:-.02em; margin-bottom:2px; }
.page-subtitle { font-size:13.5px; color:#6b7280; margin-bottom:1.6rem; }

.note-box {
  background:#111827; border-left:4px solid #6b7280;
  border-radius:8px; padding:14px 18px;
  color:#9ca3af; font-size:13.5px; line-height:1.6;
}

section[data-testid="stSidebar"] { background:#0d1117; border-right:1px solid #1f2937; }
#MainMenu, footer, header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏥 India Health Atlas")
    st.markdown(
        "<p style='color:#6b7280;font-size:13px;margin-top:-8px;margin-bottom:20px;'>"
        "NFHS-5 · 706 Districts</p>",
        unsafe_allow_html=True,
    )
    st.divider()
    page = st.radio(
        "Navigate",
        ["Overview", "Atlas Rankings", "Gender Gap", "Change Tracker", "What-If Simulator"],
        label_visibility='collapsed',
    )
    st.divider()
    st.markdown(
        "<p style='color:#374151;font-size:11px;'>Data: NFHS-5 (2019–21)<br>"
        "Source: IIPS, Government of India</p>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_all():
    paths = [SCORES_CSV, GENDER_CSV, SURPRISE_CSV, CHANGE_CSV]
    missing = [os.path.basename(p) for p in paths if not os.path.exists(p)]
    if missing:
        return None, None, None, None, missing
    return (
        pd.read_csv(SCORES_CSV),
        pd.read_csv(GENDER_CSV),
        pd.read_csv(SURPRISE_CSV),
        pd.read_csv(CHANGE_CSV),
        [],
    )


scores, gender, surprise, change, missing_files = load_all()

if missing_files:
    st.error(f"Missing processed data files: {', '.join(missing_files)}")
    st.info(
        "Run these four scripts first from the `india_health_atlas/` folder:\n\n"
        "```\npython src/vulnerability_score.py\n"
        "python src/surprise_states.py\n"
        "python src/gender_gap.py\n"
        "python src/change_tracker.py\n```"
    )
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
#  SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def metric_card(label, value, sub, border_color=BORDER):
    fs = '20px' if len(str(value)) > 14 else '24px'
    st.markdown(
        f"""<div class="metric-card" style="border-color:{border_color}">
          <div class="metric-label">{label}</div>
          <div class="metric-value" style="font-size:{fs}">{value}</div>
          <div class="metric-sub">{sub}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "Overview":
    st.markdown('<div class="page-title">India Health Atlas</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">District Health Vulnerability Score (DHVS) · '
        'NFHS-5 · 706 Districts · 36 States & UTs</div>',
        unsafe_allow_html=True,
    )

    worst_state = scores.groupby('state')['dhvs'].mean().idxmax()
    n_high      = int((scores['vulnerability_band'].isin(['High', 'Critical'])).sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("Districts Mapped",  "706",                       "36 states & UTs")
    with c2: metric_card("Most Vulnerable",   scores.iloc[0]['district'],  scores.iloc[0]['state'])
    with c3: metric_card("Least Vulnerable",  scores.iloc[-1]['district'], scores.iloc[-1]['state'])
    with c4: metric_card("High/Critical Risk",str(n_high),                 "districts above score 50")
    with c5: metric_card("Worst Avg State",   worst_state,                 "by average DHVS score")

    section("Score Distribution")
    left, right = st.columns([3, 2])

    with left:
        fig = go.Figure()
        for band, color in BAND_COLORS.items():
            sub = scores[scores['vulnerability_band'] == band]
            fig.add_trace(go.Histogram(
                x=sub['dhvs'], name=band, marker_color=color,
                opacity=0.85, nbinsx=18,
                hovertemplate=f'<b>{band}</b><br>Score: %{{x:.0f}}<br>Count: %{{y}}<extra></extra>',
            ))
        fig.update_layout(layout(
            title=dict(text='DHVS Score Distribution across 706 Districts',
                       font=dict(color=TEXT, size=13)),
            barmode='stack', height=360,
            xaxis=dict(title='DHVS Score'),
            yaxis=dict(title='Number of Districts'),
        ))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        state_avg = scores.groupby('state')['dhvs'].mean().sort_values()
        fig2 = go.Figure(go.Bar(
            x=state_avg.values, y=state_avg.index, orientation='h',
            marker=dict(
                color=state_avg.values,
                colorscale=[[0, '#22c55e'], [0.5, '#f59e0b'], [1, '#ef4444']],
                showscale=False,
            ),
            hovertemplate='<b>%{y}</b><br>Avg DHVS: %{x:.1f}<extra></extra>',
        ))
        fig2.update_layout(layout(
            title=dict(text='Average Vulnerability Score by State',
                       font=dict(color=TEXT, size=13)),
            height=360,
            yaxis=dict(tickfont=dict(size=8)),
            xaxis=dict(title='Avg DHVS Score'),
        ))
        st.plotly_chart(fig2, use_container_width=True)

    section("Headline Districts")
    tl, tr = st.columns(2)
    with tl:
        st.markdown("**🔴 10 Most Vulnerable Districts**")
        top10 = scores.head(10)[['dhvs_rank', 'district', 'state', 'dhvs', 'vulnerability_band']].copy()
        top10.columns = ['Rank', 'District', 'State', 'Score', 'Band']
        st.dataframe(top10, hide_index=True, use_container_width=True,
                     column_config={'Score': st.column_config.ProgressColumn(
                         'Score', min_value=0, max_value=100, format='%.1f')})
    with tr:
        st.markdown("**🟢 10 Least Vulnerable Districts**")
        bot10 = scores.tail(10)[['dhvs_rank', 'district', 'state', 'dhvs', 'vulnerability_band']].copy()
        bot10.columns = ['Rank', 'District', 'State', 'Score', 'Band']
        st.dataframe(bot10, hide_index=True, use_container_width=True,
                     column_config={'Score': st.column_config.ProgressColumn(
                         'Score', min_value=0, max_value=100, format='%.1f')})

    section("State Spotlight")
    state_stats = (
        scores.groupby('state')
        .agg(
            Districts=('district', 'count'),
            Avg_Score=('dhvs', 'mean'),
        )
        .sort_values('Avg_Score', ascending=False)
        .reset_index()
    )
    state_stats['Avg_Score'] = state_stats['Avg_Score'].round(1)
    state_stats.columns = ['State', 'Districts', 'Avg DHVS']
    st.dataframe(state_stats, hide_index=True, use_container_width=True, height=380,
                 column_config={'Avg DHVS': st.column_config.ProgressColumn(
                     'Avg DHVS', min_value=0, max_value=100, format='%.1f')})


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 2 — ATLAS RANKINGS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Atlas Rankings":
    st.markdown('<div class="page-title">Atlas Rankings</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Full district leaderboard — filter, search and explore</div>',
        unsafe_allow_html=True,
    )

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        sel_state = st.selectbox("Filter by State",
                                  ['All States'] + sorted(scores['state'].unique().tolist()))
    with fc2:
        sel_band = st.selectbox("Filter by Vulnerability Band",
                                 ['All Bands', 'Low', 'Moderate', 'High', 'Critical'])
    with fc3:
        search = st.text_input("Search District", placeholder="e.g. Araria")

    filtered = scores.copy()
    if sel_state != 'All States':
        filtered = filtered[filtered['state'] == sel_state]
    if sel_band != 'All Bands':
        filtered = filtered[filtered['vulnerability_band'] == sel_band]
    if search.strip():
        filtered = filtered[filtered['district'].str.lower().str.contains(search.lower())]

    st.markdown(
        f"<p style='color:{MUTED};font-size:13px;'>Showing {len(filtered)} of 706 districts</p>",
        unsafe_allow_html=True,
    )

    show_n = min(30, len(filtered))
    if show_n > 0:
        top_n = filtered.head(show_n)
        fig = go.Figure()
        for band, color in BAND_COLORS.items():
            sub = top_n[top_n['vulnerability_band'] == band]
            if sub.empty:
                continue
            fig.add_trace(go.Bar(
                y=sub['district'] + '  ·  ' + sub['state'].str[:10],
                x=sub['dhvs'], name=band, orientation='h',
                marker_color=color, opacity=0.85,
                hovertemplate='<b>%{y}</b><br>DHVS: %{x:.1f}<extra></extra>',
            ))
        fig.update_layout(layout(
            title=dict(text=f'Top {show_n} Districts (filtered)',
                       font=dict(color=TEXT, size=13)),
            barmode='stack',
            height=max(380, show_n * 22),
            yaxis=dict(tickfont=dict(size=8.5)),
            xaxis=dict(title='DHVS Score'),
        ))
        st.plotly_chart(fig, use_container_width=True)

    section("Full Table")
    disp_cols = ['dhvs_rank', 'district', 'state', 'dhvs', 'vulnerability_band',
                 'stunting', 'anaemia_women', 'child_underweight',
                 'institutional_births', 'immunisation']
    disp_cols = [c for c in disp_cols if c in filtered.columns]
    disp = filtered[disp_cols].copy()
    col_names = ['Rank', 'District', 'State', 'DHVS', 'Band',
                 'Stunting%', 'Anaemia%', 'Underweight%', 'Inst.Birth%', 'Immunis.%']
    disp.columns = col_names[:len(disp_cols)]
    st.dataframe(disp, hide_index=True, use_container_width=True, height=440,
                 column_config={'DHVS': st.column_config.ProgressColumn(
                     'DHVS', min_value=0, max_value=100, format='%.1f')})

    if os.path.exists(SURPRISE_CSV):
        section("Surprise Districts")
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("**🔴 Richer but Sicker** — development doesn't match health")
            rs = surprise[surprise['surprise_type'] == 'Richer but Sicker'].nlargest(12, 'dhvs')
            st.dataframe(
                rs[['district', 'state', 'dhvs', 'development_proxy']].rename(
                    columns={'dhvs': 'DHVS', 'development_proxy': 'Dev. Proxy'}),
                hide_index=True, use_container_width=True,
            )
        with sc2:
            st.markdown("**🟢 Poorer but Healthier** — outperforming their constraints")
            ph = surprise[surprise['surprise_type'] == 'Poorer but Healthier'].nsmallest(12, 'dhvs')
            st.dataframe(
                ph[['district', 'state', 'dhvs', 'development_proxy']].rename(
                    columns={'dhvs': 'DHVS', 'development_proxy': 'Dev. Proxy'}),
                hide_index=True, use_container_width=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 3 — GENDER GAP
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Gender Gap":
    st.markdown('<div class="page-title">Gender Health Gap</div>', unsafe_allow_html=True)
    st.markdown(
        "<div class='page-subtitle'>"
        "Districts where women's health is significantly worse than the overall picture suggests"
        "</div>",
        unsafe_allow_html=True,
    )

    hidden_thresh = gender['hidden_gap'].quantile(0.85)
    worst_g   = gender.nlargest(1, 'gender_gap_score').iloc[0]
    worst_hg  = gender.nlargest(1, 'hidden_gap').iloc[0]
    n_hidden  = int((gender['hidden_gap'] > hidden_thresh).sum())
    avg_g     = gender['gender_gap_score'].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Worst Women's Health", worst_g['district'],  worst_g['state'])
    with c2: metric_card("Largest Hidden Gap",   worst_hg['district'], worst_hg['state'])
    with c3: metric_card("Avg Gender Gap Score", f"{avg_g:.1f}",       "out of 100")
    with c4: metric_card("High Hidden Gap",      str(n_hidden),        "districts (top 15%)")

    section("Gender Score vs Overall Vulnerability")
    left, right = st.columns([3, 2])

    with left:
        normal = gender[gender['hidden_gap'] <= hidden_thresh]
        hidden = gender[gender['hidden_gap'] >  hidden_thresh]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=normal['dhvs'], y=normal['gender_gap_score'],
            mode='markers', name='Within normal range',
            marker=dict(color='#374151', size=5, opacity=0.5),
            hovertemplate='<b>%{customdata[0]}</b> · %{customdata[1]}'
                          '<br>DHVS: %{x:.1f}  Gender: %{y:.1f}<extra></extra>',
            customdata=normal[['district', 'state']].values,
        ))
        fig.add_trace(go.Scatter(
            x=hidden['dhvs'], y=hidden['gender_gap_score'],
            mode='markers', name='Hidden gender gap (top 15%)',
            marker=dict(color='#f472b6', size=9, opacity=0.85,
                        line=dict(color='#fff', width=0.4)),
            hovertemplate='<b>%{customdata[0]}</b> · %{customdata[1]}'
                          '<br>DHVS: %{x:.1f}  Gender: %{y:.1f}<extra></extra>',
            customdata=hidden[['district', 'state']].values,
        ))
        fig.add_hline(y=gender['gender_gap_score'].median(),
                      line=dict(color='#374151', dash='dash', width=1))
        fig.add_vline(x=gender['dhvs'].median(),
                      line=dict(color='#374151', dash='dash', width=1))
        fig.update_layout(layout(
            title=dict(text='Each dot = one district · Pink = hidden gender crisis',
                       font=dict(color=TEXT, size=13)),
            height=420,
            xaxis=dict(title='Overall DHVS Score'),
            yaxis=dict(title='Gender Gap Score'),
        ))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("**Top 15 — Largest Hidden Gender Gap**")
        top_h = gender.nlargest(15, 'hidden_gap')[
            ['district', 'state', 'hidden_gap', 'gender_gap_score', 'dhvs']
        ].copy()
        top_h.columns = ['District', 'State', 'Gap', 'Gender Score', 'DHVS']
        st.dataframe(top_h, hide_index=True, use_container_width=True, height=420)

    section("District Deep-Dive — Women's Indicators")
    all_d     = sorted(gender['district'].unique().tolist())
    default_d = gender.nlargest(1, 'gender_gap_score').iloc[0]['district']
    sel_dist  = st.selectbox("Select a district to inspect",
                              all_d, index=all_d.index(default_d))

    row       = gender[gender['district'] == sel_dist].iloc[0]
    ind_keys  = ['anaemia_women', 'child_marriage', 'teen_pregnancy', 'unmet_fp', 'low_bmi']
    ind_labs  = ['Anaemia in Women', 'Child Marriage', 'Teen Pregnancy',
                 'Unmet FP Need', 'Low BMI']
    values    = [float(row[k]) if k in row.index else 0 for k in ind_keys]
    nat_avg   = [float(gender[k].mean()) if k in gender.columns else 0 for k in ind_keys]
    theta     = ind_labs + [ind_labs[0]]

    fig_r = go.Figure()
    fig_r.add_trace(go.Scatterpolar(
        r=nat_avg + [nat_avg[0]], theta=theta, fill='toself',
        name='National Average',
        line=dict(color='#6b7280', width=1.5),
        fillcolor='rgba(107,114,128,0.08)',
    ))
    fig_r.add_trace(go.Scatterpolar(
        r=values + [values[0]], theta=theta, fill='toself',
        name=sel_dist,
        line=dict(color='#f472b6', width=2),
        fillcolor='rgba(244,114,182,0.12)',
    ))
    fig_r.update_layout(
        paper_bgcolor=CARD,
        font=dict(family='DM Sans', color=MUTED),
        polar=dict(
            bgcolor=CARD,
            radialaxis=dict(visible=True, range=[0, 100], gridcolor=BORDER,
                            tickfont=dict(color=MUTED, size=8)),
            angularaxis=dict(gridcolor=BORDER, tickfont=dict(color=TEXT, size=10)),
        ),
        height=400,
        legend=dict(bgcolor=CARD, bordercolor=BORDER, font=dict(color=TEXT)),
        title=dict(text=f"Women's Health Profile — {sel_dist} vs National Average",
                   font=dict(color=TEXT, size=13)),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    st.plotly_chart(fig_r, use_container_width=True)

    section("Full Gender Gap Table")
    disp_g = gender[['gender_rank', 'district', 'state', 'gender_gap_score',
                      'hidden_gap', 'dhvs', 'gender_band']].copy()
    disp_g.columns = ['G.Rank', 'District', 'State', 'Gender Score', 'Hidden Gap', 'DHVS', 'Band']
    st.dataframe(disp_g.sort_values('G.Rank'), hide_index=True,
                 use_container_width=True, height=400,
                 column_config={'Gender Score': st.column_config.ProgressColumn(
                     'Gender Score', min_value=0, max_value=100, format='%.1f')})


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 4 — CHANGE TRACKER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Change Tracker":
    st.markdown('<div class="page-title">Change Tracker</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Every district classified into one of 4 quadrants '
        'based on vulnerability and peer comparison</div>',
        unsafe_allow_html=True,
    )

    QUAD_DESC = {
        'Bad & Declining':    'Emergency — high vulnerability, falling behind state peers',
        'Bad but Improving':  'Needs support — still vulnerable but moving right direction',
        'Good but Declining': 'Hidden risk — looks healthy now but early warning signs',
        'Good & Improving':   'Model districts — low vulnerability, outperforming peers',
    }
    quad_counts = change['quadrant'].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    for col, quad in zip([c1, c2, c3, c4], QUAD_COLORS):
        count = int(quad_counts.get(quad, 0))
        with col:
            metric_card(quad, str(count),
                        f"{count/len(change)*100:.0f}% · {QUAD_DESC[quad][:35]}…",
                        border_color=QUAD_COLORS[quad] + '55')

    section("District Quadrant Scatter")
    sel_quads = st.multiselect("Show quadrants", list(QUAD_COLORS.keys()),
                                default=list(QUAD_COLORS.keys()))
    fig = go.Figure()
    for quad, color in QUAD_COLORS.items():
        if quad not in sel_quads:
            continue
        sub = change[change['quadrant'] == quad]
        fig.add_trace(go.Scatter(
            x=sub['score_delta'], y=sub['dhvs'],
            mode='markers', name=quad,
            marker=dict(color=color, size=7, opacity=0.72, line=dict(width=0)),
            hovertemplate='<b>%{customdata[0]}</b> · %{customdata[1]}'
                          '<br>DHVS: %{y:.1f}  Delta: %{x:.1f}<extra></extra>',
            customdata=sub[['district', 'state']].values,
        ))
    fig.add_hline(y=change['dhvs'].median(),
                  line=dict(color='#374151', dash='dash', width=1))
    fig.add_vline(x=change['score_delta'].median(),
                  line=dict(color='#374151', dash='dash', width=1))
    fig.update_layout(layout(
        title=dict(text='Each dot = one district · Click quadrant in legend to toggle',
                   font=dict(color=TEXT, size=13)),
        height=480,
        xaxis=dict(title='Score vs State Median  →  positive = worse than state peers'),
        yaxis=dict(title='DHVS Vulnerability Score'),
    ))
    st.plotly_chart(fig, use_container_width=True)

    section("Drill Down by Quadrant")
    sel_q = st.selectbox("Select Quadrant", list(QUAD_COLORS.keys()))
    st.markdown(f"<div class='note-box'>{QUAD_DESC[sel_q]}</div>",
                unsafe_allow_html=True)
    st.markdown("")
    q_df   = change[change['quadrant'] == sel_q].sort_values('dhvs', ascending=False)
    q_cols = [c for c in ['dhvs_rank', 'district', 'state', 'dhvs',
                           'vulnerability_band', 'score_delta'] if c in q_df.columns]
    q_disp = q_df[q_cols].copy()
    q_disp.columns = ['Rank', 'District', 'State', 'DHVS', 'Band', 'Delta'][:len(q_cols)]
    st.dataframe(q_disp, hide_index=True, use_container_width=True, height=420,
                 column_config={'DHVS': st.column_config.ProgressColumn(
                     'DHVS', min_value=0, max_value=100, format='%.1f')})


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 5 — WHAT-IF SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────
elif page == "What-If Simulator":
    st.markdown('<div class="page-title">What-If Simulator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Pick any district · Adjust health indicators · '
        'Watch the vulnerability score change instantly</div>',
        unsafe_allow_html=True,
    )

    config     = load_config(CONFIG)
    ind_config = config['indicators']

    sc1, sc2 = st.columns([3, 2])
    with sc1:
        all_d    = sorted(scores['district'].unique().tolist())
        sel_dist = st.selectbox("Select District", all_d)
    with sc2:
        d_row = scores[scores['district'] == sel_dist].iloc[0]
        metric_card(
            "Current Score",
            f"{d_row['dhvs']:.1f}",
            f"Rank #{d_row['dhvs_rank']} of 706 · {d_row['vulnerability_band']}",
        )

    section("Adjust Indicators")
    st.markdown(
        "<p style='color:#6b7280;font-size:13px;margin-top:-.4rem;'>"
        "Move any slider to simulate a policy change. The score recalculates instantly.</p>",
        unsafe_allow_html=True,
    )

    current_vals = {k: float(d_row[k]) if k in d_row.index else 50.0
                    for k in ind_config}

    sim_vals = {}
    col1, col2 = st.columns(2)
    for i, (key, meta) in enumerate(ind_config.items()):
        col  = col1 if i % 2 == 0 else col2
        curr = current_vals[key]
        dirn = "⬇ lower = more vulnerable" if not meta.get('invert') \
               else "⬆ higher = less vulnerable"
        with col:
            sim_vals[key] = st.slider(
                f"{meta['label']}  (current: {curr:.1f}%)",
                min_value=0.0, max_value=100.0,
                value=curr, step=0.5, help=dirn,
            )

    # Recalculate
    def recalc(base_df, idx, sim, ind_cfg):
        df = base_df.copy()
        for k, v in sim.items():
            if k in df.columns:
                df.at[idx, k] = v
        total_w, score = 0.0, 0.0
        for k, meta in ind_cfg.items():
            if k not in df.columns:
                continue
            vals = df[k].copy()
            if meta.get('invert', False):
                vals = vals.max() - vals + vals.min()
            mn, mx = vals.min(), vals.max()
            nv      = 50.0 if mn == mx else (vals[idx] - mn) / (mx - mn) * 100
            score  += nv * meta['weight']
            total_w += meta['weight']
        return round(score, 2)

    d_idx     = scores[scores['district'] == sel_dist].index[0]
    new_score = recalc(scores, d_idx, sim_vals, ind_config)
    old_score = float(d_row['dhvs'])
    delta     = new_score - old_score
    d_color   = '#22c55e' if delta < 0 else '#ef4444' if delta > 0 else '#6b7280'
    d_arrow   = '▼' if delta < 0 else '▲' if delta > 0 else '●'

    section("Simulated Result")
    r1, r2, r3, r4 = st.columns(4)
    with r1: metric_card("Original Score",  f"{old_score:.1f}", f"Rank #{d_row['dhvs_rank']}")
    with r2: metric_card("Simulated Score", f"{new_score:.1f}", "after your changes")
    with r3:
        st.markdown(
            f"""<div class="metric-card" style="border-color:{d_color}55">
              <div class="metric-label">Score Change</div>
              <div class="metric-value" style="color:{d_color};font-size:28px">
                {d_arrow} {abs(delta):.1f}
              </div>
              <div class="metric-sub">points
                {"improvement" if delta < 0 else "worsened" if delta > 0 else "— no change"}
              </div>
            </div>""",
            unsafe_allow_html=True,
        )
    with r4: metric_card("Est. Rank Shift", f"≈ {int(abs(delta)*2.5)} places",
                          "up" if delta < 0 else "down")

    st.markdown("")
    g_col, b_col = st.columns([2, 3])

    with g_col:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=new_score,
            delta={'reference': old_score,
                   'increasing': {'color': '#ef4444'},
                   'decreasing': {'color': '#22c55e'}},
            number={'font': {'color': TEXT, 'size': 44, 'family': 'DM Sans'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': MUTED,
                         'tickfont': {'color': MUTED, 'size': 9}},
                'bar':  {'color': '#60a5fa', 'thickness': 0.25},
                'bgcolor': BORDER, 'bordercolor': BORDER,
                'steps': [
                    {'range': [0,  25], 'color': '#052e16'},
                    {'range': [25, 50], 'color': '#431407'},
                    {'range': [50, 75], 'color': '#450a0a'},
                    {'range': [75, 100], 'color': '#2d0000'},
                ],
                'threshold': {'line': {'color': '#f9fafb', 'width': 2},
                              'value': old_score},
            },
            title={'text': f"DHVS — {sel_dist}",
                   'font': {'color': MUTED, 'size': 12}},
        ))
        fig_g.update_layout(
            paper_bgcolor=CARD, height=300,
            font=dict(family='DM Sans'),
            margin=dict(l=30, r=30, t=60, b=20),
        )
        st.plotly_chart(fig_g, use_container_width=True)

    with b_col:
        labels = [meta['label'] for meta in ind_config.values()]
        before = [current_vals[k] for k in ind_config]
        after  = [sim_vals[k]    for k in ind_config]
        fig_b  = go.Figure()
        fig_b.add_trace(go.Bar(name='Current',   x=labels, y=before,
                               marker_color='#374151', opacity=0.8))
        fig_b.add_trace(go.Bar(name='Simulated', x=labels, y=after,
                               marker_color='#60a5fa', opacity=0.88))
        fig_b.update_layout(layout(
            title=dict(text='Indicator Comparison — Current vs Simulated',
                       font=dict(color=TEXT, size=13)),
            barmode='group', height=300,
            xaxis=dict(tickangle=-20, tickfont=dict(size=9)),
            yaxis=dict(title='Value (%)'),
        ))
        st.plotly_chart(fig_b, use_container_width=True)

    section("Interpretation")
    if abs(delta) < 0.5:
        msg   = (f"The changes you made have minimal impact on {sel_dist}'s score. "
                 f"Try larger adjustments to stunting or anaemia — they carry the highest weights (25% and 20%).")
        color = '#6b7280'
    elif delta < 0:
        msg   = (f"Your interventions would reduce {sel_dist}'s vulnerability by {abs(delta):.1f} points — "
                 f"a meaningful improvement. The indicator you changed most is your highest-leverage policy action.")
        color = '#22c55e'
    else:
        msg   = (f"These changes would increase {sel_dist}'s vulnerability by {delta:.1f} points. "
                 f"Useful for modelling what happens if conditions worsen — e.g. during a funding cut or drought.")
        color = '#ef4444'

    st.markdown(
        f"""<div style="background:{CARD};border:1px solid {color}33;
          border-left:4px solid {color};border-radius:8px;
          padding:16px 20px;color:#e5e7eb;font-size:14px;line-height:1.65;">
          {msg}
        </div>""",
        unsafe_allow_html=True,
    )