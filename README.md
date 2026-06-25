# India Health Atlas

**Surfaces health vulnerability patterns invisible in standard state-level reports.**

District-level public health analysis across 706 Indian districts using NFHS-5 data. Built a composite District Health Vulnerability Score (DHVS), a gender health gap index, and a 4-quadrant change classifier to identify which districts need urgent policy attention — and which ones are outperforming their constraints.

🔗 **[Live Dashboard →](https://india-health-atlas.streamlit.app)**

---

## What this project does

Standard health reporting in India aggregates to the state level, masking enormous within-state variation. Jharkhand's average looks bad — but which of its 24 districts is worst, and by how much? Kerala looks good — but are there districts silently declining?

This atlas answers those questions across four analytical lenses:

**Vulnerability Ranking** — Every district scored and ranked on a 0–100 composite index. Filter by state, band, or search by name.

**Gender Health Gap** — Districts where women's health outcomes are *disproportionately* worse than the overall DHVS suggests — a hidden crisis not visible in aggregate numbers.

**Change Tracker** — 4-quadrant classification: Bad & Declining, Bad but Improving, Good but Declining, Good & Improving. Shows which districts are moving in the wrong direction even from a healthy baseline.

**What-If Simulator** — Adjust any indicator for any district and watch the vulnerability score update instantly. A policy tool: which intervention gives the most score improvement?

---

## The DHVS Formula

The District Health Vulnerability Score is a weighted composite of 6 NFHS-5 indicators:

| Indicator | Weight | Direction |
|---|---|---|
| Stunting in children under 5 | 25% | Higher = more vulnerable |
| Anaemia in women 15–49 yrs | 20% | Higher = more vulnerable |
| Underweight children under 5 | 20% | Higher = more vulnerable |
| Institutional birth rate | 15% | Lower = more vulnerable |
| Full immunisation coverage | 10% | Lower = more vulnerable |
| 4+ antenatal care visits | 10% | Lower = more vulnerable |

Each indicator is min-max normalised to 0–100 (higher = more vulnerable). Indicators where a high raw value means a *good* outcome (births, immunisation, ANC) are inverted before normalisation using `max(vals) − vals`, so the normalization direction is consistent across all indicators. Missing NFHS values (`*`) are imputed using state medians, with national median as fallback.

Weights reflect burden-of-disease evidence: child stunting and maternal anaemia together account for 45% of the score because they signal long-term developmental consequences, not just acute deprivation.

---

## Key Findings

- **Pashchimi Singhbhum (Jharkhand)** is the most vulnerable district nationally — DHVS 74.4, driven by stunting (54.2%) and anaemia (72.1%)
- **Imphal West (Manipur)** is the least vulnerable — DHVS 10.7
- **Bihar** has the highest average state-level vulnerability
- **126 districts** (18%) score above 50 — classified as High or Critical risk
- **West Bengal** dominates the hidden gender gap list: Koch Bihar and Purba Medinipur show the largest divergence between overall DHVS and women's health outcomes
- **254 districts** are Bad & Declining — high vulnerability *and* falling behind state peers

---

## Data Sources

| Dataset | Source | Coverage |
|---|---|---|
| NFHS-5 District Factsheets | IIPS, Government of India | 706 districts, 2019–21 |
| NFHS-4 District Data | IIPS, Government of India | 640 districts, 2015–16 |
| Administrative boundaries | data.gov.in | District-level GeoJSON |

---

## Tech Stack

Python · Pandas · NumPy · Streamlit · Plotly · scikit-learn · PyYAML · GeoPandas

---

## Quick Start

```bash
git clone https://github.com/KeertiSharma1/india-health-atlas.git
cd india-health-atlas/india_health_atlas

pip install -r requirements.txt

# Place NFHS-5 CSV at data/raw/nfhs5_districts.csv, then:
python src/vulnerability_score.py
python src/gender_gap.py
python src/surprise_states.py
python src/change_tracker.py

streamlit run app/dashboard.py
```

---

*Data: NFHS-5 (2019–21) · Source: International Institute for Population Sciences (IIPS), Government of India*
