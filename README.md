# India Health Atlas

> A district-level public health vulnerability study of India using NFHS-5 government survey data.
> Combines six health indicators into a composite scoring system and surfaces findings invisible in standard reports.

---

## Project Structure

```
india_health_atlas/
│
├── data/
│   ├── raw/
│   │   └── nfhs5_districts.csv        ← Your downloaded NFHS-5 data (put it here)
│   └── processed/
│       ├── vulnerability_scores.csv   ← Generated: DHVS scores for 706 districts
│       └── surprise_districts.csv     ← Generated: quadrant classification
│
├── src/
│   ├── scoring.py                     ← Core scoring logic (imported by other scripts)
│   ├── vulnerability_score.py         ← Script 1: compute and rank district scores
│   ├── surprise_states.py             ← Script 2: find districts that defy expectations
│   ├── gender_gap.py                  ← Script 3: where women's health diverges
│   ├── change_tracker.py              ← Script 4: NFHS-4 vs NFHS-5 comparison
│   ├── infrastructure_gap.py          ← Script 5: resources vs outcomes
│   └── atlas_map.py                   ← Script 6: interactive India choropleth map
│
├── config/
│   └── indicators.yaml                ← Indicator weights — edit here, not in code
│
├── outputs/
│   ├── charts/                        ← All generated chart images
│   └── maps/                          ← All generated HTML maps
│
├── notebooks/
│   └── exploration.ipynb              ← Jupyter notebook for EDA
│
├── requirements.txt
└── README.md
```

---

## Quick Start

**Step 1 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 2 — Place your data file**

Put your downloaded NFHS-5 CSV inside `data/raw/` and name it `nfhs5_districts.csv`.

**Step 3 — Run scripts in order**
```bash
python src/vulnerability_score.py     # Script 1: generates DHVS scores
python src/surprise_states.py         # Script 2: surprise analysis
python src/gender_gap.py              # Script 3: gender gap layer
python src/change_tracker.py          # Script 4: change over time
```

---

## The DHVS Formula

```
DHVS = 0.25 × stunting
     + 0.20 × anaemia_in_women
     + 0.20 × child_underweight
     + 0.15 × (100 − institutional_births)
     + 0.10 × (100 − immunisation)
     + 0.10 × (100 − antenatal_visits)
```

All indicators normalized to 0–100 via min-max scaling across all districts.
Higher DHVS = more vulnerable. Weights are configurable in `config/indicators.yaml`.

---

## Data Sources

| Source | What it provides |
|---|---|
| NFHS-5 district factsheets (IIPS) | 706 districts × 109 health indicators |
| NFHS-4 district factsheets | Same indicators, 2015–16 round (for change tracking) |
| data.gov.in | PHC count, doctor availability per 1000 population |

---
