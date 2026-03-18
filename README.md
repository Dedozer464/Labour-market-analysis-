# South African Labour Market Analysis

A statistical analysis of unemployment dynamics in South Africa using a
**Quarterly Labour Force Survey (QLFS)-style dataset** — modelled on
StatsSA's official microdata structure.

South Africa has one of the highest unemployment rates in the world. This project
goes beyond the headline number to examine *who* is unemployed, *where*, and
*why* — and models the probability of unemployment using logistic regression.

---

## What This Project Does

1. Generates a realistic 8,000-person QLFS-style survey dataset
2. Produces descriptive breakdowns by province, education, age, gender, and sector
3. Analyses youth unemployment, NEET rates, and discouraged workers
4. Models income distribution across formal and informal employment sectors
5. Fits a **logistic regression** to predict unemployment probability from
   demographic and educational characteristics

---

## Key Metrics Covered

| Metric | Definition |
|---|---|
| Official unemployment rate | % of labour force actively searching for work |
| Expanded unemployment rate | Includes discouraged workers (stopped searching) |
| Absorption rate | Employed as % of working-age population |
| NEET rate | % of 15–34 year-olds Not in Education, Employment, or Training |
| Discouraged workers | Unemployed who have given up searching |

---

## Analysis Modules

| Module | Description |
|---|---|
| `descriptive_overview` | Headline stats: unemployment, absorption, NEET, discouraged |
| `breakdown_plots` | Unemployment by province, education, age/gender, sector |
| `youth_neet_analysis` | Comparative NEET and discouraged worker rates: 15–24 vs 25–34 |
| `income_distribution` | Violin plots (sector) and bar chart (education) for income |
| `logistic_regression_model` | Predict P(unemployed) — includes AUC-ROC and confusion matrix |
| `gender_education_heatmap` | Intersectional heatmap: unemployment rate by education × gender |

---

## Key Findings

- **Youth unemployment is the most severe fault line.** The 15–24 age cohort
  faces unemployment rates above 55% in this dataset — consistent with StatsSA's
  published figures — with a large share classified as NEET.

- **Education is the single strongest protective factor.** Moving from no
  schooling to a certificate or diploma is associated with a dramatic reduction
  in unemployment probability. Matric alone reduces risk, but is insufficient
  without further qualifications in the current market.

- **Provincial inequality is extreme.** Western Cape and Gauteng show
  significantly lower unemployment than Eastern Cape, Limpopo, and North West —
  reflecting structural differences in economic activity, not just skills supply.

- **Women with low education face compounded disadvantage.** The education ×
  gender heatmap reveals that women without matric face substantially higher
  unemployment than men at the same education level. This gap narrows sharply at
  certificate and degree level.

- **Informal employment is the buffer, not a solution.** A large share of
  "employed" workers in the informal sector earn below R5,000/month — close to
  or below subsistence. The formal/informal income gap is stark.

- **The logistic model achieves AUC ≈ 0.75**, confirming that demographic and
  educational variables alone predict unemployment meaningfully — but roughly
  25% of variance is unexplained, pointing to unobserved factors (networks,
  geography, job availability, discrimination).

---

## Project Structure

```
sa_labour_market/
├── labour_market_analysis.py   ← main script
├── requirements.txt
├── data/
│   └── labour_market_data.csv  ← generated on first run
└── outputs/
    ├── 01_unemployment_breakdown.png
    ├── 02_youth_neet_analysis.png
    ├── 03_income_distribution.png
    ├── 04_logistic_regression.png
    └── 05_gender_education_heatmap.png
```

---

## How to Run

```bash
# Navigate to the project folder
cd sa_labour_market

# Install dependencies
pip install -r requirements.txt

# Run the full analysis
python labour_market_analysis.py
```

All charts are saved to `outputs/`. The dataset CSV is saved to `data/` on first run.

---

## Data Sources & Methodology

| Variable | Source | Notes |
|---|---|---|
| Survey structure | StatsSA QLFS | Quarterly Labour Force Survey microdata |
| Unemployment definitions | StatsSA / ILO | Official and expanded definitions |
| Provincial unemployment rates | StatsSA QLFS Q3 2023 | Used for simulation calibration |
| Income ranges by sector | StatsSA QLFS / SARS data | Median monthly earnings |
| NEET definition | StatsSA Youth report | 15–34 cohort |

> **Note:** This project uses **simulated microdata calibrated to match
> published StatsSA QLFS Q3 2023 patterns**. Unemployment rates, income
> distributions, and demographic splits are designed to reflect real SA
> conditions. To use actual QLFS microdata (available from StatsSA DataFirst),
> replace `generate_labour_dataset()` with your own data loader.

---

## Model: Logistic Regression

**Target variable:** `unemployed` (1 = unemployed, 0 = employed)

**Features:**
- Education level (encoded ordinal)
- Age group
- Gender
- Province
- Years of experience (proxy)

**Evaluation:**
- Train/test split: 75% / 25%, stratified
- Metric: AUC-ROC, classification report, confusion matrix
- Standardisation: StandardScaler applied before fitting

---

## Skills Demonstrated

- Survey microdata simulation and analysis
- Descriptive statistics across multiple demographic dimensions
- Binary logistic regression with sklearn
- AUC-ROC curve and confusion matrix interpretation
- Intersectional analysis (education × gender heatmap)
- Violin plots, grouped bar charts, pie charts, heatmaps
- End-to-end Python pipeline with modular functions

---

## Dependencies

```
pandas >= 2.0
numpy >= 1.24
matplotlib >= 3.7
seaborn >= 0.12
scikit-learn >= 1.3
scipy >= 1.11
```

---

## Author

**Retshidisitswe Sebekedi** 
[https://github.com/Dedozer464] | [https://www.linkedin.com/in/retshidisitswe-sebekedi-151212398?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app]
