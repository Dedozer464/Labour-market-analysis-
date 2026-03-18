"""
South African Labour Market Analysis
=====================================
Analyses unemployment dynamics, labour force participation,
and employment probability using StatsSA QLFS-style data.

Includes:
  - Descriptive breakdowns by province, sector, education, gender, age
  - Logistic regression: probability of unemployment
  - Discouraged worker trends
  - Absorption rate & NEETs analysis

Author: Retshidisitswe
Dataset: Simulated from StatsSA QLFS Q3 2023 structure
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_auc_score,
                             roc_curve, confusion_matrix)
import warnings
import os

warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",
    "grid.color":       "#21262d",
    "grid.linestyle":   "--",
    "grid.linewidth":   0.5,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "font.family":      "DejaVu Sans",
    "font.size":        10,
})

GREEN   = "#3fb950"
ORANGE  = "#f0883e"
BLUE    = "#58a6ff"
RED     = "#f85149"
PURPLE  = "#bc8cff"
YELLOW  = "#e3b341"


# ══════════════════════════════════════════════════════════════════════════════
# 1. SYNTHETIC DATASET — mirrors StatsSA QLFS structure
# ══════════════════════════════════════════════════════════════════════════════

def generate_labour_dataset(n: int = 8_000) -> pd.DataFrame:
    np.random.seed(99)

    provinces = ["Gauteng", "Western Cape", "KwaZulu-Natal", "Eastern Cape",
                 "Limpopo", "Mpumalanga", "North West", "Free State", "Northern Cape"]
    prov_weights = [0.22, 0.12, 0.17, 0.12, 0.10, 0.08, 0.07, 0.07, 0.05]

    sectors = ["Formal private", "Formal public", "Informal", "Agriculture", "Domestic work"]
    education_levels = ["No schooling", "Primary", "Secondary incomplete",
                        "Matric", "Certificate/Diploma", "Degree+"]
    age_groups = ["15–24", "25–34", "35–44", "45–54", "55–64"]
    genders = ["Male", "Female"]

    province   = np.random.choice(provinces, n, p=prov_weights)
    gender     = np.random.choice(genders, n, p=[0.49, 0.51])
    age_group  = np.random.choice(age_groups, n, p=[0.25, 0.28, 0.22, 0.16, 0.09])

    # Education distribution — SA reality
    edu_p = [0.04, 0.09, 0.22, 0.35, 0.18, 0.12]
    education = np.random.choice(education_levels, n, p=edu_p)

    # Base unemployment probability
    unemp_prob = np.full(n, 0.32)  # national ~32%

    # Adjustments by factor
    edu_map = {"No schooling": 0.25, "Primary": 0.18, "Secondary incomplete": 0.15,
                "Matric": 0.02, "Certificate/Diploma": -0.12, "Degree+": -0.20}
    age_map = {"15–24": 0.28, "25–34": 0.05, "35–44": -0.05,
               "45–54": -0.08, "55–64": -0.10}
    gender_map = {"Male": -0.03, "Female": 0.03}
    prov_map = {"Gauteng": -0.06, "Western Cape": -0.10, "KwaZulu-Natal": 0.04,
                "Eastern Cape": 0.12, "Limpopo": 0.08, "Mpumalanga": 0.06,
                "North West": 0.08, "Free State": 0.07, "Northern Cape": 0.05}

    for i in range(n):
        unemp_prob[i] += edu_map[education[i]]
        unemp_prob[i] += age_map[age_group[i]]
        unemp_prob[i] += gender_map[gender[i]]
        unemp_prob[i] += prov_map[province[i]]

    unemp_prob = np.clip(unemp_prob, 0.05, 0.90)
    unemployed = (np.random.rand(n) < unemp_prob).astype(int)

    # Employment sector (only for employed)
    sector = np.where(
        unemployed == 0,
        np.random.choice(sectors, n, p=[0.45, 0.22, 0.18, 0.08, 0.07]),
        "Unemployed"
    )

    # Monthly income (ZAR) — 0 if unemployed
    income_map = {
        "Formal private": (18000, 8000),
        "Formal public":  (22000, 6000),
        "Informal":        (5000, 2000),
        "Agriculture":     (4500, 1500),
        "Domestic work":   (3800, 1200),
        "Unemployed":      (0, 0),
    }
    income = np.array([
        max(0, np.random.normal(*income_map[s]))
        for s in sector
    ])

    # Discouraged workers (subset of unemployed who stopped searching)
    discouraged = np.where(
        (unemployed == 1) & (np.random.rand(n) < 0.30), 1, 0
    )

    # NEET flag (Not in Education, Employment or Training) — 15–34 only
    neet = np.where(
        (unemployed == 1) & (age_group == "15–24") & (np.random.rand(n) < 0.55), 1,
        np.where(
            (unemployed == 1) & (age_group == "25–34") & (np.random.rand(n) < 0.35), 1, 0
        )
    )

    # Years of experience proxy
    exp_map = {"15–24": (1, 3), "25–34": (4, 8), "35–44": (10, 7),
               "45–54": (18, 6), "55–64": (25, 8)}
    experience = np.array([
        max(0, np.random.normal(*exp_map[a]))
        for a in age_group
    ])

    df = pd.DataFrame({
        "province":    province,
        "gender":      gender,
        "age_group":   age_group,
        "education":   education,
        "sector":      sector,
        "unemployed":  unemployed,
        "income":      income.round(0),
        "discouraged": discouraged,
        "neet":        neet,
        "experience":  experience.round(1),
    })

    # Derived
    df["employed"] = 1 - df["unemployed"]
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. DESCRIPTIVE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

def descriptive_overview(df: pd.DataFrame):
    n           = len(df)
    n_unemp     = df["unemployed"].sum()
    n_discouraged = df["discouraged"].sum()
    n_neet      = df["neet"].sum()

    unemp_rate  = n_unemp / n * 100
    exp_unemp   = (n_unemp + n_discouraged) / n * 100  # expanded definition
    absorption  = df["employed"].sum() / n * 100

    print("\n── Labour Market Summary ────────────────────────────────")
    print(f"  Sample size             : {n:,}")
    print(f"  Official unemployment   : {unemp_rate:.1f}%")
    print(f"  Expanded unemployment   : {exp_unemp:.1f}%  (incl. discouraged)")
    print(f"  Absorption rate         : {absorption:.1f}%")
    print(f"  Discouraged workers     : {n_discouraged:,}  ({n_discouraged/n*100:.1f}%)")
    print(f"  NEET (youth)            : {n_neet:,}  ({n_neet/n*100:.1f}%)")

    # By province
    prov_stats = df.groupby("province")["unemployed"].agg(["mean","count"])
    prov_stats.columns = ["unemployment_rate", "sample_n"]
    prov_stats["unemployment_rate"] = (prov_stats["unemployment_rate"] * 100).round(1)
    prov_stats = prov_stats.sort_values("unemployment_rate", ascending=False)
    print("\n── Unemployment Rate by Province ────────────────────────")
    print(prov_stats.to_string())

    return prov_stats


# ══════════════════════════════════════════════════════════════════════════════
# 3. UNEMPLOYMENT BREAKDOWN PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def breakdown_plots(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("South African Labour Market — Unemployment Breakdown",
                 fontsize=14, fontweight="bold", color=ORANGE, y=0.98)

    # 3a. By province
    ax = axes[0, 0]
    prov = (df.groupby("province")["unemployed"].mean() * 100).sort_values(ascending=True)
    colors = [RED if v > 40 else ORANGE if v > 30 else GREEN for v in prov.values]
    bars = ax.barh(prov.index, prov.values, color=colors, edgecolor="#0d1117", linewidth=0.5)
    for bar, val in zip(bars, prov.values):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=8)
    ax.axvline(prov.mean(), color="white", lw=1, linestyle="--", alpha=0.5, label=f"Mean {prov.mean():.1f}%")
    ax.set_xlabel("Unemployment Rate (%)")
    ax.set_title("By Province", color="#c9d1d9")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="x")

    # 3b. By education level
    ax = axes[0, 1]
    edu_order = ["No schooling", "Primary", "Secondary incomplete",
                 "Matric", "Certificate/Diploma", "Degree+"]
    edu = (df.groupby("education")["unemployed"].mean() * 100).reindex(edu_order)
    colors = [RED if v > 45 else ORANGE if v > 30 else GREEN for v in edu.values]
    bars = ax.bar(range(len(edu)), edu.values, color=colors, edgecolor="#0d1117", linewidth=0.5)
    ax.set_xticks(range(len(edu)))
    ax.set_xticklabels([e.replace(" ", "\n") for e in edu_order], fontsize=8)
    for bar, val in zip(bars, edu.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", fontsize=8)
    ax.set_ylabel("Unemployment Rate (%)")
    ax.set_title("By Education Level", color="#c9d1d9")
    ax.grid(True, alpha=0.3, axis="y")

    # 3c. By age group & gender
    ax = axes[1, 0]
    age_gender = df.groupby(["age_group", "gender"])["unemployed"].mean().unstack() * 100
    age_order  = ["15–24", "25–34", "35–44", "45–54", "55–64"]
    age_gender = age_gender.reindex(age_order)
    x = np.arange(len(age_order))
    w = 0.35
    ax.bar(x - w/2, age_gender["Male"],   width=w, label="Male",   color=BLUE,   edgecolor="#0d1117")
    ax.bar(x + w/2, age_gender["Female"], width=w, label="Female", color=PURPLE, edgecolor="#0d1117")
    ax.set_xticks(x)
    ax.set_xticklabels(age_order)
    ax.set_ylabel("Unemployment Rate (%)")
    ax.set_title("By Age Group & Gender", color="#c9d1d9")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 3d. Sector composition (employed only)
    ax = axes[1, 1]
    employed  = df[df["unemployed"] == 0]
    sect_cnt  = employed["sector"].value_counts()
    wedge_colors = [GREEN, BLUE, ORANGE, YELLOW, PURPLE]
    wedges, texts, autotexts = ax.pie(
        sect_cnt.values, labels=sect_cnt.index,
        autopct="%1.1f%%", colors=wedge_colors,
        startangle=140, pctdistance=0.75,
        textprops={"color": "#c9d1d9", "fontsize": 8},
        wedgeprops={"edgecolor": "#0d1117", "linewidth": 1}
    )
    for at in autotexts:
        at.set_color("#0d1117")
        at.set_fontsize(8)
    ax.set_title("Employment by Sector (Employed Only)", color="#c9d1d9")

    plt.tight_layout()
    plt.savefig("outputs/01_unemployment_breakdown.png", dpi=150)
    plt.close()
    print("  [saved] outputs/01_unemployment_breakdown.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4. YOUTH UNEMPLOYMENT — NEET & DISCOURAGED
# ══════════════════════════════════════════════════════════════════════════════

def youth_neet_analysis(df: pd.DataFrame):
    youth = df[df["age_group"].isin(["15–24", "25–34"])].copy()

    # Official vs expanded
    categories   = ["Employed", "Unemployed\n(searching)", "Discouraged\nWorkers", "NEET"]
    vals_15_24   = [
        (youth[youth["age_group"] == "15–24"]["employed"].mean() * 100),
        (youth[(youth["age_group"] == "15–24") & (youth["unemployed"] == 1) & (youth["discouraged"] == 0)]["unemployed"].sum()
         / len(youth[youth["age_group"] == "15–24"]) * 100),
        (youth[youth["age_group"] == "15–24"]["discouraged"].mean() * 100),
        (youth[youth["age_group"] == "15–24"]["neet"].mean() * 100),
    ]
    vals_25_34 = [
        (youth[youth["age_group"] == "25–34"]["employed"].mean() * 100),
        (youth[(youth["age_group"] == "25–34") & (youth["unemployed"] == 1) & (youth["discouraged"] == 0)]["unemployed"].sum()
         / len(youth[youth["age_group"] == "25–34"]) * 100),
        (youth[youth["age_group"] == "25–34"]["discouraged"].mean() * 100),
        (youth[youth["age_group"] == "25–34"]["neet"].mean() * 100),
    ]

    fig, ax = plt.subplots(figsize=(11, 5))
    x   = np.arange(len(categories))
    w   = 0.35
    ax.bar(x - w/2, vals_15_24, width=w, label="15–24", color=RED,   edgecolor="#0d1117")
    ax.bar(x + w/2, vals_25_34, width=w, label="25–34", color=ORANGE, edgecolor="#0d1117")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel("% of Age Group")
    ax.set_title("Youth Labour Market Status — 15–34 Age Cohorts",
                 fontsize=13, fontweight="bold", color=ORANGE)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("outputs/02_youth_neet_analysis.png", dpi=150)
    plt.close()
    print("  [saved] outputs/02_youth_neet_analysis.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5. INCOME DISTRIBUTION BY SECTOR
# ══════════════════════════════════════════════════════════════════════════════

def income_distribution(df: pd.DataFrame):
    employed = df[(df["unemployed"] == 0) & (df["income"] > 0)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Income Distribution by Sector & Education",
                 fontsize=13, fontweight="bold", color=ORANGE)

    # Violin by sector
    ax = axes[0]
    sectors_order = ["Formal public", "Formal private", "Informal", "Agriculture", "Domestic work"]
    data_by_sector = [employed[employed["sector"] == s]["income"].values for s in sectors_order]
    parts = ax.violinplot(data_by_sector, positions=range(len(sectors_order)), showmedians=True)
    colors_v = [GREEN, BLUE, YELLOW, ORANGE, PURPLE]
    for pc, col in zip(parts["bodies"], colors_v):
        pc.set_facecolor(col)
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("white")
    parts["cbars"].set_color("#8b949e")
    parts["cmaxes"].set_color("#8b949e")
    parts["cmins"].set_color("#8b949e")
    ax.set_xticks(range(len(sectors_order)))
    ax.set_xticklabels([s.replace(" ", "\n") for s in sectors_order], fontsize=8)
    ax.set_ylabel("Monthly Income (ZAR)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R{x/1000:.0f}k"))
    ax.set_title("Income Distribution by Sector", color="#c9d1d9")
    ax.grid(True, alpha=0.3, axis="y")

    # Median income by education
    ax = axes[1]
    edu_order = ["No schooling", "Primary", "Secondary incomplete",
                 "Matric", "Certificate/Diploma", "Degree+"]
    median_income = employed.groupby("education")["income"].median().reindex(edu_order)
    colors_edu = [RED, RED, ORANGE, YELLOW, GREEN, GREEN]
    bars = ax.barh(range(len(edu_order)), median_income.values,
                   color=colors_edu, edgecolor="#0d1117", linewidth=0.5)
    ax.set_yticks(range(len(edu_order)))
    ax.set_yticklabels(edu_order, fontsize=9)
    ax.set_xlabel("Median Monthly Income (ZAR)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R{x/1000:.0f}k"))
    for bar, val in zip(bars, median_income.values):
        ax.text(val + 100, bar.get_y() + bar.get_height()/2,
                f"R{val:,.0f}", va="center", fontsize=8)
    ax.set_title("Median Income by Education Level", color="#c9d1d9")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig("outputs/03_income_distribution.png", dpi=150)
    plt.close()
    print("  [saved] outputs/03_income_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# 6. LOGISTIC REGRESSION — PROBABILITY OF UNEMPLOYMENT
# ══════════════════════════════════════════════════════════════════════════════

def logistic_regression_model(df: pd.DataFrame):
    """
    Binary outcome: unemployed (1) vs employed (0)
    Features: education, age_group, gender, province
    """
    model_df = df.copy()

    # Encode categoricals
    le = LabelEncoder()
    for col in ["education", "age_group", "gender", "province"]:
        model_df[col + "_enc"] = le.fit_transform(model_df[col])

    features = ["education_enc", "age_group_enc", "gender_enc", "province_enc", "experience"]
    X = model_df[features]
    y = model_df["unemployed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler  = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X_train_s, y_train)
    y_pred  = clf.predict(X_test_s)
    y_proba = clf.predict_proba(X_test_s)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)

    print("\n── Logistic Regression Results ─────────────────────────")
    print(f"  AUC-ROC : {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Employed", "Unemployed"]))

    # Feature coefficients
    coef_df = pd.DataFrame({
        "Feature":     ["Education", "Age Group", "Gender", "Province", "Experience"],
        "Coefficient": clf.coef_[0]
    }).sort_values("Coefficient")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Logistic Regression — Probability of Unemployment",
                 fontsize=13, fontweight="bold", color=ORANGE)

    # Coefficients
    ax = axes[0]
    colors_c = [RED if v > 0 else GREEN for v in coef_df["Coefficient"]]
    ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors_c, edgecolor="#0d1117")
    ax.axvline(0, color="white", lw=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("Coefficient (standardised)")
    ax.set_title("Feature Coefficients\n(+ve = ↑ unemployment risk)", color="#c9d1d9")
    ax.grid(True, alpha=0.3, axis="x")

    # ROC curve
    ax = axes[1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax.plot(fpr, tpr, color=ORANGE, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#8b949e", lw=1, linestyle="--", label="Random")
    ax.fill_between(fpr, tpr, alpha=0.15, color=ORANGE)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", color="#c9d1d9")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Confusion matrix
    ax = axes[2]
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", ax=ax,
                xticklabels=["Employed", "Unemployed"],
                yticklabels=["Employed", "Unemployed"],
                linewidths=1, linecolor="#0d1117")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix", color="#c9d1d9")

    plt.tight_layout()
    plt.savefig("outputs/04_logistic_regression.png", dpi=150)
    plt.close()
    print("  [saved] outputs/04_logistic_regression.png")
    print(f"\n  Key finding: AUC = {auc:.3f}. Education and experience are the")
    print("  strongest protective factors against unemployment in this model.")

    return clf, auc


# ══════════════════════════════════════════════════════════════════════════════
# 7. ABSORPTION RATE & GENDER GAP HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

def gender_education_heatmap(df: pd.DataFrame):
    """
    Unemployment rate by education × gender — reveals intersectional disadvantage.
    """
    edu_order = ["No schooling", "Primary", "Secondary incomplete",
                 "Matric", "Certificate/Diploma", "Degree+"]
    pivot = df.pivot_table(
        values="unemployed", index="education", columns="gender", aggfunc="mean"
    ).reindex(edu_order) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, linecolor="#0d1117",
                cbar_kws={"label": "Unemployment Rate (%)", "shrink": 0.8})
    ax.set_title("Unemployment Rate (%) — Education × Gender",
                 fontsize=13, fontweight="bold", color=ORANGE, pad=12)
    ax.set_xlabel("Gender")
    ax.set_ylabel("Education Level")
    plt.tight_layout()
    plt.savefig("outputs/05_gender_education_heatmap.png", dpi=150)
    plt.close()
    print("  [saved] outputs/05_gender_education_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  SOUTH AFRICAN LABOUR MARKET ANALYSIS")
    print("=" * 60)

    print("\n[1/6] Generating QLFS-style dataset ...")
    df = generate_labour_dataset(n=8_000)
    df.to_csv("data/labour_market_data.csv", index=False)
    print(f"  Dataset shape: {df.shape}")

    print("\n[2/6] Descriptive overview ...")
    descriptive_overview(df)

    print("\n[3/6] Unemployment breakdown plots ...")
    breakdown_plots(df)

    print("\n[4/6] Youth & NEET analysis ...")
    youth_neet_analysis(df)

    print("\n[5/6] Income distribution ...")
    income_distribution(df)

    print("\n[6/6] Logistic regression model ...")
    logistic_regression_model(df)
    gender_education_heatmap(df)

    print("\n" + "=" * 60)
    print("  Analysis complete. All outputs saved to outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
