#!/usr/bin/env python3
"""
Comprehensive Exploratory Data Analysis (EDA) Script
====================================================
Generates visual figures and statistical analysis for BITE412L:
  1. Target Distributions (Continuous Loss % & Binary Loss Risk)
  2. Thermal Dynamics & Excursions vs Loss
  3. Humidity & Environmental Stress vs Loss
  4. Logistics Delays, Waiting Times & Disruption Causes vs Loss
  5. Shelf-Life (Days-to-Expiry) vs Loss
  6. Commodity-wise & Supply-Chain-Stage-wise Patterns
  7. Feature Correlation & Target Dependency Heatmap
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

DATASET_PATH = "dataset/synthetic_food_supply_chain.csv"
OUTPUT_DIR = "reports/eda_visualizations"

# Style settings
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8

def generate_all_eda():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 80)
    print("RUNNING COMPREHENSIVE EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 80)

    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded {len(df):,} records.")

    # -------------------------------------------------------------
    # 1. Target Distributions
    # -------------------------------------------------------------
    print("[1/7] Plotting Target Distributions...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Subplot A: Continuous synthetic_loss_percentage
    sns.histplot(df["synthetic_loss_percentage"], bins=50, kde=True, ax=axes[0], color="#1E88E5", alpha=0.6)
    p70 = np.percentile(df["synthetic_loss_percentage"], 70)
    mean_val = df["synthetic_loss_percentage"].mean()
    median_val = df["synthetic_loss_percentage"].median()

    axes[0].axvline(mean_val, color="#D81B60", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.2f}%")
    axes[0].axvline(median_val, color="#004D40", linestyle=":", linewidth=1.5, label=f"Median: {median_val:.2f}%")
    axes[0].axvline(p70, color="#FFC107", linestyle="-", linewidth=2.0, label=f"70th Pct (Risk Cutoff): {p70:.2f}%")
    axes[0].set_title("Distribution of Synthetic Food Loss Percentage", fontsize=13, fontweight='bold', pad=12)
    axes[0].set_xlabel("Synthetic Loss Percentage (%)", fontsize=11)
    axes[0].set_ylabel("Frequency (Records)", fontsize=11)
    axes[0].legend(frameon=True, facecolor='white', framealpha=0.9)

    # Subplot B: Binary loss_risk
    risk_counts = df["loss_risk"].value_counts().sort_index()
    colors = ["#43A047", "#E53935"]
    axes[1].pie(risk_counts, labels=["Low / Standard Risk (0)", "High Loss Risk (1)"],
                autopct='%1.1f%%', startangle=140, colors=colors,
                wedgeprops=dict(width=0.4, edgecolor='w', linewidth=2),
                textprops={'fontsize': 11, 'fontweight': 'bold'})
    axes[1].set_title("Binary Loss Risk Target Classification Balance", fontsize=13, fontweight='bold', pad=12)

    plt.tight_layout()
    fig1_path = os.path.join(OUTPUT_DIR, "eda_1_target_distributions.png")
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {fig1_path}")

    # -------------------------------------------------------------
    # 2. Temperature vs Food Loss
    # -------------------------------------------------------------
    print("[2/7] Plotting Temperature Dynamics vs Food Loss...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Scatter of simulated_temperature vs loss_pct by temp_class
    palette = {"ambient": "#FB8C00", "cold": "#039BE5", "frozen": "#5E35B1"}
    sample_df = df.sample(min(3000, len(df)), random_state=42)
    sns.scatterplot(data=sample_df, x="simulated_temperature", y="synthetic_loss_percentage",
                    hue="temp_class", palette=palette, alpha=0.6, s=25, ax=axes[0])
    axes[0].set_title("Simulated Sensor Temperature vs. Food Loss %", fontsize=13, fontweight='bold', pad=12)
    axes[0].set_xlabel("Effective Sensor Temperature (°C)", fontsize=11)
    axes[0].set_ylabel("Synthetic Food Loss Percentage (%)", fontsize=11)
    axes[0].legend(title="Commodity Class", frameon=True)

    # Boxplot of loss for breach vs in-band
    df["has_temp_breach"] = df["temp_deviation_score"] > 0
    sns.boxplot(data=df, x="has_temp_breach", y="synthetic_loss_percentage", ax=axes[1],
                palette=["#66BB6A", "#EF5350"])
    axes[1].set_xticklabels(["In-Band Safe Temperature", "Thermal Excursion Breach"])
    axes[1].set_title("Food Loss: Safe Temperature vs. Cold-Chain Excursion", fontsize=13, fontweight='bold', pad=12)
    axes[1].set_xlabel("Cold Chain Integrity Status", fontsize=11)
    axes[1].set_ylabel("Synthetic Food Loss Percentage (%)", fontsize=11)

    plt.tight_layout()
    fig2_path = os.path.join(OUTPUT_DIR, "eda_2_temperature_vs_loss.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {fig2_path}")

    # -------------------------------------------------------------
    # 3. Humidity vs Food Loss
    # -------------------------------------------------------------
    print("[3/7] Plotting Humidity vs Food Loss...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=300)

    sns.regplot(data=sample_df, x="Humidity", y="synthetic_loss_percentage",
                scatter_kws={'alpha': 0.3, 'color': '#0288D1', 's': 20},
                line_kws={'color': '#D32F2F', 'linewidth': 2}, ax=axes[0])
    axes[0].axvspan(40, 70, color='#C8E6C9', alpha=0.3, label="Optimal RH Band (40-70%)")
    axes[0].set_title("Relative Humidity (%) vs. Synthetic Loss %", fontsize=13, fontweight='bold', pad=12)
    axes[0].set_xlabel("Relative Humidity (%)", fontsize=11)
    axes[0].set_ylabel("Synthetic Food Loss (%)", fontsize=11)
    axes[0].legend(frameon=True)

    df["humidity_stress_zone"] = pd.cut(df["Humidity"], bins=[0, 40, 70, 100], labels=["Low (<40%)", "Optimal (40-70%)", "High (>70%)"])
    sns.barplot(data=df, x="humidity_stress_zone", y="synthetic_loss_percentage", ax=axes[1],
                palette=["#FFB74D", "#81C784", "#E57373"], ci=95)
    axes[1].set_title("Mean Food Loss by Humidity Stress Zone", fontsize=13, fontweight='bold', pad=12)
    axes[1].set_xlabel("Relative Humidity Zone", fontsize=11)
    axes[1].set_ylabel("Mean Loss Percentage (%)", fontsize=11)

    plt.tight_layout()
    fig3_path = os.path.join(OUTPUT_DIR, "eda_3_humidity_vs_loss.png")
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {fig3_path}")

    # -------------------------------------------------------------
    # 4. Logistics Delay & Waiting Time vs Food Loss
    # -------------------------------------------------------------
    print("[4/7] Plotting Logistics Delays & Waiting Times vs Food Loss...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=300)

    # Waiting Time Trend
    sns.regplot(data=sample_df, x="Waiting_Time", y="synthetic_loss_percentage", ax=axes[0],
                scatter_kws={'alpha': 0.3, 'color': '#7E57C2', 's': 18},
                line_kws={'color': '#FF7043', 'linewidth': 2})
    axes[0].set_title("Vehicle Waiting Time vs. Synthetic Food Loss", fontsize=13, fontweight='bold', pad=12)
    axes[0].set_xlabel("Waiting Time (Minutes)", fontsize=11)
    axes[0].set_ylabel("Synthetic Food Loss (%)", fontsize=11)

    # Delay Reason Impact
    order = df.groupby("Logistics_Delay_Reason")["synthetic_loss_percentage"].mean().sort_values(ascending=False).index
    sns.barplot(data=df, x="Logistics_Delay_Reason", y="synthetic_loss_percentage",
                order=order, palette="mako", ax=axes[1])
    axes[1].set_title("Mean Food Loss by Logistics Disruption Reason", fontsize=13, fontweight='bold', pad=12)
    axes[1].set_xlabel("Logistics Delay Reason", fontsize=11)
    axes[1].set_ylabel("Mean Loss Percentage (%)", fontsize=11)
    axes[1].tick_params(axis='x', rotation=20)

    plt.tight_layout()
    fig4_path = os.path.join(OUTPUT_DIR, "eda_4_logistics_delay_vs_loss.png")
    plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {fig4_path}")

    # -------------------------------------------------------------
    # 5. Shelf-Life (Days-to-Expiry) vs Food Loss
    # -------------------------------------------------------------
    print("[5/7] Plotting Shelf-Life (Days-to-Expiry) vs Food Loss...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=300)

    sns.lineplot(data=df, x="days_to_expiry", y="synthetic_loss_percentage", color="#00897B", linewidth=2.5, ax=axes[0])
    axes[0].set_title("Mean Food Loss Progression as Products Approach Expiry", fontsize=13, fontweight='bold', pad=12)
    axes[0].set_xlabel("Remaining Shelf-Life (Days to Expiry)", fontsize=11)
    axes[0].set_ylabel("Synthetic Loss Percentage (%)", fontsize=11)
    axes[0].invert_xaxis()  # So moving left-to-right simulates counting down to expiration!

    # Packaging Material impact
    pack_order = df.groupby("packaging_material")["synthetic_loss_percentage"].mean().sort_values(ascending=False).index
    sns.boxplot(data=df, x="packaging_material", y="synthetic_loss_percentage",
                order=pack_order, palette="Set2", ax=axes[1])
    axes[1].set_title("Food Loss Distribution Across Packaging Materials", fontsize=13, fontweight='bold', pad=12)
    axes[1].set_xlabel("Packaging Material", fontsize=11)
    axes[1].set_ylabel("Synthetic Loss Percentage (%)", fontsize=11)

    plt.tight_layout()
    fig5_path = os.path.join(OUTPUT_DIR, "eda_5_shelf_life_vs_loss.png")
    plt.savefig(fig5_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {fig5_path}")

    # -------------------------------------------------------------
    # 6. Commodity & Stage Patterns
    # -------------------------------------------------------------
    print("[6/7] Plotting Commodity & Supply Chain Stage Patterns...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # Top 12 Commodities by Mean Loss
    top_comms = df.groupby("commodity")["synthetic_loss_percentage"].mean().sort_values(ascending=False).head(12)
    sns.barplot(x=top_comms.values, y=top_comms.index, palette="flare", ax=axes[0])
    axes[0].set_title("Top 12 Most Vulnerable Commodities (Highest Mean Loss %)", fontsize=13, fontweight='bold', pad=12)
    axes[0].set_xlabel("Mean Loss Percentage (%)", fontsize=11)
    axes[0].set_ylabel("Commodity", fontsize=11)

    # Loss by Supply Chain Stage
    stage_order = df.groupby("food_supply_stage")["synthetic_loss_percentage"].mean().sort_values(ascending=False).index
    sns.boxplot(data=df, x="food_supply_stage", y="synthetic_loss_percentage",
                order=stage_order, palette="Blues_r", ax=axes[1])
    axes[1].set_title("Food Loss Distribution Across Supply Chain Stages", fontsize=13, fontweight='bold', pad=12)
    axes[1].set_xlabel("Food Supply Stage", fontsize=11)
    axes[1].set_ylabel("Synthetic Loss Percentage (%)", fontsize=11)
    axes[1].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    fig6_path = os.path.join(OUTPUT_DIR, "eda_6_commodity_and_stage_patterns.png")
    plt.savefig(fig6_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {fig6_path}")

    # -------------------------------------------------------------
    # 7. Correlation Heatmap
    # -------------------------------------------------------------
    print("[7/7] Plotting Feature Correlation Heatmap...")
    plt.figure(figsize=(12, 10), dpi=300)

    numeric_cols = [
        "simulated_temperature", "Humidity", "Waiting_Time", "Logistics_Delay",
        "Inventory_Level", "Asset_Utilization", "Demand_Forecast",
        "User_Transaction_Amount", "User_Purchase_Frequency", "days_to_expiry",
        "synthetic_loss_percentage", "loss_risk"
    ]
    corr = df[numeric_cols].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.6, vmin=-0.6, center=0,
                annot=True, fmt=".2f", square=True, linewidths=.5, cbar_kws={"shrink": .8})
    plt.title("Operational Feature Correlation with Food Loss and Risk", fontsize=14, fontweight='bold', pad=14)

    fig7_path = os.path.join(OUTPUT_DIR, "eda_7_correlation_matrix.png")
    plt.savefig(fig7_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {fig7_path}")

    print("=" * 80)
    print(f"ALL EDA FIGURES SUCCESSFULLY GENERATED IN: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    generate_all_eda()
