#!/usr/bin/env python3
"""
Model Diagnostic Visualizations Generator
=========================================
Generates 6 publication-grade figures in reports/model_visualizations/:
  1. model_1_regression_actual_vs_predicted.png
  2. model_2_regression_residuals.png
  3. model_3_classification_roc_pr_curves.png
  4. model_4_classification_confusion_matrices.png
  5. model_5_feature_importance_comparison.png
  6. model_6_leaderboard_comparison.png
"""

import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

PRED_PATH = "reports/model_evaluations/test_predictions.csv"
METRICS_PATH = "reports/model_evaluations/comprehensive_model_metrics.json"
IMP_PATH = "reports/model_evaluations/feature_importance.json"
OUTPUT_DIR = "reports/model_visualizations"

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8

def generate_visualizations():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 85)
    print("GENERATING MODEL DIAGNOSTIC VISUALIZATIONS")
    print("=" * 85)

    if not os.path.exists(PRED_PATH) or not os.path.exists(METRICS_PATH):
        print("Required evaluation files not found. Run train_and_evaluate_models.py first.")
        return

    pred_df = pd.read_csv(PRED_PATH)
    with open(METRICS_PATH) as f:
        metrics_data = json.load(f)
    with open(IMP_PATH) as f:
        imp_data = json.load(f)

    # -------------------------------------------------------------
    # 1. Actual vs Predicted Scatter (Parity Plot)
    # -------------------------------------------------------------
    print("[1/6] Rendering Actual vs. Predicted Parity Plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    y_true = pred_df["true_loss_percentage"]
    
    # Model A: Ridge Regression
    y_pred_ridge = pred_df["pred_reg_ridge_regression"]
    sns.scatterplot(x=y_true, y=y_pred_ridge, alpha=0.3, color="#1976D2", s=18, ax=axes[0])
    max_val = max(y_true.max(), y_pred_ridge.max()) + 2
    axes[0].plot([0, max_val], [0, max_val], "r--", linewidth=2, label="Ideal Parity (y = x)")
    axes[0].set_title("Ridge Regression: Actual vs. Predicted Loss %", fontsize=12, fontweight='bold', pad=10)
    axes[0].set_xlabel("Ground Truth Synthetic Loss (%)", fontsize=11)
    axes[0].set_ylabel("Predicted Loss (%)", fontsize=11)
    axes[0].set_xlim(0, max_val)
    axes[0].set_ylim(0, max_val)
    axes[0].legend(frameon=True)

    # Model B: HistGradientBoosting
    y_pred_hgb = pred_df["pred_reg_histgradientboosting"]
    sns.scatterplot(x=y_true, y=y_pred_hgb, alpha=0.3, color="#7B1FA2", s=18, ax=axes[1])
    axes[1].plot([0, max_val], [0, max_val], "r--", linewidth=2, label="Ideal Parity (y = x)")
    axes[1].set_title("HistGradientBoosting: Actual vs. Predicted Loss %", fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xlabel("Ground Truth Synthetic Loss (%)", fontsize=11)
    axes[1].set_ylabel("Predicted Loss (%)", fontsize=11)
    axes[1].set_xlim(0, max_val)
    axes[1].set_ylim(0, max_val)
    axes[1].legend(frameon=True)

    plt.tight_layout()
    fig1 = os.path.join(OUTPUT_DIR, "model_1_regression_actual_vs_predicted.png")
    plt.savefig(fig1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {fig1}")

    # -------------------------------------------------------------
    # 2. Residual Distribution & Homoscedasticity Analysis
    # -------------------------------------------------------------
    print("[2/6] Rendering Residual Analysis Plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    residuals = y_true - y_pred_ridge
    sns.histplot(residuals, bins=50, kde=True, color="#00897B", ax=axes[0])
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Error Baseline")
    axes[0].set_title("Regression Residuals Distribution (Ridge)", fontsize=12, fontweight='bold', pad=10)
    axes[0].set_xlabel("Residual Error (Actual - Predicted %)", fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)
    axes[0].legend(frameon=True)

    sns.scatterplot(x=y_pred_ridge, y=residuals, alpha=0.3, color="#FB8C00", s=18, ax=axes[1])
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_title("Residuals vs. Fitted Values (Homoscedasticity Check)", fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xlabel("Fitted Values (Predicted Loss %)", fontsize=11)
    axes[1].set_ylabel("Residual Error (%)", fontsize=11)

    plt.tight_layout()
    fig2 = os.path.join(OUTPUT_DIR, "model_2_regression_residuals.png")
    plt.savefig(fig2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {fig2}")

    # -------------------------------------------------------------
    # 3. Multi-Model ROC and Precision-Recall Curves
    # -------------------------------------------------------------
    print("[3/6] Rendering ROC & Precision-Recall Curves...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    y_clf_true = pred_df["true_loss_risk"]
    clf_models = [
        ("HistGradientBoosting", "prob_clf_histgradientboosting", "#E53935"),
        ("Logistic Regression", "prob_clf_logistic_regression", "#1E88E5"),
        ("Random Forest", "prob_clf_random_forest", "#43A047"),
        ("Extra Trees", "prob_clf_extra_trees", "#8E24AA"),
        ("Decision Tree", "prob_clf_decision_tree", "#FB8C00")
    ]

    for label, col, color in clf_models:
        if col in pred_df.columns:
            probs = pred_df[col]
            fpr, tpr, _ = roc_curve(y_clf_true, probs)
            prec, rec, _ = precision_recall_curve(y_clf_true, probs)
            
            # Extract AUC from metrics
            auc_val = next((m["ROC-AUC"] for m in metrics_data["classification"]["leaderboard"] if m["Model"] == label), 0.5)
            pr_val = next((m["PR-AUC"] for m in metrics_data["classification"]["leaderboard"] if m["Model"] == label), 0.3)

            axes[0].plot(fpr, tpr, color=color, linewidth=2, label=f"{label} (AUC = {auc_val:.3f})")
            axes[1].plot(rec, prec, color=color, linewidth=2, label=f"{label} (AP = {pr_val:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.500)")
    axes[0].set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, fontweight='bold', pad=10)
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    axes[0].set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    axes[0].legend(loc="lower right", frameon=True)

    baseline_pr = y_clf_true.mean()
    axes[1].axhline(baseline_pr, color="black", linestyle="--", label=f"Baseline Prevalance ({baseline_pr:.1%})")
    axes[1].set_title("Precision-Recall (PR) Curves", fontsize=13, fontweight='bold', pad=10)
    axes[1].set_xlabel("Recall", fontsize=11)
    axes[1].set_ylabel("Precision", fontsize=11)
    axes[1].legend(loc="upper right", frameon=True)

    plt.tight_layout()
    fig3 = os.path.join(OUTPUT_DIR, "model_3_classification_roc_pr_curves.png")
    plt.savefig(fig3, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {fig3}")

    # -------------------------------------------------------------
    # 4. Confusion Matrices (Raw Counts and Normalized Rates)
    # -------------------------------------------------------------
    print("[4/6] Rendering Confusion Matrices for Champion Classifier...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    best_clf_col = "pred_clf_histgradientboosting"
    cm = confusion_matrix(y_clf_true, pred_df[best_clf_col])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    labels = ["Low Risk (0)", "High Risk (1)"]

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0],
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 13, "weight": "bold"})
    axes[0].set_title("Champion Classifier: Raw Confusion Matrix", fontsize=12, fontweight='bold', pad=10)
    axes[0].set_xlabel("Predicted Label", fontsize=11)
    axes[0].set_ylabel("Actual Ground Truth Label", fontsize=11)

    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Greens", cbar=False, ax=axes[1],
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 13, "weight": "bold"})
    axes[1].set_title("Champion Classifier: Normalized Confusion Rates", fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xlabel("Predicted Label", fontsize=11)
    axes[1].set_ylabel("Actual Ground Truth Label", fontsize=11)

    plt.tight_layout()
    fig4 = os.path.join(OUTPUT_DIR, "model_4_classification_confusion_matrices.png")
    plt.savefig(fig4, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {fig4}")

    # -------------------------------------------------------------
    # 5. Feature Importance Comparison
    # -------------------------------------------------------------
    print("[5/6] Rendering Permutation Feature Importance Charts...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    reg_imp_df = pd.DataFrame(imp_data["regression_importances"]).head(12)
    clf_imp_df = pd.DataFrame(imp_data["classification_importances"]).head(12)

    sns.barplot(data=reg_imp_df, x="importance_mean", y="feature", palette="Blues_r", ax=axes[0])
    axes[0].set_title(f"Permutation Importance: Regressor ({imp_data['regression_champion']})", fontsize=12, fontweight='bold', pad=10)
    axes[0].set_xlabel("Mean Decrease in R² Score on Permutation", fontsize=11)
    axes[0].set_ylabel("Operational Feature", fontsize=11)

    sns.barplot(data=clf_imp_df, x="importance_mean", y="feature", palette="Reds_r", ax=axes[1])
    axes[1].set_title(f"Permutation Importance: Classifier ({imp_data['classification_champion']})", fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xlabel("Mean Decrease in Accuracy on Permutation", fontsize=11)
    axes[1].set_ylabel("Operational Feature", fontsize=11)

    plt.tight_layout()
    fig5 = os.path.join(OUTPUT_DIR, "model_5_feature_importance_comparison.png")
    plt.savefig(fig5, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {fig5}")

    # -------------------------------------------------------------
    # 6. Leaderboard Comparison Bar Charts
    # -------------------------------------------------------------
    print("[6/6] Rendering Model Leaderboard Bar Charts...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    reg_df = pd.DataFrame(metrics_data["regression"]["leaderboard"]).sort_values("R2 Score")
    sns.barplot(data=reg_df, x="R2 Score", y="Model", palette="crest", ax=axes[0])
    axes[0].set_title("Regression Models: R² Score Comparison (Test Set)", fontsize=12, fontweight='bold', pad=10)
    axes[0].set_xlabel("R² Score (Higher is Better)", fontsize=11)
    axes[0].set_xlim(-0.05, 0.6)
    for p in axes[0].patches:
        w = p.get_width()
        axes[0].annotate(f"{w:.3f}", (max(w, 0.01) + 0.01, p.get_y() + p.get_height() / 2),
                         va='center', fontsize=10, fontweight='bold')

    clf_df = pd.DataFrame(metrics_data["classification"]["leaderboard"]).sort_values("ROC-AUC")
    sns.barplot(data=clf_df, x="ROC-AUC", y="Model", palette="flare", ax=axes[1])
    axes[1].set_title("Classification Models: ROC-AUC Comparison (Test Set)", fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xlabel("ROC-AUC Score (Higher is Better)", fontsize=11)
    axes[1].set_xlim(0.4, 0.95)
    for p in axes[1].patches:
        w = p.get_width()
        axes[1].annotate(f"{w:.3f}", (w + 0.01, p.get_y() + p.get_height() / 2),
                         va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    fig6 = os.path.join(OUTPUT_DIR, "model_6_leaderboard_comparison.png")
    plt.savefig(fig6, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {fig6}")

    print("=" * 85)
    print(f"ALL 6 DIAGNOSTIC VISUALIZATIONS SAVED IN: {OUTPUT_DIR}")
    print("=" * 85)

if __name__ == "__main__":
    generate_visualizations()
