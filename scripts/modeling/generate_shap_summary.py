#!/usr/bin/env python3
"""
Global SHAP Feature Importance Summary Generator
=================================================
Computes mean |SHAP value| across a sample of the test set for both champion
pipelines and renders a single publication-style comparison figure:
  reports/model_visualizations/model_7_shap_feature_importance.png

This complements the existing per-prediction explanations (see
scripts/inference/shap_explainer.py) with a GLOBAL view: which features
matter most for the model overall, not just for one shipment.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
os.chdir(PROJECT_ROOT)

INFERENCE_DIR = os.path.join(PROJECT_ROOT, "scripts", "inference")
if INFERENCE_DIR not in sys.path:
    sys.path.insert(0, INFERENCE_DIR)

from shap_explainer import PredictionExplainer, FEATURE_LABELS  # noqa: E402
from inference_engine import REQUIRED_FEATURES  # noqa: E402

REGRESSOR_PATH = "models/champion_food_loss_regressor.joblib"
CLASSIFIER_PATH = "models/champion_loss_risk_classifier.joblib"
TEST_PATH = "dataset/splits/test_set.csv"
OUTPUT_DIR = "reports/model_visualizations"
SAMPLE_SIZE = 300  # rows drawn from the test split to estimate global importance

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8


def compute_global_importance(explainer, prep, groups, df, top_n=12, is_classifier=False):
    """Runs the explainer's transform + underlying shap explainer across a
    sample and returns mean |impact| per original feature, sorted desc."""
    transformed = prep.transform(df[REQUIRED_FEATURES])
    if is_classifier:
        shap_values = explainer(transformed).values
        if shap_values.ndim > 2:
            shap_values = shap_values[:, :, 1]
    else:
        shap_values = explainer(transformed).values

    agg = {}
    for col, sl in groups:
        col_impact = np.abs(shap_values[:, sl]).sum(axis=1).mean()
        agg[col] = agg.get(col, 0.0) + col_impact

    ranked = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:top_n]
    labels = [FEATURE_LABELS.get(c, c) for c, _ in ranked]
    values = [v for _, v in ranked]
    return labels[::-1], values[::-1]  # reversed for horizontal barh top-down


def main():
    print("=" * 85)
    print("GENERATING GLOBAL SHAP FEATURE IMPORTANCE SUMMARY")
    print("=" * 85)

    if not (os.path.exists(REGRESSOR_PATH) and os.path.exists(CLASSIFIER_PATH)):
        print("Champion models not found. Train models first.")
        return
    if not os.path.exists(TEST_PATH):
        print(f"Test split not found at {TEST_PATH}.")
        return

    regressor = joblib.load(REGRESSOR_PATH)
    classifier = joblib.load(CLASSIFIER_PATH)

    test_df = pd.read_csv(TEST_PATH)
    sample_df = test_df.sample(n=min(SAMPLE_SIZE, len(test_df)), random_state=42)

    explainer_bundle = PredictionExplainer(regressor, classifier, REQUIRED_FEATURES)

    reg_labels, reg_values = compute_global_importance(
        explainer_bundle.reg_explainer, explainer_bundle.reg_prep,
        explainer_bundle._reg_groups, sample_df, is_classifier=False
    )
    clf_labels, clf_values = compute_global_importance(
        explainer_bundle.clf_explainer, explainer_bundle.clf_prep,
        explainer_bundle._clf_groups, sample_df, is_classifier=True
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    axes[0].barh(reg_labels, reg_values, color="#4C72B0", edgecolor="white", height=0.65)
    axes[0].set_title("Ridge Regressor — Global Feature Importance\n(mean |SHAP| — % loss units)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Mean |SHAP value| (percentage points of predicted loss)")

    axes[1].barh(clf_labels, clf_values, color="#DD8452", edgecolor="white", height=0.65)
    axes[1].set_title("HistGradientBoosting Classifier — Global Feature Importance\n(mean |SHAP| — log-odds units)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Mean |SHAP value| (log-odds contribution to risk)")

    fig.suptitle(f"Global SHAP Feature Importance (n={len(sample_df)} test shipments)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "model_7_shap_feature_importance.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
