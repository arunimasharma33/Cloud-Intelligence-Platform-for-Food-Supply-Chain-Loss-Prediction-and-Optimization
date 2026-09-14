#!/usr/bin/env python3
"""
Comprehensive Model Training, Benchmarking & Evaluation Pipeline
================================================================
Trains and evaluates 6 Regression and 6 Classification models on the
leakage-free BITE412L dataset:

Regression Models:
  1. Dummy (Mean Baseline)
  2. Ridge Regression
  3. Decision Tree Regressor
  4. Random Forest Regressor
  5. Histogram Gradient Boosting Regressor
  6. Extra Trees Regressor

Classification Models:
  1. Dummy (Stratified Chance Baseline)
  2. Logistic Regression
  3. Decision Tree Classifier
  4. Random Forest Classifier
  5. Histogram Gradient Boosting Classifier
  6. Extra Trees Classifier

Outputs:
  - models/champion_food_loss_regressor.joblib
  - models/champion_loss_risk_classifier.joblib
  - reports/model_evaluations/comprehensive_model_metrics.json
  - reports/model_evaluations/test_predictions.csv
  - reports/model_evaluations/feature_importance.json
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier,
    ExtraTreesRegressor,
    ExtraTreesClassifier
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from sklearn.inspection import permutation_importance

TRAIN_PATH = "dataset/train_set.csv"
TEST_PATH = "dataset/test_set.csv"
MODELS_DIR = "models"
REPORTS_DIR = "reports/model_evaluations"

def run_pipeline():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("=" * 85)
    print("BITE412L COMPREHENSIVE MACHINE LEARNING TRAINING & BENCHMARKING ENGINE")
    print("=" * 85)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f"Loaded Train set: {train_df.shape[0]:,} records | Test set: {test_df.shape[0]:,} records")

    targets = ["synthetic_loss_percentage", "loss_risk"]
    metadata = ["Timestamp", "Asset_ID"]
    feature_cols = [c for c in train_df.columns if c not in targets and c not in metadata]

    numeric_features = train_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = train_df[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"Operational Feature Matrix: {len(feature_cols)} total features")
    print(f"  • Numeric ({len(numeric_features)}): {numeric_features}")
    print(f"  • Categorical ({len(categorical_features)}): {categorical_features}")

    # Standard Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )

    X_train = train_df[feature_cols]
    y_train_reg = train_df["synthetic_loss_percentage"]
    y_train_clf = train_df["loss_risk"]

    X_test = test_df[feature_cols]
    y_test_reg = test_df["synthetic_loss_percentage"]
    y_test_clf = test_df["loss_risk"]

    predictions_df = pd.DataFrame({
        "true_loss_percentage": y_test_reg,
        "true_loss_risk": y_test_clf
    })

    # =========================================================================
    # PART 1: REGRESSION BENCHMARK
    # =========================================================================
    print("\n" + "=" * 85)
    print("PART 1: REGRESSION BENCHMARK — PREDICTING 'synthetic_loss_percentage'")
    print("=" * 85)

    reg_candidates = {
        "Dummy (Mean)": DummyRegressor(strategy="mean"),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=150, max_depth=8, learning_rate=0.08, random_state=42),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1)
    }

    reg_metrics = []
    best_reg_name, best_reg_pipe, best_reg_r2 = None, None, -np.inf

    for name, model in reg_candidates.items():
        print(f"Training [{name:<22}] ... ", end="", flush=True)
        pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train_reg)
        preds = pipe.predict(X_test)
        predictions_df[f"pred_reg_{name.replace(' ', '_').lower()}"] = preds

        mae = mean_absolute_error(y_test_reg, preds)
        mse = mean_squared_error(y_test_reg, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_reg, preds)
        mape = mean_absolute_percentage_error(np.maximum(y_test_reg, 0.01), np.maximum(preds, 0.01))

        reg_metrics.append({
            "Model": name,
            "MAE (%)": round(float(mae), 3),
            "MSE (%²)": round(float(mse), 3),
            "RMSE (%)": round(float(rmse), 3),
            "R2 Score": round(float(r2), 4),
            "MAPE": round(float(mape), 4)
        })
        print(f"DONE -> MAE: {mae:.3f}%, RMSE: {rmse:.3f}%, R²: {r2:.4f}")

        if r2 > best_reg_r2:
            best_reg_r2 = r2
            best_reg_name = name
            best_reg_pipe = pipe

    reg_leaderboard = pd.DataFrame(reg_metrics).sort_values(by="R2 Score", ascending=False)
    print("\n" + "-" * 85)
    print("REGRESSION MODEL LEADERBOARD:")
    print("-" * 85)
    print(reg_leaderboard.to_string(index=False))

    champion_reg_file = os.path.join(MODELS_DIR, "champion_food_loss_regressor.joblib")
    joblib.dump(best_reg_pipe, champion_reg_file)
    print(f"\n✓ Champion Regressor: {best_reg_name} (R² = {best_reg_r2:.4f}) saved to {champion_reg_file}")

    # =========================================================================
    # PART 2: CLASSIFICATION BENCHMARK
    # =========================================================================
    print("\n" + "=" * 85)
    print("PART 2: CLASSIFICATION BENCHMARK — PREDICTING 'loss_risk' (EARLY WARNING)")
    print("=" * 85)

    clf_candidates = {
        "Dummy (Stratified)": DummyClassifier(strategy="stratified", random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=150, max_depth=8, learning_rate=0.08, random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1)
    }

    clf_metrics = []
    best_clf_name, best_clf_pipe, best_clf_auc = None, None, -np.inf
    confusion_matrices = {}

    for name, model in clf_candidates.items():
        print(f"Training [{name:<22}] ... ", end="", flush=True)
        pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train_clf)
        preds = pipe.predict(X_test)
        predictions_df[f"pred_clf_{name.replace(' ', '_').lower()}"] = preds

        if hasattr(pipe, "predict_proba"):
            probs = pipe.predict_proba(X_test)[:, 1]
            predictions_df[f"prob_clf_{name.replace(' ', '_').lower()}"] = probs
            auc = roc_auc_score(y_test_clf, probs)
            pr_auc = average_precision_score(y_test_clf, probs)
        else:
            probs = preds.astype(float)
            predictions_df[f"prob_clf_{name.replace(' ', '_').lower()}"] = probs
            auc = 0.5
            pr_auc = 0.3

        acc = accuracy_score(y_test_clf, preds)
        prec = precision_score(y_test_clf, preds, zero_division=0)
        rec = recall_score(y_test_clf, preds, zero_division=0)
        f1 = f1_score(y_test_clf, preds, zero_division=0)

        cm = confusion_matrix(y_test_clf, preds)
        confusion_matrices[name] = cm.tolist()

        clf_metrics.append({
            "Model": name,
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1 Score": round(float(f1), 4),
            "ROC-AUC": round(float(auc), 4),
            "PR-AUC": round(float(pr_auc), 4)
        })
        print(f"DONE -> Acc: {acc:.2%}, F1: {f1:.4f}, ROC-AUC: {auc:.4f}")

        if auc > best_clf_auc:
            best_clf_auc = auc
            best_clf_name = name
            best_clf_pipe = pipe

    clf_leaderboard = pd.DataFrame(clf_metrics).sort_values(by="ROC-AUC", ascending=False)
    print("\n" + "-" * 85)
    print("CLASSIFICATION MODEL LEADERBOARD:")
    print("-" * 85)
    print(clf_leaderboard.to_string(index=False))

    champion_clf_file = os.path.join(MODELS_DIR, "champion_loss_risk_classifier.joblib")
    joblib.dump(best_clf_pipe, champion_clf_file)
    print(f"\n✓ Champion Classifier: {best_clf_name} (ROC-AUC = {best_clf_auc:.4f}) saved to {champion_clf_file}")

    # =========================================================================
    # PART 3: FEATURE IMPORTANCE AUDIT
    # =========================================================================
    print("\n" + "=" * 85)
    print("PART 3: FEATURE IMPORTANCE COMPUTATION FOR CHAMPION MODELS")
    print("=" * 85)
    
    # Compute permutation importance on sample of test set for speed & comparability across model types
    sample_idx = np.random.RandomState(42).choice(len(X_test), size=min(1200, len(X_test)), replace=False)
    X_test_sample = X_test.iloc[sample_idx]
    y_test_reg_sample = y_test_reg.iloc[sample_idx]
    y_test_clf_sample = y_test_clf.iloc[sample_idx]

    print("Computing Permutation Feature Importance for Champion Regressor...")
    perm_reg = permutation_importance(best_reg_pipe, X_test_sample, y_test_reg_sample, n_repeats=5, random_state=42, n_jobs=-1)
    reg_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": perm_reg.importances_mean,
        "importance_std": perm_reg.importances_std
    }).sort_values(by="importance_mean", ascending=False)

    print("Computing Permutation Feature Importance for Champion Classifier...")
    perm_clf = permutation_importance(best_clf_pipe, X_test_sample, y_test_clf_sample, n_repeats=5, random_state=42, n_jobs=-1)
    clf_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": perm_clf.importances_mean,
        "importance_std": perm_clf.importances_std
    }).sort_values(by="importance_mean", ascending=False)

    feature_importance_dict = {
        "regression_champion": best_reg_name,
        "regression_importances": reg_imp.to_dict(orient="records"),
        "classification_champion": best_clf_name,
        "classification_importances": clf_imp.to_dict(orient="records")
    }

    # =========================================================================
    # PART 4: PERSISTENCE & INFERENCE VERIFICATION
    # =========================================================================
    # Save predictions
    pred_path = os.path.join(REPORTS_DIR, "test_predictions.csv")
    predictions_df.to_csv(pred_path, index=False)
    print(f"\n✓ Saved test predictions to: {pred_path}")

    # Save metrics JSON
    metrics_path = os.path.join(REPORTS_DIR, "comprehensive_model_metrics.json")
    full_report = {
        "regression": {
            "leaderboard": reg_metrics,
            "champion_model": best_reg_name,
            "best_r2": best_reg_r2
        },
        "classification": {
            "leaderboard": clf_metrics,
            "champion_model": best_clf_name,
            "best_roc_auc": best_clf_auc,
            "confusion_matrices": confusion_matrices
        }
    }
    with open(metrics_path, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"✓ Saved full metrics report to: {metrics_path}")

    # Save feature importance JSON
    imp_path = os.path.join(REPORTS_DIR, "feature_importance.json")
    with open(imp_path, "w") as f:
        json.dump(feature_importance_dict, f, indent=2)
    print(f"✓ Saved feature importance data to: {imp_path}")

    # Test Reloading & Sanity Inference Test
    print("\nTesting Champion Model Deserialization & Sample Edge Inference...")
    loaded_reg = joblib.load(champion_reg_file)
    loaded_clf = joblib.load(champion_clf_file)

    sample_input = X_test.iloc[[0]]
    sample_reg_pred = float(loaded_reg.predict(sample_input)[0])
    sample_clf_risk = int(loaded_clf.predict(sample_input)[0])
    sample_clf_prob = float(loaded_clf.predict_proba(sample_input)[0, 1])

    print("Sample Telemetry Row:")
    for col, val in sample_input.iloc[0].items():
        print(f"  • {col}: {val}")
    print("\nInference Output:")
    print(f"  -> Predicted Loss Percentage: {sample_reg_pred:.2f}%")
    print(f"  -> Predicted Loss Risk Flag:  {sample_clf_risk} (Probability: {sample_clf_prob:.1%})")
    print(f"  -> Actual Values: Loss % = {y_test_reg.iloc[0]:.2f}%, Risk = {y_test_clf.iloc[0]}")

    print("\n" + "=" * 85)
    print("ALL MODELS TRAINED, EVALUATED, AND VALIDATED SUCCESSFULLY.")
    print("=" * 85)

if __name__ == "__main__":
    run_pipeline()
