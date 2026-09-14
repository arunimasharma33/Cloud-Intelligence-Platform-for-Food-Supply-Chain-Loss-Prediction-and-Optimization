#!/usr/bin/env python3
"""
ML Dataset Preparation and Leakage Audit Script
==============================================
Prepares the clean, leak-free, ML-ready dataset for downstream regression and classification:
  1. Identifies and removes target leakage variables and redundant fields
  2. Binds calibrated sensor reading (temperature_celsius)
  3. Enforces physical boundaries (clipping coordinate jitter)
  4. Extracts operational temporal features
  5. Exports ml_ready_food_supply_chain.csv and stratified Train/Test splits
  6. Evaluates baseline models to verify valid predictive signal without leakage
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score, f1_score

INPUT_PATH = "dataset/synthetic_food_supply_chain.csv"
OUTPUT_ML_PATH = "dataset/ml_ready_food_supply_chain.csv"
OUTPUT_TRAIN_PATH = "dataset/train_set.csv"
OUTPUT_TEST_PATH = "dataset/test_set.csv"
METADATA_PATH = "reports/ml_feature_metadata.json"

def prepare_ml_dataset():
    print("=" * 80)
    print("BITE412L MACHINE LEARNING DATASET PREPARATION & LEAKAGE AUDIT")
    print("=" * 80)

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded raw synthetic dataset: {df.shape[0]:,} rows × {df.shape[1]} columns.")

    # 1. Coordinate Clipping (Fixing 3% Gaussian jitter out-of-bound coordinates)
    df["Latitude"] = df["Latitude"].clip(-90.0, 90.0)
    df["Longitude"] = df["Longitude"].clip(-180.0, 180.0)
    print("  ✓ Clipped GPS Latitude to [-90, 90] and Longitude to [-180, 180].")

    # 2. Temporal Feature Extraction
    df["dt"] = pd.to_datetime(df["Timestamp"])
    df["hour_of_day"] = df["dt"].dt.hour
    df["day_of_week"] = df["dt"].dt.dayofweek
    df["month"] = df["dt"].dt.month
    print("  ✓ Extracted temporal features: hour_of_day, day_of_week, month.")

    # 3. Sensor Consolidation
    # simulated_temperature is the true effective cargo sensor reading calibrated per commodity
    df["temperature_celsius"] = df["simulated_temperature"]

    # 4. Target Leakage & Redundancy Audit
    columns_to_drop = [
        "dt",
        # Redundant raw temperatures
        "Temperature",
        "raw_temperature",
        "simulated_temperature",
        # Redundant stage lookup column
        "fao_stage",
        # --- CRITICAL TARGET LEAKAGE PREVENTION ---
        # These 4 columns are direct arithmetic intermediate variables from the generation formula:
        # loss = base(fao_base_loss_mean) + w_t * temp_deviation_score + w_h * humidity_deviation_score + w_e * expiry_pressure_score
        "fao_base_loss_mean",
        "temp_deviation_score",
        "humidity_deviation_score",
        "expiry_pressure_score"
    ]

    clean_df = df.drop(columns=columns_to_drop)

    # Feature categorization
    target_cols = ["synthetic_loss_percentage", "loss_risk"]
    metadata_cols = ["Timestamp", "Asset_ID"]
    
    feature_cols = [c for c in clean_df.columns if c not in target_cols and c not in metadata_cols]
    numeric_features = clean_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = clean_df[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()

    print("\nFeature Matrix Specification:")
    print(f"  • Total ML Input Features: {len(feature_cols)}")
    print(f"  • Numeric Features ({len(numeric_features)}): {numeric_features}")
    print(f"  • Categorical Features ({len(categorical_features)}): {categorical_features}")
    print(f"  • Metadata / Identifier Cols ({len(metadata_cols)}): {metadata_cols}")
    print(f"  • Targets ({len(target_cols)}): {target_cols}")

    # Reorder columns logically: Metadata -> Features -> Targets
    ordered_cols = metadata_cols + feature_cols + target_cols
    ml_ready_df = clean_df[ordered_cols]

    # Save complete ML ready dataset
    ml_ready_df.to_csv(OUTPUT_ML_PATH, index=False)
    print(f"\n✓ Saved clean ML dataset: {OUTPUT_ML_PATH} ({ml_ready_df.shape[0]:,} rows × {ml_ready_df.shape[1]} columns)")

    # 5. Stratified Train / Test Split (80% Train, 20% Test)
    train_df, test_df = train_test_split(
        ml_ready_df,
        test_size=0.20,
        random_state=42,
        stratify=ml_ready_df["loss_risk"]
    )
    train_df.to_csv(OUTPUT_TRAIN_PATH, index=False)
    test_df.to_csv(OUTPUT_TEST_PATH, index=False)
    print(f"✓ Saved stratified Train split: {OUTPUT_TRAIN_PATH} ({len(train_df):,} rows)")
    print(f"✓ Saved stratified Test split:  {OUTPUT_TEST_PATH} ({len(test_df):,} rows)")
    print(f"  Train positive loss_risk rate: {train_df['loss_risk'].mean():.2%}")
    print(f"  Test positive loss_risk rate:  {test_df['loss_risk'].mean():.2%}")

    # 6. Sanity Baseline ML Validation (Confirm realistic learning without trivial 1.0 leakage)
    print("\n[Sanity Benchmark] Running Baseline Model Validation on Operational Features...")
    
    # Simple One-Hot / Ordinal preparation for benchmark
    X_train_enc = pd.get_dummies(train_df[feature_cols], drop_first=True)
    X_test_enc = pd.get_dummies(test_df[feature_cols], drop_first=True)
    X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join='left', axis=1, fill_value=0)

    # A. Regression Sanity Check
    rf_reg = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
    rf_reg.fit(X_train_enc, train_df["synthetic_loss_percentage"])
    reg_preds = rf_reg.predict(X_test_enc)
    r2 = r2_score(test_df["synthetic_loss_percentage"], reg_preds)
    rmse = np.sqrt(mean_squared_error(test_df["synthetic_loss_percentage"], reg_preds))
    print(f"  [Regression] Baseline Random Forest -> R² = {r2:.3f}, RMSE = {rmse:.2f}%")

    # B. Classification Sanity Check
    rf_clf = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
    rf_clf.fit(X_train_enc, train_df["loss_risk"])
    clf_probs = rf_clf.predict_proba(X_test_enc)[:, 1]
    clf_preds = rf_clf.predict(X_test_enc)
    acc = accuracy_score(test_df["loss_risk"], clf_preds)
    auc = roc_auc_score(test_df["loss_risk"], clf_probs)
    f1 = f1_score(test_df["loss_risk"], clf_preds)
    print(f"  [Classification] Baseline Random Forest -> ROC-AUC = {auc:.3f}, Accuracy = {acc:.2%}, F1 = {f1:.3f}")

    # Save feature metadata
    metadata = {
        "dataset_shape": {"rows": ml_ready_df.shape[0], "columns": ml_ready_df.shape[1]},
        "target_variables": {
            "continuous_regression": "synthetic_loss_percentage",
            "binary_classification": "loss_risk"
        },
        "dropped_leakage_and_redundant_columns": columns_to_drop,
        "input_features": {
            "numeric": numeric_features,
            "categorical": categorical_features,
            "total_count": len(feature_cols)
        },
        "splits": {
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "stratification_column": "loss_risk"
        },
        "baseline_sanity_metrics": {
            "regression_r2": float(r2),
            "regression_rmse": float(rmse),
            "classification_roc_auc": float(auc),
            "classification_accuracy": float(acc),
            "classification_f1": float(f1)
        }
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved ML feature metadata to: {METADATA_PATH}")
    print("=" * 80)
    print("DATASET PREPARATION & AUDIT COMPLETE.")
    print("=" * 80)

if __name__ == "__main__":
    prepare_ml_dataset()
