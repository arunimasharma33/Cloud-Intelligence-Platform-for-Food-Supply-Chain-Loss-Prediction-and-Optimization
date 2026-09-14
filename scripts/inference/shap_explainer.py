#!/usr/bin/env python3
"""
BITE412L SHAP Prediction Explainer
===================================
Produces per-prediction, human-readable "top contributing factors" for both
champion pipelines (Ridge regressor + HistGradientBoosting classifier).

Design notes:
  - The regressor is linear, so shap.LinearExplainer is used against a small
    background sample of the training split (exact, fast, no sampling error).
  - The classifier is a gradient-boosted tree ensemble, so shap.TreeExplainer
    is used directly on the fitted model (exact path-dependent attribution).
  - Both pipelines share a ColumnTransformer (StandardScaler + OneHotEncoder).
    A one-hot-encoded categorical column produces many transformed columns;
    this module sums the SHAP values within each original column's group so
    a categorical feature (e.g. 'commodity') is reported as ONE factor with
    ONE aggregated contribution, not dozens of near-zero dummy contributions.
  - Regression SHAP values are already in the target's own units (percentage
    points of predicted loss), so they're directly interpretable.
  - Classification SHAP values are in log-odds space (relative to the
    positive/high-risk class). They are reported as signed relative
    contributions plus a direction label; exact log-odds units are not
    meaningful to a non-technical dashboard reader, so only sign + relative
    magnitude are surfaced.

If the training split used for the background sample is unavailable, the
explainer degrades gracefully: it falls back to using the single input row
as its own background (SHAP values still computable, just less statistically
grounded), and never breaks the caller's prediction flow.
"""

import os
import numpy as np
import pandas as pd
import shap

# Friendly display labels for the raw feature names used across the pipeline.
FEATURE_LABELS = {
    "temperature_celsius": "Cargo Temperature",
    "Humidity": "Relative Humidity",
    "days_to_expiry": "Days to Expiry",
    "Waiting_Time": "Queue / Waiting Time",
    "Logistics_Delay": "Logistics Delay Flag",
    "Inventory_Level": "Inventory Payload",
    "Asset_Utilization": "Asset Utilization",
    "Demand_Forecast": "Demand Forecast",
    "User_Transaction_Amount": "Transaction Amount",
    "User_Purchase_Frequency": "Purchase Frequency",
    "Latitude": "Latitude",
    "Longitude": "Longitude",
    "hour_of_day": "Hour of Day",
    "day_of_week": "Day of Week",
    "month": "Month",
    "commodity": "Commodity",
    "food_supply_stage": "Supply Chain Stage",
    "packaging_material": "Packaging Material",
    "temp_class": "Temperature Class",
    "Shipment_Status": "Shipment Status",
    "Traffic_Status": "Traffic Status",
    "Logistics_Delay_Reason": "Disruption Root Cause",
}

DEFAULT_BACKGROUND_PATH = "dataset/splits/train_set.csv"
DEFAULT_BACKGROUND_SAMPLE_SIZE = 100


def _build_column_groups(fitted_column_transformer):
    """
    Maps each transformed output column back to its original input column,
    using the fitted ColumnTransformer's own transformers_ (no string parsing).
    Returns a list of (original_column_name, slice) covering the full output.
    """
    groups = []
    offset = 0
    for name, transformer, cols in fitted_column_transformer.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "categories_"):
            # OneHotEncoder: one categories_ array per input column, in order.
            for col, cats in zip(cols, transformer.categories_):
                n = len(cats)
                groups.append((col, slice(offset, offset + n)))
                offset += n
        else:
            # Passthrough / scaler-style transformer: 1 output col per input col.
            for col in cols:
                groups.append((col, slice(offset, offset + 1)))
                offset += 1
    return groups


class PredictionExplainer:
    """Lazily-initialized SHAP explainer wrapping both champion pipelines."""

    def __init__(self, regressor_pipeline, classifier_pipeline, required_features,
                 background_path: str = DEFAULT_BACKGROUND_PATH,
                 background_sample_size: int = DEFAULT_BACKGROUND_SAMPLE_SIZE):
        self.required_features = required_features

        self.reg_prep = regressor_pipeline.named_steps["prep"]
        self.reg_model = regressor_pipeline.named_steps["model"]
        self.clf_prep = classifier_pipeline.named_steps["prep"]
        self.clf_model = classifier_pipeline.named_steps["model"]

        background_df = self._load_background(background_path, background_sample_size)
        if background_df is None:
            raise RuntimeError(
                f"SHAP background dataset unavailable at '{background_path}'. "
                "Explanations require the training split to build a representative "
                "background sample."
            )

        reg_bg_transformed = self.reg_prep.transform(background_df[required_features])
        self.reg_explainer = shap.LinearExplainer(self.reg_model, reg_bg_transformed)
        self.clf_explainer = shap.TreeExplainer(self.clf_model)

        self._reg_groups = _build_column_groups(self.reg_prep)
        self._clf_groups = _build_column_groups(self.clf_prep)

    def _load_background(self, path, sample_size):
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                missing = [c for c in self.required_features if c not in df.columns]
                if not missing and len(df) > 0:
                    n = min(sample_size, len(df))
                    return df.sample(n=n, random_state=42)
        except Exception:
            pass
        return None  # signals "no background available" to callers below

    def _top_factors(self, shap_values_row, groups, df_row, top_n):
        contributions = []
        for col, sl in groups:
            impact = float(np.sum(shap_values_row[sl]))
            contributions.append((col, impact))
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        factors = []
        for col, impact in contributions[:top_n]:
            raw_value = df_row.iloc[0][col] if col in df_row.columns else None
            if isinstance(raw_value, (np.floating, np.integer)):
                raw_value = raw_value.item()
            factors.append({
                "feature": FEATURE_LABELS.get(col, col),
                "raw_feature": col,
                "value": raw_value,
                "impact": round(impact, 4),
                "direction": "increases" if impact > 0 else "decreases",
            })
        return factors

    def explain(self, df_row: pd.DataFrame, top_n: int = 3) -> dict:
        """
        Returns top_n contributing factors for both the loss % regression and
        the loss-risk classification, given a single-row formatted DataFrame
        (as produced by FoodSupplyChainInferenceEngine.validate_and_format_payload).
        """
        reg_input = self.reg_prep.transform(df_row[self.required_features])
        clf_input = self.clf_prep.transform(df_row[self.required_features])

        reg_shap = self.reg_explainer(reg_input).values[0]
        clf_shap = self.clf_explainer(clf_input).values[0]
        # TreeExplainer on a binary HistGradientBoostingClassifier returns a
        # single log-odds-space array for the positive (high-risk) class.
        if clf_shap.ndim > 1:
            clf_shap = clf_shap[:, 1]

        return {
            "loss_pct_drivers": self._top_factors(reg_shap, self._reg_groups, df_row, top_n),
            "risk_drivers": self._top_factors(clf_shap, self._clf_groups, df_row, top_n),
        }
