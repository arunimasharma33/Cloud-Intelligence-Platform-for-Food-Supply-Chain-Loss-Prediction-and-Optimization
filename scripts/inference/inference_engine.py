#!/usr/bin/env python3
"""
BITE412L Local Inference and Decision Support System
===================================================
Loads the trained champion ML pipelines and provides real-time inference,
risk assessment, and actionable operational recommendations for refrigerated
and ambient food transport.

Features:
  - Validates and coerces all 22 operational telematics features.
  - Automatically derives temporal features if a 'Timestamp' string is supplied.
  - Automatically infers 'temp_class' from 'commodity' if omitted.
  - Generates continuous loss % prediction and binary risk classification probability.
  - Decision support rule engine generating tailored logistics interventions.
  - Formats output as an ANSI colored visual console card or structured JSON.
"""

import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

REGRESSOR_PATH = "models/champion_food_loss_regressor.joblib"
CLASSIFIER_PATH = "models/champion_loss_risk_classifier.joblib"

# The exact 22 features required by the champion pipelines
REQUIRED_FEATURES = [
    "Latitude", "Longitude", "Inventory_Level", "Shipment_Status",
    "Humidity", "Traffic_Status", "Waiting_Time", "User_Transaction_Amount",
    "User_Purchase_Frequency", "Logistics_Delay_Reason", "Asset_Utilization",
    "Demand_Forecast", "Logistics_Delay", "commodity", "food_supply_stage",
    "packaging_material", "days_to_expiry", "temp_class", "hour_of_day",
    "day_of_week", "month", "temperature_celsius"
]

FROZEN_KEYWORDS = ["frozen", "ice cream"]
COLD_KEYWORDS = ["dairy", "milk", "yogurt", "yoghurt", "cheese", "meat",
                 "egg", "fish", "poultry", "seafood", "butter"]

TEMP_BANDS = {
    "frozen": (-18.0, -12.0),
    "cold": (0.0, 8.0),
    "ambient": (10.0, 25.0)
}

def infer_temp_class(commodity: str) -> str:
    """Infers temperature class (frozen, cold, ambient) from commodity name."""
    c_lower = commodity.lower()
    for kw in FROZEN_KEYWORDS:
        if kw in c_lower:
            return "frozen"
    for kw in COLD_KEYWORDS:
        if kw in c_lower:
            return "cold"
    return "ambient"

class FoodSupplyChainInferenceEngine:
    def __init__(self, reg_path: str = REGRESSOR_PATH, clf_path: str = CLASSIFIER_PATH):
        if not os.path.exists(reg_path):
            raise FileNotFoundError(f"Champion Regressor not found at: {reg_path}")
        if not os.path.exists(clf_path):
            raise FileNotFoundError(f"Champion Classifier not found at: {clf_path}")

        self.regressor = joblib.load(reg_path)
        self.classifier = joblib.load(clf_path)
        self.feature_names = REQUIRED_FEATURES
        self._explainer = None          # lazily built on first explained predict()
        self._explainer_init_failed = False

    def _get_explainer(self):
        """Lazily builds the SHAP explainer on first use so CLI/batch calls
        that never request an explanation pay zero extra startup cost."""
        if self._explainer is not None or self._explainer_init_failed:
            return self._explainer
        try:
            from shap_explainer import PredictionExplainer
            self._explainer = PredictionExplainer(
                self.regressor, self.classifier, self.feature_names
            )
        except Exception as e:
            print(f"  [warn] SHAP explainability unavailable: {e}", file=sys.stderr)
            self._explainer_init_failed = True
        return self._explainer

    def validate_and_format_payload(self, raw_payload: Dict[str, Any]) -> pd.DataFrame:
        """Validates incoming dictionary and produces standardized 22-feature DataFrame."""
        payload = dict(raw_payload)

        # 1. Coordinate / Sensor Alias Harmonization
        if "simulated_temperature" in payload and "temperature_celsius" not in payload:
            payload["temperature_celsius"] = payload.pop("simulated_temperature")
        if "Temperature" in payload and "temperature_celsius" not in payload:
            payload["temperature_celsius"] = payload.pop("Temperature")

        # 2. Derive temporal features if Timestamp string is supplied
        if "Timestamp" in payload:
            try:
                dt = pd.to_datetime(payload["Timestamp"])
                payload.setdefault("hour_of_day", dt.hour)
                payload.setdefault("day_of_week", dt.dayofweek)
                payload.setdefault("month", dt.month)
            except Exception:
                pass

        # Defaults for temporal features if not provided
        payload.setdefault("hour_of_day", 12)
        payload.setdefault("day_of_week", 2)
        payload.setdefault("month", 6)

        # 3. Commodity & Temp Class
        if "commodity" not in payload:
            raise ValueError("Payload missing mandatory field: 'commodity'")
        
        if "temp_class" not in payload or payload["temp_class"] not in TEMP_BANDS:
            payload["temp_class"] = infer_temp_class(payload["commodity"])

        # 4. Defaults for operational metadata if missing
        payload.setdefault("Shipment_Status", "In Transit")
        payload.setdefault("Traffic_Status", "Clear")
        payload.setdefault("Logistics_Delay_Reason", "Unknown")
        payload.setdefault("food_supply_stage", "Transport")
        payload.setdefault("packaging_material", "cardboard")
        payload.setdefault("Inventory_Level", 350.0)
        payload.setdefault("Asset_Utilization", 80.0)
        payload.setdefault("Demand_Forecast", 200.0)
        payload.setdefault("User_Transaction_Amount", 200.0)
        payload.setdefault("User_Purchase_Frequency", 5.0)
        payload.setdefault("Waiting_Time", 15.0)
        payload.setdefault("Logistics_Delay", 0.0)
        payload.setdefault("days_to_expiry", 15)
        payload.setdefault("Humidity", 55.0)
        payload.setdefault("Latitude", 25.0)
        payload.setdefault("Longitude", 45.0)

        if "temperature_celsius" not in payload:
            # Fallback to mid of safe band
            band = TEMP_BANDS[payload["temp_class"]]
            payload["temperature_celsius"] = (band[0] + band[1]) / 2.0

        # Physical boundary checks and clipping
        payload["Latitude"] = float(np.clip(payload["Latitude"], -90.0, 90.0))
        payload["Longitude"] = float(np.clip(payload["Longitude"], -180.0, 180.0))
        payload["Humidity"] = float(np.clip(payload["Humidity"], 0.0, 100.0))
        payload["days_to_expiry"] = int(np.clip(payload["days_to_expiry"], 1, 60))
        payload["temperature_celsius"] = float(payload["temperature_celsius"])
        payload["Waiting_Time"] = float(max(0.0, payload["Waiting_Time"]))

        # Build DataFrame with exact feature order
        df = pd.DataFrame([payload])
        missing = [col for col in self.feature_names if col not in df.columns]
        if missing:
            raise ValueError(f"Payload could not be formatted. Missing features: {missing}")

        return df[self.feature_names]

    def predict(self, raw_payload: Dict[str, Any], explain: bool = False, explain_top_n: int = 3) -> Dict[str, Any]:
        """Runs validation, models inference, and operational action recommendation.

        If explain=True, also attaches a 'explanation' key with the top_n
        SHAP-derived contributing factors behind the loss % and risk predictions.
        """
        df = self.validate_and_format_payload(raw_payload)

        # 1. Regression Prediction
        loss_pred = float(self.regressor.predict(df)[0])
        loss_pred = float(np.clip(loss_pred, 0.0, 100.0))

        # 2. Classification Prediction
        risk_pred = int(self.classifier.predict(df)[0])
        if hasattr(self.classifier, "predict_proba"):
            risk_prob = float(self.classifier.predict_proba(df)[0, 1])
        else:
            risk_prob = float(risk_pred)

        # 3. Operational Rule Assessment
        recommendations = self._generate_recommendations(df.iloc[0], loss_pred, risk_pred, risk_prob)

        result = {
            "commodity": str(df.iloc[0]["commodity"]),
            "temp_class": str(df.iloc[0]["temp_class"]),
            "temperature_celsius": float(df.iloc[0]["temperature_celsius"]),
            "humidity": float(df.iloc[0]["Humidity"]),
            "days_to_expiry": int(df.iloc[0]["days_to_expiry"]),
            "waiting_time_min": float(df.iloc[0]["Waiting_Time"]),
            "shipment_status": str(df.iloc[0]["Shipment_Status"]),
            "predicted_loss_percentage": round(loss_pred, 2),
            "predicted_loss_risk": risk_pred,
            "risk_probability_percentage": round(risk_prob * 100.0, 1),
            "urgency_level": recommendations["urgency"],
            "primary_stress_factors": recommendations["stress_factors"],
            "recommended_action": recommendations["action"],
            "action_code": recommendations["action_code"]
        }

        if explain:
            explainer = self._get_explainer()
            if explainer is not None:
                try:
                    result["explanation"] = explainer.explain(df, top_n=explain_top_n)
                except Exception as e:
                    result["explanation"] = {"error": f"Explanation unavailable: {e}"}
            else:
                result["explanation"] = {"error": "Explanation unavailable: SHAP explainer failed to initialize."}

        return result

    def _generate_recommendations(self, row: pd.Series, loss_pred: float, risk_pred: int, risk_prob: float) -> Dict[str, Any]:
        """Maps model predictions and physical sensor states to actionable logistics operations."""
        temp = row["temperature_celsius"]
        temp_class = row["temp_class"]
        safe_band = TEMP_BANDS.get(temp_class, (10.0, 25.0))
        hum = row["Humidity"]
        expiry = row["days_to_expiry"]
        waiting_time = row["Waiting_Time"]

        stress_factors = []
        # Check Thermal Excursion
        if temp < safe_band[0] or temp > safe_band[1]:
            diff = temp - safe_band[1] if temp > safe_band[1] else safe_band[0] - temp
            stress_factors.append(f"Temperature Excursion ({temp:+.1f}°C vs safe band {safe_band[0]} to {safe_band[1]}°C, breach: {diff:+.1f}°C)")

        # Check Humidity Stress
        if hum > 75.0:
            stress_factors.append(f"High Relative Humidity ({hum:.1f}% > 75% fungal risk)")
        elif hum < 35.0:
            stress_factors.append(f"Low Relative Humidity ({hum:.1f}% < 35% dehydration risk)")

        # Check Shelf-Life
        if expiry <= 3:
            stress_factors.append(f"Critical Expiry Pressure ({expiry} day(s) remaining)")
        elif expiry <= 7:
            stress_factors.append(f"Approaching Expiry ({expiry} days remaining)")

        # Check Transit Delays
        if waiting_time >= 45.0:
            stress_factors.append(f"Severe Transit Queue Delay ({waiting_time:.1f} minutes)")

        has_cold_chain_breach = (temp_class in ["cold", "frozen"]) and (temp < safe_band[0] or temp > safe_band[1])

        # Decision Logic
        if (risk_pred == 1 and (has_cold_chain_breach or loss_pred >= 18.0 or risk_prob >= 0.70)) or (has_cold_chain_breach and risk_prob >= 0.50):
            urgency = "CRITICAL EMERGENCY"
            action_code = "ACT_CRITICAL_REROUTE"
            if has_cold_chain_breach:
                action = (f"CRITICAL COLD-CHAIN BREACH ({temp:+.1f}°C vs safe band {safe_band[0]} to {safe_band[1]}°C): "
                          "Active refrigeration failure detected. Immediately reroute shipment to the nearest cold-storage "
                          "logistics depot. Trigger driver emergency cooling override.")
            else:
                action = ("CRITICAL SPOILAGE RISK: Accelerated degradation imminent. "
                          "Expedite priority docking at next distribution hub. Authorize emergency unloading.")
        elif risk_pred == 1 or risk_prob >= 0.50:
            urgency = "HIGH RISK WARNING"
            action_code = "ACT_PRIORITY_DISPATCH"
            if expiry <= 5:
                action = ("SHELF-LIFE COMPROMISED: Prioritize shipment for immediate First-In-First-Out (FIFO) "
                          "retail display or secondary flash markdown.")
            else:
                action = ("ELEVATED SPOILAGE RISK: Inspect cargo bay seals and ventilation. "
                          "Avoid intermediate rest-stop delays to preserve product integrity.")
        elif loss_pred > 10.0:
            urgency = "MODERATE ADVISORY"
            action_code = "ACT_MONITOR_TELEMETRY"
            action = ("TRANSIT ADVISORY: Cargo within acceptable parameters but nearing threshold limits. "
                      "Maintain live telemetry monitoring at 5-minute sampling frequency.")
        else:
            urgency = "OPTIMAL"
            action_code = "ACT_PROCEED_SCHEDULED"
            action = ("CARGO OPTIMAL: All environmental and logistics conditions are safe. "
                      "Proceed along scheduled distribution route as planned.")

        return {
            "urgency": urgency,
            "action_code": action_code,
            "action": action,
            "stress_factors": stress_factors if stress_factors else ["None detected (All parameters nominal)"]
        }

def print_console_card(result: Dict[str, Any]):
    """Renders a beautiful visual terminal card displaying prediction & operational actions."""
    urgency = result["urgency_level"]
    if "CRITICAL" in urgency:
        color = "\033[91m"  # Red
        tag_bg = "\033[41m\033[97m"
    elif "HIGH" in urgency:
        color = "\033[93m"  # Yellow
        tag_bg = "\033[43m\033[30m"
    elif "MODERATE" in urgency:
        color = "\033[94m"  # Blue
        tag_bg = "\033[44m\033[97m"
    else:
        color = "\033[92m"  # Green
        tag_bg = "\033[42m\033[97m"
    reset = "\033[0m"
    bold = "\033[1m"

    print("\n" + color + "╔" + "═" * 78 + "╗" + reset)
    title = f" BITE412L CLOUD INTELLIGENCE: REAL-TIME SHIPMENT INFERENCE CARD "
    print(color + f"║ {bold}{title:<76}{reset}{color} ║" + reset)
    print(color + "╠" + "═" * 78 + "╣" + reset)

    # Commodity & Sensor Conditions
    c_info = f"Commodity: {result['commodity']} ({result['temp_class'].upper()} Chain)"
    t_info = f"Temp: {result['temperature_celsius']:+.1f}°C | Humidity: {result['humidity']:.1f}% | Expiry: {result['days_to_expiry']} days"
    print(f"║ {bold}{c_info:<76}{reset} ║")
    print(f"║ {t_info:<76} ║")
    print(color + "╟" + "─" * 78 + "╢" + reset)

    # Model Predictions
    risk_label = "HIGH RISK (ALERT)" if result["predicted_loss_risk"] == 1 else "LOW / NORMAL RISK"
    prob_str = f"Loss Risk Probability: {result['risk_probability_percentage']}%  [{risk_label}]"
    loss_str = f"Estimated Food Loss:   {result['predicted_loss_percentage']}%"
    urgency_badge = f"STATUS: [{result['urgency_level']}]"

    print(f"║ {bold}{loss_str:<76}{reset} ║")
    print(f"║ {bold}{prob_str:<76}{reset} ║")
    print(f"║ {tag_bg} {urgency_badge} {reset}{' ' * (75 - len(urgency_badge))} ║")
    print(color + "╟" + "─" * 78 + "╢" + reset)

    # Stress Factors
    print(f"║ {bold}Primary Environmental Stress Factors:{reset}{' ' * 39} ║")
    for factor in result["primary_stress_factors"]:
        f_line = f"  • {factor}"
        print(f"║ {f_line:<76} ║")

    print(color + "╟" + "─" * 78 + "╢" + reset)
    # Action Recommendation
    print(f"║ {bold}Recommended Operational Logistics Action:{reset}{' ' * 35} ║")
    words = result["recommended_action"].split()
    line = "  "
    for w in words:
        if len(line) + len(w) + 1 > 74:
            print(f"║ {color}{line:<76}{reset} ║")
            line = "  " + w
        else:
            line += (" " if line != "  " else "") + w
    if line:
        print(f"║ {color}{line:<76}{reset} ║")

    print(color + "╚" + "═" * 78 + "╝" + reset + "\n")

def run_demo():
    """Runs inference across 3 real-world edge scenarios."""
    print("=" * 80)
    print("BITE412L LOCAL INFERENCE DEMO — 3 REALISTIC EDGE TELEMETRY SCENARIOS")
    print("=" * 80)

    engine = FoodSupplyChainInferenceEngine()

    scenarios = [
        {
            "name": "Scenario 1: Optimal Ambient Cereal Transport (Maize)",
            "payload": {
                "commodity": "Maize (corn)",
                "food_supply_stage": "Transport",
                "packaging_material": "cardboard",
                "days_to_expiry": 28,
                "temperature_celsius": 18.5,
                "Humidity": 54.0,
                "Waiting_Time": 12.0,
                "Logistics_Delay": 0.0,
                "Logistics_Delay_Reason": "Unknown",
                "Shipment_Status": "In Transit",
                "Traffic_Status": "Clear",
                "hour_of_day": 14,
                "day_of_week": 2,
                "month": 8
            }
        },
        {
            "name": "Scenario 2: Critical Cold-Chain Excursion Breach (Fresh Milk)",
            "payload": {
                "commodity": "Milk (fresh whole dairy)",
                "food_supply_stage": "Transport",
                "packaging_material": "plastic",
                "days_to_expiry": 6,
                "temperature_celsius": 17.2,  # Safe is 0 to 8°C! Severe breach!
                "Humidity": 78.5,
                "Waiting_Time": 58.0,
                "Logistics_Delay": 1.0,
                "Logistics_Delay_Reason": "Mechanical Failure",
                "Shipment_Status": "Delayed",
                "Traffic_Status": "Heavy",
                "hour_of_day": 15,
                "day_of_week": 4,
                "month": 7
            }
        },
        {
            "name": "Scenario 3: Perishable Soft Produce Under Shelf-Life Pressure (Apricots)",
            "payload": {
                "commodity": "Apricots",
                "food_supply_stage": "Retail",
                "packaging_material": "cardboard",
                "days_to_expiry": 2,  # Only 2 days left!
                "temperature_celsius": 26.5,
                "Humidity": 68.0,
                "Waiting_Time": 42.0,
                "Logistics_Delay": 0.8,
                "Logistics_Delay_Reason": "Traffic",
                "Shipment_Status": "In Transit",
                "Traffic_Status": "Heavy",
                "hour_of_day": 17,
                "day_of_week": 5,
                "month": 6
            }
        }
    ]

    for sc in scenarios:
        print(f"\n>>> Running: {sc['name']}")
        result = engine.predict(sc["payload"])
        print_console_card(result)

def main():
    parser = argparse.ArgumentParser(description="BITE412L Real-Time Supply Chain Food Loss Inference Engine")
    parser.add_argument("--demo", action="store_true", help="Run 3 real-world preset test scenarios")
    parser.add_argument("--json", type=str, help="Input telemetry JSON string")
    parser.add_argument("--file", type=str, help="Path to input JSON file containing shipment record")
    parser.add_argument("--batch", type=str, help="Path to CSV or JSON file containing batch shipments")
    parser.add_argument("--output", type=str, help="Optional output JSON path to save inference results")
    parser.add_argument("--json-output", action="store_true", help="Output raw JSON instead of colored console card")
    parser.add_argument("--explain", action="store_true", help="Attach SHAP top-factor explanation to the result")

    args = parser.parse_args()

    if args.demo or len(sys.argv) == 1:
        run_demo()
        return

    engine = FoodSupplyChainInferenceEngine()

    if args.json:
        payload = json.loads(args.json)
        result = engine.predict(payload, explain=args.explain)
        if args.json_output:
            print(json.dumps(result, indent=2))
        else:
            print_console_card(result)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)

    elif args.file:
        with open(args.file) as f:
            payload = json.load(f)
        result = engine.predict(payload, explain=args.explain)
        if args.json_output:
            print(json.dumps(result, indent=2))
        else:
            print_console_card(result)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)

    elif args.batch:
        if args.batch.endswith(".csv"):
            df = pd.read_csv(args.batch)
            records = df.to_dict(orient="records")
        else:
            with open(args.batch) as f:
                records = json.load(f)

        results = [engine.predict(r) for r in records]
        out_df = pd.DataFrame(results)
        print(f"\nBatch Inference Complete for {len(results):,} shipments:")
        print(out_df[["commodity", "temperature_celsius", "predicted_loss_percentage", "risk_probability_percentage", "urgency_level"]].to_string(index=False))

        if args.output:
            out_df.to_json(args.output, orient="records", indent=2)
            print(f"Saved batch results to: {args.output}")

if __name__ == "__main__":
    main()
