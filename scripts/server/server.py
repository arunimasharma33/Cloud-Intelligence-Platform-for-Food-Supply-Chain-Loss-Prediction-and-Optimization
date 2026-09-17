#!/usr/bin/env python3
"""
Food Supply Chain Risk Platform — Flask REST API Backend
=========================================================
Wraps the existing inference engine as a local HTTP API consumed by
the web dashboard (templates/dashboard.html).

Endpoints:
  POST /api/predict          — Run full inference (+ SHAP explanation) on a shipment payload
  POST /api/predict/batch    — Run inference on an uploaded CSV of shipments
  POST /api/simulate/<id>    — Run one of 3 preset edge scenarios
  GET  /api/metrics          — Return model benchmark leaderboard data
  GET  /api/health           — System health & model status ping

Usage:
  python3 scripts/server/server.py
  (Dashboard opens automatically at http://localhost:5050)
"""

import os
import sys
import json
import webbrowser
import threading
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS

# Windows consoles default to a non-UTF-8 codepage, which crashes the
# decorative print() calls below (checkmarks, box-drawing borders).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Determine absolute project root (scripts/server/ → scripts/ → project root)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
os.chdir(PROJECT_ROOT)

INFERENCE_DIR = os.path.join(PROJECT_ROOT, "scripts", "inference")
if INFERENCE_DIR not in sys.path:
    sys.path.insert(0, INFERENCE_DIR)

# pyrefly: ignore [missing-import]
from inference_engine import FoodSupplyChainInferenceEngine

# ─── App Setup ────────────────────────────────────────────────────────────────
TEMPLATE_DIR = os.path.join(PROJECT_ROOT, "templates")
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=PROJECT_ROOT, static_url_path="")
CORS(app)

# ─── Load Engine Once at Startup ──────────────────────────────────────────────
print("  Loading champion ML pipelines...")
try:
    engine = FoodSupplyChainInferenceEngine()
    print("  ✓ Regressor and Classifier loaded successfully.")
except FileNotFoundError as e:
    print(f"  ✗ ERROR: {e}")
    engine = None

# ─── Load Static Assets ───────────────────────────────────────────────────────
METRICS_PATH = os.path.join(PROJECT_ROOT, "reports/model_evaluations/comprehensive_model_metrics.json")
SCENARIOS_PATH = os.path.join(PROJECT_ROOT, "reports/sample_shipments.json")

with open(METRICS_PATH) as f:
    MODEL_METRICS = json.load(f)

with open(SCENARIOS_PATH) as f:
    SAMPLE_SCENARIOS = {s["scenario_id"]: s for s in json.load(f)}

MAX_BATCH_ROWS = 2000  # safety cap for a single uploaded CSV

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def serve_dashboard():
    return render_template("dashboard.html")

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": engine is not None,
        "regressor": "champion_food_loss_regressor.joblib",
        "classifier": "champion_loss_risk_classifier.joblib",
        "dataset_scale": 30000,
        "feature_count": 22,
        "version": "2.5"
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    if engine is None:
        return jsonify({"error": "Inference engine not loaded. Check model paths."}), 503

    payload = request.get_json(force=True)
    if not payload:
        return jsonify({"error": "Empty or invalid JSON payload."}), 400

    # Explanation is on by default for single-shipment predictions (fast,
    # <100ms) but can be skipped with ?explain=false for lighter clients.
    explain = request.args.get("explain", "true").lower() != "false"

    try:
        result = engine.predict(payload, explain=explain)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500

@app.route("/api/predict/schema", methods=["GET"])
def predict_schema():
    """Describes the shipment payload shape accepted by /api/predict, so a
    client can build/validate a form without hardcoding the feature list."""
    if engine is None:
        return jsonify({"error": "Inference engine not loaded."}), 503
    return jsonify({
        "required_features": engine.feature_names,
        "mandatory_fields": ["commodity"],
        "notes": "All fields besides 'commodity' have sensible server-side defaults if omitted."
    })

@app.route("/api/predict/batch", methods=["POST"])
def predict_batch():
    """
    Accepts a multipart/form-data upload with a CSV file under the 'file'
    field. Each row is run through the same validation + inference path as
    a single /api/predict call. Explanations are NOT computed per-row here
    (they'd dominate runtime on large files) — batch mode trades per-row
    SHAP detail for throughput across many shipments at once.
    """
    if engine is None:
        return jsonify({"error": "Inference engine not loaded. Check model paths."}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Attach a CSV under the 'file' field."}), 400

    upload = request.files["file"]
    if not upload.filename:
        return jsonify({"error": "Empty filename."}), 400
    if not upload.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only .csv files are supported."}), 400

    try:
        df = pd.read_csv(upload)
    except Exception as e:
        return jsonify({"error": f"Could not parse CSV: {e}"}), 400

    if df.empty:
        return jsonify({"error": "Uploaded CSV has no rows."}), 400

    truncated = False
    if len(df) > MAX_BATCH_ROWS:
        df = df.head(MAX_BATCH_ROWS)
        truncated = True

    records = df.to_dict(orient="records")
    # Drop NaN keys row-by-row so optional/missing CSV cells don't get passed
    # through as literal NaN floats (which the engine's validators reject).
    cleaned_records = [
        {k: v for k, v in r.items() if pd.notna(v)} for r in records
    ]

    results = []
    errors = []
    for i, record in enumerate(cleaned_records):
        try:
            results.append(engine.predict(record))
        except Exception as e:
            errors.append({"row": i, "error": str(e), "input": record})

    high_risk_count = sum(1 for r in results if r["predicted_loss_risk"] == 1)
    avg_loss_pct = round(sum(r["predicted_loss_percentage"] for r in results) / len(results), 2) if results else 0.0

    return jsonify({
        "summary": {
            "rows_submitted": len(df),
            "rows_succeeded": len(results),
            "rows_failed": len(errors),
            "high_risk_count": high_risk_count,
            "average_predicted_loss_percentage": avg_loss_pct,
            "truncated_to_max_rows": truncated,
            "max_batch_rows": MAX_BATCH_ROWS,
        },
        "results": results,
        "errors": errors,
    })

@app.route("/api/scenarios", methods=["GET"])
def list_scenarios():
    """Lists preset simulation scenarios (id + description only) so a client
    can populate a scenario picker without hardcoding IDs from sample_shipments.json."""
    return jsonify([
        {"scenario_id": sid, "description": sc["description"]}
        for sid, sc in SAMPLE_SCENARIOS.items()
    ])

@app.route("/api/simulate/<scenario_id>", methods=["POST"])
def simulate(scenario_id):
    if engine is None:
        return jsonify({"error": "Inference engine not loaded."}), 503

    sc = SAMPLE_SCENARIOS.get(scenario_id)
    if not sc:
        return jsonify({"error": f"Unknown scenario ID: {scenario_id}"}), 404

    try:
        result = engine.predict(sc["payload"])
        result["scenario_description"] = sc["description"]
        result["scenario_id"] = sc["scenario_id"]
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/metrics", methods=["GET"])
def metrics():
    return jsonify(MODEL_METRICS)

# ─── Serve report images ───────────────────────────────────────────────────────
@app.route("/reports/<path:filename>")
def serve_report(filename):
    return send_from_directory(os.path.join(PROJECT_ROOT, "reports"), filename)

# ─── Entry Point ──────────────────────────────────────────────────────────────
def open_browser():
    webbrowser.open("http://localhost:5050")

if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  Food Supply Chain Risk Platform — API Server")
    print("═" * 60)
    print("  Dashboard → http://localhost:5050")
    print("  API Base  → http://localhost:5050/api")
    print("  Press Ctrl+C to stop.\n")
    threading.Timer(1.2, open_browser).start()
    app.run(host="0.0.0.0", port=5050, debug=False)
