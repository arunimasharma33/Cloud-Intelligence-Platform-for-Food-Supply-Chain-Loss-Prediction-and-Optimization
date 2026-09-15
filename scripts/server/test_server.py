#!/usr/bin/env python3
"""
Unit and Integration Test Suite for the Flask REST API Server
===============================================================
Verifies:
  1. Health and metrics endpoints respond with expected shape
  2. Scenario discovery lists all presets from sample_shipments.json
  3. Single-shipment prediction succeeds for a valid payload
  4. Invalid/empty prediction payloads are rejected with the right status
  5. Simulate endpoint runs known scenarios and rejects unknown ones
  6. Batch CSV prediction succeeds and reports per-row summary stats
  7. Predict schema endpoint describes the accepted feature set
"""

import io
import json
import unittest

import server as server_module


class TestServerAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        server_module.app.testing = True
        cls.client = server_module.app.test_client()

    def test_01_health(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["models_loaded"])
        self.assertIn("regressor", data)
        self.assertIn("classifier", data)

    def test_02_metrics(self):
        resp = self.client.get("/api/metrics")
        self.assertEqual(resp.status_code, 200)
        self.assertIsInstance(resp.get_json(), dict)

    def test_03_list_scenarios(self):
        resp = self.client.get("/api/scenarios")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(len(data), 3)
        ids = {s["scenario_id"] for s in data}
        self.assertIn("SCENARIO_01_OPTIMAL_AMBIENT", ids)
        for s in data:
            self.assertIn("description", s)

    def test_04_predict_valid_payload(self):
        resp = self.client.post("/api/predict", json={
            "commodity": "Maize (corn)",
            "temperature_celsius": 18.0,
            "days_to_expiry": 28,
            "Humidity": 50.0
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("predicted_loss_percentage", data)
        self.assertIn("predicted_loss_risk", data)
        self.assertIn("urgency_level", data)

    def test_05_predict_empty_payload_rejected(self):
        resp = self.client.post(
            "/api/predict",
            data="",
            content_type="application/json"
        )
        self.assertEqual(resp.status_code, 400)

    def test_06_predict_missing_commodity_rejected(self):
        resp = self.client.post("/api/predict", json={"temperature_celsius": 10.0})
        self.assertEqual(resp.status_code, 422)

    def test_07_simulate_known_scenario(self):
        resp = self.client.post("/api/simulate/SCENARIO_02_COLD_CHAIN_BREACH")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["scenario_id"], "SCENARIO_02_COLD_CHAIN_BREACH")
        self.assertIn("scenario_description", data)
        self.assertIn(data["urgency_level"], [
            "OPTIMAL", "MODERATE ADVISORY", "HIGH RISK WARNING", "CRITICAL EMERGENCY"
        ])

    def test_08_simulate_unknown_scenario_404(self):
        resp = self.client.post("/api/simulate/NOT_A_REAL_SCENARIO")
        self.assertEqual(resp.status_code, 404)

    def test_09_predict_batch_csv(self):
        csv_bytes = (
            b"commodity,temperature_celsius,days_to_expiry,Humidity\n"
            b"Maize (corn),18.0,28,50.0\n"
            b"Milk (fresh whole dairy),17.2,6,78.5\n"
        )
        resp = self.client.post(
            "/api/predict/batch",
            data={"file": (io.BytesIO(csv_bytes), "shipments.csv")},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["summary"]["rows_submitted"], 2)
        self.assertEqual(data["summary"]["rows_succeeded"], 2)
        self.assertEqual(len(data["results"]), 2)

    def test_10_predict_batch_rejects_non_csv(self):
        resp = self.client.post(
            "/api/predict/batch",
            data={"file": (io.BytesIO(b"not a csv"), "shipments.txt")},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 400)

    def test_11_predict_schema(self):
        resp = self.client.get("/api/predict/schema")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(len(data["required_features"]), 22)
        self.assertIn("commodity", data["mandatory_fields"])


if __name__ == "__main__":
    unittest.main()
