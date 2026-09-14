#!/usr/bin/env python3
"""
Unit and Integration Test Suite for BITE412L Inference Engine
============================================================
Verifies:
  1. Engine initialization & model loading
  2. Input validation, coordinate clipping & fallback feature derivation
  3. Prediction shape & valid numerical ranges
  4. Operational decision recommendation logic across scenarios
  5. JSON output consistency
"""

import os
import json
import unittest
import numpy as np
from inference_engine import FoodSupplyChainInferenceEngine, REQUIRED_FEATURES

class TestInferenceEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = FoodSupplyChainInferenceEngine()

    def test_01_engine_initialization(self):
        """Confirm models and features loaded successfully."""
        self.assertIsNotNone(self.engine.regressor)
        self.assertIsNotNone(self.engine.classifier)
        self.assertEqual(len(self.engine.feature_names), 22)

    def test_02_feature_formatting_and_derivation(self):
        """Test timestamp parsing and temp_class automatic inference."""
        raw_input = {
            "commodity": "Milk (whole pasteurized)",
            "temperature_celsius": 4.5,
            "Timestamp": "2024-05-18 10:30:00",
            "days_to_expiry": 10
        }
        df = self.engine.validate_and_format_payload(raw_input)
        self.assertEqual(len(df.columns), 22)
        self.assertEqual(df["temp_class"].iloc[0], "cold")
        self.assertEqual(df["hour_of_day"].iloc[0], 10)
        self.assertEqual(df["month"].iloc[0], 5)

    def test_03_coordinate_clipping(self):
        """Test that out-of-bound coordinates are clipped properly."""
        raw_input = {
            "commodity": "Wheat",
            "Latitude": 120.0,   # Out of bound > 90
            "Longitude": -200.0, # Out of bound < -180
            "temperature_celsius": 20.0
        }
        df = self.engine.validate_and_format_payload(raw_input)
        self.assertEqual(df["Latitude"].iloc[0], 90.0)
        self.assertEqual(df["Longitude"].iloc[0], -180.0)

    def test_04_optimal_scenario_prediction(self):
        """Test optimal ambient cereal shipment gives low loss and normal status."""
        payload = {
            "commodity": "Maize (corn)",
            "temperature_celsius": 18.0,
            "days_to_expiry": 28,
            "Humidity": 50.0,
            "Waiting_Time": 10.0,
            "Logistics_Delay": 0.0,
            "Shipment_Status": "In Transit"
        }
        res = self.engine.predict(payload)
        self.assertIn("predicted_loss_percentage", res)
        self.assertIn("predicted_loss_risk", res)
        self.assertIn("risk_probability_percentage", res)
        self.assertIn("urgency_level", res)
        self.assertEqual(res["predicted_loss_risk"], 0)
        self.assertEqual(res["urgency_level"], "OPTIMAL")

    def test_05_cold_chain_breach_detection(self):
        """Test that extreme thermal excursion on dairy triggers critical alert."""
        payload = {
            "commodity": "Milk (fresh whole dairy)",
            "temperature_celsius": 17.5, # Safe is 0 to 8°C
            "days_to_expiry": 5,
            "Humidity": 78.0,
            "Waiting_Time": 60.0,
            "Logistics_Delay": 1.0,
            "Shipment_Status": "Delayed"
        }
        res = self.engine.predict(payload)
        self.assertEqual(res["predicted_loss_risk"], 1)
        self.assertIn("CRITICAL", res["urgency_level"])
        self.assertIn("ACT_CRITICAL_REROUTE", res["action_code"])

    def test_06_sample_shipments_file(self):
        """Test batch loading and running all sample shipments from file."""
        sample_path = "reports/sample_shipments.json"
        self.assertTrue(os.path.exists(sample_path))
        with open(sample_path) as f:
            samples = json.load(f)
        self.assertEqual(len(samples), 3)

        for s in samples:
            res = self.engine.predict(s["payload"])
            self.assertGreaterEqual(res["predicted_loss_percentage"], 0.0)
            self.assertLessEqual(res["predicted_loss_percentage"], 100.0)
            self.assertIn(res["predicted_loss_risk"], [0, 1])

if __name__ == "__main__":
    unittest.main()
