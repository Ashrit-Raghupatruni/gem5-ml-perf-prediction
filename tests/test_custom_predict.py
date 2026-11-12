#!/usr/bin/env python3
"""
Unit and Regression Test Suite for Custom Input Predictions
Project: ML-Based Microarchitectural Performance Prediction (ARM gem5)
"""

import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import tempfile
import unittest
import numpy as np
import pandas as pd
import joblib

from ml.predict_custom import sanitize_and_fill_defaults, predict_custom_configurations, DEFAULT_MICROARCH


class TestCustomPredictPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Locate champion model
        cls.model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml", "saved_models", "best_model_ipc.joblib"))
        if os.path.exists(cls.model_path):
            cls.pipeline = joblib.load(cls.model_path)
        else:
            cls.pipeline = None

    def test_sanitize_and_fill_defaults(self):
        """
        Tests that omitting optional parameters fills them with physically valid microarchitectural defaults.
        """
        partial_df = pd.DataFrame([{
            'workload': 'matmul',
            'fetch_width': 8,
            'rob_entries': 256,
            'l1d_size_kb': 64,
            'l2_size_kb': 2048,
            'clk_freq_ghz': 3.0
        }])

        sanitized = sanitize_and_fill_defaults(partial_df)

        # Check required columns populated
        for col in DEFAULT_MICROARCH.keys():
            self.assertIn(col, sanitized.columns)

        # Check intelligent derived defaults
        self.assertEqual(sanitized.iloc[0]['decode_width'], 8)
        self.assertEqual(sanitized.iloc[0]['issue_width'], 4)
        self.assertEqual(sanitized.iloc[0]['commit_width'], 4)
        self.assertEqual(sanitized.iloc[0]['iq_entries'], 64)
        self.assertEqual(sanitized.iloc[0]['phys_int_regs'], int(256 * 0.8))

    def test_single_prediction_output_schema(self):
        """
        Tests single configuration inference produces valid IPC, uncertainty, CI, and cost proxies.
        """
        if self.pipeline is None:
            self.skipTest("Champion model joblib not found for integration test.")

        df_input = pd.DataFrame([{
            'workload': 'matmul',
            'fetch_width': 8,
            'issue_width': 8,
            'rob_entries': 256,
            'l1d_size_kb': 64,
            'l2_size_kb': 2048,
            'clk_freq_ghz': 3.0
        }])

        df_results, inference_ms = predict_custom_configurations(self.pipeline, df_input)

        self.assertEqual(len(df_results), 1)
        row = df_results.iloc[0]

        # Verify predicted values are non-null and within physically realistic bounds
        self.assertGreater(row['predicted_ipc'], 0.1)
        self.assertLess(row['predicted_ipc'], 10.0)
        self.assertGreaterEqual(row['uncertainty_std'], 0.0)
        self.assertLessEqual(row['ci95_lower'], row['predicted_ipc'])
        self.assertGreaterEqual(row['ci95_upper'], row['predicted_ipc'])
        self.assertGreater(row['area_cost_proxy'], 0.0)
        self.assertGreater(row['power_proxy'], 0.0)
        self.assertGreater(row['energy_proxy'], 0.0)
        self.assertGreater(inference_ms, 0.0)

    def test_batch_csv_prediction(self):
        """
        Tests batch CSV inference across multiple workloads and verifies export integrity.
        """
        if self.pipeline is None:
            self.skipTest("Champion model joblib not found for integration test.")

        batch_configs = [
            {'workload': 'matmul', 'fetch_width': 4, 'rob_entries': 128, 'clk_freq_ghz': 2.4},
            {'workload': 'stream', 'fetch_width': 8, 'rob_entries': 256, 'clk_freq_ghz': 3.0},
            {'workload': 'quicksort', 'fetch_width': 2, 'rob_entries': 64, 'clk_freq_ghz': 1.5},
            {'workload': 'hash', 'fetch_width': 4, 'rob_entries': 96, 'clk_freq_ghz': 2.0}
        ]
        df_batch = pd.DataFrame(batch_configs)

        df_results, inference_ms = predict_custom_configurations(self.pipeline, df_batch)

        self.assertEqual(len(df_results), 4)
        for _, row in df_results.iterrows():
            self.assertFalse(np.isnan(row['predicted_ipc']))
            self.assertFalse(np.isnan(row['uncertainty_std']))
            self.assertFalse(np.isnan(row['area_cost_proxy']))

        # Test CSV export roundtrip
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            df_results.to_csv(tmp_path, index=False)
            df_loaded = pd.read_csv(tmp_path)
            self.assertEqual(len(df_loaded), 4)
            self.assertIn('predicted_ipc', df_loaded.columns)
            self.assertIn('ci95_lower', df_loaded.columns)
            self.assertIn('area_cost_proxy', df_loaded.columns)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == '__main__':
    unittest.main()
