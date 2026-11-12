"""Comprehensive automated test suite for FastAPI backend endpoints."""
import os
import sys
import unittest
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend.schemas import SingleHardwareConfig
from backend.services import predict_single, predict_batch_df, get_pareto_reference

class TestBackendServices(unittest.TestCase):
    
    def test_single_predict_valid(self):
        payload = {
            "workload": "matmul",
            "fetch_width": 8,
            "issue_width": 8,
            "rob_entries": 256,
            "l1d_size_kb": 64,
            "l2_size_kb": 2048,
            "clk_freq_ghz": 3.0
        }
        res = predict_single(payload)
        self.assertIn("ipc", res)
        self.assertGreater(res["ipc"], 0.5)
        self.assertIn("uncertainty_sigma", res)
        self.assertIn("area_cost_proxy", res)
        self.assertIn("ratios", res)
        self.assertIn("rob_to_issue_ratio", res["ratios"])

    def test_single_predict_stream_memory_bound(self):
        payload = {
            "workload": "stream",
            "fetch_width": 4,
            "issue_width": 4,
            "rob_entries": 128,
            "clk_freq_ghz": 2.4
        }
        res = predict_single(payload)
        self.assertIn("ipc", res)
        # Stream has lower IPC due to memory latency
        self.assertLess(res["ipc"], 2.0)

    def test_batch_predict_csv(self):
        csv_path = os.path.join(REPO_ROOT, "examples", "custom_inputs_template.csv")
        df = pd.read_csv(csv_path)
        res = predict_batch_df(df)
        self.assertEqual(res["total_samples"], 9)
        self.assertGreaterEqual(len(res["pareto_frontier"]), 1)
        self.assertIn("mean_ipc", res["summary"])

    def test_pareto_reference(self):
        ref = get_pareto_reference()
        self.assertIsInstance(ref, list)

if __name__ == "__main__":
    unittest.main()
