#!/usr/bin/env python3
"""
Automated End-to-End Regression & Pipeline Test Suite
Project: ML-Based Microarchitectural Performance Prediction (ARM gem5)
"""

import os
import sys
import json
import tempfile
import unittest
import numpy as np
import pandas as pd

from scripts.generate_design_space import generate_lhs_design_space, save_and_split_shards
from scripts.parse_dataset import parse_single_stats, harvest_dataset
from ml.predict_and_screen import generate_candidate_design_space, compute_cost_and_energy_proxies, find_pareto_frontier


class TestDesignSpaceGenerator(unittest.TestCase):
    def test_sample_counts_and_shards(self):
        configs = generate_lhs_design_space(num_samples=40, seed=42)
        self.assertEqual(len(configs), 40)
        
        # Check workload balance
        wls = [c['workload'] for c in configs]
        self.assertEqual(wls.count('matmul'), 10)
        self.assertEqual(wls.count('stream'), 10)
        self.assertEqual(wls.count('quicksort'), 10)
        self.assertEqual(wls.count('hash'), 10)
        
        # Test sharding with remainder
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_and_split_shards(configs, tmp_dir, num_shards=3)
            master_file = os.path.join(tmp_dir, "sweep_configs.json")
            self.assertTrue(os.path.exists(master_file))
            
            s1 = json.load(open(os.path.join(tmp_dir, "shards", "shard_1.json")))
            s2 = json.load(open(os.path.join(tmp_dir, "shards", "shard_2.json")))
            s3 = json.load(open(os.path.join(tmp_dir, "shards", "shard_3.json")))
            self.assertEqual(len(s1) + len(s2) + len(s3), 40)
            self.assertEqual(len(s1), 14)
            self.assertEqual(len(s2), 13)
            self.assertEqual(len(s3), 13)


class TestStatsParser(unittest.TestCase):
    def test_parse_valid_stats(self):
        with tempfile.NamedTemporaryFile('w', delete=False) as f:
            f.write("simInsts  1000000\n")
            f.write("board.processor.cores.core.numCycles  1250000\n")
            f.write("simSeconds  0.0005\n")
            f.write("board.processor.cores.core.ipc  0.8000\n")
            f.write("board.cache_hierarchy.l2_node.l1d-cache-0.cache.overallMisses::total  500\n")
            f.name_path = f.name
            
        try:
            stats, reason = parse_single_stats(f.name_path)
            self.assertIsNotNone(stats)
            self.assertEqual(reason, "OK")
            self.assertEqual(stats['sim_insts'], 1000000)
            self.assertEqual(stats['num_cycles'], 1250000)
            self.assertAlmostEqual(stats['ipc'], 0.8)
            self.assertAlmostEqual(stats['l1d_mpki'], 0.5)
        finally:
            if os.path.exists(f.name_path):
                os.remove(f.name_path)

    def test_parse_corrupt_stats(self):
        with tempfile.NamedTemporaryFile('w', delete=False) as f:
            f.write("simInsts  0\n")
            f.name_path = f.name
            
        try:
            stats, reason = parse_single_stats(f.name_path)
            self.assertIsNone(stats)
        finally:
            if os.path.exists(f.name_path):
                os.remove(f.name_path)


class TestCandidateScreeningAndPareto(unittest.TestCase):
    def test_candidate_generation_and_pareto(self):
        df_candidates = generate_candidate_design_space(num_candidates=100)
        self.assertEqual(len(df_candidates), 100)
        
        # Fake predicted IPC
        df_candidates['predicted_ipc'] = np.random.uniform(0.5, 2.0, size=len(df_candidates))
        area, power, energy = compute_cost_and_energy_proxies(df_candidates)
        df_candidates['hardware_cost_proxy'] = area
        
        pareto_df = find_pareto_frontier(df_candidates, cost_col='hardware_cost_proxy', perf_col='predicted_ipc')
        self.assertIn('is_pareto', pareto_df.columns)
        self.assertTrue(pareto_df['is_pareto'].sum() > 0)


if __name__ == '__main__':
    unittest.main()
