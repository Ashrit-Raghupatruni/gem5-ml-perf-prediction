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

from scripts.generate_design_space import generate_lhs_design_space, save_configs, positive_int
from scripts.parse_dataset import parse_single_stats, audit_simulation_manifests
from ml.predict_and_screen import generate_candidate_design_space, compute_cost_and_energy_proxies, find_pareto_frontier
from ml.train_models import build_preprocessor, get_model_candidates, get_feature_definitions, engineer_microarchitectural_features
from ml.validate_and_evaluate import evaluate_optimizer_performance
from sklearn.pipeline import Pipeline


class TestDesignSpaceGenerator(unittest.TestCase):
    def test_arbitrary_sample_counts(self):
        """
        Tests user-selected sample counts (100, 180, 300, 501) and exact generation.
        """
        for sample_count in [100, 180, 300, 501]:
            configs = generate_lhs_design_space(num_samples=sample_count, seed=42, independent_widths=True)
            self.assertEqual(len(configs), sample_count, f"Failed exact generation for sample_count={sample_count}")
            
            # Verify no duplicate config_ids
            cids = [c['config_id'] for c in configs]
            self.assertEqual(len(cids), len(set(cids)), f"Duplicate config IDs detected for count={sample_count}")

    def test_workload_distribution_remainder_handling(self):
        """
        Tests fair remainder distribution across workloads when sample count is not divisible by 4.
        """
        # 501 % 4 = 1 remainder -> matmul should get 126, others 125
        configs = generate_lhs_design_space(num_samples=501, seed=42, independent_widths=True)
        wls = [c['workload'] for c in configs]
        self.assertEqual(wls.count('matmul'), 126)
        self.assertEqual(wls.count('stream'), 125)
        self.assertEqual(wls.count('quicksort'), 125)
        self.assertEqual(wls.count('hash'), 125)
        self.assertEqual(sum([wls.count(wl) for wl in ['matmul', 'stream', 'quicksort', 'hash']]), 501)

        # 102 % 4 = 2 remainder -> matmul 26, stream 26, quicksort 25, hash 25
        configs_102 = generate_lhs_design_space(num_samples=102, seed=42, independent_widths=True)
        wls_102 = [c['workload'] for c in configs_102]
        self.assertEqual(wls_102.count('matmul'), 26)
        self.assertEqual(wls_102.count('stream'), 26)
        self.assertEqual(wls_102.count('quicksort'), 25)
        self.assertEqual(wls_102.count('hash'), 25)

    def test_save_configs_and_boundaries(self):
        """
        Tests that generated configs save properly to a single master file with exact count and no duplicate IDs.
        """
        for total in [100, 300, 501]:
            configs = generate_lhs_design_space(num_samples=total, seed=42, independent_widths=True)
            with tempfile.TemporaryDirectory() as tmp_dir:
                cfg_path = save_configs(configs, tmp_dir, filename="sweep_configs.json")
                self.assertTrue(os.path.exists(cfg_path))
                
                with open(cfg_path) as f:
                    saved_configs = json.load(f)
                
                self.assertEqual(len(saved_configs), total)
                
                # Check uniqueness of IDs
                c_ids = [c['config_id'] for c in saved_configs]
                self.assertEqual(len(c_ids), len(set(c_ids)))

    def test_invalid_sample_count_inputs(self):
        """
        Tests rejection and clear error raising for negative, zero, or non-integer inputs.
        """
        with self.assertRaises(ValueError):
            generate_lhs_design_space(num_samples=0)
        with self.assertRaises(ValueError):
            generate_lhs_design_space(num_samples=-50)
        with self.assertRaises(ValueError):
            generate_lhs_design_space(num_samples=100.5)
        with self.assertRaises(ValueError):
            generate_lhs_design_space(num_samples="three_hundred")
            
        with self.assertRaises(Exception):
            positive_int("0")
        with self.assertRaises(Exception):
            positive_int("-10")
        with self.assertRaises(Exception):
            positive_int("abc")

    def test_monotonic_hardware_constraints(self):
        configs = generate_lhs_design_space(num_samples=100, seed=42, independent_widths=True)
        for c in configs:
            self.assertGreaterEqual(c['fetch_width'], c['decode_width'])
            self.assertGreaterEqual(c['decode_width'], c['issue_width'])
            self.assertGreaterEqual(c['commit_width'], c['issue_width'])
            self.assertGreaterEqual(c['rob_entries'], c['iq_entries'])

    def test_orthogonal_dimensions_no_correlations(self):
        configs = generate_lhs_design_space(num_samples=250, seed=42, independent_widths=True)
        df_cfg = pd.DataFrame(configs)
        corr_matrix = df_cfg[['l2_assoc', 'clk_freq_ghz', 'l1d_size_kb', 'l2_size_kb']].corr().abs()
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                val = corr_matrix.iloc[i, j]
                self.assertLess(val, 0.35, f"Unintended correlation {corr_matrix.index[i]} vs {corr_matrix.columns[j]}: {val}")


class TestStatsParserAndProvenance(unittest.TestCase):
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

    def test_provenance_audit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            m1 = {
                'environment_provenance': {
                    'hostname': 'host1',
                    'gem5_binary_sha256': 'abcdef123456',
                    'git_commit': 'deadbeef'
                }
            }
            with open(os.path.join(tmp_dir, "simulation_manifest.json"), "w") as f:
                json.dump(m1, f)
            audit = audit_simulation_manifests([tmp_dir])
            self.assertEqual(audit['total_manifests_found'], 1)
            self.assertTrue(audit['is_gem5_build_homogeneous'])


class TestOptimizerMetrics(unittest.TestCase):
    def test_missing_reference_dataset_handling(self):
        df_val = pd.DataFrame([{'simulated_ipc': 1.5}, {'simulated_ipc': 2.0}])
        res = evaluate_optimizer_performance(df_val, "non_existent_dataset.csv", percentile_threshold=0.90)
        self.assertIn('UNAVAILABLE', res['status'])
        self.assertIsNone(res['simple_regret'])
        self.assertIsNone(res['top_k_hit_rate_pct'])

    def test_evaluated_optimizer_metrics(self):
        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.csv') as f:
            df_ref = pd.DataFrame({'ipc': np.linspace(0.5, 3.0, 50)})
            df_ref.to_csv(f.name, index=False)
            f_path = f.name
            
        try:
            df_val = pd.DataFrame([{'simulated_ipc': 2.9}, {'simulated_ipc': 2.8}])
            res = evaluate_optimizer_performance(df_val, f_path, percentile_threshold=0.90)
            self.assertEqual(res['status'], 'EVALUATED')
            self.assertEqual(res['percentile_used'], 0.90)
            self.assertAlmostEqual(res['simple_regret'], 3.0 - 2.9, places=3)
            self.assertEqual(res['top_k_hit_rate_pct'], 100.0)
        finally:
            if os.path.exists(f_path):
                os.remove(f_path)


class TestCandidateScreeningAndPareto(unittest.TestCase):
    def test_candidate_generation_and_pareto(self):
        df_candidates = generate_candidate_design_space(num_candidates=100)
        self.assertEqual(len(df_candidates), 100)
        
        df_candidates['predicted_ipc'] = np.random.uniform(0.5, 2.0, size=len(df_candidates))
        area, power, energy = compute_cost_and_energy_proxies(df_candidates)
        df_candidates['hardware_cost_proxy'] = area
        
        pareto_df = find_pareto_frontier(df_candidates, cost_col='hardware_cost_proxy', perf_col='predicted_ipc')
        self.assertIn('is_pareto', pareto_df.columns)
        self.assertTrue(pareto_df['is_pareto'].sum() > 0)


class TestSyntheticPipelineIntegration(unittest.TestCase):
    
    def test_feature_engineering_monotonicity(self):
        """
        Tests that engineered microarchitectural ratios are non-negative, finite, and strictly conform to physical bounds.
        """
        df_candidates = generate_candidate_design_space(num_candidates=50)
        df_eng = engineer_microarchitectural_features(df_candidates)
        
        self.assertIn('pipe_issue_fetch_ratio', df_eng.columns)
        self.assertIn('rob_to_issue_ratio', df_eng.columns)
        self.assertIn('cache_l2_l1d_ratio', df_eng.columns)
        self.assertIn('total_l1_cache_kb', df_eng.columns)
        self.assertIn('peak_issue_bandwidth', df_eng.columns)
        
        # Check physical invariants
        self.assertTrue((df_eng['pipe_issue_fetch_ratio'] > 0).all())
        self.assertTrue((df_eng['pipe_issue_fetch_ratio'] <= 1.0).all())
        self.assertTrue((df_eng['rob_to_issue_ratio'] >= 8.0).all())
        self.assertTrue(np.isfinite(df_eng['cache_l2_l1d_ratio']).all())

    def test_all_four_models_present(self):
        """
        Tests that exactly the required model suite (KNN, RandomForest, HistGradientBoosting, and optional XGBoost)
        are instantiated with consistent interfaces.
        """
        models = get_model_candidates()
        self.assertIn('KNN', models)
        self.assertIn('RandomForest', models)
        self.assertIn('HistGradientBoosting', models)
        # Verify no unauthorized models (e.g. Ridge, SVR, etc.)
        allowed = {'KNN', 'RandomForest', 'HistGradientBoosting', 'XGBoost'}
        self.assertTrue(set(models.keys()).issubset(allowed))

    def test_synthetic_train_and_screen(self):
        cat_cols, num_cols = get_feature_definitions()
        all_features = cat_cols + num_cols
        
        n_samples = 40
        np.random.seed(42)
        synthetic_data = {
            'workload': np.random.choice(['matmul', 'stream', 'quicksort', 'hash'], n_samples),
            'branch_predictor': np.random.choice(['TournamentBP', 'BiModeBP', 'LocalBP'], n_samples),
            'fetch_width': np.random.choice([2, 4, 8], n_samples),
            'decode_width': np.random.choice([2, 4, 8], n_samples),
            'issue_width': np.random.choice([2, 4, 8], n_samples),
            'commit_width': np.random.choice([2, 4, 8], n_samples),
            'rob_entries': np.random.choice([64, 96, 128, 192], n_samples),
            'iq_entries': np.random.choice([32, 48, 64], n_samples),
            'lq_entries': np.random.choice([16, 24, 32], n_samples),
            'sq_entries': np.random.choice([16, 24, 32], n_samples),
            'phys_int_regs': np.random.choice([128, 160, 192], n_samples),
            'phys_fp_regs': np.random.choice([128, 160, 192], n_samples),
            'l1d_size_kb': np.random.choice([16, 32, 64], n_samples),
            'l1d_assoc': np.random.choice([2, 4, 8], n_samples),
            'l1i_size_kb': np.random.choice([16, 32], n_samples),
            'l1i_assoc': np.random.choice([2, 4], n_samples),
            'l2_size_kb': np.random.choice([256, 512], n_samples),
            'l2_assoc': np.random.choice([4, 8], n_samples),
            'clk_freq_ghz': np.random.choice([1.5, 2.0, 3.0], n_samples),
            'ipc': np.random.uniform(0.5, 3.0, n_samples)
        }
        df_syn = pd.DataFrame(synthetic_data)
        df_syn = engineer_microarchitectural_features(df_syn)
        
        candidates = get_model_candidates()
        self.assertIn('KNN', candidates)
        
        preprocessor = build_preprocessor()
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', candidates['KNN'])
        ])
        pipeline.fit(df_syn[all_features], df_syn['ipc'])
        
        preds = pipeline.predict(df_syn[all_features])
        self.assertEqual(len(preds), n_samples)
        self.assertTrue((preds > 0).all())


if __name__ == '__main__':
    unittest.main()
