#!/usr/bin/env python3
"""
Active Learning & Uncertainty-Driven Adaptive Sampler
Project: ML-Based Microarchitectural Performance Prediction (ARM gem5)
Strictly adheres to: documents/schema.md, documents/rules.md
"""

import os
import sys

# Ensure repository root is in sys.path when executed directly as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import argparse
import numpy as np
import pandas as pd
import joblib

try:
    from ml.predict_and_screen import generate_candidate_design_space, predict_with_uncertainty
except ModuleNotFoundError:
    from predict_and_screen import generate_candidate_design_space, predict_with_uncertainty


def select_active_learning_queries(model_path, pool_size=2000, n_queries=15, strategy='uncertainty', output_json="configs/active_queries.json"):
    """
    Selects the most informative microarchitectural configurations from an unlabeled candidate pool
    to run through gem5 simulation, maximizing model improvement with minimal simulation budget.
    
    Strategies:
    - 'uncertainty': Maximize predictive variance (std dev across ensemble trees)
    - 'ucb': Upper Confidence Bound (exploration + exploitation: mu + 1.5 * sigma)
    - 'random': Baseline random sampling
    """
    if not os.path.exists(model_path):
        print(f"[ERROR] Trained model pipeline not found: {model_path}")
        sys.exit(1)
        
    print(f"[INFO] Loading surrogate ML pipeline from: {model_path}")
    pipeline = joblib.load(model_path)
    
    print(f"[INFO] Generating unlabeled candidate pool of {pool_size} microarchitectural designs...")
    df_pool = generate_candidate_design_space(num_candidates=pool_size)
    
    print(f"[INFO] Computing predictions and uncertainty for candidate pool...")
    mean_preds, std_preds = predict_with_uncertainty(pipeline, df_pool)
    df_pool['predicted_ipc'] = mean_preds
    df_pool['uncertainty_std'] = std_preds
    df_pool['ucb'] = mean_preds + (1.5 * std_preds)
    
    if strategy == 'uncertainty':
        print(f"[INFO] Selecting top {n_queries} designs with highest predictive uncertainty (variance)...")
        selected_df = df_pool.sort_values(by='uncertainty_std', ascending=False).head(n_queries).copy()
    elif strategy == 'ucb':
        print(f"[INFO] Selecting top {n_queries} designs maximizing Upper Confidence Bound (UCB)...")
        selected_df = df_pool.sort_values(by='ucb', ascending=False).head(n_queries).copy()
    else:
        print(f"[INFO] Selecting {n_queries} random designs...")
        selected_df = df_pool.sample(n=n_queries, random_state=42).copy()
        
    queries = []
    print("\n" + "=" * 115)
    print(f"      ACTIVE LEARNING QUERIES FOR ADAPTIVE GEM5 SAMPLING (Strategy: {strategy.upper()})")
    print("=" * 115)
    print(f"{'Query ID':<10} | {'Workload':<10} | {'W':<3} | {'ROB':<5} | {'L1D':<5} | {'L2':<5} | {'Freq':<7} | {'Pred IPC':<10} | {'Uncertainty (Std)':<18}")
    print("-" * 115)
    
    for idx, (_, row) in enumerate(selected_df.iterrows(), 1):
        qid = f"AL_Q{idx:02d}"
        print(f"{qid:<10} | {row['workload']:<10} | {row['fetch_width']:<3} | {row['rob_entries']:<5} | {row['l1d_size_kb']:<5} | {row['l2_size_kb']:<5} | {row['clk_freq_ghz']:<5.1f}GHz | {row['predicted_ipc']:<10.4f} | +/-{row['uncertainty_std']:<16.4f}")
        
        queries.append({
            'config_id': qid,
            'workload': row['workload'],
            'fetch_width': int(row['fetch_width']),
            'decode_width': int(row['decode_width']),
            'issue_width': int(row['issue_width']),
            'commit_width': int(row['commit_width']),
            'rob_entries': int(row['rob_entries']),
            'iq_entries': int(row['iq_entries']),
            'lq_entries': int(row['lq_entries']),
            'sq_entries': int(row['sq_entries']),
            'phys_int_regs': int(row['phys_int_regs']),
            'phys_fp_regs': int(row['phys_fp_regs']),
            'branch_predictor': str(row['branch_predictor']),
            'l1d_size_kb': int(row['l1d_size_kb']),
            'l1d_assoc': int(row['l1d_assoc']),
            'l1i_size_kb': int(row['l1i_size_kb']),
            'l1i_assoc': int(row['l1i_assoc']),
            'l2_size_kb': int(row['l2_size_kb']),
            'l2_assoc': int(row['l2_assoc']),
            'clk_freq_ghz': float(row['clk_freq_ghz']),
            'predicted_ipc': float(row['predicted_ipc']),
            'uncertainty_std': float(row['uncertainty_std'])
        })
    print("=" * 115)
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(queries, f, indent=2)
    print(f"[SUCCESS] Exported {len(queries)} active learning queries to: {output_json}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Uncertainty-Guided Active Learning Candidate Query Selector")
    parser.add_argument("--model", type=str, default="ml/saved_models/best_model_ipc.joblib", help="Path to trained surrogate pipeline")
    parser.add_argument("--pool-size", type=int, default=2000, help="Unlabeled design pool size (default: 2000)")
    parser.add_argument("--queries", type=int, default=15, help="Number of informative samples to query (default: 15)")
    parser.add_argument("--strategy", type=str, default="uncertainty", choices=["uncertainty", "ucb", "random"], help="Query strategy")
    parser.add_argument("--output", type=str, default="configs/active_queries.json", help="Path to output queries JSON")
    args = parser.parse_args()
    
    select_active_learning_queries(args.model, args.pool_size, args.queries, args.strategy, args.output)
