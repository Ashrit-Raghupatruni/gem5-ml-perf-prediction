#!/usr/bin/env python3
"""
Large-Scale Candidate Screening & Pareto Frontier Optimizer
Project: ML-Based Microarchitectural Performance Prediction (ARM gem5)
Strictly adheres to: documents/schema.md, documents/architecture.md, documents/design_system.md
"""

import os
import sys
import json
import time
import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


def generate_candidate_design_space(num_candidates=5000, workloads=None):
    """
    Synthesizes a legal combinatorial design space covering pipeline widths,
    out-of-order buffers, multi-level cache hierarchies, and frequencies.
    """
    if workloads is None:
        workloads = ['matmul', 'stream', 'quicksort', 'hash']
        
    widths = [2, 4, 8]
    rob_sizes = [64, 96, 128, 192, 256]
    l1d_sizes = [16, 32, 64, 128]
    l1d_assocs = [2, 4, 8]
    l1i_sizes = [16, 32, 64]
    l1i_assocs = [2, 4, 8]
    l2_sizes = [256, 512, 1024, 2048]
    l2_assocs = [4, 8, 16]
    freqs = [1.5, 2.0, 2.4, 3.0]
    branch_preds = ['TournamentBP', 'BiModeBP', 'LocalBP']
    
    candidates = []
    np.random.seed(42)
    
    total_samples_per_workload = num_candidates // len(workloads)
    
    for wl in workloads:
        for _ in range(total_samples_per_workload):
            w = int(np.random.choice(widths))
            rob = int(np.random.choice(rob_sizes))
            
            # Legality constraints: LQ + SQ <= ROB, IQ <= ROB
            iq = min(96, max(32, int(rob * 0.4)))
            lq = min(64, max(16, int(rob * 0.25)))
            sq = min(64, max(16, int(rob * 0.25)))
            phys_regs = min(256, max(128, int(rob * 0.8)))
            
            l1d_s = int(np.random.choice(l1d_sizes))
            l1d_a = int(np.random.choice(l1d_assocs))
            l1i_s = int(np.random.choice(l1i_sizes))
            l1i_a = int(np.random.choice(l1i_assocs))
            l2_s = int(np.random.choice(l2_sizes))
            l2_a = int(np.random.choice(l2_assocs))
            freq = float(np.random.choice(freqs))
            bp = str(np.random.choice(branch_preds))
            
            candidates.append({
                'workload': wl,
                'fetch_width': w,
                'decode_width': w,
                'issue_width': w,
                'commit_width': w,
                'rob_entries': rob,
                'iq_entries': iq,
                'lq_entries': lq,
                'sq_entries': sq,
                'phys_int_regs': phys_regs,
                'phys_fp_regs': phys_regs,
                'branch_predictor': bp,
                'l1d_size_kb': l1d_s,
                'l1d_assoc': l1d_a,
                'l1i_size_kb': l1i_s,
                'l1i_assoc': l1i_a,
                'l2_size_kb': l2_s,
                'l2_assoc': l2_a,
                'clk_freq_ghz': freq
            })
            
    df_candidates = pd.DataFrame(candidates)
    return df_candidates


def compute_hardware_cost_proxy(df):
    """
    Computes an estimated hardware complexity / area proxy:
    Cost = 1.5*Width + 0.05*ROB + 0.8*log2(L1D) + 0.4*log2(L2) + 0.2*Assoc
    """
    cost = (
        1.5 * df['fetch_width'] +
        0.05 * df['rob_entries'] +
        0.8 * np.log2(df['l1d_size_kb']) +
        0.4 * np.log2(df['l2_size_kb']) +
        0.2 * df['l1d_assoc'] +
        0.1 * df['l2_assoc']
    )
    return cost


def find_pareto_frontier(df, cost_col='hardware_cost_proxy', perf_col='predicted_ipc'):
    """
    Extracts non-dominated Pareto-optimal configurations maximizing performance while minimizing cost.
    """
    sorted_df = df.sort_values(by=[cost_col, perf_col], ascending=[True, False]).copy()
    
    pareto_mask = []
    current_max_perf = -1.0
    
    for _, row in sorted_df.iterrows():
        perf = row[perf_col]
        if perf > current_max_perf:
            pareto_mask.append(True)
            current_max_perf = perf
        else:
            pareto_mask.append(False)
            
    sorted_df['is_pareto'] = pareto_mask
    return sorted_df


def plot_pareto_frontier(df, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(9, 6), dpi=300)
    sns.set_theme(style="whitegrid")
    
    # Non-pareto points
    non_pareto = df[~df['is_pareto']]
    plt.scatter(non_pareto['hardware_cost_proxy'], non_pareto['predicted_ipc'],
                c="#b0bec5", alpha=0.35, s=25, label="Dominated Design Points")
    
    # Pareto points
    pareto = df[df['is_pareto']].sort_values(by='hardware_cost_proxy')
    plt.plot(pareto['hardware_cost_proxy'], pareto['predicted_ipc'],
             color="#2ca02c", linestyle="-", linewidth=2.5, label="Pareto-Optimal Envelope")
    plt.scatter(pareto['hardware_cost_proxy'], pareto['predicted_ipc'],
                c="#d62728", s=60, edgecolors="black", zorder=5, label="Pareto Frontier Configs")
    
    plt.title("Microarchitecture Design Space: Pareto Frontier (IPC vs Silicon Cost Proxy)", fontsize=13, fontweight="bold")
    plt.xlabel("Estimated Silicon Hardware Cost Proxy (Area/Power Complexity)", fontsize=11)
    plt.ylabel("Surrogate-Predicted IPC (ARM gem5)", fontsize=11)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Pareto frontier plot saved to: {out_path}")


def screen_candidates(model_path, num_candidates=5000, top_k=5, output_json="dataset/top5_candidates.json"):
    if not os.path.exists(model_path):
        print(f"[ERROR] Trained model artifact not found at: {model_path}")
        print("Please train and export the surrogate model using ml/train_models.py first.")
        sys.exit(1)
        
    print(f"[INFO] Loading surrogate ML pipeline from: {model_path}")
    pipeline = joblib.load(model_path)
    
    print(f"[INFO] Synthesizing {num_candidates} legal candidate microarchitectural configurations...")
    df_candidates = generate_candidate_design_space(num_candidates=num_candidates)
    
    print("[INFO] Running millisecond batch surrogate inference...")
    t_start = time.perf_counter()
    predicted_ipc = pipeline.predict(df_candidates)
    inference_duration_ms = (time.perf_counter() - t_start) * 1000.0
    
    df_candidates['predicted_ipc'] = predicted_ipc
    df_candidates['hardware_cost_proxy'] = compute_hardware_cost_proxy(df_candidates)
    
    print(f"[SUCCESS] Evaluated {len(df_candidates)} configurations in {inference_duration_ms:.2f} ms ({inference_duration_ms / len(df_candidates):.4f} ms/config).")
    
    df_evaluated = find_pareto_frontier(df_candidates)
    
    # Select Top-K optimal designs on Pareto frontier
    pareto_candidates = df_evaluated[df_evaluated['is_pareto']].copy()
    if len(pareto_candidates) < top_k:
        selected_candidates = df_evaluated.sort_values(by='predicted_ipc', ascending=False).head(top_k)
    else:
        # Sample evenly across cost spectrum along the pareto front
        indices = np.linspace(0, len(pareto_candidates) - 1, top_k, dtype=int)
        selected_candidates = pareto_candidates.iloc[indices]
        
    print("\n" + "=" * 110)
    print(f"      TOP-{top_k} PREDICTED PARETO-OPTIMAL CONFIGURATIONS FOR GEM5 VALIDATION")
    print("=" * 110)
    print(f"{'ID':<8} | {'Workload':<10} | {'Width':<6} | {'ROB':<6} | {'L1D (KiB)':<10} | {'L2 (KiB)':<9} | {'Freq':<7} | {'Pred IPC':<10} | {'Cost Proxy':<10}")
    print("-" * 110)
    
    top_candidates_list = []
    for idx, (_, row) in enumerate(selected_candidates.iterrows(), 1):
        cid = f"TOP_{idx:02d}"
        print(f"{cid:<8} | {row['workload']:<10} | {row['fetch_width']:<6} | {row['rob_entries']:<6} | {row['l1d_size_kb']:<10} | {row['l2_size_kb']:<9} | {row['clk_freq_ghz']:<5.1f}GHz | {row['predicted_ipc']:<10.4f} | {row['hardware_cost_proxy']:<10.2f}")
        
        top_candidates_list.append({
            'candidate_id': cid,
            'workload': row['workload'],
            'params': {
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
                'clk_freq_ghz': float(row['clk_freq_ghz'])
            },
            'predicted_metrics': {
                'predicted_ipc': float(row['predicted_ipc'])
            },
            'hardware_cost_proxy': float(row['hardware_cost_proxy'])
        })
    print("=" * 110)
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(top_candidates_list, f, indent=2)
    print(f"[SUCCESS] Top-{top_k} candidate specifications saved to: {output_json}")
    
    plot_pareto_frontier(df_evaluated, "visualizations/pareto_frontier.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Large-Scale Design Space Screener & Pareto Optimizer")
    parser.add_argument("--model", type=str, default="ml/saved_models/best_model_ipc.joblib", help="Path to trained model pipeline")
    parser.add_argument("--candidates", type=int, default=5000, help="Number of virtual candidate designs to evaluate (default: 5000)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of optimal designs to export for validation (default: 5)")
    parser.add_argument("--output", type=str, default="dataset/top5_candidates.json", help="Path to export top candidates JSON")
    args = parser.parse_args()
    
    screen_candidates(args.model, args.candidates, args.top_k, args.output)
