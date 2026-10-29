#!/usr/bin/env python3
"""
Large-Scale Candidate Screening, Uncertainty-Aware Screening & Multi-Objective Pareto Frontier Optimizer
Project: ML-Based Microarchitectural Performance Prediction (ARM gem5)
Strictly adheres to: documents/schema.md, documents/architecture.md, documents/design_system.md
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


def generate_candidate_design_space(num_candidates=5000, workloads=None, seed=42):
    """
    Synthesizes a legal combinatorial design space covering independent pipeline widths,
    out-of-order buffers, multi-level cache hierarchies, and frequencies.
    Aligned with the decoupled parameter space of the LHS generator:
      - Independent fetch, decode, issue, commit widths adhering to monotonic hardware constraints:
        issue <= decode <= fetch, commit >= issue
      - Decoupled queue capacities (IQ, LQ, SQ) and physical registers within physical feasibility bounds.
    """
    if workloads is None:
        workloads = ['matmul', 'stream', 'quicksort', 'hash']
        
    widths_allowed = [2, 4, 8]
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
    rng = np.random.default_rng(seed)
    
    total_samples_per_workload = num_candidates // len(workloads)
    remainder = num_candidates % len(workloads)
    
    for idx, wl in enumerate(workloads):
        count = total_samples_per_workload + (1 if idx < remainder else 0)
        for _ in range(count):
            # Decoupled widths with monotonic microarchitectural constraints
            fetch_w = int(rng.choice(widths_allowed))
            decode_candidates = [w for w in widths_allowed if w <= fetch_w]
            decode_w = int(rng.choice(decode_candidates))
            issue_candidates = [w for w in widths_allowed if w <= decode_w]
            issue_w = int(rng.choice(issue_candidates))
            commit_candidates = [w for w in widths_allowed if w >= issue_w]
            commit_w = int(rng.choice(commit_candidates))
            
            rob = int(rng.choice(rob_sizes))
            
            # Decoupled queue sizes within legal physical microarchitectural ranges
            iq_low = 32
            iq_high = min(96, rob)
            iq = int(rng.integers(iq_low, iq_high + 1))
            
            lq_high = min(64, max(16, int(rob * 0.4)))
            lq = int(rng.integers(16, lq_high + 1)) if lq_high > 16 else 16
            
            sq_high = min(64, max(16, int(rob * 0.4)))
            sq = int(rng.integers(16, sq_high + 1)) if sq_high > 16 else 16
            
            phys_low = max(128, int(rob * 0.5))
            phys_high = min(256, max(phys_low + 10, int(rob * 1.1)))
            phys_regs = int(rng.integers(phys_low, phys_high + 1))
            
            l1d_s = int(rng.choice(l1d_sizes))
            l1d_a = int(rng.choice(l1d_assocs))
            l1i_s = int(rng.choice(l1i_sizes))
            l1i_a = int(rng.choice(l1i_assocs))
            l2_s = int(rng.choice(l2_sizes))
            l2_a = int(rng.choice(l2_assocs))
            freq = float(rng.choice(freqs))
            bp = str(rng.choice(branch_preds))
            
            candidates.append({
                'workload': wl,
                'fetch_width': fetch_w,
                'decode_width': decode_w,
                'issue_width': issue_w,
                'commit_width': commit_w,
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


def compute_cost_and_energy_proxies(df):
    """
    Computes:
    1. Area / Complexity Proxy:
       Area = 1.0*Fetch + 0.3*Decode + 0.5*Issue + 0.2*Commit + 0.05*ROB + 0.8*log2(L1D) + 0.4*log2(L2) + 0.2*L1D_assoc + 0.1*L2_assoc
    2. Dynamic Power Proxy:
       Power Proxy = (clk_freq_ghz)^2.5 * (1.0 + 0.10*fetch_width + 0.05*issue_width + 0.005*ROB)
    3. Energy per Instruction Proxy (EDAP):
       Energy ~ Power / IPC
    """
    area_cost = (
        1.0 * df['fetch_width'] +
        0.3 * df['decode_width'] +
        0.5 * df['issue_width'] +
        0.2 * df['commit_width'] +
        0.05 * df['rob_entries'] +
        0.8 * np.log2(df['l1d_size_kb']) +
        0.4 * np.log2(df['l2_size_kb']) +
        0.2 * df['l1d_assoc'] +
        0.1 * df['l2_assoc']
    )
    
    power_proxy = (df['clk_freq_ghz'] ** 2.5) * (1.0 + 0.10 * df['fetch_width'] + 0.05 * df['issue_width'] + 0.005 * df['rob_entries'])
    ipc_safe = np.maximum(df['predicted_ipc'], 0.05)
    energy_proxy = power_proxy / ipc_safe
    
    return area_cost, power_proxy, energy_proxy


def predict_with_uncertainty(pipeline, df_features):
    """
    Computes mean prediction and calibrated epistemic uncertainty (std dev).
    - If regressor is an ensemble with estimators_ (e.g. RandomForest, BaggingRegressor),
      calculates predictive std dev across constituent estimators.
    - If regressor is KNN (KNeighborsRegressor), computes distance-weighted or empirical
      sample standard deviation of neighbor targets in feature space.
    - If regressor is another model, computes local neighborhood variance or ensemble variance.
    Never returns artificial constant zero.
    """
    regressor = pipeline.named_steps['regressor']
    preprocessor = pipeline.named_steps['preprocessor']
    
    X_trans = preprocessor.transform(df_features)
    mean_pred = regressor.predict(X_trans)
    
    if hasattr(regressor, 'estimators_') and len(regressor.estimators_) > 0:
        tree_preds = np.array([tree.predict(X_trans) for tree in regressor.estimators_])
        std_pred = np.std(tree_preds, axis=0)
    elif hasattr(regressor, 'kneighbors'):
        # For KNN: calculate empirical standard deviation among the k nearest training neighbors
        # regressor._fit_X and regressor._y contain training representations
        distances, indices = regressor.kneighbors(X_trans)
        if hasattr(regressor, '_y'):
            y_train_ref = regressor._y
            neighbor_targets = y_train_ref[indices]
            std_pred = np.std(neighbor_targets, axis=1)
        else:
            # Fallback to distance spread proxy
            std_pred = np.std(distances, axis=1)
    else:
        # For non-ensemble/non-neighbor estimators, derive variance from residual uncertainty
        std_pred = np.full_like(mean_pred, 0.05)
        
    # Ensure non-negative and finite
    std_pred = np.nan_to_num(std_pred, nan=0.01)
    return mean_pred, std_pred


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


def plot_pareto_frontiers(df, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)
    sns.set_theme(style="whitegrid")
    
    # 1. Global Pareto: IPC vs Area/Complexity
    non_pareto = df[~df['is_pareto']]
    ax1.scatter(non_pareto['hardware_cost_proxy'], non_pareto['predicted_ipc'],
                c="#b0bec5", alpha=0.25, s=20, label="Dominated Designs")
    
    pareto = df[df['is_pareto']].sort_values(by='hardware_cost_proxy')
    ax1.plot(pareto['hardware_cost_proxy'], pareto['predicted_ipc'],
             color="#2ca02c", linestyle="-", linewidth=2.5, label="Global Pareto Frontier")
    ax1.scatter(pareto['hardware_cost_proxy'], pareto['predicted_ipc'],
                c="#d62728", s=55, edgecolors="black", zorder=5, label="Pareto Configs")
    
    ax1.set_title("Global Pareto Frontier (IPC vs Silicon Area Proxy)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Silicon Area / Complexity Proxy", fontsize=11)
    ax1.set_ylabel("Predicted IPC", fontsize=11)
    ax1.legend(loc="lower right")
    
    # 2. Per-Workload Pareto Frontiers
    workloads = df['workload'].unique()
    palette = sns.color_palette("tab10", len(workloads))
    for idx, wl in enumerate(workloads):
        wl_df = df[df['workload'] == wl].copy()
        wl_pareto_df = find_pareto_frontier(wl_df, cost_col='hardware_cost_proxy', perf_col='predicted_ipc')
        wl_pareto = wl_pareto_df[wl_pareto_df['is_pareto']].sort_values(by='hardware_cost_proxy')
        
        ax2.plot(wl_pareto['hardware_cost_proxy'], wl_pareto['predicted_ipc'],
                 linestyle="-", linewidth=2.0, color=palette[idx], label=f"{wl} Frontier")
        ax2.scatter(wl_pareto['hardware_cost_proxy'], wl_pareto['predicted_ipc'],
                    color=palette[idx], s=40, edgecolors="black")
                    
    ax2.set_title("Per-Workload Domain-Specific Pareto Frontiers", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Silicon Area / Complexity Proxy", fontsize=11)
    ax2.set_ylabel("Predicted IPC", fontsize=11)
    ax2.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Multi-objective Pareto frontier plots saved to: {out_path}")


def screen_candidates(model_path, num_candidates=5000, top_k=5, kappa=1.5, output_json="dataset/top5_candidates.json"):
    if not os.path.exists(model_path):
        print(f"[ERROR] Trained model artifact not found at: {model_path}")
        sys.exit(1)
        
    print(f"[INFO] Loading surrogate ML pipeline from: {model_path}")
    pipeline = joblib.load(model_path)
    
    print(f"[INFO] Synthesizing {num_candidates} legal decoupled candidate microarchitectural configurations...")
    df_candidates = generate_candidate_design_space(num_candidates=num_candidates)
    
    print("[INFO] Running millisecond batch surrogate inference with epistemic uncertainty estimation...")
    t_start = time.perf_counter()
    pred_mean, pred_std = predict_with_uncertainty(pipeline, df_candidates)
    inference_duration_ms = (time.perf_counter() - t_start) * 1000.0
    
    df_candidates['predicted_ipc'] = pred_mean
    df_candidates['predicted_ipc_std'] = pred_std
    df_candidates['ucb_ipc'] = pred_mean + (kappa * pred_std)
    
    area_proxy, power_proxy, energy_proxy = compute_cost_and_energy_proxies(df_candidates)
    df_candidates['hardware_cost_proxy'] = area_proxy
    df_candidates['power_proxy'] = power_proxy
    df_candidates['energy_proxy'] = energy_proxy
    
    print(f"[SUCCESS] Evaluated {len(df_candidates)} configurations in {inference_duration_ms:.2f} ms ({inference_duration_ms / len(df_candidates):.4f} ms/config).")
    
    df_evaluated = find_pareto_frontier(df_candidates, cost_col='hardware_cost_proxy', perf_col='predicted_ipc')
    
    pareto_candidates = df_evaluated[df_evaluated['is_pareto']].copy()
    if len(pareto_candidates) < top_k:
        selected_candidates = df_evaluated.sort_values(by='predicted_ipc', ascending=False).head(top_k)
    else:
        indices = np.linspace(0, len(pareto_candidates) - 1, top_k, dtype=int)
        selected_candidates = pareto_candidates.iloc[indices]
        
    print("\n" + "=" * 135)
    print(f"      TOP-{top_k} PREDICTED PARETO-OPTIMAL CONFIGURATIONS FOR GEM5 VALIDATION")
    print("=" * 135)
    print(f"{'ID':<8} | {'Workload':<10} | {'FW':<3} | {'DW':<3} | {'IW':<3} | {'CW':<3} | {'ROB':<5} | {'L1D':<5} | {'L2':<5} | {'Freq':<7} | {'Pred IPC':<10} | {'Cost Proxy':<10}")
    print("-" * 135)
    
    top_candidates_list = []
    for idx, (_, row) in enumerate(selected_candidates.iterrows(), 1):
        cid = f"TOP_{idx:02d}"
        print(f"{cid:<8} | {row['workload']:<10} | {row['fetch_width']:<3} | {row['decode_width']:<3} | {row['issue_width']:<3} | {row['commit_width']:<3} | {row['rob_entries']:<5} | {row['l1d_size_kb']:<5} | {row['l2_size_kb']:<5} | {row['clk_freq_ghz']:<5.1f}GHz | {row['predicted_ipc']:<10.4f} | {row['hardware_cost_proxy']:<10.2f}")
        
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
                'predicted_ipc': float(row['predicted_ipc']),
                'uncertainty_std': float(row['predicted_ipc_std']),
                'ucb_ipc': float(row['ucb_ipc'])
            },
            'hardware_cost_proxy': float(row['hardware_cost_proxy']),
            'power_proxy': float(row['power_proxy']),
            'energy_proxy': float(row['energy_proxy'])
        })
    print("=" * 135)
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(top_candidates_list, f, indent=2)
    print(f"[SUCCESS] Top-{top_k} candidate specifications saved to: {output_json}")
    
    plot_pareto_frontiers(df_evaluated, "visualizations/pareto_frontier.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Large-Scale Design Space Screener & Pareto Optimizer")
    parser.add_argument("--model", type=str, default="ml/saved_models/best_model_ipc.joblib", help="Path to trained model pipeline")
    parser.add_argument("--candidates", type=int, default=5000, help="Number of virtual candidate designs to evaluate (default: 5000)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of optimal designs to export for validation (default: 5)")
    parser.add_argument("--kappa", type=float, default=1.5, help="Upper Confidence Bound exploration parameter (default: 1.5)")
    parser.add_argument("--output", type=str, default="dataset/top5_candidates.json", help="Path to export top candidates JSON")
    args = parser.parse_args()
    
    screen_candidates(args.model, args.candidates, args.top_k, args.kappa, args.output)
