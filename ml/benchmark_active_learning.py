#!/usr/bin/env python3
"""
Active Learning Empirical Benchmarking Engine with Multi-Seed Statistical Validation
Project: ML-Based Microarchitectural Performance Prediction (ARM gem5)
Compares Uncertainty Sampling vs UCB vs Random Sampling under identical simulation budgets across repeated seeds.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ml.train_models import build_preprocessor, get_feature_definitions
from ml.predict_and_screen import generate_candidate_design_space, predict_with_uncertainty


def run_active_learning_benchmark(dataset_path, output_dir="visualizations", initial_budget=20, query_step=5, max_budget=70, n_seeds=5):
    """
    Simulates sample-efficiency curves across query strategies on ground-truth gem5 data
    averaged over multiple random seeds to ensure statistical significance.
    """
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset not found: {dataset_path}")
        sys.exit(1)
        
    df = pd.read_csv(dataset_path)
    if len(df) < max_budget:
        max_budget = int(len(df) * 0.8)
        
    cat_cols, num_cols = get_feature_definitions()
    all_features = cat_cols + num_cols
    
    strategies = ['Uncertainty Sampling', 'UCB Sampling', 'Random Sampling']
    budgets = list(range(initial_budget, max_budget + 1, query_step))
    
    # Store runs across seeds: strat -> budget -> list of metric values
    metrics_r2 = {strat: {b: [] for b in budgets} for strat in strategies}
    metrics_mape = {strat: {b: [] for b in budgets} for strat in strategies}
    metrics_best_ipc = {strat: {b: [] for b in budgets} for strat in strategies}
    
    print(f"[INFO] Benchmarking Active Learning strategies over {n_seeds} random seeds across budgets {budgets}...")
    
    seeds = [42 + i * 17 for i in range(n_seeds)]
    
    for seed in seeds:
        rng = np.random.default_rng(seed)
        test_indices = rng.choice(len(df), size=int(len(df) * 0.25), replace=False)
        train_pool_indices = np.setdiff1d(np.arange(len(df)), test_indices)
        
        test_df = df.iloc[test_indices].copy()
        X_test = test_df[all_features]
        y_test = test_df['ipc']
        
        for strat in strategies:
            available_pool = list(train_pool_indices.copy())
            current_train_idx = list(rng.choice(available_pool, size=initial_budget, replace=False))
            available_pool = list(np.setdiff1d(available_pool, current_train_idx))
            
            for b in budgets:
                train_sub_df = df.iloc[current_train_idx].copy()
                X_tr = train_sub_df[all_features]
                y_tr = train_sub_df['ipc']
                
                pipeline = Pipeline([
                    ('preprocessor', build_preprocessor()),
                    ('regressor', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1))
                ])
                pipeline.fit(X_tr, y_tr)
                
                y_pred = pipeline.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mape = mean_absolute_percentage_error(y_test, y_pred) * 100.0
                best_ipc_found = float(y_tr.max())
                
                metrics_r2[strat][b].append(r2)
                metrics_mape[strat][b].append(mape)
                metrics_best_ipc[strat][b].append(best_ipc_found)
                
                if b == budgets[-1] or len(available_pool) < query_step:
                    break
                    
                candidate_pool_df = df.iloc[available_pool].copy()
                mean_preds, std_preds = predict_with_uncertainty(pipeline, candidate_pool_df[all_features])
                
                if strat == 'Uncertainty Sampling':
                    ranked_idx = np.argsort(std_preds)[::-1][:query_step]
                elif strat == 'UCB Sampling':
                    ucb_scores = mean_preds + (1.5 * std_preds)
                    ranked_idx = np.argsort(ucb_scores)[::-1][:query_step]
                else:
                    ranked_idx = rng.choice(len(available_pool), size=query_step, replace=False)
                    
                new_queries = [available_pool[i] for i in ranked_idx]
                current_train_idx.extend(new_queries)
                available_pool = list(np.setdiff1d(available_pool, new_queries))
                
    summary_results = {}
    for strat in strategies:
        summary_results[strat] = {
            'budgets': budgets,
            'r2_mean': [float(np.mean(metrics_r2[strat][b])) for b in budgets],
            'r2_std': [float(np.std(metrics_r2[strat][b])) for b in budgets],
            'mape_mean': [float(np.mean(metrics_mape[strat][b])) for b in budgets],
            'mape_std': [float(np.std(metrics_mape[strat][b])) for b in budgets],
            'best_ipc_mean': [float(np.mean(metrics_best_ipc[strat][b])) for b in budgets],
            'best_ipc_std': [float(np.std(metrics_best_ipc[strat][b])) for b in budgets]
        }
        
    os.makedirs(output_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    sns.set_theme(style="whitegrid")
    
    colors = {'Uncertainty Sampling': '#2ca02c', 'UCB Sampling': '#1f77b4', 'Random Sampling': '#d62728'}
    markers = {'Uncertainty Sampling': 'o', 'UCB Sampling': 's', 'Random Sampling': '^'}
    
    # 1. R2 with confidence bands
    for strat in strategies:
        means = np.array(summary_results[strat]['r2_mean'])
        stds = np.array(summary_results[strat]['r2_std'])
        ax1.plot(budgets, means, label=strat, color=colors[strat], marker=markers[strat], linewidth=2.0)
        ax1.fill_between(budgets, means - stds, means + stds, color=colors[strat], alpha=0.15)
    ax1.set_xlabel("gem5 Simulation Budget (Sample Count)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Held-Out Test $R^2$ Score", fontsize=11, fontweight="bold")
    ax1.set_title(f"Sample Efficiency: Surrogate Accuracy (Mean +/- 1 Std Dev, N={n_seeds} seeds)", fontsize=11, fontweight="bold")
    ax1.legend(loc="lower right")
    
    # 2. Best IPC Found with confidence bands
    for strat in strategies:
        means = np.array(summary_results[strat]['best_ipc_mean'])
        stds = np.array(summary_results[strat]['best_ipc_std'])
        ax2.plot(budgets, means, label=strat, color=colors[strat], marker=markers[strat], linewidth=2.0)
        ax2.fill_between(budgets, means - stds, means + stds, color=colors[strat], alpha=0.15)
    ax2.set_xlabel("gem5 Simulation Budget (Sample Count)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Highest Simulated IPC Identified", fontsize=11, fontweight="bold")
    ax2.set_title(f"Optimization Trajectory: Best Design Identified ({n_seeds} seeds)", fontsize=11, fontweight="bold")
    ax2.legend(loc="lower right")
    
    plt.tight_layout()
    curve_plot_path = os.path.join(output_dir, "active_learning_learning_curves.png")
    plt.savefig(curve_plot_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Exported Active Learning benchmark learning curves to: {curve_plot_path}")
    
    os.makedirs("dataset", exist_ok=True)
    summary_path = os.path.join("dataset", "active_learning_benchmark.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_results, f, indent=2)
    print(f"[SUCCESS] Saved Active Learning multi-seed benchmark JSON to: {summary_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Active Learning Sample Efficiency Benchmarking Engine")
    parser.add_argument("--dataset", type=str, default="dataset/dataset_cleaned.csv", help="Clean dataset path")
    parser.add_argument("--output-dir", type=str, default="visualizations", help="Output directory for plots")
    parser.add_argument("--initial-budget", type=int, default=25, help="Seed simulation budget")
    parser.add_argument("--query-step", type=int, default=5, help="Samples queried per iteration")
    parser.add_argument("--max-budget", type=int, default=75, help="Maximum budget cut-off")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds to average")
    args = parser.parse_args()
    
    run_active_learning_benchmark(args.dataset, args.output_dir, args.initial_budget, args.query_step, args.max_budget, args.seeds)
