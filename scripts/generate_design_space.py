#!/usr/bin/env python3
"""
Latin Hypercube Sampling (LHS) Generator & Multi-Shard Splitter
Project: ML-Based Microarchitectural Performance Prediction (ARM gem5)
Strictly adheres to: documents/schema.md, Plan 1 (3-Member / 3-Laptop Execution)
"""

import os
import json
import argparse
import numpy as np
from scipy.stats import qmc


def generate_lhs_design_space(num_samples=180, seed=42):
    """
    Samples microarchitectural space uniformly across 4 benchmarks.
    """
    workloads = ['matmul', 'stream', 'quicksort', 'hash']
    samples_per_workload = num_samples // len(workloads)
    
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
    
    sampler = qmc.LatinHypercube(d=10, seed=seed)
    
    all_configs = []
    cfg_counter = 1
    
    for wl in workloads:
        sample_matrix = sampler.random(n=samples_per_workload)
        for s in sample_matrix:
            w = widths[int(s[0] * len(widths))]
            rob = rob_sizes[int(s[1] * len(rob_sizes))]
            
            # Legality constraints
            iq = min(96, max(32, int(rob * 0.4)))
            lq = min(64, max(16, int(rob * 0.25)))
            sq = min(64, max(16, int(rob * 0.25)))
            phys_regs = min(256, max(128, int(rob * 0.8)))
            
            l1d_s = l1d_sizes[int(s[2] * len(l1d_sizes))]
            l1d_a = l1d_assocs[int(s[3] * len(l1d_assocs))]
            l1i_s = l1i_sizes[int(s[4] * len(l1i_sizes))]
            l1i_a = l1i_assocs[int(s[5] * len(l1i_assocs))]
            l2_s = l2_sizes[int(s[6] * len(l2_sizes))]
            l2_a = l2_assocs[int(s[7] * len(l2_assocs))]
            freq = freqs[int(s[8] * len(freqs))]
            bp = branch_preds[int(s[9] * len(branch_preds))]
            
            all_configs.append({
                'config_id': f"CFG_{cfg_counter:03d}",
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
            cfg_counter += 1
            
    return all_configs


def save_and_split_shards(configs, output_dir, num_shards=3):
    os.makedirs(output_dir, exist_ok=True)
    master_path = os.path.join(output_dir, "sweep_configs.json")
    with open(master_path, 'w') as f:
        json.dump(configs, f, indent=2)
    print(f"[SUCCESS] Exported master sweep configs ({len(configs)} points) to: {master_path}")
    
    shards_dir = os.path.join(output_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)
    
    shard_size = len(configs) // num_shards
    for i in range(num_shards):
        shard_configs = configs[i * shard_size : (i + 1) * shard_size]
        shard_path = os.path.join(shards_dir, f"shard_{i+1}.json")
        with open(shard_path, 'w') as f:
            json.dump(shard_configs, f, indent=2)
        print(f"[SUCCESS] Exported Shard {i+1} ({len(shard_configs)} points) to: {shard_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LHS Design Space Generator & Sharder")
    parser.add_argument("--samples", type=int, default=180, help="Total simulation sample points (default: 180)")
    parser.add_argument("--shards", type=int, default=3, help="Number of team shards to split across laptops (default: 3)")
    parser.add_argument("--output-dir", type=str, default="configs", help="Directory to save config files")
    args = parser.parse_args()
    
    configs = generate_lhs_design_space(num_samples=args.samples)
    save_and_split_shards(configs, args.output_dir, num_shards=args.shards)
