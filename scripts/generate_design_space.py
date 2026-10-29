#!/usr/bin/env python3
"""
Latin Hypercube Sampling (LHS) Generator with Independent Parameter Decoupling
Project: ML-Based Microarchitectural Performance Prediction (ARM gem5)
Strictly adheres to: documents/schema.md, Plan 1 (3-Member / 3-Laptop Execution)
"""

import os
import json
import argparse
import numpy as np
from scipy.stats import qmc


def generate_lhs_design_space(num_samples=180, seed=42, independent_widths=True):
    """
    Samples microarchitectural space uniformly across 4 benchmarks.
    When independent_widths=True, every independently varied parameter is assigned
    its own unique dimension in the LHS matrix (d=17):
      dim 0: fetch_width
      dim 1: decode_width
      dim 2: issue_width
      dim 3: commit_width
      dim 4: rob_entries
      dim 5: iq_entries
      dim 6: lq_entries
      dim 7: sq_entries
      dim 8: phys_int_regs
      dim 9: l1d_size_kb
      dim 10: l1d_assoc
      dim 11: l1i_size_kb
      dim 12: l1i_assoc
      dim 13: l2_size_kb
      dim 14: l2_assoc
      dim 15: clk_freq_ghz
      dim 16: branch_predictor
    Ensures zero dimension-reuse or unintended inter-parameter correlations.
    """
    workloads = ['matmul', 'stream', 'quicksort', 'hash']
    base_samples = num_samples // len(workloads)
    remainder = num_samples % len(workloads)
    
    wl_samples = {wl: base_samples + (1 if idx < remainder else 0) for idx, wl in enumerate(workloads)}
    
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
    
    # 17 orthogonal LHS dimensions for decoupled microarchitecture
    dim = 17 if independent_widths else 10
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    
    all_configs = []
    seen_hashes = set()
    cfg_counter = 1
    
    for wl in workloads:
        count = wl_samples[wl]
        if count == 0:
            continue
            
        sample_matrix = sampler.random(n=count)
        for s in sample_matrix:
            if independent_widths:
                # Decoupled widths with monotonic legality constraints
                fetch_w = widths_allowed[int(s[0] * len(widths_allowed))]
                # Decode width <= fetch width
                decode_candidates = [w for w in widths_allowed if w <= fetch_w]
                decode_w = decode_candidates[int(s[1] * len(decode_candidates))]
                # Issue width <= decode width
                issue_candidates = [w for w in widths_allowed if w <= decode_w]
                issue_w = issue_candidates[int(s[2] * len(issue_candidates))]
                # Commit width >= issue width
                commit_candidates = [w for w in widths_allowed if w >= issue_w]
                commit_w = commit_candidates[int(s[3] * len(commit_candidates))]
                
                rob = rob_sizes[int(s[4] * len(rob_sizes))]
                
                # Decoupled queue sizes with physical legality bounds
                # IQ: 32 to min(96, ROB)
                iq_low = 32
                iq_high = min(96, rob)
                iq = int(iq_low + s[5] * (iq_high - iq_low))
                
                # LQ and SQ: 16 to min(64, int(ROB * 0.5))
                lq_high = min(64, max(16, int(rob * 0.4)))
                lq = int(16 + s[6] * (lq_high - 16)) if lq_high > 16 else 16
                
                sq_high = min(64, max(16, int(rob * 0.4)))
                sq = int(16 + s[7] * (sq_high - 16)) if sq_high > 16 else 16
                
                # Physical registers: max(128, int(ROB * 0.5)) to min(256, int(ROB * 1.2))
                phys_low = max(128, int(rob * 0.5))
                phys_high = min(256, max(phys_low + 10, int(rob * 1.1)))
                phys_regs = int(phys_low + s[8] * (phys_high - phys_low))
                
                # Caches, clocks, branch predictors - EACH HAS ITS OWN EXCLUSIVE DIMENSION:
                l1d_s = l1d_sizes[int(s[9] * len(l1d_sizes))]
                l1d_a = l1d_assocs[int(s[10] * len(l1d_assocs))]
                l1i_s = l1i_sizes[int(s[11] * len(l1i_sizes))]
                l1i_a = l1i_assocs[int(s[12] * len(l1i_assocs))]
                l2_s = l2_sizes[int(s[13] * len(l2_sizes))]
                l2_a = l2_assocs[int(s[14] * len(l2_assocs))]
                freq = freqs[int(s[15] * len(freqs))]
                bp = branch_preds[int(s[16] * len(branch_preds))]
            else:
                # Coupled mode for backward compatibility
                w = widths_allowed[int(s[0] * len(widths_allowed))]
                fetch_w = decode_w = issue_w = commit_w = w
                rob = rob_sizes[int(s[1] * len(rob_sizes))]
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
                
            cfg_tuple = (wl, fetch_w, decode_w, issue_w, commit_w, rob, iq, lq, sq, phys_regs, bp, l1d_s, l1d_a, l1i_s, l1i_a, l2_s, l2_a, freq)
            if cfg_tuple in seen_hashes:
                freq = freqs[(freqs.index(freq) + 1) % len(freqs)]
                cfg_tuple = (wl, fetch_w, decode_w, issue_w, commit_w, rob, iq, lq, sq, phys_regs, bp, l1d_s, l1d_a, l1i_s, l1i_a, l2_s, l2_a, freq)
                
            seen_hashes.add(cfg_tuple)
            
            all_configs.append({
                'config_id': f"CFG_{cfg_counter:03d}",
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
    
    total_configs = len(configs)
    k, m = divmod(total_configs, num_shards)
    shards = [configs[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_shards)]
    
    for i, shard_configs in enumerate(shards):
        shard_path = os.path.join(shards_dir, f"shard_{i+1}.json")
        with open(shard_path, 'w') as f:
            json.dump(shard_configs, f, indent=2)
        print(f"[SUCCESS] Exported Shard {i+1} ({len(shard_configs)} points) to: {shard_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LHS Design Space Generator & Sharder with Decoupled Parameters")
    parser.add_argument("--samples", type=int, default=180, help="Total simulation sample points (default: 180)")
    parser.add_argument("--shards", type=int, default=3, help="Number of team shards to split across laptops (default: 3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for LHS sampling (default: 42)")
    parser.add_argument("--coupled", action="store_true", help="Force coupled widths and proportional queues (legacy mode)")
    parser.add_argument("--output-dir", type=str, default="configs", help="Directory to save config files")
    args = parser.parse_args()
    
    configs = generate_lhs_design_space(num_samples=args.samples, seed=args.seed, independent_widths=not args.coupled)
    save_and_split_shards(configs, args.output_dir, num_shards=args.shards)
