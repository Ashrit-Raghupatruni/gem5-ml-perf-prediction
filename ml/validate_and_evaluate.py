#!/usr/bin/env python3
"""
Ground-Truth Validation & Simulation Time Reduction Engine
Project: ML-Based Microarchitectural Performance Prediction (ARM gem5)
Strictly adheres to: documents/schema.md, documents/rules.md, documents/design_system.md
"""

import os
import sys
import json
import time
import re
import argparse
import subprocess
import socket
import hashlib
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def parse_gem5_stats(stats_path):
    """
    Parses cycle-accurate stats from m5out/stats.txt
    """
    if not os.path.exists(stats_path):
        return None
    stats = {}
    with open(stats_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('---') or line.startswith('#'):
                continue
            parts = re.split(r'\s+', line, maxsplit=2)
            if len(parts) >= 2:
                stats[parts[0]] = parts[1]
                
    sim_insts = int(stats.get('simInsts', 0))
    num_cycles = int(stats.get('board.processor.cores.core.numCycles', 0))
    ipc = float(stats.get('board.processor.cores.core.ipc', 0.0))
    if ipc == 0.0 and num_cycles > 0 and sim_insts > 0:
        ipc = sim_insts / num_cycles
        
    return {
        'sim_insts': sim_insts,
        'num_cycles': num_cycles,
        'ipc': ipc
    }


def execute_validation_simulation(gem5_bin, script_path, benchmark_bin, params, output_dir, timeout=300):
    """
    Launches actual cycle-accurate gem5 simulation for a candidate design.
    """
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        gem5_bin,
        "-d", output_dir,
        script_path,
        "--binary", benchmark_bin,
        "--width", str(params['fetch_width']),
        "--rob", str(params['rob_entries']),
        "--iq", str(params['iq_entries']),
        "--lq", str(params['lq_entries']),
        "--sq", str(params['sq_entries']),
        "--phys-regs", str(params['phys_int_regs']),
        "--branch-pred", str(params['branch_predictor']),
        "--l1d-size", f"{params['l1d_size_kb']}KiB",
        "--l1d-assoc", str(params['l1d_assoc']),
        "--l1i-size", f"{params['l1i_size_kb']}KiB",
        "--l1i-assoc", str(params['l1i_assoc']),
        "--l2-size", f"{params['l2_size_kb']}KiB",
        "--l2-assoc", str(params['l2_assoc']),
        "--clk-freq", f"{params['clk_freq_ghz']}GHz"
    ]
    
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, text=True)
        wallclock_sec = time.perf_counter() - t0
        return proc.returncode, wallclock_sec
    except subprocess.TimeoutExpired:
        return -1, timeout


def plot_speedup_comparison(exhaustive_hours, ml_guided_hours, speedup_ratio, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(8, 5.5), dpi=300)
    sns.set_theme(style="whitegrid")
    
    categories = ['Exhaustive gem5 Simulation\n(5,000 Candidates)', 'ML-Guided Exploration\n(DoE + ML + Validation)']
    runtimes = [exhaustive_hours, ml_guided_hours]
    colors = ['#e53935', '#1e88e5']
    
    bars = plt.bar(categories, runtimes, color=colors, width=0.45, edgecolor='black', linewidth=1.2)
    plt.ylabel("Total Exploration Time (Hours)", fontsize=11, fontweight="bold")
    plt.title("Design Space Exploration: Simulation Time Reduction", fontsize=13, fontweight="bold")
    
    for bar, val in zip(bars, runtimes):
        y_pos = bar.get_height()
        if y_pos > 1.0:
            text = f"{val:.1f} hrs"
        else:
            text = f"{val * 60.0:.1f} min"
        plt.text(bar.get_x() + bar.get_width() / 2.0, y_pos * 1.02, text,
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
                 
    plt.gca().text(0.5, 0.75, f"Compute Speedup:\n{speedup_ratio:.1f}x Faster",
                   transform=plt.gca().transAxes, ha='center', va='center',
                   fontsize=14, fontweight='bold', color='#1565c0',
                   bbox=dict(boxstyle='round,pad=0.6', facecolor='#e3f2fd', edgecolor='#90caf9', linewidth=2))
                   
    plt.ylim(0, max(runtimes) * 1.18)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Speedup comparison plot saved to: {out_path}")


def validate_candidates(candidates_json, gem5_bin, script_path, benchmarks_dir, output_csv="dataset/validation_results.csv"):
    if not os.path.exists(candidates_json):
        print(f"[ERROR] Candidates JSON not found at: {candidates_json}")
        print("Please screen candidates and export top designs using ml/predict_and_screen.py first.")
        sys.exit(1)
        
    with open(candidates_json, 'r') as f:
        candidates = json.load(f)
        
    print("=" * 110)
    print("      GROUND-TRUTH VALIDATION OF TOP PREDICTED MICROARCHITECTURES (gem5 ARM)")
    print("=" * 110)
    print(f"Validating {len(candidates)} candidate designs via cycle-accurate gem5 simulation...")
    
    validation_results = []
    total_val_sim_time = 0.0
    
    for c in candidates:
        cid = c['candidate_id']
        wl = c['workload']
        params = c['params']
        pred_ipc = c['predicted_metrics']['predicted_ipc']
        
        bench_bin = os.path.join(benchmarks_dir, f"{wl}-arm")
        out_dir = f"outputs/validation_{cid}_{wl}"
        
        print(f"\n-> Launching gem5 Ground-Truth Simulation: {cid} ({wl})...")
        ret_code, sim_wallclock = execute_validation_simulation(gem5_bin, script_path, bench_bin, params, out_dir)
        total_val_sim_time += sim_wallclock
        
        if ret_code != 0:
            print(f"[FAIL] Simulation failed or timed out for {cid} (Return code: {ret_code})")
            continue
            
        stats = parse_gem5_stats(os.path.join(out_dir, "stats.txt"))
        if not stats or stats['ipc'] == 0:
            print(f"[FAIL] Unable to parse IPC from stats for {cid}")
            continue
            
        sim_ipc = stats['ipc']
        abs_err = abs(pred_ipc - sim_ipc)
        pct_err = (abs_err / sim_ipc) * 100.0
        
        print(f"   [COMPLETED in {sim_wallclock:.1f}s] Pred IPC: {pred_ipc:.4f} | Sim IPC: {sim_ipc:.4f} | Error: {pct_err:.2f}%")
        
        validation_results.append({
            'candidate_id': cid,
            'workload': wl,
            'predicted_ipc': pred_ipc,
            'simulated_ipc': sim_ipc,
            'absolute_error': abs_err,
            'percentage_error': pct_err,
            'sim_wallclock_sec': sim_wallclock
        })
        
    if not validation_results:
        print("[ERROR] No successful validation runs completed.")
        sys.exit(1)
        
    df_val = pd.DataFrame(validation_results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_val.to_csv(output_csv, index=False)
    print(f"\n[SUCCESS] Exported validation comparison table to: {output_csv}")
    
    mean_val_error = df_val['percentage_error'].mean()
    avg_sim_time = df_val['sim_wallclock_sec'].mean()
    
    # Calculate simulation time reduction across 5,000 design space
    exhaustive_sim_hours = (5000 * avg_sim_time) / 3600.0
    # DoE time (180 runs) + ML inference (<1 min) + Validation time
    doe_sim_hours = (180 * avg_sim_time) / 3600.0
    ml_guided_hours = doe_sim_hours + (total_val_sim_time / 3600.0) + (10.0 / 3600.0)
    speedup = exhaustive_sim_hours / ml_guided_hours if ml_guided_hours > 0 else 0.0
    
    print("\n" + "=" * 110)
    print("                     FINAL EMPIRICAL VALIDATION & SPEEDUP SUMMARY")
    print("=" * 110)
    print(f"Top-5 Mean Validation Error (MAPE) : {mean_val_error:.2f}% (Threshold: <= 7.0%)")
    print(f"Exhaustive 5,000 Simulation Time   : {exhaustive_sim_hours:.2f} Hours (~{exhaustive_sim_hours / 24.0:.1f} Days)")
    print(f"ML-Guided Exploration Time         : {ml_guided_hours:.2f} Hours (~{ml_guided_hours * 60.0:.1f} Minutes)")
    print(f"Effective Simulation Speedup       : {speedup:.1f}x Compute Reduction")
    print("=" * 110)
    
    plot_speedup_comparison(exhaustive_sim_hours, ml_guided_hours, speedup, "visualizations/speedup_comparison.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ground-Truth Validation & Speedup Quantification Engine")
    parser.add_argument("--candidates", type=str, default="dataset/top5_candidates.json", help="Path to top candidates JSON")
    parser.add_argument("--gem5", type=str, default="/home/ashrit/gem5/build/ARM/gem5.opt", help="Path to gem5 ARM binary")
    parser.add_argument("--script", type=str, default="configs/arm_o3_parametric.py", help="Parametric gem5 configuration script")
    parser.add_argument("--benchmarks", type=str, default="benchmarks", help="Path to benchmark binaries directory")
    parser.add_argument("--output", type=str, default="dataset/validation_results.csv", help="Path to output validation CSV")
    args = parser.parse_args()
    
    validate_candidates(args.candidates, args.gem5, args.script, args.benchmarks, args.output)
