#!/usr/bin/env python3
"""
Parallel Shard/Master gem5 Sweep Execution Runner
Project: ML-Based Microarchitectural Performance Prediction (ARM gem5)
Strictly adheres to: documents/security.md, documents/user_flow.md, Plan 1
"""

import os
import sys
import json
import time
import re
import argparse
import subprocess
from multiprocessing import Pool


def verify_simulation_output(run_out_dir):
    """
    Verifies that simulation completed properly:
    1. stats.txt exists and is non-empty
    2. simInsts exists and is > 0
    """
    stats_file = os.path.join(run_out_dir, "stats.txt")
    if not os.path.exists(stats_file) or os.path.getsize(stats_file) == 0:
        return False, "Missing or empty stats.txt"
    
    sim_insts = 0
    try:
        with open(stats_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if 'simInsts' in line:
                    parts = re.split(r'\s+', line.strip(), maxsplit=2)
                    if len(parts) >= 2 and parts[0] == 'simInsts':
                        try:
                            sim_insts = int(parts[1])
                            break
                        except ValueError:
                            pass
    except Exception as e:
        return False, f"Error reading stats.txt: {e}"
        
    if sim_insts <= 0:
        return False, f"Invalid simInsts ({sim_insts})"
        
    return True, f"Valid run (simInsts={sim_insts})"


def run_single_simulation(task_args):
    cfg, gem5_bin, script_path, benchmarks_dir, base_out_dir, timeout = task_args
    cid = cfg['config_id']
    wl = cfg['workload']
    run_out_dir = os.path.join(base_out_dir, f"run_{cid}_{wl}")
    os.makedirs(run_out_dir, exist_ok=True)
    
    bench_bin = os.path.join(benchmarks_dir, f"{wl}-arm")
    
    # Retrieve discrete widths or fallback to fetch_width
    fetch_w = cfg.get('fetch_width', 4)
    decode_w = cfg.get('decode_width', fetch_w)
    issue_w = cfg.get('issue_width', fetch_w)
    commit_w = cfg.get('commit_width', fetch_w)
    
    cmd = [
        gem5_bin,
        "-d", run_out_dir,
        script_path,
        "--binary", bench_bin,
        "--fetch-width", str(fetch_w),
        "--decode-width", str(decode_w),
        "--issue-width", str(issue_w),
        "--commit-width", str(commit_w),
        "--rob", str(cfg['rob_entries']),
        "--iq", str(cfg['iq_entries']),
        "--lq", str(cfg['lq_entries']),
        "--sq", str(cfg['sq_entries']),
        "--phys-regs", str(cfg['phys_int_regs']),
        "--branch-pred", str(cfg['branch_predictor']),
        "--l1d-size", f"{cfg['l1d_size_kb']}KiB",
        "--l1d-assoc", str(cfg['l1d_assoc']),
        "--l1i-size", f"{cfg['l1i_size_kb']}KiB",
        "--l1i-assoc", str(cfg['l1i_assoc']),
        "--l2-size", f"{cfg['l2_size_kb']}KiB",
        "--l2-assoc", str(cfg['l2_assoc']),
        "--clk-freq", f"{cfg['clk_freq_ghz']}GHz"
    ]
    
    log_file = os.path.join(run_out_dir, "run.log")
    t0 = time.perf_counter()
    status = "SUCCESS"
    details = ""
    try:
        with open(log_file, "w") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
        duration = time.perf_counter() - t0
        if proc.returncode != 0:
            status = f"FAILED (Exit code {proc.returncode})"
            details = f"Process exit code: {proc.returncode}"
        else:
            # Rigorous check on outputs
            is_valid, reason = verify_simulation_output(run_out_dir)
            if not is_valid:
                status = f"FAILED ({reason})"
                details = reason
            else:
                details = reason
    except subprocess.TimeoutExpired:
        duration = timeout
        status = "TIMEOUT"
        details = f"Timed out after {timeout}s"
        
    print(f"[{cid}] ({wl}) FetchW={fetch_w} IssueW={issue_w} ROB={cfg['rob_entries']} Freq={cfg['clk_freq_ghz']}GHz -> {status} ({duration:.1f}s)")
    return {
        'config_id': cid,
        'workload': wl,
        'status': status,
        'duration_sec': duration,
        'output_dir': run_out_dir,
        'details': details
    }


def execute_parallel_sweep(shard_json, gem5_bin, script_path, benchmarks_dir, out_dir, workers=4, timeout=300):
    if not os.path.exists(shard_json):
        print(f"[ERROR] Configuration shard file not found: {shard_json}")
        sys.exit(1)
        
    with open(shard_json, 'r') as f:
        configs = json.load(f)
        
    print("=" * 110)
    print(f"   STARTING PARALLEL GEM5 SWEEP: {len(configs)} CONFIGURATIONS | WORKERS: {workers}")
    print(f"   Configs Shard : {shard_json}")
    print(f"   gem5 Binary   : {gem5_bin}")
    print(f"   Outputs Dir   : {out_dir}")
    print("=" * 110)
    
    os.makedirs(out_dir, exist_ok=True)
    tasks = [(cfg, gem5_bin, script_path, benchmarks_dir, out_dir, timeout) for cfg in configs]
    
    t_start = time.perf_counter()
    with Pool(processes=workers) as pool:
        results = pool.map(run_single_simulation, tasks)
        
    total_wall_clock = time.perf_counter() - t_start
    total_cpu_seconds = sum(r['duration_sec'] for r in results)
    success_count = sum(1 for r in results if r['status'] == "SUCCESS")
    
    manifest = {
        'shard_file': shard_json,
        'total_configs': len(configs),
        'success_count': success_count,
        'failed_count': len(configs) - success_count,
        'workers': workers,
        'total_wall_clock_sec': total_wall_clock,
        'total_cpu_seconds': total_cpu_seconds,
        'simulations': results
    }
    manifest_path = os.path.join(out_dir, "simulation_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"[INFO] Saved run manifest: {manifest_path}")
    
    print("=" * 110)
    print(f"   SWEEP FINISHED: {success_count}/{len(configs)} Succeeded in {total_wall_clock / 60.0:.2f} Minutes (Wall-clock) | Aggregate CPU: {total_cpu_seconds / 60.0:.2f} Core-Minutes")
    print("=" * 110)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parallel gem5 Simulation Sweep Runner")
    parser.add_argument("--shard", type=str, required=True, help="Path to config shard JSON file")
    parser.add_argument("--gem5", type=str, default="/home/ashrit/gem5/build/ARM/gem5.opt", help="gem5 binary path")
    parser.add_argument("--script", type=str, default="configs/arm_o3_parametric.py", help="gem5 python script")
    parser.add_argument("--benchmarks", type=str, default="benchmarks", help="Benchmarks directory")
    parser.add_argument("--out-dir", type=str, default="outputs", help="Base outputs directory")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker threads")
    parser.add_argument("--timeout", type=int, default=300, help="Per-run watchdog timeout in seconds")
    args = parser.parse_args()
    
    execute_parallel_sweep(args.shard, args.gem5, args.script, args.benchmarks, args.out_dir, args.workers, args.timeout)
