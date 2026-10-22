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
import argparse
import subprocess
from multiprocessing import Pool


def run_single_simulation(task_args):
    cfg, gem5_bin, script_path, benchmarks_dir, base_out_dir, timeout = task_args
    cid = cfg['config_id']
    wl = cfg['workload']
    run_out_dir = os.path.join(base_out_dir, f"run_{cid}_{wl}")
    os.makedirs(run_out_dir, exist_ok=True)
    
    bench_bin = os.path.join(benchmarks_dir, f"{wl}-arm")
    
    cmd = [
        gem5_bin,
        "-d", run_out_dir,
        script_path,
        "--binary", bench_bin,
        "--width", str(cfg['fetch_width']),
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
    try:
        with open(log_file, "w") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
        duration = time.perf_counter() - t0
        if proc.returncode != 0:
            status = f"FAILED (Exit code {proc.returncode})"
    except subprocess.TimeoutExpired:
        duration = timeout
        status = "TIMEOUT"
        
    print(f"[{cid}] ({wl}) Width={cfg['fetch_width']} ROB={cfg['rob_entries']} Freq={cfg['clk_freq_ghz']}GHz -> {status} ({duration:.1f}s)")
    return {
        'config_id': cid,
        'workload': wl,
        'status': status,
        'duration_sec': duration,
        'output_dir': run_out_dir
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
    
    tasks = [(cfg, gem5_bin, script_path, benchmarks_dir, out_dir, timeout) for cfg in configs]
    
    t_start = time.perf_counter()
    with Pool(processes=workers) as pool:
        results = pool.map(run_single_simulation, tasks)
        
    total_time = time.perf_counter() - t_start
    success_count = sum(1 for r in results if r['status'] == "SUCCESS")
    
    print("=" * 110)
    print(f"   SWEEP FINISHED: {success_count}/{len(configs)} Succeeded in {total_time / 60.0:.2f} Minutes")
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
