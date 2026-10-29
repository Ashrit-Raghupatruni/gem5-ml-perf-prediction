#!/usr/bin/env python3
"""
gem5 Stats Harvester & Dataset Unifier with Quality Verification
Project: ML-Based Microarchitectural Performance Prediction (ARM gem5)
Strictly adheres to: documents/schema.md, documents/security.md
"""

import os
import sys
import re
import json
import glob
import hashlib
import argparse
import pandas as pd


MANDATORY_STAT_KEYS = [
    'simInsts',
    'board.processor.cores.core.numCycles',
    'simSeconds'
]


def parse_single_stats(stats_file):
    """
    Parses gem5 stats.txt. Returns None if mandatory performance counters are missing
    or if the simulation did not complete properly (simInsts <= 0 or numCycles <= 0).
    """
    if not os.path.exists(stats_file) or os.path.getsize(stats_file) == 0:
        return None, "stats.txt missing or empty"
        
    stats = {}
    with open(stats_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('---') or line.startswith('#'):
                continue
            parts = re.split(r'\s+', line, maxsplit=2)
            if len(parts) >= 2:
                stats[parts[0]] = parts[1]
                
    # Strict validation of mandatory counters
    if 'simInsts' not in stats:
        return None, "simInsts metric missing"
    if 'board.processor.cores.core.numCycles' not in stats:
        return None, "core.numCycles metric missing"
        
    try:
        sim_insts = int(stats['simInsts'])
        num_cycles = int(stats['board.processor.cores.core.numCycles'])
        sim_seconds = float(stats.get('simSeconds', 0.0))
    except ValueError as e:
        return None, f"Counter parsing error: {e}"
        
    if sim_insts <= 0 or num_cycles <= 0:
        return None, f"Non-positive instructions ({sim_insts}) or cycles ({num_cycles})"
        
    # IPC calculation
    if 'board.processor.cores.core.ipc' in stats:
        try:
            ipc = float(stats['board.processor.cores.core.ipc'])
        except ValueError:
            ipc = sim_insts / num_cycles
    else:
        ipc = sim_insts / num_cycles
        
    cpi = 1.0 / ipc if ipc > 0 else 0.0
    
    # Cache and branch statistics (with safe lookup)
    l1d_raw = (stats.get('board.cache_hierarchy.l2_node.l1d-cache-0.cache.overallMisses::total') or
               stats.get('board.cache_hierarchy.l1d-cache-0.cache.overallMisses::total') or
               stats.get('board.processor.cores.core.dcache.overallMisses::total') or '0')
    l1d_misses = int(l1d_raw)
    l1d_mpki = (l1d_misses * 1000.0) / sim_insts
    
    l2_raw = (stats.get('board.cache_hierarchy.l2-cache-0.cache.overallMisses::total') or
              stats.get('board.l2cache.overallMisses::total') or '0')
    l2_misses = int(l2_raw)
    l2_mpki = (l2_misses * 1000.0) / sim_insts
    
    cond_incorrect = int(stats.get('board.processor.cores.core.branchPred.condIncorrect',
                         stats.get('board.processor.cores.core.branchPred.cond_incorrect', 0)))
    cond_predicted = int(stats.get('board.processor.cores.core.branchPred.condPredicted',
                         stats.get('board.processor.cores.core.branchPred.cond_predicted', 0)))
    branch_mispred_rate = (cond_incorrect / cond_predicted) if cond_predicted > 0 else 0.0
    
    dram_reads = int(stats.get('board.memory.mem_ctrl0.dram.readReqs',
                     stats.get('board.memory.readReqs', 0)))
    dram_writes = int(stats.get('board.memory.mem_ctrl0.dram.writeReqs',
                      stats.get('board.memory.writeReqs', 0)))
    dram_total_reqs = dram_reads + dram_writes
    
    parsed = {
        'sim_seconds': sim_seconds,
        'sim_insts': sim_insts,
        'num_cycles': num_cycles,
        'ipc': ipc,
        'cpi': cpi,
        'l1d_misses': l1d_misses,
        'l1d_mpki': l1d_mpki,
        'l2_misses': l2_misses,
        'l2_mpki': l2_mpki,
        'branch_mispred_count': cond_incorrect,
        'branch_mispred_rate': branch_mispred_rate,
        'dram_reads': dram_reads,
        'dram_writes': dram_writes,
        'dram_total_reqs': dram_total_reqs
    }
    return parsed, "OK"


def harvest_dataset(configs_json, output_dirs, out_csv, quality_report_path=None):
    if not os.path.exists(configs_json):
        print(f"[ERROR] Configs JSON not found: {configs_json}")
        sys.exit(1)
        
    with open(configs_json, 'r') as f:
        configs = json.load(f)
        
    cfg_map = {c['config_id']: c for c in configs}
    rows = []
    
    scanned_runs = 0
    missing_stats = 0
    invalid_stats = 0
    missing_config = 0
    rejection_reasons = {}
    
    for base_dir in output_dirs:
        for run_dir in glob.glob(os.path.join(base_dir, "run_*")):
            scanned_runs += 1
            stats_file = os.path.join(run_dir, "stats.txt")
            if not os.path.exists(stats_file):
                missing_stats += 1
                rejection_reasons["missing_stats_file"] = rejection_reasons.get("missing_stats_file", 0) + 1
                continue
                
            dir_name = os.path.basename(run_dir)
            parts = dir_name.split('_')
            if len(parts) >= 2:
                cid = f"{parts[1]}_{parts[2]}" if len(parts) > 2 and parts[1] == 'CFG' else parts[1]
            else:
                missing_config += 1
                continue
                
            cfg = cfg_map.get(cid)
            if not cfg:
                missing_config += 1
                rejection_reasons["config_id_not_in_manifest"] = rejection_reasons.get("config_id_not_in_manifest", 0) + 1
                continue
                
            stats, reason = parse_single_stats(stats_file)
            if not stats:
                invalid_stats += 1
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                continue
                
            row = dict(cfg)
            row.update(stats)
            rows.append(row)
            
    if not rows:
        print("[ERROR] No valid simulation stats found to harvest.")
        sys.exit(1)
        
    df = pd.DataFrame(rows)
    
    # Check duplicates
    raw_count = len(df)
    df = df.drop_duplicates(subset=['config_id', 'workload'], keep='last')
    duplicates_removed = raw_count - len(df)
    
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[SUCCESS] Harvested {len(df)} validated samples to: {out_csv}")
    
    # Compute SHA-256 checksum
    with open(out_csv, 'rb') as f:
        checksum = hashlib.sha256(f.read()).hexdigest()
    sha_path = f"{out_csv}.sha256"
    with open(sha_path, 'w') as f:
        f.write(f"{checksum}  {os.path.basename(out_csv)}\n")
    print(f"[SUCCESS] Generated integrity manifest: {sha_path}")
    
    # Generate Quality Report
    if quality_report_path is None:
        quality_report_path = os.path.join(os.path.dirname(out_csv), "dataset_quality_report.json")
        
    report = {
        'total_runs_scanned': scanned_runs,
        'valid_runs_harvested': len(df),
        'duplicates_removed': duplicates_removed,
        'missing_stats_count': missing_stats,
        'invalid_stats_count': invalid_stats,
        'rejection_breakdown': rejection_reasons,
        'workload_distribution': df['workload'].value_counts().to_dict(),
        'feature_summary': {
            'ipc_min': float(df['ipc'].min()),
            'ipc_max': float(df['ipc'].max()),
            'ipc_mean': float(df['ipc'].mean()),
            'ipc_std': float(df['ipc'].std()),
            'cycles_min': int(df['num_cycles'].min()),
            'cycles_max': int(df['num_cycles'].max()),
            'sim_insts_min': int(df['sim_insts'].min()),
            'sim_insts_max': int(df['sim_insts'].max())
        },
        'sha256': checksum
    }
    with open(quality_report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"[SUCCESS] Quality report generated: {quality_report_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="gem5 Output Harvester & Dataset Creator")
    parser.add_argument("--configs", type=str, default="configs/sweep_configs.json", help="Master sweep configs JSON")
    parser.add_argument("--inputs", nargs="+", default=["outputs"], help="List of outputs directories to search")
    parser.add_argument("--output", type=str, default="dataset/dataset_cleaned.csv", help="Clean dataset destination CSV")
    parser.add_argument("--quality-report", type=str, default=None, help="Path for quality report JSON")
    args = parser.parse_args()
    
    harvest_dataset(args.configs, args.inputs, args.output, args.quality_report)
