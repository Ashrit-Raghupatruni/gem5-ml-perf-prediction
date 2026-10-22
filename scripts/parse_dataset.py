#!/usr/bin/env python3
"""
gem5 Stats Harvester & Dataset Unifier
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


def parse_single_stats(stats_file):
    stats = {}
    if not os.path.exists(stats_file):
        return None
    with open(stats_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('---') or line.startswith('#'):
                continue
            parts = re.split(r'\s+', line, maxsplit=2)
            if len(parts) >= 2:
                stats[parts[0]] = parts[1]
                
    sim_insts = int(stats.get('simInsts', 0))
    num_cycles = int(stats.get('board.processor.cores.core.numCycles', 0))
    sim_seconds = float(stats.get('simSeconds', 0.0))
    ipc = float(stats.get('board.processor.cores.core.ipc', 0.0))
    if ipc == 0.0 and num_cycles > 0 and sim_insts > 0:
        ipc = sim_insts / num_cycles
    cpi = 1.0 / ipc if ipc > 0 else 0.0
    
    l1d_misses = int(stats.get('board.cache_hierarchy.l2_node.l1d-cache-0.cache.overallMisses::total', 
                     stats.get('board.cache_hierarchy.l1d-cache-0.cache.overallMisses::total', 0)))
    l1d_mpki = (l1d_misses * 1000.0) / sim_insts if sim_insts > 0 else 0.0
    
    l2_misses = int(stats.get('board.cache_hierarchy.l2-cache-0.cache.overallMisses::total', 0))
    l2_mpki = (l2_misses * 1000.0) / sim_insts if sim_insts > 0 else 0.0
    
    cond_incorrect = int(stats.get('board.processor.cores.core.branchPred.condIncorrect', 0))
    cond_predicted = int(stats.get('board.processor.cores.core.branchPred.condPredicted', 0))
    branch_mispred_rate = (cond_incorrect / cond_predicted) if cond_predicted > 0 else 0.0
    
    dram_reads = int(stats.get('board.memory.mem_ctrl0.dram.readReqs', 
                     stats.get('board.memory.readReqs', 0)))
    dram_writes = int(stats.get('board.memory.mem_ctrl0.dram.writeReqs', 
                      stats.get('board.memory.writeReqs', 0)))
    dram_total_reqs = dram_reads + dram_writes
    
    return {
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


def harvest_dataset(configs_json, output_dirs, out_csv):
    with open(configs_json, 'r') as f:
        configs = json.load(f)
        
    cfg_map = {c['config_id']: c for c in configs}
    rows = []
    
    for base_dir in output_dirs:
        for run_dir in glob.glob(os.path.join(base_dir, "run_*")):
            stats_file = os.path.join(run_dir, "stats.txt")
            if not os.path.exists(stats_file):
                continue
                
            dir_name = os.path.basename(run_dir)
            parts = dir_name.split('_')
            if len(parts) >= 2:
                cid = f"{parts[1]}_{parts[2]}" if len(parts) > 2 and parts[1] == 'CFG' else parts[1]
            else:
                continue
                
            cfg = cfg_map.get(cid)
            if not cfg:
                continue
                
            stats = parse_single_stats(stats_file)
            if not stats or stats['sim_insts'] == 0:
                continue
                
            row = dict(cfg)
            row.update(stats)
            rows.append(row)
            
    if not rows:
        print("[ERROR] No valid simulation stats found to harvest.")
        sys.exit(1)
        
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[SUCCESS] Harvested {len(df)} samples to: {out_csv}")
    
    # Compute SHA-256 checksum
    with open(out_csv, 'rb') as f:
        checksum = hashlib.sha256(f.read()).hexdigest()
    sha_path = f"{out_csv}.sha256"
    with open(sha_path, 'w') as f:
        f.write(f"{checksum}  {os.path.basename(out_csv)}\n")
    print(f"[SUCCESS] Generated integrity manifest: {sha_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="gem5 Output Harvester & Dataset Creator")
    parser.add_argument("--configs", type=str, default="configs/sweep_configs.json", help="Master sweep configs JSON")
    parser.add_argument("--inputs", nargs="+", default=["outputs"], help="List of outputs directories to search")
    parser.add_argument("--output", type=str, default="dataset/dataset_cleaned.csv", help="Clean dataset destination CSV")
    args = parser.parse_args()
    
    harvest_dataset(args.configs, args.inputs, args.output)
