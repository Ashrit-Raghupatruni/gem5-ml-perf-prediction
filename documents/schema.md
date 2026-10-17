# Data Schemas & Model Architecture Specification

## 1. Overview
This document specifies the exact schemas, column definitions, data types, normalization rules, and serialization formats across all stages of the data and machine learning pipeline.

---

## 2. Dataset Schema (`dataset/dataset_cleaned.csv`)

### 2.1 Microarchitectural Input Features ($X$)

| Column Name | Data Type | Units / Scale | Legal Values / Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| `config_id` | `string` | ID | `CFG_001` .. `CFG_160` | Unique configuration identifier |
| `workload` | `string` | Categorical | `matmul`, `stream`, `quicksort`, `hash` | Target benchmark workload |
| `fetch_width` | `integer` | Insts/cycle | `2, 4, 8` | Pipeline fetch stage width |
| `decode_width` | `integer` | Insts/cycle | `2, 4, 8` | Instruction decode width |
| `issue_width` | `integer` | Insts/cycle | `2, 4, 8` | Reservation station issue width |
| `commit_width` | `integer` | Insts/cycle | `2, 4, 8` | Reorder buffer commit width |
| `rob_entries` | `integer` | Count | `64, 96, 128, 192, 256` | Reorder Buffer (ROB) capacity |
| `iq_entries` | `integer` | Count | `32, 48, 64, 96` | Instruction Queue entries |
| `lq_entries` | `integer` | Count | `16, 32, 48, 64` | Load Queue entries |
| `sq_entries` | `integer` | Count | `16, 32, 48, 64` | Store Queue entries |
| `phys_int_regs` | `integer` | Count | `128, 192, 256` | Physical integer register count |
| `phys_fp_regs` | `integer` | Count | `128, 192, 256` | Physical floating-point register count |
| `branch_predictor` | `string` | Categorical | `TournamentBP`, `BiModeBP`, `LocalBP` | Branch predictor algorithm |
| `l1d_size_kb` | `integer` | Kilobytes | `16, 32, 64, 128` | L1 Data Cache capacity |
| `l1d_assoc` | `integer` | Ways | `2, 4, 8` | L1 Data Cache associativity |
| `l1i_size_kb` | `integer` | Kilobytes | `16, 32, 64` | L1 Instruction Cache capacity |
| `l1i_assoc` | `integer` | Ways | `2, 4, 8` | L1 Instruction Cache associativity |
| `l2_size_kb` | `integer` | Kilobytes | `256, 512, 1024, 2048` | Unified L2 Cache capacity |
| `l2_assoc` | `integer` | Ways | `4, 8, 16` | L2 Cache associativity |
| `clk_freq_ghz` | `float` | GHz | `1.5, 2.0, 2.4, 3.0` | Core clock frequency |

### 2.2 Target Performance Metrics ($Y$)

| Column Name | Data Type | Formula / gem5 Stats Source | Description |
| :--- | :--- | :--- | :--- |
| `sim_seconds` | `float` | `simSeconds` | Total simulated execution time in seconds |
| `sim_insts` | `integer` | `simInsts` | Total simulated instructions committed |
| `num_cycles` | `integer` | `board.processor.cores.core.numCycles` | Total processor clock cycles elapsed |
| `ipc` | `float` | `board.processor.cores.core.ipc` (or `simInsts / numCycles`) | Primary target: Instructions per Cycle |
| `cpi` | `float` | `1.0 / ipc` | Cycles per Instruction |
| `l1d_misses` | `integer` | `l1d-cache.overall_misses::total` | Total L1 data cache misses |
| `l1d_mpki` | `float` | `(l1d_misses * 1000.0) / sim_insts` | L1 Data Cache Misses Per Kilo-Instructions |
| `l2_misses` | `integer` | `l2-cache.overall_misses::total` | Total L2 cache misses |
| `l2_mpki` | `float` | `(l2_misses * 1000.0) / sim_insts` | L2 Cache Misses Per Kilo-Instructions |
| `branch_mispred` | `integer` | `branchPred.condIncorrect` | Count of mispredicted conditional branches |
| `branch_mispred_rate` | `float` | `condIncorrect / (condIncorrect + condCorrect)` | Branch misprediction ratio |
| `dram_reads` | `integer` | `mem_ctrls.readReqs` | DRAM read memory requests |
| `dram_writes` | `integer` | `mem_ctrls.writeReqs` | DRAM write memory requests |
| `dram_total_reqs` | `integer` | `dram_reads + dram_writes` | Total memory controller transactions |

---

## 3. Candidate Screening Schema (`dataset/top5_candidates.json`)
The candidate screening table holds the unseen candidate microarchitectures evaluated by the surrogate model:

```json
[
  {
    "candidate_id": "CAND_0412",
    "workload": "matmul",
    "params": {
      "fetch_width": 8,
      "rob_entries": 256,
      "l1d_size_kb": 128,
      "l1d_assoc": 8,
      "l2_size_kb": 2048,
      "l2_assoc": 16,
      "clk_freq_ghz": 3.0,
      "branch_predictor": "TournamentBP"
    },
    "predicted_metrics": {
      "predicted_ipc": 2.391,
      "predicted_l1d_mpki": 3.12,
      "predicted_l2_mpki": 0.45
    },
    "hardware_cost_proxy": 82.5,
    "pareto_rank": 1
  }
]
```

---

## 4. Ground-Truth Validation Schema (`dataset/validation_results.csv`)

| Column | Type | Description |
| :--- | :--- | :--- |
| `candidate_id` | `string` | Top candidate identifier |
| `workload` | `string` | Benchmark tested |
| `predicted_ipc` | `float` | IPC predicted by ML model in <1 ms |
| `simulated_ipc` | `float` | IPC measured via actual gem5 simulation |
| `absolute_error` | `float` | $\|\text{predicted\_ipc} - \text{simulated\_ipc}\|$ |
| `percentage_error` | `float` | $(\text{absolute\_error} / \text{simulated\_ipc}) \times 100\%$ |
| `sim_wallclock_sec` | `float` | Host CPU runtime taken by gem5 run |
| `ml_inference_sec` | `float` | Host CPU runtime taken by ML model prediction |

---

## 5. Model Serialization & Artifact Format
Trained models are packaged in `ml/saved_models/` using `joblib` accompanied by a metadata JSON file:
```
ml/saved_models/
  ├── best_model_ipc.joblib       # Serialized scikit-learn Pipeline (Scaler + Regressor)
  ├── model_metadata.json         # Training timestamp, git commit, hyperparameters, CV metrics
  └── feature_names.json          # Ordered list of feature names post-one-hot encoding
```
