# Manual Execution Guide: Commands Reference

This document provides a clean, sequential list of all commands to run the entire project manually from start to finish in **Ubuntu 24.04 (WSL)**.

---

## 📌 Prerequisites Check

Ensure you are in the project root directory in your Ubuntu terminal:

```bash
cd /home/ashrit/gem5/gem5-ml-perf-prediction
```

Ensure all Python packages are installed:

```bash
pip install -r requirements.txt --break-system-packages
```

---

## 🧪 Step 0: Run Unit Tests & Sanity Check

Runs the automated unit test suite verifying design space generation, remainder distribution, stats parsing, and candidate screening:

```bash
python3 -m unittest discover tests -v
```

---

## 🚀 Step-by-Step Manual Workflow

### Step 1: Compile the ARM64 Benchmark Binaries

Cross-compiles the 4 benchmark programs (`matmul`, `stream`, `quicksort`, `hash`) into static AArch64 executables:

```bash
make -C benchmarks
```

*Verification check (all 4 should show `ARM aarch64, statically linked`):*
```bash
file benchmarks/*-arm
```

---

### Step 2: Generate the Design Space (Latin Hypercube Sampling)

Generates 180 Latin Hypercube Sampled configurations and splits them into 3 shards for team distribution:

```bash
python3 scripts/generate_design_space.py --samples 180 --shards 3 --seed 42 --output-dir configs
```

Outputs created:
- `configs/sweep_configs.json` (Master 180 configurations)
- `configs/shards/shard_1.json` (60 configs for Laptop 1)
- `configs/shards/shard_2.json` (60 configs for Laptop 2)
- `configs/shards/shard_3.json` (60 configs for Laptop 3)

---

### Step 3: Run the Parallel Simulation Sweep

#### Option A: Running All 180 Configurations on One Machine
```bash
python3 scripts/run_parallel_sweep.py \
  --shard configs/sweep_configs.json \
  --gem5 /home/ashrit/gem5/build/ARM/gem5.opt \
  --script configs/arm_o3_parametric.py \
  --benchmarks benchmarks \
  --out-dir outputs \
  --workers 4
```

#### Option B: Running Shard 1 on Laptop 1 (Member A)
```bash
python3 scripts/run_parallel_sweep.py \
  --shard configs/shards/shard_1.json \
  --gem5 /home/ashrit/gem5/build/ARM/gem5.opt \
  --script configs/arm_o3_parametric.py \
  --benchmarks benchmarks \
  --out-dir outputs/shard_1 \
  --workers 4
```

*(On Laptop 2 for Member B: substitute `--shard configs/shards/shard_2.json --out-dir outputs/shard_2`)*  
*(On Laptop 3 for Member C: substitute `--shard configs/shards/shard_3.json --out-dir outputs/shard_3`)*

*Note: The runner validates that `stats.txt` exists and `simInsts > 0` before declaring success, and records wall-clock and aggregate core times into `simulation_manifest.json`.*

---

### Step 4: Harvest Stats & Build the Clean Dataset

Extracts cycle-accurate counters (`simInsts`, `ipc`, `l1d_mpki`, `l2_mpki`, `branch_mispred_rate`, `dram_total_reqs`) into a structured CSV file with strict validation and generates a quality report:

#### If you ran all 180 on one machine:
```bash
python3 scripts/parse_dataset.py \
  --configs configs/sweep_configs.json \
  --inputs outputs \
  --output dataset/dataset_cleaned.csv
```

#### If combining results from 3 team laptops:
Place the output folders from all 3 laptops into `outputs/` and run:
```bash
python3 scripts/parse_dataset.py \
  --configs configs/sweep_configs.json \
  --inputs outputs/shard_1 outputs/shard_2 outputs/shard_3 \
  --output dataset/dataset_cleaned.csv
```

Outputs created:
- `dataset/dataset_cleaned.csv`
- `dataset/dataset_cleaned.csv.sha256` (Integrity checksum)
- `dataset/dataset_quality_report.json` (Rejection and counter summary)

*Verification check:*
```bash
head -n 5 dataset/dataset_cleaned.csv
wc -l dataset/dataset_cleaned.csv
```

---

### Step 5: Train, Cross-Validate & Evaluate Models

Partitions data into 80% train and 20% untouched test split (stratified by workload). Performs 5-Fold Cross-Validation on the training partition, Leave-One-Workload-Out validation, computes 95% bootstrap confidence intervals, and permutation importance on the test set:

```bash
python3 ml/train_models.py \
  --dataset dataset/dataset_cleaned.csv \
  --output-dir ml/saved_models \
  --target ipc
```

Outputs created:
- `ml/saved_models/best_model_ipc.joblib` (Trained pipeline)
- `ml/saved_models/model_metadata.json` (Full metrics, 95% CIs, workload generalization)
- `ml/saved_models/oof_predictions.csv` (Out-of-fold cross-validation predictions)
- `ml/saved_models/test_predictions.csv` (Held-out test set predictions)
- `visualizations/feature_importance.png` (Permutation importance on test set)
- `visualizations/predicted_vs_actual.png` (Predicted vs actual scatter on test set)

---

### Step 6: Screen 5,000 Candidates & Multi-Objective Pareto Optimization

Evaluates 5,000 virtual candidate designs in milliseconds, calculates epistemic uncertainty across trees ($\sigma$), Upper Confidence Bound ($\text{UCB} = \mu + 1.5\sigma$), Area/Power/Energy proxies, maps global and per-workload Pareto frontiers, and exports top candidates:

```bash
python3 ml/predict_and_screen.py \
  --model ml/saved_models/best_model_ipc.joblib \
  --candidates 5000 \
  --top-k 5 \
  --kappa 1.5 \
  --output dataset/top5_candidates.json
```

Outputs created:
- `dataset/top5_candidates.json`
- `visualizations/pareto_frontier.png`

---

### Step 7 (Optional / Advanced): Active Learning Query Selection

Identifies the top informative candidate configurations with highest predictive uncertainty from an unlabeled candidate pool:

```bash
python3 ml/active_learning.py \
  --model ml/saved_models/best_model_ipc.joblib \
  --pool-size 2000 \
  --queries 15 \
  --strategy uncertainty \
  --output configs/active_queries.json
```

Outputs created:
- `configs/active_queries.json`

---

### Step 8: Ground-Truth gem5 Validation & Speedup Proof

Simulates the Top-5 predicted microarchitectures in cycle-accurate gem5, verifies prediction error ($\le 7\%$), and quantifies defensible host wall-clock speedup and aggregate core-hour reduction:

```bash
python3 ml/validate_and_evaluate.py \
  --candidates dataset/top5_candidates.json \
  --gem5 /home/ashrit/gem5/build/ARM/gem5.opt \
  --script configs/arm_o3_parametric.py \
  --benchmarks benchmarks \
  --output dataset/validation_results.csv \
  --inputs outputs
```

Outputs created:
- `dataset/validation_results.csv`
- `dataset/validation_summary.json`
- `visualizations/speedup_comparison.png`

---

## 🧹 Maintenance Commands

### Copy Generated Visualizations & Datasets to Windows File Explorer:
```bash
cp -r visualizations/* /mnt/c/Users/ashri/Documents/antigravity/dazzling-pythagoras/gem5-ml-perf-prediction/visualizations/
cp -r dataset/* /mnt/c/Users/ashri/Documents/antigravity/dazzling-pythagoras/gem5-ml-perf-prediction/dataset/
cp -r ml/saved_models/* /mnt/c/Users/ashri/Documents/antigravity/dazzling-pythagoras/gem5-ml-perf-prediction/ml/saved_models/
```

### Clean All Generated Simulation Outputs (Reset):
```bash
rm -rf outputs/* dataset/* visualizations/* ml/saved_models/* benchmarks/*-arm configs/active_queries.json
```
