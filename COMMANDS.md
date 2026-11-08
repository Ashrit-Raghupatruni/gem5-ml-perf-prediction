# Manual Execution Guide: Commands Reference

This document provides a clean, sequential list of all commands to run the entire project manually from start to finish in **Ubuntu 24.04 (WSL)** with a **fully user-configurable sample count**.

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

Runs the automated unit test suite verifying design space generation across arbitrary sample counts (e.g. 100, 180, 300, 501, 2000), remainder distribution, stats parsing, and candidate screening:

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

Generates a fully user-configurable number of Latin Hypercube Sampled configurations (e.g., 300, 500, or 2000) with independent parameter decoupling ($d=17$ orthogonal dimensions) and fair remainder distribution across workloads into a single master sweep configuration:

```bash
# Example A: 300 configurations (75 per workload)
python3 scripts/generate_design_space.py --samples 300 --seed 42 --output-dir configs

# Example B: 500 configurations (125 per workload)
python3 scripts/generate_design_space.py --samples 500 --seed 42 --output-dir configs

# Example C: 2000 configurations for high-fidelity production exploration (500 per workload)
python3 scripts/generate_design_space.py --samples 2000 --seed 42 --output-dir configs
```

Outputs created:
- `configs/sweep_configs.json` (Master configurations file matching the requested count)

---

### Step 3: Run the Parallel Simulation Sweep

Runs the full design space sweep using parallel worker processes:
```bash
python3 scripts/run_parallel_sweep.py \
  --configs configs/sweep_configs.json \
  --gem5 /home/ashrit/gem5/build/ARM/gem5.opt \
  --script configs/arm_o3_parametric.py \
  --benchmarks benchmarks \
  --out-dir outputs \
  --workers 4
```

*Execution Details & Workers:*
- `--workers 4`: Spawns 4 parallel concurrent worker processes executing gem5 simulations across CPU cores simultaneously. For an 8-core CPU, you can scale to `--workers 8` to accelerate sweep execution without oversubscribing the host.
- The runner automatically verifies that `stats.txt` exists and `simInsts > 0` before recording completion in `outputs/simulation_manifest.json`.

---

### Step 4: Harvest Stats & Build Clean Feature and Target Datasets

Extracts cycle-accurate counters (`simInsts`, `ipc`, `l1d_mpki`, `l2_mpki`, `branch_mispred_rate`, `dram_total_reqs`) into structured CSV files with strict validation. It automatically generates the unified dataset as well as separate feature and output CSVs:

```bash
python3 scripts/parse_dataset.py \
  --configs configs/sweep_configs.json \
  --inputs outputs \
  --output dataset/dataset_cleaned.csv
```

Outputs created:
- Unified dataset: `dataset/dataset_cleaned.csv` and integrity checksum `dataset/dataset_cleaned.csv.sha256`
- Separated input features: `dataset/dataset_features.csv` and `outputs/features.csv` (19 features + `config_id`)
- Separated target outputs: `dataset/dataset_outputs.csv` and `outputs/outputs.csv` (15 execution metrics + `config_id`)
- Summary quality reports: `dataset/dataset_quality_report.json` and `outputs/dataset_quality_report.json`

*Verification check:*
```bash
head -n 5 outputs/features.csv
head -n 5 outputs/outputs.csv
wc -l outputs/features.csv outputs/outputs.csv dataset/dataset_cleaned.csv
```

---

### Step 5: Train, Cross-Validate & Evaluate Models

Partitions data into 80% train and 20% untouched test split (stratified by workload). Evaluates the four core models (KNN, Random Forest, HistGradientBoosting, XGBoost) under 5-Fold Cross-Validation, Leave-One-Workload-Out validation, computes 95% bootstrap confidence intervals, and permutation importance on the test set:

```bash
python3 ml/train_models.py \
  --dataset dataset/dataset_cleaned.csv \
  --output-dir ml/saved_models \
  --target ipc
```

Outputs created:
- `ml/saved_models/best_model_ipc.joblib` (Trained champion pipeline)
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
- `dataset/validation_results.csv` and `outputs/validation_results.csv`
- `dataset/validation_summary.json` and `outputs/validation_summary.json`
- `visualizations/speedup_comparison.png`

---

## 🧹 Maintenance Commands

### Copy Generated Visualizations & Datasets to Windows File Explorer:
```bash
cp -r outputs/* /mnt/c/Users/ashri/Documents/antigravity/dazzling-pythagoras/gem5-ml-perf-prediction/outputs/
cp -r visualizations/* /mnt/c/Users/ashri/Documents/antigravity/dazzling-pythagoras/gem5-ml-perf-prediction/visualizations/
cp -r dataset/* /mnt/c/Users/ashri/Documents/antigravity/dazzling-pythagoras/gem5-ml-perf-prediction/dataset/
cp -r ml/saved_models/* /mnt/c/Users/ashri/Documents/antigravity/dazzling-pythagoras/gem5-ml-perf-prediction/ml/saved_models/
```

### Clean All Generated Simulation Outputs (Reset):
```bash
rm -rf outputs/* dataset/* visualizations/* ml/saved_models/* benchmarks/*-arm configs/active_queries.json
```
