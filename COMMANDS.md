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
python3 scripts/generate_design_space.py --samples 180 --shards 3 --output-dir configs
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

---

### Step 4: Harvest Stats & Build the Clean Dataset

Extracts cycle-accurate counters (`simInsts`, `ipc`, `l1d_mpki`, `l2_mpki`, `branch_mispred_rate`, `dram_total_reqs`) into a structured CSV file:

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

*Verification check:*
```bash
head -n 5 dataset/dataset_cleaned.csv
wc -l dataset/dataset_cleaned.csv
```

---

### Step 5: Train & Cross-Validate Machine Learning Models

Trains Ridge, Random Forest, and HistGradientBoosting regressors under 5-Fold Cross-Validation, selects the champion model, and generates evaluation plots:

```bash
python3 ml/train_models.py \
  --dataset dataset/dataset_cleaned.csv \
  --output-dir ml/saved_models \
  --target ipc
```

Outputs created:
- `ml/saved_models/best_model_ipc.joblib`
- `ml/saved_models/model_metadata.json`
- `visualizations/feature_importance.png`
- `visualizations/predicted_vs_actual.png`

---

### Step 6: Screen 5,000 Candidates & Find Pareto-Optimal Designs

Evaluates 5,000 virtual candidate designs in $<100\text{ ms}$, calculates the hardware complexity proxy score, maps the Pareto frontier, and exports the Top-5 designs:

```bash
python3 ml/predict_and_screen.py \
  --model ml/saved_models/best_model_ipc.joblib \
  --candidates 5000 \
  --top-k 5 \
  --output dataset/top5_candidates.json
```

Outputs created:
- `dataset/top5_candidates.json`
- `visualizations/pareto_frontier.png`

---

### Step 7: Ground-Truth gem5 Validation & Speedup Proof

Simulates the Top-5 predicted microarchitectures in cycle-accurate gem5, verifies prediction accuracy ($\le 7\%$ error), and quantifies overall simulation speedup ($\ge 25\times$):

```bash
python3 ml/validate_and_evaluate.py \
  --candidates dataset/top5_candidates.json \
  --gem5 /home/ashrit/gem5/build/ARM/gem5.opt \
  --script configs/arm_o3_parametric.py \
  --benchmarks benchmarks \
  --output dataset/validation_results.csv
```

Outputs created:
- `dataset/validation_results.csv`
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
rm -rf outputs/* dataset/* visualizations/* ml/saved_models/* benchmarks/*-arm
```
