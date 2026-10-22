# Project Implementation Checklist (`todo.md`)

## Status Legend
- `[ ]` Not Started
- `[/]` In Progress
- `[x]` Completed & Verified

---

## Phase 1: Environment & Benchmark Suite Setup
- [x] Create project directories (`benchmarks/`, `configs/`, `scripts/`, `ml/`, `dataset/`, `outputs/`, `visualizations/`).
- [x] Implement `benchmarks/matmul.c` (Compute-bound floating-point matrix multiplication, calibrated bounds).
- [x] Implement `benchmarks/stream.c` (Memory-bound vector triad).
- [x] Implement `benchmarks/quicksort.c` (Branch-heavy recursive sorting).
- [x] Implement `benchmarks/hash.c` (Integer ALU and bit-manipulation hash).
- [x] Create `benchmarks/Makefile` with `aarch64-linux-gnu-gcc -O2 -static`.
- [x] Compile benchmarks and verify AArch64 static ELF binary signatures.

---

## Phase 2: Parametric gem5 ARM O3 Simulation Engine
- [x] Develop `configs/arm_o3_parametric.py` wrapping gem5's `ArmO3CPU`.
- [x] Expose pipeline parameters: Fetch/Decode/Issue/Commit width (2, 4, 8).
- [x] Expose Out-of-Order structures: ROB (64-256), IQ (32-96), LQ/SQ, PhysRegs.
- [x] Expose 2-level cache hierarchy: L1D (16K-128K, assoc 2-8), L1I (16K-64K), L2 (256K-2M, assoc 4-16).
- [x] Expose clock frequency (1.5GHz to 3.0GHz) and branch predictors (Tournament, BiMode, Local).
- [x] Execute single-run smoke test on `matmul-arm` and confirm `stats.txt` generates with valid counters.

---

## Phase 3: Latin Hypercube Sampling & Parallel Sweep Runner
- [x] Implement `scripts/generate_design_space.py` using Latin Hypercube Sampling (LHS) and shard splitting.
- [x] Generate 180 balanced configuration points across the 4 workloads split into 3 shards.
- [x] Implement `scripts/run_parallel_sweep.py` with multi-worker process pool.
- [x] Integrate 300-second execution watchdog and individual log directory isolation (`outputs/run_{id}/`).
- [x] Execute parallel sweep with real-time progress monitoring.

---

## Phase 4: Stat Extraction & Feature Store Pipeline
- [x] Implement `scripts/parse_dataset.py` to harvest `stats.txt` files.
- [x] Extract primary targets: `ipc`, `cpi`, `sim_seconds`, `l1d_mpki`, `l2_mpki`, `branch_mispred_rate`, `dram_total_reqs`.
- [x] Verify dataset completeness and generate `dataset/dataset_cleaned.csv` (129 rows) and SHA-256 manifest.

---

## Phase 5: Machine Learning Model Exploration & Cross-Validation
- [x] Implement `ml/train_models.py` with Scikit-Learn pipeline.
- [x] Train Baseline Linear Ridge Regression model ($R^2 = 0.6716$, $\rho = 0.7728$).
- [x] Train Random Forest Regressor ($R^2 = 0.9388$, $\text{MAPE} = 8.64\%$, $\rho = 0.9418$).
- [x] Train Histogram Gradient Boosted Trees ($R^2 = 0.9184$, $\text{MAPE} = 10.00\%$, $\rho = 0.9264$).
- [x] Perform 5-fold cross-validation and evaluate $R^2$, MAPE, RMSE, and Spearman rank correlation ($\rho$).
- [x] Generate feature importance plots (`visualizations/feature_importance.png`) and actual vs predicted scatter plots (`visualizations/predicted_vs_actual.png`).
- [x] Save champion model pipeline to `ml/saved_models/best_model_ipc.joblib` and `model_metadata.json`.

---

## Phase 6: Large-Scale Screening & Top Configuration Prediction
- [x] Implement `ml/predict_and_screen.py` generating 5,000 candidate microarchitectural designs.
- [x] Execute millisecond batch inference across 5,000 candidates in 202.7 ms (0.0405 ms/config).
- [x] Compute hardware complexity proxy score and map the Pareto frontier (`visualizations/pareto_frontier.png`).
- [x] Select and export top-5 predicted optimal configurations to `dataset/top5_candidates.json`.

---

## Phase 7: Ground-Truth gem5 Validation & Simulation Speedup Report
- [x] Implement `ml/validate_and_evaluate.py`.
- [x] Execute cycle-accurate gem5 simulations on the top-5 predicted configurations.
- [x] Compute Top-5 prediction error (**2.57% Mean MAPE**, well under the $\le 7\%$ threshold).
- [x] Calculate total simulation time reduction and compute speedup factor (**26.9× speedup**: 12.11 hours down to 0.45 hours).
- [x] Generate validation comparison table (`dataset/validation_results.csv`) and speedup chart (`visualizations/speedup_comparison.png`).
