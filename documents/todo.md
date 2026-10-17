# Project Implementation Checklist (`todo.md`)

## Status Legend
- `[ ]` Not Started
- `[/]` In Progress
- `[x]` Completed & Verified

---

## Phase 1: Environment & Benchmark Suite Setup
- [ ] Create project directories (`benchmarks/`, `configs/`, `scripts/`, `ml/`, `dataset/`, `outputs/`, `visualizations/`) in `/home/ashrit/gem5/project`.
- [ ] Implement `benchmarks/matmul.c` (Compute-bound floating-point matrix multiplication, calibrated bounds).
- [ ] Implement `benchmarks/stream.c` (Memory-bound vector triad).
- [ ] Implement `benchmarks/quicksort.c` (Branch-heavy recursive sorting).
- [ ] Implement `benchmarks/hash.c` (Integer ALU and bit-manipulation hash).
- [ ] Create `benchmarks/Makefile` with `aarch64-linux-gnu-gcc -O2 -static`.
- [ ] Compile benchmarks and verify AArch64 static ELF binary signatures.

---

## Phase 2: Parametric gem5 ARM O3 Simulation Engine
- [ ] Develop `configs/arm_o3_parametric.py` wrapping gem5's `ArmO3CPU`.
- [ ] Expose pipeline parameters: Fetch/Decode/Issue/Commit width (2, 4, 8).
- [ ] Expose Out-of-Order structures: ROB (64-256), IQ (32-96), LQ/SQ, PhysRegs.
- [ ] Expose 2-level cache hierarchy: L1D (16K-128K, assoc 2-8), L1I (16K-64K), L2 (256K-2M, assoc 4-16).
- [ ] Expose clock frequency (1.5GHz to 3.0GHz) and branch predictors (Tournament, BiMode, Local).
- [ ] Execute single-run smoke test on `matmul-arm` and confirm `stats.txt` generates with valid counters.

---

## Phase 3: Latin Hypercube Sampling & Parallel Sweep Runner
- [ ] Implement `scripts/generate_design_space.py` using Latin Hypercube Sampling (LHS).
- [ ] Generate 160 balanced configuration points across the 4 workloads.
- [ ] Implement `scripts/run_parallel_sweep.py` with multi-worker process pool.
- [ ] Integrate 300-second execution watchdog and individual log directory isolation (`outputs/run_{id}/`).
- [ ] Execute parallel sweep over 160 configurations with real-time progress monitoring.

---

## Phase 4: Stat Extraction & Feature Store Pipeline
- [ ] Implement `scripts/parse_dataset.py` to harvest `stats.txt` files.
- [ ] Extract primary targets: `ipc`, `cpi`, `sim_seconds`, `l1d_mpki`, `l2_mpki`, `branch_mispred_rate`, `dram_total_reqs`.
- [ ] Verify dataset completeness and generate `dataset/dataset_cleaned.csv` and SHA-256 manifest.

---

## Phase 5: Machine Learning Model Exploration & Cross-Validation
- [ ] Implement `ml/train_models.py` with Scikit-Learn / LightGBM pipeline.
- [ ] Train Baseline Linear & Ridge Regression models.
- [ ] Train Random Forest Regressor and Gradient Boosted Trees (`HistGradientBoostingRegressor`).
- [ ] Perform 5-fold cross-validation and compute $R^2$, MAPE, RMSE, and Spearman rank correlation ($\rho$).
- [ ] Generate feature importance plots (`visualizations/feature_importance.png`) and actual vs predicted scatter plots (`visualizations/predicted_vs_actual.png`).
- [ ] Save best trained model pipeline to `ml/saved_models/best_model_ipc.joblib`.

---

## Phase 6: Large-Scale Screening & Top Configuration Prediction
- [ ] Implement `ml/predict_and_screen.py` generating 5,000+ candidate microarchitectural designs.
- [ ] Execute millisecond batch inference across 5,000 candidates using the saved surrogate model.
- [ ] Compute hardware complexity proxy score and map the Pareto frontier.
- [ ] Select and export top-5 predicted optimal configurations to `dataset/top5_candidates.json`.

---

## Phase 7: Ground-Truth gem5 Validation & Simulation Speedup Report
- [ ] Implement `ml/validate_and_evaluate.py`.
- [ ] Execute cycle-accurate gem5 simulations on the top-5 predicted configurations.
- [ ] Compute Top-5 prediction error ($\text{MAPE} \le 7\%$).
- [ ] Calculate total simulation time reduction and compute speedup factor ($\ge 30\times$).
- [ ] Generate comprehensive final summary tables and plots for research submission.
