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
- [x] Expose pipeline parameters: Fetch/Decode/Issue/Commit width (2, 4, 8) with discrete width propagation.
- [x] Expose Out-of-Order structures: ROB (64-256), IQ (32-96), LQ/SQ, PhysRegs.
- [x] Expose 2-level cache hierarchy: L1D (16K-128K, assoc 2-8), L1I (16K-64K), L2 (256K-2M, assoc 4-16).
- [x] Expose clock frequency (1.5GHz to 3.0GHz) and branch predictors (Tournament, BiMode, Local).
- [x] Execute single-run smoke test on `matmul-arm` and confirm `stats.txt` generates with valid counters.

---

## Phase 3: Latin Hypercube Sampling & Parallel Sweep Runner
- [x] Implement `scripts/generate_design_space.py` using Latin Hypercube Sampling (LHS) and shard splitting without dropping remainder samples.
- [x] Add `--seed` parameter and candidate deduplication to design space generator.
- [x] Implement `scripts/run_parallel_sweep.py` with multi-worker process pool, discrete widths propagation, and wallclock/CPU time manifest logging.
- [x] Harden run verification: require valid non-empty `stats.txt` and `simInsts > 0` before declaring success.
- [x] Integrate 300-second execution watchdog and individual log directory isolation (`outputs/run_{id}/`).

---

## Phase 4: Stat Extraction & Feature Store Pipeline
- [x] Implement `scripts/parse_dataset.py` to harvest `stats.txt` files with strict counter verification.
- [x] Reject simulations missing mandatory counters (`simInsts`, `numCycles`, `simSeconds`) or with non-positive values.
- [x] Generate comprehensive quality report (`dataset/dataset_quality_report.json`) tracking scanned, valid, invalid runs, and rejection reasons.
- [x] Verify dataset completeness and generate `dataset/dataset_cleaned.csv` and SHA-256 manifest.

---

## Phase 5: Machine Learning Model Exploration & Cross-Validation
- [x] Implement `ml/train_models.py` with Scikit-Learn pipeline.
- [x] Partition dataset into 80% train and 20% untouched held-out test split (stratified by workload).
- [x] Perform 5-fold cross-validation strictly on the training partition for model comparison.
- [x] Evaluate cross-workload generalization using GroupKFold (Leave-One-Workload-Out).
- [x] Save out-of-fold (OOF) predictions (`ml/saved_models/oof_predictions.csv`).
- [x] Compute 95% bootstrap confidence intervals for test set $R^2$, MAPE, RMSE, MAE, and Spearman $\rho$.
- [x] Compute permutation feature importance strictly on held-out test data (`visualizations/feature_importance.png`).
- [x] Export champion model pipeline to `ml/saved_models/best_model_ipc.joblib` and comprehensive `model_metadata.json`.

---

## Phase 6: Large-Scale Screening & Top Configuration Prediction
- [x] Implement `ml/predict_and_screen.py` generating 5,000 candidate microarchitectural designs.
- [x] Compute epistemic uncertainty bands ($\sigma$) across random forest ensemble trees and Upper Confidence Bound ($\text{UCB} = \mu + \kappa \cdot \sigma$).
- [x] Formulate multi-objective optimization: Area/Complexity Proxy and Dynamic Power/Energy Proxy.
- [x] Extract global and per-workload Pareto frontiers (`visualizations/pareto_frontier.png`).
- [x] Select and export top-k predicted optimal configurations to `dataset/top5_candidates.json`.

---

## Phase 7: Active Learning & Adaptive Exploration
- [x] Implement `ml/active_learning.py` supporting uncertainty sampling, UCB exploration, and random baseline.
- [x] Select most informative queries from unlabeled candidate pool for subsequent gem5 simulation iterations.

---

## Phase 8: Ground-Truth gem5 Validation & Simulation Speedup Report
- [x] Implement `ml/validate_and_evaluate.py` with discrete width propagation.
- [x] Execute cycle-accurate gem5 simulations on the top predicted configurations.
- [x] Differentiate between host wall-clock time and aggregate core-hours in exploration speedup quantification.
- [x] Measure actual DoE run durations from manifests to eliminate average-based time approximations.
- [x] Generate validation summary (`dataset/validation_summary.json`), validation table (`dataset/validation_results.csv`), and speedup chart (`visualizations/speedup_comparison.png`).

---

## Phase 9: Automated Testing & Continuous Integration
- [x] Implement comprehensive unit test suite in `tests/test_pipeline.py`.
- [x] Add GitHub Actions CI workflow in `.github/workflows/ci.yml`.
