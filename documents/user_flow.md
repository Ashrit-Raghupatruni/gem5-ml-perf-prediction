# User Flow & Operational Journey

## 1. User Personas
- **Hardware Architecture Researcher / Student**: Wants to explore microarchitectural design spaces without running days of exhaustive simulation, focusing on the tradeoff between core complexity, cache sizing, and performance.
- **Compiler / Systems Engineer**: Wants fast performance estimates for target ARM ISA binaries across varied core configurations.
- **ML Engineer**: Wants clean tabular feature datasets with ground-truth cycle-level physics for surrogate regression modeling.

---

## 2. End-to-End User Journey Map

```
+----------------------------------------------------------------------------------------------------+
|                                         COMPLETE USER JOURNEY                                      |
+----------------------------------------------------------------------------------------------------+

  [Phase 1: Environment & Setup]
  $ cd /home/ashrit/gem5/project
  $ make -C benchmarks
  Outputs: benchmarks/matmul-arm, stream-arm, quicksort-arm, hash-arm
            |
            v
  [Phase 2: Single-Run Functional Smoke Test]
  $ python3 configs/arm_o3_parametric.py --test-run
  Verifies: ARM O3 CPU boots, executes benchmark, writes m5out/stats.txt with expected counters
            |
            v
  [Phase 3: Design-of-Experiments (DoE) Sampling]
  $ python3 scripts/generate_design_space.py --num-samples 160 --output configs/sweep_configs.json
  Outputs: 160 balanced Latin Hypercube sampled configuration points across 4 benchmarks
            |
            v
  [Phase 4: Parallel Simulation Sweep]
  $ python3 scripts/run_parallel_sweep.py --configs configs/sweep_configs.json --workers 6
  Monitors: Real-time progress bar, individual run logs, timeout watchdog (300s)
            |
            v
  [Phase 5: Data Harvesting & Validation]
  $ python3 scripts/parse_dataset.py --input-dir outputs/ --output dataset/dataset_cleaned.csv
  Outputs: Clean tabular CSV with 18 input features and 7 target metrics (IPC, MPKI, DRAM, etc.)
            |
            v
  [Phase 6: Machine Learning Training & Evaluation]
  $ python3 ml/train_models.py --dataset dataset/dataset_cleaned.csv
  Displays: 5-Fold Cross Validation Results (R2, MAPE, RMSE, Spearman rho)
  Saves: ml/saved_models/best_model_ipc.joblib and feature_importance.png
            |
            v
  [Phase 7: Screening 5,000 Candidates & Pareto Selection]
  $ python3 ml/predict_and_screen.py --model ml/saved_models/best_model_ipc.joblib
  Screens: 5,000+ candidate microarchitectures in <1 second
  Outputs: dataset/top5_candidates.json (Pareto-optimal top configs)
            |
            v
  [Phase 8: Ground-Truth gem5 Validation & Speedup Report]
  $ python3 ml/validate_and_evaluate.py --candidates dataset/top5_candidates.json
  Runs: 5 real gem5 simulations for ground-truth confirmation
  Reports: Actual vs Predicted IPC, Validation Error (MAPE), and Simulation Speedup Ratio (~40x)
+----------------------------------------------------------------------------------------------------+
```

---

## 3. Detailed Command-Line Workflows & Interactions

### Step 1: Workload Compilation
```bash
cd /home/ashrit/gem5/project
make -C benchmarks
```
- **Expected Outcome**: Generates 4 statically compiled AArch64 binaries.
- **Verification**: `file benchmarks/*-arm` confirms `ELF 64-bit LSB executable, ARM aarch64, statically linked`.

### Step 2: Sampling Space Generation
```bash
python3 scripts/generate_design_space.py \
  --num-samples 160 \
  --output configs/sweep_configs.json \
  --seed 42
```
- Generates Latin Hypercube samples covering pipeline widths, ROB sizes, L1/L2 caches, clock frequencies, and branch predictors.
- Verifies that parameter combinations are microarchitecturally legal (e.g., $LQ + SQ \le ROB$, $CommitWidth \ge FetchWidth$).

### Step 3: Parallel gem5 Execution
```bash
python3 scripts/run_parallel_sweep.py \
  --configs configs/sweep_configs.json \
  --gem5 /home/ashrit/gem5/build/ARM/gem5.opt \
  --output-dir outputs/ \
  --workers 6
```
- Visual feedback:
  ```
  [1/160] (matmul) Width=4, ROB=128, L1D=64K, Freq=2.4GHz -> [COMPLETED] (Time: 8.4s)
  [2/160] (stream) Width=2, ROB=64,  L1D=16K, Freq=1.5GHz -> [COMPLETED] (Time: 11.2s)
  ...
  Progress: [========================================] 100% (160/160 Finished)
  Total Elapsed Time: 18m 42s | Success: 160 | Failures: 0
  ```

### Step 4: Dataset Parsing & Cleaning
```bash
python3 scripts/parse_dataset.py \
  --input-dir outputs/ \
  --output dataset/dataset_cleaned.csv
```
- Scans all `outputs/run_*/stats.txt`.
- Extracts `simInsts`, `numCycles`, `ipc`, `l1d.overall_misses`, `l2.overall_misses`, `branchPred.condIncorrect`, `dram.total_requests`.
- Detects missing keys or abnormal outliers and logs warnings.

### Step 5: Model Training & Cross-Validation
```bash
python3 ml/train_models.py \
  --dataset dataset/dataset_cleaned.csv \
  --target ipc \
  --cv-folds 5
```
- Trains Ridge, Random Forest, and Gradient Boosted Trees.
- Prints comparative evaluation metrics table.
- Generates `visualizations/predicted_vs_actual.png` and `visualizations/feature_importance.png`.

### Step 6: 5,000-Configuration Screening
```bash
python3 ml/predict_and_screen.py \
  --model ml/saved_models/best_model_ipc.joblib \
  --workload matmul \
  --num-candidates 5000 \
  --top-k 5
```
- Performs batch inference in ~0.08 seconds.
- Computes Pareto efficiency front (IPC vs Area Proxy).
- Emits `dataset/top5_candidates.json`.

### Step 7: Ground-Truth Validation & Speedup Proof
```bash
python3 ml/validate_and_evaluate.py \
  --candidates dataset/top5_candidates.json \
  --gem5 /home/ashrit/gem5/build/ARM/gem5.opt
```
- Executes the top-5 predicted configurations in gem5.
- Computes the prediction accuracy on previously unseen top points:
  $$\text{MAPE}_{\text{validation}} = \frac{1}{K}\sum_{i=1}^K \left|\frac{\text{IPC}_{\text{pred}} - \text{IPC}_{\text{sim}}}{\text{IPC}_{\text{sim}}}\right| \times 100\%$$
- Quantifies the simulation budget speedup:
  $$\text{Speedup} = \frac{\text{Exhaustive Simulation Hours}}{\text{DoE Simulation Hours} + \text{Inference Time} + \text{Validation Simulation Hours}}$$

---

## 4. Error Handling & Recovery Matrix

| Failure Mode | Root Cause | Automated Recovery / Fallback |
| :--- | :--- | :--- |
| **gem5 Simulation Hang / Timeout** | Workload iteration count too large or deadlock in pipeline | Watchdog terminates process at 300s, records `status=TIMEOUT` in manifest, skips stats parsing. |
| **Missing `stats.txt` Metric** | Incompatible counter name across gem5 versions | Multi-key fallback search in parser (e.g. `core.ipc` $\to$ `simInsts / numCycles`). |
| **Out-of-Memory in Worker Pool** | Too many parallel instances running large caches | Automatic concurrency throttle based on available free system RAM. |
| **Low Model Accuracy ($R^2 < 0.85$)** | High variance or missing non-linear interaction terms | Automatic feature engineering adds interaction features ($\text{L1D} \times \text{ROB}$, $\text{Width} \times \text{ROB}$). |
