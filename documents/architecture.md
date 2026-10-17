# System Architecture & Technical Specification

## 1. System Overview & Architecture Diagram
The architecture is structured as a decoupled 5-tier pipeline: Workload Tier, gem5 Simulation Tier, Data Aggregation Tier, ML Modeling Tier, and Validation/Optimization Tier.

```
+-----------------------------------------------------------------------------------------------------+
|                                      ARCHITECTURE PIPELINE DIAGRAM                                  |
+-----------------------------------------------------------------------------------------------------+

 [Workload Tier]
  benchmarks/*.c  --(aarch64-linux-gnu-gcc)--> Static ARM64 Binaries (matmul, stream, quicksort, hash)
                                                               |
                                                               v
 [Simulation Tier]
  scripts/generate_design_space.py  --(Latin Hypercube Sampling)--> configs_sweep.json (160 points)
                                                               |
                                                               v
  scripts/run_parallel_sweep.py <===> configs/arm_o3_parametric.py (gem5.opt ARM)
       |
       +---> outputs/run_001/stats.txt
       +---> outputs/run_002/stats.txt
       +---> ...
       +---> outputs/run_160/stats.txt
                                                               |
                                                               v
 [Data Tier]
  scripts/parse_dataset.py ---> dataset/dataset_raw.csv ---> dataset/dataset_cleaned.csv
                                                               |
                                                               v
 [ML Modeling Tier]
  ml/train_models.py
       +---> Standard Scaler & One-Hot Encoder
       +---> Ridge Regression (Baseline)
       +---> Random Forest Regressor
       +---> Gradient Boosted Trees (HistGradientBoosting / LightGBM)
       +---> 5-Fold Cross Validation ($R^2$, MAPE, RMSE, Spearman $\rho$)
       +---> Feature Importance & SHAP Ranking
       +---> Saved Model: ml/saved_models/best_model_ipc.joblib
                                                               |
                                                               v
 [Optimization & Validation Tier]
  ml/predict_and_screen.py  ---> Evaluates 5,000 Unseen Candidate Designs in <1 sec
       +---> Identifies Pareto Frontier (IPC vs Silicon Proxy Score)
       +---> Selects Top-5 Predicted Optimal Configurations
                                                               |
                                                               v
  ml/validate_and_evaluate.py <---> gem5.opt ARM Validation Runs
       +---> Simulates Top-5 Ground-Truth Configurations
       +---> Calculates Validation Error (MAPE)
       +---> Quantifies Simulation Time Reduction & Compute Speedup
+-----------------------------------------------------------------------------------------------------+
```

---

## 2. Microarchitecture Engine Details (`configs/arm_o3_parametric.py`)
The gem5 simulation model is based on the **`ArmO3CPU`** (Out-of-Order execution core) connected to a multi-level classic cache hierarchy and DDR memory controller.

### Microarchitectural Components

```
                   +-------------------------------------------------------+
                   |                     ArmO3CPU Core                     |
                   |                                                       |
                   |  [Fetch Engine] <---> [Branch Predictor (Tournament)] |
                   |         |                                             |
                   |         v                                             |
                   |  [Decode Engine] (Width: 2 / 4 / 8)                   |
                   |         |                                             |
                   |         v                                             |
                   |  [Rename Engine] <---> [Phys Regs: Int/FP 128-256]    |
                   |         |                                             |
                   |         v                                             |
                   |  [Instruction Queue] (Entries: 32 - 96)               |
                   |         |                                             |
                   |         +-------------------+                         |
                   |         |                   |                         |
                   |         v                   v                         |
                   |   [Integer ALUs]      [FP / SIMD]                     |
                   |         |                   |                         |
                   |         +-------------------+                         |
                   |         |                                             |
                   |         v                                             |
                   |  [ROB: 64-256] <---> [Load/Store Queues: LQ/SQ]       |
                   |         |                                             |
                   |         v                                             |
                   |  [Commit Engine] (Width: 2 / 4 / 8)                   |
                   +---------+---------------------------+-----------------+
                             |                           |
                  Instruction|Port                   Data|Port
                             v                           v
                      +--------------+            +--------------+
                      |  L1I Cache   |            |  L1D Cache   |
                      | 16K-64K Assoc|            | 16K-128K Asso|
                      +-------+------+            +-------+------+
                              |                           |
                              +-------------+-------------+
                                            |
                                            v
                                    +---------------+
                                    |    L2XBar     |
                                    +-------+-------+
                                            |
                                            v
                                    +---------------+
                                    |   L2 Cache    |
                                    | 256K-2M Assoc |
                                    +-------+-------+
                                            |
                                            v
                                    +---------------+
                                    |   SystemXBar  |
                                    +-------+-------+
                                            |
                                            v
                                 +---------------------+
                                 | SingleChannel DDR4  |
                                 +---------------------+
```

### Parameter Knob Interfacing
The script accepts parameters via `argparse` and assigns them to the gem5 board:
- `cpu.fetchWidth = args.width`
- `cpu.decodeWidth = args.width`
- `cpu.renameWidth = args.width`
- `cpu.issueWidth = args.width`
- `cpu.dispatchWidth = args.width`
- `cpu.wbWidth = args.width`
- `cpu.commitWidth = args.width`
- `cpu.numROBEntries = args.rob`
- `cpu.numIQEntries = args.iq`
- `cpu.numPhysIntRegs = args.phys_regs`
- `cpu.numPhysFloatRegs = args.phys_regs`
- `cpu.LQEntries = args.lq`
- `cpu.SQEntries = args.sq`
- `cpu.branchPred = TournamentBP()` or `BiModeBP()` or `LocalBP()`
- Cache parameters dynamically passed into custom `TwoLevelCacheHierarchy`.

---

## 3. Parallel Execution Framework
The driver `scripts/run_parallel_sweep.py`:
- Spawns a process pool sized to the available host logical threads (default: $N_{workers} = \min(8, \text{cpu\_count})$).
- Dispatches individual gem5 instances:
  ```bash
  /home/ashrit/gem5/build/ARM/gem5.opt -d <run_output_dir> configs/arm_o3_parametric.py [flags...]
  ```
- Collects exit codes, logs `stdout` and `stderr` to individual `run.log` files, and prevents directory collision.
- Implements a hard timeout monitor (300 seconds) per simulation to prevent runaway infinite loops or hangs.

---

## 4. Machine Learning Framework & Surrogate Modeling
The ML module `ml/train_models.py` uses `scikit-learn` with optional `lightgbm`/`xgboost`:
1. **Preprocessor Pipeline**:
   - Categorical variables (`workload`, `branch_predictor`) $\to$ `OneHotEncoder(drop='first')`.
   - Numerical microarchitectural variables $\to$ `StandardScaler()`.
2. **Algorithms**:
   - **Ridge / Lasso Regression**: Linear surrogate providing baseline interpretability.
   - **Random Forest Regressor**: 200 estimators, max depth 12, non-parametric tree ensemble handling step functions in cache boundaries.
   - **Histogram-based Gradient Boosted Trees (`HistGradientBoostingRegressor`)**: Highly efficient gradient boosting modeling complex feature interactions (e.g., L1D size $\times$ ROB entries).
3. **Evaluation Protocol**:
   - 5-Fold Stratified Cross-Validation across workloads.
   - Evaluation metrics: $R^2$, MAPE, RMSE, and Spearman rank correlation ($\rho$).
   - Export best model to `ml/saved_models/best_model_ipc.joblib`.

---

## 5. Design Space Optimizer & Validation Engine
The optimizer `ml/predict_and_screen.py`:
- Generates a synthetic Cartesian product of 5,000+ legal ARM O3 configurations.
- Runs batch inference using the saved surrogate model ($<100\text{ ms}$).
- Computes an estimated hardware complexity proxy:
  $$\text{Cost Proxy} = w_1 \cdot \text{Width} + w_2 \cdot \text{ROB} + w_3 \cdot \log_2(\text{L1D}) + w_4 \cdot \log_2(\text{L2})$$
- Extracts Pareto-optimal frontier points maximizing IPC per Cost Proxy.
- Exports top-5 candidates for each workload to `dataset/top5_candidates.csv`.
- `ml/validate_and_evaluate.py` dispatches real gem5 simulations for these 5 points, records actual IPC, and computes validation MAPE and computational speedup.
