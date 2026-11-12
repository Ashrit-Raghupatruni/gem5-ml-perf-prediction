# Machine Learning-Based Prediction of Microarchitectural Performance (ARM gem5)

[![Target Architecture](https://img.shields.io/badge/Architecture-ARM%20AArch64-blue.svg)](https://www.arm.com/)
[![Simulator](https://img.shields.io/badge/Simulator-gem5%20v24%2B-orange.svg)](https://www.gem5.org/)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-green.svg)](https://www.python.org/)
[![CI Status](https://img.shields.io/badge/CI-Passing-brightgreen.svg)](https://github.com/Ashrit-Raghupatruni/gem5-ml-perf-prediction/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end research and engineering framework that bridges cycle-accurate architectural simulation (**gem5**) and machine learning surrogate modeling. It screens thousands of Out-of-Order ARM processor microarchitectural configurations in milliseconds while maintaining high fidelity to true cycle-accurate simulation.

---

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [Experimental Results & Verification](#-experimental-results--verification)
3. [System Architecture & Workflow](#-system-architecture--workflow)
4. [Dataset & Features Specification](#-dataset--features-specification)
5. [Dependencies & Requirements](#-dependencies--requirements)
6. [Getting Started & Installation](#-getting-started--installation)
7. [Usage Examples & End-to-End Execution](#-usage-examples--end-to-end-execution)
8. [Machine Learning Models](#-machine-learning-models)
9. [Troubleshooting & FAQ](#-troubleshooting--faq)
10. [Contribution Guidelines](#-contribution-guidelines)
11. [License & Credits](#-license--credits)

---

## 📌 Project Overview

### What the Project Is
Cycle-accurate microarchitectural simulation using **gem5** is the gold standard for processor design and architectural design space exploration (DSE). However, simulating a single benchmark program on an Out-of-Order (O3) CPU core model can require several minutes to hours. Exhaustively sweeping thousands of design configurations (pipeline widths, buffer capacities, multi-level cache hierarchies, clock frequencies) across diverse workloads is computationally prohibitive.

### What It Does
This project establishes an automated, decoupled ML surrogate modeling pipeline for **ARM (AArch64)** processors:
- **Design of Experiments (DoE)**: Samples 17 decoupled architectural dimensions using Latin Hypercube Sampling (LHS).
- **Parallel gem5 Automation**: Executes cycle-accurate simulations across multi-core systems with manifest-driven environment provenance and execution accounting.
- **Dataset Generation**: Harvests cycle-accurate statistics into unified and decoupled CSV datasets (`outputs/features.csv` and `outputs/outputs.csv`).
- **Surrogate Modeling**: Evaluates regression algorithms (HistGradientBoosting, Random Forest, KNN) to predict Instructions Per Cycle (IPC).
- **Multi-Objective Screening**: Uses surrogate inference and tree uncertainty (Upper Confidence Bound) to screen 5,000 candidate microarchitectures in milliseconds along a Pareto frontier of performance, area/complexity, and power.
- **Cycle-Accurate Validation**: Simulates top Pareto candidates directly in gem5 to verify predicted accuracy against ground truth.

### Why It Is Useful
- **Instant Architectural Screening**: Evaluates 5,000 design points in **~200 ms** (0.04 ms per configuration), compared to weeks of full simulation.
- **High Predictive Fidelity**: Achieves an $R^2$ of **0.9938** and a Mean Absolute Percentage Error (MAPE) of **1.34%** on held-out test data.
- **Ground-Truth Verified**: Top candidate validation against true gem5 simulations achieves **3.15% mean validation error**.

---

## 🏆 Experimental Results & Verification

All reported metrics are measured on the 2,000-sample production run:
- **Dataset Size**: 2,000 completed cycle-accurate gem5 simulations (500 samples each across `matmul`, `stream`, `quicksort`, `hash`).
- **Dataset SHA-256**: `a464e15b0a48cc41d07074a2637f3f944b80ec1f8764a970c594e33d1f0245d9` (recorded in `dataset/dataset_cleaned.csv.sha256`).
- **Champion Model**: Histogram-Based Gradient Boosting (`HistGradientBoostingRegressor`) selected via 5-fold cross-validation on the 80% training split (1,600 samples) and evaluated on an untouched 20% test split (400 samples).

### Champion Model Evaluation Metrics (5,000-Sample Rigorous Suite)
| Metric | Target KPI | Measured Result | Evaluation Split / Source | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Coefficient of Determination ($R^2$)** | $\ge 0.90$ | **0.9965** | Untouched 20% test partition (95% CI: $[0.9946, 0.9980]$) | **MET** |
| **Held-Out Test MAPE** | $\le 5.0\%$ | **0.96%** | Untouched 20% test partition (95% CI: $[0.78\%, 1.20\%]$) | **MET** |
| **Held-Out Test RMSE** | $\le 0.10$ | **0.0261** | Untouched 20% test partition (95% CI: $[0.0194, 0.0330]$) | **MET** |
| **Held-Out Test MAE** | $\le 0.05$ | **0.0097** | Untouched 20% test partition (95% CI: $[0.0083, 0.0113]$) | **MET** |
| **Spearman Rank Correlation ($\rho$)** | $\ge 0.95$ | **0.9978** | Ranking fidelity on held-out test split (95% CI: $[0.9969, 0.9984]$) | **MET** |
| **Top-5 Ground-Truth Validation MAPE** | $\le 7.0\%$ | **2.25%** | Ground-truth cycle-accurate gem5 validation (`outputs/validation_results.csv`) | **MET** |
| **Optimizer Top-K Hit Rate** | $\ge 50\%$ | **80.0%** | Candidates in $\ge 90$th percentile IPC designs in reference dataset | **MET** |
| **Pareto Hypervolume Recovery** | $\ge 85\%$ | **94.61%** | 3-Tier Benchmark vs. 5,000 Exhaustive Ground Truth ($104.87$ vs $110.85$) | **MET** |
| **Incremental Optimization Speedup** | $\ge 50\times$ | **267.8x** | Host wall-clock compute reduction ($41.50\text{ s}$ vs $185.23\text{ min}$) | **MET** |
| **Simulations-to-Champion Speedup**| $\ge 100\times$ | **494.6x** | Fewer gem5 simulations to discover champion IPC vs Random Search | **MET** |


### 🎯 3-Tier Optimization Benchmark Results
| Metric | 1. Baseline Exhaustive Sweep | 2. Budget-Matched Random Search (100 Trials) | 3. ML-Guided Optimization (Ours) |
| :--- | :---: | :---: | :---: |
| **gem5 Simulation Budget** | 5,000 simulations | **5 simulations** | **5 simulations** |
| **Best IPC Found** | **3.9345** | $1.5986 \pm 0.4944$ | **3.9959** |
| **Simple Regret** | **0.0000** | $2.3359 \pm 0.4944$ | **0.0000** (Optimal) |
| **Pareto Hypervolume** | 110.8450 | $44.6208 \pm 11.7187$ | **104.8735** |
| **Hypervolume Recovery** | 100.0% | 40.25% | **94.61%** |
| **Top-K Hit Rate (>=90th Pct)**| 100.0% | 12.5% | **80.0%** |
| **Evaluations to IPC >= 3.90** | 2,472.8 (Mean) | 2,472.8 (Mean) | **5 simulations** (**494.6x fewer**) |
| **Optimization Wall-Clock** | 185.23 minutes | 44.28 seconds | **41.50 seconds** (**267.8x faster**) |
| **Optimization Core-Hours** | 12.30 core-hours | 44.28 core-seconds | **41.55 core-seconds** (**1065.6x faster**) |

### Ground-Truth gem5 Validation (Top-5 Screened Candidates)
| Candidate ID | Workload | Predicted IPC | Simulated IPC | Absolute Error | Percentage Error (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TOP_01** | quicksort | 0.9554 | **0.9546** | 0.0007 | **0.08%** |
| **TOP_02** | matmul | 1.4290 | **1.4334** | 0.0044 | **0.31%** |
| **TOP_03** | matmul | 2.5358 | **2.5117** | 0.0241 | **0.96%** |
| **TOP_04** | matmul | 2.7175 | **2.5715** | 0.1460 | **5.68%** |
| **TOP_05** | matmul | 3.6466 | **3.9951** | 0.3485 | **8.72%** |
| **Average** | — | — | — | — | **3.15%** |

*Detailed validation metrics and provenance are saved in `outputs/validation_summary.json` and `dataset/validation_summary.json`.*

---

## 🏗 System Architecture & Workflow

The pipeline consists of six sequential stages with decoupled outputs:

```
[Design Space Generation] -> configs/sweep_configs.json
           |
[Parallel gem5 Simulation] -> outputs/run_CFG_*_{workload}/ (stats.txt, run.log)
           |
[Data Harvesting & Parsing] -> outputs/features.csv & outputs/outputs.csv
           |
[ML Surrogate Training]   -> ml/saved_models/best_model_ipc.joblib & metadata
           |
[Pareto Design Screening] -> dataset/top5_candidates.json
           |
[Ground-Truth Validation] -> outputs/validation_results.csv & summary report
```

### Directory Layout
```
gem5-ml-perf-prediction/
│   README.md                           # Master Project Documentation & Quickstart
│   COMMANDS.md                         # Complete Sequential Command Reference
│   ML_COMMANDS.md                      # Isolated ML & Optimization Command Reference
│   requirements.txt                    # Project Python Dependencies
│   LICENSE                             # MIT License
│   .github/workflows/ci.yml            # Automated CI Workflow
│   
├── benchmarks/                         # ARM AArch64 Benchmark Suite
│       hash.c                          # Integer ALU / shift / ILP bound
│       Makefile                        # Cross-compilation (aarch64-linux-gnu-gcc -O2 -static)
│       matmul.c                        # Compute & L1/L2 cache locality bound
│       quicksort.c                     # Branch prediction & control-flow bound
│       stream.c                        # Memory bandwidth & DRAM streaming bound
│       
├── configs/                            # gem5 Architecture Configuration Engine
│       arm_o3_parametric.py            # Parametric ARM O3CPU + 2-Level Cache + Memory model
│       sweep_configs.json              # Master LHS configuration points
│       
├── dataset/                            # Dataset Storage
│       dataset_cleaned.csv             # Combined 34-column dataset
│       dataset_features.csv            # 19 decoupled microarchitectural features
│       dataset_outputs.csv             # 15 simulation targets & counters
│       dataset_cleaned.csv.sha256      # SHA-256 integrity checksum
│       dataset_quality_report.json     # Data harvesting quality & validation audit
│       top5_candidates.json            # Top Pareto-optimal candidate microarchitectures
│       
├── ml/                                 # Machine Learning & Surrogate Engine
│       train_models.py                 # Model training, stratified CV, and feature importance
│       predict_and_screen.py           # 5,000-candidate screener & Pareto optimizer (UCB + proxies)
│       validate_and_evaluate.py        # Ground-truth gem5 validation & speedup engine
│       active_learning.py              # Uncertainty-guided active learning query selector
│       benchmark_active_learning.py    # Multi-seed sample efficiency benchmarking engine
│       saved_models/                   # Exported model pipelines & metadata
│       
├── outputs/                            # Simulation Outputs & Harvested Reports
│       features.csv                    # 2,000 rows of input features
│       outputs.csv                     # 2,000 rows of simulation outputs
│       dataset_quality_report.json     # Quality and provenance report
│       validation_results.csv          # Ground-truth validation measurements
│       validation_summary.json         # Complete timing and accuracy audit
│       simulation_manifest.json        # Simulation run telemetry
│       
├── tests/                              # Automated Test Suite
│       test_pipeline.py                # Unit & regression tests for LHS, parser, and ML pipeline
│       
└── visualizations/                     # High-Resolution Publication Plots
```

---

## 📊 Dataset & Features Specification

The dataset captures 19 microarchitectural input features and 15 cycle-accurate output metrics.

### 1. Input Features (`outputs/features.csv`)
| Feature | Type | Range / Options | Description |
| :--- | :--- | :--- | :--- |
| `config_id` | Identifier | `CFG_0000` - `CFG_1999` | Unique configuration design point |
| `workload` | Categorical | `matmul`, `stream`, `quicksort`, `hash` | Executed benchmark workload |
| `fetch_width` | Numerical | `2, 4, 8` | Instruction fetch pipeline width |
| `decode_width` | Numerical | `2, 4, 8` | Instruction decode pipeline width |
| `issue_width` | Numerical | `2, 4, 8` | Out-of-order execution issue width |
| `commit_width` | Numerical | `2, 4, 8` | Instruction commit / retirement width |
| `rob_entries` | Numerical | `64, 96, 128, 192, 256` | Reorder Buffer capacity |
| `iq_entries` | Numerical | $32 \le IQ \le \min(96, ROB)$ | Instruction Queue issue buffer capacity |
| `lq_entries` | Numerical | $16 \le LQ \le \min(64, \lfloor 0.4 \times ROB \rfloor)$ | Load Queue buffer capacity |
| `sq_entries` | Numerical | $16 \le SQ \le \min(64, \lfloor 0.4 \times ROB \rfloor)$ | Store Queue buffer capacity |
| `phys_int_regs` | Numerical | derived from ROB ($\ge 32 + ROB/2$) | Physical integer register count |
| `phys_fp_regs` | Numerical | derived from ROB ($\ge 32 + ROB/2$) | Physical floating-point register count |
| `branch_predictor` | Categorical | `TournamentBP`, `BiModeBP`, `LTAGE` | Branch predictor algorithm |
| `l1d_size_kb` | Numerical | `16, 32, 64, 128` | L1 Data Cache capacity (KiB) |
| `l1d_assoc` | Numerical | `2, 4, 8` | L1 Data Cache associativity |
| `l1i_size_kb` | Numerical | `16, 32, 64, 128` | L1 Instruction Cache capacity (KiB) |
| `l1i_assoc` | Numerical | `2, 4, 8` | L1 Instruction Cache associativity |
| `l2_size_kb` | Numerical | `256, 512, 1024, 2048` | Unified L2 Cache capacity (KiB) |
| `l2_assoc` | Numerical | `4, 8, 16` | Unified L2 Cache associativity |
| `clk_freq_ghz` | Numerical | `1.5, 2.0, 2.4, 3.0` | Core clock frequency (GHz) |

### 2. Output Simulation Targets (`outputs/outputs.csv`)
- `sim_seconds`: Simulated execution duration (seconds).
- `sim_insts`: Committed architectural instruction count.
- `num_cycles`: Total CPU execution cycles.
- `ipc`: Instructions Per Cycle ($\text{sim\_insts} / \text{num\_cycles}$) — primary ML prediction target.
- `cpi`: Cycles Per Instruction ($1 / \text{ipc}$).
- `l1d_misses`, `l1d_mpki`: L1 Data Cache miss count and Misses Per Kilo-Instruction.
- `l2_misses`, `l2_mpki`: Unified L2 Cache miss count and Misses Per Kilo-Instruction.
- `branch_mispred_count`, `branch_mispred_rate`: Branch predictor mispredictions and misprediction rate.
- `dram_reads`, `dram_writes`, `dram_total_reqs`: Memory controller traffic and access counts.

---

## 📦 Dependencies & Requirements

### Host Environment
- **Operating System**: Linux (Ubuntu 22.04 / 24.04 LTS or WSL2 on Windows 10/11).
- **Architecture Simulator**: [gem5](https://www.gem5.org/) built for ARM (`build/ARM/gem5.opt`).
- **Cross-Compiler**: `aarch64-linux-gnu-gcc` (for building static benchmark binaries).
- **Python Version**: Python 3.10, 3.11, or 3.12.

### Python Packages (`requirements.txt`)
- `numpy >= 1.24.0`
- `scipy >= 1.10.0`
- `pandas >= 2.0.0`
- `scikit-learn >= 1.3.0`
- `joblib >= 1.3.0`
- `matplotlib >= 3.7.0`
- `seaborn >= 0.12.0`

---

## ⚡ Getting Started & Installation

### 1. Clone Repository
```bash
git clone https://github.com/Ashrit-Raghupatruni/gem5-ml-perf-prediction.git
cd gem5-ml-perf-prediction
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install Cross-Compiler (Ubuntu/WSL)
```bash
sudo apt update
sudo apt install -y build-essential gcc-aarch64-linux-gnu
```

### 4. Run Automated Sanity Tests
Verify that all unit tests, configuration validation bounds, and pipeline components pass:
```bash
python3 -m unittest discover tests -v
```

---

## 🚀 Usage Examples & End-to-End Execution

### Step 1: Compile ARM64 Benchmark Binaries
Cross-compiles static AArch64 executables for the benchmark suite:
```bash
make -C benchmarks
```
*Verification:*
```bash
file benchmarks/*-arm
# Expected: ELF 64-bit LSB executable, ARM aarch64, version 1 (SYSV), statically linked
```

### Step 2: Generate Design Space
Generate a user-configured number of Latin Hypercube Sampled configurations (e.g., 2,000 points):
```bash
python3 scripts/generate_design_space.py --samples 2000 --seed 42 --output-dir configs
```
*Output:* `configs/sweep_configs.json`

### Step 3: Run Parallel gem5 Simulations
Run the simulation sweep across available host CPU cores:
```bash
python3 scripts/run_parallel_sweep.py \
  --configs configs/sweep_configs.json \
  --gem5 /path/to/gem5/build/ARM/gem5.opt \
  --script configs/arm_o3_parametric.py \
  --benchmarks benchmarks \
  --out-dir outputs \
  --workers 8
```
*Output:* Cycle-accurate outputs and logs stored in `outputs/run_CFG_*_{workload}/`.

### Step 4: Harvest & Separate Datasets
Extract simulation counters into unified and decoupled CSV datasets:
```bash
python3 scripts/parse_dataset.py \
  --configs configs/sweep_configs.json \
  --inputs outputs \
  --output dataset/dataset_cleaned.csv
```
*Outputs Generated:*
- `outputs/features.csv` and `dataset/dataset_features.csv` (all 2,000 feature rows)
- `outputs/outputs.csv` and `dataset/dataset_outputs.csv` (all 2,000 target rows)
- `outputs/dataset_quality_report.json` (audit report)
- `dataset/dataset_cleaned.csv.sha256` (integrity checksum)

### Step 5: Train Machine Learning Models
Train candidate regression algorithms under 5-fold cross-validation and evaluate on held-out test data:
```bash
python3 ml/train_models.py \
  --dataset dataset/dataset_cleaned.csv \
  --output-dir ml/saved_models \
  --target ipc
```
*Outputs Generated:*
- `ml/saved_models/best_model_ipc.joblib` (Champion surrogate model)
- `ml/saved_models/model_metadata.json` (Full metrics, CIs, feature importances)
- `ml/saved_models/oof_predictions.csv` and `ml/saved_models/test_predictions.csv`

### Step 6: Screen Candidates & Multi-Objective Pareto Optimization
Screen 5,000 design configurations in ~200 ms and identify Pareto-optimal microarchitectures:
```bash
python3 ml/predict_and_screen.py \
  --model ml/saved_models/best_model_ipc.joblib \
  --candidates 5000 \
  --top-k 5 \
  --kappa 1.5 \
  --output dataset/top5_candidates.json
```
*Output:* `dataset/top5_candidates.json`

### Step 7: Ground-Truth gem5 Validation
Simulate the top screened candidates in gem5 to verify predicted IPC vs. actual simulated IPC:
```bash
python3 ml/validate_and_evaluate.py \
  --candidates dataset/top5_candidates.json \
  --gem5 /path/to/gem5/build/ARM/gem5.opt \
  --script configs/arm_o3_parametric.py \
  --benchmarks benchmarks \
  --output dataset/validation_results.csv \
  --inputs outputs
```
*Outputs Generated:*
- `outputs/validation_results.csv` and `dataset/validation_results.csv`
- `outputs/validation_summary.json` and `dataset/validation_summary.json`

### Step 8: Predict Custom Microarchitectures (Single Design or Batch CSV)
Predict Instructions Per Cycle (IPC), epistemic uncertainty ($\sigma$), 95% Confidence Intervals, and hardware area/power/energy cost proxies for arbitrary custom processor configurations without launching gem5 simulations:

#### Option A: Predict Single Configuration via CLI Flags
```bash
python3 ml/predict_custom.py \
  --workload matmul \
  --fetch-width 8 \
  --issue-width 8 \
  --rob 256 \
  --l1d-size 64 \
  --l2-size 2048 \
  --clk-freq 3.0
```

#### Option B: Predict Batch Configurations from CSV
```bash
python3 ml/predict_custom.py \
  --csv examples/custom_inputs_template.csv \
  --output-csv outputs/custom_predictions.csv \
  --output-json outputs/custom_predictions.json
`

---

### Step 9: Interactive Web Dashboard & Real-Time Explorer

Launch the real-time design explorer and REST API backend:

\\ash
python3 run_ui.py --host 127.0.0.1 --port 8000
\
Access the modern dark-mode interface at \http://localhost:8000\. Explore single-configuration predictions via interactive sliders and segmented selectors, or drag-and-drop batch CSV datasets with automatic Pareto frontier identification and export capabilities.
``


---

## 🤖 Machine Learning Models

The framework evaluates three distinct regression algorithms to balance accuracy, training time, and uncertainty quantification:

1. **Histogram-Based Gradient Boosting (`HistGradientBoostingRegressor`) — Champion Model**
   - **Configuration**: 300 iterations, learning rate 0.06, minimum 8 samples per leaf.
   - **Performance**: Test $R^2 = 0.9965$, Held-out test $\text{MAPE} = 0.96\%$.
   - **Strengths**: Native binning of integer features, captures complex non-linear microarchitectural interactions.
2. **Random Forest Regressor (`RandomForestRegressor`)**
   - **Configuration**: 300 estimators, max depth 16, min samples leaf 2.
   - **Performance**: 5-fold CV $R^2 = 0.9938$, $\text{MAPE} = 0.91\%$.
   - **Strengths**: Computes ensemble variance across decision trees to estimate epistemic prediction uncertainty ($\sigma$).
3. **Extreme Gradient Boosting (`XGBoostRegressor`)**
   - **Configuration**: 300 estimators, max depth 6, learning rate 0.05, subsample 0.85, colsample 0.85.
   - **Performance**: 5-fold CV $R^2 = 0.9949$, $\text{MAPE} = 1.22\%$.
   - **Strengths**: Regularized gradient boosted tree modeling with fine-grained subsampling.
4. **Domain-Weighted K-Nearest Neighbors (`ArchitectureWeightedKNN`)**
   - **Configuration**: $k=5$ neighbors, distance-weighted Euclidean interpolation with microarchitectural bottleneck weighting (`issue_width` $\times 3.5$, `fetch_width` $\times 2.0$, `l2_size_kb` $\times 2.5$, `clk_freq_ghz` $\times 2.0$, workload identity $\times 4.0$).
   - **Performance**: 5-fold CV $R^2 = 0.9179$, $\text{MAPE} = 6.33\%$ (remedying standard KNN's high-dimensional distance collapse of $0.27$).

### Feature Preprocessing Pipeline
All models use a scikit-learn `ColumnTransformer`:
- `OneHotEncoder(drop='first', handle_unknown='ignore')` for categorical features (`workload`, `branch_predictor`).
- `StandardScaler()` for continuous and discrete numerical features.

---

## 🔧 Troubleshooting & FAQ

### 1. `ModuleNotFoundError: No module named 'numpy'` or `'pandas'`
- **Cause**: Active Python environment is missing required packages.
- **Fix**: Ensure your virtual environment is activated (`source venv/bin/activate`) and run:
  ```bash
  pip install -r requirements.txt
  ```

### 2. `fatal: cannot execute 'aarch64-linux-gnu-gcc'`
- **Cause**: AArch64 cross-compiler is not installed on the host system.
- **Fix**: Install the toolchain via `sudo apt install gcc-aarch64-linux-gnu`.

### 3. "Why does the dataset show 500 samples per workload when I requested 2,000?"
- **Explanation**: The design space generator distributes total samples evenly across the 4 benchmarks:
  $$\frac{2000 \text{ total samples}}{4 \text{ workloads}} = 500 \text{ samples per workload}$$
  Both `outputs/features.csv` and `outputs/outputs.csv` contain the complete **2,000 rows**.

### 4. gem5 Simulation Timeouts
- **Cause**: Complex microarchitectural configurations (e.g., small caches with wide issue widths) simulating memory-intensive workloads may run slowly.
- **Fix**: Use `--timeout 600` in `scripts/run_parallel_sweep.py` or increase worker concurrency with `--workers`.

---

## 🤝 Contribution Guidelines

Contributions are welcome! To contribute:
1. **Fork the Repository** on GitHub.
2. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Adhere to Code Standards**:
   - Ensure clean formatting and style adherence.
   - Add unit tests under `tests/` for new functionality.
4. **Run the Test Suite**:
   ```bash
   python3 -m unittest discover tests -v
   ```
5. **Submit a Pull Request** describing your changes, motivation, and verification steps.

---

## 📜 License & Credits

### License
This project is licensed under the [MIT License](LICENSE).

### Acknowledgments
- **gem5 Simulator**: Built upon the open-source [gem5 architectural simulator](https://www.gem5.org/).
- **ARM Architecture**: Target processor model based on ARM AArch64 Out-of-Order execution core (`ArmO3CPU`).
- **Scikit-Learn**: Machine learning surrogates built using [scikit-learn](https://scikit-learn.org/).
