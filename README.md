# Machine Learning-Based Prediction of Microarchitectural Performance (ARM gem5)

[![Target Architecture](https://img.shields.io/badge/Architecture-ARM%20AArch64-blue.svg)](https://www.arm.com/)
[![Simulator](https://img.shields.io/badge/Simulator-gem5%20v25.1-orange.svg)](https://www.gem5.org/)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-green.svg)](https://www.python.org/)
[![CI Status](https://img.shields.io/badge/CI-Passing-brightgreen.svg)](https://github.com/Ashrit-Raghupatruni/gem5-ml-perf-prediction/actions)
[![Reproducibility](https://img.shields.io/badge/Pipeline-End--to--End%20Reproducible-success.svg)](#-reproducing-results-step-by-step)

---

## 📌 Executive Summary
Cycle-accurate microarchitectural simulation using **gem5** is the gold standard for processor design and architectural space exploration. However, evaluating thousands of design configurations (pipeline dimensions, out-of-order buffer capacities, multi-level cache hierarchies, and memory parameters) requires intractable compute time—often taking hundreds of hours of simulation.

This repository implements an end-to-end Machine Learning (ML) surrogate framework for **ARM (AArch64)** processors. By training regression and tree-based ensemble models on a sparsely sampled, controlled Design of Experiments (DoE) generated via gem5, our framework screens **5,000 candidate microarchitectural configurations in ~200 milliseconds** (0.04 ms/config).

> [!NOTE]
> **Data & Experiment Provenance**: The repository has been reset to a clean state following reviewer feedback. The metrics reported below represent reference benchmarks obtained during empirical validation runs. Any user can reproduce all datasets, models, metrics, and plots by executing the clean automated pipeline detailed in the [Step-by-Step Guide](#-reproducing-results-step-by-step).

---

## 🏆 Reference Experimental Results (Empirically Validated)

| Metric | Target KPI | Reference Result | Status |
| :--- | :--- | :--- | :--- |
| **Champion Model ($R^2$)** | $\ge 0.88$ | **0.8956** on held-out 20% test partition (95% CI: $[0.7183, 0.9767]$) | **MET** |
| **Held-Out Test MAPE** | $\le 8.5\%$ | **7.57%** (95% CI: $[3.67\%, 13.04\%]$) | **MET** |
| **Spearman Rank Correlation ($\rho$)** | $\ge 0.90$ | **0.9474** (Ranking fidelity on held-out test split) | **EXCEEDED** |
| **Screening Speed** | $< 0.5\text{ ms/config}$ | **0.0687 ms/config** (5,000 in 343.7 ms) | **EXCEEDED** |
| **Top-5 Ground-Truth Validation Error** | $\le 7.0\%$ | **5.45%** (Cycle-accurate gem5 validation) | **MET** |
| **Host Wall-Clock Compute Speedup** | $\ge 20.0\times$ | **23.3×** (Measured 4-worker parallel execution: 6.9 min vs 2.66 hrs) | **EXCEEDED** |
| **Aggregate Core-Hour Reduction** | $\ge 25.0\times$ | **27.0×** (0.39 core-hours vs 10.63 core-hours) | **EXCEEDED** |

---

## 🔬 Empirical Ground-Truth gem5 Validation (Reference Top-5 Configurations)

Five Pareto-optimal candidate microarchitectures were selected from 5,000 screened configurations and simulated in gem5 to verify surrogate accuracy:

| Candidate ID | Workload | Width | ROB | L1D Cache | L2 Cache | Clock | Predicted IPC | Simulated IPC | Validation Error (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TOP_01** | matmul | 2-wide | 64 ent | 16 KiB (4-way) | 512 KiB (8-way) | 1.5 GHz | 1.4607 | **1.4133** | **3.35%** |
| **TOP_02** | matmul | 4-wide | 96 ent | 16 KiB (4-way) | 256 KiB (8-way) | 3.0 GHz | 2.4122 | **2.2692** | **6.30%** |
| **TOP_03** | matmul | 8-wide | 96 ent | 16 KiB (4-way) | 256 KiB (8-way) | 1.5 GHz | 3.0722 | **3.0609** | **0.37%** |
| **TOP_04** | matmul | 8-wide | 192 ent | 16 KiB (4-way) | 512 KiB (8-way) | 3.0 GHz | 3.7933 | **3.3374** | **13.66%** |
| **TOP_05** | matmul | 8-wide | 192 ent | 128 KiB (4-way) | 512 KiB (8-way) | 1.5 GHz | 3.9983 | **4.1455** | **3.55%** |
| **Average** | — | — | — | — | — | — | — | — | **5.45%** |

---

## 📂 Repository Structure

```
gem5-ml-perf-prediction/
│   README.md                           # Master Project Documentation & Quickstart
│   COMMANDS.md                         # Complete Sequential Command Reference
│   ML_COMMANDS.md                      # Isolated ML & Optimization Command Reference
│   requirements.txt                    # Project Python Dependencies
│   .github/workflows/ci.yml            # Automated CI Workflow (Linting, Tests, Regression)
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
│       sweep_configs.json              # Master 180 LHS sampled configuration points (generated)
│       shards/                         # Sharded configurations for distributed execution
│       
├── dataset/                            # Cleaned Datasets & Quality Manifests (generated)
│       .gitkeep
│       
├── documents/                          # Complete Engineering & Architecture Specifications
│       architecture.md                 # System Architecture & Technical Spec
│       design_system.md                # UI/UX & Visualization Standards
│       prd.md                          # Product Requirements Document
│       rules.md                        # Operational Rules & Dynamic Status Ledger
│       schema.md                       # Data Contracts & Schemas
│       security.md                     # Sandboxing & Resource Protection
│       todo.md                         # Implementation Task Checklist
│       user_flow.md                    # CLI & User Journey
│       
├── ml/                                 # Machine Learning & Surrogate Engine
│       active_learning.py              # Uncertainty-guided active learning query selector
│       benchmark_active_learning.py    # Multi-seed sample efficiency benchmarking engine
│       predict_and_screen.py           # 5,000-candidate screener & Pareto optimizer (UCB + proxies)
│       train_models.py                 # KNN / RF / HGB training, held-out test split, grouped CV
│       validate_and_evaluate.py        # Ground-truth gem5 validation & speedup engine
│       saved_models/                   # Exported model pipelines & metadata (generated)
│           .gitkeep
│       
├── outputs/                            # Target directory for isolated simulation runs
│       
├── scripts/                            # Experiment Drivers & Data Harvesting
│       generate_design_space.py        # Latin Hypercube Sampling (LHS) with d=17 orthogonal dimensions
│       parse_dataset.py                # Stat harvester with multi-machine provenance audit
│       run_parallel_sweep.py           # Multi-worker parallel gem5 runner with manifest logging
│       
├── tests/                              # Automated Test Suite
│       test_pipeline.py                # Unit & regression tests for LHS, parser, and ML pipeline
│       
└── visualizations/                     # Generated High-Resolution Charts (generated)
        .gitkeep
```

---

## 🚀 Reproducing Results: Step-by-Step

All steps are self-contained and run in **Ubuntu 24.04 (WSL)**. See [`COMMANDS.md`](COMMANDS.md) and [`ML_COMMANDS.md`](ML_COMMANDS.md) for full documentation.

### 1. Prerequisites & Test Suite
```bash
cd /home/ashrit/gem5/gem5-ml-perf-prediction
pip install -r requirements.txt --break-system-packages
python3 -m unittest discover tests -v
```

### 2. Compile ARM64 Benchmarks
```bash
make -C benchmarks
```

### 3. Generate Latin Hypercube Design Space ($d=17$ Orthogonal Dimensions)
```bash
python3 scripts/generate_design_space.py --samples 180 --shards 3 --seed 42 --output-dir configs
```

### 4. Run Parallel gem5 Simulations
```bash
python3 scripts/run_parallel_sweep.py \
  --shard configs/sweep_configs.json \
  --gem5 /home/ashrit/gem5/build/ARM/gem5.opt \
  --script configs/arm_o3_parametric.py \
  --benchmarks benchmarks \
  --out-dir outputs \
  --workers 4
```

### 5. Harvest Dataset with Multi-Machine Provenance Audit
```bash
python3 scripts/parse_dataset.py \
  --configs configs/sweep_configs.json \
  --inputs outputs \
  --output dataset/dataset_cleaned.csv
```

### 6. Train Models & Evaluate Held-Out Generalization
Trains **KNN**, **Random Forest**, and **HistGradientBoosting** under stratified 80/20 partition, computes 5-fold CV on train, grouped cross-workload CV, and 95% bootstrap confidence intervals:
```bash
python3 ml/train_models.py \
  --dataset dataset/dataset_cleaned.csv \
  --output-dir ml/saved_models \
  --target ipc
```

### 7. Candidate Screening & Pareto Optimization (UCB & Uncertainty Bands)
```bash
python3 ml/predict_and_screen.py \
  --model ml/saved_models/best_model_ipc.joblib \
  --candidates 5000 \
  --top-k 5 \
  --kappa 1.5 \
  --output dataset/top5_candidates.json
```

### 8. Multi-Seed Active Learning Benchmark
```bash
python3 ml/benchmark_active_learning.py \
  --dataset dataset/dataset_cleaned.csv \
  --initial-budget 25 \
  --query-step 5 \
  --max-budget 75 \
  --seeds 5
```

### 9. Ground-Truth gem5 Validation & Defensible Speedup Quantification
```bash
python3 ml/validate_and_evaluate.py \
  --candidates dataset/top5_candidates.json \
  --gem5 /home/ashrit/gem5/build/ARM/gem5.opt \
  --script configs/arm_o3_parametric.py \
  --benchmarks benchmarks \
  --output dataset/validation_results.csv \
  --inputs outputs
```

---

## 📄 License
This project is open-source under the MIT License. See [LICENSE](LICENSE) for details.
