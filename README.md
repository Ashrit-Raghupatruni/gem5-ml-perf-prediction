# Machine Learning-Based Prediction of Microarchitectural Performance (ARM gem5)

[![Target Architecture](https://img.shields.io/badge/Architecture-ARM%20AArch64-blue.svg)](https://www.arm.com/)
[![Simulator](https://img.shields.io/badge/Simulator-gem5%20v24+-orange.svg)](https://www.gem5.org/)
[![Python](https://img.shields.io/badge/Python-3.12+-green.svg)](https://www.python.org/)
[![ML Framework](https://img.shields.io/badge/ML-scikit--learn%20%7C%20LightGBM-purple.svg)](https://scikit-learn.org/)

---

## 📌 Executive Summary
Cycle-accurate microarchitectural simulation using **gem5** is the gold standard for processor design and architectural space exploration. However, evaluating thousands of design configurations (pipeline dimensions, out-of-order buffer capacities, multi-level cache hierarchies, and memory parameters) requires intractable compute time—often taking hundreds of hours of simulation.

This project implements an end-to-end Machine Learning (ML) surrogate framework for **ARM (AArch64)** processors. By training regression and tree-based ensemble models on a sparsely sampled, controlled Design of Experiments (DoE) generated via gem5, our framework screens **5,000+ candidate microarchitectural configurations in milliseconds**.

The core contribution focuses on **prediction accuracy, ranking fidelity (Spearman $\rho$), and quantifiable simulation time reduction ($\ge 30\times$)**, validated against cycle-accurate gem5 ground-truth simulations of predicted optimal processor configurations.

---

## 🎯 Key Objectives & Success Criteria
- **Target ISA**: ARM AArch64 (`ArmO3CPU` out-of-order timing model).
- **Workload Diversity**: 4 diverse benchmarks (`matmul`, `stream`, `quicksort`, `hash`).
- **Target Output Metrics**: IPC, CPI, execution time, L1D/L2 MPKI, branch misprediction rate, DRAM total requests.
- **Model Accuracy**: $R^2 \ge 0.88$, $\text{MAPE} \le 8\%$, Spearman rank correlation $\rho \ge 0.90$.
- **Simulation Time Reduction**: $\ge 30\times$ speedup over exhaustive cycle-accurate simulation.
- **Ground-Truth Validation**: Empirical gem5 verification of predicted top-5 candidate configurations with validation error $\le 7\%$.

---

## 📂 Repository Structure

```
gem5/project/
├── README.md                           # Master Project Documentation & Quickstart
├── requirements.txt                    # Python Dependencies (pandas, scikit-learn, etc.)
├── documents/                          # Complete Engineering & Architecture Specifications
│   ├── prd.md                          # Product Requirements Document
│   ├── architecture.md                 # Technical Spec & System Architecture
│   ├── user_flow.md                    # CLI Journey & Operational Workflow
│   ├── design_system.md                # Visualization, Tabular & CLI Standards
│   ├── schema.md                       # Data Contracts, Feature & Output Schemas
│   ├── security.md                     # Sandboxing, Watchdogs & Resource Protection
│   ├── todo.md                         # Implementation Plan & Task Checklist
│   └── rules.md                        # Operational Rules & Dynamic Task Ledger
├── benchmarks/                         # ARM AArch64 Benchmark Suite
│   ├── Makefile                        # Cross-compilation using aarch64-linux-gnu-gcc
│   ├── matmul.c                        # Compute & L1/L2 cache bottleneck
│   ├── stream.c                        # Memory bandwidth & DRAM bottleneck
│   ├── quicksort.c                     # Branch prediction & control-flow bottleneck
│   └── hash.c                          # Integer ALU & irregular access bottleneck
├── configs/                            # gem5 Architecture Configuration Engine
│   └── arm_o3_parametric.py            # Parametric ARM O3CPU + Cache + Memory model
├── scripts/                            # Experiment Drivers & Data Harvesting
│   ├── generate_design_space.py        # Latin Hypercube Sampling (LHS) generator
│   ├── run_parallel_sweep.py           # Multi-worker parallel gem5 sweep manager
│   └── parse_dataset.py                # Harvester extracting stats.txt into dataset.csv
├── ml/                                 # Machine Learning & Surrogate Engine
│   ├── train_models.py                 # Ridge, Random Forest, HistGBDT / LightGBM
│   ├── predict_and_screen.py           # 5,000-candidate screening & Pareto optimizer
│   ├── validate_and_evaluate.py        # Ground-truth gem5 validation & speedup report
│   └── saved_models/                   # Serialized model pipelines & metadata
├── dataset/                            # Tabular Datasets & Manifests
│   ├── dataset_cleaned.csv             # Clean feature store with 18 inputs & 7 targets
│   └── top5_candidates.json            # Pareto-optimal configurations for validation
└── visualizations/                     # 300-DPI Publication-Ready Plots
    ├── predicted_vs_actual.png         # Model correlation scatter with y=x line
    ├── feature_importance.png          # Microarchitecture knob impact ranking
    ├── pareto_frontier.png             # IPC vs Hardware Complexity frontier
    └── speedup_comparison.png          # Exhaustive vs ML-guided time comparison
```

---

## ⚙️ Prerequisites & Environment Setup

### 1. Host Requirements
- **OS**: Linux / WSL2 (Ubuntu 24.04 LTS).
- **gem5 ARM Binary**: `/home/ashrit/gem5/build/ARM/gem5.opt` (built with `scons build/ARM/gem5.opt`).
- **Cross-Compiler**: `aarch64-linux-gnu-gcc`.
- **Python**: Python 3.12+ with pip.

### 2. Python Dependencies
Install required data science packages:
```bash
pip install -r requirements.txt
```
*(Dependencies: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`, `joblib`)*

---

## 🚀 Quickstart & Workflow Walkthrough

### Step 1: Compile ARM64 Benchmarks
Cross-compile the 4 benchmark programs into static AArch64 ELF executables:
```bash
make -C benchmarks
```

### Step 2: Test gem5 ARM O3 Pipeline (Smoke Test)
Verify the parametric gem5 configuration script with a single fast run:
```bash
python3 configs/arm_o3_parametric.py --test-run
```

### Step 3: Generate Sampling Space (Latin Hypercube Sampling)
Sample 160 representative microarchitectural points covering the design space:
```bash
python3 scripts/generate_design_space.py --num-samples 160 --output configs/sweep_configs.json
```

### Step 4: Run Parallel gem5 Simulations
Execute simulations across multi-core host CPU using the process pool manager:
```bash
python3 scripts/run_parallel_sweep.py --configs configs/sweep_configs.json --workers 6
```

### Step 5: Harvest & Clean Dataset
Extract cycle-accurate counters from `stats.txt` into a structured tabular dataset:
```bash
python3 scripts/parse_dataset.py --input-dir outputs/ --output dataset/dataset_cleaned.csv
```

### Step 6: Train & Evaluate Surrogate Models
Train Ridge, Random Forest, and Gradient Boosted Trees with 5-fold cross-validation:
```bash
python3 ml/train_models.py --dataset dataset/dataset_cleaned.csv
```

### Step 7: Screen 5,000 Candidates & Identify Optimal Designs
Evaluate 5,000 candidate microarchitectures in <100 ms and extract Pareto-optimal designs:
```bash
python3 ml/predict_and_screen.py --model ml/saved_models/best_model_ipc.joblib
```

### Step 8: Validate with Ground-Truth gem5 Simulations
Simulate the predicted top-5 configurations and measure speedup:
```bash
python3 ml/validate_and_evaluate.py --candidates dataset/top5_candidates.json
```

---

## 📖 Detailed Documentation Index
All detailed engineering, architectural, and operational specifications are available in the [`documents/`](./documents) directory:

- 📋 [**Product Requirements Document (PRD)**](./documents/prd.md): Objectives, microarchitecture knobs, metrics, and KPIs.
- 🏗️ [**System Architecture & Tech Spec**](./documents/architecture.md): gem5 O3 pipeline, cache hierarchies, and ML architecture.
- 🧭 [**User Flow & CLI Journey**](./documents/user_flow.md): Step-by-step CLI execution and operational lifecycle.
- 🎨 [**Design System & Visualization**](./documents/design_system.md): Tabular formatting, CLI banners, and 300-DPI plot specs.
- 📊 [**Data Schemas & Contracts**](./documents/schema.md): Complete feature dictionaries, formulas, and JSON formats.
- 🔒 [**Security & Sandboxing**](./documents/security.md): Subprocess watchdogs, memory limits, and dataset hashing.
- ✅ [**Implementation Checklist (`todo.md`)**](./documents/todo.md): Granular phase-by-phase task tracking.
- 📜 [**Rules of Engagement & Status Ledger (`rules.md`)**](./documents/rules.md): Operational rules and live progress ledger.

---

## 🛡️ Operational Protocol Note
Per **Rule 1** in [`documents/rules.md`](./documents/rules.md), all engineers and autonomous agents are required to update `todo.md` and the dynamic Status Ledger in `rules.md` immediately following the completion of any task or phase.
