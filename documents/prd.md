# Product Requirements Document (PRD)

## Project Title
**Machine Learning-Based Prediction of Microarchitectural Performance (ARM gem5)**

---

## 1. Executive Summary & Problem Statement
In computer architecture exploration, cycle-accurate architectural simulators such as **gem5** are the gold standard for evaluating microarchitectural design tradeoffs (pipeline depth/width, out-of-order buffer sizing, cache hierarchies, and memory subsystems). However, cycle-level simulation is computationally expensive and slow: simulating a few billion instructions can take hours or days per design point. 

Given an architectural parameter space consisting of pipeline dimensions ($2, 4, 8\text{-wide}$), ROB capacities ($64\text{ to }256$), L1/L2 cache sizes and associativities, and clock frequencies across multiple benchmark workloads, exhaustive search requires testing thousands to millions of configurations—an intractable requirement for realistic design timelines.

### The Objective
This project establishes whether machine-learning (ML) regression and ensemble tree-based models can accurately predict cycle-accurate microarchitectural performance directly from architectural parameters and workload characteristics on the **ARM (AArch64)** architecture.

By training surrogate ML models on a controlled, sparsely sampled design-of-experiments (DoE) dataset generated via gem5, the surrogate model can screen tens of thousands of candidate processor configurations in milliseconds. The primary contribution of this project focuses on **prediction accuracy, ranking fidelity, and drastic simulation time reduction**, substantiated through actual gem5 validation runs of predicted optimal designs.

---

## 2. Key Objectives & Success Metrics (KPIs)

| Objective | Target Metric | Success Threshold | Justification |
| :--- | :--- | :--- | :--- |
| **Model Prediction Accuracy** | Coefficient of Determination ($R^2$) | $R^2 \ge 0.88$ | Ensures models capture non-linear parameter interactions. |
| **Prediction Error** | Mean Absolute Percentage Error (MAPE) | $\text{MAPE} \le 8.0\%$ | Guarantees predicted metrics closely track ground-truth simulation. |
| **Ranking Fidelity** | Spearman Rank Correlation ($\rho$) | $\rho \ge 0.90$ | Hardware architects care most about which design is superior. |
| **Simulation Time Reduction** | Compute Budget Speedup | $\ge 30\times$ speedup | Demonstrates quantifiable reduction over exhaustive simulation. |
| **Top-Config Validation** | Top-5 Validation Error | $\text{Error} \le 7.0\%$ | Confirms optimal predictions hold true when simulated in gem5. |

---

## 3. Workload Suite Specifications (ARM AArch64)
The dataset must capture diverse microarchitectural bottlenecks rather than uniform synthetic loops. The benchmark suite includes:

1. **`matmul` (Compute & Cache Bottleneck)**:
   - Dense double-precision matrix multiplication ($N \times N$, $N=64, 128$).
   - High compute intensity, nested loops, temporal data reuse in L1/L2.
2. **`stream` (Memory Bandwidth & DRAM Bottleneck)**:
   - Vector triad operation: $A[i] = B[i] + \alpha \cdot C[i]$.
   - Streaming memory accesses stressing DRAM controllers, fill buffers, and cache miss queues.
3. **`quicksort` (Branch & Control-Flow Bottleneck)**:
   - Recursive sorting on pseudo-random integers.
   - Irregular, data-dependent conditional branches stressing the branch predictor (TournamentBP/BiModeBP).
4. **`hash` (Integer ALU & Irregular Access Bottleneck)**:
   - Cryptographic/string hash mixing algorithms.
   - High instruction-level parallelism (ILP) with integer shifts, XORs, and small working sets.

All benchmarks are cross-compiled using `aarch64-linux-gnu-gcc -static -O2` targeting AArch64.

---

## 4. Microarchitectural Feature Space (Inputs $X$)
The simulation framework exposes the following knobs on the ARM Out-of-Order processor (`ArmO3CPU`):

### Pipeline & Execution Resources
- **Fetch / Decode / Rename / Issue / Dispatch / Commit Width**: $\{2, 4, 8\}$
- **Reorder Buffer (ROB) Entries**: $\{64, 96, 128, 192, 256\}$
- **Instruction Queue (IQ) Entries**: $\{32, 48, 64, 96\}$
- **Load Queue (LQ) & Store Queue (SQ) Entries**: $\{16, 32, 48, 64\}$
- **Physical Register File (Int & Float)**: $\{128, 192, 256\}$

### Cache & Memory Hierarchy
- **L1 Data Cache (L1D)**:
  - Size: $\{16\text{KiB}, 32\text{KiB}, 64\text{KiB}, 128\text{KiB}\}$
  - Associativity: $\{2, 4, 8\}$
- **L1 Instruction Cache (L1I)**:
  - Size: $\{16\text{KiB}, 32\text{KiB}, 64\text{KiB}\}$
  - Associativity: $\{2, 4, 8\}$
- **L2 Cache (Unified)**:
  - Size: $\{256\text{KiB}, 512\text{KiB}, 1\text{MiB}, 2\text{MiB}\}$
  - Associativity: $\{4, 8, 16\}$
- **Clock Frequency**: $\{1.5\text{GHz}, 2.0\text{GHz}, 2.4\text{GHz}, 3.0\text{GHz}\}$
- **Branch Predictor**: $\{\text{TournamentBP}, \text{BiModeBP}, \text{LocalBP}\}$

---

## 5. Target Output Metrics (Outputs $Y$)
Extracted directly from `stats.txt`:
1. **IPC** (`Instructions Per Cycle`)
2. **CPI** (`Cycles Per Instruction`)
3. **Simulated Execution Time** (`simSeconds`)
4. **L1D Cache MPKI** (Misses Per Kilo-Instructions)
5. **L2 Cache MPKI**
6. **Branch Misprediction Rate & Branch MPKI**
7. **DRAM Total Memory Requests** (Reads + Writes)

---

## 6. Functional Requirements
1. **Automated Latin Hypercube Sampling (LHS)**: Sample $N=160$ design points avoiding clustering and ensuring uniform coverage of parameter dimensions.
2. **Parallel Sweep Harness**: Parallelize gem5 runs across multi-core host CPU using Python's `multiprocessing` with per-run log isolation.
3. **Stat Harvester**: Robust parser extracting metrics with fallback sanity checks and regex validation.
4. **Multi-Model ML Pipeline**: Train and compare Ridge Regression, Random Forest, and Gradient Boosted Trees (XGBoost/LightGBM/HistGradientBoosting).
5. **Pareto Screening & Fast Optimizer**: Evaluate $\ge 5,000$ virtual configurations to identify optimal Pareto frontiers (IPC vs. Silicon Footprint).
6. **Ground-Truth Validation**: Re-execute top predicted configurations in gem5, calculate prediction delta and quantify simulation speedup.
