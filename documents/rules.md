# Project Operational Rules & Execution Protocol (`rules.md`)

## 1. Prime Directive & Operational Rules

> [!IMPORTANT]
> ### RULE 1: MANDATORY CONTINUOUS TASK STATUS UPDATES
> **The AI Assistant and any contributing engineer MUST explicitly update this document (`rules.md`) and `todo.md` after completing EVERY task or sub-phase.**
> - You may NOT proceed to a subsequent phase without updating the Status Ledger in Section 4 of this file.
> - Each entry in the Status Ledger must record:
>   1. Task / Phase ID
>   2. Completion Timestamp
>   3. Evidence / Artifact Generated (e.g. path to binary, log file, CSV, or plot)
>   4. Resulting Status (`[COMPLETED]`, `[FAILED]`, `[IN PROGRESS]`)
> - Before claiming project completion, all entries in Section 4 must be verified and marked as `[COMPLETED]`.

---

### RULE 2: FAIL-FAST VALIDATION BEFORE SWEEPS
- Never launch large simulation sweeps without first running a single-configuration smoke test (`configs/arm_o3_parametric.py --test-run`) and validating that:
  1. The gem5 execution exits cleanly with status code `0`.
  2. The output `stats.txt` contains valid values for all required counters (`simInsts`, `ipc`, `l1d.overall_misses`, `l2.overall_misses`, `branchPred.condIncorrect`).

---

### RULE 3: RIGOROUS ML ACCEPTANCE CRITERIA
- No machine learning model may be reported as acceptable unless:
  1. It undergoes 5-Fold Stratified Cross-Validation across workloads.
  2. The model achieves $R^2 \ge 0.85$ and $\text{MAPE} \le 8.5\%$ on holdout folds.
  3. The Spearman rank correlation satisfies $\rho \ge 0.88$ (ensuring ranking fidelity).

---

### RULE 4: GROUND-TRUTH EMPIRICAL VALIDATION REQUIRED
- Under no circumstances may performance optimization claims be made based solely on ML predictions.
- The predicted top configurations (top-5) **MUST** be simulated through actual gem5 runs, and the validation delta ($\|\text{IPC}_{\text{pred}} - \text{IPC}_{\text{sim}}\|$) must be reported explicitly.
- The simulation time reduction ratio must be quantitatively proven using measured wallclock runtimes.

---

### RULE 5: ISOLATION & RESOURCE LIMITS
- All parallel gem5 processes must be wrapped with a 300-second watchdog timer.
- Output directories must be strictly isolated per run ID (`outputs/run_{id}/`).
- Host memory and core concurrency must be bounded to preserve host system stability.

---

## 2. Directory & Path Standards
- **WSL Project Root**: `/home/ashrit/gem5/project`
- **gem5 Binary**: `/home/ashrit/gem5/build/ARM/gem5.opt`
- **ARM Cross-Compiler**: `/usr/bin/aarch64-linux-gnu-gcc`
- **Windows Workspace Mirror**: `c:\Users\ashri\Documents\antigravity\dazzling-pythagoras\`
- **Documentation Directory**: `/home/ashrit/gem5/project/documents/` and `dazzling-pythagoras/documents/`

---

## 3. Communication & Code Quality Standards
- Preserve all existing comments and documentation.
- Python code must follow PEP 8 standards with descriptive variable naming and docstrings.
- Scripts must handle missing files, corrupt stats, and CLI flag variations gracefully without uncaught exceptions.

---

## 4. Live Task Status Ledger (Dynamic Tracking)
*This ledger must be updated continuously after every task.*

| Task ID | Description | Target Phase | Status | Timestamp (UTC) | Verification Evidence / Output Artifact |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DOC-01** | Create `/documents` folder & PRD (`prd.md`) | Planning | `[COMPLETED]` | 2026-10-17 08:18 | `documents/prd.md` created & reviewed |
| **DOC-02** | Create Technical Architecture (`architecture.md`) | Planning | `[COMPLETED]` | 2026-10-17 08:18 | `documents/architecture.md` created & reviewed |
| **DOC-03** | Create User Flow & CLI Journey (`user_flow.md`) | Planning | `[COMPLETED]` | 2026-10-17 08:18 | `documents/user_flow.md` created & reviewed |
| **DOC-04** | Create Design System & Styling (`design_system.md`) | Planning | `[COMPLETED]` | 2026-10-17 08:19 | `documents/design_system.md` created & reviewed |
| **DOC-05** | Create Schema Specifications (`schema.md`) | Planning | `[COMPLETED]` | 2026-10-17 08:19 | `documents/schema.md` created & reviewed |
| **DOC-06** | Create Security & Safeguards (`security.md`) | Planning | `[COMPLETED]` | 2026-10-17 08:19 | `documents/security.md` created & reviewed |
| **DOC-07** | Create Implementation Checklist (`todo.md`) | Planning | `[COMPLETED]` | 2026-10-17 08:19 | `documents/todo.md` created & reviewed |
| **DOC-08** | Create Operational Protocol & Rules (`rules.md`) | Planning | `[COMPLETED]` | 2026-10-17 08:20 | `documents/rules.md` created & initialized |
| **DOC-09** | Create Master Project README (`README.md`) in root | Planning | `[COMPLETED]` | 2026-10-17 08:28 | `README.md` created & synced to WSL root |
| **GIT-01** | Git init, initial commit & push to GitHub `origin/main` | Setup | `[COMPLETED]` | 2026-10-17 08:31 | Pushed to `Ashrit-Raghupatruni/gem5-ml-perf-prediction` |
| **PLN-01** | Split Plans into Plan 1 (Dataset) & Plan 2 (Implementation) | Planning | `[COMPLETED]` | 2026-10-22 12:38 | `plan_1_dataset_generation.md` & `plan_2_implementation.md` |
| **ML-CODE**| Implement Plan 2 ML Engine Code (train, screen, validate) | Phase 5-7 | `[COMPLETED]` | 2026-10-22 13:22 | `ml/train_models.py`, `predict_and_screen.py`, `validate_and_evaluate.py` |
| **PH1-01** | Implement ARM64 Benchmark Suite (4 benchmarks) | Phase 1 | `[ ]` | Pending | `benchmarks/*.c` |
| **PH1-02** | Compile Benchmarks with aarch64 gcc | Phase 1 | `[ ]` | Pending | Static ELF verification |
| **PH2-01** | Develop Parametric ARM O3 CPU Config Script | Phase 2 | `[ ]` | Pending | `configs/arm_o3_parametric.py` |
| **PH2-02** | Execute Single-Run Smoke Test & Stat Verification | Phase 2 | `[ ]` | Pending | `m5out/stats.txt` verification |
| **PH3-01** | Generate Latin Hypercube Sampling Matrix (160 pts) | Phase 3 | `[ ]` | Pending | `configs/sweep_configs.json` |
| **PH3-02** | Execute Parallel gem5 Sweep with Process Pool | Phase 3 | `[ ]` | Pending | 160 completed runs in `outputs/` |
| **PH4-01** | Harvest & Clean gem5 Simulation Dataset | Phase 4 | `[ ]` | Pending | `dataset/dataset_cleaned.csv` |
| **PH5-01** | Train & Cross-Validate Regression & Tree Models | Phase 5 | `[ ]` | Pending | $R^2 \ge 0.88$, MAPE, Spearman $\rho$ |
| **PH5-02** | Export Model Pipeline & Feature Importance Plots | Phase 5 | `[ ]` | Pending | `best_model_ipc.joblib` |
| **PH6-01** | Screen 5,000 Candidates & Identify Pareto Optimal Top-5 | Phase 6 | `[ ]` | Pending | `dataset/top5_candidates.json` |
| **PH7-01** | Run Ground-Truth Validation in gem5 | Phase 7 | `[ ]` | Pending | Measured Top-5 Simulated IPC |
| **PH7-02** | Compute Speedup & Final Evaluation Report | Phase 7 | `[ ]` | Pending | Final project report & walkthrough |
