# Design System: Visualization, Reporting & Terminal Standards

## 1. Overview
The design system establishes uniform visual, tabular, and terminal presentation rules across all scripts, artifacts, and figures in the project. Since this is an engineering research tool, polish, clarity, readability, and reproducibility are paramount.

---

## 2. Terminal UI & CLI Standards

### 2.1 Execution Banner
Every script execution outputs a standardized hardware execution signature banner:
```
==============================================================================================================
   PROJECT: ML-BASED MICROARCHITECTURAL PERFORMANCE PREDICTION (ARM gem5)
   Host Name   : DESKTOP-WSL-ARM
   Machine ID  : 7F8E9D0A1B2C
   Target ISA  : ARM (AArch64) | Engine: ArmO3CPU (gem5 v24+)
   Timestamp   : 2026-10-17 08:30:00 UTC
==============================================================================================================
```

### 2.2 Tabular Output Formatting
All tabular CLI reports adhere to rigid column widths with unicode dividers:
```
+---------+-------+---------+----------+----------+----------+---------+---------+-----------+
| Config  | Bench | Width   | ROB      | L1D Size | L2 Size  | Sim IPC | Pred IPC| Error (%) |
+---------+-------+---------+----------+----------+----------+---------+---------+-----------+
| TOP-01  | matmul| 8-wide  | 256 ent  | 128 KiB  | 2048 KiB | 2.4182  | 2.3910  | 1.12%     |
| TOP-02  | matmul| 8-wide  | 192 ent  | 128 KiB  | 1024 KiB | 2.3510  | 2.3325  | 0.79%     |
| TOP-03  | matmul| 4-wide  | 256 ent  | 64 KiB   | 1024 KiB | 1.8490  | 1.8120  | 2.00%     |
| TOP-04  | matmul| 4-wide  | 128 ent  | 64 KiB   | 512 KiB  | 1.7105  | 1.7480  | 2.19%     |
| TOP-05  | matmul| 4-wide  | 96 ent   | 32 KiB   | 512 KiB  | 1.5830  | 1.5420  | 2.59%     |
+---------+-------+---------+----------+----------+----------+---------+---------+-----------+
```

### 2.3 Status Indicators & Badges
- `[PASS]` / `[SUCCESS]`: Green `#28a745`
- `[INFO]` / `[RUNNING]`: Cyan/Blue `#17a2b8`
- `[WARN]` / `[FALLBACK]`: Amber/Yellow `#ffc107`
- `[FAIL]` / `[TIMEOUT]`: Red `#dc3545`

---

## 3. Matplotlib & Data Visualization Guidelines
All generated plots are saved to `visualizations/` at **300 DPI** in PNG format.

### 3.1 Color Palette
- **Primary Line / Scatter**: `#1f77b4` (Deep Blue)
- **Ideal / Reference Line ($y=x$)**: `#d62728` (Crimson, dashed)
- **Secondary / Comparison**: `#ff7f0e` (Safety Orange)
- **Pareto Frontier**: `#2ca02c` (Emerald Green)
- **Background & Grids**: `#ffffff` (White canvas) with `#e0e0e0` subtle grid lines (alpha 0.5)

### 3.2 Typography & Sizing
- **Font Family**: Sans-serif (`DejaVu Sans`, `Helvetica`, `Arial`)
- **Title Size**: 14 pt, Bold
- **Axis Label Size**: 12 pt, Medium
- **Tick Label Size**: 10 pt
- **Legend Size**: 10 pt, frame alpha 0.9, fancy box

### 3.3 Core Charts Specifications
1. **Actual vs. Predicted Scatter (`predicted_vs_actual.png`)**:
   - $X$-axis: Simulated Ground-Truth IPC (gem5)
   - $Y$-axis: ML Model Predicted IPC
   - Includes diagonal $y = x$ reference line.
   - Text annotation box showing $R^2$, MAPE, and Spearman $\rho$.
2. **Feature Importance Bar Chart (`feature_importance.png`)**:
   - Horizontal bar chart sorting microarchitectural knobs by Gini importance / SHAP value.
   - Distinct color bands for Core (Width, ROB, IQ), Cache (L1D, L2), and System (Freq, Predictor).
3. **Pareto Frontier Plot (`pareto_frontier.png`)**:
   - $X$-axis: Microarchitectural Complexity / Estimated Area Proxy Score
   - $Y$-axis: IPC
   - Highlights the Pareto optimal envelope and shows dominated designs in grey.
4. **Simulation Time Reduction Chart (`speedup_comparison.png`)**:
   - Grouped bar chart comparing:
     * Exhaustive gem5 Simulation Time (Hours)
     * ML-Guided Design Space Exploration Time (DoE + Train + Validation) (Hours)
   - Clear speedup factor badge (e.g. `38.4x Speedup`).
