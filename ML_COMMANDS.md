# Machine Learning & Optimization Pipeline Guide (`ML_COMMANDS.md`)

This guide isolates all Machine Learning (ML), surrogate modeling, candidate screening, active learning, and validation commands.

---

## 📌 Prerequisites

Ensure you are in the repository root directory in your Ubuntu terminal:

```bash
cd /home/ashrit/gem5/gem5-ml-perf-prediction
```

Ensure all Python dependencies are installed:

```bash
pip install -r requirements.txt --break-system-packages
```

---

## 🔬 1. Model Training & Cross-Validation (`ml/train_models.py`)

Trains candidate regression models (Ridge, Random Forest, HistGradientBoosting) with:
- **80% Train / 20% Untouched Test split** (stratified by workload).
- **5-Fold Cross-Validation** strictly on the train partition.
- **GroupKFold (Leave-One-Workload-Out)** cross-validation to assess cross-workload generalization.
- **95% Bootstrap Confidence Intervals** on held-out test data.
- **Permutation feature importance** computed strictly on held-out test data.
- **Out-of-fold (OOF)** prediction export.

```bash
python3 ml/train_models.py \
  --dataset dataset/dataset_cleaned.csv \
  --output-dir ml/saved_models \
  --target ipc
```

### Outputs Generated:
- `ml/saved_models/best_model_ipc.joblib` *(Champion model pipeline)*
- `ml/saved_models/model_metadata.json` *(95% CIs, CV scores, test set metrics, grouped CV metrics)*
- `ml/saved_models/oof_predictions.csv` *(Out-of-fold CV predictions on training data)*
- `ml/saved_models/test_predictions.csv` *(Predictions on untouched 20% test partition)*
- `visualizations/feature_importance.png` *(Permutation feature importance chart)*
- `visualizations/predicted_vs_actual.png` *(Held-out test set scatter plot)*

---

## ⚡ 2. Design Screening & Multi-Objective Pareto Optimization (`ml/predict_and_screen.py`)

Performs millisecond batch inference across 5,000 legal microarchitectural design candidates:
- Computes **Epistemic Uncertainty ($\sigma$)** across random forest trees.
- Evaluates **Upper Confidence Bound** ($\text{UCB} = \mu + 1.5\sigma$).
- Computes **Area / Complexity**, **Dynamic Power**, and **Energy Proxies**.
- Maps both **Global** and **Per-Workload Pareto Frontiers**.
- Exports the Top-$K$ Pareto-optimal designs for gem5 ground-truth validation.

```bash
python3 ml/predict_and_screen.py \
  --model ml/saved_models/best_model_ipc.joblib \
  --candidates 5000 \
  --top-k 5 \
  --kappa 1.5 \
  --output dataset/top5_candidates.json
```

### Outputs Generated:
- `dataset/top5_candidates.json` *(Top-5 Pareto-optimal candidate microarchitectures)*
- `visualizations/pareto_frontier.png` *(Global & per-workload Pareto frontier charts)*

---

## 🎯 3. Active Learning & Adaptive Sampling (`ml/active_learning.py`)

Queries an unlabeled pool of candidate configurations to identify the most informative designs for the next iteration of gem5 simulations:
- Strategy options: `uncertainty` (maximum predictive variance), `ucb` (upper confidence bound), or `random`.

```bash
python3 ml/active_learning.py \
  --model ml/saved_models/best_model_ipc.joblib \
  --pool-size 2000 \
  --queries 15 \
  --strategy uncertainty \
  --output configs/active_queries.json
```

### Alternative Strategies:
```bash
# Upper Confidence Bound (exploration + exploitation):
python3 ml/active_learning.py --strategy ucb --queries 15

# Random baseline sampling:
python3 ml/active_learning.py --strategy random --queries 15
```

### Outputs Generated:
- `configs/active_queries.json` *(15 candidate specs prioritized for simulation)*

---

## 🔍 4. Ground-Truth gem5 Validation & Defensible Speedup (`ml/validate_and_evaluate.py`)

Validates the ML surrogate against cycle-accurate gem5 simulations:
- Runs full cycle-accurate gem5 simulations on the Top-5 predicted microarchitectures.
- Compares predicted vs. simulated IPC and calculates percentage error ($\le 7\%$).
- Scrapes actual measured wall-clock and CPU times from `outputs/simulation_manifest.json`.
- Calculates **Host Wall-Clock Speedup** (parallel 4-worker execution) and **Aggregate Core-Hour Speedup** against a 5,000-point exhaustive baseline.

```bash
python3 ml/validate_and_evaluate.py \
  --candidates dataset/top5_candidates.json \
  --gem5 /home/ashrit/gem5/build/ARM/gem5.opt \
  --script configs/arm_o3_parametric.py \
  --benchmarks benchmarks \
  --output dataset/validation_results.csv \
  --inputs outputs
```

### Outputs Generated:
- `dataset/validation_results.csv` *(Per-candidate ground-truth vs predicted IPC with errors)*
- `dataset/validation_summary.json` *(Speedup metrics and validation error summary)*
- `visualizations/speedup_comparison.png` *(Wall-clock and core-hour speedup charts)*

---

## 💾 5. Copy ML Artifacts to Windows

To view all trained models, metadata, and visualization figures in Windows File Explorer:

```bash
cp -r ml/saved_models/* /mnt/c/Users/ashri/Documents/antigravity/dazzling-pythagoras/gem5-ml-perf-prediction/ml/saved_models/
cp -r dataset/* /mnt/c/Users/ashri/Documents/antigravity/dazzling-pythagoras/gem5-ml-perf-prediction/dataset/
cp -r visualizations/* /mnt/c/Users/ashri/Documents/antigravity/dazzling-pythagoras/gem5-ml-perf-prediction/visualizations/
```
