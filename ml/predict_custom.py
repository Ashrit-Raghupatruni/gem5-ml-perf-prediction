#!/usr/bin/env python3
"""
Custom Input Microarchitectural Performance Prediction CLI
Project: ML-Based Microarchitectural Performance Prediction (ARM gem5)
Strictly adheres to: documents/schema.md, documents/architecture.md, documents/design_system.md

Supports:
1. Single configuration inference via CLI arguments or JSON string
2. Batch inference on custom CSV or JSON files
3. Automatic microarchitectural ratio derivation and physical constraint verification
4. Calibrated epistemic uncertainty, 95% confidence intervals, and hardware cost proxies
"""

import os
import sys

# Ensure repository root is in sys.path when executed directly as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
import argparse
import numpy as np
import pandas as pd
import joblib

try:
    from ml.predict_and_screen import predict_with_uncertainty, compute_cost_and_energy_proxies, engineer_microarchitectural_features
    from ml.train_models import get_feature_definitions
except ModuleNotFoundError:
    from predict_and_screen import predict_with_uncertainty, compute_cost_and_energy_proxies, engineer_microarchitectural_features
    from train_models import get_feature_definitions


DEFAULT_MICROARCH = {
    'workload': 'matmul',
    'fetch_width': 4,
    'decode_width': 4,
    'issue_width': 4,
    'commit_width': 4,
    'rob_entries': 128,
    'iq_entries': 64,
    'lq_entries': 32,
    'sq_entries': 32,
    'phys_int_regs': 192,
    'phys_fp_regs': 192,
    'branch_predictor': 'BiModeBP',
    'l1d_size_kb': 64,
    'l1d_assoc': 2,
    'l1i_size_kb': 32,
    'l1i_assoc': 2,
    'l2_size_kb': 1024,
    'l2_assoc': 8,
    'clk_freq_ghz': 2.0
}


def sanitize_and_fill_defaults(df_input):
    """
    Ensures all expected microarchitectural columns exist.
    Fills in omitted parameters intelligently using physical architectural rules:
      - decode_width = fetch_width (if missing)
      - issue_width = min(decode_width, 4) (if missing)
      - commit_width = issue_width (if missing)
      - phys_regs = max(128, int(rob * 0.8)) (if missing)
      - iq_entries = min(64, rob) (if missing)
      - lq/sq_entries = min(32, int(rob * 0.3)) (if missing)
    """
    df = df_input.copy()

    for col, default_val in DEFAULT_MICROARCH.items():
        if col not in df.columns:
            if col == 'decode_width' and 'fetch_width' in df.columns:
                df['decode_width'] = df['fetch_width']
            elif col == 'issue_width' and 'decode_width' in df.columns:
                df['issue_width'] = np.minimum(df['decode_width'], 4)
            elif col == 'commit_width' and 'issue_width' in df.columns:
                df['commit_width'] = df['issue_width']
            elif col in ['phys_int_regs', 'phys_fp_regs'] and 'rob_entries' in df.columns:
                df[col] = np.maximum(128, (df['rob_entries'] * 0.8).astype(int))
            elif col == 'iq_entries' and 'rob_entries' in df.columns:
                df['iq_entries'] = np.minimum(64, df['rob_entries'])
            elif col in ['lq_entries', 'sq_entries'] and 'rob_entries' in df.columns:
                df[col] = np.maximum(16, np.minimum(32, (df['rob_entries'] * 0.3).astype(int)))
            else:
                df[col] = default_val

    # Type enforcement
    int_cols = [
        'fetch_width', 'decode_width', 'issue_width', 'commit_width',
        'rob_entries', 'iq_entries', 'lq_entries', 'sq_entries',
        'phys_int_regs', 'phys_fp_regs',
        'l1d_size_kb', 'l1d_assoc', 'l1i_size_kb', 'l1i_assoc',
        'l2_size_kb', 'l2_assoc'
    ]
    for c in int_cols:
        if c in df.columns:
            df[c] = df[c].astype(int)

    df['clk_freq_ghz'] = df['clk_freq_ghz'].astype(float)
    df['workload'] = df['workload'].astype(str)
    df['branch_predictor'] = df['branch_predictor'].astype(str)

    return df


def predict_custom_configurations(pipeline, df_configs, alpha=0.05):
    """
    Takes an arbitrary DataFrame of configurations, computes microarchitectural features,
    predicts Instructions Per Cycle (IPC), calculates uncertainty and confidence intervals,
    and computes hardware area/power/energy cost proxies.
    """
    df_clean = sanitize_and_fill_defaults(df_configs)
    df_engineered = engineer_microarchitectural_features(df_clean)

    t_start = time.perf_counter()
    pred_mean, pred_std = predict_with_uncertainty(pipeline, df_engineered)
    inference_ms = (time.perf_counter() - t_start) * 1000.0

    df_out = df_clean.copy()
    df_out['predicted_ipc'] = np.round(pred_mean, 4)
    df_out['uncertainty_std'] = np.round(pred_std, 4)

    # 95% Confidence Interval (z = 1.96 for normal approximation)
    z_score = 1.95996
    ci_lower = np.maximum(0.01, pred_mean - (z_score * pred_std))
    ci_upper = pred_mean + (z_score * pred_std)
    df_out['ci95_lower'] = np.round(ci_lower, 4)
    df_out['ci95_upper'] = np.round(ci_upper, 4)

    # Upper Confidence Bound (exploration-exploitation proxy)
    df_out['ucb_ipc'] = np.round(pred_mean + (1.5 * pred_std), 4)

    # Hardware cost proxies
    area_proxy, power_proxy, energy_proxy = compute_cost_and_energy_proxies(df_out)
    df_out['area_cost_proxy'] = np.round(area_proxy, 2)
    df_out['power_proxy'] = np.round(power_proxy, 2)
    df_out['energy_proxy'] = np.round(energy_proxy, 2)

    return df_out, inference_ms


def print_prediction_results_table(df_results, title="CUSTOM MICROARCHITECTURAL PERFORMANCE PREDICTIONS"):
    """
    Renders a formatted terminal table of prediction outputs.
    """
    print("\n" + "=" * 135)
    print(f"      {title}")
    print("=" * 135)
    print(f"{'Idx':<4} | {'Workload':<10} | {'W':<3} | {'ROB':<5} | {'L1D':<5} | {'L2':<5} | {'Freq':<7} | {'Pred IPC':<10} | {'Uncertainty':<12} | {'95% CI':<18} | {'Cost Proxy':<10}")
    print("-" * 135)

    display_limit = min(50, len(df_results))
    for idx in range(display_limit):
        r = df_results.iloc[idx]
        ci_str = f"[{r['ci95_lower']:.3f}, {r['ci95_upper']:.3f}]"
        print(f"{idx+1:<4} | {r['workload']:<10} | {r['fetch_width']:<3} | {r['rob_entries']:<5} | {r['l1d_size_kb']:<5} | {r['l2_size_kb']:<5} | {r['clk_freq_ghz']:<5.1f}GHz | {r['predicted_ipc']:<10.4f} | +/-{r['uncertainty_std']:<10.4f} | {ci_str:<18} | {r['area_cost_proxy']:<10.2f}")

    if len(df_results) > display_limit:
        print(f"... and {len(df_results) - display_limit} more configurations (see output file).")

    print("=" * 135)


def main():
    parser = argparse.ArgumentParser(
        description="Predict ARM gem5 performance (IPC) for custom single or batch microarchitectural inputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model path
    # Resolve default model path gracefully whether invoked from repo root or parent workspace
    default_model = "ml/saved_models/best_model_ipc.joblib"
    local_model = os.path.join(os.path.dirname(__file__), "saved_models", "best_model_ipc.joblib")
    if not os.path.exists(default_model) and os.path.exists(local_model):
        default_model = local_model
    parser.add_argument("--model", type=str, default=default_model, help="Path to trained model pipeline")

    # Batch input options
    parser.add_argument("--csv", type=str, default=None, help="Path to input CSV containing candidate configurations")
    parser.add_argument("--json", type=str, default=None, help="Path to input JSON file containing list of configurations")

    # Direct single-input options
    parser.add_argument("--config-str", type=str, default=None, help="Inline JSON string specifying a single configuration")
    parser.add_argument("--workload", type=str, default=None, choices=['matmul', 'stream', 'quicksort', 'hash'], help="Target benchmark workload")
    parser.add_argument("--fetch-width", type=int, default=None, help="Fetch width (2, 4, 8)")
    parser.add_argument("--decode-width", type=int, default=None, help="Decode width (<= fetch_width)")
    parser.add_argument("--issue-width", type=int, default=None, help="Issue width (<= decode_width)")
    parser.add_argument("--commit-width", type=int, default=None, help="Commit width (>= issue_width)")
    parser.add_argument("--rob", type=int, default=None, help="Reorder Buffer (ROB) entries (64 to 256)")
    parser.add_argument("--iq", type=int, default=None, help="Instruction Queue (IQ) entries")
    parser.add_argument("--lq", type=int, default=None, help="Load Queue (LQ) entries")
    parser.add_argument("--sq", type=int, default=None, help="Store Queue (SQ) entries")
    parser.add_argument("--phys-regs", type=int, default=None, help="Physical registers (int and fp)")
    parser.add_argument("--branch-pred", type=str, default=None, choices=['TournamentBP', 'BiModeBP', 'LocalBP'], help="Branch predictor")
    parser.add_argument("--l1d-size", type=int, default=None, help="L1D cache size in KiB (16, 32, 64, 128)")
    parser.add_argument("--l1d-assoc", type=int, default=None, help="L1D cache associativity (2, 4, 8)")
    parser.add_argument("--l1i-size", type=int, default=None, help="L1I cache size in KiB (16, 32, 64)")
    parser.add_argument("--l1i-assoc", type=int, default=None, help="L1I cache associativity (2, 4, 8)")
    parser.add_argument("--l2-size", type=int, default=None, help="L2 cache size in KiB (256, 512, 1024, 2048)")
    parser.add_argument("--l2-assoc", type=int, default=None, help="L2 cache associativity (4, 8, 16)")
    parser.add_argument("--clk-freq", type=float, default=None, help="Core clock frequency in GHz (1.5, 2.0, 2.4, 3.0)")

    # Output options
    parser.add_argument("--output-csv", type=str, default=None, help="Path to export predictions as CSV")
    parser.add_argument("--output-json", type=str, default=None, help="Path to export predictions as JSON")

    args = parser.parse_args()

    # 1. Load trained surrogate pipeline
    if not os.path.exists(args.model):
        print(f"[ERROR] Trained surrogate model not found at: {args.model}")
        print("Please train a model first using: python3 ml/train_models.py")
        sys.exit(1)

    print(f"[INFO] Loading trained champion ML pipeline from: {args.model}")
    pipeline = joblib.load(args.model)

    # 2. Ingest Inputs
    input_records = []

    if args.csv is not None:
        if not os.path.exists(args.csv):
            print(f"[ERROR] Input CSV not found at: {args.csv}")
            sys.exit(1)
        print(f"[INFO] Ingesting batch configurations from CSV: {args.csv}")
        df_input = pd.read_csv(args.csv)
        source_desc = f"Batch CSV ({len(df_input)} items: {args.csv})"

    elif args.json is not None:
        if not os.path.exists(args.json):
            print(f"[ERROR] Input JSON file not found at: {args.json}")
            sys.exit(1)
        print(f"[INFO] Ingesting batch configurations from JSON: {args.json}")
        with open(args.json, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            df_input = pd.DataFrame(data)
        else:
            df_input = pd.DataFrame([data])
        source_desc = f"JSON file ({len(df_input)} items: {args.json})"

    elif args.config_str is not None:
        try:
            cfg = json.loads(args.config_str)
            df_input = pd.DataFrame([cfg] if isinstance(cfg, dict) else cfg)
            source_desc = "Inline JSON string"
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse inline JSON config: {e}")
            sys.exit(1)

    else:
        # Build single configuration from direct CLI flags (with intelligent defaults)
        single_cfg = DEFAULT_MICROARCH.copy()
        if args.workload is not None: single_cfg['workload'] = args.workload
        if args.fetch_width is not None: single_cfg['fetch_width'] = args.fetch_width
        if args.decode_width is not None: single_cfg['decode_width'] = args.decode_width
        elif args.fetch_width is not None: single_cfg['decode_width'] = args.fetch_width

        if args.issue_width is not None: single_cfg['issue_width'] = args.issue_width
        elif args.decode_width is not None: single_cfg['issue_width'] = min(args.decode_width, 4)

        if args.commit_width is not None: single_cfg['commit_width'] = args.commit_width
        elif args.issue_width is not None: single_cfg['commit_width'] = single_cfg['issue_width']

        if args.rob is not None: single_cfg['rob_entries'] = args.rob
        if args.iq is not None: single_cfg['iq_entries'] = args.iq
        if args.lq is not None: single_cfg['lq_entries'] = args.lq
        if args.sq is not None: single_cfg['sq_entries'] = args.sq
        if args.phys_regs is not None:
            single_cfg['phys_int_regs'] = args.phys_regs
            single_cfg['phys_fp_regs'] = args.phys_regs
        if args.branch_pred is not None: single_cfg['branch_predictor'] = args.branch_pred
        if args.l1d_size is not None: single_cfg['l1d_size_kb'] = args.l1d_size
        if args.l1d_assoc is not None: single_cfg['l1d_assoc'] = args.l1d_assoc
        if args.l1i_size is not None: single_cfg['l1i_size_kb'] = args.l1i_size
        if args.l1i_assoc is not None: single_cfg['l1i_assoc'] = args.l1i_assoc
        if args.l2_size is not None: single_cfg['l2_size_kb'] = args.l2_size
        if args.l2_assoc is not None: single_cfg['l2_assoc'] = args.l2_assoc
        if args.clk_freq is not None: single_cfg['clk_freq_ghz'] = args.clk_freq

        df_input = pd.DataFrame([single_cfg])
        source_desc = f"Single CLI configuration ({single_cfg['workload']}, W={single_cfg['fetch_width']}, ROB={single_cfg['rob_entries']}, L2={single_cfg['l2_size_kb']}KiB, Freq={single_cfg['clk_freq_ghz']}GHz)"

    # 3. Perform Prediction
    df_results, inference_ms = predict_custom_configurations(pipeline, df_input)

    # 4. Display Results
    print(f"\n[SUCCESS] Completed predictions for {len(df_results)} configurations in {inference_ms:.2f} ms ({inference_ms / len(df_results):.4f} ms/config).")
    print(f"Source: {source_desc}")
    print_prediction_results_table(df_results)

    # 5. Export Outputs (if requested)
    if args.output_csv is not None:
        os.makedirs(os.path.dirname(args.output_csv) if os.path.dirname(args.output_csv) else ".", exist_ok=True)
        df_results.to_csv(args.output_csv, index=False)
        print(f"[SUCCESS] Exported predictions to CSV: {args.output_csv}")

    if args.output_json is not None:
        os.makedirs(os.path.dirname(args.output_json) if os.path.dirname(args.output_json) else ".", exist_ok=True)
        records = df_results.to_dict(orient='records')
        with open(args.output_json, 'w') as f:
            json.dump(records, f, indent=2)
        print(f"[SUCCESS] Exported predictions to JSON: {args.output_json}")


if __name__ == '__main__':
    main()
