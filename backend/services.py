"""Backend ML inference services wrapping the pre-trained champion surrogate."""
import os
import sys
import time
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ml.predict_custom import (
    sanitize_and_fill_defaults,
    predict_custom_configurations,
    DEFAULT_MICROARCH
)
from ml.predict_and_screen import engineer_microarchitectural_features

MODEL_PATH = os.path.join(REPO_ROOT, "ml", "saved_models", "best_model_ipc.joblib")
TOP5_PATH = os.path.join(REPO_ROOT, "dataset", "top5_candidates.json")

# Cached model instance
_CACHED_PIPELINE = None

def get_pipeline():
    global _CACHED_PIPELINE
    if _CACHED_PIPELINE is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Champion surrogate model not found at: {MODEL_PATH}")
        _CACHED_PIPELINE = joblib.load(MODEL_PATH)
    return _CACHED_PIPELINE

def predict_single(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    pipeline = get_pipeline()
    
    # 1. Convert dict to single-row DataFrame
    df_raw = pd.DataFrame([config_dict])
    
    # 2. Run standard custom prediction pipeline
    df_out, inference_ms = predict_custom_configurations(pipeline, df_raw)
    row = df_out.iloc[0]
    
    # 3. Engineer ratios for radar visualization
    df_feat = engineer_microarchitectural_features(sanitize_and_fill_defaults(df_raw))
    ratios = {
        "rob_to_issue_ratio": round(float(df_feat["rob_to_issue_ratio"].iloc[0]), 2),
        "pipe_issue_fetch_ratio": round(float(df_feat["pipe_issue_fetch_ratio"].iloc[0]), 2),
        "pipe_commit_issue_ratio": round(float(df_feat["pipe_commit_issue_ratio"].iloc[0]), 2),
        "iq_to_rob_ratio": round(float(df_feat["iq_to_rob_ratio"].iloc[0]), 3),
        "ls_to_rob_ratio": round(float(df_feat["ls_to_rob_ratio"].iloc[0]), 3),
        "cache_l2_l1d_ratio": round(float(df_feat["cache_l2_l1d_ratio"].iloc[0]), 2),
        "phys_regs_to_rob_ratio": round(float(df_feat["phys_regs_to_rob_ratio"].iloc[0]), 2)
    }
    
    filled = {}
    for col in df_out.columns:
        val = row[col]
        if isinstance(val, (np.integer, int)):
            filled[col] = int(val)
        elif isinstance(val, (np.floating, float)):
            filled[col] = round(float(val), 4)
        else:
            filled[col] = str(val)
            
    return {
        "ipc": float(row["predicted_ipc"]),
        "uncertainty_sigma": float(row["uncertainty_std"]),
        "ci_lower": float(row["ci95_lower"]),
        "ci_upper": float(row["ci95_upper"]),
        "ucb": float(row["ucb_ipc"]),
        "area_cost_proxy": float(row["area_cost_proxy"]),
        "dynamic_power_proxy": float(row["power_proxy"]),
        "energy_per_instruction_proxy": float(row["energy_proxy"]),
        "latency_ms": round(inference_ms, 2),
        "ratios": ratios,
        "filled_config": filled
    }

def predict_batch_df(df_input: pd.DataFrame) -> Dict[str, Any]:
    pipeline = get_pipeline()
    
    df_out, inference_ms = predict_custom_configurations(pipeline, df_input)
    records = df_out.to_dict(orient="records")
    
    # Compute Pareto non-dominated points (Maximize IPC, Minimize Area Cost)
    pareto_candidates = []
    for i, r in enumerate(records):
        dominated = False
        for j, other in enumerate(records):
            if i != j:
                if (other["predicted_ipc"] >= r["predicted_ipc"] and 
                    other["area_cost_proxy"] <= r["area_cost_proxy"] and 
                    (other["predicted_ipc"] > r["predicted_ipc"] or other["area_cost_proxy"] < r["area_cost_proxy"])):
                    dominated = True
                    break
        if not dominated:
            pareto_candidates.append(r)
            
    pareto_sorted = sorted(pareto_candidates, key=lambda x: x["predicted_ipc"], reverse=True)
    
    summary = {
        "mean_ipc": round(float(df_out["predicted_ipc"].mean()), 4),
        "max_ipc": round(float(df_out["predicted_ipc"].max()), 4),
        "min_ipc": round(float(df_out["predicted_ipc"].min()), 4),
        "mean_area": round(float(df_out["area_cost_proxy"].mean()), 2),
        "pareto_optimal_count": len(pareto_sorted),
        "total_inference_ms": round(inference_ms, 2)
    }
    
    return {
        "total_samples": len(records),
        "predictions": records,
        "pareto_frontier": pareto_sorted,
        "summary": summary
    }

def get_pareto_reference() -> List[Dict[str, Any]]:
    if os.path.exists(TOP5_PATH):
        try:
            with open(TOP5_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return []
