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

# Reference Pareto frontier coordinates from Top-5 validated Pareto points
# (Area, IPC)
PARETO_BENCHMARK = [
    (14.4, 0.9546),
    (17.2, 1.4322),
    (20.4, 2.5117),
    (28.0, 3.0578),
    (38.0, 3.9959)
]

def evaluate_design_quality(ipc: float, area: float, filled_config: Dict[str, Any], ratios: Dict[str, float]) -> Dict[str, Any]:
    """
    Evaluates microarchitectural design quality based on Pareto frontier distance,
    area-efficiency trade-offs, and physical hardware queue/cache ratios.
    """
    workload = str(filled_config.get("workload", "matmul")).lower()
    fetch_w = int(filled_config.get("fetch_width", 4))
    issue_w = int(filled_config.get("issue_width", 4))
    rob = int(filled_config.get("rob_entries", 128))
    l1d = int(filled_config.get("l1d_size_kb", 64))
    l2 = int(filled_config.get("l2_size_kb", 1024))
    freq = float(filled_config.get("clk_freq_ghz", 2.4))
    
    # Interpolate expected frontier IPC at this area proxy:
    # (14.4 -> 0.95), (17.2 -> 1.43), (20.4 -> 2.51), (28.0 -> 3.06), (38.0 -> 3.99)
    if area <= 14.4:
        expected_frontier_ipc = 0.95 * (area / 14.4)
    elif area <= 17.2:
        expected_frontier_ipc = 0.95 + (1.43 - 0.95) * ((area - 14.4) / (17.2 - 14.4))
    elif area <= 20.4:
        expected_frontier_ipc = 1.43 + (2.51 - 1.43) * ((area - 17.2) / (20.4 - 17.2))
    elif area <= 28.0:
        expected_frontier_ipc = 2.51 + (3.06 - 2.51) * ((area - 20.4) / (28.0 - 20.4))
    elif area <= 38.0:
        expected_frontier_ipc = 3.06 + (3.99 - 3.06) * ((area - 28.0) / (38.0 - 28.0))
    else:
        expected_frontier_ipc = 3.99
        
    # Relative frontier performance ratio (1.0 = on frontier, >1.0 = exceeds, <0.7 = poor)
    frontier_ratio = min(1.2, max(0.2, ipc / max(0.1, expected_frontier_ipc)))
    
    # 2. Compute Efficiency Metric (IPC / Area Proxy * 10)
    eff_ratio = round((ipc / max(5.0, area)) * 10.0, 2)
    
    # 3. Detect Specific Microarchitectural Bottlenecks
    issues = []
    recommendations = []
    
    # Bottleneck A: Width mismatch
    if issue_w > fetch_w:
        issues.append(f"Issue width ({issue_w}) exceeds fetch width ({fetch_w}), causing front-end pipeline starvation.")
        recommendations.append(f"Align fetch width to {issue_w} or lower issue width to {fetch_w}.")
        
    # Bottleneck B: Memory wall / cache choking
    if workload == "stream":
        if l1d <= 32 and rob >= 192:
            issues.append("Memory-bound streaming workload with small L1D cache (<=32K) cannot keep large ROB filled.")
            recommendations.append("Increase L1D to 64K/128K or scale down ROB to 96 to save 30% silicon area.")
        elif ipc < 1.0 and area > 25.0:
            issues.append("High silicon area invested in execution units that remain stalled waiting on DRAM bandwidth.")
            recommendations.append("Stream is bandwidth-limited: reduce issue width and ROB to cut area without IPC drop.")
    elif workload == "matmul":
        if l2 < 1024 and issue_w >= 6:
            issues.append(f"Wide compute core ({issue_w}-wide) throttled by small L2 cache ({l2}K) on dense matrix reuse.")
            recommendations.append("Upgrade L2 cache to 2048 KB to unlock substantial IPC headroom.")
        elif rob <= 64 and issue_w >= 4:
            issues.append(f"ROB ({rob}) too small to expose loop-level instruction parallelism for {issue_w}-wide execution.")
            recommendations.append("Increase ROB capacity to 128 or 192.")
            
    # Bottleneck C: Cache capacity inversion
    if l2 <= l1d:
        issues.append(f"L2 cache ({l2}K) is not sufficiently larger than L1D ({l1d}K), causing frequent victim evictions.")
        recommendations.append(f"Scale L2 cache to at least 4x L1D capacity.")
        
    # 4. Synthesize Score & Classification
    base_score = int(frontier_ratio * 80)
    if eff_ratio >= 1.0:
        base_score += 10
        
    final_score = max(20, min(100, base_score - (len(issues) * 12)))
    strictly_dominated = (frontier_ratio < 0.75)
    
    if final_score >= 82:
        status = "Optimal"
        diag = "Near Pareto-optimal frontier! High throughput achieved with strong silicon area utilization."
        rec = "Balanced configuration ready for gem5 cycle-accurate sign-off."
    elif final_score >= 68:
        status = "Good"
        diag = "Well-balanced microarchitecture with healthy pipeline and cache parameters."
        rec = "Good baseline design. Minor tuning of L2 or frequency can yield further efficiency."
    elif final_score >= 50:
        status = "Sub-optimal"
        diag = "Sub-optimal trade-off: Area cost is elevated relative to the realized IPC throughput."
        rec = recommendations[0] if recommendations else "Consider reducing ROB entries or core width to optimize area."
    else:
        status = "Inefficient"
        diag = "Dominated configuration: Severe resource bottleneck or over-provisioned pipeline."
        rec = recommendations[0] if recommendations else "Rebalance pipeline widths and memory hierarchy."
        
    if issues:
        diag = " ".join(issues)
    if recommendations:
        rec = " ".join(recommendations)
        
    return {
        "status": status,
        "score": int(final_score),
        "efficiency_ratio": float(eff_ratio),
        "is_pareto_dominated": bool(strictly_dominated),
        "diagnosis": diag,
        "recommendation": rec
    }

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
            
    eval_result = evaluate_design_quality(float(row["predicted_ipc"]), float(row["area_cost_proxy"]), filled, ratios)
    
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
        "filled_config": filled,
        "evaluation": eval_result
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
