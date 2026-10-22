#!/usr/bin/env python3
"""
Machine Learning Training & Cross-Validation Engine
Project: ML-Based Microarchitectural Performance Prediction (ARM gem5)
Strictly adheres to: documents/schema.md, documents/design_system.md, documents/rules.md
"""

import os
import sys
import json
import argparse
import socket
import hashlib
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error, root_mean_squared_error
from scipy.stats import spearmanr
import joblib


def print_execution_banner():
    hostname = socket.gethostname()
    machine_hash = hashlib.sha256(hostname.encode()).hexdigest().upper()[:12]
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    print("=" * 110)
    print("   PROJECT: ML-BASED MICROARCHITECTURAL PERFORMANCE PREDICTION (ARM gem5)")
    print(f"   Host Name   : {hostname}")
    print(f"   Machine ID  : {machine_hash}")
    print("   Target ISA  : ARM (AArch64) | Engine: ArmO3CPU (gem5 v24+)")
    print(f"   Timestamp   : {timestamp}")
    print("=" * 110)


def get_feature_definitions():
    """
    Returns categorical and numerical feature names specified in documents/schema.md
    """
    categorical_features = ['workload', 'branch_predictor']
    numerical_features = [
        'fetch_width', 'decode_width', 'issue_width', 'commit_width',
        'rob_entries', 'iq_entries', 'lq_entries', 'sq_entries',
        'phys_int_regs', 'phys_fp_regs',
        'l1d_size_kb', 'l1d_assoc',
        'l1i_size_kb', 'l1i_assoc',
        'l2_size_kb', 'l2_assoc',
        'clk_freq_ghz'
    ]
    return categorical_features, numerical_features


def build_preprocessor():
    categorical_features, numerical_features = get_feature_definitions()
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ],
        remainder='drop'
    )
    return preprocessor


def get_model_candidates():
    models = {
        'Ridge': Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        'HistGradientBoosting': HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08, min_samples_leaf=5, random_state=42)
    }
    return models


def evaluate_models_cross_validation(X, y, cv_folds=5):
    preprocessor = build_preprocessor()
    models = get_model_candidates()
    results = {}
    
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    print(f"\n--- Running {cv_folds}-Fold Stratified Cross-Validation ---")
    print(f"{'Model':<22} | {'R2 Score (mean)':<16} | {'MAPE (%) (mean)':<16} | {'RMSE (mean)':<14} | {'Spearman rho':<14}")
    print("-" * 92)
    
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        r2_scores = []
        mape_scores = []
        rmse_scores = []
        spearman_scores = []
        
        for train_idx, val_idx in kf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)
            
            r2 = r2_score(y_val, y_pred)
            mape = mean_absolute_percentage_error(y_val, y_pred) * 100.0
            rmse = root_mean_squared_error(y_val, y_pred)
            rho, _ = spearmanr(y_val, y_pred)
            
            r2_scores.append(r2)
            mape_scores.append(mape)
            rmse_scores.append(rmse)
            spearman_scores.append(rho if not np.isnan(rho) else 0.0)
            
        mean_r2 = np.mean(r2_scores)
        mean_mape = np.mean(mape_scores)
        mean_rmse = np.mean(rmse_scores)
        mean_spearman = np.mean(spearman_scores)
        
        results[name] = {
            'r2': mean_r2,
            'mape': mean_mape,
            'rmse': mean_rmse,
            'spearman': mean_spearman,
            'pipeline': pipeline
        }
        
        print(f"{name:<22} | {mean_r2:<16.4f} | {mean_mape:<16.2f}% | {mean_rmse:<14.4f} | {mean_spearman:<14.4f}")
        
    print("-" * 92)
    return results


def plot_feature_importance(pipeline, X, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    regressor = pipeline.named_steps['regressor']
    preprocessor = pipeline.named_steps['preprocessor']
    
    cat_feature_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out())
    _, numerical_features = get_feature_definitions()
    all_feature_names = cat_feature_names + numerical_features
    
    if hasattr(regressor, 'feature_importances_'):
        importances = regressor.feature_importances_
    elif hasattr(regressor, 'coef_'):
        importances = np.abs(regressor.coef_)
    else:
        return
        
    indices = np.argsort(importances)[::-1]
    sorted_features = [all_feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    plt.figure(figsize=(10, 6), dpi=300)
    sns.set_theme(style="whitegrid")
    sns.barplot(x=sorted_importances, y=sorted_features, palette="viridis")
    plt.title("Microarchitectural Feature Importance Ranking", fontsize=14, fontweight="bold")
    plt.xlabel("Importance Score / Normalized Coefficient", fontsize=12)
    plt.ylabel("Microarchitectural Feature", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Feature importance plot saved to: {out_path}")


def plot_predicted_vs_actual(y_actual, y_pred, out_path, r2, mape, rho):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(7, 7), dpi=300)
    sns.set_theme(style="whitegrid")
    
    plt.scatter(y_actual, y_pred, alpha=0.7, color="#1f77b4", edgecolors="k", s=45, label="Simulated Runs")
    min_val = min(y_actual.min(), y_pred.min()) * 0.95
    max_val = max(y_actual.max(), y_pred.max()) * 1.05
    plt.plot([min_val, max_val], [min_val, max_val], color="#d62728", linestyle="--", linewidth=2, label="Ideal (y = x)")
    
    stats_text = f"$R^2 = {r2:.4f}$\n$\\mathrm{{MAPE}} = {mape:.2f}\\%$\n$\\rho = {rho:.4f}$"
    plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.title("Surrogate Model: Predicted vs Actual IPC (gem5 ARM)", fontsize=13, fontweight="bold")
    plt.xlabel("Ground-Truth Simulated IPC (gem5)", fontsize=11)
    plt.ylabel("Predicted IPC (ML Surrogate)", fontsize=11)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Predicted vs Actual scatter plot saved to: {out_path}")


def train_and_export(dataset_path, output_model_dir, target_metric='ipc'):
    print_execution_banner()
    
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset not found at: {dataset_path}")
        print("Please place the harvested dataset file as specified in documents/schema.md before executing training.")
        sys.exit(1)
        
    df = pd.read_csv(dataset_path)
    print(f"[INFO] Successfully loaded dataset: {len(df)} samples, {len(df.columns)} columns.")
    
    categorical_features, numerical_features = get_feature_definitions()
    missing_cols = [c for c in categorical_features + numerical_features if c not in df.columns]
    if missing_cols:
        print(f"[ERROR] Missing required feature columns in dataset: {missing_cols}")
        sys.exit(1)
    if target_metric not in df.columns:
        print(f"[ERROR] Target metric '{target_metric}' not present in dataset columns.")
        sys.exit(1)
        
    X = df[categorical_features + numerical_features]
    y = df[target_metric]
    
    # Evaluate across candidate algorithms
    cv_results = evaluate_models_cross_validation(X, y, cv_folds=5)
    
    # Champion Selection based on highest R2
    best_model_name = max(cv_results, key=lambda k: cv_results[k]['r2'])
    best_res = cv_results[best_model_name]
    print(f"\n[CHAMPION MODEL] Selected: {best_model_name} (R2={best_res['r2']:.4f}, MAPE={best_res['mape']:.2f}%)")
    
    # Retrain champion model on full dataset
    preprocessor = build_preprocessor()
    final_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', get_model_candidates()[best_model_name])
    ])
    final_pipeline.fit(X, y)
    
    y_full_pred = final_pipeline.predict(X)
    full_r2 = r2_score(y, y_full_pred)
    full_mape = mean_absolute_percentage_error(y, y_full_pred) * 100.0
    full_rho, _ = spearmanr(y, y_full_pred)
    
    os.makedirs(output_model_dir, exist_ok=True)
    model_save_path = os.path.join(output_model_dir, f"best_model_{target_metric}.joblib")
    joblib.dump(final_pipeline, model_save_path)
    print(f"[SUCCESS] Exported champion model pipeline to: {model_save_path}")
    
    metadata = {
        'target_metric': target_metric,
        'model_architecture': best_model_name,
        'cv_folds': 5,
        'cv_mean_r2': float(best_res['r2']),
        'cv_mean_mape': float(best_res['mape']),
        'cv_mean_rmse': float(best_res['rmse']),
        'cv_mean_spearman': float(best_res['spearman']),
        'full_dataset_r2': float(full_r2),
        'full_dataset_mape': float(full_mape),
        'full_dataset_spearman': float(full_rho),
        'categorical_features': categorical_features,
        'numerical_features': numerical_features,
        'training_timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
        'dataset_sample_count': len(df)
    }
    
    metadata_save_path = os.path.join(output_model_dir, "model_metadata.json")
    with open(metadata_save_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[SUCCESS] Exported training metadata to: {metadata_save_path}")
    
    # Visualizations
    plot_feature_importance(final_pipeline, X, "visualizations/feature_importance.png")
    plot_predicted_vs_actual(y, y_full_pred, "visualizations/predicted_vs_actual.png", full_r2, full_mape, full_rho)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ML-Based ARM Microarchitectural Performance Model Trainer")
    parser.add_argument("--dataset", type=str, default="dataset/dataset_cleaned.csv", help="Path to cleaned dataset CSV")
    parser.add_argument("--output-dir", type=str, default="ml/saved_models", help="Directory to save trained model artifacts")
    parser.add_argument("--target", type=str, default="ipc", help="Target performance metric (default: ipc)")
    args = parser.parse_args()
    
    train_and_export(args.dataset, args.output_dir, args.target)
