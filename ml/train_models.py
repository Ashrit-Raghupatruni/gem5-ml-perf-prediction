#!/usr/bin/env python3
"""
Machine Learning Training, Strict Cross-Validation & Generalization Engine
Project: ML-Based Microarchitectural Performance Prediction (ARM gem5)
Strictly adheres to: documents/schema.md, documents/design_system.md, documents/rules.md
"""

import os
import sys
import json
import argparse
import socket
import hashlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error, root_mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr
import joblib


def print_execution_banner():
    hostname = socket.gethostname()
    machine_hash = hashlib.sha256(hostname.encode()).hexdigest().upper()[:12]
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
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
        'KNN': KNeighborsRegressor(n_neighbors=5, weights='distance'),
        'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        'HistGradientBoosting': HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08, min_samples_leaf=5, random_state=42)
    }
    return models


def evaluate_models_cross_validation(X_train, y_train, cv_folds=5):
    """
    Standard K-Fold Cross Validation evaluated strictly on the TRAINING split.
    Records out-of-fold predictions.
    """
    models = get_model_candidates()
    results = {}
    
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    print(f"\n--- Running {cv_folds}-Fold Cross-Validation on Train Split ({len(X_train)} samples) ---")
    print(f"{'Model':<22} | {'R2 Score (mean)':<16} | {'MAPE (%) (mean)':<16} | {'RMSE (mean)':<14} | {'Spearman rho':<14}")
    print("-" * 92)
    
    for name, model in models.items():
        preprocessor = build_preprocessor()
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        r2_scores = []
        mape_scores = []
        rmse_scores = []
        spearman_scores = []
        oof_preds = np.zeros(len(y_train))
        
        for train_idx, val_idx in kf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            pipeline.fit(X_tr, y_tr)
            y_pred = pipeline.predict(X_val)
            oof_preds[val_idx] = y_pred
            
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
            'oof_preds': oof_preds
        }
        
        print(f"{name:<22} | {mean_r2:<16.4f} | {mean_mape:<16.2f}% | {mean_rmse:<14.4f} | {mean_spearman:<14.4f}")
        
    print("-" * 92)
    return results


def evaluate_grouped_cross_validation(X, y, groups):
    """
    Evaluates cross-workload generalization using GroupKFold (Leave-One-Workload-Out).
    """
    unique_groups = groups.unique()
    n_splits = len(unique_groups)
    gkf = GroupKFold(n_splits=n_splits)
    
    print(f"\n--- Running Workload-Grouped Cross-Validation (Leave-One-Workload-Out: {list(unique_groups)}) ---")
    print(f"{'Held-Out Workload':<22} | {'R2 Score':<12} | {'MAPE (%)':<12} | {'RMSE':<12} | {'Spearman rho':<12}")
    print("-" * 80)
    
    group_results = {}
    r2_list = []
    mape_list = []
    rmse_list = []
    rho_list = []
    
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        held_out_wl = groups.iloc[test_idx].iloc[0]
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        
        preprocessor = build_preprocessor()
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1))
        ])
        
        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_te)
        
        r2 = r2_score(y_te, y_pred)
        mape = mean_absolute_percentage_error(y_te, y_pred) * 100.0
        rmse = root_mean_squared_error(y_te, y_pred)
        rho, _ = spearmanr(y_te, y_pred)
        rho = rho if not np.isnan(rho) else 0.0
        
        r2_list.append(r2)
        mape_list.append(mape)
        rmse_list.append(rmse)
        rho_list.append(rho)
        
        group_results[held_out_wl] = {'r2': r2, 'mape': mape, 'rmse': rmse, 'spearman': rho}
        print(f"{held_out_wl:<22} | {r2:<12.4f} | {mape:<11.2f}% | {rmse:<12.4f} | {rho:<12.4f}")
        
    print("-" * 80)
    mean_summary = {
        'mean_r2': float(np.mean(r2_list)),
        'mean_mape': float(np.mean(mape_list)),
        'mean_rmse': float(np.mean(rmse_list)),
        'mean_spearman': float(np.mean(rho_list)),
        'by_workload': group_results
    }
    print(f"{'Mean Cross-Workload':<22} | {mean_summary['mean_r2']:<12.4f} | {mean_summary['mean_mape']:<11.2f}% | {mean_summary['mean_rmse']:<12.4f} | {mean_summary['mean_spearman']:<12.4f}")
    return mean_summary


def compute_bootstrap_confidence_intervals(y_true, y_pred, n_bootstraps=1000, alpha=0.05, seed=42):
    """
    Computes 95% bootstrap confidence intervals for key evaluation metrics.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    
    boot_r2 = []
    boot_mape = []
    boot_rmse = []
    boot_mae = []
    boot_rho = []
    
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        yt = y_true_arr[idx]
        yp = y_pred_arr[idx]
        
        boot_r2.append(r2_score(yt, yp))
        boot_mape.append(mean_absolute_percentage_error(yt, yp) * 100.0)
        boot_rmse.append(root_mean_squared_error(yt, yp))
        boot_mae.append(mean_absolute_error(yt, yp))
        rho, _ = spearmanr(yt, yp)
        boot_rho.append(rho if not np.isnan(rho) else 0.0)
        
    low_pct = (alpha / 2.0) * 100
    high_pct = (1.0 - alpha / 2.0) * 100
    
    return {
        'r2_ci95': [float(np.percentile(boot_r2, low_pct)), float(np.percentile(boot_r2, high_pct))],
        'mape_ci95': [float(np.percentile(boot_mape, low_pct)), float(np.percentile(boot_mape, high_pct))],
        'rmse_ci95': [float(np.percentile(boot_rmse, low_pct)), float(np.percentile(boot_rmse, high_pct))],
        'mae_ci95': [float(np.percentile(boot_mae, low_pct)), float(np.percentile(boot_mae, high_pct))],
        'spearman_ci95': [float(np.percentile(boot_rho, low_pct)), float(np.percentile(boot_rho, high_pct))]
    }


def compute_and_plot_permutation_importance(pipeline, X_test, y_test, out_path):
    """
    Computes permutation feature importance strictly on the HELD-OUT test split.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    perm_importance = permutation_importance(pipeline, X_test, y_test, n_repeats=15, random_state=42, n_jobs=-1)
    
    sorted_idx = perm_importance.importances_mean.argsort()[::-1]
    feature_names = np.array(X_test.columns)[sorted_idx]
    importances_mean = perm_importance.importances_mean[sorted_idx]
    importances_std = perm_importance.importances_std[sorted_idx]
    
    plt.figure(figsize=(10, 6), dpi=300)
    sns.set_theme(style="whitegrid")
    y_pos = np.arange(len(feature_names))
    plt.barh(y_pos, importances_mean, xerr=importances_std, align='center', color="#2ca02c", ecolor='black', capsize=4, alpha=0.85)
    plt.yticks(y_pos, feature_names)
    plt.gca().invert_yaxis()
    plt.title("Permutation Feature Importance on Held-Out Test Set (gem5 ARM)", fontsize=13, fontweight="bold")
    plt.xlabel("Mean Permutation Importance (Drop in $R^2$)", fontsize=11)
    plt.ylabel("Microarchitectural Feature", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Permutation feature importance plot saved to: {out_path}")
    
    return {feature_names[i]: float(importances_mean[i]) for i in range(len(feature_names))}


def plot_predicted_vs_actual(y_actual, y_pred, out_path, r2, mape, rho, title_prefix="Held-Out Test Set"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(7, 7), dpi=300)
    sns.set_theme(style="whitegrid")
    
    plt.scatter(y_actual, y_pred, alpha=0.75, color="#1f77b4", edgecolors="k", s=50, label="Test Simulations")
    min_val = min(y_actual.min(), y_pred.min()) * 0.95
    max_val = max(y_actual.max(), y_pred.max()) * 1.05
    plt.plot([min_val, max_val], [min_val, max_val], color="#d62728", linestyle="--", linewidth=2, label="Ideal (y = x)")
    
    stats_text = f"$R^2 = {r2:.4f}$\n$\\mathrm{{MAPE}} = {mape:.2f}\\%$\n$\\rho = {rho:.4f}$"
    plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray'))
    
    plt.title(f"ML Surrogate Performance ({title_prefix})", fontsize=13, fontweight="bold")
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
        sys.exit(1)
        
    df = pd.read_csv(dataset_path)
    print(f"[INFO] Successfully loaded dataset: {len(df)} samples, {len(df.columns)} columns.")
    
    categorical_features, numerical_features = get_feature_definitions()
    all_features = categorical_features + numerical_features
    missing_cols = [c for c in all_features if c not in df.columns]
    if missing_cols:
        print(f"[ERROR] Missing required feature columns in dataset: {missing_cols}")
        sys.exit(1)
    if target_metric not in df.columns:
        print(f"[ERROR] Target metric '{target_metric}' not present in dataset columns.")
        sys.exit(1)
        
    X = df[all_features]
    y = df[target_metric]
    
    # 1. Stratified 80/20 Train/Test Partition (stratified by workload)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=df['workload']
    )
    print(f"[INFO] Partitioned dataset: {len(X_train)} Train samples (80%), {len(X_test)} Untouched Test samples (20%).")
    
    # 2. Cross-Validation on Train Split ONLY
    cv_results = evaluate_models_cross_validation(X_train, y_train, cv_folds=5)
    
    # Select Champion Model based on CV R2 on Train Split
    best_model_name = max(cv_results, key=lambda k: cv_results[k]['r2'])
    best_res = cv_results[best_model_name]
    print(f"\n[CHAMPION MODEL] Selected based on CV Train: {best_model_name} (CV R2={best_res['r2']:.4f}, CV MAPE={best_res['mape']:.2f}%)")
    
    # Save Out-Of-Fold predictions for the champion model
    os.makedirs(output_model_dir, exist_ok=True)
    oof_df = X_train.copy()
    oof_df['true_' + target_metric] = y_train
    oof_df['oof_pred_' + target_metric] = best_res['oof_preds']
    oof_path = os.path.join(output_model_dir, "oof_predictions.csv")
    oof_df.to_csv(oof_path, index=False)
    print(f"[INFO] Saved Out-Of-Fold (OOF) predictions to: {oof_path}")
    
    # 3. Fit Champion Pipeline strictly on X_train, y_train
    preprocessor = build_preprocessor()
    final_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', get_model_candidates()[best_model_name])
    ])
    final_pipeline.fit(X_train, y_train)
    
    # 4. Strict Evaluation on UNTOUCHED Test Set
    y_test_pred = final_pipeline.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100.0
    test_rmse = root_mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rho, _ = spearmanr(y_test, y_test_pred)
    test_rho = test_rho if not np.isnan(test_rho) else 0.0
    
    # 5. Bootstrap Confidence Intervals on Test Set
    ci_results = compute_bootstrap_confidence_intervals(y_test, y_test_pred, n_bootstraps=1000, alpha=0.05, seed=42)
    
    print("\n" + "=" * 92)
    print("   STRICT INDEPENDENT TEST EVALUATION (Held-out 20% Partition)")
    print("=" * 92)
    print(f"   Test Set Size     : {len(X_test)} samples")
    print(f"   R2 Score          : {test_r2:.4f}  (95% CI: [{ci_results['r2_ci95'][0]:.4f}, {ci_results['r2_ci95'][1]:.4f}])")
    print(f"   MAPE              : {test_mape:.2f}%  (95% CI: [{ci_results['mape_ci95'][0]:.2f}%, {ci_results['mape_ci95'][1]:.2f}%])")
    print(f"   RMSE              : {test_rmse:.4f}  (95% CI: [{ci_results['rmse_ci95'][0]:.4f}, {ci_results['rmse_ci95'][1]:.4f}])")
    print(f"   MAE               : {test_mae:.4f}  (95% CI: [{ci_results['mae_ci95'][0]:.4f}, {ci_results['mae_ci95'][1]:.4f}])")
    print(f"   Spearman Rho      : {test_rho:.4f}  (95% CI: [{ci_results['spearman_ci95'][0]:.4f}, {ci_results['spearman_ci95'][1]:.4f}])")
    print("=" * 92)
    
    # 6. Evaluate Grouped Cross-Validation (Leave-One-Workload-Out)
    grouped_cv_summary = evaluate_grouped_cross_validation(X, y, df['workload'])
    
    # 7. Permutation Feature Importance computed STRICTLY on held-out test data
    perm_dict = compute_and_plot_permutation_importance(final_pipeline, X_test, y_test, "visualizations/feature_importance.png")
    
    # 8. Plot Test Set Predicted vs Actual
    plot_predicted_vs_actual(y_test, y_test_pred, "visualizations/predicted_vs_actual.png", test_r2, test_mape, test_rho, title_prefix="Held-Out 20% Test Set")
    
    # 9. Save test set predictions
    test_eval_df = X_test.copy()
    test_eval_df['true_' + target_metric] = y_test
    test_eval_df['pred_' + target_metric] = y_test_pred
    test_eval_path = os.path.join(output_model_dir, "test_predictions.csv")
    test_eval_df.to_csv(test_eval_path, index=False)
    
    # 10. Export trained model and comprehensive metadata
    model_save_path = os.path.join(output_model_dir, f"best_model_{target_metric}.joblib")
    joblib.dump(final_pipeline, model_save_path)
    print(f"\n[SUCCESS] Exported champion model pipeline to: {model_save_path}")
    
    metadata = {
        'target_metric': target_metric,
        'model_architecture': best_model_name,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'train_cv_5fold': {
            'mean_r2': float(best_res['r2']),
            'mean_mape': float(best_res['mape']),
            'mean_rmse': float(best_res['rmse']),
            'mean_spearman': float(best_res['spearman'])
        },
        'held_out_test_metrics': {
            'r2': float(test_r2),
            'mape': float(test_mape),
            'rmse': float(test_rmse),
            'mae': float(test_mae),
            'spearman': float(test_rho),
            'confidence_intervals_95': ci_results
        },
        'grouped_cross_workload_cv': grouped_cv_summary,
        'permutation_importance_test': perm_dict,
        'categorical_features': categorical_features,
        'numerical_features': numerical_features,
        'training_timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    }
    
    metadata_save_path = os.path.join(output_model_dir, "model_metadata.json")
    with open(metadata_save_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[SUCCESS] Exported rigorous training metadata to: {metadata_save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ML-Based ARM Microarchitectural Performance Model Trainer")
    parser.add_argument("--dataset", type=str, default="dataset/dataset_cleaned.csv", help="Path to cleaned dataset CSV")
    parser.add_argument("--output-dir", type=str, default="ml/saved_models", help="Directory to save trained model artifacts")
    parser.add_argument("--target", type=str, default="ipc", help="Target performance metric (default: ipc)")
    args = parser.parse_args()
    
    train_and_export(args.dataset, args.output_dir, args.target)
