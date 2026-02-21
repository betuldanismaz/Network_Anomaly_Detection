"""
XGBoost Training Script for Network Intrusion Detection System

Date: 2026-02-15
"""

import os
import sys
import json
import time
import warnings
from typing import Tuple, Dict, Any

# Fix Unicode encoding for Windows (emoji support)
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

warnings.filterwarnings('ignore')

# ANSI Color Codes for Terminal Output
GREEN = '\033[92m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
RED = '\033[91m'
MAGENTA = '\033[95m'
RESET = '\033[0m'

# Try to import XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print(f"{RED}{'='*80}{RESET}")
    print(f"{RED}❌ CRITICAL ERROR: XGBoost is not installed!{RESET}")
    print(f"{RED}   Please install it: pip install xgboost{RESET}")
    print(f"{RED}{'='*80}{RESET}")
    sys.exit(1)


def print_header(title: str, color: str = CYAN) -> None:
    """Print a formatted section header."""
    print(f"\n{color}{'='*80}{RESET}")
    print(f"{color}{title}{RESET}")
    print(f"{color}{'='*80}{RESET}\n")


def detect_gpu_config() -> Dict[str, Any]:

    print(f"{YELLOW}🔍 Step 1: Detecting Hardware Configuration...{RESET}")
    
    xgb_version = xgb.__version__
    print(f"   ✅ XGBoost Version: {xgb_version}")
    
    config = {
        'version': xgb_version,
        'gpu_available': False,
        'device_param': None,
        'tree_method': None
    }
    
    # Check GPU availability
    try:
        # Try to build a small DMatrix on GPU
        test_data = xgb.DMatrix(np.random.rand(10, 5), label=np.random.randint(0, 2, 10))
        
        # XGBoost 2.0+ uses 'device' parameter
        major_version = int(xgb_version.split('.')[0])
        if major_version >= 2:
            params = {'device': 'cuda', 'tree_method': 'hist'}
            config['device_param'] = 'cuda'
            config['tree_method'] = 'hist'
        else:
            params = {'tree_method': 'gpu_hist'}
            config['tree_method'] = 'gpu_hist'
        
        # Test GPU training
        xgb.train(params, test_data, num_boost_round=1, verbose_eval=False)
        config['gpu_available'] = True
        
        print(f"{GREEN}   ✅ GPU DETECTED! Training will use CUDA acceleration.{RESET}")
        if config['device_param']:
            print(f"   📌 Using: device='{config['device_param']}', tree_method='{config['tree_method']}'")
        else:
            print(f"   📌 Using: tree_method='{config['tree_method']}'")
            
    except Exception as e:
        print(f"{RED}   ⚠️  GPU NOT AVAILABLE!{RESET}")
        print(f"{RED}   ⚠️  Error: {str(e)}{RESET}")
        print(f"{YELLOW}   ⚠️  Falling back to CPU training (this will be SLOW for 2.8M samples){RESET}")
        config['tree_method'] = 'hist'
    
    return config


def load_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    print(f"{YELLOW}📂 Step 2: Loading Datasets...{RESET}")
    
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    # Verify files exist
    for path, name in [(train_path, "Train"), (val_path, "Validation"), (test_path, "Test")]:
        if not os.path.exists(path):
            print(f"{RED}❌ Error: {name} file not found at {path}{RESET}")
            sys.exit(1)
    
    # Load data with optimized dtypes (float32 to save memory)
    print(f"   ⏳ Loading data (optimized for memory efficiency)...")
    
    train_df = pd.read_csv(train_path, dtype=np.float32)
    val_df = pd.read_csv(val_path, dtype=np.float32)
    test_df = pd.read_csv(test_path, dtype=np.float32)
    
    print(f"   ✅ Train Set: {train_df.shape} ({train_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB)")
    print(f"   ✅ Val Set:   {val_df.shape} ({val_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB)")
    print(f"   ✅ Test Set:  {test_df.shape} ({test_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB)")
    
    # Class distribution
    train_dist = train_df['Label'].value_counts()
    print(f"\n   📊 Training Set Class Distribution:")
    print(f"      - Benign (0): {train_dist.get(0.0, 0):,.0f} ({train_dist.get(0.0, 0)/len(train_df)*100:.2f}%)")
    print(f"      - Attack (1): {train_dist.get(1.0, 0):,.0f} ({train_dist.get(1.0, 0)/len(train_df)*100:.2f}%)")
    
    return train_df, val_df, test_df


def prepare_datasets(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple:
    """
    Separate features and targets, calculate class weights.
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, scale_pos_weight)
    """
    print(f"\n{YELLOW}⚙️  Step 3: Preparing Features and Calculating Class Weights...{RESET}")
    
    # Separate features and target
    X_train = train_df.drop('Label', axis=1)
    y_train = train_df['Label'].astype(int)
    
    X_val = val_df.drop('Label', axis=1)
    y_val = val_df['Label'].astype(int)
    
    X_test = test_df.drop('Label', axis=1)
    y_test = test_df['Label'].astype(int)
    
    # Calculate scale_pos_weight dynamically
    num_negative = (y_train == 0).sum()
    num_positive = (y_train == 1).sum()
    scale_pos_weight = num_negative / num_positive if num_positive > 0 else 1.0
    
    print(f"   ✅ Features: {X_train.shape[1]} columns")
    print(f"   ✅ Dynamic Class Weight Calculation:")
    print(f"      - Negative samples: {num_negative:,}")
    print(f"      - Positive samples: {num_positive:,}")
    print(f"      - scale_pos_weight: {scale_pos_weight:.4f}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scale_pos_weight


def train_model(X_train: pd.DataFrame, y_train: pd.Series, 
                X_val: pd.DataFrame, y_val: pd.Series,
                gpu_config: Dict[str, Any], scale_pos_weight: float) -> xgb.XGBClassifier:
    """
    Train XGBoost model with early stopping.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        gpu_config: GPU configuration dictionary
        scale_pos_weight: Class weight for imbalanced data
        
    Returns:
        Trained XGBoost model
    """
    print(f"\n{YELLOW}🚀 Step 4: Training XGBoost Model...{RESET}")
    
    # Configure hyperparameters
    params = {
        'n_estimators': 1000,
        'max_depth': 7,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 1,
        'early_stopping_rounds': 50  # XGBoost 2.0+ requires this in init, not fit()
    }
    
    # Add GPU parameters if available
    if gpu_config['gpu_available']:
        if gpu_config['device_param']:
            params['device'] = gpu_config['device_param']
        params['tree_method'] = gpu_config['tree_method']
    else:
        params['tree_method'] = 'hist'
    
    print(f"   📋 Hyperparameters:")
    for key, value in params.items():
        print(f"      - {key}: {value}")
    
    # Initialize model
    model = xgb.XGBClassifier(**params)
    
    # Train with early stopping
    print(f"\n   ⏳ Training with Early Stopping (patience=50)...")
    print(f"   {'─'*76}")
    
    start_time = time.time()
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    
    training_time = time.time() - start_time
    
    print(f"   {'─'*76}")
    print(f"{GREEN}   ✅ Training Complete!{RESET}")
    print(f"   ⏱️  Training Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"   🌲 Best Iteration: {model.best_iteration}")
    print(f"   📊 Best Score (LogLoss): {model.best_score:.6f}")
    
    return model


def optimize_threshold(model: xgb.XGBClassifier, X_val: pd.DataFrame, y_val: pd.Series) -> Tuple[float, Dict]:
    """
    Find optimal decision threshold by maximizing F1-Score on validation set.
    
    Args:
        model: Trained XGBoost model
        X_val, y_val: Validation data
        
    Returns:
        Tuple of (optimal_threshold, metrics_dict)
    """
    print(f"\n{YELLOW} Step 5: Optimizing Decision Threshold...{RESET}")
    print(f"   Goal: Maximize F1-Score (balanced Precision & Recall)")
    
    # Get probability predictions
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Test thresholds from 0.01 to 0.99
    thresholds = np.arange(0.01, 1.00, 0.01)
    f1_scores = []
    
    print(f"   ⏳ Testing {len(thresholds)} thresholds...")
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_val, y_pred)
        f1_scores.append(f1)
    
    # Find optimal threshold
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    
    # Calculate metrics at optimal threshold
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    
    metrics = {
        'threshold': float(optimal_threshold),
        'f1_score': float(optimal_f1),
        'precision': float(precision_score(y_val, y_pred_optimal)),
        'recall': float(recall_score(y_val, y_pred_optimal)),
        'accuracy': float(accuracy_score(y_val, y_pred_optimal))
    }
    
    print(f"{GREEN}   ✅ Optimal Threshold Found: {optimal_threshold:.4f}{RESET}")
    print(f"   📊 Validation Metrics at Optimal Threshold:")
    print(f"      - F1-Score:  {metrics['f1_score']:.4f}")
    print(f"      - Precision: {metrics['precision']:.4f}")
    print(f"      - Recall:    {metrics['recall']:.4f}")
    print(f"      - Accuracy:  {metrics['accuracy']:.4f}")
    
    return optimal_threshold, metrics


def evaluate_model(model: xgb.XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series,
                   optimal_threshold: float) -> Dict[str, Any]:

    print(f"\n{YELLOW}📊 Step 6: Final Evaluation on Test Set...{RESET}")
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_baseline = (y_pred_proba >= 0.5).astype(int)
    y_pred_optimized = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Baseline metrics (threshold = 0.5)
    print(f"\n   {CYAN}Baseline Results (Threshold = 0.5):{RESET}")
    baseline_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_baseline),
        'precision': precision_score(y_test, y_pred_baseline),
        'recall': recall_score(y_test, y_pred_baseline),
        'f1': f1_score(y_test, y_pred_baseline)
    }
    
    cm_baseline = confusion_matrix(y_test, y_pred_baseline)
    tn_base, fp_base, fn_base, tp_base = cm_baseline.ravel()
    
    print(f"      - Accuracy:  {baseline_metrics['accuracy']:.4f}")
    print(f"      - Precision: {baseline_metrics['precision']:.4f}")
    print(f"      - Recall:    {baseline_metrics['recall']:.4f}")
    print(f"      - F1-Score:  {baseline_metrics['f1']:.4f}")
    print(f"      - Confusion: TN={tn_base:,}, FP={fp_base:,}, FN={fn_base:,}, TP={tp_base:,}")
    
    # Optimized metrics
    print(f"\n   {GREEN}Optimized Results (Threshold = {optimal_threshold:.4f}):{RESET}")
    optimized_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_optimized),
        'precision': precision_score(y_test, y_pred_optimized),
        'recall': recall_score(y_test, y_pred_optimized),
        'f1': f1_score(y_test, y_pred_optimized)
    }
    
    cm_optimized = confusion_matrix(y_test, y_pred_optimized)
    tn_opt, fp_opt, fn_opt, tp_opt = cm_optimized.ravel()
    
    print(f"      - Accuracy:  {optimized_metrics['accuracy']:.4f}")
    print(f"      - Precision: {optimized_metrics['precision']:.4f}")
    print(f"      - Recall:    {optimized_metrics['recall']:.4f}")
    print(f"      - F1-Score:  {optimized_metrics['f1']:.4f}")
    print(f"      - Confusion: TN={tn_opt:,}, FP={fp_opt:,}, FN={fn_opt:,}, TP={tp_opt:,}")
    
    # ROC-AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    print(f"\n   📈 ROC-AUC Score: {roc_auc:.4f}")
    
    # Inference latency
    print(f"\n   ⚡ Measuring Inference Latency...")
    sample_size = min(10000, len(X_test))
    X_sample = X_test.iloc[:sample_size]
    
    start_time = time.time()
    _ = model.predict_proba(X_sample)
    elapsed = time.time() - start_time
    
    latency_ms = (elapsed / sample_size) * 1000
    throughput = sample_size / elapsed
    
    print(f"      - Samples tested: {sample_size:,}")
    print(f"      - Total time: {elapsed:.4f} seconds")
    print(f"      - Latency: {latency_ms:.4f} ms/sample")
    print(f"      - Throughput: {throughput:.2f} samples/second")
    
    # Compile results
    results = {
        'baseline': baseline_metrics,
        'optimized': optimized_metrics,
        'confusion_matrices': {
            'baseline': cm_baseline.tolist(),
            'optimized': cm_optimized.tolist()
        },
        'roc_auc': float(roc_auc),
        'inference': {
            'latency_ms': float(latency_ms),
            'throughput': float(throughput)
        },
        'predictions': {
            'y_pred_proba': y_pred_proba,
            'y_pred_baseline': y_pred_baseline,
            'y_pred_optimized': y_pred_optimized
        }
    }
    
    return results


def generate_visualizations(model: xgb.XGBClassifier, results: Dict, 
                           X_train: pd.DataFrame, y_test: pd.Series,
                           optimal_threshold: float, reports_dir: str) -> None:

    print(f"\n{YELLOW}🎨 Step 7: Generating Visualizations...{RESET}")
    
    os.makedirs(reports_dir, exist_ok=True)
    
    # 1. Confusion Matrix Comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    cm_baseline = np.array(results['confusion_matrices']['baseline'])
    cm_optimized = np.array(results['confusion_matrices']['optimized'])
    
    # Baseline
    sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Predicted Benign', 'Predicted Attack'],
                yticklabels=['Actual Benign', 'Actual Attack'],
                annot_kws={'size': 14})
    axes[0].set_title(f'Baseline (Threshold = 0.5)\nF1: {results["baseline"]["f1"]:.4f}',
                      fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    
    # Optimized
    sns.heatmap(cm_optimized, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=['Predicted Benign', 'Predicted Attack'],
                yticklabels=['Actual Benign', 'Actual Attack'],
                annot_kws={'size': 14})
    axes[1].set_title(f'Optimized (Threshold = {optimal_threshold:.4f})\nF1: {results["optimized"]["f1"]:.4f}',
                      fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    cm_path = os.path.join(reports_dir, "xgb_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Confusion Matrix saved to: {cm_path}")
    plt.close()
    
    # 2. Feature Importance (Top 20)
    feature_importance = model.get_booster().get_score(importance_type='gain')
    feature_importance = pd.DataFrame({
        'feature': list(feature_importance.keys()),
        'importance': list(feature_importance.values())
    }).sort_values('importance', ascending=False).head(20)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance, y='feature', x='importance', palette='viridis')
    plt.title('XGBoost Feature Importance (Top 20 - Gain)', fontsize=16, fontweight='bold')
    plt.xlabel('Importance Score (Gain)', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    
    importance_path = os.path.join(reports_dir, "xgb_feature_importance.png")
    plt.savefig(importance_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Feature Importance saved to: {importance_path}")
    plt.close()
    
    # 3. Learning Curve
    eval_results = model.evals_result()
    epochs = len(eval_results['validation_0']['logloss'])
    x_axis = range(0, epochs)
    
    plt.figure(figsize=(12, 6))
    plt.plot(x_axis, eval_results['validation_0']['logloss'], label='Validation LogLoss', linewidth=2)
    plt.axvline(x=model.best_iteration, color='red', linestyle='--', 
                label=f'Best Iteration ({model.best_iteration})', linewidth=2)
    plt.xlabel('Boosting Round', fontsize=12)
    plt.ylabel('LogLoss', fontsize=12)
    plt.title('XGBoost Learning Curve (Validation Set)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    learning_path = os.path.join(reports_dir, "xgb_learning_curve.png")
    plt.savefig(learning_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Learning Curve saved to: {learning_path}")
    plt.close()
    
    # 4. ROC Curve
    y_pred_proba = results['predictions']['y_pred_proba']
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = results['roc_auc']
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'XGBoost (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title('ROC Curve - XGBoost Model', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    roc_path = os.path.join(reports_dir, "xgb_roc_curve.png")
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ ROC Curve saved to: {roc_path}")
    plt.close()
    
    # 5. Precision-Recall Curve
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall_vals, precision_vals, color='blue', lw=2, label='PR Curve')
    
    # Mark optimal threshold
    y_pred_opt = results['predictions']['y_pred_optimized']
    opt_precision = precision_score(y_test, y_pred_opt)
    opt_recall = recall_score(y_test, y_pred_opt)
    
    plt.scatter([opt_recall], [opt_precision], color='red', s=200, marker='*', 
                zorder=5, label=f'Optimal (T={optimal_threshold:.4f})')
    plt.xlabel('Recall (Attack Detection Rate)', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve with Optimal Threshold', fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    pr_path = os.path.join(reports_dir, "xgb_precision_recall.png")
    plt.savefig(pr_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Precision-Recall Curve saved to: {pr_path}")
    plt.close()


def save_artifacts(model: xgb.XGBClassifier, optimal_threshold: float, 
                   val_metrics: Dict, test_metrics: Dict, gpu_config: Dict,
                   models_dir: str) -> None:

    print(f"\n{YELLOW}💾 Step 8: Saving Model and Configuration...{RESET}")
    
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(models_dir, "xgboost_model.pkl")
    joblib.dump(model, model_path)
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"   ✅ Model saved to: {model_path} ({model_size_mb:.2f} MB)")
    
    # Save threshold (plain text)
    threshold_txt_path = os.path.join(models_dir, "threshold_xgb.txt")
    with open(threshold_txt_path, 'w') as f:
        f.write(str(optimal_threshold))
    print(f"   ✅ Threshold saved to: {threshold_txt_path}")
    
    # Save configuration (JSON)
    config = {
        'model_type': 'XGBoost',
        'xgboost_version': xgb.__version__,
        'training_date': '2026-02-15',
        'gpu_config': {
            'gpu_available': gpu_config['gpu_available'],
            'device': gpu_config.get('device_param'),
            'tree_method': gpu_config['tree_method']
        },
        'hyperparameters': {
            'n_estimators': model.n_estimators,
            'max_depth': model.max_depth,
            'learning_rate': model.learning_rate,
            'subsample': model.subsample,
            'colsample_bytree': model.colsample_bytree,
            'scale_pos_weight': float(model.scale_pos_weight),
            'best_iteration': int(model.best_iteration),
            'best_score': float(model.best_score)
        },
        'optimal_threshold': optimal_threshold,
        'validation_metrics': val_metrics,
        'test_metrics': {
            'baseline': test_metrics['baseline'],
            'optimized': test_metrics['optimized'],
            'roc_auc': test_metrics['roc_auc'],
            'inference_latency_ms': test_metrics['inference']['latency_ms'],
            'throughput_samples_per_sec': test_metrics['inference']['throughput']
        }
    }
    
    config_path = os.path.join(models_dir, "xgb_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"   ✅ Configuration saved to: {config_path}")


def print_final_summary(val_metrics: Dict, test_metrics: Dict, optimal_threshold: float) -> None:
    """Print final summary and key takeaways."""
    print_header("🏁 TRAINING COMPLETE - SUMMARY", GREEN)
    
    print(f"{CYAN}📊 Key Results:{RESET}\n")
    
    print(f"   {MAGENTA}Optimal Threshold:{RESET} {optimal_threshold:.4f}")
    print(f"\n   {MAGENTA}Test Set Performance (Optimized):{RESET}")
    print(f"      - Accuracy:  {test_metrics['optimized']['accuracy']:.4f}")
    print(f"      - Precision: {test_metrics['optimized']['precision']:.4f}")
    print(f"      - Recall:    {test_metrics['optimized']['recall']:.4f}")
    print(f"      - F1-Score:  {test_metrics['optimized']['f1']:.4f}")
    print(f"      - ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    
    print(f"\n   {MAGENTA}Performance Metrics:{RESET}")
    print(f"      - Inference Latency: {test_metrics['inference']['latency_ms']:.4f} ms/sample")
    print(f"      - Throughput: {test_metrics['inference']['throughput']:.2f} samples/sec")
    
    # Improvement over baseline
    f1_improvement = test_metrics['optimized']['f1'] - test_metrics['baseline']['f1']
    recall_improvement = test_metrics['optimized']['recall'] - test_metrics['baseline']['recall']
    
    print(f"\n   {MAGENTA}Threshold Optimization Impact:{RESET}")
    print(f"      - F1-Score improvement: {f1_improvement:+.4f}")
    print(f"      - Recall improvement: {recall_improvement:+.4f}")
    
    print(f"\n{GREEN}✅ Model is ready for production deployment!{RESET}")
    print(f"{CYAN}📁 Artifacts saved to 'models/' and 'reports/figures/xgboost/'{RESET}\n")


def main():
    """Main training pipeline."""
    print_header("🚀 XGBoost Training Pipeline - Network Intrusion Detection", CYAN)
    
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    data_dir = os.path.join(project_root, "data", "processed_randomforest")
    models_dir = os.path.join(project_root, "models")
    reports_dir = os.path.join(project_root, "reports", "figures", "xgboost")
    
    try:
        # Step 1: Detect GPU
        gpu_config = detect_gpu_config()
        
        # Step 2: Load data
        train_df, val_df, test_df = load_data(data_dir)
        
        # Step 3: Prepare datasets
        X_train, y_train, X_val, y_val, X_test, y_test, scale_pos_weight = prepare_datasets(
            train_df, val_df, test_df
        )
        
        # Step 4: Train model
        model = train_model(X_train, y_train, X_val, y_val, gpu_config, scale_pos_weight)
        
        # Step 5: Optimize threshold
        optimal_threshold, val_metrics = optimize_threshold(model, X_val, y_val)
        
        # Step 6: Evaluate on test set
        test_results = evaluate_model(model, X_test, y_test, optimal_threshold)
        
        # Step 7: Generate visualizations
        generate_visualizations(model, test_results, X_train, y_test, optimal_threshold, reports_dir)
        
        # Step 8: Save artifacts
        save_artifacts(model, optimal_threshold, val_metrics, test_results, gpu_config, models_dir)
        
        # Final summary
        print_final_summary(val_metrics, test_results, optimal_threshold)
        
    except Exception as e:
        print(f"\n{RED}{'='*80}{RESET}")
        print(f"{RED}❌ FATAL ERROR: {str(e)}{RESET}")
        print(f"{RED}{'='*80}{RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()