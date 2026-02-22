"""
XGBoost Training Script — 3-Class Network Intrusion Detection (CICIDS2017)
Classes: Benign (0) | Volumetric (1) | Semantic (2)

Objective: multi:softprob (outputs class probabilities for OvR ROC-AUC)
Imbalance handling: compute_sample_weight('balanced') instead of scale_pos_weight

Date: 2026-02-22
"""

import os
import sys
import json
import time
import warnings
from typing import Tuple, Dict, Any
from datetime import datetime

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
    roc_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings('ignore')

# ANSI Color Codes for Terminal Output
GREEN = '\033[92m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
RED = '\033[91m'
MAGENTA = '\033[95m'
RESET = '\033[0m'

# Class definitions
CLASS_NAMES = ['Benign', 'Volumetric', 'Semantic']
CLASS_LABELS_FULL = ['Benign (0)', 'Volumetric (1)', 'Semantic (2)']
CLASS_IDS = [0, 1, 2]

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
    """Detect GPU availability for XGBoost acceleration."""

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
        test_data = xgb.DMatrix(np.random.rand(10, 5), label=np.random.randint(0, 3, 10))

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
    """Load train, val, and test CSVs from the 3-class processed data directory."""

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

    # 3-class distribution
    print(f"\n   📊 Training Set Class Distribution:")
    train_dist = train_df['Label'].value_counts().sort_index()
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        count = train_dist.get(float(cls_id), 0)
        pct = count / len(train_df) * 100
        marker = " ⚠️  (minority)" if cls_name == "Semantic" else ""
        print(f"      - {cls_name} ({cls_id}): {count:,.0f} ({pct:.2f}%){marker}")

    return train_df, val_df, test_df


def prepare_datasets(train_df: pd.DataFrame, val_df: pd.DataFrame,
                     test_df: pd.DataFrame) -> Tuple:
    """
    Separate features and targets, calculate sample weights for 3-class imbalance.

    WHY sample_weight instead of scale_pos_weight?
    scale_pos_weight is binary-only. For multi-class imbalance handling,
    we compute per-sample weights inversely proportional to class frequencies.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, sample_weights)
    """
    print(f"\n{YELLOW}⚙️  Step 3: Preparing Features and Calculating Class Weights...{RESET}")

    # Separate features and target
    X_train = train_df.drop('Label', axis=1)
    y_train = train_df['Label'].astype(int)

    X_val = val_df.drop('Label', axis=1)
    y_val = val_df['Label'].astype(int)

    X_test = test_df.drop('Label', axis=1)
    y_test = test_df['Label'].astype(int)

    # Compute balanced sample weights for 3-class imbalance
    # WHY: Semantic class is only ~6.25% — without weighting, XGBoost will ignore it.
    # compute_sample_weight('balanced') assigns weights inversely proportional to
    # class frequencies, giving Semantic samples ~12-16x higher weight than Benign.
    sample_weights = compute_sample_weight('balanced', y_train)

    print(f"   ✅ Features: {X_train.shape[1]} columns")
    print(f"   ✅ Sample Weight Calculation (balanced):")
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        cls_mask = y_train == cls_id
        cls_weight = sample_weights[cls_mask].mean() if cls_mask.any() else 0
        cls_count = cls_mask.sum()
        print(f"      - {cls_name} ({cls_id}): {cls_count:,} samples, avg weight = {cls_weight:.4f}")

    return X_train, y_train, X_val, y_val, X_test, y_test, sample_weights


def train_model(X_train: pd.DataFrame, y_train: pd.Series,
                X_val: pd.DataFrame, y_val: pd.Series,
                gpu_config: Dict[str, Any],
                sample_weights: np.ndarray) -> xgb.XGBClassifier:
    """
    Train XGBoost 3-class model with early stopping.

    WHY multi:softprob over multi:softmax?
    softprob outputs per-class probabilities (N×3 matrix), required for
    ROC-AUC computation and per-class analysis. softmax only outputs
    predicted class indices.
    """
    print(f"\n{YELLOW}🚀 Step 4: Training XGBoost Model (3-Class)...{RESET}")

    # Configure hyperparameters for 3-class
    params = {
        'n_estimators': 1000,
        'max_depth': 7,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'multi:softprob',  # 3-class: outputs probabilities
        'num_class': 3,                 # Required for multi-class objectives
        'eval_metric': 'mlogloss',      # Multi-class log loss
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 1,
        'early_stopping_rounds': 50
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

    # Train with early stopping and sample weights
    print(f"\n   ⏳ Training with Early Stopping (patience=50)...")
    print(f"   ℹ️  Using sample_weight for 3-class imbalance (replaces scale_pos_weight)")
    print(f"   {'─'*76}")

    start_time = time.time()

    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    training_time = time.time() - start_time

    print(f"   {'─'*76}")
    print(f"{GREEN}   ✅ Training Complete!{RESET}")
    print(f"   ⏱️  Training Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"   🌲 Best Iteration: {model.best_iteration}")
    print(f"   📊 Best Score (mlogloss): {model.best_score:.6f}")

    return model


def evaluate_model(model: xgb.XGBClassifier, X_test: pd.DataFrame,
                   y_test: pd.Series) -> Dict[str, Any]:
    """
    Evaluate the 3-class model on the test set.

    Computes:
    - Per-class precision, recall, F1, support
    - Macro and weighted averages
    - 3×3 confusion matrix
    - OvR ROC-AUC per class + macro average
    - Inference latency
    """
    print(f"\n{YELLOW}📊 Step 5: Final Evaluation on Test Set...{RESET}")

    # Get predictions — model.predict() returns class labels (0, 1, 2)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)  # N×3 probability matrix

    # Macro-averaged metrics
    accuracy = accuracy_score(y_test, y_pred)
    macro_precision = precision_score(y_test, y_pred, average='macro')
    macro_recall = recall_score(y_test, y_pred, average='macro')
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\n   {CYAN}3-Class Test Results (Macro-Averaged):{RESET}")
    print(f"      - Accuracy:        {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"      - Macro Precision: {macro_precision:.4f} ({macro_precision*100:.2f}%)")
    print(f"      - Macro Recall:    {macro_recall:.4f} ({macro_recall*100:.2f}%)")
    print(f"      - Macro F1-Score:  {macro_f1:.4f} ({macro_f1*100:.2f}%)")
    print(f"      - Weighted F1:     {weighted_f1:.4f} ({weighted_f1*100:.2f}%)")

    # Per-class metrics
    per_class = {}
    print(f"\n   {CYAN}Per-Class Metrics:{RESET}")
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        cls_p = precision_score(y_test, y_pred, labels=[cls_id], average='macro')
        cls_r = recall_score(y_test, y_pred, labels=[cls_id], average='macro')
        cls_f = f1_score(y_test, y_pred, labels=[cls_id], average='macro')
        support = int((y_test == cls_id).sum())
        per_class[cls_name] = {
            'precision': float(cls_p),
            'recall': float(cls_r),
            'f1': float(cls_f),
            'support': support
        }
        marker = " ⚠️  (minority ~6%)" if cls_name == "Semantic" else ""
        print(f"      - {cls_name:12s}  P={cls_p:.4f}  R={cls_r:.4f}  F1={cls_f:.4f}  n={support:,}{marker}")

    # 3×3 Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n   📋 3×3 Confusion Matrix:")
    print(f"   {'─'*65}")
    header = f"{'':>18s}  {'Pred Benign':>12s}  {'Pred Volum.':>12s}  {'Pred Seman.':>12s}"
    print(f"   {header}")
    print(f"   {'─'*65}")
    for i, cls_name in enumerate(CLASS_NAMES):
        row = "  ".join(f"{cm[i, j]:>12,}" for j in range(3))
        print(f"   {cls_name:>18s}  {row}")
    print(f"   {'─'*65}")

    # Per-class accuracy
    print(f"\n   Per-Class Accuracy:")
    for i, cls_name in enumerate(CLASS_NAMES):
        total = cm[i, :].sum()
        correct = cm[i, i]
        pct = correct / total * 100 if total > 0 else 0
        print(f"      - {cls_name}: {correct:,} / {total:,} ({pct:.2f}%)")

    # OvR ROC-AUC
    y_test_bin = label_binarize(y_test, classes=CLASS_IDS)
    roc_data = {}

    print(f"\n   📈 OvR ROC-AUC per Class:")
    for i, cls_name in enumerate(CLASS_NAMES):
        fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        auc_i = auc(fpr_i, tpr_i)
        roc_data[cls_name] = {'fpr': fpr_i, 'tpr': tpr_i, 'auc': float(auc_i)}
        print(f"      - {cls_name}: AUC = {auc_i:.4f}")

    # Macro-average ROC
    all_fpr = np.unique(np.concatenate([roc_data[n]['fpr'] for n in CLASS_NAMES]))
    mean_tpr = np.zeros_like(all_fpr)
    for n in CLASS_NAMES:
        mean_tpr += np.interp(all_fpr, roc_data[n]['fpr'], roc_data[n]['tpr'])
    mean_tpr /= len(CLASS_NAMES)
    macro_auc = float(auc(all_fpr, mean_tpr))
    roc_data['macro'] = {'fpr': all_fpr, 'tpr': mean_tpr, 'auc': macro_auc}
    print(f"      - Macro-Average AUC: {macro_auc:.4f}")

    # Detailed classification report
    print(f"\n   📊 Detailed Classification Report:")
    print(f"   {'─'*65}")
    print(classification_report(y_test, y_pred,
                                target_names=CLASS_LABELS_FULL,
                                digits=4))

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
        'accuracy': float(accuracy),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'per_class': per_class,
        'confusion_matrix': cm,
        'roc_data': roc_data,
        'macro_auc': macro_auc,
        'inference': {
            'latency_ms': float(latency_ms),
            'throughput': float(throughput)
        },
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'total_samples': len(y_test)
    }

    return results


def generate_visualizations(model: xgb.XGBClassifier, results: Dict,
                            X_train: pd.DataFrame, y_test: pd.Series,
                            reports_dir: str) -> None:
    """Generate 3-class visualizations: confusion matrix, feature importance, learning curve, ROC curves."""

    print(f"\n{YELLOW}🎨 Step 6: Generating Visualizations...{RESET}")

    os.makedirs(reports_dir, exist_ok=True)

    # ========================================================================
    # 1. 3×3 Confusion Matrix Heatmap
    # ========================================================================
    cm = results['confusion_matrix']
    cm_pct = cm / cm.sum() * 100

    annot = np.empty_like(cm, dtype=object)
    for i in range(3):
        for j in range(3):
            annot[i, j] = f'{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)'

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=2, linecolor='white',
                annot_kws={'size': 12})
    plt.title('3-Class Confusion Matrix — XGBoost (Test Set)',
              fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=13)
    plt.xlabel('Predicted Label', fontsize=13)
    plt.tight_layout()

    cm_path = os.path.join(reports_dir, "xgb_3class_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Confusion Matrix saved to: {cm_path}")
    plt.close()

    # ========================================================================
    # 2. Feature Importance (Top 20 by Gain)
    # ========================================================================
    feature_importance = model.get_booster().get_score(importance_type='gain')
    feature_importance = pd.DataFrame({
        'feature': list(feature_importance.keys()),
        'importance': list(feature_importance.values())
    }).sort_values('importance', ascending=False).head(20)

    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance, y='feature', x='importance', palette='viridis')
    plt.title('XGBoost Feature Importance (Top 20 — Gain) — 3-Class',
              fontsize=16, fontweight='bold')
    plt.xlabel('Importance Score (Gain)', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()

    importance_path = os.path.join(reports_dir, "xgb_3class_feature_importance.png")
    plt.savefig(importance_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Feature Importance saved to: {importance_path}")
    plt.close()

    # Print top 10 features
    print(f"\n   📊 Top 10 Features by Gain:")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
        print(f"      {i+1:2d}. {row['feature']:30s} {row['importance']:.4f}")

    # ========================================================================
    # 3. Learning Curve (mlogloss)
    # ========================================================================
    eval_results = model.evals_result()
    metric_key = 'mlogloss'
    epochs = len(eval_results['validation_0'][metric_key])
    x_axis = range(0, epochs)

    plt.figure(figsize=(12, 6))
    plt.plot(x_axis, eval_results['validation_0'][metric_key],
             label='Validation mlogloss', linewidth=2)
    plt.axvline(x=model.best_iteration, color='red', linestyle='--',
                label=f'Best Iteration ({model.best_iteration})', linewidth=2)
    plt.xlabel('Boosting Round', fontsize=12)
    plt.ylabel('Multi-class LogLoss (mlogloss)', fontsize=12)
    plt.title('XGBoost Learning Curve — 3-Class (Validation Set)',
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    learning_path = os.path.join(reports_dir, "xgb_3class_learning_curve.png")
    plt.savefig(learning_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Learning Curve saved to: {learning_path}")
    plt.close()

    # ========================================================================
    # 4. OvR Multi-Class ROC Curves
    # ========================================================================
    roc_data = results['roc_data']
    class_colors = ['#2196F3', '#FF5722', '#4CAF50']  # benign, volumetric, semantic

    plt.figure(figsize=(10, 8))

    for i, cls_name in enumerate(CLASS_NAMES):
        d = roc_data[cls_name]
        plt.plot(d['fpr'], d['tpr'], color=class_colors[i], lw=2,
                 label=f'{cls_name} (AUC = {d["auc"]:.4f})')

    # Macro-average ROC
    d_macro = roc_data['macro']
    plt.plot(d_macro['fpr'], d_macro['tpr'], color='black', lw=2.5, linestyle='--',
             label=f'Macro-Avg (AUC = {d_macro["auc"]:.4f})')

    plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle=':', alpha=0.5)

    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('One-vs-Rest ROC Curves — XGBoost 3-Class', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    roc_path = os.path.join(reports_dir, "xgb_3class_roc_curves.png")
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ ROC Curves saved to: {roc_path}")
    plt.close()


def save_artifacts(model: xgb.XGBClassifier, test_results: Dict,
                   gpu_config: Dict, models_dir: str) -> None:
    """Save the trained model and 3-class configuration JSON."""

    print(f"\n{YELLOW}💾 Step 7: Saving Model and Configuration...{RESET}")

    os.makedirs(models_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(models_dir, "xgb_3class_model.pkl")
    joblib.dump(model, model_path)
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"   ✅ Model saved to: {model_path} ({model_size_mb:.2f} MB)")

    # Build per-class metrics for JSON (must be JSON-serializable)
    per_class_json = {}
    for cls_name in CLASS_NAMES:
        pc = test_results['per_class'][cls_name]
        per_class_json[cls_name] = {
            'precision': pc['precision'],
            'recall': pc['recall'],
            'f1': pc['f1'],
            'support': pc['support']
        }

    # Build ROC-AUC per class for JSON
    roc_auc_json = {}
    for cls_name in CLASS_NAMES:
        roc_auc_json[cls_name] = test_results['roc_data'][cls_name]['auc']
    roc_auc_json['macro'] = test_results['macro_auc']

    # Save configuration (JSON) — no threshold fields
    config = {
        'model_type': 'XGBoost 3-Class',
        'xgboost_version': xgb.__version__,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'classification': '3-class (Benign / Volumetric / Semantic)',
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'imbalance_handling': 'compute_sample_weight(balanced)',
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
            'num_class': 3,
            'best_iteration': int(model.best_iteration),
            'best_score_mlogloss': float(model.best_score)
        },
        'test_metrics': {
            'accuracy': test_results['accuracy'],
            'macro_precision': test_results['macro_precision'],
            'macro_recall': test_results['macro_recall'],
            'macro_f1': test_results['macro_f1'],
            'weighted_f1': test_results['weighted_f1'],
            'macro_roc_auc': test_results['macro_auc'],
            'per_class': per_class_json,
            'roc_auc_per_class': roc_auc_json,
            'inference_latency_ms': test_results['inference']['latency_ms'],
            'throughput_samples_per_sec': test_results['inference']['throughput'],
            'total_test_samples': test_results['total_samples']
        }
    }

    config_path = os.path.join(models_dir, "xgb_3class_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"   ✅ Configuration saved to: {config_path}")


def print_final_summary(test_results: Dict) -> None:
    """Print final 3-class summary and key takeaways."""
    print_header("🏁 TRAINING COMPLETE — 3-CLASS SUMMARY", GREEN)

    print(f"{CYAN}📊 Key Results:{RESET}\n")

    print(f"   {MAGENTA}Test Set Performance (Macro-Averaged):{RESET}")
    print(f"      - Accuracy:        {test_results['accuracy']:.4f}")
    print(f"      - Macro Precision: {test_results['macro_precision']:.4f}")
    print(f"      - Macro Recall:    {test_results['macro_recall']:.4f}")
    print(f"      - Macro F1-Score:  {test_results['macro_f1']:.4f}  ⭐")
    print(f"      - Weighted F1:     {test_results['weighted_f1']:.4f}")
    print(f"      - Macro ROC-AUC:   {test_results['macro_auc']:.4f}")

    print(f"\n   {MAGENTA}Per-Class Performance:{RESET}")
    for cls_name in CLASS_NAMES:
        m = test_results['per_class'][cls_name]
        marker = " ⚠️  (minority class, ~6%)" if cls_name == "Semantic" else ""
        print(f"      - {cls_name:12s}  P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}  n={m['support']:,}{marker}")

    print(f"\n   {MAGENTA}OvR ROC-AUC per Class:{RESET}")
    for cls_name in CLASS_NAMES:
        roc_auc = test_results['roc_data'][cls_name]['auc']
        print(f"      - {cls_name:12s}: {roc_auc:.4f}")
    print(f"      - {'Macro-Avg':12s}: {test_results['macro_auc']:.4f}")

    print(f"\n   {MAGENTA}Inference Performance:{RESET}")
    print(f"      - Latency: {test_results['inference']['latency_ms']:.4f} ms/sample")
    print(f"      - Throughput: {test_results['inference']['throughput']:.2f} samples/sec")

    print(f"\n{GREEN}✅ 3-Class model is ready for production deployment!{RESET}")
    print(f"{CYAN}📁 Artifacts saved to 'models/' and 'reports/figures/xgboost/'{RESET}\n")


def main():
    """Main 3-class training pipeline."""
    print_header("🚀 XGBoost 3-Class Training Pipeline — Network Intrusion Detection", CYAN)
    print(f"   Classes: Benign (0) | Volumetric (1) | Semantic (2)\n")

    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))

    data_dir = os.path.join(project_root, "data", "processed_ml")
    models_dir = os.path.join(project_root, "models")
    reports_dir = os.path.join(project_root, "reports", "figures", "xgboost")

    try:
        # Step 1: Detect GPU
        gpu_config = detect_gpu_config()

        # Step 2: Load data
        train_df, val_df, test_df = load_data(data_dir)

        # Step 3: Prepare datasets (sample_weights replaces scale_pos_weight)
        X_train, y_train, X_val, y_val, X_test, y_test, sample_weights = prepare_datasets(
            train_df, val_df, test_df
        )

        # Step 4: Train model (no threshold optimization — direct 3-class predict)
        model = train_model(X_train, y_train, X_val, y_val, gpu_config, sample_weights)

        # Step 5: Evaluate on test set (no threshold — model.predict() returns 0/1/2)
        test_results = evaluate_model(model, X_test, y_test)

        # Step 6: Generate visualizations
        generate_visualizations(model, test_results, X_train, y_test, reports_dir)

        # Step 7: Save artifacts
        save_artifacts(model, test_results, gpu_config, models_dir)

        # Final summary
        print_final_summary(test_results)

    except Exception as e:
        print(f"\n{RED}{'='*80}{RESET}")
        print(f"{RED}❌ FATAL ERROR: {str(e)}{RESET}")
        print(f"{RED}{'='*80}{RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()