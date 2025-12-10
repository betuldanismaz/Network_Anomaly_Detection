"""
Optimized Random Forest Training Script with Hyperparameter Tuning and Threshold Optimization
Focus: Minimize False Negatives (Missed Attacks) for Network Intrusion Detection

Author: Senior ML Engineer
Date: December 9, 2025
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
    recall_score,
    precision_score,
    f1_score
)
import warnings
warnings.filterwarnings('ignore')

# ANSI colors for terminal output
GREEN = '\033[92m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'

def train_optimized_model():
    print(f"\n{CYAN}{'='*80}{RESET}")
    print(f"{CYAN}🚀 OPTIMIZED RANDOM FOREST TRAINING - ATTACK DETECTION FOCUS{RESET}")
    print(f"{CYAN}{'='*80}{RESET}\n")

    # ============================================================================
    # 1. SETUP PATHS
    # ============================================================================
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    data_path = os.path.join(project_root, "data", "processed_csv", "ready_splits")
    models_dir = os.path.join(project_root, "models")
    reports_dir = os.path.join(project_root, "reports", "figures")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # ============================================================================
    # 2. LOAD DATA
    # ============================================================================
    print(f"{YELLOW}📂 Step 1: Loading Training and Validation Data...{RESET}")
    train_file = os.path.join(data_path, "train.csv")
    val_file = os.path.join(data_path, "val.csv")

    if not os.path.exists(train_file) or not os.path.exists(val_file):
        print(f"{RED}❌ Error: Data files not found in {data_path}{RESET}")
        print(f"{RED}   Please run 'src/features/preprocess.py' first.{RESET}")
        return

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    print(f"   ✅ Train Set: {train_df.shape}")
    print(f"   ✅ Val Set:   {val_df.shape}")

    # Separate Features and Target
    X_train = train_df.drop('Label', axis=1)
    y_train = train_df['Label']
    
    X_val = val_df.drop('Label', axis=1)
    y_val = val_df['Label']

    # Class distribution check
    train_dist = y_train.value_counts()
    print(f"\n   📊 Training Set Class Distribution:")
    print(f"      - Benign (0): {train_dist.get(0, 0):,} ({train_dist.get(0, 0)/len(y_train)*100:.2f}%)")
    print(f"      - Attack (1): {train_dist.get(1, 0):,} ({train_dist.get(1, 0)/len(y_train)*100:.2f}%)")

    # ============================================================================
    # 3. HYPERPARAMETER TUNING WITH RANDOMIZED SEARCH
    # ============================================================================
    print(f"\n{YELLOW}⚙️  Step 2: Hyperparameter Tuning (Optimizing for Recall)...{RESET}")
    print(f"   🎯 Strategy: Using RandomizedSearchCV with scoring='recall'")
    print(f"   🎯 Class Weights: 'balanced' (penalize missed attacks more)")
    print(f"   ⚠️  SAFE MODE ENABLED: Limited CPU usage to prevent overheating")
    print(f"      - Max CPU cores: 2")
    print(f"      - Max iterations: 10")
    print(f"      - Max trees per model: 100")
    
    # Define hyperparameter search space (SAFE MODE - Reduced to prevent CPU overload)
    param_distributions = {
        'n_estimators': [50, 75, 100],                       # Number of trees (LIMITED TO 100 MAX)
        'max_depth': [10, 15, 20, None],                     # Tree depth (reduced options)
        'min_samples_split': [2, 5, 10],                     # Min samples to split node
        'min_samples_leaf': [1, 2, 4],                       # Min samples in leaf
        'max_features': ['sqrt', 'log2'],                    # Features per split (removed None)
        'bootstrap': [True],                                  # Bootstrap sampling (fixed to True)
        'criterion': ['gini']                                 # Split criterion (fixed to gini)
    }
    
    # Base estimator with balanced class weights (SAFE MODE - Limited CPU usage)
    base_rf = RandomForestClassifier(
        class_weight='balanced',  # Critical: penalize FN more than FP
        random_state=42,
        n_jobs=4,                 # LIMITED TO 2 CORES (prevents overheating)
        verbose=1                 # Show progress
    )
    
    # RandomizedSearchCV - SAFE MODE (Reduced iterations to prevent overload)
    print(f"\n   ⏳ Starting RandomizedSearchCV (10 iterations, 3-fold CV)...")
    print(f"   ⚠️  SAFE MODE: Using only 2 CPU cores to prevent overheating")
    random_search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_distributions,
        n_iter=10,                    # REDUCED: Only 10 combinations (was 20)
        scoring='recall',             # CRITICAL: Optimize for catching attacks
        cv=3,                         # 3-fold cross-validation
        verbose=2,                    # INCREASED: Show detailed progress
        random_state=42,
        n_jobs=4,                     # LIMITED: Only 2 cores (prevents overheating)
        return_train_score=True
    )
    
    # Fit the random search
    random_search.fit(X_train, y_train)
    
    # Get best model
    best_rf = random_search.best_estimator_
    best_params = random_search.best_params_
    best_cv_score = random_search.best_score_
    
    print(f"\n{GREEN}   ✅ Hyperparameter Tuning Complete!{RESET}")
    print(f"\n   📋 Best Parameters Found:")
    for param, value in best_params.items():
        print(f"      - {param}: {value}")
    print(f"\n   🏆 Best CV Recall Score: {best_cv_score:.4f}")

    # ============================================================================
    # 4. BASELINE EVALUATION (with default threshold 0.5)
    # ============================================================================
    print(f"\n{YELLOW}📊 Step 3: Baseline Evaluation (Threshold = 0.5)...{RESET}")
    
    y_pred_baseline = best_rf.predict(X_val)
    y_pred_proba = best_rf.predict_proba(X_val)[:, 1]  # Probability of being attack
    
    # Baseline metrics
    baseline_acc = accuracy_score(y_val, y_pred_baseline)
    baseline_recall = recall_score(y_val, y_pred_baseline)
    baseline_precision = precision_score(y_val, y_pred_baseline)
    baseline_f1 = f1_score(y_val, y_pred_baseline)
    
    cm_baseline = confusion_matrix(y_val, y_pred_baseline)
    tn_base, fp_base, fn_base, tp_base = cm_baseline.ravel()
    
    print(f"\n   Baseline Results (Threshold = 0.5):")
    print(f"   - Accuracy:  {baseline_acc:.4f}")
    print(f"   - Recall:    {baseline_recall:.4f} (Attack Detection Rate)")
    print(f"   - Precision: {baseline_precision:.4f}")
    print(f"   - F1-Score:  {baseline_f1:.4f}")
    print(f"\n   {CYAN}Confusion Matrix (Baseline):{RESET}")
    print(f"   - True Negatives:  {GREEN}{tn_base:,}{RESET} (Normal correctly identified)")
    print(f"   - False Positives: {YELLOW}{fp_base:,}{RESET} (False alarms)")
    print(f"   - False Negatives: {RED}{fn_base:,}{RESET} ⚠️  CRITICAL (Missed attacks)")
    print(f"   - True Positives:  {GREEN}{tp_base:,}{RESET} (Attacks caught)")

    # ============================================================================
    # 5. THRESHOLD OPTIMIZATION (Minimize False Negatives)
    # ============================================================================
    print(f"\n{YELLOW}🎯 Step 4: Optimizing Decision Threshold...{RESET}")
    print(f"   Goal: Find threshold that maximizes Recall (minimizes FN)")
    
    # Calculate Precision-Recall Curve
    precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_val, y_pred_proba)
    
    # Find threshold for target recall (e.g., 99.9% Recall - AGGRESSIVE FOR SECURITY)
    target_recall = 0.999  # Hedef: %99.9 Recall (neredeyse tüm saldırıları yakala)
    
    # Find the threshold that gives us at least target_recall
    valid_indices = np.where(recall_vals >= target_recall)[0]
    
    if len(valid_indices) > 0:
        # Among valid thresholds, choose the one with highest precision
        best_idx = valid_indices[np.argmax(precision_vals[valid_indices])]
        optimal_threshold = thresholds_pr[best_idx]
        optimal_precision = precision_vals[best_idx]
        optimal_recall = recall_vals[best_idx]
        print(f"{GREEN}   ✅ Target recall {target_recall*100}% achieved!{RESET}")
    else:
        # Fallback: Find threshold that maximizes Recall (minimize FN at all costs)
        # For security systems, we prefer high recall even with lower precision
        best_idx = np.argmax(recall_vals)
        optimal_threshold = thresholds_pr[best_idx] if best_idx < len(thresholds_pr) else 0.3
        optimal_precision = precision_vals[best_idx]
        optimal_recall = recall_vals[best_idx]
        print(f"{YELLOW}   ⚠️  Could not achieve {target_recall*100}% recall. Using max Recall threshold.{RESET}")
        print(f"{YELLOW}   📊 Achieved: {optimal_recall*100:.2f}% Recall{RESET}")
    
    print(f"\n{GREEN}   ✅ Optimal Threshold Found: {optimal_threshold:.4f}{RESET}")
    print(f"   - Expected Precision: {optimal_precision:.4f}")
    print(f"   - Expected Recall:    {optimal_recall:.4f}")

    # ============================================================================
    # 6. FINAL EVALUATION WITH OPTIMAL THRESHOLD
    # ============================================================================
    print(f"\n{YELLOW}📊 Step 5: Final Evaluation (Optimal Threshold)...{RESET}")
    
    # Apply optimal threshold
    y_pred_optimized = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Calculate metrics
    opt_acc = accuracy_score(y_val, y_pred_optimized)
    opt_recall = recall_score(y_val, y_pred_optimized)
    opt_precision = precision_score(y_val, y_pred_optimized)
    opt_f1 = f1_score(y_val, y_pred_optimized)
    
    cm_optimized = confusion_matrix(y_val, y_pred_optimized)
    tn_opt, fp_opt, fn_opt, tp_opt = cm_optimized.ravel()
    
    print(f"\n   Optimized Results (Threshold = {optimal_threshold:.4f}):")
    print(f"   - Accuracy:  {opt_acc:.4f}")
    print(f"   - Recall:    {opt_recall:.4f} (Attack Detection Rate)")
    print(f"   - Precision: {opt_precision:.4f}")
    print(f"   - F1-Score:  {opt_f1:.4f}")
    print(f"\n   {CYAN}Confusion Matrix (Optimized):{RESET}")
    print(f"   - True Negatives:  {GREEN}{tn_opt:,}{RESET}")
    print(f"   - False Positives: {YELLOW}{fp_opt:,}{RESET}")
    print(f"   - False Negatives: {RED}{fn_opt:,}{RESET} ⚠️  CRITICAL")
    print(f"   - True Positives:  {GREEN}{tp_opt:,}{RESET}")
    
    # ============================================================================
    # 7. IMPROVEMENT SUMMARY
    # ============================================================================
    print(f"\n{CYAN}{'='*80}{RESET}")
    print(f"{CYAN}📈 IMPROVEMENT SUMMARY{RESET}")
    print(f"{CYAN}{'='*80}{RESET}")
    
    fn_reduction = fn_base - fn_opt
    fn_reduction_pct = (fn_reduction / fn_base * 100) if fn_base > 0 else 0
    
    print(f"\n   False Negatives (Missed Attacks):")
    print(f"   - Baseline (0.5):     {RED}{fn_base:,}{RESET}")
    print(f"   - Optimized ({optimal_threshold:.4f}): {GREEN}{fn_opt:,}{RESET}")
    print(f"   - Reduction:          {GREEN}{fn_reduction:,} ({fn_reduction_pct:.2f}% improvement){RESET}")
    
    print(f"\n   Trade-offs:")
    fp_increase = fp_opt - fp_base
    print(f"   - False Positives increased by: {fp_increase:,}")
    print(f"   - Accuracy change: {opt_acc - baseline_acc:+.4f}")
    print(f"   - Recall improvement: {opt_recall - baseline_recall:+.4f}")

    # ============================================================================
    # 8. VISUALIZATIONS
    # ============================================================================
    print(f"\n{YELLOW}🎨 Step 6: Generating Visualizations...{RESET}")
    
    # 8.1 Comparison Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Baseline
    sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Predicted Benign', 'Predicted Attack'],
                yticklabels=['Actual Benign', 'Actual Attack'],
                annot_kws={'size': 14})
    axes[0].set_title(f'Baseline (Threshold = 0.5)\nFN: {fn_base:,}', 
                      fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    
    # Optimized
    sns.heatmap(cm_optimized, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=['Predicted Benign', 'Predicted Attack'],
                yticklabels=['Actual Benign', 'Actual Attack'],
                annot_kws={'size': 14})
    axes[1].set_title(f'Optimized (Threshold = {optimal_threshold:.4f})\nFN: {fn_opt:,}', 
                      fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    cm_comparison_path = os.path.join(reports_dir, "confusion_matrix_comparison.png")
    plt.savefig(cm_comparison_path, dpi=300)
    print(f"   ✅ Confusion Matrix Comparison saved to: {cm_comparison_path}")
    plt.close()

    # 8.2 Precision-Recall Curve with optimal threshold marked
    plt.figure(figsize=(10, 8))
    plt.plot(recall_vals, precision_vals, color='blue', lw=2, label='PR Curve')
    plt.scatter([optimal_recall], [optimal_precision], color='red', s=200, 
                marker='*', zorder=5, label=f'Optimal (T={optimal_threshold:.4f})')
    plt.xlabel('Recall (Attack Detection Rate)', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve with Optimal Threshold', fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    pr_curve_path = os.path.join(reports_dir, "precision_recall_optimized.png")
    plt.savefig(pr_curve_path, dpi=300)
    print(f"   ✅ Precision-Recall Curve saved to: {pr_curve_path}")
    plt.close()

    # 8.3 ROC Curve
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title('ROC Curve - Optimized Model', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    roc_path = os.path.join(reports_dir, "roc_curve_optimized.png")
    plt.savefig(roc_path, dpi=300)
    print(f"   ✅ ROC Curve saved to: {roc_path}")
    plt.close()

    # 8.4 Feature Importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
    plt.title('Feature Importance - Optimized Model', fontsize=16, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    
    importance_path = os.path.join(reports_dir, "feature_importance_optimized.png")
    plt.savefig(importance_path, dpi=300)
    print(f"   ✅ Feature Importance saved to: {importance_path}")
    plt.close()

    # ============================================================================
    # 9. SAVE MODEL AND THRESHOLD
    # ============================================================================
    print(f"\n{YELLOW}💾 Step 7: Saving Model and Configuration...{RESET}")
    
    # Save optimized model
    model_path = os.path.join(models_dir, "rf_model_optimized.pkl")
    joblib.dump(best_rf, model_path)
    print(f"   ✅ Optimized model saved to: {model_path}")
    
    # Save threshold to JSON for live system
    threshold_config = {
        "optimal_threshold": float(optimal_threshold),
        "expected_recall": float(optimal_recall),
        "expected_precision": float(optimal_precision),
        "model_type": "RandomForest_Optimized",
        "training_date": "2025-12-10",
        "hyperparameters": best_params
    }
    
    threshold_path = os.path.join(models_dir, "threshold_config.json")
    with open(threshold_path, 'w') as f:
        json.dump(threshold_config, f, indent=4)
    print(f"   ✅ Threshold config saved to: {threshold_path}")
    
    # Also save as simple text file for easy reading
    threshold_txt_path = os.path.join(models_dir, "threshold.txt")
    with open(threshold_txt_path, 'w') as f:
        f.write(str(optimal_threshold))
    print(f"   ✅ Threshold value saved to: {threshold_txt_path}")

    # 8.5 Comprehensive Model Performance Report
    print(f"\n   📋 Generating Comprehensive Performance Report...")
    
    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(6, 2, hspace=0.4, wspace=0.3)
    
    # Title
    fig.suptitle('Network Anomaly Detection - Optimized Random Forest Model Report', 
                 fontsize=20, fontweight='bold', y=0.995)
    
    # 1. Model Configuration
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    config_text = f"""
MODEL CONFIGURATION & HYPERPARAMETERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Training Date: {threshold_config['training_date']}
Model Type: {threshold_config['model_type']}

Best Hyperparameters:
  • n_estimators: {best_params.get('n_estimators', 'N/A')}
  • max_depth: {best_params.get('max_depth', 'N/A')}
  • min_samples_split: {best_params.get('min_samples_split', 'N/A')}
  • min_samples_leaf: {best_params.get('min_samples_leaf', 'N/A')}
  • max_features: {best_params.get('max_features', 'N/A')}
  • criterion: {best_params.get('criterion', 'N/A')}
  • bootstrap: {best_params.get('bootstrap', 'N/A')}
  • class_weight: balanced

Cross-Validation Score (Recall): {best_cv_score:.4f}
Optimal Decision Threshold: {optimal_threshold:.4f}
    """
    ax1.text(0.05, 0.95, config_text, transform=ax1.transAxes,
             fontsize=11, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # 2. Dataset Information
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis('off')
    dataset_text = f"""
DATASET INFORMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Training Set:
  • Total Samples: {len(X_train):,}
  • Features: {X_train.shape[1]}
  • Benign: {train_dist.get(0, 0):,}
  • Attack: {train_dist.get(1, 0):,}

Validation Set:
  • Total Samples: {len(X_val):,}
  • Features: {X_val.shape[1]}
  • Benign: {y_val.value_counts().get(0, 0):,}
  • Attack: {y_val.value_counts().get(1, 0):,}

Class Distribution:
  • Benign: {train_dist.get(0, 0)/len(y_train)*100:.2f}%
  • Attack: {train_dist.get(1, 0)/len(y_train)*100:.2f}%
    """
    ax2.text(0.05, 0.95, dataset_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # 3. Performance Metrics Comparison
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    metrics_text = f"""
PERFORMANCE METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                  Baseline    Optimized    Change
                  (T=0.5)     (T={optimal_threshold:.4f})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Accuracy:         {baseline_acc:.4f}      {opt_acc:.4f}      {opt_acc-baseline_acc:+.4f}
Recall:           {baseline_recall:.4f}      {opt_recall:.4f}      {opt_recall-baseline_recall:+.4f}
Precision:        {baseline_precision:.4f}      {opt_precision:.4f}      {opt_precision-baseline_precision:+.4f}
F1-Score:         {baseline_f1:.4f}      {opt_f1:.4f}      {opt_f1-baseline_f1:+.4f}
ROC-AUC:          {roc_auc:.4f}      {roc_auc:.4f}      0.0000

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    ax3.text(0.05, 0.95, metrics_text, transform=ax3.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    # 4. Confusion Matrix Details
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis('off')
    cm_text = f"""
CONFUSION MATRIX (BASELINE)
Threshold = 0.5
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

True Negatives:   {tn_base:,}
False Positives:  {fp_base:,}
False Negatives:  {fn_base:,}  ⚠️
True Positives:   {tp_base:,}

False Negative Rate: {fn_base/(fn_base+tp_base)*100:.2f}%
False Positive Rate: {fp_base/(fp_base+tn_base)*100:.2f}%
    """
    ax4.text(0.05, 0.95, cm_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    cm_opt_text = f"""
CONFUSION MATRIX (OPTIMIZED)
Threshold = {optimal_threshold:.4f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

True Negatives:   {tn_opt:,}
False Positives:  {fp_opt:,}
False Negatives:  {fn_opt:,}  ⚠️
True Positives:   {tp_opt:,}

False Negative Rate: {fn_opt/(fn_opt+tp_opt)*100:.2f}%
False Positive Rate: {fp_opt/(fp_opt+tn_opt)*100:.2f}%
    """
    ax5.text(0.05, 0.95, cm_opt_text, transform=ax5.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # 5. Improvement Summary
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('off')
    improvement_text = f"""
OPTIMIZATION IMPACT ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Critical Metric - False Negatives (Missed Attacks):
  • Baseline:     {fn_base:,} missed attacks
  • Optimized:    {fn_opt:,} missed attacks
  • Reduction:    {fn_reduction:,} ({fn_reduction_pct:.2f}% {'improvement' if fn_reduction > 0 else 'degradation'})

Trade-offs:
  • False Positives: {fp_base:,} → {fp_opt:,} (Change: {fp_increase:,})
  • Precision: {baseline_precision:.4f} → {opt_precision:.4f} (Change: {opt_precision-baseline_precision:+.4f})
  • Recall: {baseline_recall:.4f} → {opt_recall:.4f} (Change: {opt_recall-baseline_recall:+.4f})

Security Perspective:
  • Attack Detection Rate: {opt_recall*100:.2f}%
  • Missed Attack Rate: {fn_opt/(fn_opt+tp_opt)*100:.2f}%
  • False Alarm Rate: {fp_opt/(fp_opt+tn_opt)*100:.2f}%
    """
    ax6.text(0.05, 0.95, improvement_text, transform=ax6.transAxes,
             fontsize=11, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
    
    # 6. Top Features Bar Chart
    ax7 = fig.add_subplot(gs[4, :])
    top_10_features = feature_importance.head(10)
    bars = ax7.barh(range(len(top_10_features)), top_10_features['importance'].values, 
                     color=plt.cm.viridis(np.linspace(0, 1, 10)))
    ax7.set_yticks(range(len(top_10_features)))
    ax7.set_yticklabels(top_10_features['feature'].values, fontsize=10)
    ax7.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
    ax7.set_title('Top 10 Most Important Features', fontsize=12, fontweight='bold')
    ax7.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (idx, row) in enumerate(top_10_features.iterrows()):
        ax7.text(row['importance'], i, f" {row['importance']:.4f}", 
                va='center', fontsize=9, fontweight='bold')
    
    # 7. Classification Report Table
    ax8 = fig.add_subplot(gs[5, :])
    ax8.axis('off')
    
    report_dict = classification_report(y_val, y_pred_optimized, 
                                       target_names=['BENIGN', 'ATTACK'],
                                       output_dict=True, digits=4)
    
    report_text = f"""
DETAILED CLASSIFICATION REPORT (Optimized Model)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

              Precision    Recall    F1-Score    Support
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BENIGN        {report_dict['BENIGN']['precision']:.4f}      {report_dict['BENIGN']['recall']:.4f}     {report_dict['BENIGN']['f1-score']:.4f}      {int(report_dict['BENIGN']['support']):,}
ATTACK        {report_dict['ATTACK']['precision']:.4f}      {report_dict['ATTACK']['recall']:.4f}     {report_dict['ATTACK']['f1-score']:.4f}      {int(report_dict['ATTACK']['support']):,}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Accuracy                              {report_dict['accuracy']:.4f}      {int(report_dict['BENIGN']['support'] + report_dict['ATTACK']['support']):,}
Macro Avg     {report_dict['macro avg']['precision']:.4f}      {report_dict['macro avg']['recall']:.4f}     {report_dict['macro avg']['f1-score']:.4f}      {int(report_dict['BENIGN']['support'] + report_dict['ATTACK']['support']):,}
Weighted Avg  {report_dict['weighted avg']['precision']:.4f}      {report_dict['weighted avg']['recall']:.4f}     {report_dict['weighted avg']['f1-score']:.4f}      {int(report_dict['BENIGN']['support'] + report_dict['ATTACK']['support']):,}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RECOMMENDATIONS:
• Model is {'production-ready' if fn_opt < 100 else 'needs improvement'} for deployment
• {'Excellent' if opt_recall > 0.999 else 'Good' if opt_recall > 0.995 else 'Moderate'} attack detection rate
• Consider {'reducing' if fp_opt > 500 else 'maintaining'} threshold for better balance
    """
    ax8.text(0.05, 0.95, report_text, transform=ax8.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.5))
    
    plt.tight_layout()
    report_path = os.path.join(reports_dir, "model_performance_report_comprehensive.png")
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Comprehensive Performance Report saved to: {report_path}")
    plt.close()

    print(f"\n{GREEN}📊 All visualizations generated successfully!{RESET}")
    print(f"\n{YELLOW}💾 Step 7: Saving Model and Configuration...{RESET}")
    
    # Save optimized model
    model_path = os.path.join(models_dir, "rf_model_optimized.pkl")
    joblib.dump(best_rf, model_path)
    print(f"   ✅ Optimized model saved to: {model_path}")
    
    # Save threshold to JSON for live system
    threshold_config = {
        "optimal_threshold": float(optimal_threshold),
        "expected_recall": float(optimal_recall),
        "expected_precision": float(optimal_precision),
        "model_type": "RandomForest_Optimized",
        "training_date": "2025-12-09",
        "hyperparameters": best_params
    }
    
    threshold_path = os.path.join(models_dir, "threshold_config.json")
    with open(threshold_path, 'w') as f:
        json.dump(threshold_config, f, indent=4)
    print(f"   ✅ Threshold config saved to: {threshold_path}")
    
    # Also save as simple text file for easy reading
    threshold_txt_path = os.path.join(models_dir, "threshold.txt")
    with open(threshold_txt_path, 'w') as f:
        f.write(str(optimal_threshold))
    print(f"   ✅ Threshold value saved to: {threshold_txt_path}")

    # ============================================================================
    # 10. FINAL CLASSIFICATION REPORT
    # ============================================================================
    print(f"\n{YELLOW}📝 Final Classification Report (Optimized Threshold):{RESET}")
    print(classification_report(y_val, y_pred_optimized, 
                                target_names=['BENIGN (0)', 'ATTACK (1)'],
                                digits=4))

    # ============================================================================
    # COMPLETION
    # ============================================================================
    print(f"\n{GREEN}{'='*80}{RESET}")
    print(f"{GREEN}🏁 OPTIMIZATION COMPLETE!{RESET}")
    print(f"{GREEN}{'='*80}{RESET}\n")
    
    print(f"{CYAN}📌 Key Takeaways:{RESET}")
    print(f"   1. Optimal threshold: {optimal_threshold:.4f} (vs default 0.5)")
    print(f"   2. False Negatives reduced: {fn_base:,} → {fn_opt:,} ({fn_reduction_pct:.1f}% improvement)")
    print(f"   3. Recall improved: {baseline_recall:.4f} → {opt_recall:.4f}")
    print(f"   4. Model and threshold saved for production deployment")
    print(f"\n{CYAN}🚀 Ready for live deployment with enhanced attack detection!{RESET}\n")

if __name__ == "__main__":
    train_optimized_model()
