"""
Random Forest Training Script for Network Intrusion Detection System
3-CLASS CLASSIFICATION: Benign (0), Volumetric (1), Semantic (2)
Date: 2026-02-21
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
    recall_score,
    precision_score,
    f1_score
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# ANSI colors for terminal output
GREEN = '\033[92m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'

# Class label mapping
CLASS_NAMES = ['Benign', 'Volumetric', 'Semantic']
CLASS_LABELS_FULL = ['Benign (0)', 'Volumetric (1)', 'Semantic (2)']


def train_3class_model():
    print(f"\n{CYAN}{'='*80}{RESET}")
    print(f"{CYAN}🚀 3-CLASS RANDOM FOREST TRAINING - NETWORK INTRUSION DETECTION{RESET}")
    print(f"{CYAN}   Classes: Benign (0) | Volumetric (1) | Semantic (2){RESET}")
    print(f"{CYAN}{'='*80}{RESET}\n")

    # ============================================================================
    # 1. SETUP PATHS
    # ============================================================================
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))

    data_path = os.path.join(project_root, "data", "processed_ml")
    models_dir = os.path.join(project_root, "models")
    reports_dir = os.path.join(project_root, "reports", "figures")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # ============================================================================
    # 2. LOAD DATA
    # ============================================================================
    print(f"{YELLOW}📂 Step 1: Loading Training, Validation, and Test Data...{RESET}")
    train_file = os.path.join(data_path, "train.csv")
    val_file = os.path.join(data_path, "val.csv")
    test_file = os.path.join(data_path, "test.csv")

    for fpath, fname in [(train_file, "train.csv"), (val_file, "val.csv"), (test_file, "test.csv")]:
        if not os.path.exists(fpath):
            print(f"{RED}❌ Error: {fname} not found in {data_path}{RESET}")
            print(f"{RED}   Please run the 3-class preprocessing pipeline first.{RESET}")
            return

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)

    print(f"   ✅ Train Set: {train_df.shape}")
    print(f"   ✅ Val Set:   {val_df.shape}")
    print(f"   ✅ Test Set:  {test_df.shape}")

    # Separate Features and Target
    X_train = train_df.drop('Label', axis=1)
    y_train = train_df['Label']

    X_val = val_df.drop('Label', axis=1)
    y_val = val_df['Label']

    X_test = test_df.drop('Label', axis=1)
    y_test = test_df['Label']

    # Class distribution check
    train_dist = y_train.value_counts().sort_index()
    val_dist = y_val.value_counts().sort_index()
    test_dist = y_test.value_counts().sort_index()

    print(f"\n   📊 Training Set Class Distribution:")
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        count = train_dist.get(cls_id, 0)
        pct = count / len(y_train) * 100
        print(f"      - {cls_name} ({cls_id}): {count:,} ({pct:.2f}%)")

    print(f"\n   📊 Validation Set Class Distribution:")
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        count = val_dist.get(cls_id, 0)
        pct = count / len(y_val) * 100
        print(f"      - {cls_name} ({cls_id}): {count:,} ({pct:.2f}%)")

    print(f"\n   📊 Test Set Class Distribution:")
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        count = test_dist.get(cls_id, 0)
        pct = count / len(y_test) * 100
        print(f"      - {cls_name} ({cls_id}): {count:,} ({pct:.2f}%)")

    # ============================================================================
    # 3. HYPERPARAMETER TUNING WITH RANDOMIZED SEARCH
    # ============================================================================
    print(f"\n{YELLOW}⚙️  Step 2: Hyperparameter Tuning (Optimizing for Macro F1)...{RESET}")
    print(f"   🎯 Strategy: Using RandomizedSearchCV with scoring='f1_macro'")
    print(f"   🎯 Class Weights: 'balanced' (critical for imbalanced Semantic class)")
    print(f"      - Max iterations: 10")
    print(f"      - Cross-validation folds: 3")
    print(f"      - Parallel jobs: 4")

    # Define hyperparameter search space
    param_distributions = {
        'n_estimators': [50, 75, 100],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True],
        'criterion': ['gini']
    }

    # Base estimator with balanced class weights
    base_rf = RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
        n_jobs=4,
        verbose=1
    )

    # RandomizedSearchCV with f1_macro scoring
    print(f"\n   ⏳ Starting RandomizedSearchCV (10 iterations, 3-fold CV)...")
    random_search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_distributions,
        n_iter=10,
        scoring='f1_macro',
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=4,
        return_train_score=True
    )

    # Fit the random search
    train_start = time.time()
    random_search.fit(X_train, y_train)
    train_time = time.time() - train_start

    # Get best model
    best_rf = random_search.best_estimator_
    best_params = random_search.best_params_
    best_cv_score = random_search.best_score_

    print(f"\n{GREEN}   ✅ Hyperparameter Tuning Complete! (Training time: {train_time:.1f}s){RESET}")
    print(f"\n   📋 Best Parameters Found:")
    for param, value in best_params.items():
        print(f"      - {param}: {value}")
    print(f"\n   🏆 Best CV Macro F1 Score: {best_cv_score:.4f}")

    # ============================================================================
    # 4. EVALUATION ON VALIDATION SET
    # ============================================================================
    print(f"\n{YELLOW}📊 Step 3: Evaluation on Validation Set...{RESET}")

    y_pred_val = best_rf.predict(X_val)
    y_pred_proba_val = best_rf.predict_proba(X_val)

    # Validation metrics (macro-averaged)
    val_acc = accuracy_score(y_val, y_pred_val)
    val_recall = recall_score(y_val, y_pred_val, average='macro')
    val_precision = precision_score(y_val, y_pred_val, average='macro')
    val_f1 = f1_score(y_val, y_pred_val, average='macro')

    print(f"\n   Validation Results (Macro-Averaged):")
    print(f"   - Accuracy:       {val_acc:.4f}")
    print(f"   - Macro Recall:   {val_recall:.4f}")
    print(f"   - Macro Precision:{val_precision:.4f}")
    print(f"   - Macro F1-Score: {val_f1:.4f}")

    # Per-class metrics on validation
    print(f"\n   {CYAN}Per-Class Metrics (Validation):{RESET}")
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        cls_recall = recall_score(y_val, y_pred_val, labels=[cls_id], average='macro')
        cls_precision = precision_score(y_val, y_pred_val, labels=[cls_id], average='macro')
        cls_f1 = f1_score(y_val, y_pred_val, labels=[cls_id], average='macro')
        print(f"   - {cls_name:12s}  P={cls_precision:.4f}  R={cls_recall:.4f}  F1={cls_f1:.4f}")

    # Confusion matrix
    cm_val = confusion_matrix(y_val, y_pred_val)
    print(f"\n   {CYAN}Confusion Matrix (Validation):{RESET}")
    print(f"   {'':>12s}  {'Pred Benign':>12s}  {'Pred Volum.':>12s}  {'Pred Seman.':>12s}")
    for i, cls_name in enumerate(CLASS_NAMES):
        row = "  ".join(f"{cm_val[i, j]:>12,}" for j in range(3))
        print(f"   {cls_name:>12s}  {row}")

    # ============================================================================
    # 5. EVALUATION ON TEST SET
    # ============================================================================
    print(f"\n{YELLOW}📊 Step 4: Final Evaluation on Test Set...{RESET}")

    y_pred_test = best_rf.predict(X_test)
    y_pred_proba_test = best_rf.predict_proba(X_test)

    # Test metrics (macro-averaged)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_recall = recall_score(y_test, y_pred_test, average='macro')
    test_precision = precision_score(y_test, y_pred_test, average='macro')
    test_f1 = f1_score(y_test, y_pred_test, average='macro')

    print(f"\n   Test Results (Macro-Averaged):")
    print(f"   - Accuracy:       {test_acc:.4f}")
    print(f"   - Macro Recall:   {test_recall:.4f}")
    print(f"   - Macro Precision:{test_precision:.4f}")
    print(f"   - Macro F1-Score: {test_f1:.4f}")

    # Per-class metrics on test
    per_class_metrics = {}
    print(f"\n   {CYAN}Per-Class Metrics (Test):{RESET}")
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        cls_recall = recall_score(y_test, y_pred_test, labels=[cls_id], average='macro')
        cls_precision = precision_score(y_test, y_pred_test, labels=[cls_id], average='macro')
        cls_f1 = f1_score(y_test, y_pred_test, labels=[cls_id], average='macro')
        per_class_metrics[cls_name] = {
            'precision': float(cls_precision),
            'recall': float(cls_recall),
            'f1': float(cls_f1)
        }
        print(f"   - {cls_name:12s}  P={cls_precision:.4f}  R={cls_recall:.4f}  F1={cls_f1:.4f}")

    # Confusion matrix on test
    cm_test = confusion_matrix(y_test, y_pred_test)
    print(f"\n   {CYAN}Confusion Matrix (Test):{RESET}")
    print(f"   {'':>12s}  {'Pred Benign':>12s}  {'Pred Volum.':>12s}  {'Pred Seman.':>12s}")
    for i, cls_name in enumerate(CLASS_NAMES):
        row = "  ".join(f"{cm_test[i, j]:>12,}" for j in range(3))
        print(f"   {cls_name:>12s}  {row}")

    # ============================================================================
    # 6. VISUALIZATIONS
    # ============================================================================
    print(f"\n{YELLOW}🎨 Step 5: Generating Visualizations...{RESET}")

    # 6.1 Confusion Matrix (3×3 Heatmap)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                annot_kws={'size': 14})
    plt.title('3-Class Confusion Matrix — Random Forest (Test Set)',
              fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=13)
    plt.xlabel('Predicted Label', fontsize=13)
    plt.tight_layout()

    cm_path = os.path.join(reports_dir, "rf_3class_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    print(f"   ✅ Confusion Matrix saved to: {cm_path}")
    plt.close()

    # 6.2 One-vs-Rest (OvR) Multi-Class ROC Curves
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = 3

    fpr = {}
    tpr = {}
    roc_auc = {}
    colors = ['#2196F3', '#FF5722', '#4CAF50']

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba_test[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

    plt.figure(figsize=(10, 8))
    for i, cls_name in enumerate(CLASS_NAMES):
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                 label=f'{cls_name} (AUC = {roc_auc[i]:.4f})')

    plt.plot(fpr['macro'], tpr['macro'], color='black', lw=2.5, linestyle='--',
             label=f'Macro-Average (AUC = {roc_auc["macro"]:.4f})')
    plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle=':', alpha=0.5)

    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('One-vs-Rest ROC Curves — Random Forest (Test Set)',
              fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    roc_path = os.path.join(reports_dir, "rf_3class_roc_curves.png")
    plt.savefig(roc_path, dpi=300)
    print(f"   ✅ ROC Curves saved to: {roc_path}")
    plt.close()

    # 6.3 Feature Importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_rf.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
    plt.title('Feature Importance — 3-Class Random Forest', fontsize=16, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()

    importance_path = os.path.join(reports_dir, "rf_3class_feature_importance.png")
    plt.savefig(importance_path, dpi=300)
    print(f"   ✅ Feature Importance saved to: {importance_path}")
    plt.close()

    # ============================================================================
    # 7. COMPREHENSIVE PERFORMANCE REPORT FIGURE
    # ============================================================================
    print(f"\n   📋 Generating Comprehensive Performance Report...")

    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(6, 2, hspace=0.4, wspace=0.3)

    # Title
    fig.suptitle('Network Intrusion Detection — 3-Class Random Forest Performance Report',
                 fontsize=20, fontweight='bold', y=0.995)

    # 7.1 Model Configuration
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    config_text = f"""
MODEL CONFIGURATION & HYPERPARAMETERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Training Date: 2026-02-21
Model Type: RandomForest_3Class
Classification: 3-Class (Benign / Volumetric / Semantic)
Scoring Metric: f1_macro
Training Time: {train_time:.1f} seconds

Best Hyperparameters:
  • n_estimators: {best_params.get('n_estimators', 'N/A')}
  • max_depth: {best_params.get('max_depth', 'N/A')}
  • min_samples_split: {best_params.get('min_samples_split', 'N/A')}
  • min_samples_leaf: {best_params.get('min_samples_leaf', 'N/A')}
  • max_features: {best_params.get('max_features', 'N/A')}
  • criterion: {best_params.get('criterion', 'N/A')}
  • class_weight: balanced

Best CV Macro F1 Score: {best_cv_score:.4f}
    """
    ax1.text(0.05, 0.95, config_text, transform=ax1.transAxes,
             fontsize=11, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # 7.2 Dataset Information
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis('off')
    dataset_text = f"""
DATASET INFORMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Training Set:
  • Total Samples: {len(X_train):,}
  • Features: {X_train.shape[1]}
  • Benign:     {train_dist.get(0, 0):,} ({train_dist.get(0, 0)/len(y_train)*100:.1f}%)
  • Volumetric: {train_dist.get(1, 0):,} ({train_dist.get(1, 0)/len(y_train)*100:.1f}%)
  • Semantic:   {train_dist.get(2, 0):,} ({train_dist.get(2, 0)/len(y_train)*100:.1f}%)

Test Set:
  • Total Samples: {len(X_test):,}
  • Features: {X_test.shape[1]}
  • Benign:     {test_dist.get(0, 0):,} ({test_dist.get(0, 0)/len(y_test)*100:.1f}%)
  • Volumetric: {test_dist.get(1, 0):,} ({test_dist.get(1, 0)/len(y_test)*100:.1f}%)
  • Semantic:   {test_dist.get(2, 0):,} ({test_dist.get(2, 0)/len(y_test)*100:.1f}%)
    """
    ax2.text(0.05, 0.95, dataset_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    # 7.3 Performance Metrics
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    metrics_text = f"""
PERFORMANCE METRICS (Test Set)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Overall:
  • Accuracy:        {test_acc:.4f}
  • Macro Precision: {test_precision:.4f}
  • Macro Recall:    {test_recall:.4f}
  • Macro F1-Score:  {test_f1:.4f}
  • Macro ROC-AUC:   {roc_auc['macro']:.4f}

Per-Class ROC-AUC:
  • Benign:     {roc_auc[0]:.4f}
  • Volumetric: {roc_auc[1]:.4f}
  • Semantic:   {roc_auc[2]:.4f}
    """
    ax3.text(0.05, 0.95, metrics_text, transform=ax3.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    # 7.4 Per-Class Detailed Metrics
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    report_dict = classification_report(y_test, y_pred_test,
                                        target_names=CLASS_LABELS_FULL,
                                        output_dict=True, digits=4)

    per_class_text = f"""
PER-CLASS PERFORMANCE (Test Set)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                 Precision    Recall    F1-Score    Support
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Benign (0)       {report_dict['Benign (0)']['precision']:.4f}      {report_dict['Benign (0)']['recall']:.4f}     {report_dict['Benign (0)']['f1-score']:.4f}      {int(report_dict['Benign (0)']['support']):,}
Volumetric (1)   {report_dict['Volumetric (1)']['precision']:.4f}      {report_dict['Volumetric (1)']['recall']:.4f}     {report_dict['Volumetric (1)']['f1-score']:.4f}      {int(report_dict['Volumetric (1)']['support']):,}
Semantic (2)     {report_dict['Semantic (2)']['precision']:.4f}      {report_dict['Semantic (2)']['recall']:.4f}     {report_dict['Semantic (2)']['f1-score']:.4f}      {int(report_dict['Semantic (2)']['support']):,}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Accuracy                                {test_acc:.4f}      {len(y_test):,}
Macro Avg        {report_dict['macro avg']['precision']:.4f}      {report_dict['macro avg']['recall']:.4f}     {report_dict['macro avg']['f1-score']:.4f}      {len(y_test):,}
Weighted Avg     {report_dict['weighted avg']['precision']:.4f}      {report_dict['weighted avg']['recall']:.4f}     {report_dict['weighted avg']['f1-score']:.4f}      {len(y_test):,}
    """
    ax4.text(0.05, 0.95, per_class_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.5))

    # 7.5 Confusion Matrix Details
    ax5 = fig.add_subplot(gs[3, :])
    ax5.axis('off')
    cm_detail_text = f"""
CONFUSION MATRIX BREAKDOWN (Test Set)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                    Predicted Benign    Predicted Volumetric    Predicted Semantic
Actual Benign       {cm_test[0,0]:>12,}         {cm_test[0,1]:>12,}           {cm_test[0,2]:>12,}
Actual Volumetric   {cm_test[1,0]:>12,}         {cm_test[1,1]:>12,}           {cm_test[1,2]:>12,}
Actual Semantic     {cm_test[2,0]:>12,}         {cm_test[2,1]:>12,}           {cm_test[2,2]:>12,}

Per-Class Accuracy:
  • Benign correctly classified:     {cm_test[0,0]:,} / {cm_test[0,:].sum():,} ({cm_test[0,0]/cm_test[0,:].sum()*100:.2f}%)
  • Volumetric correctly classified: {cm_test[1,1]:,} / {cm_test[1,:].sum():,} ({cm_test[1,1]/cm_test[1,:].sum()*100:.2f}%)
  • Semantic correctly classified:   {cm_test[2,2]:,} / {cm_test[2,:].sum():,} ({cm_test[2,2]/cm_test[2,:].sum()*100:.2f}%)
    """
    ax5.text(0.05, 0.95, cm_detail_text, transform=ax5.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    # 7.6 Top Features Bar Chart
    ax6 = fig.add_subplot(gs[4, :])
    top_10_features = feature_importance.head(10)
    bars = ax6.barh(range(len(top_10_features)), top_10_features['importance'].values,
                    color=plt.cm.viridis(np.linspace(0, 1, 10)))
    ax6.set_yticks(range(len(top_10_features)))
    ax6.set_yticklabels(top_10_features['feature'].values, fontsize=10)
    ax6.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
    ax6.set_title('Top 10 Most Important Features', fontsize=12, fontweight='bold')
    ax6.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (idx, row) in enumerate(top_10_features.iterrows()):
        ax6.text(row['importance'], i, f" {row['importance']:.4f}",
                 va='center', fontsize=9, fontweight='bold')

    # 7.7 Recommendations
    ax7 = fig.add_subplot(gs[5, :])
    ax7.axis('off')

    semantic_f1 = per_class_metrics['Semantic']['f1']
    rec_text = f"""
RECOMMENDATIONS & NOTES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• Macro F1-Score: {test_f1:.4f} — {'Excellent' if test_f1 > 0.95 else 'Good' if test_f1 > 0.90 else 'Moderate' if test_f1 > 0.80 else 'Needs improvement'}
• Semantic class (minority, ~6%): F1={semantic_f1:.4f} — {'Strong' if semantic_f1 > 0.90 else 'Acceptable' if semantic_f1 > 0.80 else 'Needs attention'} detection
• class_weight='balanced' used to compensate for class imbalance
• Model uses predict() directly — no binary threshold optimization needed for 3-class
• All metrics computed with average='macro' to equally weight each class
    """
    ax7.text(0.05, 0.95, rec_text, transform=ax7.transAxes,
             fontsize=11, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))

    plt.tight_layout()
    report_path = os.path.join(reports_dir, "rf_3class_performance_report.png")
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Comprehensive Performance Report saved to: {report_path}")
    plt.close()

    print(f"\n{GREEN}📊 All visualizations generated successfully!{RESET}")

    # ============================================================================
    # 8. SAVE MODEL AND CONFIGURATION
    # ============================================================================
    print(f"\n{YELLOW}💾 Step 6: Saving Model and Configuration...{RESET}")

    # Save trained model
    model_path = os.path.join(models_dir, "rf_3class_model.pkl")
    joblib.dump(best_rf, model_path)
    print(f"   ✅ 3-Class model saved to: {model_path}")

    # Save configuration JSON (no threshold — direct predict())
    config = {
        "model_type": "RandomForest_3Class",
        "classification": "3-class",
        "classes": {str(i): name for i, name in enumerate(CLASS_NAMES)},
        "training_date": "2026-02-21",
        "training_time_seconds": round(train_time, 1),
        "scoring_metric": "f1_macro",
        "best_cv_f1_macro": round(float(best_cv_score), 4),
        "hyperparameters": {k: (int(v) if isinstance(v, (np.integer,)) else
                            float(v) if isinstance(v, (np.floating,)) else v)
                           for k, v in best_params.items()},
        "test_metrics": {
            "accuracy": round(float(test_acc), 4),
            "macro_precision": round(float(test_precision), 4),
            "macro_recall": round(float(test_recall), 4),
            "macro_f1": round(float(test_f1), 4),
            "macro_roc_auc": round(float(roc_auc['macro']), 4)
        },
        "per_class_metrics": {
            name: {k: round(v, 4) for k, v in metrics.items()}
            for name, metrics in per_class_metrics.items()
        },
        "per_class_roc_auc": {
            CLASS_NAMES[i]: round(float(roc_auc[i]), 4) for i in range(n_classes)
        },
        "dataset": {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "n_features": X_train.shape[1],
            "feature_names": list(X_train.columns)
        }
    }

    config_path = os.path.join(models_dir, "rf_3class_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"   ✅ Config saved to: {config_path}")

    # ============================================================================
    # 9. FINAL CLASSIFICATION REPORT
    # ============================================================================
    print(f"\n{YELLOW}📝 Final Classification Report (Test Set):{RESET}")
    print(classification_report(y_test, y_pred_test,
                                target_names=CLASS_LABELS_FULL,
                                digits=4))

    # ============================================================================
    # COMPLETION
    # ============================================================================
    print(f"\n{GREEN}{'='*80}{RESET}")
    print(f"{GREEN}🏁 3-CLASS TRAINING COMPLETE!{RESET}")
    print(f"{GREEN}{'='*80}{RESET}\n")

    print(f"{CYAN}📌 Key Takeaways:{RESET}")
    print(f"   1. Best CV Macro F1: {best_cv_score:.4f}")
    print(f"   2. Test Macro F1:    {test_f1:.4f}")
    print(f"   3. Test Accuracy:    {test_acc:.4f}")
    print(f"   4. Macro ROC-AUC:    {roc_auc['macro']:.4f}")

    print(f"\n{CYAN}📊 Per-Class Performance (Test):{RESET}")
    for cls_name in CLASS_NAMES:
        m = per_class_metrics[cls_name]
        marker = " ⚠️  (minority class)" if cls_name == "Semantic" else ""
        print(f"   - {cls_name:12s}  P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}{marker}")

    print(f"\n{CYAN}💾 Saved Artifacts:{RESET}")
    print(f"   - Model:       {model_path}")
    print(f"   - Config:      {config_path}")
    print(f"   - Confusion:   {cm_path}")
    print(f"   - ROC Curves:  {roc_path}")
    print(f"   - Features:    {importance_path}")
    print(f"   - Report:      {report_path}")
    print(f"\n{CYAN}🚀 Ready for deployment with 3-class intrusion detection!{RESET}\n")


if __name__ == "__main__":
    train_3class_model()
