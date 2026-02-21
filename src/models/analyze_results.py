"""
3-Class Random Forest — Performance Analytics Suite
Network Intrusion Detection (CICIDS2017)
Classes: Benign (0) | Volumetric (1) | Semantic (2)

Features:
- Interactive metric gauges with color-coded status indicators
- 3×3 confusion matrix heatmap with per-class breakdowns
- Per-class and macro-averaged performance comparison charts
- Security risk assessment with actionable recommendations
- Hyperparameter visualization and model explainability
- Executive summary report with deployment readiness score

Date: 2026-02-22
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Wedge, FancyBboxPatch
from matplotlib.gridspec import GridSpec
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

# Professional plotting configuration
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Professional color palette
COLORS = {
    'primary': '#2C3E50',
    'success': '#27AE60',
    'warning': '#F39C12',
    'danger': '#E74C3C',
    'info': '#3498DB',
    'light': '#ECF0F1',
    'dark': '#34495E',
    'purple': '#9B59B6',
    'teal': '#16A085',
    'benign': '#2196F3',
    'volumetric': '#FF5722',
    'semantic': '#4CAF50',
}

# ANSI colors for terminal output
GREEN = '\033[92m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
RED = '\033[91m'
BOLD = '\033[1m'
RESET = '\033[0m'

# Class definitions
CLASS_NAMES = ['Benign', 'Volumetric', 'Semantic']
CLASS_IDS = [0, 1, 2]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_header():
    """Print professional ASCII art header"""
    header = f"""
{CYAN}{'═'*90}
╔════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                        ║
║     🛡️  NETWORK INTRUSION DETECTION — 3-CLASS RF PERFORMANCE ANALYTICS 🛡️             ║
║                                                                                        ║
║             Classes: Benign (0) | Volumetric (1) | Semantic (2)                        ║
║                                                                                        ║
╚════════════════════════════════════════════════════════════════════════════════════════╝
{'═'*90}{RESET}
    """
    print(header)


def compute_metrics_from_data(y_true, y_pred, y_pred_proba):
    """Compute comprehensive 3-class metrics from actual predictions"""

    # Overall metrics (macro-averaged)
    acc = accuracy_score(y_true, y_pred)
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    weighted_recall = recall_score(y_true, y_pred, average='weighted')
    weighted_precision = precision_score(y_true, y_pred, average='weighted')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Per-class metrics
    per_class = {}
    for cls_id, cls_name in zip(CLASS_IDS, CLASS_NAMES):
        cls_r = recall_score(y_true, y_pred, labels=[cls_id], average='macro')
        cls_p = precision_score(y_true, y_pred, labels=[cls_id], average='macro')
        cls_f = f1_score(y_true, y_pred, labels=[cls_id], average='macro')
        support = int((y_true == cls_id).sum())
        per_class[cls_name] = {
            'precision': cls_p,
            'recall': cls_r,
            'f1': cls_f,
            'support': support
        }

    # OvR ROC-AUC
    y_true_bin = label_binarize(y_true, classes=CLASS_IDS)
    roc_data = {}
    for i, cls_name in enumerate(CLASS_NAMES):
        fpr_i, tpr_i, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        auc_i = auc(fpr_i, tpr_i)
        roc_data[cls_name] = {'fpr': fpr_i, 'tpr': tpr_i, 'auc': auc_i}

    # Macro-average ROC
    all_fpr = np.unique(np.concatenate([roc_data[n]['fpr'] for n in CLASS_NAMES]))
    mean_tpr = np.zeros_like(all_fpr)
    for n in CLASS_NAMES:
        mean_tpr += np.interp(all_fpr, roc_data[n]['fpr'], roc_data[n]['tpr'])
    mean_tpr /= len(CLASS_NAMES)
    macro_auc = auc(all_fpr, mean_tpr)
    roc_data['macro'] = {'fpr': all_fpr, 'tpr': mean_tpr, 'auc': macro_auc}

    metrics = {
        'accuracy': acc,
        'macro_recall': macro_recall,
        'macro_precision': macro_precision,
        'macro_f1': macro_f1,
        'weighted_recall': weighted_recall,
        'weighted_precision': weighted_precision,
        'weighted_f1': weighted_f1,
        'confusion_matrix': cm,
        'per_class': per_class,
        'roc_data': roc_data,
        'macro_auc': macro_auc,
        'total_samples': len(y_true),
    }

    return metrics


def get_deployment_status(metrics):
    """Determine deployment readiness status for 3-class model"""

    macro_f1 = metrics['macro_f1']
    accuracy = metrics['accuracy']
    semantic_f1 = metrics['per_class']['Semantic']['f1']

    # Score based on macro F1, accuracy, and minority class performance
    score = (
        macro_f1 * 35 +
        accuracy * 25 +
        semantic_f1 * 25 +
        metrics['macro_auc'] * 15
    ) * 100

    if score >= 95 and macro_f1 >= 0.95 and semantic_f1 >= 0.90:
        status = "EXCELLENT"
        color = COLORS['success']
        icon = "✓"
        recommendation = "Ready for Production Deployment"
    elif score >= 90 and macro_f1 >= 0.90 and semantic_f1 >= 0.80:
        status = "GOOD"
        color = COLORS['info']
        icon = "○"
        recommendation = "Acceptable for Deployment"
    elif score >= 85:
        status = "ACCEPTABLE"
        color = COLORS['warning']
        icon = "△"
        recommendation = "Monitor Performance Closely"
    else:
        status = "NEEDS IMPROVEMENT"
        color = COLORS['danger']
        icon = "⚠"
        recommendation = "Further Tuning Required"

    return {
        'status': status,
        'color': color,
        'icon': icon,
        'score': score,
        'recommendation': recommendation
    }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_gauge_chart(ax, value, title, subtitle="", threshold_good=0.95, threshold_ok=0.90):
    """Create a professional circular gauge chart"""

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')

    # Determine color based on value
    if value >= threshold_good:
        color = COLORS['success']
    elif value >= threshold_ok:
        color = COLORS['warning']
    else:
        color = COLORS['danger']

    # Draw background arc
    bg_arc = Wedge((0, 0), 1, 0, 180, width=0.25, facecolor='lightgray', alpha=0.3)
    ax.add_patch(bg_arc)

    # Draw value arc
    angle = 180 * value
    value_arc = Wedge((0, 0), 1, 0, angle, width=0.25, facecolor=color, alpha=0.9)
    ax.add_patch(value_arc)

    # Add value text
    ax.text(0, 0.1, f'{value*100:.2f}%',
            ha='center', va='center', fontsize=36, fontweight='bold', color=color)

    # Add title and subtitle
    ax.text(0, -0.5, title,
            ha='center', va='center', fontsize=14, fontweight='bold')
    if subtitle:
        ax.text(0, -0.7, subtitle,
                ha='center', va='center', fontsize=9, style='italic', color='gray')


def create_confusion_matrix_3class(ax, cm):
    """Create 3×3 confusion matrix heatmap"""

    # Calculate percentages
    cm_pct = cm / cm.sum() * 100

    # Create custom annotations
    annot = np.empty_like(cm, dtype=object)
    for i in range(3):
        for j in range(3):
            annot[i, j] = f'{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)'

    # Plot heatmap
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=True,
                square=True, linewidths=2, linecolor='white',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                ax=ax, cbar_kws={'label': 'Count'},
                annot_kws={'size': 10})

    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('3-Class Confusion Matrix', fontsize=13, fontweight='bold', pad=15)

    # Add per-class accuracy below the matrix
    per_class_acc = [cm[i, i] / cm[i, :].sum() * 100 for i in range(3)]
    text = " | ".join(f"{CLASS_NAMES[i]}: {per_class_acc[i]:.1f}%" for i in range(3))
    ax.text(0.5, -0.12, f"Per-Class Accuracy: {text}",
            transform=ax.transAxes, ha='center',
            fontsize=9, color=COLORS['dark'], fontweight='bold')


def create_metrics_comparison_3class(ax, metrics):
    """Create horizontal bar chart comparing macro and per-class metrics"""

    metric_names = []
    metric_values = []
    colors_list = []

    # Macro-averaged metrics
    macro_items = [
        ('Macro F1', metrics['macro_f1'], COLORS['purple']),
        ('Macro Recall', metrics['macro_recall'], COLORS['success']),
        ('Macro Precision', metrics['macro_precision'], COLORS['info']),
        ('Accuracy', metrics['accuracy'], COLORS['teal']),
        ('Macro ROC-AUC', metrics['macro_auc'], COLORS['primary']),
    ]

    for name, val, col in macro_items:
        metric_names.append(name)
        metric_values.append(val * 100)
        colors_list.append(col)

    # Create bars
    bars = ax.barh(metric_names, metric_values, color=colors_list,
                   alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, metric_values)):
        ax.text(val + 0.5, i, f'{val:.2f}%', va='center', fontsize=10, fontweight='bold')

    ax.set_xlim(0, 105)
    ax.set_xlabel('Score (%)', fontsize=11, fontweight='bold')
    ax.set_title('Overall Performance Metrics', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)


def create_per_class_comparison(ax, metrics):
    """Create grouped bar chart showing per-class P/R/F1"""

    x = np.arange(len(CLASS_NAMES))
    width = 0.25

    precisions = [metrics['per_class'][n]['precision'] * 100 for n in CLASS_NAMES]
    recalls = [metrics['per_class'][n]['recall'] * 100 for n in CLASS_NAMES]
    f1s = [metrics['per_class'][n]['f1'] * 100 for n in CLASS_NAMES]

    bars1 = ax.bar(x - width, precisions, width, label='Precision',
                   color=COLORS['info'], alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x, recalls, width, label='Recall',
                   color=COLORS['success'], alpha=0.8, edgecolor='black')
    bars3 = ax.bar(x + width, f1s, width, label='F1-Score',
                   color=COLORS['purple'], alpha=0.8, edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=11, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=11)
    ax.set_title('Per-Class Performance Breakdown', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add value labels on top of bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')


def create_roc_curves_3class(ax, roc_data):
    """Create One-vs-Rest multi-class ROC curves"""

    class_colors = [COLORS['benign'], COLORS['volumetric'], COLORS['semantic']]

    for i, cls_name in enumerate(CLASS_NAMES):
        d = roc_data[cls_name]
        ax.plot(d['fpr'], d['tpr'], color=class_colors[i], lw=2,
                label=f'{cls_name} (AUC = {d["auc"]:.4f})')

    # Macro-average
    d_macro = roc_data['macro']
    ax.plot(d_macro['fpr'], d_macro['tpr'], color='black', lw=2.5, linestyle='--',
            label=f'Macro-Avg (AUC = {d_macro["auc"]:.4f})')

    ax.plot([0, 1], [0, 1], color='grey', lw=1, linestyle=':', alpha=0.5)

    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('One-vs-Rest ROC Curves', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)


def create_status_indicator(ax, deployment_status):
    """Create deployment status indicator"""

    status = deployment_status['status']
    color = deployment_status['color']
    icon = deployment_status['icon']
    score = deployment_status['score']

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Draw status box
    status_box = FancyBboxPatch((0.1, 0.25), 0.8, 0.5,
                                boxstyle="round,pad=0.02",
                                facecolor=color, edgecolor=COLORS['dark'],
                                linewidth=3, alpha=0.2)
    ax.add_patch(status_box)

    # Add content
    ax.text(0.5, 0.85, 'DEPLOYMENT STATUS', ha='center', fontsize=12, fontweight='bold')
    ax.text(0.5, 0.55, icon, ha='center', fontsize=60, fontweight='bold', color=color)
    ax.text(0.5, 0.35, status, ha='center', fontsize=18, fontweight='bold', color=color)
    ax.text(0.5, 0.15, f'Score: {score:.1f}/100', ha='center', fontsize=11, fontweight='bold')
    ax.text(0.5, 0.05, deployment_status['recommendation'],
            ha='center', fontsize=10, style='italic')


def create_hyperparameters_table(ax, config):
    """Create hyperparameters display table"""

    ax.axis('off')

    hyperparams = config['hyperparameters']

    text = "╔═══════════════════════════════════╗\n"
    text += "║     HYPERPARAMETERS               ║\n"
    text += "╠═══════════════════════════════════╣\n\n"

    param_map = {
        'n_estimators': 'Trees',
        'max_depth': 'Max Depth',
        'min_samples_split': 'Min Split',
        'min_samples_leaf': 'Min Leaf',
        'max_features': 'Max Features',
        'criterion': 'Criterion',
        'bootstrap': 'Bootstrap'
    }

    for key, label in param_map.items():
        value = hyperparams.get(key, 'N/A')
        if value is None:
            value = 'None'
        text += f"  {label:15s}: {value}\n"

    text += f"  {'Class Weight':15s}: balanced\n"
    text += f"\n  {'Scoring':15s}: f1_macro\n"
    text += f"  {'Classification':15s}: 3-class\n"

    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.7', facecolor='lightyellow',
                     edgecolor=COLORS['dark'], linewidth=2, alpha=0.9))


def create_risk_assessment_3class(ax, metrics):
    """Create security risk assessment panel for 3-class model"""

    ax.axis('off')

    pc = metrics['per_class']
    semantic_recall = pc['Semantic']['recall'] * 100
    volumetric_recall = pc['Volumetric']['recall'] * 100
    benign_recall = pc['Benign']['recall'] * 100

    # Risk level based on minority class (Semantic) performance
    if semantic_recall >= 95:
        risk_level = "MINIMAL"
        risk_color = COLORS['success']
    elif semantic_recall >= 85:
        risk_level = "LOW"
        risk_color = COLORS['info']
    elif semantic_recall >= 70:
        risk_level = "MODERATE"
        risk_color = COLORS['warning']
    else:
        risk_level = "HIGH"
        risk_color = COLORS['danger']

    cm = metrics['confusion_matrix']

    text = f"""
╔════════════════════════════════════════════════════╗
║       3-CLASS SECURITY RISK ASSESSMENT             ║
╚════════════════════════════════════════════════════╝

Risk Level: {risk_level}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DETECTION PERFORMANCE BY CLASS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ Benign detection:      {benign_recall:.2f}% ({cm[0,0]:,} / {cm[0,:].sum():,})
✓ Volumetric detection:  {volumetric_recall:.2f}% ({cm[1,1]:,} / {cm[1,:].sum():,})
{'✓' if semantic_recall >= 85 else '⚠'} Semantic detection:    {semantic_recall:.2f}% ({cm[2,2]:,} / {cm[2,:].sum():,})

Macro F1-Score:          {metrics['macro_f1']*100:.2f}%
Macro ROC-AUC:           {metrics['macro_auc']*100:.2f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ class_weight='balanced' used for imbalance handling
✓ f1_macro scoring ensures all classes contribute equally
⚠ Semantic class (~6%) is the hardest to detect
⚠ Monitor Semantic recall closely in production
    """

    text_color = 'darkgreen' if risk_level in ["MINIMAL", "LOW"] else 'black'

    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            fontsize=9.5, verticalalignment='top', family='monospace',
            color=text_color,
            bbox=dict(boxstyle='round,pad=0.7', facecolor='white',
                     edgecolor=risk_color, linewidth=3, alpha=0.95))


# ============================================================================
# MAIN DASHBOARD FUNCTIONS
# ============================================================================

def create_executive_dashboard(config, metrics, reports_dir):
    """Create comprehensive executive dashboard for 3-class RF"""

    print(f"\n{YELLOW}📊 Step 3: Creating Executive Dashboard...{RESET}")

    deployment_status = get_deployment_status(metrics)

    # Create large figure with professional layout
    fig = plt.figure(figsize=(26, 20))
    gs = GridSpec(5, 4, figure=fig, hspace=0.45, wspace=0.35,
                  left=0.05, right=0.95, top=0.94, bottom=0.04)

    # Main title
    title = f"Network Intrusion Detection — 3-Class Random Forest Analytics Dashboard\n"
    title += f"Model: {config['model_type']} | Date: {config['training_date']} | "
    title += f"Score: {deployment_status['score']:.1f}/100 ({deployment_status['status']})"
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)

    # ========================================================================
    # ROW 1: MAIN METRICS GAUGES
    # ========================================================================

    # Macro F1 Gauge
    ax1 = fig.add_subplot(gs[0, 0])
    create_gauge_chart(ax1, metrics['macro_f1'], 'MACRO F1',
                       'Primary Optimization Target', 0.95, 0.90)

    # Accuracy Gauge
    ax2 = fig.add_subplot(gs[0, 1])
    create_gauge_chart(ax2, metrics['accuracy'], 'ACCURACY',
                       'Overall Correctness', 0.98, 0.95)

    # Macro Recall Gauge
    ax3 = fig.add_subplot(gs[0, 2])
    create_gauge_chart(ax3, metrics['macro_recall'], 'MACRO RECALL',
                       'Average Detection Rate', 0.95, 0.90)

    # Semantic F1 Gauge (minority class focus)
    semantic_f1 = metrics['per_class']['Semantic']['f1']
    ax4 = fig.add_subplot(gs[0, 3])
    create_gauge_chart(ax4, semantic_f1, 'SEMANTIC F1',
                       'Minority Class (~6%)', 0.90, 0.80)

    # ========================================================================
    # ROW 2: CONFUSION MATRIX + OVERALL METRICS
    # ========================================================================

    # 3×3 Confusion Matrix
    ax5 = fig.add_subplot(gs[1, 0:2])
    create_confusion_matrix_3class(ax5, metrics['confusion_matrix'])

    # Overall Metrics Comparison
    ax6 = fig.add_subplot(gs[1, 2:4])
    create_metrics_comparison_3class(ax6, metrics)

    # ========================================================================
    # ROW 3: PER-CLASS COMPARISON + ROC CURVES
    # ========================================================================

    # Per-class P/R/F1 grouped bar chart
    ax7 = fig.add_subplot(gs[2, 0:2])
    create_per_class_comparison(ax7, metrics)

    # OvR ROC Curves
    ax8 = fig.add_subplot(gs[2, 2:4])
    create_roc_curves_3class(ax8, metrics['roc_data'])

    # ========================================================================
    # ROW 4: STATUS + HYPERPARAMETERS
    # ========================================================================

    # Deployment Status
    ax9 = fig.add_subplot(gs[3, 0])
    create_status_indicator(ax9, deployment_status)

    # Hyperparameters
    ax10 = fig.add_subplot(gs[3, 1])
    create_hyperparameters_table(ax10, config)

    # Risk Assessment (spans right half)
    ax11 = fig.add_subplot(gs[3, 2:4])
    create_risk_assessment_3class(ax11, metrics)

    # ========================================================================
    # ROW 5: CLASS DISTRIBUTION PIE + DETAILED CLASSIFICATION REPORT
    # ========================================================================

    # Class distribution pie chart
    ax12 = fig.add_subplot(gs[4, 0])
    supports = [metrics['per_class'][n]['support'] for n in CLASS_NAMES]
    pie_colors = [COLORS['benign'], COLORS['volumetric'], COLORS['semantic']]
    wedges, texts, autotexts = ax12.pie(
        supports, labels=CLASS_NAMES, colors=pie_colors,
        autopct='%1.1f%%', startangle=90, pctdistance=0.85,
        textprops={'fontsize': 10})
    for t in autotexts:
        t.set_fontweight('bold')
    ax12.set_title('Test Set Class Distribution', fontsize=12, fontweight='bold')

    # Detailed classification report text
    ax13 = fig.add_subplot(gs[4, 1:4])
    ax13.axis('off')

    pc = metrics['per_class']
    report_text = f"""
CLASSIFICATION REPORT (Test Set)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                 Precision    Recall    F1-Score    Support
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Benign (0)       {pc['Benign']['precision']:.4f}      {pc['Benign']['recall']:.4f}     {pc['Benign']['f1']:.4f}      {pc['Benign']['support']:,}
Volumetric (1)   {pc['Volumetric']['precision']:.4f}      {pc['Volumetric']['recall']:.4f}     {pc['Volumetric']['f1']:.4f}      {pc['Volumetric']['support']:,}
Semantic (2)     {pc['Semantic']['precision']:.4f}      {pc['Semantic']['recall']:.4f}     {pc['Semantic']['f1']:.4f}      {pc['Semantic']['support']:,}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Accuracy                               {metrics['accuracy']:.4f}      {metrics['total_samples']:,}
Macro Avg        {metrics['macro_precision']:.4f}      {metrics['macro_recall']:.4f}     {metrics['macro_f1']:.4f}      {metrics['total_samples']:,}
Weighted Avg     {metrics['weighted_precision']:.4f}      {metrics['weighted_recall']:.4f}     {metrics['weighted_f1']:.4f}      {metrics['total_samples']:,}
    """
    ax13.text(0.02, 0.95, report_text, transform=ax13.transAxes,
              fontsize=10, verticalalignment='top', family='monospace',
              bbox=dict(boxstyle='round,pad=0.5', facecolor='lavender',
                       edgecolor=COLORS['dark'], linewidth=2, alpha=0.9))

    # Save dashboard
    dashboard_path = os.path.join(reports_dir, "rf_3class_executive_dashboard.png")
    plt.savefig(dashboard_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"{GREEN}✅ Executive Dashboard saved: {dashboard_path}{RESET}")
    plt.close()

    return dashboard_path


def create_detailed_numeric_report(config, metrics, reports_dir):
    """Create detailed textual numeric report for 3-class model"""

    print(f"\n{YELLOW}📋 Step 4: Creating Detailed Numeric Report...{RESET}")

    deployment_status = get_deployment_status(metrics)
    pc = metrics['per_class']
    cm = metrics['confusion_matrix']

    fig, ax = plt.subplots(figsize=(18, 16))
    ax.axis('off')

    report_text = f"""
╔════════════════════════════════════════════════════════════════════════════════════════════╗
║         NETWORK INTRUSION DETECTION — 3-CLASS RF COMPREHENSIVE NUMERICAL REPORT           ║
╚════════════════════════════════════════════════════════════════════════════════════════════╝

MODEL INFORMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model Type:                  {config['model_type']}
Training Date:               {config['training_date']}
Classification:              3-Class (Benign / Volumetric / Semantic)
Optimization Metric:         f1_macro (Macro-Averaged F1-Score)
Deployment Score:            {deployment_status['score']:.2f}/100 ({deployment_status['status']})


MACRO-AVERAGED PERFORMANCE METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Accuracy:                    {metrics['accuracy']:.6f}  ({metrics['accuracy']*100:.2f}%)
Macro Precision:             {metrics['macro_precision']:.6f}  ({metrics['macro_precision']*100:.2f}%)
Macro Recall:                {metrics['macro_recall']:.6f}  ({metrics['macro_recall']*100:.2f}%)
Macro F1-Score:              {metrics['macro_f1']:.6f}  ({metrics['macro_f1']*100:.2f}%)  ★ PRIMARY
Macro ROC-AUC:               {metrics['macro_auc']:.6f}  ({metrics['macro_auc']*100:.2f}%)


PER-CLASS PERFORMANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                  Precision    Recall    F1-Score    Support
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Benign (0):       {pc['Benign']['precision']:.4f}      {pc['Benign']['recall']:.4f}     {pc['Benign']['f1']:.4f}      {pc['Benign']['support']:,}   (~80%)
Volumetric (1):   {pc['Volumetric']['precision']:.4f}      {pc['Volumetric']['recall']:.4f}     {pc['Volumetric']['f1']:.4f}      {pc['Volumetric']['support']:,}   (~14%)
Semantic (2):     {pc['Semantic']['precision']:.4f}      {pc['Semantic']['recall']:.4f}     {pc['Semantic']['f1']:.4f}      {pc['Semantic']['support']:,}    (~6%) ⚠ minority


CONFUSION MATRIX (Test Set)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Samples:               {metrics['total_samples']:,}

                    Pred Benign    Pred Volumetric    Pred Semantic    Total
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Actual Benign       {cm[0,0]:>10,}        {cm[0,1]:>10,}          {cm[0,2]:>10,}      {cm[0,:].sum():>10,}
Actual Volumetric   {cm[1,0]:>10,}        {cm[1,1]:>10,}          {cm[1,2]:>10,}      {cm[1,:].sum():>10,}
Actual Semantic     {cm[2,0]:>10,}        {cm[2,1]:>10,}          {cm[2,2]:>10,}      {cm[2,:].sum():>10,}

Per-Class Accuracy:
  • Benign:     {cm[0,0]:,} / {cm[0,:].sum():,} = {cm[0,0]/cm[0,:].sum()*100:.2f}%
  • Volumetric: {cm[1,1]:,} / {cm[1,:].sum():,} = {cm[1,1]/cm[1,:].sum()*100:.2f}%
  • Semantic:   {cm[2,2]:,} / {cm[2,:].sum():,} = {cm[2,2]/cm[2,:].sum()*100:.2f}%


ROC-AUC (One-vs-Rest)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Benign AUC:                  {metrics['roc_data']['Benign']['auc']:.6f}
Volumetric AUC:              {metrics['roc_data']['Volumetric']['auc']:.6f}
Semantic AUC:                {metrics['roc_data']['Semantic']['auc']:.6f}
Macro-Average AUC:           {metrics['macro_auc']:.6f}


HYPERPARAMETERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Algorithm:                   Random Forest Classifier (3-Class)
n_estimators:                {config['hyperparameters'].get('n_estimators', 'N/A')}
max_depth:                   {config['hyperparameters'].get('max_depth', 'None (unlimited)')}
min_samples_split:           {config['hyperparameters'].get('min_samples_split', 'N/A')}
min_samples_leaf:            {config['hyperparameters'].get('min_samples_leaf', 'N/A')}
max_features:                {config['hyperparameters'].get('max_features', 'N/A')}
criterion:                   {config['hyperparameters'].get('criterion', 'N/A')}
class_weight:                balanced


DEPLOYMENT RECOMMENDATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status:                      {deployment_status['status']} ({deployment_status['score']:.1f}/100)
Recommendation:              {deployment_status['recommendation']}

✓ STRENGTHS:
  • Macro F1 = {metrics['macro_f1']*100:.2f}% — balanced across all 3 classes
  • class_weight='balanced' compensates for imbalanced Semantic class
  • No binary threshold needed — direct 3-class predict()

⚠ CONSIDERATIONS:
  • Semantic class (~6%) is the minority — monitor closely
  • Volumetric attacks may have sub-types with variable detectability
  • Consider ensemble or stacking with other classifiers for improvement

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Report Generated: {config['training_date']}
Model Version: {config['model_type']}
Analysis Tool: 3-Class Performance Analytics Suite v4.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """

    ax.text(0.02, 0.99, report_text, transform=ax.transAxes,
            fontsize=8.5, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='white',
                     edgecolor=COLORS['primary'], linewidth=2.5, alpha=1))

    plt.tight_layout()
    report_path = os.path.join(reports_dir, "rf_3class_detailed_numeric_report.png")
    plt.savefig(report_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"{GREEN}✅ Detailed Report saved: {report_path}{RESET}")
    plt.close()

    return report_path


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def load_model_and_data():
    """Load model, config, and test data; generate predictions"""

    print_header()

    print(f"{YELLOW}📂 Step 1: Loading Model, Config, and Test Data...{RESET}")

    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))

    models_dir = os.path.join(project_root, "models")
    reports_dir = os.path.join(project_root, "reports", "figures")
    data_path = os.path.join(project_root, "data", "processed_ml")

    os.makedirs(reports_dir, exist_ok=True)

    # Load config
    config_path = os.path.join(models_dir, "rf_3class_config.json")
    if not os.path.exists(config_path):
        print(f"\n{RED}{'='*90}")
        print(f"❌ ERROR: Model configuration not found!")
        print(f"{'='*90}")
        print(f"\n📁 Missing file: {config_path}")
        print(f"\n{YELLOW}💡 SOLUTION:{RESET}")
        print(f"   1. Train the 3-class model first:")
        print(f"      {CYAN}python src/models/train_randomforest.py{RESET}")
        print(f"   2. Ensure training completes successfully")
        print(f"   3. Verify rf_3class_config.json exists in models/ directory")
        print(f"\n{RED}{'='*90}{RESET}\n")
        return None

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Load model
    model_path = os.path.join(models_dir, "rf_3class_model.pkl")
    if not os.path.exists(model_path):
        print(f"\n{RED}❌ ERROR: Model file not found: {model_path}{RESET}")
        return None

    model = joblib.load(model_path)

    # Load test data
    test_file = os.path.join(data_path, "test.csv")
    if not os.path.exists(test_file):
        print(f"\n{RED}❌ ERROR: Test data not found: {test_file}{RESET}")
        return None

    test_df = pd.read_csv(test_file)
    X_test = test_df.drop('Label', axis=1)
    y_test = test_df['Label']

    print(f"{GREEN}✅ All files loaded successfully{RESET}\n")
    print(f"{CYAN}═══════════════════════════════════════════════════════════════{RESET}")
    print(f"{CYAN}Model Configuration Summary:{RESET}")
    print(f"{CYAN}═══════════════════════════════════════════════════════════════{RESET}")
    print(f"  📦 Model Type:        {BOLD}{config['model_type']}{RESET}")
    print(f"  📅 Training Date:     {config['training_date']}")
    print(f"  🎯 Scoring Metric:    {config.get('scoring_metric', 'f1_macro')}")
    print(f"  🎯 Best CV F1 Macro:  {config.get('best_cv_f1_macro', 'N/A')}")
    print(f"  📊 Test Samples:      {len(y_test):,}")
    print(f"  📊 Features:          {X_test.shape[1]}")
    print(f"{CYAN}═══════════════════════════════════════════════════════════════{RESET}")

    # Generate predictions
    print(f"\n{YELLOW}📊 Step 2: Generating Predictions on Test Set...{RESET}")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Compute metrics
    metrics = compute_metrics_from_data(y_test.values, y_pred, y_pred_proba)

    print(f"{GREEN}✅ Predictions and metrics computed{RESET}")
    print(f"\n{CYAN}Quick Summary:{RESET}")
    print(f"   • Accuracy:        {metrics['accuracy']*100:6.2f}%")
    print(f"   • Macro Precision: {metrics['macro_precision']*100:6.2f}%")
    print(f"   • Macro Recall:    {metrics['macro_recall']*100:6.2f}%")
    print(f"   • Macro F1-Score:  {metrics['macro_f1']*100:6.2f}%  ⭐")
    print(f"   • Macro ROC-AUC:   {metrics['macro_auc']*100:6.2f}%")

    print(f"\n{CYAN}Per-Class F1 Scores:{RESET}")
    for cls_name in CLASS_NAMES:
        marker = " ⚠️  (minority)" if cls_name == "Semantic" else ""
        f1 = metrics['per_class'][cls_name]['f1']
        print(f"   • {cls_name:12s}: {f1*100:.2f}%{marker}")

    return config, metrics, reports_dir


def main():
    """Main execution function"""

    try:
        # Step 1-2: Load and compute
        result = load_model_and_data()
        if result is None:
            return

        config, metrics, reports_dir = result

        # Step 3: Create executive dashboard
        dashboard_path = create_executive_dashboard(config, metrics, reports_dir)

        # Step 4: Create detailed numeric report
        report_path = create_detailed_numeric_report(config, metrics, reports_dir)

        # Final summary
        print(f"\n{GREEN}{'═'*90}")
        print(f"{'🎉 3-CLASS ANALYSIS COMPLETE! 🎉':^90}")
        print(f"{'═'*90}{RESET}\n")

        print(f"{CYAN}📊 Generated Reports:{RESET}")
        print(f"   1. {BOLD}Executive Dashboard{RESET}:    rf_3class_executive_dashboard.png")
        print(f"   2. {BOLD}Detailed Numeric Report{RESET}: rf_3class_detailed_numeric_report.png")

        print(f"\n{CYAN}📁 Location:{RESET} reports/figures/")

        print(f"\n{GREEN}✓ All visualizations generated successfully!{RESET}")
        print(f"{GREEN}✓ Reports are production-ready for stakeholder review{RESET}")

        # Key metrics
        deployment = get_deployment_status(metrics)
        pc = metrics['per_class']
        print(f"\n{CYAN}═══════════════════════════════════════════════════════════════{RESET}")
        print(f"{CYAN}Key Performance Indicators:{RESET}")
        print(f"{CYAN}═══════════════════════════════════════════════════════════════{RESET}")
        print(f"  🎯 Deployment Score:  {deployment['score']:.1f}/100 ({deployment['status']})")
        print(f"  📈 Macro F1-Score:    {metrics['macro_f1']*100:.2f}%")
        print(f"  📈 Macro Recall:      {metrics['macro_recall']*100:.2f}%")
        print(f"  📈 Macro Precision:   {metrics['macro_precision']*100:.2f}%")
        print(f"  📈 Macro ROC-AUC:     {metrics['macro_auc']*100:.2f}%")
        print(f"  📈 Accuracy:          {metrics['accuracy']*100:.2f}%")

        print(f"\n{CYAN}Per-Class Performance:{RESET}")
        for cls_name in CLASS_NAMES:
            m = pc[cls_name]
            marker = " ⚠️  (minority class, ~6%)" if cls_name == "Semantic" else ""
            print(f"  {cls_name:12s}  P={m['precision']*100:.2f}%  R={m['recall']*100:.2f}%  F1={m['f1']*100:.2f}%{marker}")

        print(f"{CYAN}═══════════════════════════════════════════════════════════════{RESET}\n")

    except Exception as e:
        print(f"\n{RED}{'='*90}")
        print(f"❌ ERROR: An unexpected error occurred!")
        print(f"{'='*90}")
        print(f"\n{str(e)}")
        print(f"\n{RED}{'='*90}{RESET}\n")
        raise


if __name__ == "__main__":
    main()