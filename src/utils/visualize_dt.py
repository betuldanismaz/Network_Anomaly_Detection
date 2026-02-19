import os
import sys
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import confusion_matrix

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def setup_output_directories(base_reports_dir: str) -> dict:
    reports_path = Path(base_reports_dir)
    figures_dir = reports_path / "figures"
    text_dir = reports_path / "text_exports"
    
    figures_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directories ready:")
    print(f"   - Figures: {figures_dir}")
    print(f"   - Text exports: {text_dir}")
    
    return {
        'figures': figures_dir,
        'text_exports': text_dir
    }


def load_model(model_path: str) -> DecisionTreeClassifier:
    model_file = Path(model_path)
    
    if not model_file.exists():
        raise FileNotFoundError(
            f"❌ Model file not found: {model_file}\n"
            f"   Please train the model first using train_dt.py"
        )
    
    print(f"📦 Loading trained model from: {model_file}")
    
    try:
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        
        if not isinstance(model, DecisionTreeClassifier):
            raise TypeError(f"Expected DecisionTreeClassifier, got {type(model)}")
        
        print(f"   ✓ Model loaded successfully!")
        print(f"   - Tree depth: {model.tree_.max_depth}")
        print(f"   - Number of features: {model.n_features_in_}")
        print(f"   - Number of nodes: {model.tree_.node_count}")
        
        return model
        
    except Exception as e:
        raise Exception(f"❌ Error loading model: {str(e)}")


def load_test_data(data_path: str) -> tuple:
    data_file = Path(data_path)
    
    if not data_file.exists():
        raise FileNotFoundError(f"❌ Test data not found: {data_file}")
    
    print(f"\n📂 Loading test data from: {data_file}")
    df = pd.read_csv(data_file)
    print(f"   ✓ Loaded {len(df):,} test samples")
    
    if 'Label' not in df.columns:
        raise ValueError("❌ 'Label' column not found in test data")
    
    y_test = df['Label'].values
    X_test = df.drop('Label', axis=1).values
    feature_names = df.drop('Label', axis=1).columns.tolist()
    
    print(f"   - Features: {len(feature_names)}")
    print(f"   - Class distribution: {np.bincount(y_test)}")
    
    return X_test, y_test, feature_names


def visualize_tree_structure(
    model: DecisionTreeClassifier,
    feature_names: list,
    output_path: Path,
    max_depth: int = 4
) -> None:
    print(f"\n🌳 Generating tree structure visualization (top {max_depth} levels)...")
    
    plt.figure(figsize=(20, 12))
    
    plot_tree(
        model,
        max_depth=max_depth,
        filled=True,
        rounded=True,
        feature_names=feature_names,
        class_names=['Normal', 'Attack'],
        fontsize=10,
        proportion=True
    )
    
    plt.title(
        f'Decision Tree Structure (Top {max_depth} Levels)\n'
        f'Total Tree Depth: {model.tree_.max_depth} | Total Nodes: {model.tree_.node_count}',
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved: {output_path}")


def visualize_feature_importance(
    model: DecisionTreeClassifier,
    feature_names: list,
    output_path: Path,
    top_n: int = 10
) -> None:
    print(f"\n📊 Generating feature importance visualization (Top {top_n})...")
    
    importances = model.feature_importances_
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(12, 8))
    
    ax = sns.barplot(
        data=importance_df,
        x='Importance',
        y='Feature',
        palette='viridis'
    )
    
    for i, (importance, feature) in enumerate(zip(importance_df['Importance'], importance_df['Feature'])):
        ax.text(importance, i, f' {importance:.4f}', va='center', fontweight='bold')
    
    plt.title(
        f'Top {top_n} Most Important Features\n'
        'Feature Importance = Contribution to Decision-Making',
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
    plt.ylabel('Feature Name', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved: {output_path}")
    print(f"\n   📌 Top 5 Most Important Features:")
    for idx, row in importance_df.head(5).iterrows():
        print(f"      {row['Feature']}: {row['Importance']:.4f}")


def visualize_confusion_matrix(
    model: DecisionTreeClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_path: Path
) -> None:
    print(f"\n🎯 Generating confusion matrix...")
    
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(10, 8))
    
    annotations = np.array([
        [f'{count:,}\n({percent:.2f}%)' for count, percent in zip(row_counts, row_percents)]
        for row_counts, row_percents in zip(cm, cm_percent)
    ])
    
    sns.heatmap(
        cm,
        annot=annotations,
        fmt='s',
        cmap='Blues',
        square=True,
        linewidths=2,
        cbar_kws={'label': 'Number of Samples'},
        xticklabels=['Normal', 'Attack'],
        yticklabels=['Normal', 'Attack']
    )
    
    plt.title(
        'Confusion Matrix - Decision Tree Performance\n'
        'Shows Prediction Accuracy Breakdown on Test Set',
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    plt.ylabel('Actual Class', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=12, fontweight='bold')
    
    tn, fp, fn, tp = cm.ravel()
    interpretation = (
        f'\n\nInterpretation:\n'
        f'True Negatives (TN): {tn:,} - Normal traffic correctly identified\n'
        f'False Positives (FP): {fp:,} - Normal traffic incorrectly flagged\n'
        f'False Negatives (FN): {fn:,} - Attacks missed (Critical!)\n'
        f'True Positives (TP): {tp:,} - Attacks correctly detected'
    )
    
    plt.figtext(0.5, -0.15, interpretation, ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved: {output_path}")
    print(f"\n   📊 Confusion Matrix Breakdown:")
    print(f"      TN (Normal → Normal):  {tn:,} ({cm_percent[0][0]:.2f}%)")
    print(f"      FP (Normal → Attack):  {fp:,} ({cm_percent[0][1]:.2f}%)")
    print(f"      FN (Attack → Normal):  {fn:,} ({cm_percent[1][0]:.2f}%)")
    print(f"      TP (Attack → Attack):  {tp:,} ({cm_percent[1][1]:.2f}%)")


def export_tree_rules(
    model: DecisionTreeClassifier,
    feature_names: list,
    output_path: Path
) -> None:
    print(f"\n📜 Exporting decision tree rules to text...")
    
    tree_rules = export_text(
        model,
        feature_names=feature_names,
        spacing=3
    )
    
    header = f"""
{'='*80}
DECISION TREE RULES - NETWORK INTRUSION DETECTION
{'='*80}

Model Information:
- Tree Depth: {model.tree_.max_depth}
- Total Nodes: {model.tree_.node_count}
- Leaf Nodes: {model.tree_.n_leaves}
- Features Used: {model.n_features_in_}

How to Read:
- Each line shows a decision rule (threshold test)
- Indentation shows the tree hierarchy (depth)
- 'class' shows the final prediction at leaf nodes
- 'value' shows [normal_count, attack_count] at each node

{'='*80}

"""
    
    with open(output_path, 'w') as f:
        f.write(header)
        f.write(tree_rules)
        f.write(f"\n{'='*80}\n")
        f.write("END OF DECISION TREE RULES\n")
        f.write(f"{'='*80}\n")
    
    rules_lines = tree_rules.split('\n')
    preview_lines = min(20, len(rules_lines))
    
    print(f"   ✓ Saved: {output_path}")
    print(f"\n   📄 Preview (first {preview_lines} lines):")
    print("   " + "-" * 70)
    for line in rules_lines[:preview_lines]:
        print(f"   {line}")
    print("   " + "-" * 70)
    print(f"   ... (see full rules in {output_path})")


def main():
    print("\n" + "="*80)
    print("🎨 DECISION TREE VISUALIZATION & INTERPRETATION")
    print("="*80)
    
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    MODEL_PATH = PROJECT_ROOT / "models" / "dt_model.pkl"
    DATA_PATH = PROJECT_ROOT / "data" / "processed_randomforest" / "test.csv"
    REPORTS_DIR = PROJECT_ROOT / "reports"
    
    print(f"\n⚙️  Configuration:")
    print(f"   Project Root: {PROJECT_ROOT}")
    print(f"   Model Path: {MODEL_PATH}")
    print(f"   Data Path: {DATA_PATH}")
    print(f"   Reports Dir: {REPORTS_DIR}")
    
    output_dirs = setup_output_directories(REPORTS_DIR)
    
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"\n{e}")
        sys.exit(1)
    
    try:
        X_test, y_test, feature_names = load_test_data(DATA_PATH)
    except Exception as e:
        print(f"\n{e}")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*80)
    
    tree_structure_path = output_dirs['figures'] / "dt_structure_top4_levels.png"
    visualize_tree_structure(model, feature_names, tree_structure_path, max_depth=4)
    
    feature_importance_path = output_dirs['figures'] / "dt_feature_importance.png"
    visualize_feature_importance(model, feature_names, feature_importance_path, top_n=10)
    
    confusion_matrix_path = output_dirs['figures'] / "dt_confusion_matrix.png"
    visualize_confusion_matrix(model, X_test, y_test, confusion_matrix_path)
    
    print("\n" + "="*80)
    print("📝 EXPORTING INTERPRETABLE RULES")
    print("="*80)
    
    rules_path = output_dirs['text_exports'] / "decision_tree_rules.txt"
    export_tree_rules(model, feature_names, rules_path)
    
    print("\n" + "="*80)
    print("✅ VISUALIZATION COMPLETE!")
    print("="*80)
    
    print(f"\n📁 Generated Files:")
    print(f"\n   Visualizations (PNG, 300 DPI):")
    print(f"   ✓ {tree_structure_path}")
    print(f"   ✓ {feature_importance_path}")
    print(f"   ✓ {confusion_matrix_path}")
    
    print(f"\n   Text Exports:")
    print(f"   ✓ {rules_path}")
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Review the tree structure to understand key decision points")
    print(f"   2. Check feature importance to see which metrics matter most")
    print(f"   3. Analyze the confusion matrix for model strengths/weaknesses")
    print(f"   4. Read the text rules to validate they make domain sense")
    print(f"   5. Include these visualizations in your reports/presentations")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
