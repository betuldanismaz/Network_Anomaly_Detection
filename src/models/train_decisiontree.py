"""
Decision Tree — 3-Class Classification
Network Intrusion Detection on CICIDS2017
Classes: Benign (0) | Volumetric (1) | Semantic (2)

Author: betül
Date: 2026-03-02
"""

import os
import sys
from pathlib import Path
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import ParameterGrid

# WHY: We add the parent directory to the path so Python can find our config module.
# This allows us to run this script from different locations (e.g., project root or src/).
sys.path.append(str(Path(__file__).parent.parent.parent))

# WHY: Import the TOP_FEATURES list from config.py to ensure consistency across models.
# This way, all models use the same feature set, making comparisons fair.
from src.config import TOP_FEATURES

# Class label mapping
CLASS_NAMES = ['Benign', 'Volumetric', 'Semantic']
CLASS_LABELS_FULL = ['Benign (0)', 'Volumetric (1)', 'Semantic (2)']

# Project root (two levels up from src/models/)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def load_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, val, and test CSVs from the 3-class processed data directory."""

    data_path = Path(data_dir)
    train_path = data_path / "train.csv"
    val_path = data_path / "val.csv"
    test_path = data_path / "test.csv"

    # WHY: Check file existence explicitly to provide a helpful error message.
    for fpath, name in [(train_path, "train.csv"), (val_path, "val.csv"), (test_path, "test.csv")]:
        if not fpath.exists():
            raise FileNotFoundError(f"{name} not found at: {fpath}")

    print(f"📂 Loading training data from: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"   ✓ Loaded {len(train_df):,} training samples")

    print(f"📂 Loading validation data from: {val_path}")
    val_df = pd.read_csv(val_path)
    print(f"   ✓ Loaded {len(val_df):,} validation samples")

    print(f"📂 Loading test data from: {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"   ✓ Loaded {len(test_df):,} test samples")

    return train_df, val_df, test_df


def prepare_features_and_labels(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str]
) -> tuple:
    """Extract feature matrices and label vectors from all splits.

    Strategy: Train on train.csv only, evaluate on both val and test.
    Reasoning: Keeping val separate from training prevents data leakage
    and gives an honest validation metric alongside the final test score.
    """

    print("\n🔧 Preparing features and labels...")
    print(f"   Using {len(feature_cols)} selected features from TOP_FEATURES")

    # WHY: Ensure all expected features exist in the DataFrame.
    missing_features = set(feature_cols) - set(train_df.columns)
    if missing_features:
        raise ValueError(f"Missing features in data: {missing_features}")

    X_train = train_df[feature_cols].values
    X_val = val_df[feature_cols].values
    X_test = test_df[feature_cols].values

    y_train = train_df['Label'].values
    y_val = val_df['Label'].values
    y_test = test_df['Label'].values

    print(f"   ✓ X_train shape: {X_train.shape}")
    print(f"   ✓ X_val shape:   {X_val.shape}")
    print(f"   ✓ X_test shape:  {X_test.shape}")

    # Class distribution for all 3 classes
    print(f"\n   📊 Training Set Class Distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique, counts):
        cls_name = CLASS_NAMES[int(label)] if int(label) < len(CLASS_NAMES) else f"Unknown"
        pct = count / len(y_train) * 100
        print(f"      - {cls_name} ({int(label)}): {count:,} samples ({pct:.2f}%)")

    print(f"\n   📊 Validation Set Class Distribution:")
    unique_v, counts_v = np.unique(y_val, return_counts=True)
    for label, count in zip(unique_v, counts_v):
        cls_name = CLASS_NAMES[int(label)] if int(label) < len(CLASS_NAMES) else f"Unknown"
        pct = count / len(y_val) * 100
        print(f"      - {cls_name} ({int(label)}): {count:,} samples ({pct:.2f}%)")

    print(f"\n   📊 Test Set Class Distribution:")
    unique_t, counts_t = np.unique(y_test, return_counts=True)
    for label, count in zip(unique_t, counts_t):
        cls_name = CLASS_NAMES[int(label)] if int(label) < len(CLASS_NAMES) else f"Unknown"
        pct = count / len(y_test) * 100
        print(f"      - {cls_name} ({int(label)}): {count:,} samples ({pct:.2f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


def tune_decision_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = 42
) -> dict:
    """Grid search over key hyperparameters, scored on validation macro F1.

    WHY a grid search instead of fixed params?
    A single decision tree is sensitive to depth and leaf-size settings.
    Deeper trees (depth 20+) capture finer Benign/Volumetric/Semantic boundaries,
    but too deep without leaf guards causes noisy leaves that hurt precision.
    Rather than hand-tuning, we let the validation set decide the best combo.
    """

    # WHY these ranges?
    # - max_depth: 15-30 — shallow (10) was under-fitting for 3-class;
    #   None (unlimited) risks noisy leaves, so we cap at 30 and let
    #   min_samples_leaf handle pruning instead.
    # - min_samples_leaf: 20-100 — a leaf with fewer samples is likely noise;
    #   this is a structural pruning guard that replaces the blunt max_depth cap.
    # - class_weight: 'balanced' vs custom — 'balanced' can over-boost Semantic
    #   (weight ~5.3×), custom {0:1,1:2,2:4} is softer and preserves more precision.
    param_grid = {
        'max_depth':        [15, 20, 30],
        'min_samples_leaf': [20, 50, 100],
        'class_weight':     ['balanced', {0: 1.0, 1: 2.0, 2: 4.0}],
    }

    print(f"\n🔍 Hyperparameter Grid Search (scored on Validation Macro F1)...")
    print(f"   Grid: max_depth={param_grid['max_depth']}, "
          f"min_samples_leaf={param_grid['min_samples_leaf']}, "
          f"class_weight=['balanced', custom]")
    print(f"   Total combinations: {len(list(ParameterGrid(param_grid)))}")

    best_f1   = -1.0
    best_params = {}
    results = []

    for i, params in enumerate(ParameterGrid(param_grid), start=1):
        clf = DecisionTreeClassifier(
            max_depth=params['max_depth'],
            min_samples_leaf=params['min_samples_leaf'],
            min_samples_split=params['min_samples_leaf'] * 2,  # always 2× leaf floor
            criterion='entropy',
            class_weight=params['class_weight'],
            random_state=random_state
        )
        clf.fit(X_train, y_train)
        y_pred_val = clf.predict(X_val)
        val_f1 = f1_score(y_val, y_pred_val, average='macro')
        val_p  = precision_score(y_val, y_pred_val, average='macro')
        val_r  = recall_score(y_val, y_pred_val, average='macro')

        cw_label = 'balanced' if params['class_weight'] == 'balanced' else 'custom'
        results.append((val_f1, params, cw_label))
        print(f"   [{i:2d}/18] depth={str(params['max_depth']):>3s}  "
              f"leaf={params['min_samples_leaf']:>3d}  cw={cw_label:>8s}  "
              f"→ P={val_p:.4f}  R={val_r:.4f}  F1={val_f1:.4f}"
              + (" ✅ best" if val_f1 > best_f1 else ""))

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_params = params

    print(f"\n   🏆 Best Val Macro F1: {best_f1:.4f}")
    print(f"   Best params: max_depth={best_params['max_depth']}, "
          f"min_samples_leaf={best_params['min_samples_leaf']}, "
          f"class_weight={'balanced' if best_params['class_weight'] == 'balanced' else 'custom'}")

    return best_params


def train_decision_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    best_params: dict,
    random_state: int = 42
) -> DecisionTreeClassifier:
    """Train the final 3-class Decision Tree using hyperparameters from grid search."""

    max_depth        = best_params['max_depth']
    min_samples_leaf = best_params['min_samples_leaf']
    class_weight     = best_params['class_weight']
    cw_label = 'balanced' if class_weight == 'balanced' else 'custom {0:1, 1:2, 2:4}'

    print(f"\n🌳 Training Final Decision Tree Classifier (3-Class)...")
    print(f"   Hyperparameters (from grid search):")
    print(f"   - max_depth:         {max_depth}")
    print(f"   - min_samples_leaf:  {min_samples_leaf}  (structural pruning guard)")
    print(f"   - min_samples_split: {min_samples_leaf * 2}  (= 2 × leaf floor)")
    print(f"   - criterion:         'entropy'  (information gain, better for multi-class)")
    print(f"   - class_weight:      {cw_label}")

    # WHY entropy over gini?
    # entropy (information gain) tends to produce more balanced split choices
    # in multi-class settings because it penalises impure nodes more aggressively.
    # WHY min_samples_leaf instead of relying purely on max_depth?
    # A leaf with only a handful of samples is likely noise. min_samples_leaf
    # acts as a pruning floor — the tree grows to max_depth but won't create
    # leaves that represent fewer than N samples, preserving precision.
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_leaf * 2,
        criterion='entropy',
        class_weight=class_weight,
        random_state=random_state
    )

    print("   Training in progress...")
    model.fit(X_train, y_train)
    print("   ✓ Training complete!")

    n_nodes      = model.tree_.node_count
    n_leaves     = model.tree_.n_leaves
    actual_depth = model.tree_.max_depth
    print(f"\n   Tree Statistics:")
    print(f"   - Total nodes:  {n_nodes}")
    print(f"   - Leaf nodes:   {n_leaves}")
    print(f"   - Actual depth: {actual_depth}")
    print(f"   - Classes:      {model.n_classes_}")

    return model


def evaluate_model(
    model: DecisionTreeClassifier,
    X_data: np.ndarray,
    y_data: np.ndarray,
    split_name: str = "Test"
) -> dict:
    """Evaluate the 3-class model using macro-averaged metrics."""

    print(f"\n📊 Evaluating model on {split_name} set...")

    y_pred = model.predict(X_data)

    # WHY macro-averaged metrics?
    # average='macro' computes the metric for each class independently and then
    # takes the unweighted mean. This gives equal importance to all 3 classes,
    # including the minority Semantic class.
    accuracy = accuracy_score(y_data, y_pred)
    macro_precision = precision_score(y_data, y_pred, average='macro')
    macro_recall = recall_score(y_data, y_pred, average='macro')
    macro_f1 = f1_score(y_data, y_pred, average='macro')
    weighted_f1 = f1_score(y_data, y_pred, average='weighted')

    cm = confusion_matrix(y_data, y_pred)

    print(f"\n   ✅ {split_name} Results (Macro-Averaged):")
    print(f"   - Accuracy:        {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   - Macro Precision: {macro_precision:.4f} ({macro_precision*100:.2f}%)")
    print(f"   - Macro Recall:    {macro_recall:.4f} ({macro_recall*100:.2f}%)")
    print(f"   - Macro F1-Score:  {macro_f1:.4f} ({macro_f1*100:.2f}%)")
    print(f"   - Weighted F1:     {weighted_f1:.4f} ({weighted_f1*100:.2f}%)")

    # Per-class metrics
    print(f"\n   � Per-Class Metrics ({split_name}):")
    per_class = {}
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        cls_p = precision_score(y_data, y_pred, labels=[cls_id], average='macro')
        cls_r = recall_score(y_data, y_pred, labels=[cls_id], average='macro')
        cls_f = f1_score(y_data, y_pred, labels=[cls_id], average='macro')
        marker = " ⚠️  (minority ~6%)" if cls_name == "Semantic" else ""
        print(f"   - {cls_name:12s}  P={cls_p:.4f}  R={cls_r:.4f}  F1={cls_f:.4f}{marker}")
        per_class[cls_name] = {'precision': cls_p, 'recall': cls_r, 'f1': cls_f}

    # 3×3 Confusion Matrix
    print(f"\n   📋 3×3 Confusion Matrix ({split_name}):")
    print("   " + "-" * 65)
    header = f"{'':>18s}  {'Pred Benign':>12s}  {'Pred Volum.':>12s}  {'Pred Seman.':>12s}"
    print(f"   {header}")
    print("   " + "-" * 65)
    for i, cls_name in enumerate(CLASS_NAMES):
        row = "  ".join(f"{cm[i, j]:>12,}" for j in range(3))
        print(f"   {cls_name:>18s}  {row}")
    print("   " + "-" * 65)

    # Per-class accuracy
    print(f"\n   Per-Class Accuracy:")
    for i, cls_name in enumerate(CLASS_NAMES):
        total = cm[i, :].sum()
        correct = cm[i, i]
        pct = correct / total * 100 if total > 0 else 0
        print(f"   - {cls_name}: {correct:,} / {total:,} ({pct:.2f}%)")

    # Detailed classification report
    print(f"\n   📊 Detailed Classification Report ({split_name}):")
    print("   " + "-" * 65)
    print(classification_report(y_data, y_pred,
                                target_names=CLASS_LABELS_FULL,
                                digits=4))

    return {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'per_class': per_class
    }


def plot_confusion_matrix(cm: np.ndarray, reports_dir: str) -> str:
    """Save a 3×3 confusion matrix heatmap to reports/figures/."""

    print("\n🎨 Generating Confusion Matrix Visualization...")

    os.makedirs(reports_dir, exist_ok=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                annot_kws={'size': 14})
    plt.title('3-Class Confusion Matrix — Decision Tree (Test Set)',
              fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=13)
    plt.xlabel('Predicted Label', fontsize=13)
    plt.tight_layout()

    cm_path = os.path.join(reports_dir, "dt_3class_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    print(f"   ✓ Confusion Matrix saved to: {cm_path}")
    plt.close()

    return cm_path


def plot_feature_importance(model: DecisionTreeClassifier,
                            feature_names: list[str],
                            reports_dir: str) -> str:
    """Plot horizontal bar chart of feature importances and save to file."""

    print("\n🎨 Generating Feature Importance Visualization...")

    os.makedirs(reports_dir, exist_ok=True)

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
    plt.title('Feature Importance — 3-Class Decision Tree', fontsize=16, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()

    importance_path = os.path.join(reports_dir, "dt_3class_feature_importance.png")
    plt.savefig(importance_path, dpi=300)
    print(f"   ✓ Feature Importance saved to: {importance_path}")
    plt.close()

    # Print top features to terminal
    print(f"\n   📊 Top 10 Features by Importance:")
    for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
        print(f"      {i+1:2d}. {row['feature']:30s} {row['importance']:.4f}")

    return importance_path


def export_tree_rules(
    model: DecisionTreeClassifier,
    feature_names: list[str]
) -> str:
    """Export the decision tree rules as readable text."""

    print("\n📜 Exporting Decision Tree Rules...")
    print("   (This shows the exact if-else logic the model uses)")

    # WHY: export_text creates a readable text representation of the tree
    tree_rules = export_text(
        model,
        feature_names=feature_names,
        spacing=3
    )

    print("\n" + "=" * 80)
    print("DECISION TREE RULES (If-Then-Else Logic) — 3-Class")
    print("=" * 80)
    print(tree_rules)
    print("=" * 80)

    print("\n   💡 How to read this:")
    print("   - Each line shows a decision rule (threshold)")
    print("   - Indentation shows tree depth/hierarchy")
    print("   - 'class' shows the final prediction at each leaf:")
    print("     class 0 = Benign, class 1 = Volumetric, class 2 = Semantic")
    print("   - 'value' shows [benign_count, volumetric_count, semantic_count] at that node")

    return tree_rules


def save_model(model: DecisionTreeClassifier, output_path: str) -> None:
    """Save the trained model using pickle."""

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Saving trained model...")
    print(f"   Output path: {output_file}")

    with open(output_file, 'wb') as f:
        pickle.dump(model, f)

    file_size_kb = output_file.stat().st_size / 1024
    print(f"   ✓ Model saved successfully! (Size: {file_size_kb:.2f} KB)")

    print("\n   📌 To load this model later, use:")
    print(f"      with open('{output_path}', 'rb') as f:")
    print(f"          model = pickle.load(f)")


def main():
    """Main training pipeline for 3-class Decision Tree."""

    print("\n" + "=" * 80)
    print("🚀 DECISION TREE TRAINING — 3-CLASS NETWORK INTRUSION DETECTION")
    print("   Classes: Benign (0) | Volumetric (1) | Semantic (2)")
    print("=" * 80)

    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    DATA_DIR = os.path.join(ROOT, 'data', 'processed_ml')
    MODEL_OUTPUT_PATH = os.path.join(ROOT, 'models', 'dt_3class_model.pkl')
    RULES_OUTPUT_PATH = os.path.join(ROOT, 'models', 'dt_3class_rules.txt')
    REPORTS_DIR = os.path.join(ROOT, 'reports', 'figures')

    RANDOM_STATE = 42  # WHY 42? It's the "Answer to Life, Universe, and Everything" 😊

    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================
    try:
        train_df, val_df, test_df = load_data(DATA_DIR)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Tip: Make sure the 3-class preprocessed data exists in data/processed_ml/.")
        print("   Run the preprocessing script first if needed.")
        sys.exit(1)

    # ========================================================================
    # STEP 2: PREPARE FEATURES AND LABELS
    # ========================================================================
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_features_and_labels(
            train_df, val_df, test_df, TOP_FEATURES
        )
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

    # ========================================================================
    # STEP 3: HYPERPARAMETER SEARCH → TRAIN BEST MODEL
    # ========================================================================
    best_params = tune_decision_tree(
        X_train, y_train,
        X_val,   y_val,
        random_state=RANDOM_STATE
    )

    model = train_decision_tree(
        X_train,
        y_train,
        best_params=best_params,
        random_state=RANDOM_STATE
    )

    # ========================================================================
    # STEP 4: EVALUATE MODEL (Validation + Test)
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 EVALUATION")
    print("=" * 80)

    val_metrics = evaluate_model(model, X_val, y_val, split_name="Validation")
    test_metrics = evaluate_model(model, X_test, y_test, split_name="Test")

    # ========================================================================
    # STEP 5: VISUALIZATIONS
    # ========================================================================
    cm_path = plot_confusion_matrix(test_metrics['confusion_matrix'], REPORTS_DIR)
    fi_path = plot_feature_importance(model, TOP_FEATURES, REPORTS_DIR)

    # ========================================================================
    # STEP 6: EXPORT TREE RULES (INTERPRETABILITY)
    # ========================================================================
    tree_rules = export_tree_rules(model, TOP_FEATURES)

    rules_file = Path(RULES_OUTPUT_PATH)
    rules_file.parent.mkdir(parents=True, exist_ok=True)
    with open(rules_file, 'w') as f:
        f.write(tree_rules)
    print(f"\n   ✓ Rules saved to: {rules_file}")

    # ========================================================================
    # STEP 7: SAVE MODEL
    # ========================================================================
    save_model(model, MODEL_OUTPUT_PATH)

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("✅ 3-CLASS TRAINING COMPLETE!")
    print("=" * 80)

    print(f"\n📊 Final Model Performance (Test Set):")
    print(f"   - Accuracy:        {test_metrics['accuracy']:.4f}")
    print(f"   - Macro Precision: {test_metrics['macro_precision']:.4f}")
    print(f"   - Macro Recall:    {test_metrics['macro_recall']:.4f}")
    print(f"   - Macro F1-Score:  {test_metrics['macro_f1']:.4f}")
    print(f"   - Weighted F1:     {test_metrics['weighted_f1']:.4f}")

    print(f"\n� Per-Class Performance (Test Set):")
    for cls_name in CLASS_NAMES:
        m = test_metrics['per_class'][cls_name]
        marker = " ⚠️  (minority class, ~6%)" if cls_name == "Semantic" else ""
        print(f"   - {cls_name:12s}  P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}{marker}")

    print(f"\n💾 Saved Artifacts:")
    print(f"   - Model:             {MODEL_OUTPUT_PATH}")
    print(f"   - Tree Rules:        {RULES_OUTPUT_PATH}")
    print(f"   - Confusion Matrix:  {cm_path}")
    print(f"   - Feature Importance: {fi_path}")

    print("\n🎯 Next Steps:")
    print("   1. Review the decision tree rules above to validate they make sense for 3-class")
    print("   2. Check Semantic class performance — it's the minority (~6%)")
    print("   3. If macro F1 is low, consider increasing max_depth or using ensemble methods")
    print("   4. Compare with Random Forest, XGBoost, and LSTM models")
    print("   5. Deploy the model using the saved .pkl file")

    print("\n" + "=" * 80)


if __name__ == "__main__":

    main()
