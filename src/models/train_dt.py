"""
Decision Tree binary classification 

Author: betül
Date: 05.02.2026
""" 

import os
import sys
from pathlib import Path
import pickle

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# WHY: We add the parent directory to the path so Python can find our config module.
# This allows us to run this script from different locations (e.g., project root or src/).
sys.path.append(str(Path(__file__).parent.parent.parent))

# WHY: Import the TOP_FEATURES list from config.py to ensure consistency across models.
# This way, all models use the same feature set, making comparisons fair.
from src.config import TOP_FEATURES


def load_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:

    data_path = Path(data_dir)
    train_path = data_path / "train.csv"
    test_path = data_path / "test.csv"
    
    # WHY: Check file existence explicitly to provide a helpful error message.
    # Better than letting pandas fail with a cryptic error.
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found at: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at: {test_path}")
    
    print(f"📂 Loading training data from: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"   ✓ Loaded {len(train_df):,} training samples")
    
    print(f"📂 Loading test data from: {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"   ✓ Loaded {len(test_df):,} test samples")
    
    return train_df, test_df


def prepare_features_and_labels(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    feature_cols: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # WHY: We use TOP_FEATURES to select only the most important features.
    # This reduces noise and potential overfitting. Feature selection was done
    # during preprocessing (likely using techniques like feature importance or correlation).
    print("\n🔧 Preparing features and labels...")
    print(f"   Using {len(feature_cols)} selected features from TOP_FEATURES")
    
    # WHY: Ensure all expected features exist in the DataFrame.
    # This catches configuration errors early.
    missing_features = set(feature_cols) - set(train_df.columns)
    if missing_features:
        raise ValueError(f"Missing features in data: {missing_features}")
    
    # WHY: Extract feature columns into X (predictor variables)
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    
    # WHY: Extract the 'Label' column into y (target variable)
    # We assume 'Label' contains binary values: 0 (Normal) or 1 (Attack)
    y_train = train_df['Label'].values
    y_test = test_df['Label'].values
    
    print(f"   ✓ X_train shape: {X_train.shape}")
    print(f"   ✓ X_test shape: {X_test.shape}")
    print(f"   ✓ Class distribution in training set:")
    unique, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique, counts):
        label_name = "Normal" if label == 0 else "Attack"
        print(f"      - {label_name} ({label}): {count:,} samples ({count/len(y_train)*100:.2f}%)")
    
    return X_train, X_test, y_train, y_test


def train_decision_tree(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    max_depth: int = 10,
    random_state: int = 42
) -> DecisionTreeClassifier:

    print(f"\n🌳 Training Decision Tree Classifier...")
    print(f"   Hyperparameters:")
    print(f"   - max_depth: {max_depth} (prevents overfitting, maintains interpretability)")
    print(f"   - random_state: {random_state} (ensures reproducibility)")
    print(f"   - criterion: 'gini' (default, measures impurity of splits)")
    
   
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state,
        criterion='gini'  # WHY gini? It's computationally efficient and works well in practice
    )
    
   
    print("   Training in progress...")
    model.fit(X_train, y_train)
    print("   ✓ Training complete!")
    
    # WHY: Print tree statistics to understand model complexity
    n_nodes = model.tree_.node_count
    n_leaves = model.tree_.n_leaves
    actual_depth = model.tree_.max_depth
    print(f"\n   Tree Statistics:")
    print(f"   - Total nodes: {n_nodes}")
    print(f"   - Leaf nodes (decision outcomes): {n_leaves}")
    print(f"   - Actual depth: {actual_depth}")
    
    return model


def evaluate_model(
    model: DecisionTreeClassifier, 
    X_test: np.ndarray, 
    y_test: np.ndarray
) -> dict:

    print("\n📊 Evaluating model on test set...")
    
    # WHY: Generate predictions on test data
    # The model traverses the tree for each sample, following decision rules
    y_pred = model.predict(X_test)
    
    # WHY: Calculate multiple metrics to get a complete picture of performance
    # Each metric tells us something different about the model's behavior
    
    # WHY Accuracy: Simple overall correctness percentage
    # LIMITATION: Can be misleading if one class dominates (e.g., 95% normal traffic)
    accuracy = accuracy_score(y_test, y_pred)
    

    precision = precision_score(y_test, y_pred, average='binary', pos_label=1)
    

    recall = recall_score(y_test, y_pred, average='binary', pos_label=1)
    

    f1 = f1_score(y_test, y_pred, average='binary', pos_label=1)
    

    cm = confusion_matrix(y_test, y_pred)
    
    print("\n   ✅ Evaluation Results:")
    print(f"   - Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   - Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"   - Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"   - F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    print("\n   📋 Confusion Matrix:")
    print("   " + "-" * 50)
    print(f"                    Predicted Normal  Predicted Attack")
    print(f"   Actual Normal    {cm[0][0]:>15,}  {cm[0][1]:>16,}  (FP)")
    print(f"   Actual Attack    {cm[1][0]:>15,}  {cm[1][1]:>16,}  (TP)")
    print("                    (FN)")
    print("   " + "-" * 50)
    print(f"\n   Interpretation:")
    print(f"   - True Negatives (TN):  {cm[0][0]:,} - Correctly identified normal traffic")
    print(f"   - False Positives (FP): {cm[0][1]:,} - Normal traffic incorrectly flagged as attacks")
    print(f"   - False Negatives (FN): {cm[1][0]:,} - Attacks that slipped through undetected ⚠️")
    print(f"   - True Positives (TP):  {cm[1][1]:,} - Correctly detected attacks ✓")
    
    # WHY: Print detailed classification report
    # This shows precision, recall, and F1 for BOTH classes (Normal and Attack)
    print("\n   📊 Detailed Classification Report:")
    print("   " + "-" * 50)
    target_names = ['Normal (0)', 'Attack (1)']
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'predictions': y_pred
    }


def export_tree_rules(
    model: DecisionTreeClassifier, 
    feature_names: list[str]
) -> str:

    print("\n📜 Exporting Decision Tree Rules...")
    print("   (This shows the exact if-else logic the model uses)")
    
    # WHY: export_text creates a readable text representation of the tree
    # It shows the decision path from root to each leaf
    tree_rules = export_text(
        model, 
        feature_names=feature_names,
        spacing=3        # Indentation for readability
    )
    
    print("\n" + "=" * 80)
    print("DECISION TREE RULES (If-Then-Else Logic)")
    print("=" * 80)
    print(tree_rules)
    print("=" * 80)
    
    print("\n   💡 How to read this:")
    print("   - Each line shows a decision rule (threshold)")
    print("   - Indentation shows tree depth/hierarchy")
    print("   - 'class' shows the final prediction at each leaf")
    print("   - 'value' shows [normal_count, attack_count] at that node")
    
    return tree_rules


def save_model(model: DecisionTreeClassifier, output_path: str) -> None:

    # WHY: Create parent directories if they don't exist
    # This prevents errors if the 'models/' directory doesn't exist yet
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving trained model...")
    print(f"   Output path: {output_file}")
    
    # WHY: Use 'wb' (write binary) mode for pickle files
    # Pickle creates binary data, not text
    with open(output_file, 'wb') as f:
        pickle.dump(model, f)
    
    # WHY: Verify the file was created and check its size
    file_size_kb = output_file.stat().st_size / 1024
    print(f"   ✓ Model saved successfully! (Size: {file_size_kb:.2f} KB)")
    
    print("\n   📌 To load this model later, use:")
    print(f"      with open('{output_path}', 'rb') as f:")
    print(f"          model = pickle.load(f)")


def main():

    print("\n" + "=" * 80)
    print("🚀 DECISION TREE TRAINING FOR NETWORK INTRUSION DETECTION")
    print("=" * 80)
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    # WHY: Define all paths and parameters at the top for easy modification
    
    DATA_DIR = r"d:\Projects\networkdetection\networkdetection\data\processed_randomforest"
    MODEL_OUTPUT_PATH = "models/dt_model.pkl"
    
    # WHY max_depth=10? 
    # - Prevents overfitting (too deep = memorizes training data)
    # - Maintains interpretability (too deep = too many rules to understand)
    # - Based on empirical best practices for binary classification
    MAX_DEPTH = 10
    
    RANDOM_STATE = 42  # WHY 42? It's the "Answer to Life, Universe, and Everything" 😊
    
    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================
    try:
        train_df, test_df = load_data(DATA_DIR)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Tip: Make sure the preprocessed data exists.")
        print("   Run the preprocessing script first if needed.")
        sys.exit(1)
    
    # ========================================================================
    # STEP 2: PREPARE FEATURES AND LABELS
    # ========================================================================
    try:
        X_train, X_test, y_train, y_test = prepare_features_and_labels(
            train_df, test_df, TOP_FEATURES
        )
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    
    # ========================================================================
    # STEP 3: TRAIN MODEL
    # ========================================================================
    model = train_decision_tree(
        X_train, 
        y_train, 
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE
    )
    
    # ========================================================================
    # STEP 4: EVALUATE MODEL
    # ========================================================================
    metrics = evaluate_model(model, X_test, y_test)
    
    # ========================================================================
    # STEP 5: EXPORT TREE RULES (INTERPRETABILITY)
    # ========================================================================
    tree_rules = export_tree_rules(model, TOP_FEATURES)
    
    # WHY: Optionally save rules to a text file for documentation
    rules_path = Path("models/dt_rules.txt")
    rules_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rules_path, 'w') as f:
        f.write(tree_rules)
    print(f"\n   ✓ Rules also saved to: {rules_path}")
    
    # ========================================================================
    # STEP 6: SAVE MODEL
    # ========================================================================
    save_model(model, MODEL_OUTPUT_PATH)
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\n📊 Final Model Performance:")
    print(f"   - Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   - Precision: {metrics['precision']:.4f}")
    print(f"   - Recall:    {metrics['recall']:.4f}")
    print(f"   - F1-Score:  {metrics['f1_score']:.4f}")
    
    print(f"\n💾 Model saved to: {MODEL_OUTPUT_PATH}")
    print(f"📜 Tree rules saved to: {rules_path}")
    
    print("\n🎯 Next Steps:")
    print("   1. Review the decision tree rules above to validate they make sense")
    print("   2. If recall is low, consider increasing max_depth or using ensemble methods")
    print("   3. If precision is low, consider adding more discriminative features")
    print("   4. Deploy the model using the saved .pkl file")
    print("   5. Monitor performance on real-world traffic and retrain as needed")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":

    main()
