#!/usr/bin/env python3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# TensorFlow imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
import tensorflow as tf
from tensorflow import keras

# Scikit-learn imports
from sklearn.metrics import classification_report, confusion_matrix

# =============================================================================
# Configuration
# =============================================================================
DATA_DIR = PROJECT_ROOT / "data" / "processed_lstm"
MODEL_PATH = PROJECT_ROOT / "models" / "bilstm_best.keras"
REPORTS_DIR = PROJECT_ROOT / "reports" / "bilstm"

# Class labels
CLASS_NAMES = {
    0: "Benign",
    1: "Volumetric", 
    2: "Semantic"
}

# Prediction batch size (memory safety)
BATCH_SIZE = 256


def load_test_data():
    """Load test data from .npy files."""
    print("=" * 60)
    print("📂 Loading Test Data")
    print("=" * 60)
    
    X_test_path = DATA_DIR / "X_test.npy"
    y_test_path = DATA_DIR / "y_test.npy"
    
    if not X_test_path.exists():
        raise FileNotFoundError(f"X_test.npy not found at {X_test_path}")
    if not y_test_path.exists():
        raise FileNotFoundError(f"y_test.npy not found at {y_test_path}")
    
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    
    print(f"✅ X_test shape: {X_test.shape}")
    print(f"✅ y_test shape: {y_test.shape}")
    print(f"✅ Unique classes in y_test: {np.unique(y_test)}")
    
    # Class distribution
    print("\n📊 Class Distribution in Test Set:")
    for cls_id, cls_name in CLASS_NAMES.items():
        count = np.sum(y_test == cls_id)
        pct = (count / len(y_test)) * 100
        print(f"   Class {cls_id} ({cls_name}): {count:,} samples ({pct:.2f}%)")
    
    return X_test, y_test


def load_model():
    """Load the trained BiLSTM model."""
    print("\n" + "=" * 60)
    print("🧠 Loading BiLSTM Model")
    print("=" * 60)
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    model = keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded from: {MODEL_PATH}")
    print(f"✅ Model input shape: {model.input_shape}")
    print(f"✅ Model output shape: {model.output_shape}")
    
    return model


def predict_with_batches(model, X_test, batch_size=BATCH_SIZE):
    """Generate predictions using batches for memory safety."""
    print("\n" + "=" * 60)
    print("🔮 Generating Predictions")
    print("=" * 60)
    
    n_samples = len(X_test)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    print(f"📦 Total samples: {n_samples:,}")
    print(f"📦 Batch size: {batch_size}")
    print(f"📦 Number of batches: {n_batches}")
    
    all_predictions = []
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        
        batch_X = X_test[start_idx:end_idx]
        batch_pred = model.predict(batch_X, verbose=0)
        all_predictions.append(batch_pred)
        
        # Progress indicator
        if (i + 1) % 50 == 0 or (i + 1) == n_batches:
            progress = (i + 1) / n_batches * 100
            print(f"   Progress: {i + 1}/{n_batches} batches ({progress:.1f}%)")
    
    # Concatenate all predictions
    y_pred_proba = np.concatenate(all_predictions, axis=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    print(f"\n✅ Predictions generated: {len(y_pred):,} samples")
    
    return y_pred, y_pred_proba


def generate_classification_report(y_test, y_pred):
    """Generate and print classification report."""
    print("\n" + "=" * 60)
    print("📋 Classification Report")
    print("=" * 60)
    
    # Get target names in order
    target_names = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())]
    
    report = classification_report(
        y_test, 
        y_pred, 
        target_names=target_names,
        digits=4
    )
    
    print(report)
    
    # Save report to file
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "final_classification_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("BiLSTM Model - Final Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test samples: {len(y_test):,}\n")
        f.write(f"Model: bilstm_best.keras\n\n")
        f.write(report)
    
    print(f"\n✅ Report saved to: {report_path}")
    
    return report


def generate_confusion_matrix(y_test, y_pred):
    """Generate and save confusion matrix heatmap."""
    print("\n" + "=" * 60)
    print("🎯 Confusion Matrix")
    print("=" * 60)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Print raw matrix
    print("\nRaw Confusion Matrix:")
    print(cm)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Get class names
    class_labels = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())]
    
    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar_kws={'label': 'Count'}
    )
    
    plt.title('BiLSTM Model - Confusion Matrix\n(Final Evaluation on Test Set)', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    # Save figure
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = REPORTS_DIR / "final_confusion_matrix.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Confusion matrix saved to: {save_path}")
    
    # Calculate and print accuracy per class
    print("\n📊 Per-Class Accuracy:")
    for i, cls_name in CLASS_NAMES.items():
        if cm[i].sum() > 0:
            class_acc = cm[i, i] / cm[i].sum() * 100
            print(f"   {cls_name}: {class_acc:.2f}%")
    
    # Overall accuracy
    overall_acc = np.trace(cm) / cm.sum() * 100
    print(f"\n🎯 Overall Accuracy: {overall_acc:.2f}%")
    
    return cm


def main():
    """Main evaluation function."""
    print("\n" + "=" * 60)
    print("🚀 BiLSTM Model Evaluation")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Data: {DATA_DIR}")
    print(f"Reports: {REPORTS_DIR}")
    
    try:
        # 1. Load test data
        X_test, y_test = load_test_data()
        
        # 2. Load model
        model = load_model()
        
        # 3. Generate predictions
        y_pred, y_pred_proba = predict_with_batches(model, X_test)
        
        # 4. Generate classification report
        report = generate_classification_report(y_test, y_pred)
        
        # 5. Generate confusion matrix
        cm = generate_confusion_matrix(y_test, y_pred)
        
        # Summary
        print("\n" + "=" * 60)
        print("✅ EVALUATION COMPLETE")
        print("=" * 60)
        print(f"📁 Classification Report: {REPORTS_DIR / 'final_classification_report.txt'}")
        print(f"📁 Confusion Matrix: {REPORTS_DIR / 'final_confusion_matrix.png'}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
