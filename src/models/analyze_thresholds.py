#!/usr/bin/env python3
"""
BiLSTM Threshold Analysis Script
=================================
Analyzes prediction confidence scores and calculates optimal thresholds
for a 5-Level Risk Scoring System.

Date: 2026-01-01
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# TensorFlow imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras

# Scikit-learn imports
from sklearn.metrics import precision_score, recall_score

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

BATCH_SIZE = 256


def load_data_and_model():
    """Load test data and trained model."""
    print("=" * 60)
    print("📂 Loading Data and Model")
    print("=" * 60)
    
    # Load data
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")
    
    print(f"✅ X_test shape: {X_test.shape}")
    print(f"✅ y_test shape: {y_test.shape}")
    
    # Load model
    model = keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded from: {MODEL_PATH}")
    
    return X_test, y_test, model


def predict_with_batches(model, X_test, batch_size=BATCH_SIZE):
    """Generate predictions using batches for memory safety."""
    print("\n" + "=" * 60)
    print("🔮 Generating Predictions")
    print("=" * 60)
    
    n_samples = len(X_test)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    all_predictions = []
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        
        batch_X = X_test[start_idx:end_idx]
        batch_pred = model.predict(batch_X, verbose=0)
        all_predictions.append(batch_pred)
        
        if (i + 1) % 100 == 0 or (i + 1) == n_batches:
            progress = (i + 1) / n_batches * 100
            print(f"   Progress: {i + 1}/{n_batches} ({progress:.1f}%)")
    
    y_pred_proba = np.concatenate(all_predictions, axis=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    print(f"\n✅ Predictions generated: {len(y_pred):,} samples")
    
    return y_pred, y_pred_proba


def analyze_confidence_distribution(y_test, y_pred, y_pred_proba):
    """Analyze confidence score distribution for each class."""
    print("\n" + "=" * 60)
    print("📊 Confidence Score Distribution Analysis")
    print("=" * 60)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for class_idx, class_name in CLASS_NAMES.items():
        ax = axes[class_idx]
        
        # Get samples where this class was predicted
        predicted_as_class = (y_pred == class_idx)
        
        # Get confidence scores for this class
        confidence_scores = y_pred_proba[:, class_idx]
        
        # Separate correct and incorrect predictions
        correct_mask = (y_test == class_idx) & predicted_as_class
        incorrect_mask = (y_test != class_idx) & predicted_as_class
        
        correct_confidences = confidence_scores[correct_mask]
        incorrect_confidences = confidence_scores[incorrect_mask]
        
        # Also get confidence scores for samples that ARE this class
        true_class_mask = (y_test == class_idx)
        true_class_confidences = confidence_scores[true_class_mask]
        
        # Plot histograms
        bins = np.linspace(0, 1, 51)
        
        # Correct predictions (True Positives)
        ax.hist(correct_confidences, bins=bins, alpha=0.7, color='green', 
                label=f'Correct (TP): {len(correct_confidences):,}', edgecolor='darkgreen')
        
        # Incorrect predictions (False Positives)
        ax.hist(incorrect_confidences, bins=bins, alpha=0.7, color='red', 
                label=f'Wrong (FP): {len(incorrect_confidences):,}', edgecolor='darkred')
        
        ax.set_xlabel('Confidence Score', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Class {class_idx}: {class_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Add vertical lines for reference thresholds
        ax.axvline(x=0.5, color='orange', linestyle='--', alpha=0.7, label='0.5 threshold')
        ax.axvline(x=0.9, color='purple', linestyle='--', alpha=0.7, label='0.9 threshold')
        
        # Print statistics
        print(f"\n📌 Class {class_idx} ({class_name}):")
        print(f"   Total Predicted as {class_name}: {predicted_as_class.sum():,}")
        print(f"   True Positives: {len(correct_confidences):,}")
        print(f"   False Positives: {len(incorrect_confidences):,}")
        if len(correct_confidences) > 0:
            print(f"   Correct Confidence - Mean: {correct_confidences.mean():.4f}, "
                  f"Min: {correct_confidences.min():.4f}, Max: {correct_confidences.max():.4f}")
        if len(incorrect_confidences) > 0:
            print(f"   Incorrect Confidence - Mean: {incorrect_confidences.mean():.4f}, "
                  f"Min: {incorrect_confidences.min():.4f}, Max: {incorrect_confidences.max():.4f}")
    
    plt.suptitle('BiLSTM Confidence Score Distribution by Class\n(Green=Correct, Red=Wrong)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = REPORTS_DIR / "threshold_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Visualization saved to: {save_path}")


def calculate_safe_thresholds(y_test, y_pred_proba, target_precision=0.99):
    """Calculate thresholds that achieve target precision for each class."""
    print("\n" + "=" * 60)
    print(f"🎯 Safe Threshold Analysis (Target Precision: {target_precision*100:.0f}%)")
    print("=" * 60)
    
    results = {}
    
    for class_idx, class_name in CLASS_NAMES.items():
        print(f"\n📌 Class {class_idx} ({class_name}):")
        
        # Get confidence scores for this class
        confidence_scores = y_pred_proba[:, class_idx]
        
        # True labels for this class (binary)
        y_true_binary = (y_test == class_idx).astype(int)
        
        # Try different thresholds
        thresholds = np.arange(0.30, 1.00, 0.01)
        
        best_threshold = None
        best_recall = 0
        
        for threshold in thresholds:
            # Predict as this class if confidence >= threshold
            y_pred_binary = (confidence_scores >= threshold).astype(int)
            
            # Skip if no positive predictions
            if y_pred_binary.sum() == 0:
                continue
            
            # Calculate precision and recall
            precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            
            # Check if precision meets target
            if precision >= target_precision:
                if recall > best_recall:
                    best_recall = recall
                    best_threshold = threshold
        
        if best_threshold is not None:
            # Recalculate metrics at best threshold
            y_pred_binary = (confidence_scores >= best_threshold).astype(int)
            precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            
            print(f"   ✅ Safe Threshold: {best_threshold:.2f}")
            print(f"   ✅ Precision at threshold: {precision*100:.2f}%")
            print(f"   ✅ Recall at threshold: {recall*100:.2f}%")
            print(f"   ✅ Samples above threshold: {y_pred_binary.sum():,}")
            
            results[class_idx] = {
                'threshold': best_threshold,
                'precision': precision,
                'recall': recall,
                'count': y_pred_binary.sum()
            }
        else:
            print(f"   ⚠️ No threshold found that achieves {target_precision*100:.0f}% precision")
            
            # Find the maximum precision achievable
            max_precision = 0
            max_precision_threshold = 0
            for threshold in thresholds:
                y_pred_binary = (confidence_scores >= threshold).astype(int)
                if y_pred_binary.sum() == 0:
                    continue
                precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
                if precision > max_precision:
                    max_precision = precision
                    max_precision_threshold = threshold
            
            print(f"   📊 Max achievable precision: {max_precision*100:.2f}% at threshold {max_precision_threshold:.2f}")
            
            results[class_idx] = {
                'threshold': max_precision_threshold,
                'precision': max_precision,
                'recall': None,
                'count': None
            }
    
    return results


def analyze_risk_levels(y_pred_proba):
    """Propose 5-level risk scoring based on confidence distribution."""
    print("\n" + "=" * 60)
    print("🔢 5-Level Risk Scoring Proposal")
    print("=" * 60)
    
    # Get max confidence for each prediction
    max_confidences = np.max(y_pred_proba, axis=1)
    
    # Analyze distribution
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print("\n📊 Confidence Distribution Percentiles:")
    for p in percentiles:
        val = np.percentile(max_confidences, p)
        print(f"   {p}th percentile: {val:.4f}")
    
    # Propose risk levels
    print("\n🎯 Proposed 5-Level Risk Scoring:")
    print("-" * 50)
    
    risk_levels = [
        ("CRITICAL", 0.00, 0.50, "🔴"),
        ("HIGH",     0.50, 0.75, "🟠"),
        ("MEDIUM",   0.75, 0.90, "🟡"),
        ("LOW",      0.90, 0.99, "🟢"),
        ("MINIMAL",  0.99, 1.00, "⚪"),
    ]
    
    for level_name, low, high, emoji in risk_levels:
        mask = (max_confidences >= low) & (max_confidences < high)
        count = mask.sum()
        pct = count / len(max_confidences) * 100
        print(f"   {emoji} {level_name:10s} [{low:.2f} - {high:.2f}): {count:,} samples ({pct:.2f}%)")
    
    print("-" * 50)
    print("\n💡 Interpretation:")
    print("   - CRITICAL/HIGH: Uncertain predictions, needs review")
    print("   - MEDIUM: Moderate confidence, monitor closely")
    print("   - LOW: Good confidence, standard monitoring")
    print("   - MINIMAL: Very high confidence, likely accurate")


def main():
    """Main analysis function."""
    print("\n" + "=" * 60)
    print("🚀 BiLSTM Threshold Analysis")
    print("=" * 60)
    
    try:
        # 1. Load data and model
        X_test, y_test, model = load_data_and_model()
        
        # 2. Generate predictions
        y_pred, y_pred_proba = predict_with_batches(model, X_test)
        
        # 3. Analyze confidence distribution
        analyze_confidence_distribution(y_test, y_pred, y_pred_proba)
        
        # 4. Calculate safe thresholds
        thresholds = calculate_safe_thresholds(y_test, y_pred_proba, target_precision=0.99)
        
        # 5. Propose risk levels
        analyze_risk_levels(y_pred_proba)
        
        # Summary
        print("\n" + "=" * 60)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"📁 Visualization: {REPORTS_DIR / 'threshold_analysis.png'}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
