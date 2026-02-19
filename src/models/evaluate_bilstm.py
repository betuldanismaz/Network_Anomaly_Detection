#!/usr/bin/env python3
"""
BiLSTM Model Evaluation for Network Intrusion Detection System
Comprehensive Performance Analysis and Visualization

This script evaluates the trained BiLSTM model and generates:
    - Classification report (Precision/Recall/F1 per class)
    - Confusion matrix heatmap
    - Training history plots
    - Inference speed benchmarks

Usage:
    python src/models/evaluate_bilstm.py

Author: betül
Date: 05.02.2026
"""

import os
import sys
import json
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Tuple
import pickle

# --- PATH SETUP ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

DATA_DIR = os.path.join(ROOT, 'data', 'processed_lstm')
MODELS_DIR = os.path.join(ROOT, 'models')
REPORTS_DIR = os.path.join(ROOT, 'reports', 'bilstm')

# Class names for reporting
CLASS_NAMES = ['Benign', 'Volumetric', 'Semantic']
NUM_CLASSES = 3

# Create output directory
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_test_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load preprocessed test data.
    
    Returns:
        X_test: Test sequences (N, 10, 20)
        y_test: Test labels (N,)
    
    Raises:
        FileNotFoundError: If test data files are missing
    """
    print("\n" + "=" * 70)
    print("📂 LOADING TEST DATA")
    print("=" * 70)
    
    X_test_path = os.path.join(DATA_DIR, 'X_test.npy')
    y_test_path = os.path.join(DATA_DIR, 'y_test.npy')
    
    # Verify files exist
    for path in [X_test_path, y_test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"❌ Test data file not found: {path}\n"
                f"   Please run preprocessing first: python src/features/preprocess_lstm.py"
            )
    
    print(f"   Loading from: {DATA_DIR}")
    
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    
    print(f"   ✅ X_test shape: {X_test.shape}")
    print(f"   ✅ y_test shape: {y_test.shape}")
    print(f"   ✅ Test samples: {len(y_test):,}")
    
    return X_test, y_test


def load_trained_model() -> tf.keras.Model:
    """
    Load the best BiLSTM model checkpoint.
    
    Returns:
        Trained Keras model
    
    Raises:
        FileNotFoundError: If model file is missing
    """
    print("\n" + "=" * 70)
    print("🔧 LOADING TRAINED MODEL")
    print("=" * 70)
    
    model_path = os.path.join(MODELS_DIR, 'bilstm_best.keras')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"❌ Model file not found: {model_path}\n"
            f"   Please train the model first: python src/models/train_bilstm.py"
        )
    
    print(f"   Loading from: {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    
    print(f"   ✅ Model loaded successfully")
    print(f"   ✅ Model type: {model.name}")
    
    # Display model summary
    print("\n   📋 Model Architecture:")
    model.summary()
    
    return model


def evaluate_model(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Evaluate model performance and generate predictions.
    
    Args:
        model: Trained Keras model
        X_test: Test sequences
        y_test: True labels
    
    Returns:
        y_pred_classes: Predicted class labels
        metrics: Dictionary of evaluation metrics
    """
    print("\n" + "=" * 70)
    print("📊 EVALUATING MODEL PERFORMANCE")
    print("=" * 70)
    
    # Generate predictions
    print("\n   🔄 Generating predictions...")
    y_pred_prob = model.predict(X_test, batch_size=256, verbose=1)
    y_pred_classes = np.argmax(y_pred_prob, axis=1)
    
    # Calculate overall accuracy
    accuracy = accuracy_score(y_test, y_pred_classes)
    
    # Determine present classes (handle missing classes gracefully)
    present_classes = sorted(set(y_test.tolist()))
    present_labels = [i for i in range(NUM_CLASSES) if i in present_classes]
    present_names = [CLASS_NAMES[i] for i in present_labels]
    
    print(f"\n   📋 Classes present in test set: {present_labels}")
    print(f"      Class names: {present_names}")
    
    # Generate classification report
    report_dict = classification_report(
        y_test,
        y_pred_classes,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True
    )
    
    report_text = classification_report(
        y_test,
        y_pred_classes,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        zero_division=0
    )
    
    print("\n" + "=" * 70)
    print("📝 CLASSIFICATION REPORT")
    print("=" * 70)
    print(report_text)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes, labels=list(range(NUM_CLASSES)))
    
    print("\n   📊 Confusion Matrix:")
    print("   " + "-" * 50)
    print(f"                  Predicted")
    print(f"           {'  '.join([f'{name:>10}' for name in CLASS_NAMES])}")
    print("   " + "-" * 50)
    for i, name in enumerate(CLASS_NAMES):
        row_str = f"   {name:>10} "
        row_str += "  ".join([f"{cm[i][j]:>10,}" for j in range(NUM_CLASSES)])
        print(row_str)
    print("   " + "-" * 50)
    
    metrics = {
        'accuracy': accuracy,
        'classification_report': report_dict,
        'confusion_matrix': cm.tolist(),
        'y_pred_classes': y_pred_classes,
        'y_pred_prob': y_pred_prob
    }
    
    return y_pred_classes, metrics


def measure_inference_speed(model: tf.keras.Model, X_test: np.ndarray) -> float:
    """
    Measure average inference time per sample.
    
    Args:
        model: Trained Keras model
        X_test: Test sequences
    
    Returns:
        Average inference time in milliseconds per sample
    """
    print("\n" + "=" * 70)
    print("⚡ MEASURING INFERENCE SPEED")
    print("=" * 70)
    
    # Warm-up run (TensorFlow graph compilation)
    print("   🔥 Warm-up run (graph compilation)...")
    _ = model.predict(X_test[:100], verbose=0)
    
    # Benchmark on larger sample
    test_samples = min(10000, len(X_test))
    X_benchmark = X_test[:test_samples]
    
    print(f"   ⏱️  Benchmarking on {test_samples:,} samples...")
    
    start_time = time.time()
    _ = model.predict(X_benchmark, batch_size=256, verbose=0)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_per_sample = (total_time / test_samples) * 1000  # Convert to ms
    
    print(f"\n   📈 Inference Performance:")
    print(f"      - Total time: {total_time:.4f} seconds")
    print(f"      - Samples processed: {test_samples:,}")
    print(f"      - Average time per sample: {avg_time_per_sample:.4f} ms")
    print(f"      - Throughput: {test_samples / total_time:.2f} samples/second")
    
    return avg_time_per_sample


def save_classification_report(metrics: dict) -> None:
    """
    Save classification report to text file.
    
    Args:
        metrics: Dictionary containing evaluation metrics
    """
    print("\n   💾 Saving classification report...")
    
    report_path = os.path.join(REPORTS_DIR, 'classification_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("BiLSTM MODEL - CLASSIFICATION REPORT\n")
        f.write("Network Intrusion Detection System\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: BiLSTM (Bidirectional)\n\n")
        
        f.write("Overall Metrics:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n\n")
        
        f.write("Per-Class Metrics:\n")
        f.write("-" * 50 + "\n")
        
        report_dict = metrics['classification_report']
        for i, class_name in enumerate(CLASS_NAMES):
            if class_name in report_dict:
                class_metrics = report_dict[class_name]
                f.write(f"\n{class_name} (Class {i}):\n")
                f.write(f"  Precision: {class_metrics['precision']:.4f} ({class_metrics['precision']*100:.2f}%)\n")
                f.write(f"  Recall:    {class_metrics['recall']:.4f} ({class_metrics['recall']*100:.2f}%)\n")
                f.write(f"  F1-Score:  {class_metrics['f1-score']:.4f} ({class_metrics['f1-score']*100:.2f}%)\n")
                f.write(f"  Support:   {class_metrics['support']:,} samples\n")
        
        # Macro and weighted averages
        f.write("\n" + "=" * 50 + "\n")
        f.write("Aggregated Metrics:\n")
        f.write("-" * 50 + "\n")
        
        macro_avg = report_dict['macro avg']
        weighted_avg = report_dict['weighted avg']
        
        f.write(f"\nMacro Average:\n")
        f.write(f"  Precision: {macro_avg['precision']:.4f}\n")
        f.write(f"  Recall:    {macro_avg['recall']:.4f}\n")
        f.write(f"  F1-Score:  {macro_avg['f1-score']:.4f}\n")
        
        f.write(f"\nWeighted Average:\n")
        f.write(f"  Precision: {weighted_avg['precision']:.4f}\n")
        f.write(f"  Recall:    {weighted_avg['recall']:.4f}\n")
        f.write(f"  F1-Score:  {weighted_avg['f1-score']:.4f}\n")
        
        # Confusion matrix
        f.write("\n" + "=" * 50 + "\n")
        f.write("Confusion Matrix:\n")
        f.write("-" * 50 + "\n\n")
        
        cm = np.array(metrics['confusion_matrix'])
        f.write("              Predicted\n")
        f.write("        " + "  ".join([f"{name:>10}" for name in CLASS_NAMES]) + "\n")
        f.write("-" * 50 + "\n")
        for i, name in enumerate(CLASS_NAMES):
            row = f"{name:>10} " + "  ".join([f"{cm[i][j]:>10,}" for j in range(NUM_CLASSES)])
            f.write(row + "\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"   ✅ Classification report saved: {report_path}")


def plot_confusion_matrix(cm: np.ndarray) -> None:
    """
    Create and save confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix (3x3 numpy array)
    """
    print("\n   🎨 Generating confusion matrix heatmap...")
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap with annotations
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        annot_kws={'size': 14, 'weight': 'bold'},
        cbar_kws={'label': 'Count'}
    )
    
    plt.title('BiLSTM Confusion Matrix\nNetwork Intrusion Detection (3-Class)',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
    plt.ylabel('True Label', fontsize=13, fontweight='bold')
    
    # Add percentage annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            total = cm[i, :].sum()
            if total > 0:
                pct = cm[i, j] / total * 100
                plt.text(j + 0.5, i + 0.7, f'({pct:.1f}%)',
                        ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    
    save_path = os.path.join(REPORTS_DIR, 'confusion_matrix_bilstm.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Confusion matrix saved: {save_path}")


def load_and_plot_training_history() -> None:
    """
    Load training history from the trained model and plot curves.
    
    Note: If history is not available in model, this will be skipped.
    """
    print("\n   🎨 Attempting to plot training history...")
    
    # Try to load history from pickle if saved during training
    history_path = os.path.join(MODELS_DIR, 'bilstm_history.pkl')
    
    if os.path.exists(history_path):
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        axes[0].plot(history['loss'], label='Training Loss', linewidth=2, color='#e74c3c')
        axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2, color='#3498db')
        axes[0].set_title('BiLSTM Model Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(history['accuracy'], label='Training Accuracy', linewidth=2, color='#e74c3c')
        axes[1].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='#3498db')
        axes[1].set_title('BiLSTM Model Accuracy', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(REPORTS_DIR, 'training_history.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Training history plot saved: {save_path}")
    else:
        print(f"   ⚠️  Training history not found (already created during training)")
        print(f"      Check: reports/bilstm/training_history.png")


def main() -> None:
    """Main evaluation pipeline orchestration."""
    
    print("\n" + "=" * 70)
    print("🔍 BiLSTM MODEL EVALUATION - NETWORK INTRUSION DETECTION")
    print("   Comprehensive Performance Analysis")
    print("=" * 70)
    
    # Step 1: Load test data
    X_test, y_test = load_test_data()
    
    # Step 2: Load trained model
    model = load_trained_model()
    
    # Step 3: Evaluate model
    y_pred_classes, metrics = evaluate_model(model, X_test, y_test)
    
    # Step 4: Measure inference speed
    avg_inference_time = measure_inference_speed(model, X_test)
    metrics['avg_inference_time_ms'] = avg_inference_time
    
    # Step 5: Save classification report
    save_classification_report(metrics)
    
    # Step 6: Plot confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(cm)
    
    # Step 7: Plot training history (if available)
    load_and_plot_training_history()
    
    # Final summary
    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    
    print(f"\n   📊 Performance Summary:")
    print(f"      - Overall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    
    report_dict = metrics['classification_report']
    print(f"      - Macro Avg F1-Score: {report_dict['macro avg']['f1-score']:.4f}")
    print(f"      - Weighted Avg F1-Score: {report_dict['weighted avg']['f1-score']:.4f}")
    print(f"      - Avg Inference Time: {avg_inference_time:.4f} ms/sample")
    
    print(f"\n   📁 Generated Outputs:")
    print(f"      - Classification Report: reports/bilstm/classification_report.txt")
    print(f"      - Confusion Matrix: reports/bilstm/confusion_matrix_bilstm.png")
    print(f"      - Training History: reports/bilstm/training_history.png")
    
    print(f"\n   🎯 Next Steps:")
    print(f"      1. Review classification report for per-class performance")
    print(f"      2. Examine confusion matrix for misclassification patterns")
    print(f"      3. Compare with LSTM model (models/lstm_best.keras)")
    print(f"      4. If performance is suboptimal, consider hyperparameter tuning")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
