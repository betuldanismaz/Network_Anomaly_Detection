#!/usr/bin/env python3
"""
BiLSTM Model Training for Network Intrusion Detection System
Sprint 5 - Task 1.3: Bidirectional LSTM Implementation

This script trains a BiLSTM model for 3-class network attack classification:
    - Class 0: Benign (Normal traffic)
    - Class 1: Volumetric (DoS, DDoS attacks)
    - Class 2: Semantic/Infiltration (PortScan, Web Attacks, etc.)

Features:
    - Memory-efficient tf.data.Dataset pipeline
    - Class weight handling for imbalanced data
    - Robust evaluation handling for missing classes
    - Comprehensive reporting with confusion matrix visualization
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Bidirectional, LSTM, Dense, Dropout, BatchNormalization, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter

# --- PATH SETUP ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

DATA_DIR = os.path.join(ROOT, 'data', 'processed_lstm')
MODELS_DIR = os.path.join(ROOT, 'models')
REPORTS_DIR = os.path.join(ROOT, 'reports', 'bilstm')

# --- HYPERPARAMETERS ---
EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 0.001
LSTM_UNITS_1 = 128  # First BiLSTM layer
LSTM_UNITS_2 = 64   # Second BiLSTM layer
DROPOUT_RATE = 0.3
NUM_CLASSES = 3

# Class names for reporting
CLASS_NAMES = ['Benign', 'Volumetric', 'Semantic']

# Create output directories
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def load_data_as_dataset():
    """
    Load preprocessed LSTM data using tf.data.Dataset for memory efficiency.
    
    Returns:
        train_dataset: tf.data.Dataset for training
        test_dataset: tf.data.Dataset for testing
        X_test: numpy array for evaluation (loaded separately)
        y_test: numpy array for evaluation (loaded separately)
    """
    print("\n" + "=" * 60)
    print("📂 LOADING DATA")
    print("=" * 60)
    
    # File paths
    X_train_path = os.path.join(DATA_DIR, 'X_train.npy')
    y_train_path = os.path.join(DATA_DIR, 'y_train.npy')
    X_test_path = os.path.join(DATA_DIR, 'X_test.npy')
    y_test_path = os.path.join(DATA_DIR, 'y_test.npy')
    
    # Verify files exist
    for path in [X_train_path, y_train_path, X_test_path, y_test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Data file not found: {path}")
    
    # Load data using memory mapping for initial inspection
    X_train_mmap = np.load(X_train_path, mmap_mode='r')
    y_train = np.load(y_train_path)
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    
    print(f"✅ X_train shape: {X_train_mmap.shape}")
    print(f"✅ y_train shape: {y_train.shape}")
    print(f"✅ X_test shape: {X_test.shape}")
    print(f"✅ y_test shape: {y_test.shape}")
    
    # Display class distribution
    train_dist = Counter(y_train.tolist())
    test_dist = Counter(y_test.tolist())
    print(f"\n📊 Training class distribution: {dict(train_dist)}")
    print(f"📊 Test class distribution: {dict(test_dist)}")
    
    
    # Warn about missing classes in test set
    missing_classes = set(range(NUM_CLASSES)) - set(test_dist.keys())
    if missing_classes:
        print(f"⚠️  WARNING: Test set is missing classes: {missing_classes}")
    
    # ⚠️  MEMORY-EFFICIENT LOADING FOR SMALL GPU
    # Use generator approach to avoid loading full 1.8GB dataset into GPU at once
    print("\n🔄 Creating memory-efficient tf.data.Dataset pipeline...")
    
    # Generator function for training data (streams from disk)
    def train_generator():
        """Generator that streams training data from disk."""
        indices = np.arange(len(y_train))
        np.random.seed(42)  # Reproducible shuffle
        np.random.shuffle(indices)
        
        for idx in indices:
            # Load one sample from memory-mapped array
            yield X_train_mmap[idx].astype(np.float32), np.int32(y_train[idx])
    
    # Create dataset from generator
    train_dataset = tf.data.Dataset.from_generator(
        train_generator,
        output_signature=(
            tf.TensorSpec(shape=(10, 20), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    
    # For test set, we can load directly (smaller size: ~450MB)
    X_test = X_test[:].astype(np.float32)  # Convert to float32
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)
    
    print("✅ Pipeline configured: generator → batch(256) → prefetch(AUTOTUNE)")
    print("ℹ️  Training data streams from disk (memory-efficient)")
    
    return train_dataset, test_dataset, X_test, y_test, y_train



def load_class_weights():
    """
    Load precomputed class weights from JSON file.
    Handles string keys by converting them to integers.
    
    Returns:
        dict: Class weights with integer keys
    """
    weights_path = os.path.join(MODELS_DIR, 'class_weights.json')
    
    if os.path.exists(weights_path):
        print(f"\n📦 Loading class weights from: {weights_path}")
        with open(weights_path, 'r') as f:
            weights = json.load(f)
        
        # Convert string keys to integers (JSON stores keys as strings)
        class_weights = {int(k): float(v) for k, v in weights.items()}
        
        # Ensure all classes have weights
        for c in range(NUM_CLASSES):
            if c not in class_weights:
                class_weights[c] = 1.0
                print(f"⚠️  Class {c} weight not found, defaulting to 1.0")
        
        print(f"✅ Class weights loaded: {class_weights}")
        return class_weights
    else:
        print(f"⚠️  Class weights file not found: {weights_path}")
        print("   Using default weights (1.0 for all classes)")
        return {i: 1.0 for i in range(NUM_CLASSES)}


def build_bilstm_model(input_shape, num_classes=3):
    """
    Build BiLSTM model architecture.
    
    Architecture:
        Input: (batch_size, time_steps=10, features=20)
        BiLSTM Layer 1: 128 units (returns sequences)
        BatchNormalization + Dropout(0.3)
        BiLSTM Layer 2: 64 units (returns final hidden state)
        BatchNormalization + Dropout(0.3)
        Dense: 64 units + ReLU
        Dropout(0.3)
        Output: 3 units + Softmax
    
    Args:
        input_shape: Tuple of (time_steps, features)
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras Sequential model
    """
    print("\n" + "=" * 60)
    print("🏗️  BUILDING BILSTM MODEL")
    print("=" * 60)
    
    model = Sequential([
        # Input layer
        Input(shape=input_shape, name='input'),
        
        # First BiLSTM layer - returns sequences for stacking
        Bidirectional(
            LSTM(LSTM_UNITS_1, return_sequences=True, name='lstm_1'),
            name='bilstm_1'
        ),
        BatchNormalization(name='bn_1'),
        Dropout(DROPOUT_RATE, name='dropout_1'),
        
        # Second BiLSTM layer - returns final hidden state
        Bidirectional(
            LSTM(LSTM_UNITS_2, return_sequences=False, name='lstm_2'),
            name='bilstm_2'
        ),
        BatchNormalization(name='bn_2'),
        Dropout(DROPOUT_RATE, name='dropout_2'),
        
        # Fully connected layer
        Dense(64, activation='relu', name='fc_1'),
        Dropout(DROPOUT_RATE, name='dropout_3'),
        
        # Output layer - 3 classes with Softmax
        Dense(num_classes, activation='softmax', name='output')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ Model built with input shape: {input_shape}")
    print(f"✅ Output classes: {num_classes}")
    print(f"✅ Optimizer: Adam (lr={LEARNING_RATE})")
    print(f"✅ Loss: sparse_categorical_crossentropy")
    
    return model


def create_callbacks():
    """
    Create training callbacks for early stopping, checkpointing, and LR scheduling.
    
    Returns:
        list: List of Keras callbacks
    """
    print("\n📋 Configuring callbacks...")
    
    callbacks = [
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),
        
        # Save best model checkpoint
        ModelCheckpoint(
            filepath=os.path.join(MODELS_DIR, 'bilstm_best.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
            mode='min'
        ),
        
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
            mode='min'
        ),
        
        # TensorBoard logging
        TensorBoard(
            log_dir=os.path.join(REPORTS_DIR, 'logs', datetime.now().strftime('%Y%m%d-%H%M%S')),
            histogram_freq=1
        )
    ]
    
    print(f"   ✓ EarlyStopping (patience=5)")
    print(f"   ✓ ModelCheckpoint (save_best_only)")
    print(f"   ✓ ReduceLROnPlateau (factor=0.5, patience=3)")
    print(f"   ✓ TensorBoard logging")
    
    return callbacks


def train_model(model, train_dataset, test_dataset, class_weights):
    """
    Train the BiLSTM model using tf.data.Dataset.
    
    Args:
        model: Compiled Keras model
        train_dataset: tf.data.Dataset for training
        test_dataset: tf.data.Dataset for validation
        class_weights: Dictionary of class weights
    
    Returns:
        history: Training history object
    """
    print("\n" + "=" * 60)
    print("🚀 STARTING TRAINING")
    print("=" * 60)
    
    model.summary()
    
    callbacks = create_callbacks()
    
    print(f"\n📈 Training for up to {EPOCHS} epochs...")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Class weights: {class_weights}")
    print("-" * 60)
    
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model and generate classification report.
    Handles missing classes gracefully.
    
    Args:
        model: Trained Keras model
        X_test: Test features (numpy array)
        y_test: Test labels (numpy array)
    
    Returns:
        y_pred_classes: Predicted class labels
    """
    print("\n" + "=" * 60)
    print("📊 EVALUATING MODEL")
    print("=" * 60)
    
    # Get predictions
    print("🔄 Generating predictions...")
    y_pred = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Determine which classes are present in test set
    present_classes = sorted(set(y_test.tolist()))
    present_labels = [i for i in range(NUM_CLASSES) if i in present_classes]
    present_names = [CLASS_NAMES[i] for i in present_labels]
    
    print(f"\n📋 Classes present in test set: {present_labels}")
    print(f"   Class names: {present_names}")
    
    # Classification report (handles missing classes)
    report = classification_report(
        y_test, 
        y_pred_classes, 
        labels=present_labels,
        target_names=present_names,
        zero_division=0
    )
    
    print("\n" + "=" * 60)
    print("📝 CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)
    
    # Full classification report with all classes for reference
    full_report = classification_report(
        y_test, 
        y_pred_classes, 
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        zero_division=0
    )
    
    # Save report
    report_path = os.path.join(REPORTS_DIR, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("BiLSTM Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Test Set Results:\n")
        f.write("-" * 40 + "\n")
        f.write(report)
        f.write("\n\nFull Report (All Classes):\n")
        f.write("-" * 40 + "\n")
        f.write(full_report)
        f.write(f"\n\nNote: Classes present in test set: {present_labels}")
        if len(present_labels) < NUM_CLASSES:
            missing = set(range(NUM_CLASSES)) - set(present_labels)
            f.write(f"\nWarning: Missing classes in test set: {list(missing)}")
    
    print(f"\n✅ Classification report saved to: {report_path}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes, labels=list(range(NUM_CLASSES)))
    print("\n📊 Confusion Matrix:")
    print(cm)
    
    return y_pred_classes, cm


def plot_confusion_matrix(cm, save_path):
    """
    Create and save confusion matrix visualization.
    
    Args:
        cm: Confusion matrix (numpy array)
        save_path: Path to save the plot
    """
    print("\n🎨 Generating confusion matrix plot...")
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        annot_kws={'size': 14}
    )
    
    plt.title('BiLSTM Confusion Matrix\nNetwork Intrusion Detection', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Confusion matrix plot saved to: {save_path}")


def plot_training_history(history, save_dir):
    """
    Plot and save training history (loss and accuracy curves).
    
    Args:
        history: Keras training history object
        save_dir: Directory to save plots
    """
    print("\n🎨 Generating training history plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training history plot saved to: {save_path}")


def save_model(model):
    """
    Save the trained model in multiple formats.
    
    Args:
        model: Trained Keras model
    """
    print("\n" + "=" * 60)
    print("💾 SAVING MODEL")
    print("=" * 60)
    
    # Keras format (.keras)
    keras_path = os.path.join(MODELS_DIR, 'bilstm_model.keras')
    model.save(keras_path)
    print(f"✅ Keras model saved: {keras_path}")
    
    # SavedModel format (for TensorFlow Serving)
    savedmodel_path = os.path.join(MODELS_DIR, 'bilstm_savedmodel')
    model.save(savedmodel_path)
    print(f"✅ SavedModel saved: {savedmodel_path}")
    
    # Save model config as JSON
    config_path = os.path.join(MODELS_DIR, 'bilstm_config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'input_shape': [10, 20],
            'num_classes': NUM_CLASSES,
            'lstm_units_1': LSTM_UNITS_1,
            'lstm_units_2': LSTM_UNITS_2,
            'dropout_rate': DROPOUT_RATE,
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'class_names': CLASS_NAMES,
            'created_at': datetime.now().isoformat()
        }, f, indent=2)
    print(f"✅ Model config saved: {config_path}")


def main():
    """Main training pipeline."""
    print("\n" + "=" * 70)
    print("🧠 BiLSTM MODEL TRAINING - NETWORK INTRUSION DETECTION")
    print("   Sprint 5 - Task 1.3: Bidirectional LSTM Implementation")
    print("=" * 70)
    print(f"   TensorFlow version: {tf.__version__}")
    print(f"   GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"   GPU devices: {[gpu.name for gpu in gpus]}")
        # Enable memory growth to avoid OOM
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    # 1. Load data
    train_dataset, test_dataset, X_test, y_test, y_train = load_data_as_dataset()
    
    # 2. Load class weights
    class_weights = load_class_weights()
    
    # 3. Build model
    input_shape = (10, 20)  # (time_steps, features)
    model = build_bilstm_model(input_shape, NUM_CLASSES)
    
    # 4. Train model
    history = train_model(model, train_dataset, test_dataset, class_weights)
    
    # 5. Evaluate model
    y_pred_classes, cm = evaluate_model(model, X_test, y_test)
    
    # 6. Generate plots
    plot_confusion_matrix(cm, os.path.join(REPORTS_DIR, 'confusion_matrix_bilstm.png'))
    plot_training_history(history, REPORTS_DIR)
    
    # 7. Save model
    save_model(model)
    
    # Final summary
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"   📁 Model saved to: {MODELS_DIR}")
    print(f"   📊 Reports saved to: {REPORTS_DIR}")
    print(f"   📈 Best validation loss: {min(history.history['val_loss']):.4f}")
    print(f"   🎯 Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
