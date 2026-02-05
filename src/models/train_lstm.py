#!/usr/bin/env python3
"""
LSTM 3-class classification

Author: betül
Date: 05.02.2026
""" 

import os
import sys
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter
from typing import Tuple, Dict, List

# --- REPRODUCIBILITY SETUP (MUST BE FIRST) ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# --- PATH SETUP ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

DATA_DIR = os.path.join(ROOT, 'data', 'processed_lstm')
MODELS_DIR = os.path.join(ROOT, 'models')
REPORTS_DIR = os.path.join(ROOT, 'reports', 'lstm')

# --- HYPERPARAMETERS (STRICT SPECIFICATION) ---
LSTM_UNITS_1 = 128          # First LSTM layer
LSTM_UNITS_2 = 64           # Second LSTM layer
DROPOUT_RATE = 0.3          # Dropout probability
DENSE_UNITS = 64            # Fully connected layer
NUM_CLASSES = 3             # Output classes
LEARNING_RATE = 0.001       # Adam optimizer learning rate
BATCH_SIZE = 256            # Training batch size
EPOCHS = 50                 # Maximum training epochs

# Class names for reporting
CLASS_NAMES = ['Benign', 'Volumetric', 'Semantic']

# Create output directories
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(os.path.join(REPORTS_DIR, 'logs'), exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def check_gpu_availability() -> None:
    """
    Verify GPU availability and configure memory growth.
    
    This prevents TensorFlow from allocating all GPU memory at startup,
    which can cause OOM errors in multi-process environments.
    """
    print("\n" + "=" * 70)
    print("🖥️  HARDWARE CONFIGURATION")
    print("=" * 70)
    print(f"   TensorFlow version: {tf.__version__}")
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"   ✅ GPU available: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"      - {gpu.name}")
            try:
                # Enable memory growth to prevent OOM
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"      ✓ Memory growth enabled for {gpu.name}")
            except RuntimeError as e:
                print(f"      ⚠️  Warning: {e}")
    else:
        print("   ⚠️  No GPU detected - training will use CPU (slower)")
        print("      Consider using Google Colab or cloud GPU for faster training")


def load_data_as_dataset() -> Tuple[tf.data.Dataset, tf.data.Dataset, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load preprocessed LSTM data using tf.data.Dataset for memory efficiency.
    
    Returns:
        train_dataset: Batched and prefetched training dataset
        test_dataset: Batched and prefetched test dataset
        X_test: Raw test features (for final evaluation)
        y_test: Test labels (for final evaluation)
        y_train: Training labels (for class distribution analysis)
    
    Raises:
        FileNotFoundError: If required data files are missing
    """
    print("\n" + "=" * 70)
    print("📂 LOADING DATA")
    print("=" * 70)
    
    # Define file paths
    X_train_path = os.path.join(DATA_DIR, 'X_train.npy')
    y_train_path = os.path.join(DATA_DIR, 'y_train.npy')
    X_test_path = os.path.join(DATA_DIR, 'X_test.npy')
    y_test_path = os.path.join(DATA_DIR, 'y_test.npy')
    
    # Verify all files exist
    for path in [X_train_path, y_train_path, X_test_path, y_test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"❌ Required data file not found: {path}\n"
                f"   Please run preprocessing first: python src/features/preprocess_lstm.py"
            )
    
    print(f"   Loading from: {DATA_DIR}")
    
    # ⚠️  MEMORY-EFFICIENT LOADING FOR SMALL GPU
    # Use memory mapping to avoid loading full 1.8GB dataset into GPU at once
    print("   ⚠️  Using memory-mapped loading for GPU memory efficiency...")
    
    # Load with memory mapping (keeps data on disk, loads on demand)
    X_train_mmap = np.load(X_train_path, mmap_mode='r')
    y_train = np.load(y_train_path)
    X_test_mmap = np.load(X_test_path, mmap_mode='r')
    y_test = np.load(y_test_path)
    
    print(f"   ✅ X_train shape: {X_train_mmap.shape}")
    print(f"   ✅ y_train shape: {y_train.shape}")
    print(f"   ✅ X_test shape: {X_test_mmap.shape}")
    print(f"   ✅ y_test shape: {y_test.shape}")
    
    # Display class distribution
    train_dist = Counter(y_train.tolist())
    test_dist = Counter(y_test.tolist())
    
    print(f"\n   📊 Training class distribution:")
    for cls in sorted(train_dist.keys()):
        count = train_dist[cls]
        pct = count / len(y_train) * 100
        print(f"      Class {cls} ({CLASS_NAMES[cls]}): {count:,} samples ({pct:.2f}%)")
    
    print(f"\n   📊 Test class distribution:")
    for cls in sorted(test_dist.keys()):
        count = test_dist[cls]
        pct = count / len(y_test) * 100
        print(f"      Class {cls} ({CLASS_NAMES[cls]}): {count:,} samples ({pct:.2f}%)")
    
    # Warn about missing classes
    missing_test = set(range(NUM_CLASSES)) - set(test_dist.keys())
    if missing_test:
        print(f"\n   ⚠️  WARNING: Test set missing classes: {missing_test}")
    
    # Create tf.data.Dataset pipeline using generator (GPU memory-efficient)
    print("\n   🔄 Creating memory-efficient tf.data.Dataset pipeline...")
    
    # Generator function for training data (streams from disk)
    def train_generator():
        """Generator that streams training data from disk."""
        indices = np.arange(len(y_train))
        np.random.seed(SEED)  # Reproducible shuffle
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
    X_test = X_test_mmap[:].astype(np.float32)  # Load into memory
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)
    
    print("   ✅ Pipeline configured: generator → batch(256) → prefetch(AUTOTUNE)")
    print("   ℹ️  Training data streams from disk (memory-efficient)")
    
    return train_dataset, test_dataset, X_test, y_test, y_train


def load_class_weights() -> Dict[int, float]:
    """
    Load precomputed class weights from JSON file.
    
    Class weights compensate for imbalanced datasets by penalizing
    misclassifications of minority classes more heavily.
    
    Returns:
        Dictionary mapping class indices to weights
    """
    weights_path = os.path.join(MODELS_DIR, 'class_weights.json')
    
    if os.path.exists(weights_path):
        print(f"\n   📦 Loading class weights from: {weights_path}")
        with open(weights_path, 'r') as f:
            weights = json.load(f)
        
        # Convert string keys to integers (JSON serialization artifact)
        class_weights = {int(k): float(v) for k, v in weights.items()}
        
        # Ensure all classes have weights
        for c in range(NUM_CLASSES):
            if c not in class_weights:
                class_weights[c] = 1.0
                print(f"      ⚠️  Class {c} weight not found, defaulting to 1.0")
        
        print(f"   ✅ Class weights: {class_weights}")
        return class_weights
    else:
        print(f"   ⚠️  Class weights file not found: {weights_path}")
        print("      Using uniform weights (1.0 for all classes)")
        return {i: 1.0 for i in range(NUM_CLASSES)}


def build_lstm_model(input_shape: Tuple[int, int], num_classes: int = 3) -> tf.keras.Model:
    """
    Build LSTM model architecture according to strict specification.
    
    Architecture:
        Input: (timesteps, features) = (10, 20)
        LSTM Layer 1: 128 units (return_sequences=True)
        BatchNormalization → Dropout(0.3)
        LSTM Layer 2: 64 units (return_sequences=False)
        BatchNormalization → Dropout(0.3)
        Dense: 64 units (ReLU)
        Dropout(0.3)
        Output: 3 units (Softmax)
    
    Args:
        input_shape: (timesteps, features) tuple
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras Sequential model
    """
    print("\n" + "=" * 70)
    print("🏗️  BUILDING LSTM MODEL")
    print("=" * 70)
    
    model = Sequential([
        # Input layer
        Input(shape=input_shape, name='input'),
        
        # First LSTM layer - returns sequences for stacking
        LSTM(LSTM_UNITS_1, return_sequences=True, name='lstm_1'),
        BatchNormalization(name='bn_1'),
        Dropout(DROPOUT_RATE, name='dropout_1'),
        
        # Second LSTM layer - returns final hidden state
        LSTM(LSTM_UNITS_2, return_sequences=False, name='lstm_2'),
        BatchNormalization(name='bn_2'),
        Dropout(DROPOUT_RATE, name='dropout_2'),
        
        # Fully connected layer
        Dense(DENSE_UNITS, activation='relu', name='fc_1'),
        Dropout(DROPOUT_RATE, name='dropout_3'),
        
        # Output layer - 3 classes with Softmax
        Dense(num_classes, activation='softmax', name='output')
    ], name='LSTM_Network')
    
    # Compile model (STRICT SPECIFICATION)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"   ✅ Model built with input shape: {input_shape}")
    print(f"   ✅ Architecture: LSTM({LSTM_UNITS_1}) → LSTM({LSTM_UNITS_2}) → Dense({DENSE_UNITS}) → Dense({num_classes})")
    print(f"   ✅ Optimizer: Adam (lr={LEARNING_RATE})")
    print(f"   ✅ Loss: sparse_categorical_crossentropy")
    
    return model


def create_callbacks() -> List[tf.keras.callbacks.Callback]:
    """
    Create training callbacks according to strict specification.
    
    Returns:
        List of Keras callbacks:
            - EarlyStopping: Stops training if val_loss doesn't improve
            - ModelCheckpoint: Saves best model based on val_loss
            - ReduceLROnPlateau: Reduces learning rate when val_loss plateaus
            - TensorBoard: Logs metrics for visualization
    """
    print("\n   📋 Configuring callbacks...")
    
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    
    callbacks = [
        # EarlyStopping (STRICT SPECIFICATION)
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),
        
        # ModelCheckpoint (STRICT SPECIFICATION)
        ModelCheckpoint(
            filepath=os.path.join(MODELS_DIR, 'lstm_best.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
            mode='min'
        ),
        
        # ReduceLROnPlateau (STRICT SPECIFICATION)
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
            mode='min'
        ),
        
        # TensorBoard (STRICT SPECIFICATION)
        TensorBoard(
            log_dir=os.path.join(REPORTS_DIR, 'logs', timestamp),
            histogram_freq=1,
            write_graph=True,
            write_images=False
        )
    ]
    
    print("      ✓ EarlyStopping: monitor='val_loss', patience=5, restore_best_weights=True")
    print("      ✓ ModelCheckpoint: save_best_only=True → 'models/lstm_best.keras'")
    print("      ✓ ReduceLROnPlateau: factor=0.5, patience=3, min_lr=1e-6")
    print(f"      ✓ TensorBoard: log_dir='reports/lstm/logs/{timestamp}'")
    
    return callbacks


def train_model(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    test_dataset: tf.data.Dataset,
    class_weights: Dict[int, float]
) -> tf.keras.callbacks.History:
    """
    Train LSTM model with class weighting and callbacks.
    
    Args:
        model: Compiled Keras model
        train_dataset: Training data pipeline
        test_dataset: Validation data pipeline
        class_weights: Dictionary of class weights
    
    Returns:
        Training history object containing loss/accuracy curves
    """
    print("\n" + "=" * 70)
    print("🚀 STARTING TRAINING")
    print("=" * 70)
    
    # Display model architecture
    model.summary()
    
    # Configure callbacks
    callbacks = create_callbacks()
    
    print(f"\n   📈 Training configuration:")
    print(f"      - Max epochs: {EPOCHS}")
    print(f"      - Batch size: {BATCH_SIZE}")
    print(f"      - Learning rate: {LEARNING_RATE}")
    print(f"      - Class weights: {class_weights}")
    print("\n" + "-" * 70)
    
    # Train model
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETED")
    print("=" * 70)
    
    return history


def save_model_and_config(model: tf.keras.Model) -> None:
    """
    Save trained model and configuration metadata.
    
    Args:
        model: Trained Keras model
    """
    print("\n" + "=" * 70)
    print("💾 SAVING MODEL AND CONFIGURATION")
    print("=" * 70)
    
    # Save final model
    final_model_path = os.path.join(MODELS_DIR, 'lstm_model.keras')
    model.save(final_model_path)
    print(f"   ✅ Final model saved: {final_model_path}")
    
    # Save model configuration as JSON
    config_path = os.path.join(MODELS_DIR, 'lstm_config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'model_type': 'LSTM (Unidirectional)',
            'input_shape': [10, 20],
            'num_classes': NUM_CLASSES,
            'lstm_units_1': LSTM_UNITS_1,
            'lstm_units_2': LSTM_UNITS_2,
            'dense_units': DENSE_UNITS,
            'dropout_rate': DROPOUT_RATE,
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'max_epochs': EPOCHS,
            'class_names': CLASS_NAMES,
            'random_seed': SEED,
            'created_at': datetime.now().isoformat()
        }, f, indent=2)
    print(f"   ✅ Model config saved: {config_path}")
    
    # Note about best model
    best_model_path = os.path.join(MODELS_DIR, 'lstm_best.keras')
    if os.path.exists(best_model_path):
        print(f"   ✅ Best model checkpoint: {best_model_path}")
        print("      (Use this for evaluation - it has the best val_loss)")


def plot_training_history(history: tf.keras.callbacks.History) -> None:
    """
    Plot and save training history curves.
    
    Args:
        history: Keras training history object
    """
    print("\n   🎨 Generating training history plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2, color='#e74c3c')
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='#3498db')
    axes[0].set_title('LSTM Model Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, color='#e74c3c')
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='#3498db')
    axes[1].set_title('LSTM Model Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(REPORTS_DIR, 'training_history.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Training history saved: {save_path}")


def main() -> None:
    """Main training pipeline orchestration."""
    
    print("\n" + "=" * 70)
    print("🧠 LSTM MODEL TRAINING - NETWORK INTRUSION DETECTION")
    print("   Production-Grade Unidirectional LSTM Implementation")
    print("=" * 70)
    
    # Step 1: Hardware check
    check_gpu_availability()
    
    # Step 2: Load data
    train_dataset, test_dataset, X_test, y_test, y_train = load_data_as_dataset()
    
    # Step 3: Load class weights
    class_weights = load_class_weights()
    
    # Step 4: Build model
    input_shape = (10, 20)  # (timesteps, features)
    model = build_lstm_model(input_shape, NUM_CLASSES)
    
    # Step 5: Train model
    history = train_model(model, train_dataset, test_dataset, class_weights)
    
    # Step 6: Save model and configuration
    save_model_and_config(model)
    
    # Step 7: Plot training history
    plot_training_history(history)
    
    # Final summary
    print("\n" + "=" * 70)
    print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"\n   📁 Outputs:")
    print(f"      - Best model: models/lstm_best.keras")
    print(f"      - Final model: models/lstm_model.keras")
    print(f"      - Config: models/lstm_config.json")
    print(f"      - Training history: reports/lstm/training_history.png")
    print(f"      - TensorBoard logs: reports/lstm/logs/")
    
    print(f"\n   📊 Final Metrics:")
    print(f"      - Best validation loss: {min(history.history['val_loss']):.4f}")
    print(f"      - Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"      - Total epochs trained: {len(history.history['loss'])}")
    
    print(f"\n   🔍 Next Steps:")
    print(f"      1. Evaluate model: python src/models/evaluate_lstm.py")
    print(f"      2. View TensorBoard: tensorboard --logdir reports/lstm/logs")
    print(f"      3. Compare with BiLSTM model performance")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
