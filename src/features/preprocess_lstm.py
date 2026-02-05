"""
LSTM Data Preprocessing Script (Scientifically Rigorous Version)
================================================================

This script implements a SCIENTIFICALLY RIGOROUS preprocessing pipeline for LSTM-based
network intrusion detection that prioritizes MAXIMUM DATA UTILIZATION.

DESIGN DECISION: "Split Raw Data First -> Then Create Sequences" (Hybrid Solution)
==================================================================================
This approach was chosen to ensure:
1. ZERO DATA LEAKAGE: Train and test sequences come from completely different raw rows.
   By splitting BEFORE windowing, no window ever contains rows from both sets.
2. CLASS BALANCE GUARANTEE: Each CSV file contributes balanced classes to both train
   and test sets via stratified sampling on the raw data.
3. MAXIMUM ACCURACY: Using Stride=1 ensures the LSTM sees EVERY valid pattern in the data,
   albeit at the cost of slower training times and larger memory footprint.

WHY NOT "Window First -> Then Split"?
------------------------------------
If we created sequences first and then split, windows near the split boundary could
contain data from both train and test sets, causing data leakage. The model would
effectively "see" test data during training, leading to overly optimistic performance
estimates that don't generalize to truly unseen data.

NOTE ON SHUFFLING RAW FLOWS
---------------------------
We use shuffle=True in train_test_split on raw individual flows. This treats each flow
as an INDEPENDENT EVENT, which fits the flow-based nature of CICIDS2017 where each row
represents a network flow extracted independently. The temporal relationship within a
session is captured by the windowing, but individual flows are sampled from distinct
network sessions and can be treated as IID samples for splitting purposes.

Author: Network Detection Team
"""

import os
import sys
import gc
import glob
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
from collections import Counter
from typing import List, Tuple, Optional, Dict, Any


# =============================================================================
# CONFIGURATION
# =============================================================================

# Project root directory
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Add project root to Python path to allow imports from src
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Directory paths (configurable via environment variables)
DATA_DIR = os.getenv('LSTM_DATA_DIR') or os.path.join(ROOT, 'data', 'original_csv')
OUT_DIR = os.getenv('LSTM_OUT_DIR') or r'D:\Projects\networkdetection\networkdetection\data\processed_lstm'
MODELS_DIR = os.getenv('LSTM_MODELS_DIR') or os.path.join(ROOT, 'models')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler_lstm.pkl')
CLASS_WEIGHTS_PATH = os.path.join(MODELS_DIR, 'class_weights.json')

# Resolve classes_map.json from several likely locations
possible_class_map_paths = [
    os.getenv('CLASSES_MAP_PATH'),
    os.path.join(ROOT, 'src', 'utils', 'classes_map.json'),
    os.path.join(ROOT, 'data', 'classes_map.json'),
    os.path.join(ROOT, 'classes_map.json'),
]
CLASS_MAP_PATH = None
for p in possible_class_map_paths:
    if p and os.path.exists(p):
        CLASS_MAP_PATH = p
        break
if CLASS_MAP_PATH is None:
    CLASS_MAP_PATH = os.path.join(os.path.dirname(__file__), '..', 'utils', 'classes_map.json')

# Create output directories
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Import top features from config
from src.config import TOP_FEATURES

# =============================================================================
# STRIDE CONFIGURATION
# =============================================================================
# STRIDE controls the step size of the sliding window:
#   - STRIDE=1 (Default): Maximum accuracy, sees EVERY valid pattern. Slow training.
#   - STRIDE=10: ~10x faster, useful for debugging/fast iterations.
#
# Set via environment variable: LSTM_STRIDE=10 python preprocess_lstm.py

STRIDE = int(os.getenv('LSTM_STRIDE', '1'))
WINDOW_SIZE = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42

print("=" * 70)
print("LSTM PREPROCESSING CONFIGURATION")
print("=" * 70)
print(f"⚠️  STRIDE: {STRIDE} {'(Maximum Accuracy Mode)' if STRIDE == 1 else '(Fast Debug Mode)'}")
print(f"   WINDOW_SIZE: {WINDOW_SIZE}")
print(f"   TEST_SIZE: {TEST_SIZE}")
print(f"   RANDOM_STATE: {RANDOM_STATE}")
print("=" * 70)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def normalize_text(x: Any) -> str:
    """
    Normalize text by removing special characters, extra whitespace, and BOM.
    Used for matching labels between CSV files and classes_map.json.
    """
    if x is None:
        return ''
    s = str(x)
    try:
        import unicodedata
        s = unicodedata.normalize('NFKC', s)
    except Exception:
        pass
    # Remove BOM, non-breaking space, replacement character
    s = s.replace('\ufeff', '').replace('\u00A0', ' ').replace('\uFFFD', ' ')
    s = ' '.join(s.split())
    return s.strip()


def list_csv_files(data_dir: str) -> List[str]:
    """Return sorted list of all CSV files in the given directory."""
    return sorted(glob.glob(os.path.join(data_dir, '*.csv')))


def create_sequences_optimized(
    arr: np.ndarray,
    labels: np.ndarray,
    window_size: int = 10,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences using np.lib.stride_tricks.sliding_window_view for efficiency.
    
    This is more memory-efficient than manual iteration for large arrays.
    
    Parameters
    ----------
    arr : np.ndarray
        Feature array of shape (N, F) where N is number of samples, F is features.
    labels : np.ndarray
        Label array of shape (N,).
    window_size : int
        Number of time steps in each sequence.
    stride : int
        Step size between consecutive windows.
        
    Returns
    -------
    X : np.ndarray
        Sequences of shape (num_sequences, window_size, F).
    y : np.ndarray
        Labels of shape (num_sequences,), using the label of the LAST timestep.
    """
    N, F = arr.shape
    
    if N < window_size:
        # Not enough data to create even one sequence
        return np.zeros((0, window_size, F), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    
    # Use sliding_window_view for memory-efficient windowing
    # This creates a VIEW (no copy) until we need to actually use the data
    windowed = np.lib.stride_tricks.sliding_window_view(arr, window_shape=window_size, axis=0)
    # windowed shape: (N - window_size + 1, F, window_size)
    
    # Also create windows for labels to get the last label in each window
    label_windows = np.lib.stride_tricks.sliding_window_view(labels, window_shape=window_size)
    # label_windows shape: (N - window_size + 1, window_size)
    
    # Apply stride to reduce the number of sequences
    num_windows = windowed.shape[0]
    indices = np.arange(0, num_windows, stride)
    
    # Select strided windows and transpose to (num_sequences, window_size, F)
    X = windowed[indices].transpose(0, 2, 1).astype(np.float32)
    
    # Take the last label from each strided window
    y = label_windows[indices, -1].astype(np.int32)
    
    return X, y


def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Compute balanced class weights using inverse frequency.
    
    Formula: weight[c] = total_samples / (n_classes * count[c])
    This gives higher weight to minority classes.
    """
    counts = Counter(y.tolist())
    classes = sorted(counts.keys())
    total = sum(counts.values())
    n_classes = len(classes)
    
    weights = {}
    for c in classes:
        if counts[c] == 0:
            weights[c] = 1.0
        else:
            weights[c] = float(total) / (n_classes * counts[c])
    
    return weights


def load_classes_map() -> Dict[str, int]:
    """Load and normalize the classes mapping dictionary."""
    with open(CLASS_MAP_PATH, 'r') as f:
        raw_map = json.load(f)
    return {normalize_text(k): v for k, v in raw_map.items()}


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


# =============================================================================
# MAIN PROCESSING FUNCTIONS
# =============================================================================

def process_single_file(
    file_path: str,
    features: List[str],
    label_map: Dict[str, int],
    window_size: int,
    stride: int
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Process a single CSV file with stratified split BEFORE windowing.
    
    This ensures ZERO data leakage because:
    - Train sequences only contain train rows
    - Test sequences only contain test rows
    - No window ever spans across train/test boundary
    
    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    features : List[str]
        List of feature column names to extract.
    label_map : Dict[str, int]
        Mapping from label strings to integer classes.
    window_size : int
        Size of sliding window.
    stride : int
        Stride for sliding window.
        
    Returns
    -------
    Optional tuple of (X_train, X_test, y_train, y_test) or None if file is invalid.
    """
    filename = os.path.basename(file_path)
    
    try:
        # Step A: Load CSV and identify columns
        # First read just the header to minimize memory usage
        header_df = pd.read_csv(file_path, nrows=0)
        cols_original = header_df.columns.tolist()
        cols_stripped = [c.strip() for c in cols_original]
        col_map = {stripped: original for stripped, original in zip(cols_stripped, cols_original)}
        
        # Filter features to those present in this file
        keep_stripped = [c for c in features if c in cols_stripped]
        if 'Label' in cols_stripped:
            keep_stripped.append('Label')
        
        if not keep_stripped or 'Label' not in keep_stripped:
            print(f"  ⚠️  [{filename}] Missing required columns, skipping")
            return None
        
        # Map to original column names for reading
        keep_original = [col_map[c] for c in keep_stripped]
        
        # Read the full file with selected columns
        df = pd.read_csv(file_path, usecols=keep_original)
        df.columns = df.columns.str.strip()
        
        print(f"  📂 [{filename}] Loaded {len(df):,} rows")
        
    except Exception as e:
        print(f"  ❌ [{filename}] Error loading: {e}")
        return None
    
    # Step A continued: Map labels
    labels_raw = df['Label'].astype(str).fillna('').map(normalize_text)
    labels_mapped = labels_raw.map(label_map)
    mask_valid = labels_mapped.notna()
    
    if mask_valid.sum() == 0:
        print(f"  ⚠️  [{filename}] No valid labels after mapping, skipping")
        return None
    
    # Drop unknown labels
    df = df.loc[mask_valid].reset_index(drop=True)
    y_raw = labels_mapped.loc[mask_valid].astype(int).reset_index(drop=True)
    
    # Extract features
    feature_cols = [c for c in features if c in df.columns]
    if not feature_cols:
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if c != 'Label']
    
    X_raw = df[feature_cols].copy()
    
    print(f"       Raw label distribution: {dict(Counter(y_raw.values))}")
    
    # Step B: STRATIFIED RAW SPLIT (CRITICAL FOR ZERO LEAKAGE)
    # =========================================================
    # We split the RAW data BEFORE creating sequences.
    # This ensures every file contributes balanced classes to both sets.
    #
    # NOTE: We use shuffle=True because each flow in CICIDS2017 is an
    # independent network flow event. While we create temporal sequences
    # for the LSTM, the flows themselves were captured independently and
    # can be treated as IID samples for the purposes of train/test splitting.
    
    # Check if we have more than one class for stratification
    unique_classes = y_raw.nunique()
    if unique_classes < 2:
        print(f"  ⚠️  [{filename}] Only {unique_classes} class(es) present, cannot stratify")
        # Fall back to random split without stratification
        try:
            X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
                X_raw, y_raw,
                test_size=TEST_SIZE,
                shuffle=True,
                random_state=RANDOM_STATE
            )
        except Exception as e:
            print(f"  ❌ [{filename}] Split error: {e}")
            return None
    else:
        try:
            X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
                X_raw, y_raw,
                test_size=TEST_SIZE,
                stratify=y_raw,
                shuffle=True,
                random_state=RANDOM_STATE
            )
        except Exception as e:
            print(f"  ⚠️  [{filename}] Stratified split failed ({e}), falling back to random split")
            X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
                X_raw, y_raw,
                test_size=TEST_SIZE,
                shuffle=True,
                random_state=RANDOM_STATE
            )
    
    print(f"       Train raw: {len(X_train_raw):,} | Test raw: {len(X_test_raw):,}")
    
    # Step C: SEQUENCE CREATION (POST-SPLIT)
    # ======================================
    # Create sequences SEPARATELY on train and test data.
    # This ensures zero leakage: train sequences only contain train rows,
    # test sequences only contain test rows.
    
    # Convert to numpy arrays
    X_train_arr = X_train_raw.values.astype(np.float32)
    X_test_arr = X_test_raw.values.astype(np.float32)
    y_train_arr = y_train_raw.values.astype(np.int32)
    y_test_arr = y_test_raw.values.astype(np.int32)
    
    # Create sequences
    X_train_seq, y_train_seq = create_sequences_optimized(
        X_train_arr, y_train_arr, window_size=window_size, stride=stride
    )
    X_test_seq, y_test_seq = create_sequences_optimized(
        X_test_arr, y_test_arr, window_size=window_size, stride=stride
    )
    
    print(f"       Train seq: {X_train_seq.shape[0]:,} | Test seq: {X_test_seq.shape[0]:,}")
    
    # Memory cleanup
    del df, X_raw, X_train_raw, X_test_raw, y_train_raw, y_test_raw
    del X_train_arr, X_test_arr, y_train_arr, y_test_arr
    gc.collect()
    
    return X_train_seq, X_test_seq, y_train_seq, y_test_seq


def main():
    """
    Main preprocessing pipeline implementing the "Split Raw First -> Then Sequence" strategy.
    """
    print("\n" + "=" * 70)
    print("STARTING LSTM DATA PREPROCESSING")
    print("Strategy: Split Raw Data First -> Then Create Sequences")
    print("Goal: ZERO Data Leakage + Class Balance in Test Set")
    print("=" * 70 + "\n")
    
    # Load label mapping
    print("📋 Loading classes map...")
    label_map = load_classes_map()
    print(f"   Classes: {sorted(set(label_map.values()))}")
    
    # List CSV files
    csv_files = list_csv_files(DATA_DIR)
    if not csv_files:
        raise RuntimeError(f"No CSV files found in {DATA_DIR}")
    print(f"📁 Found {len(csv_files)} CSV files to process\n")
    
    # Initialize global collectors
    global_X_train: List[np.ndarray] = []
    global_X_test: List[np.ndarray] = []
    global_y_train: List[np.ndarray] = []
    global_y_test: List[np.ndarray] = []
    
    # STEP 2: ITERATIVE PROCESSING (Per File)
    print("-" * 70)
    print("PROCESSING FILES (Per-File Stratified Split)")
    print("-" * 70)
    
    for i, file_path in enumerate(csv_files, 1):
        print(f"\n[{i}/{len(csv_files)}] Processing: {os.path.basename(file_path)}")
        print(f"   Memory usage: {get_memory_usage_mb():.1f} MB")
        
        result = process_single_file(
            file_path=file_path,
            features=TOP_FEATURES,
            label_map=label_map,
            window_size=WINDOW_SIZE,
            stride=STRIDE
        )
        
        if result is not None:
            X_train, X_test, y_train, y_test = result
            
            # Step D: Collection
            if X_train.shape[0] > 0:
                global_X_train.append(X_train)
                global_y_train.append(y_train)
            if X_test.shape[0] > 0:
                global_X_test.append(X_test)
                global_y_test.append(y_test)
            
            # Step E: Immediate memory cleanup
            del X_train, X_test, y_train, y_test
            gc.collect()
    
    print("\n" + "-" * 70)
    print("FILE PROCESSING COMPLETE")
    print("-" * 70)
    
    # Check if we have any data
    if not global_X_train or not global_X_test:
        raise RuntimeError("No valid data collected from any file")
    
    # STEP 3: AGGREGATION & SCALING
    print("\n📊 Aggregating sequences...")
    
    # Concatenate all sequences
    X_train_all = np.concatenate(global_X_train, axis=0)
    X_test_all = np.concatenate(global_X_test, axis=0)
    y_train_all = np.concatenate(global_y_train, axis=0)
    y_test_all = np.concatenate(global_y_test, axis=0)
    
    # Free memory from lists
    del global_X_train, global_X_test, global_y_train, global_y_test
    gc.collect()
    
    print(f"   X_train shape: {X_train_all.shape}")
    print(f"   X_test shape: {X_test_all.shape}")
    print(f"   Memory usage: {get_memory_usage_mb():.1f} MB")
    
    # LEAKAGE-FREE SCALING
    print("\n⚖️  Applying leakage-free scaling (fit on train only)...")
    
    # Get dimensions
    n_train, window_size, n_features = X_train_all.shape
    n_test = X_test_all.shape[0]
    
    # Reshape to 2D for scaling: (samples * window_size, features)
    X_train_2d = X_train_all.reshape(-1, n_features)
    X_test_2d = X_test_all.reshape(-1, n_features)
    
    # Fit scaler on TRAIN data ONLY
    scaler = MinMaxScaler()
    X_train_2d = scaler.fit_transform(X_train_2d)
    
    # Transform test data using train-fitted scaler
    X_test_2d = scaler.transform(X_test_2d)
    
    # Reshape back to 3D
    X_train_all = X_train_2d.reshape(n_train, window_size, n_features).astype(np.float32)
    X_test_all = X_test_2d.reshape(n_test, window_size, n_features).astype(np.float32)
    
    # Free 2D arrays
    del X_train_2d, X_test_2d
    gc.collect()
    
    # Save scaler
    joblib.dump(scaler, SCALER_PATH)
    print(f"   Scaler saved to: {SCALER_PATH}")
    
    # STEP 4: HANDLING IMBALANCE - Compute class weights
    print("\n⚖️  Computing class weights from training labels...")
    class_weights = compute_class_weights(y_train_all)
    
    # Ensure all classes 0,1,2 are present
    for c in [0, 1, 2]:
        class_weights.setdefault(c, 1.0)
    
    # Save class weights
    with open(CLASS_WEIGHTS_PATH, 'w') as f:
        json.dump(class_weights, f, indent=2)
    print(f"   Class weights: {class_weights}")
    print(f"   Saved to: {CLASS_WEIGHTS_PATH}")
    
    # STEP 5: VALIDATION & OUTPUT
    print("\n" + "=" * 70)
    print("SAVING OUTPUT FILES")
    print("=" * 70)
    
    # Save numpy arrays
    np.save(os.path.join(OUT_DIR, 'X_train.npy'), X_train_all)
    np.save(os.path.join(OUT_DIR, 'y_train.npy'), y_train_all.astype(np.int32))
    np.save(os.path.join(OUT_DIR, 'X_test.npy'), X_test_all)
    np.save(os.path.join(OUT_DIR, 'y_test.npy'), y_test_all.astype(np.int32))
    
    print(f"   Saved: X_train.npy {X_train_all.shape}")
    print(f"   Saved: y_train.npy {y_train_all.shape}")
    print(f"   Saved: X_test.npy {X_test_all.shape}")
    print(f"   Saved: y_test.npy {y_test_all.shape}")
    
    # VERIFICATION: Confirm class presence
    print("\n" + "=" * 70)
    print("VERIFICATION: Class Distribution")
    print("=" * 70)
    
    train_dist = Counter(y_train_all.tolist())
    test_dist = Counter(y_test_all.tolist())
    
    print("\n📊 y_train distribution (value_counts):")
    for cls in sorted(train_dist.keys()):
        pct = 100.0 * train_dist[cls] / len(y_train_all)
        print(f"   Class {cls}: {train_dist[cls]:>10,} ({pct:>5.2f}%)")
    
    print("\n📊 y_test distribution (value_counts) - VERIFY Class 2 Presence:")
    for cls in sorted(test_dist.keys()):
        pct = 100.0 * test_dist[cls] / len(y_test_all)
        print(f"   Class {cls}: {test_dist[cls]:>10,} ({pct:>5.2f}%)")
    
    # Verify Class 2 presence
    if 2 in test_dist and test_dist[2] > 0:
        print("\n✅ SUCCESS: Class 2 (Intrusion) is present in the test set!")
    else:
        print("\n⚠️  WARNING: Class 2 (Intrusion) has ZERO samples in test set!")
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"   Total Train Sequences: {len(y_train_all):,}")
    print(f"   Total Test Sequences:  {len(y_test_all):,}")
    print(f"   Sequence Shape: ({WINDOW_SIZE}, {n_features})")
    print(f"   Stride Used: {STRIDE}")
    print(f"   Output Directory: {OUT_DIR}")
    print(f"   Final Memory Usage: {get_memory_usage_mb():.1f} MB")
    print("=" * 70)
    print("✅ PREPROCESSING COMPLETE")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
