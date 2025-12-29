import numpy as np
import os
import sys
import json
from collections import Counter
import joblib

# LSTM DATA INTEGRITY AUDIT SCRIPT
# This script audits the preprocessed LSTM sequence data for quality and correctness

# ANSI colors for terminal output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
RESET = '\033[0m'


def print_header(title):
    """Print a formatted section header"""
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")


def print_subheader(title):
    """Print a formatted subsection header"""
    print(f"\n{CYAN}{'─'*70}")
    print(f" {title}")
    print(f"{'─'*70}{RESET}")


def check_lstm_data_health():
    """
    Comprehensive health check for LSTM preprocessed data.
    Validates sequences, labels, scaler, class weights, and data integrity.
    """
    
    # 1. Setup Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    # Get data directory from environment or use default
    data_dir = os.getenv('LSTM_OUT_DIR') or os.path.join(project_root, "data", "processed_lstm")
    models_dir = os.getenv('LSTM_MODELS_DIR') or os.path.join(project_root, "models")
    
    # Define expected files
    data_files = {
        "X_train": os.path.join(data_dir, "X_train.npy"),
        "y_train": os.path.join(data_dir, "y_train.npy"),
        "X_test": os.path.join(data_dir, "X_test.npy"),
        "y_test": os.path.join(data_dir, "y_test.npy")
    }
    
    model_files = {
        "scaler": os.path.join(models_dir, "scaler_lstm.pkl"),
        "class_weights": os.path.join(models_dir, "class_weights.json")
    }
    
    print_header("🚀 STARTING LSTM DATA HEALTH AUDIT")
    print(f"{BLUE}📁 Data Directory: {data_dir}")
    print(f"📁 Models Directory: {models_dir}{RESET}")
    
    # 2. File Existence Check
    print_header("📂 1. FILE EXISTENCE CHECK")
    
    all_files_exist = True
    for name, path in data_files.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"{GREEN}✅ {name:12s}: Found ({size_mb:.2f} MB){RESET}")
        else:
            print(f"{RED}❌ {name:12s}: NOT FOUND at {path}{RESET}")
            all_files_exist = False
    
    print()
    for name, path in model_files.items():
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            print(f"{GREEN}✅ {name:15s}: Found ({size_kb:.2f} KB){RESET}")
        else:
            print(f"{RED}❌ {name:15s}: NOT FOUND at {path}{RESET}")
            all_files_exist = False
    
    if not all_files_exist:
        print(f"\n{RED}❌ ERROR: Missing required files. Please run preprocess_lstm.py first.{RESET}")
        return
    
    # 3. Load Data
    print_header("📥 2. LOADING DATA")
    
    try:
        X_train = np.load(data_files["X_train"])
        y_train = np.load(data_files["y_train"])
        X_test = np.load(data_files["X_test"])
        y_test = np.load(data_files["y_test"])
        print(f"{GREEN}✅ Successfully loaded all numpy arrays{RESET}")
    except Exception as e:
        print(f"{RED}❌ Error loading data: {e}{RESET}")
        return
    
    try:
        scaler = joblib.load(model_files["scaler"])
        print(f"{GREEN}✅ Successfully loaded scaler{RESET}")
    except Exception as e:
        print(f"{RED}❌ Error loading scaler: {e}{RESET}")
        scaler = None
    
    try:
        with open(model_files["class_weights"], 'r') as f:
            class_weights = json.load(f)
        print(f"{GREEN}✅ Successfully loaded class weights{RESET}")
    except Exception as e:
        print(f"{RED}❌ Error loading class weights: {e}{RESET}")
        class_weights = None
    
    # 4. Shape and Dimension Verification
    print_header("📐 3. SHAPE AND DIMENSION VERIFICATION")
    
    print_subheader("Training Data")
    print(f"   X_train shape: {X_train.shape}")
    print(f"   └─ Samples:     {X_train.shape[0]:,}")
    print(f"   └─ Time Steps:  {X_train.shape[1]}")
    print(f"   └─ Features:    {X_train.shape[2]}")
    print(f"   y_train shape: {y_train.shape}")
    print(f"   └─ Samples:     {y_train.shape[0]:,}")
    
    print_subheader("Test Data")
    print(f"   X_test shape:  {X_test.shape}")
    print(f"   └─ Samples:     {X_test.shape[0]:,}")
    print(f"   └─ Time Steps:  {X_test.shape[1]}")
    print(f"   └─ Features:    {X_test.shape[2]}")
    print(f"   y_test shape:  {y_test.shape}")
    print(f"   └─ Samples:     {y_test.shape[0]:,}")
    
    # Verify dimensions match
    if X_train.ndim != 3:
        print(f"{RED}⚠️ ERROR: X_train should be 3D (samples, timesteps, features), got {X_train.ndim}D{RESET}")
    else:
        print(f"{GREEN}✅ X_train has correct 3D shape{RESET}")
    
    if X_test.ndim != 3:
        print(f"{RED}⚠️ ERROR: X_test should be 3D (samples, timesteps, features), got {X_test.ndim}D{RESET}")
    else:
        print(f"{GREEN}✅ X_test has correct 3D shape{RESET}")
    
    if y_train.ndim != 1:
        print(f"{RED}⚠️ ERROR: y_train should be 1D, got {y_train.ndim}D{RESET}")
    else:
        print(f"{GREEN}✅ y_train has correct 1D shape{RESET}")
    
    if y_test.ndim != 1:
        print(f"{RED}⚠️ ERROR: y_test should be 1D, got {y_test.ndim}D{RESET}")
    else:
        print(f"{GREEN}✅ y_test has correct 1D shape{RESET}")
    
    # Check sample count alignment
    if X_train.shape[0] != y_train.shape[0]:
        print(f"{RED}⚠️ ERROR: X_train samples ({X_train.shape[0]}) != y_train samples ({y_train.shape[0]}){RESET}")
    else:
        print(f"{GREEN}✅ Training samples aligned{RESET}")
    
    if X_test.shape[0] != y_test.shape[0]:
        print(f"{RED}⚠️ ERROR: X_test samples ({X_test.shape[0]}) != y_test samples ({y_test.shape[0]}){RESET}")
    else:
        print(f"{GREEN}✅ Test samples aligned{RESET}")
    
    # Check time steps consistency
    if X_train.shape[1] != X_test.shape[1]:
        print(f"{RED}⚠️ WARNING: Train time steps ({X_train.shape[1]}) != Test time steps ({X_test.shape[1]}){RESET}")
    else:
        print(f"{GREEN}✅ Time steps consistent across train/test{RESET}")
    
    # Check feature count consistency
    if X_train.shape[2] != X_test.shape[2]:
        print(f"{RED}⚠️ ERROR: Train features ({X_train.shape[2]}) != Test features ({X_test.shape[2]}){RESET}")
    else:
        print(f"{GREEN}✅ Feature count consistent across train/test{RESET}")
    
    # 5. Data Type Verification
    print_header("💾 4. DATA TYPE VERIFICATION")
    
    print(f"   X_train dtype: {X_train.dtype}")
    print(f"   X_test dtype:  {X_test.dtype}")
    print(f"   y_train dtype: {y_train.dtype}")
    print(f"   y_test dtype:  {y_test.dtype}")
    
    if X_train.dtype == np.float32 and X_test.dtype == np.float32:
        print(f"{GREEN}✅ Feature arrays use float32 (memory efficient){RESET}")
    elif X_train.dtype == np.float64 or X_test.dtype == np.float64:
        print(f"{YELLOW}⚠️ WARNING: Using float64. Consider float32 to save memory.{RESET}")
    
    if y_train.dtype in [np.int32, np.int64] and y_test.dtype in [np.int32, np.int64]:
        print(f"{GREEN}✅ Label arrays use integer type{RESET}")
    else:
        print(f"{RED}⚠️ WARNING: Labels should be integer type, got {y_train.dtype}{RESET}")
    
    # 6. Value Range Check (Scaling Verification)
    print_header("⚖️ 5. SCALING VERIFICATION")
    
    print_subheader("Training Data Range")
    train_min = X_train.min()
    train_max = X_train.max()
    train_mean = X_train.mean()
    train_std = X_train.std()
    
    print(f"   Min:    {train_min:.6f}")
    print(f"   Max:    {train_max:.6f}")
    print(f"   Mean:   {train_mean:.6f}")
    print(f"   Std:    {train_std:.6f}")
    
    print_subheader("Test Data Range")
    test_min = X_test.min()
    test_max = X_test.max()
    test_mean = X_test.mean()
    test_std = X_test.std()
    
    print(f"   Min:    {test_min:.6f}")
    print(f"   Max:    {test_max:.6f}")
    print(f"   Mean:   {test_mean:.6f}")
    print(f"   Std:    {test_std:.6f}")
    
    # Check if data is scaled (MinMaxScaler should give [0, 1] range)
    if -0.01 <= train_min <= 0.01 and 0.99 <= train_max <= 1.01:
        print(f"{GREEN}✅ Training data appears to be MinMax scaled to [0, 1]{RESET}")
    elif -0.01 <= test_min <= 0.01 and 0.99 <= test_max <= 1.01:
        print(f"{GREEN}✅ Test data appears to be MinMax scaled to [0, 1]{RESET}")
    else:
        print(f"{YELLOW}⚠️ WARNING: Data may not be properly scaled to [0, 1] range{RESET}")
        print(f"   Expected MinMaxScaler output, but got range [{train_min:.3f}, {train_max:.3f}]")
    
    # Check for extreme values
    if test_min < -0.1 or test_max > 1.1:
        print(f"{RED}⚠️ WARNING: Test data has values outside expected range!{RESET}")
        print(f"   This may indicate the scaler wasn't fitted properly on training data.")
    
    # 7. NaN and Infinity Check
    print_header("🧠 6. SANITY CHECK (NaNs & Infinity)")
    
    checks = [
        ("X_train", X_train),
        ("X_test", X_test),
        ("y_train", y_train),
        ("y_test", y_test)
    ]
    
    all_clean = True
    for name, arr in checks:
        nan_count = np.isnan(arr).sum()
        inf_count = np.isinf(arr).sum()
        
        if nan_count > 0 or inf_count > 0:
            print(f"{RED}⚠️ {name:12s}: Found {nan_count:,} NaNs and {inf_count:,} Inf values{RESET}")
            all_clean = False
        else:
            print(f"{GREEN}✅ {name:12s}: Clean (No NaNs or Inf){RESET}")
    
    if all_clean:
        print(f"\n{GREEN}✅ All arrays are clean!{RESET}")
    else:
        print(f"\n{RED}⚠️ WARNING: Found invalid values. This will cause training failures!{RESET}")
    
    # 8. Class Distribution Analysis
    print_header("📊 7. CLASS DISTRIBUTION ANALYSIS")
    
    print_subheader("Training Set Distribution")
    train_counts = Counter(y_train)
    train_total = len(y_train)
    
    for class_id in sorted(train_counts.keys()):
        count = train_counts[class_id]
        percentage = (count / train_total) * 100
        bar_length = int(percentage / 2)
        bar = '█' * bar_length
        print(f"   Class {class_id}: {count:7,} samples ({percentage:5.2f}%) {bar}")
    
    print_subheader("Test Set Distribution")
    test_counts = Counter(y_test)
    test_total = len(y_test)
    
    for class_id in sorted(test_counts.keys()):
        count = test_counts[class_id]
        percentage = (count / test_total) * 100
        bar_length = int(percentage / 2)
        bar = '█' * bar_length
        print(f"   Class {class_id}: {count:7,} samples ({percentage:5.2f}%) {bar}")
    
    # Check for class imbalance
    print_subheader("Imbalance Analysis")
    train_class_counts = list(train_counts.values())
    if train_class_counts:
        imbalance_ratio = max(train_class_counts) / min(train_class_counts)
        print(f"   Training imbalance ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 10:
            print(f"{RED}⚠️ SEVERE CLASS IMBALANCE detected!{RESET}")
            print(f"   Consider using class weights or resampling techniques.")
        elif imbalance_ratio > 3:
            print(f"{YELLOW}⚠️ Moderate class imbalance detected.{RESET}")
            print(f"   Class weights are recommended.")
        else:
            print(f"{GREEN}✅ Classes are reasonably balanced.{RESET}")
    
    # Verify all expected classes are present
    expected_classes = {0, 1, 2}  # Normal, Attack, (if 3-class)
    train_classes = set(train_counts.keys())
    test_classes = set(test_counts.keys())
    
    if not expected_classes.issubset(train_classes):
        missing = expected_classes - train_classes
        print(f"{RED}⚠️ WARNING: Training set missing classes: {missing}{RESET}")
    else:
        print(f"{GREEN}✅ All expected classes present in training set{RESET}")
    
    if not expected_classes.issubset(test_classes):
        missing = expected_classes - test_classes
        print(f"{YELLOW}⚠️ WARNING: Test set missing classes: {missing}{RESET}")
    else:
        print(f"{GREEN}✅ All expected classes present in test set{RESET}")
    
    # 9. Class Weights Verification
    print_header("⚖️ 8. CLASS WEIGHTS VERIFICATION")
    
    if class_weights:
        print("   Loaded class weights:")
        for class_id in sorted(class_weights.keys()):
            # Handle both string and int keys
            weight = class_weights[class_id]
            print(f"   Class {class_id}: {weight:.4f}")
        
        # Verify weights make sense
        weight_values = list(class_weights.values())
        if all(w > 0 for w in weight_values):
            print(f"{GREEN}✅ All class weights are positive{RESET}")
        else:
            print(f"{RED}⚠️ WARNING: Some class weights are non-positive!{RESET}")
        
        # Check if weights inversely correlate with class frequency
        print("\n   Weight vs Frequency Check:")
        for class_id in sorted(train_counts.keys()):
            freq = train_counts[class_id] / train_total
            weight = class_weights.get(str(class_id), class_weights.get(class_id, 1.0))
            print(f"   Class {class_id}: Frequency={freq:.4f}, Weight={weight:.4f}")
    else:
        print(f"{RED}⚠️ Class weights not loaded{RESET}")
    
    # 10. Scaler Verification
    print_header("🔧 9. SCALER VERIFICATION")
    
    if scaler:
        print(f"   Scaler type: {type(scaler).__name__}")
        
        if hasattr(scaler, 'n_features_in_'):
            print(f"   Features fitted: {scaler.n_features_in_}")
            
            if scaler.n_features_in_ != X_train.shape[2]:
                print(f"{RED}⚠️ WARNING: Scaler features ({scaler.n_features_in_}) != Data features ({X_train.shape[2]}){RESET}")
            else:
                print(f"{GREEN}✅ Scaler feature count matches data{RESET}")
        
        if hasattr(scaler, 'data_min_') and hasattr(scaler, 'data_max_'):
            print(f"   Min values shape: {scaler.data_min_.shape}")
            print(f"   Max values shape: {scaler.data_max_.shape}")
            print(f"   Feature range: [{scaler.data_min_.min():.6f}, {scaler.data_max_.max():.6f}]")
            print(f"{GREEN}✅ Scaler is fitted and ready{RESET}")
        else:
            print(f"{YELLOW}⚠️ Scaler may not be fitted properly{RESET}")
    else:
        print(f"{RED}⚠️ Scaler not loaded{RESET}")
    
    # 11. Sequence Temporal Consistency Check
    print_header("🕐 10. TEMPORAL SEQUENCE CONSISTENCY")
    
    # Sample a few sequences to check for temporal patterns
    if X_train.shape[0] > 0:
        sample_idx = min(100, X_train.shape[0] - 1)
        sample_seq = X_train[sample_idx]
        
        # Check if sequence has variation (not all same values)
        seq_std = sample_seq.std(axis=0)
        static_features = (seq_std == 0).sum()
        
        print(f"   Sample sequence index: {sample_idx}")
        print(f"   Static features (no variation): {static_features}/{X_train.shape[2]}")
        
        if static_features == X_train.shape[2]:
            print(f"{RED}⚠️ WARNING: Sample sequence has no temporal variation!{RESET}")
            print(f"   All features are constant across time steps.")
        elif static_features > X_train.shape[2] * 0.5:
            print(f"{YELLOW}⚠️ WARNING: Over 50% of features are static in sample sequence{RESET}")
        else:
            print(f"{GREEN}✅ Sequences show temporal variation{RESET}")
    
    # 12. Memory Usage Analysis
    print_header("💾 11. MEMORY USAGE ANALYSIS")
    
    def get_size_mb(arr):
        return arr.nbytes / (1024 * 1024)
    
    x_train_size = get_size_mb(X_train)
    y_train_size = get_size_mb(y_train)
    x_test_size = get_size_mb(X_test)
    y_test_size = get_size_mb(y_test)
    total_size = x_train_size + y_train_size + x_test_size + y_test_size
    
    print(f"   X_train: {x_train_size:8.2f} MB")
    print(f"   y_train: {y_train_size:8.2f} MB")
    print(f"   X_test:  {x_test_size:8.2f} MB")
    print(f"   y_test:  {y_test_size:8.2f} MB")
    print(f"   {'─'*40}")
    print(f"   Total:   {total_size:8.2f} MB")
    
    if total_size > 1000:
        print(f"\n{YELLOW}⚠️ Large dataset ({total_size:.0f} MB). Consider batch processing.{RESET}")
    else:
        print(f"\n{GREEN}✅ Dataset size is manageable{RESET}")
    
    # 13. Train/Test Split Ratio
    print_header("📊 12. TRAIN/TEST SPLIT ANALYSIS")
    
    total_samples = X_train.shape[0] + X_test.shape[0]
    train_ratio = (X_train.shape[0] / total_samples) * 100
    test_ratio = (X_test.shape[0] / total_samples) * 100
    
    print(f"   Total sequences: {total_samples:,}")
    print(f"   Train: {X_train.shape[0]:,} ({train_ratio:.1f}%)")
    print(f"   Test:  {X_test.shape[0]:,} ({test_ratio:.1f}%)")
    
    if 75 <= train_ratio <= 85:
        print(f"{GREEN}✅ Split ratio is appropriate (80/20 recommended){RESET}")
    else:
        print(f"{YELLOW}⚠️ Unusual split ratio. Standard is 80/20.{RESET}")
    
    # 14. Final Summary Report
    print_header("📋 13. SUMMARY REPORT")
    
    print(f"\n{BLUE}{'─'*70}")
    print("   DATASET OVERVIEW")
    print(f"{'─'*70}{RESET}")
    print(f"   Total Training Sequences:    {X_train.shape[0]:,}")
    print(f"   Total Test Sequences:        {X_test.shape[0]:,}")
    print(f"   Time Steps per Sequence:     {X_train.shape[1]}")
    print(f"   Features per Time Step:      {X_train.shape[2]}")
    print(f"   Number of Classes:           {len(train_counts)}")
    print(f"   Data Type:                   {X_train.dtype}")
    print(f"   Total Memory Usage:          {total_size:.2f} MB")
    
    print(f"\n{BLUE}{'─'*70}")
    print("   QUALITY CHECKS")
    print(f"{'─'*70}{RESET}")
    
    quality_score = 0
    max_score = 10
    
    # Score each check
    if X_train.ndim == 3 and X_test.ndim == 3:
        quality_score += 1
        print(f"   {GREEN}✅{RESET} Correct tensor dimensions")
    else:
        print(f"   {RED}❌{RESET} Incorrect tensor dimensions")
    
    if X_train.shape[0] == y_train.shape[0] and X_test.shape[0] == y_test.shape[0]:
        quality_score += 1
        print(f"   {GREEN}✅{RESET} Sample counts aligned")
    else:
        print(f"   {RED}❌{RESET} Sample count mismatch")
    
    if X_train.dtype == np.float32 and X_test.dtype == np.float32:
        quality_score += 1
        print(f"   {GREEN}✅{RESET} Optimal data types")
    else:
        print(f"   {YELLOW}⚠️{RESET} Suboptimal data types")
    
    if all_clean:
        quality_score += 1
        print(f"   {GREEN}✅{RESET} No NaN or Inf values")
    else:
        print(f"   {RED}❌{RESET} Contains invalid values")
    
    if -0.01 <= train_min <= 0.01 and 0.99 <= train_max <= 1.01:
        quality_score += 1
        print(f"   {GREEN}✅{RESET} Properly scaled data")
    else:
        print(f"   {YELLOW}⚠️{RESET} Scaling may be incorrect")
    
    if expected_classes.issubset(train_classes):
        quality_score += 1
        print(f"   {GREEN}✅{RESET} All classes present")
    else:
        print(f"   {RED}❌{RESET} Missing classes")
    
    if 75 <= train_ratio <= 85:
        quality_score += 1
        print(f"   {GREEN}✅{RESET} Appropriate split ratio")
    else:
        print(f"   {YELLOW}⚠️{RESET} Unusual split ratio")
    
    if scaler and hasattr(scaler, 'data_min_'):
        quality_score += 1
        print(f"   {GREEN}✅{RESET} Scaler properly fitted")
    else:
        print(f"   {RED}❌{RESET} Scaler issues")
    
    if class_weights and all(w > 0 for w in class_weights.values()):
        quality_score += 1
        print(f"   {GREEN}✅{RESET} Valid class weights")
    else:
        print(f"   {YELLOW}⚠️{RESET} Class weights issues")
    
    if static_features < X_train.shape[2] * 0.5:
        quality_score += 1
        print(f"   {GREEN}✅{RESET} Temporal variation present")
    else:
        print(f"   {YELLOW}⚠️{RESET} Limited temporal variation")
    
    # Final score
    print(f"\n{BLUE}{'─'*70}")
    print(f"   OVERALL QUALITY SCORE: {quality_score}/{max_score}")
    print(f"{'─'*70}{RESET}")
    
    if quality_score == max_score:
        print(f"\n{GREEN}🎉 EXCELLENT! Data is ready for LSTM training!{RESET}")
    elif quality_score >= 8:
        print(f"\n{GREEN}✅ GOOD! Data quality is acceptable with minor issues.{RESET}")
    elif quality_score >= 6:
        print(f"\n{YELLOW}⚠️ FAIR! Some issues detected. Review warnings above.{RESET}")
    else:
        print(f"\n{RED}❌ POOR! Significant issues detected. Fix errors before training.{RESET}")
    
    print_header("🏁 AUDIT COMPLETE")


if __name__ == "__main__":
    check_lstm_data_health()
