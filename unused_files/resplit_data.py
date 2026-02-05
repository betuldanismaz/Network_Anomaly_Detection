#!/usr/bin/env python3
"""
Stratified Data Resplit Utility
Fixes class imbalance in test set caused by time-based split.

This script:
1. Loads existing X_train, X_test, y_train, y_test from processed_lstm/
2. Merges them into X_total and y_total
3. Performs stratified split to ensure all classes in both train/test
4. Saves the new splits back to the same location
"""

import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

# --- CONFIG ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(ROOT, 'data', 'processed_lstm')

# File paths
X_TRAIN_PATH = os.path.join(DATA_DIR, 'X_train.npy')
Y_TRAIN_PATH = os.path.join(DATA_DIR, 'y_train.npy')
X_TEST_PATH = os.path.join(DATA_DIR, 'X_test.npy')
Y_TEST_PATH = os.path.join(DATA_DIR, 'y_test.npy')

# Split parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data():
    """Load existing train and test data."""
    print("=" * 60)
    print("📂 LOADING EXISTING DATA")
    print("=" * 60)
    
    # Check files exist
    for path in [X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH, Y_TEST_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ File not found: {path}")
    
    # Load data
    X_train = np.load(X_TRAIN_PATH)
    y_train = np.load(Y_TRAIN_PATH)
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)
    
    print(f"✅ X_train shape: {X_train.shape}")
    print(f"✅ y_train shape: {y_train.shape}")
    print(f"✅ X_test shape: {X_test.shape}")
    print(f"✅ y_test shape: {y_test.shape}")
    
    # Show current class distribution
    print("\n📊 CURRENT CLASS DISTRIBUTION (Before Resplit):")
    print("-" * 40)
    
    train_dist = Counter(y_train.tolist())
    test_dist = Counter(y_test.tolist())
    
    print("Training set:")
    for cls in sorted(train_dist.keys()):
        print(f"  Class {cls}: {train_dist[cls]:,} samples")
    
    print("Test set:")
    for cls in sorted(test_dist.keys()):
        print(f"  Class {cls}: {test_dist[cls]:,} samples")
    
    # Check for missing classes
    all_classes = set(train_dist.keys()) | set(test_dist.keys())
    missing_in_test = set(range(3)) - set(test_dist.keys())
    if missing_in_test:
        print(f"\n⚠️  PROBLEM: Classes {missing_in_test} missing from test set!")
    
    return X_train, y_train, X_test, y_test


def merge_data(X_train, y_train, X_test, y_test):
    """Merge train and test data back together."""
    print("\n" + "=" * 60)
    print("🔗 MERGING DATA")
    print("=" * 60)
    
    X_total = np.concatenate([X_train, X_test], axis=0)
    y_total = np.concatenate([y_train, y_test], axis=0)
    
    print(f"✅ X_total shape: {X_total.shape}")
    print(f"✅ y_total shape: {y_total.shape}")
    
    total_dist = Counter(y_total.tolist())
    print("\n📊 Total class distribution:")
    for cls in sorted(total_dist.keys()):
        pct = 100 * total_dist[cls] / len(y_total)
        print(f"  Class {cls}: {total_dist[cls]:,} samples ({pct:.2f}%)")
    
    return X_total, y_total


def stratified_split(X_total, y_total):
    """Perform stratified train/test split."""
    print("\n" + "=" * 60)
    print("✂️  STRATIFIED SPLIT")
    print("=" * 60)
    
    print(f"   test_size: {TEST_SIZE}")
    print(f"   random_state: {RANDOM_STATE}")
    print(f"   stratify: y_total (ensures class balance)")
    
    X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
        X_total,
        y_total,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_total  # KEY: Ensures proportional class distribution
    )
    
    print(f"\n✅ New X_train shape: {X_train_new.shape}")
    print(f"✅ New y_train shape: {y_train_new.shape}")
    print(f"✅ New X_test shape: {X_test_new.shape}")
    print(f"✅ New y_test shape: {y_test_new.shape}")
    
    return X_train_new, y_train_new, X_test_new, y_test_new


def save_data(X_train, y_train, X_test, y_test):
    """Save the new stratified splits."""
    print("\n" + "=" * 60)
    print("💾 SAVING NEW SPLITS")
    print("=" * 60)
    
    # Backup info
    print("⚠️  Overwriting existing files...")
    
    np.save(X_TRAIN_PATH, X_train.astype(np.float32))
    np.save(Y_TRAIN_PATH, y_train.astype(np.int32))
    np.save(X_TEST_PATH, X_test.astype(np.float32))
    np.save(Y_TEST_PATH, y_test.astype(np.int32))
    
    print(f"✅ Saved: {X_TRAIN_PATH}")
    print(f"✅ Saved: {Y_TRAIN_PATH}")
    print(f"✅ Saved: {X_TEST_PATH}")
    print(f"✅ Saved: {Y_TEST_PATH}")


def verify_split(y_train, y_test):
    """Verify that all classes exist in both splits."""
    print("\n" + "=" * 60)
    print("✅ VERIFICATION: NEW CLASS DISTRIBUTION")
    print("=" * 60)
    
    train_dist = Counter(y_train.tolist())
    test_dist = Counter(y_test.tolist())
    
    class_names = {0: 'Benign', 1: 'Volumetric', 2: 'Semantic'}
    
    print("\n📊 New Training Set:")
    print("-" * 50)
    for cls in sorted(train_dist.keys()):
        pct = 100 * train_dist[cls] / len(y_train)
        print(f"  Class {cls} ({class_names.get(cls, 'Unknown')}): {train_dist[cls]:,} samples ({pct:.2f}%)")
    
    print(f"\n  Total: {len(y_train):,} samples")
    
    print("\n📊 New Test Set:")
    print("-" * 50)
    for cls in sorted(test_dist.keys()):
        pct = 100 * test_dist[cls] / len(y_test)
        print(f"  Class {cls} ({class_names.get(cls, 'Unknown')}): {test_dist[cls]:,} samples ({pct:.2f}%)")
    
    print(f"\n  Total: {len(y_test):,} samples")
    
    # Final check
    print("\n" + "=" * 60)
    all_classes_in_train = set(train_dist.keys())
    all_classes_in_test = set(test_dist.keys())
    
    if all_classes_in_train == all_classes_in_test == {0, 1, 2}:
        print("🎉 SUCCESS: All 3 classes present in both train and test sets!")
    else:
        missing_train = {0, 1, 2} - all_classes_in_train
        missing_test = {0, 1, 2} - all_classes_in_test
        if missing_train:
            print(f"⚠️  WARNING: Classes {missing_train} missing from train set")
        if missing_test:
            print(f"⚠️  WARNING: Classes {missing_test} missing from test set")
    
    print("=" * 60)


def main():
    """Main execution flow."""
    print("\n" + "=" * 70)
    print("🔄 STRATIFIED DATA RESPLIT UTILITY")
    print("   Fixing class imbalance in test set")
    print("=" * 70)
    
    # 1. Load existing data
    X_train, y_train, X_test, y_test = load_data()
    
    # 2. Merge back together
    X_total, y_total = merge_data(X_train, y_train, X_test, y_test)
    
    # 3. Stratified split
    X_train_new, y_train_new, X_test_new, y_test_new = stratified_split(X_total, y_total)
    
    # 4. Save new splits
    save_data(X_train_new, y_train_new, X_test_new, y_test_new)
    
    # 5. Verify
    verify_split(y_train_new, y_test_new)
    
    print("\n✅ RESPLIT COMPLETE!")
    print("   You can now retrain the BiLSTM model with balanced test data.")
    print("   Run: python src/models/train_bilstm.py")


if __name__ == '__main__':
    main()
