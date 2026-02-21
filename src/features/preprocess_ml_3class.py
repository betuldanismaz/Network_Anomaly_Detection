"""
3-Class ML Data Preprocessing Script
=====================================

Produces train.csv, val.csv, test.csv for a 3-class classifier
(0 = Benign, 1 = Volumetric, 2 = Semantic) from raw CICIDS2017 CSVs.

Design: Per-file stratified splitting → aggregate → leakage-free MinMaxScaler.

WHY per-file stratified split?
  Each of the 8 CICIDS2017 files covers different attack types and time periods.
  Some classes are extremely rare in certain files (e.g. Thursday has only 36
  Infiltration rows). Splitting each file independently with stratification
  guarantees that even these rare classes appear in train, val, AND test sets.
  If we concatenated all raw data first and then split, the 36 Infiltration rows
  could end up entirely in one split, leaving the others without that class.

Output convention (matches binary pipeline):
  - Each CSV has TOP_FEATURES columns + "Label" column
  - Downstream training scripts do: X = df.drop('Label', axis=1), y = df['Label']

"""

import os
import sys
import gc
import glob
import json
import unicodedata
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================

# Project root directory (script lives in src/features/)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Add project root to Python path so `from src.config import ...` works
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Directory paths
DATA_DIR   = os.path.join(ROOT, 'data', 'original_csv')
OUT_DIR    = os.path.join(ROOT, 'data', 'processed_ml')
MODELS_DIR = os.path.join(ROOT, 'models')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler_ml_3class.pkl')

# Resolve classes_map.json — same resolution logic as preprocess_lstm.py
# so the script works regardless of where the file has been placed.
_candidate_class_map_paths = [
    os.getenv('CLASSES_MAP_PATH'),
    os.path.join(ROOT, 'src', 'utils', 'classes_map.json'),
    os.path.join(ROOT, 'reports', 'data', 'classes_map.json'),
    os.path.join(ROOT, 'data', 'classes_map.json'),
    os.path.join(ROOT, 'classes_map.json'),
]
CLASS_MAP_PATH: Optional[str] = None
for _p in _candidate_class_map_paths:
    if _p and os.path.exists(_p):
        CLASS_MAP_PATH = _p
        break
if CLASS_MAP_PATH is None:
    # Last-resort default (will fail at load time with a clear message)
    CLASS_MAP_PATH = os.path.join(ROOT, 'src', 'utils', 'classes_map.json')

# Import the curated top-20 features shared across ML pipelines
from src.config import TOP_FEATURES

# Splitting parameters
RANDOM_STATE = 42

# Create output directories
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def normalize_text(x: Any) -> str:
    """
    Normalize a label string: strip BOM, collapse whitespace, NFKC-normalize.
    This handles the quirky CICIDS2017 label formatting (e.g. extra spaces in
    'Web Attack  Brute Force') so labels match the keys in classes_map.json.
    """
    if x is None:
        return ''
    s = str(x)
    s = unicodedata.normalize('NFKC', s)
    # Remove BOM, non-breaking space, replacement character
    s = s.replace('\ufeff', '').replace('\u00A0', ' ').replace('\uFFFD', ' ')
    s = ' '.join(s.split())
    return s.strip()


def load_classes_map() -> Dict[str, int]:
    """Load classes_map.json and normalize all keys for reliable matching."""
    with open(CLASS_MAP_PATH, 'r', encoding='utf-8') as f:
        raw_map = json.load(f)
    return {normalize_text(k): v for k, v in raw_map.items()}


def list_csv_files(data_dir: str) -> List[str]:
    """Return sorted list of CSV files in the given directory."""
    return sorted(glob.glob(os.path.join(data_dir, '*.csv')))


def get_memory_mb() -> float:
    """Return current process RSS in MB (0.0 if psutil unavailable)."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


# =============================================================================
# PER-FILE PROCESSING
# =============================================================================

def process_single_file(
    file_path: str,
    features: List[str],
    label_map: Dict[str, int],
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Load one CICIDS2017 CSV, map labels to 3 classes, and perform a per-file
    stratified 3-way split (80/10/10).

    Returns (df_train, df_val, df_test) — each containing TOP_FEATURES + Label.
    Returns None if the file cannot be processed.
    """
    filename = os.path.basename(file_path)

    try:
        # ── A. LOAD & MAP ──────────────────────────────────────────────────
        # Read header first to resolve column-name whitespace issues
        header_df = pd.read_csv(file_path, nrows=0)
        cols_original = header_df.columns.tolist()
        cols_stripped = [c.strip() for c in cols_original]
        col_map = {stripped: original for stripped, original in zip(cols_stripped, cols_original)}

        # Only read the columns we actually need (saves RAM on wide CSVs)
        keep_stripped = [c for c in features if c in cols_stripped]
        if 'Label' in cols_stripped:
            keep_stripped.append('Label')

        if not keep_stripped or 'Label' not in keep_stripped:
            print(f"  [WARN] [{filename}] Missing required columns (need Label + features), skipping")
            return None

        keep_original = [col_map[c] for c in keep_stripped]
        df = pd.read_csv(file_path, usecols=keep_original)
        df.columns = df.columns.str.strip()

        print(f"  📂 [{filename}] Loaded {len(df):,} rows")

        # Map raw label strings → 0/1/2 integers via classes_map.json
        labels_raw = df['Label'].astype(str).fillna('').map(normalize_text)
        labels_mapped = labels_raw.map(label_map)
        mask_valid = labels_mapped.notna()

        n_dropped = (~mask_valid).sum()
        if n_dropped > 0:
            print(f"     [WARN] [{filename}] Dropped {n_dropped:,} rows with unmapped labels")

        if mask_valid.sum() == 0:
            print(f"  [WARN] [{filename}] No valid labels after mapping, skipping")
            return None

        df = df.loc[mask_valid].reset_index(drop=True)
        df['Label'] = labels_mapped.loc[mask_valid].astype(int).values

        # Replace inf with NaN and then fill NaN with 0 for numeric stability
        feature_cols = [c for c in features if c in df.columns]
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

        label_dist = dict(Counter(df['Label'].values))
        print(f"     Raw label distribution: {label_dist}")

    except Exception as e:
        print(f"  [ERROR] [{filename}] Failed to load/map: {e}")
        return None

    # ── B. PER-FILE STRATIFIED 3-WAY SPLIT ─────────────────────────────
    # First split: 80% train, 20% temp
    # Second split: temp → 50/50 = 10% val, 10% test (of the original)
    #
    # We use stratification so that even rare classes (e.g. 36 Infiltration
    # rows on Thursday) are proportionally represented in every split.
    y = df['Label']

    try:
        # Check minimum class count for stratification
        min_class_count = y.value_counts().min()

        if min_class_count >= 2:
            # Enough samples to stratify both splits
            df_train, df_temp = train_test_split(
                df, test_size=0.2, stratify=y,
                shuffle=True, random_state=RANDOM_STATE
            )
            y_temp = df_temp['Label']

            # Second split: need at least 2 per class in temp for stratification
            min_temp_count = y_temp.value_counts().min()
            if min_temp_count >= 2:
                df_val, df_test = train_test_split(
                    df_temp, test_size=0.5, stratify=y_temp,
                    shuffle=True, random_state=RANDOM_STATE
                )
            else:
                print(f"     [WARN] [{filename}] Too few samples in temp for stratified val/test, falling back")
                df_val, df_test = train_test_split(
                    df_temp, test_size=0.5,
                    shuffle=True, random_state=RANDOM_STATE
                )
        else:
            # Too few samples for any stratification
            print(f"     [WARN] [{filename}] Class with only {min_class_count} sample(s), non-stratified split")
            df_train, df_temp = train_test_split(
                df, test_size=0.2,
                shuffle=True, random_state=RANDOM_STATE
            )
            df_val, df_test = train_test_split(
                df_temp, test_size=0.5,
                shuffle=True, random_state=RANDOM_STATE
            )
    except Exception as e:
        # Absolute fallback: non-stratified splits
        print(f"     [WARN] [{filename}] Stratified split failed ({e}), using random split")
        df_train, df_temp = train_test_split(
            df, test_size=0.2, shuffle=True, random_state=RANDOM_STATE
        )
        df_val, df_test = train_test_split(
            df_temp, test_size=0.5, shuffle=True, random_state=RANDOM_STATE
        )

    print(f"     Split → train: {len(df_train):,} | val: {len(df_val):,} | test: {len(df_test):,}")

    return df_train, df_val, df_test


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("3-CLASS ML DATA PREPROCESSING")
    print("Strategy: Per-File Stratified Split → Aggregate → Leakage-Free Scaling")
    print("Classes: 0=Benign  |  1=Volumetric  |  2=Semantic")
    print("=" * 70 + "\n")

    # ── Load label mapping ──────────────────────────────────────────────
    print(f"📋 Loading classes map from: {CLASS_MAP_PATH}")
    label_map = load_classes_map()
    unique_classes = sorted(set(label_map.values()))
    print(f"   Mapped classes: {unique_classes}")
    print(f"   Entries: {len(label_map)}\n")

    # ── Discover CSV files ──────────────────────────────────────────────
    csv_files = list_csv_files(DATA_DIR)
    if not csv_files:
        raise RuntimeError(f"No CSV files found in {DATA_DIR}")
    print(f"📁 Found {len(csv_files)} CSV files in {DATA_DIR}\n")

    # ── Per-file iterative processing ───────────────────────────────────
    # We collect split DataFrames in separate lists to avoid any cross-file
    # contamination. Concatenation happens only after all files are processed.
    train_parts: List[pd.DataFrame] = []
    val_parts:   List[pd.DataFrame] = []
    test_parts:  List[pd.DataFrame] = []

    print("-" * 70)
    print("PROCESSING FILES (Per-File Stratified Split)")
    print("-" * 70)

    for i, fpath in enumerate(csv_files, 1):
        print(f"\n[{i}/{len(csv_files)}] {os.path.basename(fpath)}")
        print(f"   Memory: {get_memory_mb():.1f} MB")

        result = process_single_file(fpath, TOP_FEATURES, label_map)

        if result is not None:
            df_train, df_val, df_test = result

            # Step C: Collect
            train_parts.append(df_train)
            val_parts.append(df_val)
            test_parts.append(df_test)

            # Step D: Free per-file memory immediately
            del df_train, df_val, df_test
        else:
            print(f"   ⚠️  File skipped")

        gc.collect()

    print("\n" + "-" * 70)
    print("FILE PROCESSING COMPLETE")
    print("-" * 70)

    if not train_parts:
        raise RuntimeError("No valid data collected from any file!")

    # ── 3. AGGREGATION & LEAKAGE-FREE SCALING ──────────────────────────
    print("\n📊 Aggregating all splits...")

    df_train_all = pd.concat(train_parts, ignore_index=True)
    df_val_all   = pd.concat(val_parts,   ignore_index=True)
    df_test_all  = pd.concat(test_parts,  ignore_index=True)

    # Free the part lists
    del train_parts, val_parts, test_parts
    gc.collect()

    print(f"   Aggregated shapes: train={df_train_all.shape}, val={df_val_all.shape}, test={df_test_all.shape}")
    print(f"   Memory: {get_memory_mb():.1f} MB")

    # Separate features and labels
    feature_cols = [c for c in TOP_FEATURES if c in df_train_all.columns]
    X_train = df_train_all[feature_cols].values.astype(np.float64)
    y_train = df_train_all['Label'].values.astype(int)
    X_val   = df_val_all[feature_cols].values.astype(np.float64)
    y_val   = df_val_all['Label'].values.astype(int)
    X_test  = df_test_all[feature_cols].values.astype(np.float64)
    y_test  = df_test_all['Label'].values.astype(int)

    # Free the raw DataFrames (we'll rebuild from scaled arrays)
    del df_train_all, df_val_all, df_test_all
    gc.collect()

    # Fit scaler on TRAIN ONLY → transform all three (leakage-free)
    print("\n⚖️  Fitting MinMaxScaler on training data only (leakage-free)...")
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # Save scaler
    joblib.dump(scaler, SCALER_PATH)
    print(f"   Scaler saved → {SCALER_PATH}")

    # Rebuild DataFrames with Label column (downstream expects df.drop('Label', axis=1))
    df_train_final = pd.DataFrame(X_train, columns=feature_cols)
    df_train_final['Label'] = y_train

    df_val_final = pd.DataFrame(X_val, columns=feature_cols)
    df_val_final['Label'] = y_val

    df_test_final = pd.DataFrame(X_test, columns=feature_cols)
    df_test_final['Label'] = y_test

    # Free numpy arrays
    del X_train, X_val, X_test, y_train, y_val, y_test
    gc.collect()

    # ── 4. OUTPUT ───────────────────────────────────────────────────────
    print("\n💾 Saving output CSVs...")

    train_path = os.path.join(OUT_DIR, 'train.csv')
    val_path   = os.path.join(OUT_DIR, 'val.csv')
    test_path  = os.path.join(OUT_DIR, 'test.csv')

    df_train_final.to_csv(train_path, index=False)
    df_val_final.to_csv(val_path, index=False)
    df_test_final.to_csv(test_path, index=False)

    print(f"   ✅ {train_path}")
    print(f"   ✅ {val_path}")
    print(f"   ✅ {test_path}")

    # ── 5. VALIDATION OUTPUT ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    for name, df in [("TRAIN", df_train_final), ("VAL", df_val_final), ("TEST", df_test_final)]:
        counts = dict(Counter(df['Label'].values))
        # Ensure all 3 classes are shown (even if 0 count)
        counts_display = {c: counts.get(c, 0) for c in [0, 1, 2]}
        print(f"{name:5s} - Shape: {df.shape} | Label counts: {counts_display}")

    # Check that all 3 classes are present in every split
    all_present = True
    for name, df in [("TRAIN", df_train_final), ("VAL", df_val_final), ("TEST", df_test_final)]:
        present = set(df['Label'].unique())
        missing = {0, 1, 2} - present
        if missing:
            print(f"⚠️  {name} is missing classes: {missing}")
            all_present = False

    if all_present:
        print("✅ All 3 classes present in all splits")
    else:
        print("⚠️  WARNING: Not all 3 classes are present in every split!")

    print(f"\n📁 Output directory: {OUT_DIR}")
    print(f"⚖️  Scaler: {SCALER_PATH}")
    print(f"🧠 Memory: {get_memory_mb():.1f} MB")
    print("=" * 70)
    print("✅ 3-CLASS ML PREPROCESSING COMPLETE")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
