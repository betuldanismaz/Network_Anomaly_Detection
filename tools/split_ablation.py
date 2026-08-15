"""
E1-S03: Split Ablation — Global Shuffle vs Per-File Stratified
==============================================================

Compares two splitting strategies on the same models/hyperparameters
to quantify the effect of per-file stratified splitting on rare-class
representation (K1 claim for JNCA paper).

Usage:
    python tools/split_ablation.py                   # Full run
    python tools/split_ablation.py --skip-preprocess  # Skip Phase 1 (data already exists)
    python tools/split_ablation.py --include-dl       # Include LSTM/BiLSTM (slow)
"""

import os
import sys
import gc
import argparse
import json
import time
import warnings
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Project paths ─────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import TOP_FEATURES
from src.features.preprocess_ml_3class import (
    normalize_text,
    load_classes_map,
    list_csv_files,
)

DATA_DIR = os.path.join(ROOT, "data", "original_csv")
PERFILE_DIR = os.path.join(ROOT, "data", "processed_ml")
GLOBAL_DIR = os.path.join(ROOT, "data", "ablation_global")
REPORT_DIR = os.path.join(ROOT, "reports", "split_ablation")

RANDOM_STATE = 42
CLASS_NAMES = {0: "Benign", 1: "Volumetric", 2: "Semantic"}

# ── Hyperparameters (frozen from canonical models) ────────────────────────────
RF_PARAMS = dict(
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=10,
    min_samples_split=20,
    max_features="sqrt",
    criterion="entropy",
    class_weight={0: 1.0, 1: 2.0, 2: 4.0},
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

DT_PARAMS = dict(
    criterion="entropy",
    max_depth=30,
    min_samples_leaf=20,
    min_samples_split=40,
    class_weight={0: 1.0, 1: 2.0, 2: 4.0},
    random_state=RANDOM_STATE,
)

XGB_N_ESTIMATORS = 857
XGB_PARAMS = dict(
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


# =============================================================================
# PHASE 1 — Global-Shuffle Data Generation
# =============================================================================

def generate_global_split():
    """Load all CSVs, concatenate, single global stratified split, scale."""
    print("\n" + "=" * 70)
    print("PHASE 1: Generating Global-Shuffle Split")
    print("=" * 70)

    os.makedirs(GLOBAL_DIR, exist_ok=True)
    label_map = load_classes_map()
    csv_files = list_csv_files(DATA_DIR)
    if not csv_files:
        raise RuntimeError(f"No CSV files found in {DATA_DIR}")

    print(f"Found {len(csv_files)} CSV files\n")

    all_dfs = []
    for i, fpath in enumerate(csv_files, 1):
        fname = os.path.basename(fpath)
        print(f"  [{i}/{len(csv_files)}] Loading {fname} ...", end=" ")

        header_df = pd.read_csv(fpath, nrows=0)
        cols_stripped = [c.strip() for c in header_df.columns]
        col_map = {s: o for s, o in zip(cols_stripped, header_df.columns)}

        keep = [c for c in TOP_FEATURES if c in cols_stripped]
        if "Label" in cols_stripped:
            keep.append("Label")
        keep_original = [col_map[c] for c in keep]

        df = pd.read_csv(fpath, usecols=keep_original)
        df.columns = df.columns.str.strip()

        labels_raw = df["Label"].astype(str).fillna("").map(normalize_text)
        labels_mapped = labels_raw.map(label_map)
        mask = labels_mapped.notna()
        df = df.loc[mask].reset_index(drop=True)
        df["Label"] = labels_mapped.loc[mask].astype(int).values

        feature_cols = [c for c in TOP_FEATURES if c in df.columns]
        df[feature_cols] = (
            df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        )

        print(f"{len(df):,} rows")
        all_dfs.append(df)
        gc.collect()

    df_all = pd.concat(all_dfs, ignore_index=True)
    del all_dfs
    gc.collect()
    print(f"\nTotal rows: {len(df_all):,}")
    print(f"Label distribution: {dict(Counter(df_all['Label'].values))}")

    y = df_all["Label"]
    df_train, df_temp = train_test_split(
        df_all, test_size=0.2, stratify=y, shuffle=True, random_state=RANDOM_STATE
    )
    y_temp = df_temp["Label"]
    df_val, df_test = train_test_split(
        df_temp, test_size=0.5, stratify=y_temp, shuffle=True, random_state=RANDOM_STATE
    )
    del df_all, df_temp
    gc.collect()

    print(f"Split sizes → train: {len(df_train):,} | val: {len(df_val):,} | test: {len(df_test):,}")

    feature_cols = [c for c in TOP_FEATURES if c in df_train.columns]
    scaler = MinMaxScaler()
    df_train[feature_cols] = scaler.fit_transform(
        df_train[feature_cols].values.astype(np.float64)
    )
    df_val[feature_cols] = scaler.transform(
        df_val[feature_cols].values.astype(np.float64)
    )
    df_test[feature_cols] = scaler.transform(
        df_test[feature_cols].values.astype(np.float64)
    )

    df_train.to_csv(os.path.join(GLOBAL_DIR, "train.csv"), index=False)
    df_val.to_csv(os.path.join(GLOBAL_DIR, "val.csv"), index=False)
    df_test.to_csv(os.path.join(GLOBAL_DIR, "test.csv"), index=False)

    print(f"Saved to {GLOBAL_DIR}")
    print("Phase 1 complete.\n")


# =============================================================================
# PHASE 2 — Model Training & Evaluation
# =============================================================================

def load_split(data_dir):
    """Load train/val/test CSVs, return (X_train, y_train, X_val, y_val, X_test, y_test)."""
    df_train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    df_val = pd.read_csv(os.path.join(data_dir, "val.csv"))
    df_test = pd.read_csv(os.path.join(data_dir, "test.csv"))

    feature_cols = [c for c in TOP_FEATURES if c in df_train.columns]

    X_train = df_train[feature_cols].values
    y_train = df_train["Label"].values.astype(int)
    X_val = df_val[feature_cols].values
    y_val = df_val["Label"].values.astype(int)
    X_test = df_test[feature_cols].values
    y_test = df_test["Label"].values.astype(int)

    return X_train, y_train, X_val, y_val, X_test, y_test


def compute_metrics(y_true, y_pred):
    """Return dict of macro + per-class metrics."""
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    for cls_id, cls_name in CLASS_NAMES.items():
        p = precision_score(y_true, y_pred, labels=[cls_id], average="macro", zero_division=0)
        r = recall_score(y_true, y_pred, labels=[cls_id], average="macro", zero_division=0)
        f = f1_score(y_true, y_pred, labels=[cls_id], average="macro", zero_division=0)
        result[f"{cls_name}_precision"] = p
        result[f"{cls_name}_recall"] = r
        result[f"{cls_name}_f1"] = f
    return result


def train_rf(X_train, y_train):
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    return model


def train_dt(X_train, y_train):
    model = DecisionTreeClassifier(**DT_PARAMS)
    model.fit(X_train, y_train)
    return model


def train_xgb(X_train, y_train):
    from xgboost import XGBClassifier
    from sklearn.utils.class_weight import compute_sample_weight

    sample_weights = compute_sample_weight("balanced", y_train)

    model = XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        tree_method="hist",
        **XGB_PARAMS,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model


MODEL_REGISTRY = {
    "RF": train_rf,
    "DT": train_dt,
    "XGBoost": train_xgb,
}


def run_all_models(split_name, data_dir, include_dl=False):
    """Train all models on a given split, return list of result dicts."""
    print(f"\n{'─' * 70}")
    print(f"Training on: {split_name} ({data_dir})")
    print(f"{'─' * 70}")

    X_train, y_train, X_val, y_val, X_test, y_test = load_split(data_dir)
    print(f"  Train: {len(y_train):,} | Val: {len(y_val):,} | Test: {len(y_test):,}")
    print(f"  Train distribution: {dict(Counter(y_train))}")
    print(f"  Test  distribution: {dict(Counter(y_test))}")

    results = []
    for model_name, train_fn in MODEL_REGISTRY.items():
        print(f"\n  [{model_name}] Training ...", end=" ", flush=True)
        t0 = time.time()
        model = train_fn(X_train, y_train)
        train_time = time.time() - t0
        print(f"done ({train_time:.1f}s)")

        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        metrics["model"] = model_name
        metrics["split_method"] = split_name
        metrics["train_time_s"] = round(train_time, 1)
        results.append(metrics)

        print(f"    Macro-F1: {metrics['macro_f1']:.4f} | "
              f"Semantic-F1: {metrics['Semantic_f1']:.4f}")

        del model
        gc.collect()

    del X_train, y_train, X_val, y_val, X_test, y_test
    gc.collect()
    return results


# =============================================================================
# PHASE 3 — Class Distribution Analysis
# =============================================================================

def build_class_distribution():
    """Compare class counts across splits for both methods."""
    rows = []
    for method, data_dir in [("per_file", PERFILE_DIR), ("global", GLOBAL_DIR)]:
        for split_name in ["train", "val", "test"]:
            fpath = os.path.join(data_dir, f"{split_name}.csv")
            df = pd.read_csv(fpath, usecols=["Label"])
            counts = Counter(df["Label"].values)
            total = len(df)
            for cls_id, cls_name in CLASS_NAMES.items():
                rows.append({
                    "split_method": method,
                    "split": split_name,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "count": counts.get(cls_id, 0),
                    "pct": round(100 * counts.get(cls_id, 0) / total, 2) if total > 0 else 0,
                })
            del df
    return pd.DataFrame(rows)


# =============================================================================
# PHASE 4 — Report Generation
# =============================================================================

def generate_reports(results_df, dist_df):
    """Write comparison_table.csv, class_distribution.csv, split_ablation_report.txt."""
    os.makedirs(REPORT_DIR, exist_ok=True)

    # 1. comparison_table.csv
    table_path = os.path.join(REPORT_DIR, "comparison_table.csv")
    col_order = [
        "model", "split_method",
        "macro_f1", "macro_precision", "macro_recall", "accuracy",
        "Benign_f1", "Benign_precision", "Benign_recall",
        "Volumetric_f1", "Volumetric_precision", "Volumetric_recall",
        "Semantic_f1", "Semantic_precision", "Semantic_recall",
        "train_time_s",
    ]
    results_df[col_order].to_csv(table_path, index=False, float_format="%.4f")
    print(f"\n  Saved: {table_path}")

    # 2. class_distribution.csv
    dist_path = os.path.join(REPORT_DIR, "class_distribution.csv")
    dist_df.to_csv(dist_path, index=False)
    print(f"  Saved: {dist_path}")

    # 3. split_ablation_report.txt
    report_path = os.path.join(REPORT_DIR, "split_ablation_report.txt")
    lines = []
    lines.append("=" * 70)
    lines.append("E1-S03: Split Ablation Report")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Hypothesis: Per-file stratified split preserves rare-class")
    lines.append("representation (especially Semantic/Infiltration), yielding")
    lines.append("higher Semantic F1 than global shuffle.")
    lines.append("")

    # Per-model delta analysis
    lines.append("-" * 70)
    lines.append("MODEL-WISE COMPARISON (per_file - global)")
    lines.append("-" * 70)
    lines.append(f"{'Model':<10} {'Macro-F1 (pf)':>14} {'Macro-F1 (gl)':>14} {'Delta':>8} "
                 f"{'Sem-F1 (pf)':>12} {'Sem-F1 (gl)':>12} {'Delta':>8}")
    lines.append("-" * 70)

    models = results_df["model"].unique()
    macro_deltas = []
    semantic_deltas = []

    for m in models:
        pf = results_df[(results_df["model"] == m) & (results_df["split_method"] == "per_file")]
        gl = results_df[(results_df["model"] == m) & (results_df["split_method"] == "global")]

        if pf.empty or gl.empty:
            continue

        pf_macro = pf["macro_f1"].values[0]
        gl_macro = gl["macro_f1"].values[0]
        pf_sem = pf["Semantic_f1"].values[0]
        gl_sem = gl["Semantic_f1"].values[0]

        d_macro = pf_macro - gl_macro
        d_sem = pf_sem - gl_sem
        macro_deltas.append(d_macro)
        semantic_deltas.append(d_sem)

        lines.append(
            f"{m:<10} {pf_macro:>14.4f} {gl_macro:>14.4f} {d_macro:>+8.4f} "
            f"{pf_sem:>12.4f} {gl_sem:>12.4f} {d_sem:>+8.4f}"
        )

    lines.append("-" * 70)

    if macro_deltas:
        avg_macro_d = np.mean(macro_deltas)
        avg_sem_d = np.mean(semantic_deltas)
        lines.append(f"{'AVERAGE':<10} {'':>14} {'':>14} {avg_macro_d:>+8.4f} "
                     f"{'':>12} {'':>12} {avg_sem_d:>+8.4f}")
    lines.append("")

    # Class distribution summary
    lines.append("-" * 70)
    lines.append("CLASS DISTRIBUTION IN TEST SETS")
    lines.append("-" * 70)

    for method in ["per_file", "global"]:
        test_dist = dist_df[(dist_df["split_method"] == method) & (dist_df["split"] == "test")]
        lines.append(f"\n  {method}:")
        for _, row in test_dist.iterrows():
            lines.append(f"    {row['class_name']:<12} {row['count']:>8,} ({row['pct']:.2f}%)")

    lines.append("")

    # K1 conclusion
    lines.append("=" * 70)
    lines.append("K1 CLAIM ASSESSMENT")
    lines.append("=" * 70)
    lines.append("")

    if semantic_deltas:
        all_positive = all(d > 0 for d in semantic_deltas)
        avg_sem_d = np.mean(semantic_deltas)

        if all_positive:
            lines.append("RESULT: SUPPORTED")
            lines.append(f"Per-file stratified split yields higher Semantic F1 across")
            lines.append(f"all {len(semantic_deltas)} models (avg delta: {avg_sem_d:+.4f}).")
            lines.append(f"This confirms that per-file stratification preserves rare-class")
            lines.append(f"representation, supporting claim K1.")
        elif avg_sem_d > 0:
            lines.append("RESULT: PARTIALLY SUPPORTED")
            lines.append(f"Per-file stratified split yields higher average Semantic F1")
            lines.append(f"(avg delta: {avg_sem_d:+.4f}), but not all models show improvement.")
        else:
            lines.append("RESULT: NOT SUPPORTED")
            lines.append(f"Global shuffle shows comparable or better Semantic F1")
            lines.append(f"(avg delta: {avg_sem_d:+.4f}). K1 claim needs revision.")
    else:
        lines.append("RESULT: INSUFFICIENT DATA")

    lines.append("")
    lines.append("=" * 70)

    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"  Saved: {report_path}")

    # Print to console
    print("\n" + report_text)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="E1-S03: Split Ablation Study")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip Phase 1 (global-shuffle data already exists)")
    parser.add_argument("--include-dl", action="store_true",
                        help="Include LSTM/BiLSTM models (slow)")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("E1-S03: SPLIT ABLATION STUDY")
    print("Global Shuffle vs Per-File Stratified")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Random state: {RANDOM_STATE}")
    print(f"Features: {len(TOP_FEATURES)}")
    print(f"Models: {list(MODEL_REGISTRY.keys())}")

    t_start = time.time()

    # ── Phase 1: Data generation ──────────────────────────────────────────
    if args.skip_preprocess:
        if not os.path.exists(os.path.join(GLOBAL_DIR, "train.csv")):
            print("\nWARN: --skip-preprocess but no global data found. Running Phase 1 anyway.")
            generate_global_split()
        else:
            print("\nSkipping Phase 1 (--skip-preprocess)")
    else:
        generate_global_split()

    # ── Phase 2: Train & evaluate ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 2: Model Training & Evaluation")
    print("=" * 70)

    all_results = []
    all_results.extend(run_all_models("per_file", PERFILE_DIR))
    all_results.extend(run_all_models("global", GLOBAL_DIR))

    results_df = pd.DataFrame(all_results)

    # ── Phase 3: Class distribution ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 3: Class Distribution Analysis")
    print("=" * 70)

    dist_df = build_class_distribution()

    # ── Phase 4: Reports ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 4: Report Generation")
    print("=" * 70)

    generate_reports(results_df, dist_df)

    elapsed = time.time() - t_start
    print(f"\nTotal elapsed: {elapsed / 60:.1f} minutes")
    print("=" * 70)
    print("E1-S03 COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
