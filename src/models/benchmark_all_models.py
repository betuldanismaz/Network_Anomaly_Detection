"""
Unified Inference Latency Benchmark — All 5 Models
====================================================
Measures latency and throughput for XGBoost, Random Forest, Decision Tree,
LSTM, and BiLSTM using the same 10,000-sample protocol used in train_xgboost.py.

Results are saved to: reports/latency_benchmark.json
Run:
    python src/models/benchmark_all_models.py

Author: NIDS Project
Date:   2026-03
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # suppress CUDA/oneDNN noise
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

MODELS_DIR  = os.path.join(ROOT, "models")
REPORTS_DIR = os.path.join(ROOT, "reports")
DATA_ML     = os.path.join(ROOT, "data", "processed_ml",       "test.csv")
DATA_LSTM   = os.path.join(ROOT, "data", "processed_lstm",      "X_test.npy")

SAMPLE_SIZE = 10_000
WARMUP_SIZE = 100
LABEL_COL   = "Label"

os.makedirs(REPORTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def benchmark(fn, label=""):
    """Run fn() twice: once as warm-up, once timed. Return (latency_ms, throughput)."""
    fn()                        # warm-up
    start = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - start
    latency_ms  = (elapsed / SAMPLE_SIZE) * 1000
    throughput  = SAMPLE_SIZE / elapsed
    print(f"   ✅ {label:<20} latency={latency_ms:.4f} ms/sample  "
          f"throughput={throughput:,.0f} samples/sec")
    return latency_ms, throughput


def load_ml_data():
    """Load tabular (ML) test data — first SAMPLE_SIZE rows, drop label."""
    print(f"\n📂 Loading ML test data from: {DATA_ML}")
    df = pd.read_csv(DATA_ML, nrows=SAMPLE_SIZE + 10)
    if LABEL_COL in df.columns:
        df = df.drop(columns=[LABEL_COL])
    X = df.iloc[:SAMPLE_SIZE]
    print(f"   ✅ Shape: {X.shape}")
    return X


def load_dl_data():
    """Load sequential (DL) test data — first SAMPLE_SIZE sequences."""
    print(f"\n📂 Loading DL test data from: {DATA_LSTM}")
    X = np.load(DATA_LSTM, mmap_mode="r")[:SAMPLE_SIZE]
    print(f"   ✅ Shape: {X.shape}")
    return X


# ─────────────────────────────────────────────────────────────────────────────
# Model benchmarks
# ─────────────────────────────────────────────────────────────────────────────

def bench_xgboost(X_ml):
    print("\n" + "═" * 60)
    print("⚡ XGBoost (GPU)")
    path = os.path.join(MODELS_DIR, "xgb_3class_model.pkl")
    if not os.path.exists(path):
        print(f"   ❌ Model not found: {path}")
        return None
    model = joblib.load(path)
    lat, thr = benchmark(lambda: model.predict_proba(X_ml), "XGBoost")
    return {"model": "XGBoost (GPU)", "latency_ms": lat, "throughput": thr,
            "sample_size": SAMPLE_SIZE, "gpu": "CUDA"}


def bench_random_forest(X_ml):
    print("\n" + "═" * 60)
    print("⚡ Random Forest (CPU)")
    path = os.path.join(MODELS_DIR, "rf_3class_model.pkl")
    if not os.path.exists(path):
        print(f"   ❌ Model not found: {path}")
        return None
    model = joblib.load(path)
    lat, thr = benchmark(lambda: model.predict_proba(X_ml), "Random Forest")
    return {"model": "Random Forest", "latency_ms": lat, "throughput": thr,
            "sample_size": SAMPLE_SIZE, "gpu": "CPU"}


def bench_decision_tree(X_ml):
    print("\n" + "═" * 60)
    print("⚡ Decision Tree (CPU)")
    path = os.path.join(MODELS_DIR, "dt_3class_model.pkl")
    if not os.path.exists(path):
        print(f"   ❌ Model not found: {path}")
        return None
    model = joblib.load(path)
    lat, thr = benchmark(lambda: model.predict_proba(X_ml), "Decision Tree")
    return {"model": "Decision Tree", "latency_ms": lat, "throughput": thr,
            "sample_size": SAMPLE_SIZE, "gpu": "CPU"}


def bench_lstm(X_dl):
    print("\n" + "═" * 60)
    print("⚡ LSTM")
    path = os.path.join(MODELS_DIR, "lstm_best.keras")
    if not os.path.exists(path):
        print(f"   ❌ Model not found: {path}")
        return None
    model = tf.keras.models.load_model(path)

    # Warm-up (TF graph compilation)
    print("   🔥 Warm-up pass...")
    model.predict(X_dl[:WARMUP_SIZE], batch_size=256, verbose=0)

    lat, thr = benchmark(
        lambda: model.predict(X_dl, batch_size=256, verbose=0), "LSTM")
    return {"model": "LSTM", "latency_ms": lat, "throughput": thr,
            "sample_size": SAMPLE_SIZE, "gpu": "CUDA (if available)"}


def bench_bilstm(X_dl):
    print("\n" + "═" * 60)
    print("⚡ BiLSTM")
    path = os.path.join(MODELS_DIR, "bilstm_model.keras")
    if not os.path.exists(path):
        print(f"   ❌ Model not found: {path}")
        return None
    model = tf.keras.models.load_model(path)

    print("   🔥 Warm-up pass...")
    model.predict(X_dl[:WARMUP_SIZE], batch_size=256, verbose=0)

    lat, thr = benchmark(
        lambda: model.predict(X_dl, batch_size=256, verbose=0), "BiLSTM")
    return {"model": "BiLSTM", "latency_ms": lat, "throughput": thr,
            "sample_size": SAMPLE_SIZE, "gpu": "CUDA (if available)"}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 60)
    print("  NIDS — Unified Inference Latency Benchmark")
    print(f"  Protocol: {SAMPLE_SIZE:,} samples, 1 warm-up + 1 timed run")
    print("═" * 60)

    X_ml = load_ml_data()
    X_dl = load_dl_data()

    results = []
    for fn, arg in [
        (bench_xgboost,       X_ml),
        (bench_random_forest, X_ml),
        (bench_decision_tree, X_ml),
        (bench_lstm,          X_dl),
        (bench_bilstm,        X_dl),
    ]:
        r = fn(arg)
        if r:
            results.append(r)

    # ── Summary table ────────────────────────────────────────────
    print("\n\n" + "═" * 60)
    print("  RESULTS SUMMARY")
    print("═" * 60)
    print(f"{'Model':<22} {'Latency (ms)':<16} {'Throughput (s/s)':<20} {'GPU'}")
    print("-" * 60)
    for r in results:
        print(f"{r['model']:<22} {r['latency_ms']:<16.4f} {r['throughput']:<20,.0f} {r['gpu']}")

    # ── Save JSON ─────────────────────────────────────────────────
    out_path = os.path.join(REPORTS_DIR, "latency_benchmark.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"benchmark_protocol": f"{SAMPLE_SIZE}_samples",
                   "results": results}, f, indent=2)
    print(f"\n✅ Results saved to: {out_path}")
    print("   → Use these values to update Table 23 in 9results_analysis.html\n")


if __name__ == "__main__":
    main()
