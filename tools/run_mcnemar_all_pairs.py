#!/usr/bin/env python3
"""
McNemar istatistiksel anlamlilik testi — tum model ciftleri.

5 model (RF, XGBoost, DT, LSTM, BiLSTM) arasinda eslesik karsilastirma:
  - Tabular modeller (RF, XGBoost, DT): data/processed_ml/test.csv
  - Sequence modeller (LSTM, BiLSTM): data/processed_lstm/X_test.npy

Ciktilar:
  - reports/mcnemar_summary.txt   (birlestik paper-ready tablo)
  - reports/mcnemar_summary.csv   (makine okunur)
  - Her cift icin ayri detay raporu reports/mcnemar_<A>_vs_<B>.txt
"""

import json
import math
import os
import sys
from datetime import datetime
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_ML = os.path.join(PROJECT_ROOT, "data", "processed_ml")
DATA_LSTM = os.path.join(PROJECT_ROOT, "data", "processed_lstm")

CLASS_NAMES = ["Benign", "Volumetric", "Semantic"]

TABULAR_MODELS = ["Random Forest", "XGBoost", "Decision Tree"]
SEQUENCE_MODELS = ["LSTM", "BiLSTM"]


def mcnemar_chi2(b: int, c: int, cc: bool = True) -> Tuple[float, float]:
    n = b + c
    if n == 0:
        return 0.0, 1.0
    if cc:
        chi2 = (abs(b - c) - 1) ** 2 / n
    else:
        chi2 = (b - c) ** 2 / n
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return float(chi2), float(p)


def load_tabular_models() -> Dict[str, object]:
    import joblib

    mapping = {
        "Random Forest": "rf_3class_model.pkl",
        "XGBoost": "xgb_3class_model.pkl",
        "Decision Tree": "dt_3class_model.pkl",
    }
    models = {}
    for name, fname in mapping.items():
        path = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(path):
            print(f"  [!] {name} bulunamadi: {path}")
            continue
        try:
            models[name] = joblib.load(path)
            print(f"  [+] {name} yuklendi")
        except Exception as e:
            print(f"  [!] {name} yuklenemedi: {e}")
    return models


def load_sequence_models() -> Dict[str, object]:
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        from tensorflow import keras
    except ImportError:
        print("  [!] tensorflow bulunamadi — LSTM/BiLSTM atlaniyor")
        return {}

    mapping = {
        "LSTM": "lstm_model.keras",
        "BiLSTM": "bilstm_model.keras",
    }
    models = {}
    for name, fname in mapping.items():
        path = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(path):
            print(f"  [!] {name} bulunamadi: {path}")
            continue
        models[name] = keras.models.load_model(path)
        print(f"  [+] {name} yuklendi")
    return models


def predict_tabular(models: Dict[str, object], X: np.ndarray) -> Dict[str, np.ndarray]:
    preds = {}
    for name, model in models.items():
        preds[name] = model.predict(X)
        print(f"  [+] {name} tahminleri uretildi ({len(preds[name]):,} ornek)")
    return preds


def predict_sequence(models: Dict[str, object], X: np.ndarray) -> Dict[str, np.ndarray]:
    preds = {}
    for name, model in models.items():
        proba = model.predict(X, verbose=0)
        preds[name] = np.argmax(proba, axis=1)
        print(f"  [+] {name} tahminleri uretildi ({len(preds[name]):,} ornek)")
    return preds


def run_mcnemar_pair(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    name_a: str,
    name_b: str,
    alpha: float = 0.05,
) -> dict:
    n = len(y_true)
    f1_a = f1_score(y_true, pred_a, average="macro")
    f1_b = f1_score(y_true, pred_b, average="macro")

    a_ok = pred_a == y_true
    b_ok = pred_b == y_true

    a11 = int(np.sum(a_ok & b_ok))
    a10 = int(np.sum(a_ok & ~b_ok))
    a01 = int(np.sum(~a_ok & b_ok))
    a00 = int(np.sum(~a_ok & ~b_ok))

    chi2_cc, p_cc = mcnemar_chi2(a10, a01, cc=True)
    chi2_no, p_no = mcnemar_chi2(a10, a01, cc=False)

    per_class = {}
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        mask = y_true == cls_id
        if mask.sum() == 0:
            continue
        ca_ok = pred_a[mask] == y_true[mask]
        cb_ok = pred_b[mask] == y_true[mask]
        cb10 = int(np.sum(ca_ok & ~cb_ok))
        cb01 = int(np.sum(~ca_ok & cb_ok))
        cls_chi2, cls_p = mcnemar_chi2(cb10, cb01, cc=True)
        per_class[cls_name] = {
            "b": cb10,
            "c": cb01,
            "chi2": cls_chi2,
            "p": cls_p,
            "significant": cls_p < alpha,
        }

    return {
        "name_a": name_a,
        "name_b": name_b,
        "n_samples": n,
        "f1_a": f1_a,
        "f1_b": f1_b,
        "delta_f1": f1_a - f1_b,
        "a11": a11,
        "a10_b": a10,
        "a01_c": a01,
        "a00": a00,
        "chi2_cc": chi2_cc,
        "p_cc": p_cc,
        "chi2_no_cc": chi2_no,
        "p_no_cc": p_no,
        "significant_cc": p_cc < alpha,
        "significant_no_cc": p_no < alpha,
        "alpha": alpha,
        "per_class": per_class,
    }


def format_detail_report(r: dict) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append(f"  McNemar Testi: {r['name_a']} vs {r['name_b']}")
    lines.append("=" * 78)
    lines.append(f"Ornek sayisi (N): {r['n_samples']:,}")
    lines.append("")
    lines.append("--- Macro F1 ---")
    lines.append(f"  {r['name_a']:20s}: {r['f1_a']:.6f}")
    lines.append(f"  {r['name_b']:20s}: {r['f1_b']:.6f}")
    lines.append(f"  Fark (A - B):        {r['delta_f1']:+.6f}")
    lines.append("")
    lines.append("--- 2x2 Uyumsuzluk (contingency) tablosu ---")
    lines.append(f"  Her ikisi dogru   (n11): {r['a11']:>10,}")
    lines.append(f"  Sadece A dogru    (b)  : {r['a10_b']:>10,}")
    lines.append(f"  Sadece B dogru    (c)  : {r['a01_c']:>10,}")
    lines.append(f"  Her ikisi yanlis  (n00): {r['a00']:>10,}")
    lines.append("")
    lines.append("--- McNemar ki-kare ---")
    lines.append(f"  CC acik  : chi2 = {r['chi2_cc']:10.4f}   p = {r['p_cc']:.8f}  {'*' if r['significant_cc'] else ''}")
    lines.append(f"  CC kapali: chi2 = {r['chi2_no_cc']:10.4f}   p = {r['p_no_cc']:.8f}  {'*' if r['significant_no_cc'] else ''}")
    lines.append(f"  alpha = {r['alpha']}")
    sig_text = "ANLAMLI (H0 reddedilir)" if r["significant_cc"] else "ANLAMLI DEGIL (H0 reddedilemez)"
    lines.append(f"  Karar (CC acik): {sig_text}")
    lines.append("")
    lines.append("--- Sinif bazli McNemar (CC acik) ---")
    lines.append(f"  {'Sinif':12s} {'b':>8s} {'c':>8s} {'chi2':>10s} {'p':>12s} {'Anlamli':>8s}")
    lines.append(f"  {'-'*60}")
    for cls_name in CLASS_NAMES:
        if cls_name not in r["per_class"]:
            continue
        pc = r["per_class"][cls_name]
        sig = "Evet" if pc["significant"] else "Hayir"
        lines.append(
            f"  {cls_name:12s} {pc['b']:>8,} {pc['c']:>8,} {pc['chi2']:>10.4f} {pc['p']:>12.8f} {sig:>8s}"
        )
    lines.append("=" * 78)
    return "\n".join(lines)


def format_summary_table(results: List[dict]) -> str:
    lines = []
    lines.append("=" * 100)
    lines.append("  McNEMAR ISTATISTIKSEL ANLAMLILIK OZET TABLOSU")
    lines.append("  Network Intrusion Detection — 3-Class (Benign / Volumetric / Semantic)")
    lines.append(f"  Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 100)
    lines.append("")

    hdr = (
        f"  {'Cift':>3s}  {'Model A':>16s}  {'Model B':>16s}  "
        f"{'F1_A':>8s}  {'F1_B':>8s}  {'dF1':>8s}  "
        f"{'b':>7s}  {'c':>7s}  {'chi2':>9s}  {'p':>12s}  {'Anlamli':>8s}  {'N':>10s}"
    )
    lines.append(hdr)
    lines.append("  " + "-" * 96)

    for i, r in enumerate(results, 1):
        sig = "***" if r["p_cc"] < 0.001 else ("**" if r["p_cc"] < 0.01 else ("*" if r["p_cc"] < 0.05 else "n.s."))
        lines.append(
            f"  {i:>3d}  {r['name_a']:>16s}  {r['name_b']:>16s}  "
            f"{r['f1_a']:>8.4f}  {r['f1_b']:>8.4f}  {r['delta_f1']:>+8.4f}  "
            f"{r['a10_b']:>7,}  {r['a01_c']:>7,}  {r['chi2_cc']:>9.2f}  {r['p_cc']:>12.2e}  {sig:>8s}  {r['n_samples']:>10,}"
        )

    lines.append("")
    lines.append("  Anlamlilik: *** p<0.001, ** p<0.01, * p<0.05, n.s. p>=0.05")
    lines.append("  CC: Sureklilik duzeltmesi (Yates) uygulanmis")
    lines.append("")

    lines.append("  SINIF BAZLI McNEMAR SONUCLARI (CC acik, alpha=0.05)")
    lines.append("  " + "-" * 96)
    cls_hdr = (
        f"  {'Cift':>3s}  {'Model A':>16s}  {'Model B':>16s}  "
        f"{'Sinif':>12s}  {'b':>7s}  {'c':>7s}  {'chi2':>9s}  {'p':>12s}  {'Anlamli':>8s}"
    )
    lines.append(cls_hdr)
    lines.append("  " + "-" * 96)

    for i, r in enumerate(results, 1):
        for cls_name in CLASS_NAMES:
            if cls_name not in r["per_class"]:
                continue
            pc = r["per_class"][cls_name]
            sig = "***" if pc["p"] < 0.001 else ("**" if pc["p"] < 0.01 else ("*" if pc["p"] < 0.05 else "n.s."))
            lines.append(
                f"  {i:>3d}  {r['name_a']:>16s}  {r['name_b']:>16s}  "
                f"{cls_name:>12s}  {pc['b']:>7,}  {pc['c']:>7,}  {pc['chi2']:>9.2f}  {pc['p']:>12.2e}  {sig:>8s}"
            )
        lines.append("  " + "." * 96)

    lines.append("")
    lines.append("=" * 100)
    return "\n".join(lines)


def results_to_csv_rows(results: List[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "model_a": r["name_a"],
                "model_b": r["name_b"],
                "n_samples": r["n_samples"],
                "f1_a": round(r["f1_a"], 6),
                "f1_b": round(r["f1_b"], 6),
                "delta_f1": round(r["delta_f1"], 6),
                "b_only_a_correct": r["a10_b"],
                "c_only_b_correct": r["a01_c"],
                "chi2_cc": round(r["chi2_cc"], 4),
                "p_value_cc": r["p_cc"],
                "significant_005": r["significant_cc"],
                "chi2_no_cc": round(r["chi2_no_cc"], 4),
                "p_value_no_cc": r["p_no_cc"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    all_results: List[dict] = []

    # ── 1) Tabular modeller ──────────────────────────────────────────────
    print("\n[1/4] Tabular modeller yukleniyor...")
    tab_models = load_tabular_models()

    if len(tab_models) >= 2:
        print("\n[2/4] test.csv yukleniyor...")
        test_df = pd.read_csv(os.path.join(DATA_ML, "test.csv"))
        X_test = test_df.drop("Label", axis=1).values
        y_test_tab = test_df["Label"].values
        print(f"  Ornek: {len(y_test_tab):,}  Ozellik: {X_test.shape[1]}")

        print("\n[2/4] Tabular tahminler uretiliyor...")
        tab_preds = predict_tabular(tab_models, X_test)

        print("\n[3/4] Tabular ciftler icin McNemar testi...")
        for name_a, name_b in combinations(tab_preds.keys(), 2):
            r = run_mcnemar_pair(y_test_tab, tab_preds[name_a], tab_preds[name_b], name_a, name_b)
            all_results.append(r)
            sig = "ANLAMLI" if r["significant_cc"] else "anlamsiz"
            print(f"  {name_a} vs {name_b}: p={r['p_cc']:.2e} ({sig})")
    else:
        print("  [!] En az 2 tabular model gerekli, atlaniyor")

    # ── 2) Sequence modeller ─────────────────────────────────────────────
    print("\n[3/4] Sequence modeller yukleniyor...")
    seq_models = load_sequence_models()

    if len(seq_models) >= 2:
        print("\n[3/4] X_test.npy / y_test.npy yukleniyor...")
        X_test_seq = np.load(os.path.join(DATA_LSTM, "X_test.npy"))
        y_test_seq = np.load(os.path.join(DATA_LSTM, "y_test.npy"))
        print(f"  Ornek: {len(y_test_seq):,}  Sekans: {X_test_seq.shape[1]}x{X_test_seq.shape[2]}")

        print("\n[3/4] Sequence tahminler uretiliyor...")
        seq_preds = predict_sequence(seq_models, X_test_seq)

        print("\n[3/4] Sequence ciftler icin McNemar testi...")
        for name_a, name_b in combinations(seq_preds.keys(), 2):
            r = run_mcnemar_pair(y_test_seq, seq_preds[name_a], seq_preds[name_b], name_a, name_b)
            all_results.append(r)
            sig = "ANLAMLI" if r["significant_cc"] else "anlamsiz"
            print(f"  {name_a} vs {name_b}: p={r['p_cc']:.2e} ({sig})")
    else:
        print("  [!] En az 2 sequence model gerekli, atlaniyor")

    if not all_results:
        print("\n[HATA] Hicbir cift test edilemedi!")
        sys.exit(1)

    # ── 3) Raporlari kaydet ──────────────────────────────────────────────
    print(f"\n[4/4] Raporlar kaydediliyor... ({len(all_results)} cift)")

    for r in all_results:
        detail = format_detail_report(r)
        safe_a = r["name_a"].replace(" ", "_")
        safe_b = r["name_b"].replace(" ", "_")
        detail_path = os.path.join(REPORTS_DIR, f"mcnemar_{safe_a}_vs_{safe_b}.txt")
        with open(detail_path, "w", encoding="utf-8") as f:
            f.write(detail + "\n")
        print(f"  [+] {detail_path}")

    summary_txt = format_summary_table(all_results)
    summary_path = os.path.join(REPORTS_DIR, "mcnemar_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_txt + "\n")
    print(f"  [+] {summary_path}")

    csv_df = results_to_csv_rows(all_results)
    csv_path = os.path.join(REPORTS_DIR, "mcnemar_summary.csv")
    csv_df.to_csv(csv_path, index=False)
    print(f"  [+] {csv_path}")

    # ── 4) Sonuc ozeti ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  McNEMAR TESTI TAMAMLANDI")
    print("=" * 60)
    print(f"  Toplam cift: {len(all_results)}")
    n_sig = sum(1 for r in all_results if r["significant_cc"])
    print(f"  Anlamli (p<0.05): {n_sig}")
    print(f"  Anlamsiz:         {len(all_results) - n_sig}")
    print()
    print(summary_txt)


if __name__ == "__main__":
    main()
