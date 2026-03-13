#!/usr/bin/env python3
"""
McNemar test utility for paired model comparison.

Compares two classifiers on the same samples using:
1) Macro F1 scores
2) McNemar test on paired correctness outcomes

No external stats package is required.
"""

import argparse
import math
import os
import sys
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run McNemar test for two models on the same test set and report Macro F1 difference."
        )
    )

    parser.add_argument("--y-true", required=True, help="Path to true labels (.csv or .npy)")
    parser.add_argument("--pred-a", required=True, help="Path to model A predictions (.csv or .npy)")
    parser.add_argument("--pred-b", required=True, help="Path to model B predictions (.csv or .npy)")

    parser.add_argument(
        "--true-col",
        default=None,
        help="Column name for y_true if input is CSV with multiple columns",
    )
    parser.add_argument(
        "--pred-a-col",
        default=None,
        help="Column name for model A predictions if input is CSV with multiple columns",
    )
    parser.add_argument(
        "--pred-b-col",
        default=None,
        help="Column name for model B predictions if input is CSV with multiple columns",
    )

    parser.add_argument("--name-a", default="Model_A", help="Display name of model A")
    parser.add_argument("--name-b", default="Model_B", help="Display name of model B")
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="Significance level (default: 0.05)"
    )
    parser.add_argument(
        "--no-cc",
        action="store_true",
        help="Disable continuity correction in McNemar chi-square",
    )

    parser.add_argument(
        "--out-path",
        default=None,
        help=(
            "Optional output .txt path. If not provided, the report is saved to "
            "reports/mcnemar_<modelA>_vs_<modelB>_<timestamp>.txt"
        ),
    )

    return parser.parse_args()


def _load_from_csv(path: str, column: str = None) -> np.ndarray:
    df = pd.read_csv(path)

    if column is not None:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in {path}. Available: {list(df.columns)}")
        values = df[column].to_numpy()
    else:
        if df.shape[1] == 1:
            values = df.iloc[:, 0].to_numpy()
        else:
            raise ValueError(
                f"CSV file {path} has multiple columns. Please provide --*-col argument."
            )

    return values


def load_vector(path: str, column: str = None) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        arr = np.load(path)
        if arr.ndim != 1:
            raise ValueError(f"Expected 1D array in {path}, got shape {arr.shape}")
        return arr

    if ext == ".csv":
        return _load_from_csv(path, column)

    raise ValueError(f"Unsupported file extension for {path}. Use .csv or .npy")


def mcnemar_chi_square(b: int, c: int, continuity_correction: bool = True) -> Tuple[float, float]:
    """
    McNemar chi-square statistic and p-value for 1 degree of freedom.

    For df=1, p-value can be computed via:
    p = erfc(sqrt(chi2 / 2))
    """
    n_discordant = b + c
    if n_discordant == 0:
        return 0.0, 1.0

    if continuity_correction:
        chi2 = (abs(b - c) - 1) ** 2 / n_discordant
    else:
        chi2 = (b - c) ** 2 / n_discordant

    p_value = math.erfc(math.sqrt(chi2 / 2.0))
    return float(chi2), float(p_value)


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def resolve_output_path(args: argparse.Namespace) -> str:
    if args.out_path:
        out_path = args.out_path
    else:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        reports_dir = os.path.join(project_root, "reports")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_a = _safe_name(args.name_a)
        name_b = _safe_name(args.name_b)
        out_path = os.path.join(reports_dir, f"mcnemar_{name_a}_vs_{name_b}_{timestamp}.txt")

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    return out_path


def main() -> None:
    args = parse_args()

    y_true = load_vector(args.y_true, args.true_col)
    y_pred_a = load_vector(args.pred_a, args.pred_a_col)
    y_pred_b = load_vector(args.pred_b, args.pred_b_col)

    n = len(y_true)
    if len(y_pred_a) != n or len(y_pred_b) != n:
        raise ValueError(
            f"Length mismatch: y_true={n}, pred_a={len(y_pred_a)}, pred_b={len(y_pred_b)}"
        )

    # Align types to avoid false mismatches due to dtype differences.
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    f1_a = f1_score(y_true, y_pred_a, average="macro")
    f1_b = f1_score(y_true, y_pred_b, average="macro")
    delta_f1 = f1_a - f1_b

    a_correct = y_pred_a == y_true
    b_correct = y_pred_b == y_true

    # Contingency table cells for paired correctness:
    # b: A correct, B wrong
    # c: A wrong, B correct
    a11 = int(np.sum(a_correct & b_correct))
    a10 = int(np.sum(a_correct & ~b_correct))  # b
    a01 = int(np.sum(~a_correct & b_correct))  # c
    a00 = int(np.sum(~a_correct & ~b_correct))

    chi2, p_value = mcnemar_chi_square(a10, a01, continuity_correction=(not args.no_cc))

    lines = []
    lines.append("=" * 72)
    lines.append("McNemar Test Sonucu (Eslesik Model Karsilastirmasi)")
    lines.append("=" * 72)
    lines.append(f"Ornek sayisi: {n}")
    lines.append("")
    lines.append("Model performansi (Macro F1):")
    lines.append(f"- {args.name_a}: {f1_a:.6f}")
    lines.append(f"- {args.name_b}: {f1_b:.6f}")
    lines.append(f"- Fark ({args.name_a} - {args.name_b}): {delta_f1:+.6f}")
    lines.append("")
    lines.append("McNemar 2x2 tablosu (dogru/yanlis):")
    lines.append(f"- Her ikisi dogru   (a11): {a11}")
    lines.append(f"- Sadece {args.name_a} dogru (a10=b): {a10}")
    lines.append(f"- Sadece {args.name_b} dogru (a01=c): {a01}")
    lines.append(f"- Her ikisi yanlis  (a00): {a00}")
    lines.append("")

    cc_text = "Acilmis" if not args.no_cc else "Kapali"
    lines.append(f"McNemar ki-kare (CC={cc_text}): {chi2:.6f}")
    lines.append(f"p-degeri: {p_value:.8f}")
    lines.append(f"alpha: {args.alpha:.4f}")

    if p_value < args.alpha:
        lines.append("Karar: H0 reddedilir -> modeller arasindaki fark istatistiksel olarak anlamli.")
    else:
        lines.append("Karar: H0 reddedilemez -> fark istatistiksel olarak anlamli degil.")

    lines.append("=" * 72)

    report_text = "\n".join(lines)
    print(report_text)

    output_path = resolve_output_path(args)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")

    print(f"Rapor kaydedildi: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Hata: {exc}", file=sys.stderr)
        sys.exit(1)
