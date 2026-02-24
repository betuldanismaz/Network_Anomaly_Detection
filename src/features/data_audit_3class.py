"""
3-Class Data Audit — Quality Assurance Pipeline
=================================================

Read-only inspection of data/processed_ml/{train,val,test}.csv.
Runs 4 audit modules, prints a colour-coded terminal report,
and writes a plain-text summary to reports/data_audit_3class_summary.txt.

Author: Network Detection Team
"""

import os
import sys
import io
import numpy as np
import pandas as pd
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(ROOT, 'data', 'processed_ml')
REPORTS_DIR = os.path.join(ROOT, 'reports')
SUMMARY_PATH = os.path.join(REPORTS_DIR, 'data_audit_3class_summary.txt')

os.makedirs(REPORTS_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# ANSI COLOURS
# ═══════════════════════════════════════════════════════════════════════
GREEN  = '\033[92m'
RED    = '\033[91m'
CYAN   = '\033[96m'
YELLOW = '\033[93m'
BOLD   = '\033[1m'
RESET  = '\033[0m'

# Class names for pretty printing
CLASS_NAMES = {0: 'Benign', 1: 'Volumetric', 2: 'Semantic'}

# Floating-point tolerance for MinMaxScaler range check
EPSILON = 1e-6


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

class DualWriter:
    """Write to both the terminal (with colour) and a plain-text buffer."""

    def __init__(self):
        self._buf = io.StringIO()

    def header(self, title: str, module_num: Optional[int] = None) -> None:
        tag = f"MODULE {module_num}: " if module_num else ""
        line = f"=== {tag}{title} ==="
        border = "=" * len(line)
        print(f"\n{CYAN}{BOLD}{border}{RESET}")
        print(f"{CYAN}{BOLD}{line}{RESET}")
        print(f"{CYAN}{BOLD}{border}{RESET}")
        self._buf.write(f"\n{border}\n{line}\n{border}\n")

    def info(self, msg: str) -> None:
        print(f"  {CYAN}ℹ{RESET}  {msg}")
        self._buf.write(f"  [INFO]  {msg}\n")

    def passed(self, msg: str) -> None:
        print(f"  {GREEN}✅ PASS{RESET}  {msg}")
        self._buf.write(f"  [PASS]  {msg}\n")

    def fail(self, msg: str) -> None:
        print(f"  {RED}❌ FAIL{RESET}  {msg}")
        self._buf.write(f"  [FAIL]  {msg}\n")

    def warn(self, msg: str) -> None:
        print(f"  {YELLOW}⚠️  WARN{RESET}  {msg}")
        self._buf.write(f"  [WARN]  {msg}\n")

    def line(self, msg: str = "") -> None:
        print(f"  {msg}")
        self._buf.write(f"  {msg}\n")

    def blank(self) -> None:
        print()
        self._buf.write("\n")

    def get_text(self) -> str:
        return self._buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════
# MODULE 1 — FILE INTEGRITY & SHAPE CHECK
# ═══════════════════════════════════════════════════════════════════════

def module_shape_check(
    dfs: Dict[str, pd.DataFrame], out: DualWriter
) -> bool:
    """Verify existence (already guaranteed) and column-count consistency."""
    out.header("FILE INTEGRITY & SHAPE CHECK", module_num=1)
    ok = True

    shapes: Dict[str, Tuple[int, int]] = {}
    for name, df in dfs.items():
        shapes[name] = df.shape
        out.info(f"{name:5s}  →  {df.shape[0]:>10,} rows  ×  {df.shape[1]:>3} cols")

    col_counts = {n: s[1] for n, s in shapes.items()}
    if len(set(col_counts.values())) == 1:
        out.passed(f"All splits have identical column count ({list(col_counts.values())[0]})")
    else:
        out.fail(f"Column count mismatch across splits: {col_counts}")
        ok = False

    return ok


# ═══════════════════════════════════════════════════════════════════════
# MODULE 2 — MISSING & INFINITE VALUES CHECK
# ═══════════════════════════════════════════════════════════════════════

def module_nan_inf_check(
    dfs: Dict[str, pd.DataFrame], out: DualWriter
) -> bool:
    """Scan every column for NaN or ±Inf values."""
    out.header("MISSING & INFINITE VALUES CHECK", module_num=2)
    ok = True

    for name, df in dfs.items():
        nan_total = int(df.isna().sum().sum())
        numeric = df.select_dtypes(include=[np.number])
        inf_total = int(np.isinf(numeric).sum().sum())

        if nan_total == 0 and inf_total == 0:
            out.passed(f"{name:5s}  →  0 NaN, 0 Inf  —  CLEAN")
        else:
            ok = False
            out.fail(f"{name:5s}  →  {nan_total:,} NaN, {inf_total:,} Inf")
            # Detail per-column
            if nan_total > 0:
                nan_cols = df.columns[df.isna().any()].tolist()
                out.warn(f"       NaN columns: {nan_cols}")
            if inf_total > 0:
                inf_cols = numeric.columns[np.isinf(numeric).any()].tolist()
                out.warn(f"       Inf columns: {inf_cols}")

    return ok


# ═══════════════════════════════════════════════════════════════════════
# MODULE 3 — LABEL DISTRIBUTION & STRATIFICATION PROOF
# ═══════════════════════════════════════════════════════════════════════

def module_label_distribution(
    dfs: Dict[str, pd.DataFrame], out: DualWriter
) -> bool:
    """
    Print absolute + percentage distribution for each class per split.
    Prove stratification by showing that class proportions are consistent
    (max deviation < 0.5 percentage points across splits for each class).
    """
    out.header("LABEL DISTRIBUTION & STRATIFICATION PROOF", module_num=3)
    ok = True

    # ── Per-split distribution table ────────────────────────────────────
    dist_table: Dict[str, Dict[int, float]] = {}  # name -> {cls: pct}

    for name, df in dfs.items():
        counts = df['Label'].value_counts().sort_index()
        total = len(df)
        out.info(f"{name} label distribution ({total:,} rows):")
        split_pcts: Dict[int, float] = {}
        for cls in [0, 1, 2]:
            cnt = int(counts.get(cls, 0))
            pct = cnt / total * 100 if total > 0 else 0.0
            split_pcts[cls] = pct
            tag = CLASS_NAMES.get(cls, f"Class {cls}")
            out.line(f"    Class {cls} ({tag:11s}):  {cnt:>10,}  ({pct:6.2f}%)")
        dist_table[name] = split_pcts
        out.blank()

    # ── Stratification consistency check ────────────────────────────────
    # For each class, compute the max absolute deviation across splits.
    MAX_DEVIATION_PP = 0.5  # max acceptable deviation in percentage points
    out.info("Stratification consistency (max deviation per class across splits):")

    for cls in [0, 1, 2]:
        pcts = [dist_table[n][cls] for n in dfs]
        deviation = max(pcts) - min(pcts)
        tag = CLASS_NAMES.get(cls, f"Class {cls}")
        if deviation <= MAX_DEVIATION_PP:
            out.passed(f"Class {cls} ({tag}):  Δ = {deviation:.3f} pp  (≤ {MAX_DEVIATION_PP} pp)")
        else:
            out.warn(f"Class {cls} ({tag}):  Δ = {deviation:.3f} pp  (> {MAX_DEVIATION_PP} pp threshold)")
            # A warning, not a hard failure — small deviations are expected
            # for extremely rare classes processed per-file.

    # ── All 3 classes present? ──────────────────────────────────────────
    out.blank()
    for name, df in dfs.items():
        present = set(df['Label'].unique())
        missing = {0, 1, 2} - present
        if missing:
            out.fail(f"{name} is missing classes: {missing}")
            ok = False
        else:
            out.passed(f"{name} has all 3 classes present")

    return ok


# ═══════════════════════════════════════════════════════════════════════
# MODULE 4 — SCALING VERIFICATION (MinMaxScaler)
# ═══════════════════════════════════════════════════════════════════════

def module_scaling_check(
    dfs: Dict[str, pd.DataFrame], out: DualWriter
) -> bool:
    """
    Verify that all feature columns fall within [0, 1] (± EPSILON).
    The scaler was fit on train only, so val/test values may slightly
    exceed [0,1] if their raw values were outside the train range —
    that is expected and we flag it as a warning, not a failure.
    """
    out.header("SCALING VERIFICATION (MinMaxScaler)", module_num=4)
    ok = True

    for name, df in dfs.items():
        feature_cols = [c for c in df.columns if c != 'Label']
        features = df[feature_cols]

        col_min = features.min()
        col_max = features.max()

        global_min = col_min.min()
        global_max = col_max.max()

        out.info(f"{name:5s}  →  global min = {global_min:.8f},  global max = {global_max:.8f}")

        below_zero = col_min[col_min < -EPSILON]
        above_one  = col_max[col_max > 1.0 + EPSILON]

        if len(below_zero) == 0 and len(above_one) == 0:
            out.passed(f"{name:5s}  →  All features within [0, 1] (± {EPSILON})")
        else:
            # For train this is a definite failure; for val/test it is a warning
            # because the scaler was fit on train only.
            if name == "TRAIN":
                out.fail(f"{name} features outside [0, 1] range!")
                ok = False
            else:
                out.warn(f"{name} has features slightly outside [0, 1] (expected for val/test)")

            if len(below_zero) > 0:
                for col_name in below_zero.index:
                    out.line(f"       {col_name}: min = {below_zero[col_name]:.8f}")
            if len(above_one) > 0:
                for col_name in above_one.index:
                    out.line(f"       {col_name}: max = {above_one[col_name]:.8f}")

    return ok


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    out = DualWriter()

    out.header("3-CLASS DATA QUALITY AUDIT")
    out.info(f"Data directory: {DATA_DIR}")
    out.blank()

    # ── Load datasets ───────────────────────────────────────────────────
    split_names = ['train', 'val', 'test']
    paths = {n.upper(): os.path.join(DATA_DIR, f'{n}.csv') for n in split_names}

    # Verify file existence first
    all_exist = True
    for name, path in paths.items():
        if os.path.isfile(path):
            out.passed(f"{name} file exists  →  {os.path.basename(path)}")
        else:
            out.fail(f"{name} file NOT FOUND  →  {path}")
            all_exist = False

    if not all_exist:
        out.fail("Cannot proceed — missing data files.")
        _save_summary(out)
        return

    out.blank()
    out.info("Loading datasets into memory...")
    dfs: Dict[str, pd.DataFrame] = {}
    for name, path in paths.items():
        dfs[name] = pd.read_csv(path)
        out.info(f"  Loaded {name}: {dfs[name].shape}")

    # ── Run audit modules ───────────────────────────────────────────────
    results: OrderedDict[str, bool] = OrderedDict()

    results['1. Shape Check']          = module_shape_check(dfs, out)
    results['2. NaN / Inf Check']      = module_nan_inf_check(dfs, out)
    results['3. Label Distribution']   = module_label_distribution(dfs, out)
    results['4. Scaling Verification'] = module_scaling_check(dfs, out)

    # ── Final scorecard ─────────────────────────────────────────────────
    out.header("FINAL SCORECARD")
    total = len(results)
    passed = sum(1 for v in results.values() if v)

    for module_name, status in results.items():
        icon = f"{GREEN}PASS{RESET}" if status else f"{RED}FAIL{RESET}"
        out.line(f"  {icon}  {module_name}")

    out.blank()
    if passed == total:
        out.line(f"{GREEN}{BOLD}  ════════════════════════════════════════════{RESET}")
        out.line(f"{GREEN}{BOLD}   OVERALL: {passed}/{total} PASSED  ✅  DATA IS READY FOR TRAINING{RESET}")
        out.line(f"{GREEN}{BOLD}  ════════════════════════════════════════════{RESET}")
    else:
        out.line(f"{RED}{BOLD}  ════════════════════════════════════════════{RESET}")
        out.line(f"{RED}{BOLD}   OVERALL: {passed}/{total} PASSED  ⚠️  ISSUES DETECTED{RESET}")
        out.line(f"{RED}{BOLD}  ════════════════════════════════════════════{RESET}")

    out.blank()

    # ── Save plain-text summary ─────────────────────────────────────────
    _save_summary(out)


def _save_summary(out: DualWriter) -> None:
    """Write the colour-stripped summary to disk."""
    import re
    # Strip ANSI escape codes for the text file
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    clean_text = ansi_escape.sub('', out.get_text())

    with open(SUMMARY_PATH, 'w', encoding='utf-8') as f:
        f.write(clean_text)
    print(f"\n{CYAN}📄 Plain-text summary saved → {SUMMARY_PATH}{RESET}\n")


if __name__ == '__main__':
    main()
