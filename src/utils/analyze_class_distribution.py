import os
import glob
import json
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import unicodedata
import re

# Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# allow override via env var
DATA_DIR = os.getenv('PROCESSED_CSV_DIR') or os.path.join(ROOT_DIR, 'data', 'processed_csv')
OUT_FIG_DIR = os.getenv('REPORTS_FIGURES_DIR') or os.path.join(ROOT_DIR, 'reports', 'figures')
OUT_FIG = os.path.join(OUT_FIG_DIR, 'class_distribution_3class.png')

# Resolve classes_map.json from env or common locations
possible_class_map_paths = [
    os.getenv('CLASSES_MAP_PATH'),
    os.path.join(os.path.dirname(__file__), 'classes_map.json'),
    os.path.join(ROOT_DIR, 'src', 'utils', 'classes_map.json'),
    os.path.join(ROOT_DIR, 'classes_map.json'),
]
CLASS_MAP_PATH = next((p for p in possible_class_map_paths if p and os.path.exists(p)), None)
if CLASS_MAP_PATH is None:
    CLASS_MAP_PATH = os.path.join(os.path.dirname(__file__), 'classes_map.json')

# Expected classes
CLASS_LABELS = {0: 'Normal', 1: 'Volumetric', 2: 'Infiltration'}


def load_class_map(path):
    with open(path, 'r') as f:
        return json.load(f)


def analyze_distribution():
    class_map = load_class_map(CLASS_MAP_PATH)

    # counters
    mapped_counts = Counter()
    unknown_counts = Counter()
    total_rows = 0

    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
    if not csv_files:
        raise RuntimeError(f"No CSV files found in {DATA_DIR}")

    # Prepare normalized class_map to handle unicode/whitespace variation
    def _normalize_text(x):
        if x is None:
            return ''
        s = str(x)
        s = unicodedata.normalize('NFKC', s)
        s = s.replace('\uFFFD', ' ')
        s = s.replace('\u00A0', ' ')
        s = s.replace('\ufeff', '')
        # collapse all whitespace (tabs, multiple spaces) to single space
        s = re.sub(r"\s+", ' ', s)
        return s.strip()

    normalized_map = { _normalize_text(k): v for k, v in class_map.items() }

    for f in csv_files:
        try:
            # Read only header to detect the exact label column name (case/whitespace variations)
            header_cols = pd.read_csv(f, nrows=0).columns.tolist()
            label_col = None
            for c in header_cols:
                if c and isinstance(c, str) and ('label' == c.strip().lower() or 'label' in c.strip().lower()):
                    label_col = c
                    break
            if label_col is None:
                sample_cols = header_cols[:10]
                print(f"[WARN] 'Label' column not found in {f}. Sample columns: {sample_cols}")
                continue
            s = pd.read_csv(f, usecols=[label_col], dtype=str)[label_col]
        except Exception as e:
            print(f"[WARN] Could not read 'Label' from {f}: {e}")
            continue

        total_rows += len(s)

        # Normalize labels and map using normalized_map
        s_norm = s.fillna('').map(_normalize_text)
        mapped = s_norm.map(normalized_map)

        # Count mapped
        vc_mapped = mapped.dropna().astype(int).value_counts()
        for cls, cnt in vc_mapped.items():
            mapped_counts[int(cls)] += int(cnt)

        # Count unknown normalized label values (keep original samples for diagnostics)
        unmapped_mask = mapped.isna()
        if unmapped_mask.any():
            vc_unknown = s_norm[unmapped_mask].value_counts()
            for label_name, cnt in vc_unknown.items():
                unknown_counts[str(label_name)] += int(cnt)

    # Consolidate counts for classes 0,1,2 (ensure keys exist)
    counts = {k: int(mapped_counts.get(k, 0)) for k in CLASS_LABELS.keys()}
    unknown_total = sum(unknown_counts.values())

    # Print summary
    print("\nClass distribution summary:")
    print(f"Total rows processed: {total_rows}")
    for k, name in CLASS_LABELS.items():
        c = counts[k]
        pct = (c / total_rows * 100) if total_rows else 0
        print(f"- Class {k} ({name}): {c} rows ({pct:.4f}%)")
    print(f"- Unknown labels (not in classes_map.json): {unknown_total} rows")
    if unknown_total:
        print("  Unique unknown label names (sample):")
        for i, (lbl, cnt) in enumerate(unknown_counts.most_common(10), 1):
            print(f"    {i}. {lbl}: {cnt}")

    # Percentages
    class_counts_array = np.array([counts[k] for k in sorted(CLASS_LABELS.keys())], dtype=float)

    # Imbalance ratio: max/min among the 3 classes
    nonzero = class_counts_array[class_counts_array > 0]
    if nonzero.size == 0:
        imbalance_ratio = None
        print("No class samples found to compute imbalance ratio.")
    else:
        max_c = nonzero.max()
        min_c = nonzero.min()
        if min_c == 0:
            imbalance_ratio = float('inf')
            print("Imbalance Ratio: infinite (one or more classes have zero samples)")
        else:
            ratio = max_c / min_c
            imbalance_ratio = ratio
            print(f"Imbalance Ratio (majority / minority): {ratio:.2f}:1")

    # Visualization
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    labels = [f"{CLASS_LABELS[k]} ({k})" for k in sorted(CLASS_LABELS.keys())]
    counts_plot = [counts[k] for k in sorted(CLASS_LABELS.keys())]
    if unknown_total:
        labels.append('Unknown')
        counts_plot.append(unknown_total)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts_plot, color=['#2ca02c', '#ff7f0e', '#1f77b4', '#7f7f7f'][:len(labels)])
    ax.set_yscale('log')
    ax.set_ylabel('Count (log scale)')
    ax.set_title('Class Distribution (3-class mapping)')

    # Annotate bars with raw counts
    for bar, cnt in zip(bars, counts_plot):
        height = bar.get_height()
        ax.annotate(f"{cnt}", xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=200)
    plt.close(fig)
    print(f"Saved class distribution plot to: {OUT_FIG}")

    # Recommendations
    print("\nAutomated recommendations:")
    # Compute percentages for existing classes
    for k in sorted(CLASS_LABELS.keys()):
        c = counts[k]
        pct = (c / total_rows * 100) if total_rows else 0
        if pct < 1 and c > 0:
            print(f"⚠️ Critical Imbalance for Class {k} ({CLASS_LABELS[k]}): {pct:.4f}% -> Recommendation -> Use SMOTE or Class Weights")
    # If any class is zero
    zero_classes = [k for k in sorted(CLASS_LABELS.keys()) if counts[k] == 0]
    if zero_classes:
        print(f"⚠️ The following classes have ZERO samples: {zero_classes} -> Consider data collection or use class weighting carefully.")

    # Balanced check: if no class is below 1% and the spread is small
    pct_array = np.array([(counts[k] / total_rows * 100) if total_rows else 0 for k in sorted(CLASS_LABELS.keys())])
    if (pct_array.max() - pct_array.min()) < 5 and (pct_array.min() > 0):
        print("✅ Balanced Dataset (class percentages within 5% range)")
    else:
        if not any(pct < 1 for pct in pct_array):
            print("ℹ️ Dataset is imbalanced but not critically (<1%). Consider class weights or targeted augmentation (SMOTE on minority classes).")

    return {
        'total_rows': int(total_rows),
        'class_counts': counts,
        'unknown_counts': dict(unknown_counts),
        'imbalance_ratio': imbalance_ratio,
        'plot_path': OUT_FIG,
    }


if __name__ == '__main__':
    summary = analyze_distribution()
    # Optionally write summary to a small JSON next to the plot
    try:
        summary_path = os.path.join(os.path.dirname(OUT_FIG), 'class_distribution_summary.json')
        with open(summary_path, 'w') as sf:
            json.dump(summary, sf, indent=2)
        print(f"Wrote summary JSON to: {summary_path}")
    except Exception:
        pass
