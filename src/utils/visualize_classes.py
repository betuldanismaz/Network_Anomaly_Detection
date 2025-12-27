import os
import json
import math
import unicodedata
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE

# Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TRAIN_CSV = os.path.join(ROOT_DIR, 'data', 'processed_csv', 'ready_splits', 'train.csv')
CLASS_MAP_PATH = os.path.join(os.path.dirname(__file__), 'classes_map.json')
OUT_FIG = os.path.join(ROOT_DIR, 'reports', 'figures', 'class_clusters_tsne.png')

# Import TOP_FEATURES if available
try:
    from src.config import TOP_FEATURES
except Exception:
    TOP_FEATURES = None

# Normalization utilities
def _normalize_text(x):
    if x is None:
        return ''
    s = str(x)
    s = unicodedata.normalize('NFKC', s)
    s = s.replace('\ufeff', '')
    s = s.replace('\u00A0', ' ')
    s = s.replace('\uFFFD', ' ')
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def load_class_map(path):
    with open(path, 'r') as f:
        raw = json.load(f)
    # normalize keys
    return { _normalize_text(k): v for k, v in raw.items() }


def find_label_column(columns):
    for c in columns:
        if not isinstance(c, str):
            continue
        if 'label' == c.strip().lower() or 'label' in c.strip().lower():
            return c
    return None


def load_sample(n=5000, random_state=42):
    # Try train.csv first
    if os.path.exists(TRAIN_CSV):
        try:
            df = pd.read_csv(TRAIN_CSV, low_memory=False)
            if df.shape[0] == 0:
                raise ValueError('Empty train.csv')
            # detect label column
            label_col = find_label_column(df.columns)
            if label_col is None:
                raise ValueError('Label column not found in train.csv')
            if df.shape[0] > n:
                df = df.sample(n=n, random_state=random_state)
            return df.reset_index(drop=True), label_col
        except Exception as e:
            print(f"[WARN] Could not load/train sample from {TRAIN_CSV}: {e}")

    # Fallback: sample across processed_csv folder
    processed_dir = os.path.join(ROOT_DIR, 'data', 'processed_csv')
    csv_files = sorted([p for p in [
        os.path.join(processed_dir, 'Monday-WorkingHours.pcap_ISCX.csv'),
        os.path.join(processed_dir, 'Tuesday-WorkingHours.pcap_ISCX.csv'),
        os.path.join(processed_dir, 'Wednesday-workingHours.pcap_ISCX.csv'),
        os.path.join(processed_dir, 'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv'),
        os.path.join(processed_dir, 'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv'),
        os.path.join(processed_dir, 'Friday-WorkingHours-Morning.pcap_ISCX.csv'),
        os.path.join(processed_dir, 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv'),
        os.path.join(processed_dir, 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
    ] if os.path.exists(p)])

    if not csv_files:
        raise RuntimeError(f'No CSV sources found under {processed_dir}')

    parts = []
    for p in csv_files:
        try:
            # read a small fraction when files are large
            dfp = pd.read_csv(p, low_memory=False)
            parts.append(dfp)
        except Exception as e:
            print(f"[WARN] Could not read {p}: {e}")
    if not parts:
        raise RuntimeError('No data could be read from processed_csv files')

    big = pd.concat(parts, axis=0, ignore_index=True)
    label_col = find_label_column(big.columns)
    if label_col is None:
        raise RuntimeError('No label column found in fallback CSVs')
    if big.shape[0] > n:
        big = big.sample(n=n, random_state=random_state)
    return big.reset_index(drop=True), label_col


def stratified_sample_with_classes(target_classes=(0, 1, 2), per_class=1667, random_state=42):
    """Attempt to build a stratified sample containing at least `per_class` rows for each class.
    Reads source CSVs in chunks to avoid OOM and stops when targets are met."""
    processed_dir = os.path.join(ROOT_DIR, 'data', 'processed_csv')
    csv_files = sorted([p for p in [
        os.path.join(processed_dir, 'Monday-WorkingHours.pcap_ISCX.csv'),
        os.path.join(processed_dir, 'Tuesday-WorkingHours.pcap_ISCX.csv'),
        os.path.join(processed_dir, 'Wednesday-workingHours.pcap_ISCX.csv'),
        os.path.join(processed_dir, 'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv'),
        os.path.join(processed_dir, 'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv'),
        os.path.join(processed_dir, 'Friday-WorkingHours-Morning.pcap_ISCX.csv'),
        os.path.join(processed_dir, 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv'),
        os.path.join(processed_dir, 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
    ] if os.path.exists(p)])

    if not csv_files:
        raise RuntimeError('No source CSVs available for stratified sampling')

    class_map = load_class_map(CLASS_MAP_PATH)
    normalized_map = { _normalize_text(k): v for k, v in class_map.items() }

    collected = {c: [] for c in target_classes}
    remaining = {c: per_class for c in target_classes}

    for p in csv_files:
        if all(v <= 0 for v in remaining.values()):
            break
        try:
            for chunk in pd.read_csv(p, chunksize=200000, low_memory=False):
                lbl_col = find_label_column(chunk.columns)
                if lbl_col is None:
                    continue
                s = chunk[lbl_col].astype(str).fillna('').map(_normalize_text)
                mapped = s.map(normalized_map)
                for cls in target_classes:
                    if remaining[cls] <= 0:
                        continue
                    mask = mapped == cls
                    if not mask.any():
                        continue
                    rows = chunk.loc[mask]
                    take = min(remaining[cls], len(rows))
                    collected[cls].append(rows.sample(n=take, random_state=random_state))
                    remaining[cls] -= take
                if all(v <= 0 for v in remaining.values()):
                    break
        except Exception:
            continue

    final_parts = []
    for cls in target_classes:
        if collected[cls]:
            final_parts.append(pd.concat(collected[cls], axis=0))
    if not final_parts:
        raise RuntimeError('Could not build stratified sample; no data collected')

    sample_df = pd.concat(final_parts, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return sample_df


def preprocess_for_tsne(df, label_col, class_map, top_features=None):
    # Determine features to use
    if top_features:
        use_cols = [c for c in top_features if c in df.columns]
        if not use_cols:
            # fallback numeric
            use_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        use_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if label_col in use_cols:
        use_cols.remove(label_col)

    if not use_cols:
        raise RuntimeError('No numeric feature columns found for t-SNE')

    X = df[use_cols].copy()
    y_raw = df[label_col].astype(str).fillna('')

    # Normalize label strings and map
    y_norm = y_raw.map(_normalize_text)
    y_mapped = y_norm.map(class_map)

    # If labels are already numeric 0/1/2, keep them
    if y_mapped.isna().all():
        # try convert directly
        try:
            y_numeric = pd.to_numeric(df[label_col], errors='coerce')
            if y_numeric.notna().any():
                y_mapped = y_numeric.astype(pd.Int64Dtype())
        except Exception:
            pass

    # Drop rows without a mapped label
    mask_valid = y_mapped.notna()
    if mask_valid.sum() == 0:
        raise RuntimeError('No mapped labels found after applying class_map')

    X = X[mask_valid.values]
    y = y_mapped[mask_valid].astype(int)

    # Handle NaNs/Infs in features
    # Replace inf with nan then drop rows with NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=0)
    # align y
    y = y.loc[X.index]

    # Scale
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X.values.astype(np.float32))

    return X_scaled, y.values, use_cols


def run_tsne(X, perplexity=30, n_iter=1000, random_state=42):
    import inspect
    print('Calculations may take a moment...')
    # Build kwargs compatibly for different sklearn versions (n_iter vs max_iter)
    sig = inspect.signature(TSNE)
    kwargs = dict(n_components=2, perplexity=perplexity, random_state=random_state, init='pca')
    if 'n_iter' in sig.parameters:
        kwargs['n_iter'] = n_iter
    elif 'max_iter' in sig.parameters:
        kwargs['max_iter'] = n_iter
    tsne = TSNE(**kwargs)
    Z = tsne.fit_transform(X)
    return Z


def plot_tsne(Z, y, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    dfz = pd.DataFrame({'x': Z[:, 0], 'y': Z[:, 1], 'class': y})
    dfz['class'] = dfz['class'].astype(int)

    palette = {0: 'blue', 1: 'red', 2: 'green'}
    label_names = {0: 'Normal', 1: 'Volumetric', 2: 'Infiltration'}

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=dfz, x='x', y='y', hue='class', palette=palette, legend='full', s=10, alpha=0.7)
    # Customize legend
    handles, labels = plt.gca().get_legend_handles_labels()
    # Replace numeric labels with names
    new_labels = []
    for lab in labels:
        try:
            k = int(lab)
            new_labels.append(label_names.get(k, str(k)))
        except Exception:
            new_labels.append(lab)
    plt.title('t-SNE Visualization of Network Traffic Classes')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(title='Class', labels=new_labels)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f'Saved t-SNE plot to: {out_path}')


def main(sample_n=5000):
    class_map = load_class_map(CLASS_MAP_PATH)
    df, label_col = load_sample(n=sample_n)
    print(f'Loaded sample with shape: {df.shape}, label column: {label_col}')

    # Check whether all three classes are present; if not, build a stratified sample
    normalized_map = { _normalize_text(k): v for k, v in class_map.items() }
    mapped = df[label_col].astype(str).fillna('').map(_normalize_text).map(normalized_map)
    present = set(mapped.dropna().unique())
    required = {0, 1, 2}
    if not required.issubset(present):
        print(f"[INFO] Initial sample missing classes {required - present}. Building stratified sample to include all classes...")
        per_class = max(1, sample_n // 3)
        try:
            df = stratified_sample_with_classes(target_classes=(0,1,2), per_class=per_class)
            label_col = find_label_column(df.columns)
            print(f"[INFO] Stratified sample built with shape: {df.shape}, label column: {label_col}")
        except Exception as e:
            print(f"[WARN] Stratified sampling failed: {e}. Proceeding with original sample.")

    X, y, used_features = preprocess_for_tsne(df, label_col, class_map, TOP_FEATURES)
    print(f'Using {len(used_features)} features for t-SNE')
    Z = run_tsne(X)
    plot_tsne(Z, y, OUT_FIG)

    print('\nInterpretation:')
    print('✅ Visual inspection required: Check if colors form distinct clusters.')
    # Print quick counts
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f'Class {u}: {c} points')


if __name__ == '__main__':
    main()
