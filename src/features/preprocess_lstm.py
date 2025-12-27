import os
import glob
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from collections import Counter


# --- CONFIG ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Allow overriding directories via environment variables if you moved folders
DATA_DIR = os.getenv('LSTM_DATA_DIR') or os.path.join(ROOT, 'data', 'original_csv')
OUT_DIR = os.getenv('LSTM_OUT_DIR') or os.path.join(ROOT, 'data', 'processed_lstm')
MODELS_DIR = os.getenv('LSTM_MODELS_DIR') or os.path.join(ROOT, 'models')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler_lstm.pkl')
CLASS_WEIGHTS_PATH = os.path.join(MODELS_DIR, 'class_weights.json')

# Resolve classes_map.json from several likely locations
possible_class_map_paths = [
    os.getenv('CLASSES_MAP_PATH'),
    os.path.join(ROOT, 'src', 'utils', 'classes_map.json'),
    os.path.join(ROOT, 'src', 'utils', 'classes_map.json'),
    os.path.join(ROOT, 'data', 'classes_map.json'),
    os.path.join(ROOT, 'classes_map.json'),
]
CLASS_MAP_PATH = None
for p in possible_class_map_paths:
    if not p:
        continue
    if os.path.exists(p):
        CLASS_MAP_PATH = p
        break
if CLASS_MAP_PATH is None:
    # final fallback to package-relative utils
    CLASS_MAP_PATH = os.path.join(os.path.dirname(__file__), '..', 'utils', 'classes_map.json')

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Import top features
from src.config import TOP_FEATURES


# Parameters
WINDOW_SIZE = 10
STRIDE = 1  # change to 5 to reduce dataset size


def _normalize_text(x):
    if x is None:
        return ''
    s = str(x)
    try:
        import unicodedata
        s = unicodedata.normalize('NFKC', s)
    except Exception:
        pass
    s = s.replace('\ufeff', '').replace('\u00A0', ' ').replace('\uFFFD', ' ')
    s = ' '.join(s.split())
    return s.strip()


def list_csv_files(data_dir):
    return sorted(glob.glob(os.path.join(data_dir, '*.csv')))


def read_selected_columns(file_path, features):
    # read only needed columns to save memory
    usecols = [c for c in features if c in pd.read_csv(file_path, nrows=0).columns]
    if 'Label' in pd.read_csv(file_path, nrows=0).columns:
        usecols = usecols + ['Label']
    if not usecols:
        return None
    return pd.read_csv(file_path, usecols=usecols)


def create_sequences(arr, labels, window_size=10, stride=1):
    # arr: numpy array (N, F), labels: 1D array-like (N,)
    X_list = []
    y_list = []
    N = arr.shape[0]
    for start in range(0, N - window_size + 1, stride):
        end = start + window_size
        X_list.append(arr[start:end])
        y_list.append(int(labels[end - 1]))
    if not X_list:
        return np.zeros((0, window_size, arr.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y


def compute_class_weights(y):
    # y is 1D array of integer labels
    counts = Counter(y.tolist())
    classes = sorted(counts.keys())
    total = sum(counts.values())
    n_classes = len(classes)
    weights = {}
    for c in classes:
        if counts[c] == 0:
            weights[c] = 1.0
        else:
            weights[c] = float(total) / (n_classes * counts[c])
    return weights


def main(window_size=WINDOW_SIZE, stride=STRIDE):
    # 1) Load files and keep only TOP_FEATURES + Label
    csv_files = list_csv_files(DATA_DIR)
    if not csv_files:
        raise RuntimeError(f'No CSV files found in {DATA_DIR}')

    frames = []
    for f in csv_files:
        try:
            cols = pd.read_csv(f, nrows=0).columns.tolist()
        except Exception:
            print(f'[WARN] Could not read header of {f}, skipping')
            continue
        keep = [c for c in TOP_FEATURES if c in cols]
        if 'Label' in cols:
            keep = keep + ['Label']
        if not keep:
            continue
        try:
            df = pd.read_csv(f, usecols=keep)
            frames.append(df)
        except Exception as e:
            print(f'[WARN] Error reading {f}: {e}')
            continue

    if not frames:
        raise RuntimeError('No data loaded after scanning CSVs')

    data = pd.concat(frames, axis=0, ignore_index=True)

    # 2) Map labels
    with open(CLASS_MAP_PATH, 'r') as cf:
        raw_map = json.load(cf)
    normalized_map = { _normalize_text(k): v for k, v in raw_map.items() }

    if 'Label' not in data.columns:
        raise RuntimeError('Label column not found in merged data')

    labels_raw = data['Label'].astype(str).fillna('').map(_normalize_text)
    labels_mapped = labels_raw.map(normalized_map)
    mask_valid = labels_mapped.notna()
    if mask_valid.sum() == 0:
        raise RuntimeError('No labels mapped to classes 0/1/2; check classes_map.json')

    # Drop unknowns
    data = data.loc[mask_valid].reset_index(drop=True)
    labels = labels_mapped.loc[mask_valid].astype(int).reset_index(drop=True)

    # 3) Feature selection - ensure order of TOP_FEATURES
    feature_cols = [c for c in TOP_FEATURES if c in data.columns]
    if not feature_cols:
        # fallback numeric columns except Label
        feature_cols = [c for c in data.select_dtypes(include=[np.number]).columns.tolist() if c != 'Label']

    X_df = data[feature_cols].copy()

    # 4) Time-based split (first 80% train, last 20% test)
    N = X_df.shape[0]
    split_idx = int(N * 0.8)
    X_train_df = X_df.iloc[:split_idx].reset_index(drop=True)
    y_train = labels.iloc[:split_idx].reset_index(drop=True)
    X_test_df = X_df.iloc[split_idx:].reset_index(drop=True)
    y_test = labels.iloc[split_idx:].reset_index(drop=True)

    # 5) Scaling: fit on train only
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_df.astype(np.float32))
    X_test_scaled = scaler.transform(X_test_df.astype(np.float32))
    joblib.dump(scaler, SCALER_PATH)

    # 6) Sliding window -> sequences
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train.values, window_size=window_size, stride=stride)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test.values, window_size=window_size, stride=stride)

    # 7) Class weights (based on training sequences)
    if y_train_seq.size > 0:
        class_weights = compute_class_weights(y_train_seq)
    else:
        class_weights = {}
    # Ensure classes 0,1,2 exist in dict
    for c in [0,1,2]:
        class_weights.setdefault(c, 1.0)

    # Save class weights
    with open(CLASS_WEIGHTS_PATH, 'w') as wf:
        json.dump(class_weights, wf, indent=2)

    # 8) Save outputs
    np.save(os.path.join(OUT_DIR, 'X_train.npy'), X_train_seq.astype(np.float32))
    np.save(os.path.join(OUT_DIR, 'y_train.npy'), y_train_seq.astype(np.int32))
    np.save(os.path.join(OUT_DIR, 'X_test.npy'), X_test_seq.astype(np.float32))
    np.save(os.path.join(OUT_DIR, 'y_test.npy'), y_test_seq.astype(np.int32))

    print(f'X_train shape: {X_train_seq.shape}')
    print(f'y_train distribution: {Counter(y_train_seq)}')
    print(f'X_test shape: {X_test_seq.shape}')
    print(f'y_test distribution: {Counter(y_test_seq)}')
    print(f'Scaler saved to: {SCALER_PATH}')
    print(f'Class weights saved to: {CLASS_WEIGHTS_PATH}')


if __name__ == '__main__':
    main()
