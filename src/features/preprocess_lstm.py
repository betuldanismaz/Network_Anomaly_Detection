import os
import glob
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# --- CONFIG ---
DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data/processed_csv')
LSTM_OUT_DIR = os.path.join(os.path.dirname(__file__), '../../data/processed_lstm')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '../../models/scaler_lstm.pkl')
CLASS_MAP_PATH = os.path.join(os.path.dirname(__file__), '../utils/classes_map.json')

# Import top features from config
from src.config import TOP_FEATURES

WINDOW_SIZE = 10


# --- 1. LOAD & FILTER (ONLY SPECIFIED FILES) ---
SELECTED_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv"
]
frames = []
for fname in SELECTED_FILES:
    fpath = os.path.join(DATA_DIR, fname)
    if not os.path.exists(fpath):
        print(f"[WARN] File not found: {fpath}")
        continue
    df = pd.read_csv(fpath)
    keep_cols = [f for f in TOP_FEATURES if f in df.columns] + ['Label']
    df = df[keep_cols]
    frames.append(df)
if not frames:
    raise RuntimeError("No data loaded. Please check file paths.")
data = pd.concat(frames, axis=0, ignore_index=True)

# --- 2. MULTI-CLASS LABELING ---
with open(CLASS_MAP_PATH, 'r') as f:
    class_map = json.load(f)

def map_label(label):
    if label in class_map:
        return class_map[label]
    else:
        print(f"[WARN] Unknown label '{label}' encountered. Dropping row.")
        return -1

# Map labels
labels = data['Label'].apply(map_label)
mask = labels != -1
labels = labels[mask].astype(np.int32)
data = data.loc[mask, :].reset_index(drop=True)
data = data.drop(columns=['Label'])

# --- 3. NORMALIZATION (MinMax) ---
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.astype(np.float32))
joblib.dump(scaler, SCALER_PATH)

# --- 4. SLIDING WINDOW ---
def create_sequences(data, labels, window_size=10):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(labels.iloc[i+window_size])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

# --- 5. SPLITTING ---
num_samples = data_scaled.shape[0]
split_idx = int(num_samples * 0.8)

train_data = data_scaled[:split_idx]
train_labels = labels.iloc[:split_idx]
test_data = data_scaled[split_idx - WINDOW_SIZE:]
test_labels = labels.iloc[split_idx - WINDOW_SIZE:]

X_train, y_train = create_sequences(train_data, train_labels, WINDOW_SIZE)
X_test, y_test = create_sequences(test_data, test_labels, WINDOW_SIZE)

# --- 6. SAVING ---
os.makedirs(LSTM_OUT_DIR, exist_ok=True)
np.save(os.path.join(LSTM_OUT_DIR, 'X_train.npy'), X_train)
np.save(os.path.join(LSTM_OUT_DIR, 'y_train.npy'), y_train)
np.save(os.path.join(LSTM_OUT_DIR, 'X_test.npy'), X_test)
np.save(os.path.join(LSTM_OUT_DIR, 'y_test.npy'), y_test)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print("Preprocessing for BiLSTM complete.")
