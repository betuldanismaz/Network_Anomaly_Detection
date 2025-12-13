# Testing Checklist - Live Bridge Refactoring

## Pre-Flight Checklist

### ✅ 1. Model Files Verification

- [ ] `models/rf_model_optimized.pkl` exists
- [ ] `models/scaler.pkl` exists
- [ ] `models/threshold.txt` exists (contains float value like `0.35`)
- [ ] Models trained on same Top 20 features

**Verification Command:**

```powershell
Get-ChildItem models/ | Select-Object Name, Length, LastWriteTime
Get-Content models/threshold.txt
```

### ✅ 2. Configuration Files

- [ ] `src/config.py` exists with `TOP_FEATURES` list
- [ ] `.env` file has `NETWORK_INTERFACE` and `WHITELIST_IPS`
- [ ] `requirements.txt` updated

**Create config.py if missing:**

```python
# src/config.py
TOP_FEATURES = [
    "Flow Bytes/s",
    "Bwd Packet Length Mean",
    "Flow IAT Mean",
    "Fwd Packet Length Mean",
    "Flow Duration",
    "Fwd IAT Mean",
    "Bwd IAT Mean",
    "Flow IAT Std",
    "Fwd Packet Length Max",
    "Bwd Packet Length Max",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Header Length",
    "Bwd Header Length",
    "Flow Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
]
```

**Create .env if missing:**

```bash
NETWORK_INTERFACE=Wi-Fi
WHITELIST_IPS=192.168.1.1,127.0.0.1,8.8.8.8,localhost
```

### ✅ 3. Dependencies Installation

```powershell
# Check Python version (3.8+)
python --version

# Install requirements
pip install -r requirements.txt

# Verify critical packages
pip list | Select-String "scapy|pandas|joblib|numpy|cicflowmeter"
```

**Expected Output:**

```
cicflowmeter       0.1.8
joblib            1.3.2
numpy             1.24.3
pandas            2.0.3
scapy             2.5.0
```

### ✅ 4. Network Interface Detection

```powershell
# Test interface detection
python -c "from scapy.all import conf; print([i.name for i in conf.ifaces.values() if i.ip and i.ip != '127.0.0.1'])"
```

**Expected:** `['Wi-Fi', 'Ethernet', ...]`

### ✅ 5. CICFlowMeter Test

```powershell
# Check CLI availability
python -m cicflowmeter --help

# Test on sample PCAP
python -m cicflowmeter -f test.pcap -c test_output.csv
```

---

## Unit Testing

### Test 1: Model Loading

```python
# test_model_loading.py
import joblib
import os

MODEL_PATH = "models/rf_model_optimized.pkl"
SCALER_PATH = "models/scaler.pkl"
THRESHOLD_PATH = "models/threshold.txt"

# Test model exists and loads
assert os.path.exists(MODEL_PATH), "Model not found"
model = joblib.load(MODEL_PATH)
print(f"✅ Model loaded: {type(model)}")

# Test scaler
assert os.path.exists(SCALER_PATH), "Scaler not found"
scaler = joblib.load(SCALER_PATH)
print(f"✅ Scaler loaded: {type(scaler)}")

# Test threshold
assert os.path.exists(THRESHOLD_PATH), "Threshold file not found"
with open(THRESHOLD_PATH, 'r') as f:
    threshold = float(f.read().strip())
assert 0.0 <= threshold <= 1.0, "Invalid threshold"
print(f"✅ Threshold: {threshold}")
```

**Run:** `python test_model_loading.py`

### Test 2: TOP_FEATURES Import

```python
# test_config.py
try:
    from src.config import TOP_FEATURES
    print(f"✅ TOP_FEATURES loaded: {len(TOP_FEATURES)} features")
    assert len(TOP_FEATURES) == 20, "Expected 20 features"
except ImportError:
    print("⚠️ config.py not found, using fallback")
```

**Run:** `python test_config.py`

### Test 3: Feature Alignment

```python
# test_feature_alignment.py
import pandas as pd
from src.live_bridge import prepare_feature_frame, EXPECTED_FEATURES

# Create dummy CICFlowMeter output
dummy_data = {
    'Flow Duration': [100.0],
    'Total Fwd Packets': [10],
    'Total Backward Packets': [5],
    # ... add more CIC columns
}
df = pd.DataFrame(dummy_data)

# Test alignment
aligned = prepare_feature_frame(df)
print(f"✅ Aligned shape: {aligned.shape}")
assert aligned.shape[1] == 78, f"Expected 78 columns, got {aligned.shape[1]}"
assert list(aligned.columns) == EXPECTED_FEATURES, "Column order mismatch"
```

**Run:** `python test_feature_alignment.py`

### Test 4: Prediction Pipeline

```python
# test_prediction.py
import numpy as np
import pandas as pd
from src.live_bridge import LiveDetector

detector = LiveDetector()

# Create dummy features (20 top features)
dummy_features = pd.DataFrame(
    np.random.randn(5, 20),
    columns=detector.top_features
)

# Test prediction
predictions, probabilities = detector.process_and_predict(dummy_features)

print(f"✅ Predictions: {predictions}")
print(f"✅ Probabilities: {probabilities}")
assert len(predictions) == 5, "Wrong prediction count"
assert len(probabilities) == 5, "Wrong probability count"
assert all(p in [0, 1] for p in predictions), "Invalid predictions"
```

**Run:** `python test_prediction.py`

---

## Integration Testing

### Test 5: Full Pipeline (Dry Run)

```python
# test_full_pipeline.py
import os
from scapy.all import sniff, wrpcap

# 1. Capture 10 packets
print("📡 Capturing packets...")
packets = sniff(iface="Wi-Fi", count=10, timeout=5)
wrpcap("test_pipeline.pcap", packets)
print(f"✅ Captured {len(packets)} packets")

# 2. Run CICFlowMeter
from src.live_bridge import run_cicflowmeter_cli
success, error = run_cicflowmeter_cli("test_pipeline.pcap", "test_pipeline.csv")
assert success, f"CICFlowMeter failed: {error}"
print("✅ CICFlowMeter succeeded")

# 3. Test feature alignment
import pandas as pd
from src.live_bridge import prepare_feature_frame
df = pd.read_csv("test_pipeline.csv")
aligned = prepare_feature_frame(df)
print(f"✅ Aligned features: {aligned.shape}")

# 4. Test prediction
from src.live_bridge import LiveDetector
detector = LiveDetector()
predictions, probs = detector.process_and_predict(aligned)
print(f"✅ Predictions: {predictions}")
print(f"✅ Probabilities: {probs}")

# 5. Test logging
detector.log(aligned, predictions, probs)
print("✅ Logging succeeded")

# Cleanup
detector.shutdown()
os.remove("test_pipeline.pcap")
os.remove("test_pipeline.csv")
```

**Run:** `python test_full_pipeline.py`

### Test 6: Live Capture (1 Minute)

```powershell
# Run live bridge for 1 minute
timeout /t 60 python src/live_bridge.py
```

**Expected Console Output:**

```
🛡️  AI NETWORK IPS - OPTIMIZED MODEL
======================================================================
✅ Model loaded: models/rf_model_optimized.pkl
✅ Scaler loaded: models/scaler.pkl
✅ Threshold: 0.35 (loaded from threshold.txt)
✅ CSV writer initialized: data/live_captured_traffic.csv

🛡️  SİSTEM BAŞLATILDI | Arayüz: Wi-Fi

📡 Ağ Dinleniyor: Wi-Fi
⏹️  Durdurmak için CTRL+C yapın...

⏳ Paket toplanıyor...
   ↳ ⚙️ Analiz...
✅ [14:32:15] Trafik Temiz - Güvenli (5 Akış)
```

**Verify CSV Output:**

```powershell
# Check CSV was created
Get-Item data/live_captured_traffic.csv | Select-Object Length, LastWriteTime

# Check column count (should be 22: Timestamp + 20 features + Label + Confidence)
(Get-Content data/live_captured_traffic.csv -First 1 -Delimiter ",").Count
```

### Test 7: Attack Simulation

```python
# test_attack_simulation.py
import pandas as pd
from src.live_bridge import LiveDetector

detector = LiveDetector()

# Create attack-like features (high traffic, unusual patterns)
attack_features = pd.DataFrame({
    'Flow Bytes/s': [1000000.0],  # Very high
    'Bwd Packet Length Mean': [5000.0],  # Large packets
    'Flow IAT Mean': [0.01],  # Very fast
    'Fwd Packet Length Mean': [5000.0],
    'Flow Duration': [100.0],
    # ... fill remaining 15 features with attack-like values
})

# Pad with zeros for missing features
for col in detector.top_features:
    if col not in attack_features.columns:
        attack_features[col] = 0

# Reorder columns
attack_features = attack_features[detector.top_features]

# Predict
predictions, probs = detector.process_and_predict(attack_features)

print(f"Prediction: {predictions[0]} (1=ATTACK, 0=NORMAL)")
print(f"Confidence: {probs[0]:.4f}")
print(f"Threshold: {detector.threshold}")

if predictions[0] == 1:
    print("✅ Attack correctly detected!")
else:
    print("⚠️ Expected attack, got normal (threshold may be too high)")
```

**Run:** `python test_attack_simulation.py`

---

## Performance Testing

### Test 8: Throughput Benchmark

```python
# test_throughput.py
import time
import pandas as pd
import numpy as np
from src.live_bridge import LiveDetector

detector = LiveDetector()

# Generate 1000 random samples
features = pd.DataFrame(
    np.random.randn(1000, 20),
    columns=detector.top_features
)

# Benchmark
start = time.time()
predictions, probs = detector.process_and_predict(features)
elapsed = time.time() - start

print(f"✅ Processed 1000 samples in {elapsed:.4f} seconds")
print(f"✅ Throughput: {1000/elapsed:.2f} samples/sec")
print(f"✅ Latency: {elapsed/1000*1000:.2f} ms/sample")
```

**Expected:** `> 500 samples/sec` (2 ms/sample)

### Test 9: Memory Usage

```python
# test_memory.py
import tracemalloc
from src.live_bridge import LiveDetector

tracemalloc.start()

# Initialize detector
detector = LiveDetector()

current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"✅ Current memory: {current / 1024 / 1024:.2f} MB")
print(f"✅ Peak memory: {peak / 1024 / 1024:.2f} MB")
```

**Expected:** `< 500 MB`

---

## Professor Verification Test

### Test 10: Wireshark Comparison

**Goal:** Capture same traffic with both Live Bridge and Wireshark, verify feature extraction accuracy.

**Steps:**

1. Start Wireshark capture on Wi-Fi interface
2. Start Live Bridge: `python src/live_bridge.py`
3. Generate traffic: `curl https://example.com` (repeat 10 times)
4. Stop both captures (Ctrl+C)
5. Export Wireshark: File → Export Packet Dissections → CSV
6. Compare:
   - Packet count: Wireshark vs `temp_live.csv`
   - IP addresses: Match source/destination IPs
   - Byte counts: Compare "Total Length of Fwd Packets"
   - Timing: Compare "Flow Duration"

**Acceptance Criteria:**

- ✅ Packet count matches ±5%
- ✅ IP addresses identical
- ✅ Byte counts match ±10% (CICFlowMeter aggregation)
- ✅ Flow durations match ±15%

### Test 11: Data Harvest Validation

```python
# test_harvest_validation.py
import pandas as pd

# Load harvested CSV
df = pd.read_csv("data/live_captured_traffic.csv")

# Validate schema
assert 'Timestamp' in df.columns, "Missing Timestamp"
assert 'Predicted_Label' in df.columns, "Missing label"
assert 'Confidence_Score' in df.columns, "Missing confidence"

# Count feature columns (should be 20)
feature_cols = [c for c in df.columns if c not in ['Timestamp', 'Predicted_Label', 'Confidence_Score']]
assert len(feature_cols) == 20, f"Expected 20 features, got {len(feature_cols)}"

# Check data types
assert df['Predicted_Label'].dtype == int, "Label should be int"
assert df['Confidence_Score'].dtype == float, "Confidence should be float"

# Check value ranges
assert df['Predicted_Label'].isin([0, 1]).all(), "Invalid label values"
assert (df['Confidence_Score'] >= 0).all() and (df['Confidence_Score'] <= 1).all(), "Invalid confidence"

print(f"✅ CSV validation passed")
print(f"✅ Total rows: {len(df)}")
print(f"✅ Attack ratio: {df['Predicted_Label'].mean():.2%}")
```

**Run:** `python test_harvest_validation.py`

---

## Regression Testing

### Test 12: Compare Old vs New Predictions

```python
# test_regression.py
import joblib
import pandas as pd
import numpy as np

# Load old model (if available)
old_model = joblib.load("models/rf_model_v1.pkl")  # 78 features
new_model = joblib.load("models/rf_model_optimized.pkl")  # 20 features

# Generate test data (78 features)
test_data_78 = pd.read_csv("data/ready_splits/test.csv").drop(columns=['Label'])
test_labels = pd.read_csv("data/ready_splits/test.csv")['Label']

# Old predictions (all 78 features)
old_predictions = old_model.predict(test_data_78)

# New predictions (Top 20 features)
from src.config import TOP_FEATURES
test_data_20 = test_data_78[TOP_FEATURES]
scaler = joblib.load("models/scaler.pkl")
test_scaled = scaler.transform(test_data_20)
new_probs = new_model.predict_proba(test_scaled)[:, 1]
new_predictions = (new_probs >= 0.35).astype(int)  # Use threshold

# Compare accuracy
from sklearn.metrics import accuracy_score, recall_score, precision_score

old_acc = accuracy_score(test_labels, old_predictions)
new_acc = accuracy_score(test_labels, new_predictions)

print(f"Old Model (78 features): Accuracy={old_acc:.4f}")
print(f"New Model (20 features): Accuracy={new_acc:.4f}")
print(f"Difference: {new_acc - old_acc:+.4f}")

# Acceptable if difference < 5%
assert abs(new_acc - old_acc) < 0.05, "Accuracy degradation > 5%"
print("✅ Regression test passed")
```

---

## Cleanup Checklist

After testing:

- [ ] Delete temporary PCAP files: `rm temp_live.*`
- [ ] Archive test CSV: `mv test_*.csv tests/`
- [ ] Review harvested data: `head data/live_captured_traffic.csv`
- [ ] Backup database: `cp attacks.db attacks_backup_$(date +%Y%m%d).db`

---

## Final Checklist

Before production deployment:

- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Performance benchmarks meet targets
- [ ] Professor verification complete
- [ ] Regression tests show <5% accuracy change
- [ ] CSV schema validated
- [ ] Wireshark logs match
- [ ] No syntax/import errors
- [ ] Documentation updated
- [ ] `.env` file configured

**Sign-off Date:** ******\_\_\_\_******  
**Tested By:** ******\_\_\_\_******  
**Status:** ☐ PASS ☐ FAIL (explain below)

---

**Notes:**
