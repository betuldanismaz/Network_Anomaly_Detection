# 🛡️ Network Intrusion Detection System (NIDS)

A production-ready Network Intrusion Detection System featuring **multiple machine learning architectures** for comprehensive threat detection: Random Forest for real-time binary classification, Decision Tree for interpretable analysis, and BiLSTM for temporal pattern analysis with 3-class attack categorization.

**🎯 Key Capabilities:** Real-time Detection | Automated Firewall Response | Live Monitoring Dashboard | Multi-Model Architecture

---

## 📊 Quick Stats

| Model             | Task                                 | Accuracy | Precision | Recall | Use Case               |
| ----------------- | ------------------------------------ | -------- | --------- | ------ | ---------------------- |
| **Random Forest** | Binary (Normal/Attack)               | 99.73%   | 97.87%    | 99.90% | Real-time IPS          |
| **Decision Tree** | Binary (Normal/Attack)               | 99.60%   | 99.61%    | 98.08% | Interpretable Analysis |
| **BiLSTM**        | 3-Class (Benign/Volumetric/Semantic) | 97.73%   | 97.87%    | 97.73% | Temporal Analysis      |
| **LSTM**          | 3-Class (Benign/Volumetric/Semantic) | 98.15%   | 98.18%    | 98.15% | Lightweight Temporal   |

---

## 📑 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Development](#-development)
- [Dependencies](#-dependencies)

---

## ✨ Features

### 🚀 Core Capabilities

| Feature                           | Description                                                  |
| --------------------------------- | ------------------------------------------------------------ |
| **Dual-Model Architecture**       | Random Forest (real-time) + BiLSTM/LSTM (temporal analysis)  |
| **Real-Time Detection**           | 4-second packet capture windows with immediate analysis      |
| **3-Class Attack Classification** | Benign, Volumetric (DDoS), Semantic (Port Scan, Web Attacks) |
| **Automated Firewall Response**   | OS-level IP blocking (Windows/Linux)                         |
| **Live Dashboard**                | Streamlit-based monitoring with real-time statistics         |
| **Data Harvesting**               | Continuous learning pipeline for model retraining            |
| **Top 20 Features**               | Optimized feature selection for 3x faster inference          |
| **Custom Thresholding**           | Security-first decision boundaries (0.1077 for RF)           |

### 🎯 What Makes This Special


1. **Security-First ML Design**
   - **99.90% Recall** → Only 0.1% of attacks slip through
   - **Low False Positive Rate** → 97.87% precision prevents alarm fatigue
   - **Custom Threshold Optimization** → Prioritizes catching attacks over reducing false alarms

2. **Production-Ready Architecture**
   - Thread-safe buffered writes (no data loss)
   - Graceful shutdown with data persistence
   - Fallback mechanisms for feature extraction (CLI + API modes)
   - # Comprehensive error handling and logging
     **1. Multi-Model Approach**

- **Random Forest**: Fast binary classification (6-9s latency) for immediate threat response
  | **Decision Tree** | Highly interpretable model for understanding decision logic
  | **BiLSTM/LSTM** | Deep temporal analysis for sophisticated attack pattern recognition
  | **Complementary Strengths** | Speed + Interpretability + Accuracy combined

**2. Production-Ready Design**

- Thread-safe buffered writes
- Graceful shutdown with data persistence
- Comprehensive error handling and logging
- Memory-efficient batch processing for BiLSTM

**3. Advanced Attack Classification**

- **Benign** (Class 0): Normal network traffic
- **Volumetric** (Class 1): DDoS, DoS, Botnet attacks
- **Semantic** (Class 2): Port Scanning, Web Attacks, Brute Force, Infiltration

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    NETWORK TRAFFIC INPUT                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   Scapy Capture      │
              │   (4-second windows) │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  CICFlowMeter        │
              │  (78 Features)       │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Feature Selection   │
              │  (Top 20 Features)   │
              └──────────┬───────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
┌─────────────────┐           ┌─────────────────────┐
│  Random Forest  │           │      BiLSTM         │
│  Binary (0/1)   │           │  3-Class (0/1/2)    │
│  Real-time IPS  │           │  Temporal Analysis  │
└────────┬────────┘           └──────────┬──────────┘
         │                               │
         ▼                               ▼
┌─────────────────┐           ┌─────────────────────┐
│ Firewall Block  │           │  Pattern Reports    │
│ SQLite Alerts   │           │  Confusion Matrix   │
│ Live Dashboard  │           │  Classification     │
└─────────────────┘           └─────────────────────┘
```

### Model Comparison

| Aspect            | Random Forest             | BiLSTM                                            |
| ----------------- | ------------------------- | ------------------------------------------------- |
| **Input**         | Single flow (20 features) | Sequence of 10 flows (20 features × 10 timesteps) |
| **Output**        | Binary (Normal/Attack)    | 3-Class (Benign/Volumetric/Semantic)              |
| **Latency**       | 6-9 seconds               | Batch processing                                  |
| **Use Case**      | Real-time blocking        | Offline analysis, pattern detection               |
| **Training Time** | ~15 minutes               | ~2-3 hours (50 epochs)                            |
| **Model Size**    | 16.7 MB                   | 3.9 MB (BiLSTM) / ~2 MB (LSTM)                    |
| **Preprocessing** | `preprocess.py`           | `preprocess_lstm.py`                              |
| **Scaler**        | `scaler.pkl`              | `scaler_lstm.pkl`                                 |

### Component Breakdown

| Component                | Technology                  | Purpose                                    |
| ------------------------ | --------------------------- | ------------------------------------------ |
| **Packet Capture**       | Scapy                       | Raw packet sniffing from network interface |
| **Feature Extraction**   | CICFlowMeter (Java)         | 78 bidirectional flow features             |
| **Preprocessing**        | Pandas + scikit-learn       | Scaling, feature selection, sequencing     |
| **RF Model**             | scikit-learn RandomForest   | Binary classification (75 estimators)      |
| **BiLSTM/LSTM Model**    | TensorFlow/Keras            | 3-class sequential classification          |
| **Firewall Integration** | Windows Firewall / iptables | OS-level IP blocking                       |
| **Database**             | SQLite                      | Attack event logging                       |
| **Dashboard**            | Streamlit                   | Real-time monitoring UI                    |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (tested on 3.9, 3.10, 3.11)
- **Admin/Root privileges** (for packet capture and firewall)
- **Java 11+** (for CICFlowMeter)
- **Windows 10/11 or Linux** (Ubuntu 20.04+)

### Installation

**Step 1: Clone Repository**

```bash
git clone https://github.com/betuldanismaz/Network_Anomaly_Detection.git
cd Network_Anomaly_Detection/networkdetection
```

**Step 2: Create Virtual Environment**

```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

**Step 3: Install Dependencies**

```bash
pip install -r requirements.txt
```

**Step 4: Configure Environment**

```bash
# Copy template
cp .env.example .env

# Edit .env with your settings (see Configuration section)
```

### Running the System

**Option 1: Real-Time IPS (Random Forest)**

```bash
# Terminal 1: Start IPS engine
python src/live_bridge.py

# Terminal 2: Launch dashboard
streamlit run src/dashboard/app.py
```


**Option 2: BiLSTM Training & Evaluation**

```bash
# Step 1: Preprocess data for LSTM
python src/features/preprocess_lstm.py

# Step 2: Train BiLSTM model
python src/models/train_bilstm.py

# Step 3: Evaluate model
python src/models/evaluate_bilstm.py

# Option 3: Standard LSTM Training
python src/models/train_lstm.py
python src/models/evaluate_lstm.py
```

---

## 📊 Model Performance

### Random Forest (Binary Classification)

**Training Dataset:** CICIDS 2017 (2,830,743 flows)

| Metric                  | Value  | Interpretation                                   |
| ----------------------- | ------ | ------------------------------------------------ |
| **Accuracy**            | 99.73% | Overall correctness                              |
| **Precision**           | 97.87% | When predicting "attack", correct 97.87% of time |
| **Recall**              | 99.90% | Catches 99.9% of all actual attacks              |
| **F1-Score**            | 98.88% | Harmonic mean of precision & recall              |
| **ROC-AUC**             | 0.9994 | Near-perfect discriminative ability              |
| **False Negative Rate** | 0.10%  | Only 1 in 1000 attacks missed                    |

**Confusion Matrix:**

```
                Predicted
              Normal  Attack
Actual Normal  98.7%   1.3%   ← FP: 2.13%
      Attack   0.1%   99.9%   ← FN: 0.10%
```

**Top 5 Features (by Importance):**

1. Bwd Packet Length Std (14.2%)
2. Packet Length Variance (11.8%)
3. Subflow Fwd Bytes (9.3%)
4. Total Length of Fwd Packets (7.6%)
5. Flow Bytes/s (6.4%)

### Decision Tree (Binary Classification)

**Training Dataset:** CICIDS 2017 (2,830,743 flows)

| Metric                  | Value  | Interpretation                                   |
| ----------------------- | ------ | ------------------------------------------------ |
| **Accuracy**            | 99.60% | Overall correctness                              |
| **Precision**           | 99.61% | When predicting "attack", correct 99.61% of time |
| **Recall**              | 98.08% | Catches 98.08% of all actual attacks             |
| **F1-Score**            | 98.84% | Harmonic mean of precision & recall              |
| **False Negative Rate** | 1.92%  | 936 attacks missed out of 48,877                 |

**Top Feature:** `Bwd Packet Length Std` (71.37% importance)

**Configuration:**

- Max Depth: 10 levels
- Total Nodes: 433
- Criterion: Gini impurity
- Random State: 42 (reproducible)

**Visualizations:**

- Tree structure diagram (top 4 levels, 300 DPI)
- Feature importance chart (top 10 features)
- Confusion matrix heatmap
- Decision rules export (text format)

### BiLSTM (3-Class Classification)

**Architecture (BiLSTM):**

- Input: (batch_size, 10 timesteps, 20 features)
- BiLSTM Layer 1: 128 units + BatchNorm + Dropout(0.3)
- BiLSTM Layer 2: 64 units + BatchNorm + Dropout(0.3)
- Dense: 32 units (ReLU) + Dropout(0.3)
- Output: 3 units (Softmax)

**Architecture (LSTM):**

- Similar structure but uses unidirectional LSTM layers for lower latency.
- Optimized for resource-constrained environments.

**Training Configuration:**

- Epochs: 50 (with early stopping)
- Batch Size: 256
- Optimizer: Adam (lr=0.001)
- Loss: Sparse Categorical Crossentropy
- Class Weights: Balanced (computed from training data)

**Class Mapping:**

- **Class 0 (Benign)**: Normal traffic
- **Class 1 (Volumetric)**: DDoS, DoS, Botnet
- **Class 2 (Semantic)**: PortScan, Web Attack, Brute Force, Infiltration

**Performance:** ~98%+ accuracy on test set (see `reports/bilstm/final_classification_report.txt`)

---

## 📁 Project Structure

```
networkdetection/
├── 📂 data/
│   ├── original_csv/              # Raw CICIDS 2017 dataset
│   ├── processed_csv/             # Preprocessed for Random Forest
│   │   └── ready_splits/          # Train/Val/Test splits
│   ├── processed_lstm/            # Preprocessed for BiLSTM
│   │   ├── X_train.npy           # Training sequences (N, 10, 20)
│   │   ├── y_train.npy           # Training labels
│   │   ├── X_test.npy            # Test sequences
│   │   └── y_test.npy            # Test labels
│   └── live_captured_traffic.csv  # Data harvesting output
│
├── 📂 models/
│   ├── rf_optimized_model.pkl     # Random Forest (75 estimators)
│   ├── scaler.pkl                 # MinMaxScaler for RF
│   ├── threshold.txt              # RF decision boundary (0.1077)
│   ├── threshold_config.json      # RF threshold metadata
│   ├── dt_model.pkl               # Trained Decision Tree model
│   ├── dt_rules.txt               # Decision Tree rules (text)
│   ├── bilstm_best.keras          # Trained BiLSTM model
│   ├── lstm_best.keras            # Trained LSTM model
│   ├── scaler_lstm.pkl            # MinMaxScaler for BiLSTM/LSTM
│   └── class_weights.json         # BiLSTM class weights
│
├── 📂 src/
│   ├── live_bridge.py             # 🚀 Real-time IPS engine (RF)
│   ├── config.py                  # Top 20 feature definitions
│   │
│   ├── 📁 capture/
│   │   └── sniffer.py             # Scapy packet capture
│   │
│   ├── 📁 dashboard/
│   │   └── app.py                 # Streamlit monitoring dashboard
│   │
│   ├── 📁 features/
│   │   ├── preprocess.py          # RF preprocessing pipeline
│   │   ├── preprocess_lstm.py     # BiLSTM preprocessing (sequences)
│   │   └── resplit_data.py        # Data splitting utility
│   │
│   ├── 📁 models/
│   │   ├── train_randomforest.py  # RF training script
│   │   ├── train_dt.py            # Decision Tree training script
│   │   ├── train_bilstm.py        # BiLSTM training script
│   │   ├── train_lstm.py          # LSTM training script
│   │   ├── evaluate_bilstm.py     # BiLSTM evaluation script
│   │   ├── evaluate_lstm.py       # LSTM evaluation script
│   │   ├── analyze_results.py     # RF model evaluation
│   │   ├── analyze_thresholds.py  # Threshold optimization & risk scoring
│   │   ├── stress_test.py         # Performance benchmarking
│   │   └── top_20_features.json   # Feature importance rankings
│   │
│   └── 📁 utils/
│       ├── db_manager.py          # SQLite operations
│       ├── firewall_manager.py    # Firewall integration
│       ├── visualize_dt.py        # Decision Tree visualizations
│       ├── data_audit.py          # RF data quality checks
│       ├── data_audit_lstm.py     # BiLSTM data quality checks
│       ├── model_optimizer.py     # Threshold optimization
│       ├── xai_engine.py          # Explainable AI utilities
│       └── inspect_csv.py         # CSV inspection helper
│
├── 📂 reports/
│   ├── figures/                   # RF visualizations
│   ├── decisiontree/              # Decision Tree visualizations
│   │   ├── dt_structure_top4_levels.png
│   │   ├── dt_feature_importance.png
│   │   ├── dt_confusion_matrix.png
│   │   └── text_exports/decision_tree_rules.txt
│   └── bilstm/                    # BiLSTM evaluation reports
│       ├── final_classification_report.txt
│       ├── final_confusion_matrix.png
│       └── training_history/
│
├── 📂 test/
│   ├── attack_test.py             # Simulated attack scenarios
│   └── check_interfaces.py        # Network interface detector
│
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment template
└── README.md                      # This file
```

---

## 💻 Usage

### Random Forest: Real-Time IPS

**Start Monitoring:**

```bash
python src/live_bridge.py
```

**Console Output Example:**

```
======================================================================
🔧 LIVE DETECTOR INITIALIZATION
======================================================================
✅ Model Loaded: models/rf_optimized_model.pkl
✅ Scaler Loaded: models/scaler.pkl
✅ Threshold Loaded: 0.1077
✅ Top Features: 20 columns
======================================================================

[2026-01-02 18:30:15] ✅ NORMAL
  Src: 192.168.1.105 → Dst: 8.8.8.8
  Confidence: 0.0234

[2026-01-02 18:30:20] 🚨 ATTACK DETECTED!
  Src: 203.0.113.45 → Dst: 192.168.1.100
  Confidence: 0.9876
  🚫 IP blocked: 203.0.113.45
```

### BiLSTM: Training & Evaluation

**Step 1: Preprocess Data**

```bash
python src/features/preprocess_lstm.py
```

This creates:

- `data/processed_lstm/X_train.npy` - Training sequences (N, 10, 20)
- `data/processed_lstm/y_train.npy` - Training labels
- `data/processed_lstm/X_test.npy` - Test sequences
- `data/processed_lstm/y_test.npy` - Test labels
- `models/scaler_lstm.pkl` - Fitted scaler
- `models/class_weights.json` - Class weights

**Step 2: Train Model**

```bash
python src/models/train_bilstm.py
```

Output:

- `models/bilstm_best.keras` - Best model checkpoint
- `reports/bilstm/training_history/` - Loss/accuracy plots

**Step 3: Evaluate Model**

```bash
python src/models/evaluate_bilstm.py
```

Generates:

- `reports/bilstm/final_classification_report.txt`
- `reports/bilstm/final_confusion_matrix.png`

**Example Classification Report:**

```
              precision    recall  f1-score   support

      Benign     0.9850    0.9920    0.9885    150000
  Volumetric     0.9780    0.9750    0.9765     80000
    Semantic     0.9810    0.9790    0.9800     70000

    accuracy                         0.9830    300000
```

### Dashboard Usage

**Launch Dashboard:**

```bash
streamlit run src/dashboard/app.py
```

**Features:**

- 📊 Real-time metrics (total events, blocked IPs)
- 📈 Attack timeline chart
- 🥧 Action distribution (blocked vs allowed)
- 📋 Filterable event log
- 🔓 IP unblock interface

---

## ⚙️ Configuration

### Environment Variables (`.env`)

```env
# Network Interface
NETWORK_INTERFACE=Wi-Fi
# Find yours: python test/check_interfaces.py

# IP Whitelist (comma-separated, no spaces)
WHITELIST_IPS=192.168.1.1,127.0.0.1,8.8.8.8

# Random Forest Threshold (0.0 - 1.0)
THRESHOLD=0.10774313582858071
# Lower = more sensitive, Higher = fewer false alarms
```

### Random Forest Threshold Tuning

| Threshold  | Recall | Precision | Use Case                     |
| ---------- | ------ | --------- | ---------------------------- |
| **0.05**   | 99.95% | 92%       | High-security (banks, gov't) |
| **0.1077** | 99.90% | 97.87%    | **Recommended (current)**    |
| **0.20**   | 99.5%  | 99%       | Balanced production          |
| **0.50**   | 95%    | 99.8%     | Low false alarm priority     |

**Change Threshold:**

1. Edit `models/threshold.txt` → Change value
2. Or edit `.env` → `THRESHOLD=0.20`
3. Restart `live_bridge.py`

### BiLSTM Hyperparameters

Edit `src/models/train_bilstm.py`:

```python
EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 0.001
LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
DROPOUT_RATE = 0.3
WINDOW_SIZE = 10  # Sequence length
```

### Class Mapping (BiLSTM)

Edit `src/utils/classes_map.json`:

```json
{
  "BENIGN": 0,
  "DDoS": 1,
  "DoS": 1,
  "Bot": 1,
  "PortScan": 2,
  "Web Attack": 2,
  "Brute Force": 2,
  "Infiltration": 2
}
```

---

## 🛠️ Development

### Training Random Forest

```bash
python src/models/train_randomforest.py
```

Outputs:

- `models/rf_optimized_model.pkl`
- `models/scaler.pkl`
- `models/threshold.txt`
- `reports/figures/confusion_matrix.png`

### Training Decision Tree

```bash
# Train the model
python src/models/train_dt.py

# Generate visualizations
python src/utils/visualize_dt.py
```

Outputs:

- `models/dt_model.pkl` - Trained Decision Tree model
- `models/dt_rules.txt` - Human-readable decision rules
- `reports/decisiontree/dt_structure_top4_levels.png` - Tree diagram
- `reports/decisiontree/dt_feature_importance.png` - Feature importance chart
- `reports/decisiontree/dt_confusion_matrix.png` - Confusion matrix heatmap

### Training BiLSTM

```bash
# 1. Preprocess data
python src/features/preprocess_lstm.py

# 2. Train model
python src/models/train_bilstm.py

# 3. Evaluate
python src/models/evaluate_bilstm.py
```

### Training LSTM (Unidirectional)

```bash
# Train model
python src/models/train_lstm.py

# Evaluate
python src/models/evaluate_lstm.py
```

### Threshold Analysis & Risk Scoring

Analyze prediction confidence to determine optimal thresholds for 5-level risk scoring:

```bash
python src/models/analyze_thresholds.py
```

**Outputs:**

- `reports/bilstm/threshold_analysis.png`
- Precision/Recall metrics for custom thresholds
- Suggested risk levels (Critical, High, Medium, Low, Minimal)

### Data Quality Audits

**Random Forest:**

```bash
python src/utils/data_audit.py
```

Checks:

- No data leakage between splits
- Class balance
- Missing values
- Feature correlation

**BiLSTM:**

```bash
python src/utils/data_audit_lstm.py
```

Checks:

- Sequence shapes
- NaN/Inf values
- Scaling verification
- Class distribution

### Testing

**Simulate Attack:**

```bash
python test/attack_test.py
```

**Stress Test:**

```bash
python src/models/stress_test.py
```

Output:

- ⚡ Throughput: predictions/sec
- 📊 Latency: average ms
- 💾 Memory usage

---

## 📦 Dependencies

```python
# Core ML
scikit-learn==1.7.2
tensorflow
joblib

# Data Processing
numpy
pandas

# Network & Security
scapy
cicflowmeter

# Visualization
matplotlib
seaborn
plotly

# Dashboard
streamlit

# Utilities
python-dotenv
```

**Install:**

```bash
pip install -r requirements.txt
```

---

## 🔍 How It Works

### Random Forest Pipeline

1. **Capture** (4s) → Scapy sniffs packets
2. **Extract** (2-5s) → CICFlowMeter generates 78 features
3. **Preprocess** → Select top 20 features, scale (0-1)
4. **Predict** → RF outputs probability, apply threshold (0.1077)
5. **Act** → Block IP if attack detected, log to DB

### BiLSTM Pipeline

1. **Preprocess** → Load CSVs, map labels (0/1/2)
2. **Sequence** → Create sliding windows (10 timesteps)
3. **Scale** → MinMaxScaler (0-1) fitted on training data
4. **Train** → 2 BiLSTM layers + BatchNorm + Dropout
5. **Evaluate** → Generate classification report, confusion matrix

### Why Dual Models?

| Scenario              | Best Model    | Reason                         |
| --------------------- | ------------- | ------------------------------ |
| Real-time blocking    | Random Forest | 6-9s latency, immediate action |
| Pattern analysis      | BiLSTM        | Captures temporal dependencies |
| Attack categorization | BiLSTM        | 3-class granularity            |
| Resource-constrained  | Random Forest | No GPU needed                  |
| Offline forensics     | BiLSTM        | Deep pattern recognition       |

---

## 📄 License

MIT License - Copyright (c) 2026 Betul Danismaz

See [LICENSE](LICENSE) for full text.

---

## 🙏 Acknowledgements

### Datasets

- **[CICIDS 2017](https://www.unb.ca/cic/datasets/ids-2017.html)** - Canadian Institute for Cybersecurity

### Tools & Libraries

- **[Scapy](https://scapy.net/)** - Packet manipulation
- **[CICFlowMeter](https://github.com/ahlashkari/CICFlowMeter)** - Flow feature extraction
- **[scikit-learn](https://scikit-learn.org/)** - Machine learning
- **[TensorFlow](https://www.tensorflow.org/)** - Deep learning
- **[Streamlit](https://streamlit.io/)** - Dashboard framework

---

## 📧 Contact & Support


**Authors:** Betul Danismaz, Mustafa Emre Bıyık

**Repository:** [Network_Anomaly_Detection](https://github.com/betuldanismaz/Network_Anomaly_Detection)

---

<div align="center">

## ⚡ Dual-Model Architecture | Real-Time Detection | Temporal Analysis

**Star ⭐ this repo if you find it useful!**

[![GitHub Stars](https://img.shields.io/github/stars/betuldanismaz/Network_Anomaly_Detection?style=social)](https://github.com/betuldanismaz/Network_Anomaly_Detection)
[![GitHub Forks](https://img.shields.io/github/forks/betuldanismaz/Network_Anomaly_Detection?style=social)](https://github.com/betuldanismaz/Network_Anomaly_Detection)

**Made with ❤️ for cybersecurity research**

</div>
