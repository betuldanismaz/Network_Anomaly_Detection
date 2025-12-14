# 🛡️ Network Intrusion Prevention System (IPS)

A real-time, production-ready Network Intrusion Prevention System that combines machine learning-based threat detection with automated firewall response and live monitoring dashboard. Built for detecting and blocking sophisticated network attacks including DDoS, Port Scanning, Web Attacks, and Infiltration attempts.

## 🚀 Recent Updates

### ✨ Major Refactoring Complete

- **Optimized Model Integration**: Now uses `rf_model_optimized.pkl` with Top 20 features (3x faster)
- **Dynamic Threshold**: Loads optimal threshold from `models/threshold.txt` for precision control
- **Enhanced Prediction**: Uses `predict_proba()` with threshold-based classification
- **LiveDetector Class**: Unified architecture for prediction + data harvesting
- **Wireshark Verification**: Detailed packet logging for academic validation
- **Improved Data Harvest**: Thread-safe buffered CSV writes with 25-row batching

📄 **Documentation:**

- [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) - Complete changelog
- [ARCHITECTURE.md](ARCHITECTURE.md) - System flow diagrams
- [TESTING_CHECKLIST.md](TESTING_CHECKLIST.md) - Verification tests

## 🌟 Key Features

### 🔍 Real-Time Threat Detection

- **Live packet capture and analysis** using Scapy
- **ML-powered classification** with optimized Random Forest (99.8% accuracy)
- **Top 20 feature extraction** for efficient processing (down from 78)
- **Sub-second detection latency** for rapid response
- **Dynamic threshold optimization** for minimizing false negatives

### 🚨 Automated Defense

- **Automatic IP blocking** via Windows Firewall/iptables integration
- **Configurable whitelist** to protect critical infrastructure
- **Threshold-based predictions** (proba ≥ threshold → ATTACK)
- **Manual override controls** through the dashboard

### 📊 Live Monitoring Dashboard

- **Real-time statistics** (total events, blocked IPs, last attack time)
- **Interactive visualizations** (attack frequency charts, action distribution)
- **Event log viewer** with filtering and search
- **IP unblock interface** for manual intervention
- Built with Streamlit for instant deployment

### 🧠 Advanced ML Pipeline

- **Hyperparameter-tuned Random Forest** classifier
- **Feature importance analysis** → Top 20 features selected
- **Threshold optimization** via grid search (Recall-focused)
- **Comprehensive evaluation metrics** (Precision, Recall, F1, ROC-AUC)
- Support for LSTM-based sequential models

### 🗄️ Persistent Logging & Data Harvest

- **SQLite database** for attack event storage
- **Live traffic CSV logging** for model retraining (`data/live_captured_traffic.csv`)
- **Buffered writes** (25 rows OR 30 seconds, whichever first)
- **Thread-safe architecture** with background writer
- **Schema**: Timestamp + 20 features + Label + Confidence (22 columns)

## 📁 Project Structure

```
networkdetection/
├── data/
│   ├── processed_csv/              # CICIDS 2017 preprocessed datasets
│   │   ├── ready_splits/           # Train/Val/Test splits (80/10/10)
│   │   └── *.csv                   # Individual attack scenario CSVs
│   └── raw_pcap/                   # Raw packet captures (optional)
├── models/
│   ├── rf_model_v1.pkl             # Trained Random Forest model
│   ├── scaler.pkl                  # Feature scaler
│   ├── threshold_config.json       # Optimized decision threshold
│   └── top_20_features.json        # Feature selection config
├── src/
│   ├── capture/
│   │   └── sniffer.py              # Network packet capture module
│   ├── dashboard/
│   │   └── app.py                  # Streamlit monitoring dashboard
│   ├── features/
│   │   └── preprocess.py           # Data cleaning and feature engineering
│   ├── models/
│   │   ├── randomforest.py         # RF training with hyperparameter tuning
│   │   ├── train_lstm.py           # LSTM model implementation
│   │   ├── analyze_results.py      # Model evaluation and forensics
│   │   └── stress_test.py          # Performance benchmarking
│   ├── utils/
│   │   ├── db_manager.py           # SQLite operations
│   │   ├── firewall_manager.py     # OS-level IP blocking
│   │   ├── data_audit.py           # Data quality checks
│   │   ├── model_optimizer.py      # Threshold tuning
│   │   └── xai_engine.py           # Explainable AI utilities
│   ├── live_bridge.py              # Main IPS orchestration engine
│   └── config.py                   # Top 20 features configuration
├── test/
│   ├── attack_test.py              # Attack simulation scripts
│   └── check_interfaces.py         # Network interface validation
├── alerts.db                       # Attack event database
├── requirements.txt                # Python dependencies
└── .env.example                    # Environment configuration template

```

## 🚀 Quick Start

### 1. Installation

```powershell
# Clone the repository
git clone https://github.com/betuldanismaz/Network_Anomaly_Detection.git
cd Network_Anomaly_Detection

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file from the template:

```powershell
cp .env.example .env
```

Edit `.env` and configure:

```env
NETWORK_INTERFACE=Wi-Fi              # Your network interface name
WHITELIST_IPS=192.168.1.1,127.0.0.1  # Protected IPs (comma-separated)
THRESHOLD=0.45                        # Detection sensitivity (0-1)
```

### 3. Train the Model (Optional)

If you want to retrain with your own data:

```powershell
# Preprocess datasets
python src/features/preprocess.py

# Train Random Forest with optimization
python src/models/randomforest.py
```

Pre-trained models are included in `models/` directory.

### 4. Run the IPS

**Terminal 1 - Start the IPS engine:**

```powershell
python src/live_bridge.py
```

**Terminal 2 - Launch the dashboard:**

```powershell
streamlit run src/dashboard/app.py
```

Dashboard will open at `http://localhost:8501`

### 5. Test Detection (Optional)

Simulate an attack to verify the system:

```powershell
python test/attack_test.py
```

## 🎯 Usage Scenarios

### Monitor Live Traffic

The system continuously analyzes network packets and displays detections in the console:

```
🔍 Analyzing packet batch (50 packets)...
⚠️  ATTACK DETECTED: DDoS (confidence: 98.3%)
🚫 Blocking IP: 192.168.1.105
✅ Event logged to database
```

### View Dashboard

Access real-time statistics and visualizations at `http://localhost:8501`:

- **KPI Cards:** Total events, blocked IPs, last attack timestamp
- **Attack Frequency Chart:** 1-minute resolution time series
- **Action Distribution:** Pie chart of blocked vs. allowed traffic
- **Event Table:** Searchable log with all metadata

### Unblock an IP

Use the dashboard sidebar:

1. Enter IP address in "Engeli Kaldırılacak IP" field
2. Click "Unblock IP" button
3. Confirmation message appears

### Analyze Model Performance

```powershell
python src/models/analyze_results.py
```

Generates:

- Confusion matrix heatmap
- ROC curve with AUC score
- Precision-Recall curve
- Feature importance ranking
- Missed attack analysis report

## 📊 Model Performance

Trained on **CICIDS 2017** dataset with 2.8M+ samples covering 7 attack types:

| Metric                  | Score  |
| ----------------------- | ------ |
| **Accuracy**            | 99.73% |
| **Precision**           | 97.87% |
| **Recall (Attack)**     | 99.90% |
| **F1-Score**            | 98.88% |
| **ROC-AUC**             | 0.9994 |
| **False Negative Rate** | 0.10%  |

**Attack Types Detected:**

- DDoS (Distributed Denial of Service)
- PortScan (Network Reconnaissance)
- Web Attack (SQL Injection, XSS, etc.)
- Infiltration (APT, Lateral Movement)
- Botnet Traffic
- Brute Force Attacks

## 🔧 Advanced Configuration

### Threshold Tuning

Adjust detection sensitivity in `models/threshold_config.json`:

```json
{
  "optimal_threshold": 0.45,
  "precision_at_threshold": 0.9945,
  "recall_at_threshold": 0.9931
}
```

Lower values = more sensitive (more false positives)
Higher values = less sensitive (more false negatives)

### Feature Selection

Modify `src/config.py` to use different features:

```python
TOP_FEATURES = [
    "Bwd Packet Length Std",
    "Packet Length Variance",
    # ... add/remove features
]
```

### Firewall Integration

The system automatically detects your OS and uses:

- **Windows:** `netsh advfirewall` commands
- **Linux:** `iptables` rules

Customize in `src/utils/firewall_manager.py`.

## 🛠️ Development

### Run Data Quality Audit

```powershell
python src/utils/data_audit.py
```

Checks for:

- Data leakage between splits
- Class imbalance
- Missing values
- Feature correlation issues

### Stress Test the Model

```powershell
python src/models/stress_test.py
```

Measures:

- Inference latency (ms per prediction)
- Memory usage
- Throughput (predictions/second)

### Add New Attack Types

1. Add labeled data to `data/processed_csv/`
2. Update preprocessing in `src/features/preprocess.py`
3. Retrain: `python src/models/randomforest.py`
4. Update attack type mapping in `src/live_bridge.py`

## 📦 Dependencies

**Core:**

- `scapy` - Packet capture and analysis
- `scikit-learn` - Machine learning algorithms
- `pandas`, `numpy` - Data manipulation
- `streamlit` - Dashboard framework
- `joblib` - Model serialization

**Analysis:**

- `matplotlib`, `seaborn`, `plotly` - Visualization
- `tensorflow` - LSTM model support (optional)

**Utilities:**

- `python-dotenv` - Environment configuration
- `cicflowmeter` - Network flow feature extraction

See `requirements.txt` for complete list with versions.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

**Areas for Contribution:**

- Additional ML models (XGBoost, Neural Networks)
- Support for more firewall systems (pfSense, UFW)
- Real-time alerting (email, Slack, PagerDuty)
- Advanced XAI visualizations
- Docker containerization

## 📄 License

This project is licensed under the MIT License. See `LICENSE` file for details.

## 🙏 Acknowledgements

- **Dataset:** [CICIDS 2017](https://www.unb.ca/cic/datasets/ids-2017.html) by Canadian Institute for Cybersecurity
- **Kaggle Dataset:** [Network Intrusion Dataset](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset)
- **Libraries:** Scapy, scikit-learn, TensorFlow, Streamlit, and the open-source community

## 📧 Contact

**Authors:** Betul Danismaz  , Mustafa Emre Bıyık
**Repository:** [Network_Anomaly_Detection](https://github.com/betuldanismaz/Network_Anomaly_Detection)  

---

⚡ **Powered by Machine Learning | Secured by Automation | Monitored in Real-Time**
