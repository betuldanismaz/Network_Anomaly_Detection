# Network Intrusion Detection System (NIDS)

A professional, modular, and extensible Network Intrusion Detection System (NIDS) for analyzing network traffic, preprocessing data, and training machine learning models for attack detection.

## Project Structure

```
networkdetection/
├── data/
│   ├── processed_csv/      # Source CSV datasets
│   │   └── ready_splits/   # Train/Val/Test splits
│   └── raw_pcap/           # Raw packet captures
├── models/                 # Saved models (.pkl)
├── reports/
│   └── figures/            # Generated plots (Confusion Matrix, Feature Importance)
├── src/
│   ├── capture/
│   │   └── sniffer.py      # Real-time packet capture
│   ├── features/
│   │   └── preprocess.py   # Data cleaning & splitting pipeline
│   ├── models/
│   │   ├── train_rf.py     # Random Forest training script
│   │   └── analyze_results.py # Model performance & feature analysis
│   └── utils/
│       └── data_audit.py   # Data quality & leakage check
├── requirements.txt
└── README.md
```

## Features

- **Packet Capture:** Real-time network packet sniffing using Scapy (`src/capture/sniffer.py`).
- **Data Preprocessing:** Cleans, merges, and splits network traffic datasets for ML tasks (`src/features/preprocess.py`).
- **Data Quality Audit:** Automated checks for data leakage, class distribution, and sanity (`src/utils/data_audit.py`).
- **Model Training:** Random Forest implementation with automated evaluation and visualization (`src/models/train_rf.py`).
- **Forensic Analysis:** Tools to analyze feature importance and missed attacks (`src/models/analyze_results.py`).

## Getting Started

### Prerequisites

- Python 3.7+
- pip

### Installation

1. Clone the repository:
   ```sh
   git clone <repo-url>
   cd networkdetection
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

### 1. Data Preparation

- Place your raw PCAP files in `data/raw_pcap/` (if needed).
- Preprocessed CSVs should be in `data/processed_csv/`.
- **Process and split the data:**
  ```sh
  python src/features/preprocess.py
  ```
  This will generate `train.csv`, `val.csv`, and `test.csv` in `data/processed_csv/ready_splits/`.

### 2. Data Audit (Optional but Recommended)

Verify data health, check for leakage, and inspect class distribution:

```sh
python src/utils/data_audit.py
```

### 3. Model Training

Train the Random Forest model and generate performance reports:

```sh
python src/models/train_rf.py
```

- Saves the model to `models/rf_model_v1.pkl`.
- Saves the confusion matrix to `reports/figures/confusion_matrix_rf.png`.

### 4. Model Analysis

Analyze feature importance and investigate missed attacks:

```sh
python src/models/analyze_results.py
```

- Generates `reports/figures/feature_importance.png`.

### 5. Packet Sniffing

To capture and display live network packets:

```sh
python src/capture/sniffer.py
```

## Requirements

See `requirements.txt` for all dependencies:

- numpy
- pandas
- scikit-learn
- tensorflow
- scapy
- streamlit
- matplotlib
- seaborn
- joblib

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements, bug fixes, or new features.

## License

This project is licensed under the MIT License.

## Acknowledgements

- [CICIDS 2017 Dataset](https://www.unb.ca/cic/datasets/malmem-2017.html)
- Scapy, scikit-learn, TensorFlow, and the open-source community.
