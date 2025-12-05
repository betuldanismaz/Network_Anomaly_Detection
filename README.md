# Network Anomaly Detection System

Network Anomaly Detection System is a modular, extensible platform for analyzing network traffic, detecting anomalies, and identifying attacks using machine learning. The system is designed for research, prototyping, and practical deployment in cybersecurity environments.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- **Real-time Packet Capture:** Live network traffic sniffing and recording.
- **Data Preprocessing:** Automated cleaning, merging, and splitting of network datasets for ML tasks.
- **Data Quality Audit:** Tools for checking data leakage, class balance, and integrity.
- **Model Training:** Random Forest and LSTM-based training pipelines with evaluation and visualization.
- **Forensic Analysis:** Feature importance, confusion matrix, and missed attack analysis.
- **Extensible Architecture:** Modular codebase for easy integration of new models and data sources.

## Project Structure

```
networkdetection/
├── data/
│   ├── processed_csv/      # Preprocessed CSV datasets
│   │   └── ready_splits/   # Train/Val/Test splits
│   └── raw_pcap/           # Raw packet captures
├── models/                 # Saved models (.pkl)
├── reports/
│   └── figures/            # Generated plots and analysis
├── src/
│   ├── capture/            # Packet sniffing scripts
│   ├── features/           # Data preprocessing scripts
│   ├── models/             # Model training and analysis
│   └── utils/              # Utility scripts (data audit, firewall, etc.)
├── requirements.txt        # Python dependencies
└── README.md
```

## Installation

1. Clone the repository:
   ```sh
   git clone <r[epo-url](https://github.com/betuldanismaz/Network_Anomaly_Detection.git)>
   cd networkdetection
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### 1. Data Preparation

- Place raw PCAP files in `data/raw_pcap/`.
- Place preprocessed CSVs in `data/processed_csv/`.
- To process and split data:
  ```sh
  python src/features/preprocess.py
  ```
  This generates `train.csv`, `val.csv`, and `test.csv` in `data/processed_csv/ready_splits/`.

### 2. Data Audit (Recommended)

Check data health, leakage, and class distribution:

```sh
python src/utils/data_audit.py
```

### 3. Model Training

Train a Random Forest model and generate evaluation reports:

```sh
python src/models/train_rf.py
```

Model and figures are saved in `models/` and `reports/figures/`.

### 4. Model Analysis

Analyze feature importance and missed attacks:

```sh
python src/models/analyze_results.py
```

### 5. Live Packet Capture

Capture and display live network packets:

```sh
python src/capture/sniffer.py
```

## Requirements

- Python 3.7+
- See `requirements.txt` for all dependencies:
  - numpy, pandas, scikit-learn, tensorflow, scapy, streamlit, matplotlib, seaborn, joblib

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements, bug fixes, or new features. For major changes, please discuss with the maintainers first.

## License

MIT LICENCE

This project is licensed under the MIT License.

## Acknowledgements

- [CICIDS 2017 Dataset](https://www.unb.ca/cic/datasets/malmem-2017.html)
- https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset/code
- Scapy, scikit-learn, TensorFlow, and the open-source community.
