# Network Intrusion Detection System (NIDS)

A professional, modular, and extensible Network Intrusion Detection System (NIDS) for analyzing network traffic, preprocessing data, and training machine learning models for attack detection.

## Project Structure

```
networkdetection/
├── data/
│   ├── processed_csv/
│   │   ├── [CSV files: processed network traffic data]
│   │   └── ready_splits/
│   │       ├── train.csv
│   │       ├── val.csv
│   │       └── test.csv
│   └── raw_pcap/
├── models/
├── src/
│   ├── capture/
│   │   └── sniffer.py
│   ├── dashboard/
│   ├── features/
│   │   └── preprocess.py
│   └── models/
├── requirements.txt
└── README.md
```

## Features

- **Packet Capture:** Real-time network packet sniffing using Scapy (`src/capture/sniffer.py`).
- **Data Preprocessing:** Cleans, merges, and splits network traffic datasets for ML tasks (`src/features/preprocess.py`).
- **Machine Learning Ready:** Prepares data splits for training, validation, and testing.
- **Extensible:** Modular codebase for easy integration of new models, features, or dashboards.

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

### Data Preparation

- Place your raw PCAP files in `data/raw_pcap/` (if needed).
- Preprocessed CSVs should be in `data/processed_csv/`.
- To process and split the data:
  ```sh
  python src/features/preprocess.py
  ```
  This will generate `train.csv`, `val.csv`, and `test.csv` in `data/processed_csv/ready_splits/`.

### Packet Sniffing

To capture and display live network packets:

```sh
python src/capture/sniffer.py
```

### Model Training & Evaluation

- Add your model scripts to `src/models/`.
- Use the prepared data splits for training and evaluation.

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
