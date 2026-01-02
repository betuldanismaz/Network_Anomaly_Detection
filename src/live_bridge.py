#!/usr/bin/env python3
"""
Live IPS Bridge - BiLSTM Model with 5-Level Risk Scoring
=========================================================
Captures traffic, extracts CICFlowMeter features, and detects attacks using
BiLSTM time-series model with sliding window and multi-level risk assessment.

Author: NIDS Project
Date: 2026-01-01
"""

import os
import sys
import time
import shutil
import subprocess
import threading
import queue
import atexit
from datetime import datetime
from collections import deque
from typing import Tuple, Optional, Dict, Any

import joblib
import numpy as np
import pandas as pd
from scapy.all import sniff, wrpcap, rdpcap, conf
from dotenv import load_dotenv

# TensorFlow imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
import tensorflow as tf
from tensorflow.keras import models

# Load environment variables
load_dotenv()

# ---------------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# BiLSTM Model Paths
BILSTM_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "bilstm_best.keras")
SCALER_LSTM_PATH = os.path.join(PROJECT_ROOT, "models", "scaler_lstm.pkl")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")  # Fallback

# Import Top 20 Features from config
try:
    from config import TOP_FEATURES
    print(f"✅ TOP_FEATURES loaded from config.py ({len(TOP_FEATURES)} features)")
except ImportError:
    print("⚠️ config.py not found! Using fallback Top 20 features.")
    TOP_FEATURES = [
        "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Fwd Packets", "Total Length of Bwd Packets",
        "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
        "Bwd Packet Length Max", "Bwd Packet Length Min", "Flow Bytes/s",
        "Flow Packets/s", "Flow IAT Mean", "Fwd IAT Total", "Bwd IAT Total",
        "Fwd PSH Flags", "Fwd Header Length", "Fwd Packets/s", "Min Packet Length",
        "Average Packet Size"
    ]

# Utils Import
try:
    from utils.firewall_manager import block_ip
    from utils.db_manager import log_attack
except ImportError:
    print("⚠️ UYARI: firewall_manager/db_manager bulunamadı, dummy fonksiyonlar kullanılıyor.")
    def block_ip(ip_address):
        print(f"   [Simülasyon] {ip_address} engellenecekti.")
        return False
    def log_attack(*_args, **_kwargs):
        pass

# ---------------------------------------------------------------------------
# RUNTIME CONFIG
# ---------------------------------------------------------------------------
INTERFACE = os.getenv("NETWORK_INTERFACE", "")
TEMP_PCAP = "temp_live.pcap"
TEMP_CSV = "temp_live.csv"
WHITELIST_IPS = os.getenv(
    "WHITELIST_IPS",
    "192.168.1.1,127.0.0.1,0.0.0.0,localhost,8.8.8.8",
).split(",")

# Wireshark Verification Mode
WIRESHARK_VERBOSE = True

# BiLSTM Configuration
SEQUENCE_LENGTH = 10  # Sliding window size
NUM_FEATURES = 20     # Top 20 features

# ---------------------------------------------------------------------------
# 5-LEVEL RISK SCORING SYSTEM
# ---------------------------------------------------------------------------
RISK_LEVELS = {
    1: {"name": "SAFE",     "color": "🟢", "action": "ALLOW",  "description": "Normal Traffic"},
    2: {"name": "LOW",      "color": "🔵", "action": "ALLOW",  "description": "Minor Anomaly"},
    3: {"name": "MEDIUM",   "color": "🟡", "action": "ALERT",  "description": "Suspicious Activity"},
    4: {"name": "HIGH",     "color": "🟠", "action": "BLOCK",  "description": "Likely Attack"},
    5: {"name": "CRITICAL", "color": "🔴", "action": "BLOCK",  "description": "Confirmed Attack"},
}

# Class names mapping
CLASS_NAMES = {
    0: "Benign",
    1: "Volumetric",
    2: "Semantic"
}

DROP_COLS = [
    "Flow ID", "Source IP", "Src IP", "src_ip", "dst_ip",
    "Source Port", "Src Port", "src_port", "dst_port",
    "Destination IP", "Dest IP", "Destination Port", "Dest Port",
    "Timestamp", "timestamp", "Date", "protocol", "Flow_ID",
    "SimillarHTTP", "Label",
]

# ---------------------------------------------------------------------------
# DATA HARVEST CONFIG
# ---------------------------------------------------------------------------
HARVEST_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "live_captured_traffic_bilstm.csv")
HARVEST_BUFFER_SIZE = 25
HARVEST_FLUSH_INTERVAL = 30.0


def calculate_risk_level(probabilities: np.ndarray) -> Tuple[int, str, str, int]:
    """
    Calculate 5-level risk score from prediction probabilities.
    
    Args:
        probabilities: Array of shape [Prob_Benign, Prob_Volumetric, Prob_Semantic]
    
    Returns:
        Tuple of (risk_level, risk_name, action, predicted_class)
    """
    prob_benign = probabilities[0]
    prob_volumetric = probabilities[1]
    prob_semantic = probabilities[2]
    
    # Get predicted class
    predicted_class = int(np.argmax(probabilities))
    max_confidence = float(np.max(probabilities))
    
    # Benign Traffic Assessment
    if predicted_class == 0:  # Benign
        if prob_benign > 0.60:
            return 1, "SAFE", "ALLOW", predicted_class
        else:
            return 2, "LOW", "ALLOW", predicted_class
    
    # Attack Traffic Assessment (Volumetric or Semantic)
    elif predicted_class == 1:  # Volumetric Attack
        if prob_volumetric >= 0.90:
            return 5, "CRITICAL", "BLOCK", predicted_class
        elif prob_volumetric >= 0.70:
            return 4, "HIGH", "BLOCK", predicted_class
        elif prob_volumetric >= 0.50:
            return 3, "MEDIUM", "ALERT", predicted_class
        else:
            return 2, "LOW", "ALLOW", predicted_class
    
    else:  # Semantic Attack (class 2)
        if prob_semantic >= 0.85:
            return 5, "CRITICAL", "BLOCK", predicted_class
        elif prob_semantic >= 0.70:
            return 4, "HIGH", "BLOCK", predicted_class
        elif prob_semantic >= 0.50:
            return 3, "MEDIUM", "ALERT", predicted_class
        else:
            return 2, "LOW", "ALLOW", predicted_class


class BiLSTMDetector:
    """
    Live IDS/IPS using BiLSTM model with sliding window and 5-level risk scoring.
    
    Features:
    - Loads BiLSTM model trained for 3-class classification
    - Uses sliding window buffer for time-series prediction
    - Implements 5-level risk scoring system
    - Thread-safe traffic data harvesting
    """
    
    def __init__(
        self,
        csv_path: str = HARVEST_CSV_PATH,
        buffer_size: int = HARVEST_BUFFER_SIZE,
        flush_interval: float = HARVEST_FLUSH_INTERVAL,
    ):
        print("\n" + "="*70)
        print("🧠 BiLSTM DETECTOR INITIALIZATION")
        print("="*70)
        
        # 1. Load BiLSTM Model
        if not os.path.exists(BILSTM_MODEL_PATH):
            raise FileNotFoundError(f"❌ BiLSTM Model not found: {BILSTM_MODEL_PATH}")
        
        self.model = models.load_model(BILSTM_MODEL_PATH)
        print(f"✅ BiLSTM Model Loaded: {BILSTM_MODEL_PATH}")
        print(f"   Input Shape: {self.model.input_shape}")
        print(f"   Output Shape: {self.model.output_shape}")
        
        # 2. Load Scaler
        scaler_path = SCALER_LSTM_PATH if os.path.exists(SCALER_LSTM_PATH) else SCALER_PATH
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"❌ Scaler not found: {scaler_path}")
        
        self.scaler = joblib.load(scaler_path)
        print(f"✅ Scaler Loaded: {scaler_path}")
        
        # 3. Store Top Features
        self.top_features = TOP_FEATURES
        print(f"✅ Top Features: {len(self.top_features)} columns")
        
        # 4. Initialize Sliding Window Buffer
        self.sequence_buffer: deque = deque(maxlen=SEQUENCE_LENGTH)
        self.ip_buffer: deque = deque(maxlen=SEQUENCE_LENGTH)  # Track IPs
        print(f"✅ Sliding Window: {SEQUENCE_LENGTH} time steps")
        
        # 5. Traffic Logger Setup
        self.csv_path = csv_path
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        
        self._queue: queue.Queue = queue.Queue()
        self._buffer: list = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._last_flush_time = time.time()
        
        # CSV columns: Timestamp + Top20 Features + Risk columns
        self._csv_columns = (
            ["Timestamp"] + self.top_features + 
            ["Predicted_Class", "Class_Name", "Risk_Level", "Risk_Name", "Action", 
             "Prob_Benign", "Prob_Volumetric", "Prob_Semantic"]
        )
        
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        self._initialize_csv()
        
        # Start background writer thread
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True, name="TrafficLoggerWriter")
        self._writer_thread.start()
        
        atexit.register(self.shutdown)
        
        print(f"✅ Data Harvesting Active: {self.csv_path}")
        print("="*70 + "\n")
    
    def _initialize_csv(self):
        """Create CSV file with header if it doesn't exist."""
        if not os.path.exists(self.csv_path):
            try:
                header_df = pd.DataFrame(columns=self._csv_columns)
                header_df.to_csv(self.csv_path, index=False)
                print(f"   ↳ New CSV created ({len(self._csv_columns)} columns)")
            except Exception as e:
                print(f"⚠️ CSV initialization error: {e}")
        else:
            try:
                existing_df = pd.read_csv(self.csv_path, nrows=0)
                if list(existing_df.columns) != self._csv_columns:
                    print(f"⚠️ CSV schema mismatch, backing up...")
                    backup_path = self.csv_path.replace(".csv", f"_backup_{int(time.time())}.csv")
                    shutil.move(self.csv_path, backup_path)
                    self._initialize_csv()
            except Exception:
                pass
    
    def add_to_buffer(self, scaled_features: np.ndarray, src_ip: str = "Unknown"):
        """Add scaled features to the sliding window buffer."""
        self.sequence_buffer.append(scaled_features)
        self.ip_buffer.append(src_ip)
    
    def is_buffer_ready(self) -> bool:
        """Check if buffer has enough data for prediction."""
        return len(self.sequence_buffer) >= SEQUENCE_LENGTH
    
    def get_sequence(self) -> np.ndarray:
        """Get current sequence from buffer as numpy array."""
        if not self.is_buffer_ready():
            return None
        return np.array(list(self.sequence_buffer)).reshape(1, SEQUENCE_LENGTH, NUM_FEATURES)
    
    def predict(self, sequence: np.ndarray) -> Tuple[np.ndarray, int, str, str]:
        """
        Make prediction using BiLSTM model.
        
        Args:
            sequence: Numpy array of shape (1, 10, 20)
        
        Returns:
            Tuple of (probabilities, risk_level, risk_name, action)
        """
        if sequence is None:
            return None, 1, "INITIALIZING", "ALLOW"
        
        # Predict
        probabilities = self.model.predict(sequence, verbose=0)[0]
        
        # Calculate risk level
        risk_level, risk_name, action, predicted_class = calculate_risk_level(probabilities)
        
        return probabilities, risk_level, risk_name, action, predicted_class
    
    def process_and_predict(
        self, 
        features_df: pd.DataFrame, 
        src_ips=None, 
        dst_ips=None, 
        original_df=None
    ) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """
        Core prediction logic with sliding window and risk assessment.
        
        Args:
            features_df: Processed features aligned to TOP_FEATURES
            src_ips: Source IP addresses
            dst_ips: Destination IP addresses
            original_df: Original DataFrame for metric extraction
        
        Returns:
            Tuple of (predictions, result_info)
        """
        if features_df.empty:
            return None, None
        
        # 1. Filter to TOP_FEATURES only (Top 20)
        aligned_features = features_df.reindex(columns=self.top_features, fill_value=0)
        
        results = []
        
        for idx in range(len(aligned_features)):
            try:
                # Extract single row features
                row_features = aligned_features.iloc[idx].values.reshape(1, -1)
                
                # Scale features
                scaled_features = self.scaler.transform(row_features)[0]
                
                # Get source IP
                src_ip = str(src_ips.iloc[idx]) if src_ips is not None and idx < len(src_ips) else "Unknown"
                dst_ip = str(dst_ips.iloc[idx]) if dst_ips is not None and idx < len(dst_ips) else "Unknown"
                
                # Add to sliding window buffer
                self.add_to_buffer(scaled_features, src_ip)
                
                # Check if buffer is ready
                if not self.is_buffer_ready():
                    # Not enough data yet
                    result = {
                        'src_ip': src_ip,
                        'dst_ip': dst_ip,
                        'predicted_class': -1,
                        'class_name': 'Initializing',
                        'risk_level': 0,
                        'risk_name': 'INIT',
                        'action': 'BUFFER',
                        'probabilities': np.array([0.0, 0.0, 0.0]),
                        'buffer_size': len(self.sequence_buffer)
                    }
                    results.append(result)
                    print(f"⏳ Buffer filling: {len(self.sequence_buffer)}/{SEQUENCE_LENGTH}", end="\r")
                    continue
                
                # Get sequence and predict
                sequence = self.get_sequence()
                probabilities, risk_level, risk_name, action, predicted_class = self.predict(sequence)
                
                result = {
                    'src_ip': src_ip,
                    'dst_ip': dst_ip,
                    'predicted_class': predicted_class,
                    'class_name': CLASS_NAMES.get(predicted_class, 'Unknown'),
                    'risk_level': risk_level,
                    'risk_name': risk_name,
                    'action': action,
                    'probabilities': probabilities,
                    'buffer_size': len(self.sequence_buffer)
                }
                results.append(result)
                
                # Log to console with color-coded output
                self._log_prediction(result, dst_ip, original_df, idx)
                
            except Exception as e:
                print(f"⚠️ Prediction error for row {idx}: {e}")
                continue
        
        return results, None
    
    def _log_prediction(self, result: Dict, dst_ip: str, original_df: pd.DataFrame, idx: int):
        """Log prediction with color-coded risk level."""
        risk_info = RISK_LEVELS.get(result['risk_level'], RISK_LEVELS[1])
        color = risk_info['color']
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        probs = result['probabilities']
        
        # Format output based on risk level
        if result['risk_level'] >= 4:  # HIGH or CRITICAL
            print(f"\n{color} [{timestamp}] {result['risk_name']} RISK - {result['action']}")
            print(f"   Class: {result['class_name']} | Src: {result['src_ip']} → Dst: {dst_ip}")
            print(f"   Confidence: B:{probs[0]:.2%} V:{probs[1]:.2%} S:{probs[2]:.2%}")
            print("-" * 60)
        elif result['risk_level'] == 3:  # MEDIUM
            print(f"{color} [{timestamp}] {result['risk_name']}: {result['class_name']} | {result['src_ip']}")
        else:  # SAFE or LOW
            print(f"{color} [{timestamp}] {result['risk_name']} | {result['class_name']}", end="\r")
    
    def log(
        self,
        features_df: pd.DataFrame,
        results: list,
    ):
        """Queue feature data with predictions for async logging."""
        if features_df.empty or not results:
            return
        
        try:
            timestamp = datetime.now().isoformat()
            aligned_features = features_df.reindex(columns=self.top_features, fill_value=0)
            
            for idx, result in enumerate(results):
                if result['predicted_class'] == -1:  # Skip initializing entries
                    continue
                    
                row_data = {
                    "Timestamp": timestamp,
                    "Predicted_Class": result['predicted_class'],
                    "Class_Name": result['class_name'],
                    "Risk_Level": result['risk_level'],
                    "Risk_Name": result['risk_name'],
                    "Action": result['action'],
                    "Prob_Benign": float(result['probabilities'][0]),
                    "Prob_Volumetric": float(result['probabilities'][1]),
                    "Prob_Semantic": float(result['probabilities'][2]),
                }
                
                # Add Top 20 features
                if idx < len(aligned_features):
                    for col in self.top_features:
                        row_data[col] = aligned_features.iloc[idx][col]
                
                self._queue.put(row_data, block=False)
                
        except Exception as e:
            print(f"⚠️ Logging error: {e}")
    
    def _writer_loop(self):
        """Background thread that processes queue and writes to CSV."""
        while not self._stop_event.is_set():
            try:
                try:
                    row = self._queue.get(timeout=1.0)
                    with self._lock:
                        self._buffer.append(row)
                    self._queue.task_done()
                except queue.Empty:
                    pass
                
                should_flush = False
                with self._lock:
                    buffer_full = len(self._buffer) >= self.buffer_size
                    time_elapsed = (time.time() - self._last_flush_time) >= self.flush_interval
                    should_flush = (buffer_full or time_elapsed) and len(self._buffer) > 0
                
                if should_flush:
                    self._flush_buffer()
                    
            except Exception as e:
                print(f"⚠️ Writer thread error: {e}")
                time.sleep(1)
    
    def _flush_buffer(self):
        """Write buffered data to CSV file."""
        with self._lock:
            if not self._buffer:
                return
            
            rows_to_write = self._buffer.copy()
            self._buffer.clear()
            self._last_flush_time = time.time()
        
        try:
            df = pd.DataFrame(rows_to_write, columns=self._csv_columns)
            df.to_csv(self.csv_path, mode='a', header=False, index=False)
            print(f"💾 Saved {len(rows_to_write)} rows (Total: {self._get_total_rows()})")
        except Exception as e:
            print(f"⚠️ Flush error: {e}")
            with self._lock:
                self._buffer.extend(rows_to_write)
    
    def _get_total_rows(self) -> int:
        """Get approximate total row count in CSV."""
        try:
            if os.path.exists(self.csv_path):
                with open(self.csv_path, 'r', encoding='utf-8') as f:
                    return sum(1 for _ in f) - 1
        except Exception:
            pass
        return 0
    
    def shutdown(self):
        """Gracefully shutdown the detector."""
        print("\n🔄 Shutting down, flushing buffer...")
        self._stop_event.set()
        
        while not self._queue.empty():
            try:
                row = self._queue.get_nowait()
                with self._lock:
                    self._buffer.append(row)
            except queue.Empty:
                break
        
        self._flush_buffer()
        
        if self._writer_thread.is_alive():
            self._writer_thread.join(timeout=5.0)
        
        print(f"✅ All data saved -> {self.csv_path}")
    
    def get_stats(self) -> dict:
        """Return current detector statistics."""
        with self._lock:
            return {
                "buffer_size": len(self._buffer),
                "queue_size": self._queue.qsize(),
                "sequence_buffer_size": len(self.sequence_buffer),
                "total_rows": self._get_total_rows(),
                "csv_path": self.csv_path,
            }


# ---------------------------------------------------------------------------
# COLUMN RENAME MAP (CICFlowMeter -> Training Schema)
# ---------------------------------------------------------------------------
COLUMN_RENAME_MAP = {
    "flow_duration": "Flow Duration",
    "tot_fwd_pkts": "Total Fwd Packets",
    "tot_bwd_pkts": "Total Backward Packets",
    "totlen_fwd_pkts": "Total Length of Fwd Packets",
    "totlen_bwd_pkts": "Total Length of Bwd Packets",
    "fwd_pkt_len_max": "Fwd Packet Length Max",
    "fwd_pkt_len_min": "Fwd Packet Length Min",
    "fwd_pkt_len_mean": "Fwd Packet Length Mean",
    "bwd_pkt_len_max": "Bwd Packet Length Max",
    "bwd_pkt_len_min": "Bwd Packet Length Min",
    "flow_byts_s": "Flow Bytes/s",
    "flow_pkts_s": "Flow Packets/s",
    "flow_iat_mean": "Flow IAT Mean",
    "fwd_iat_tot": "Fwd IAT Total",
    "bwd_iat_tot": "Bwd IAT Total",
    "fwd_psh_flags": "Fwd PSH Flags",
    "fwd_header_len": "Fwd Header Length",
    "fwd_pkts_s": "Fwd Packets/s",
    "pkt_len_min": "Min Packet Length",
    "pkt_size_avg": "Average Packet Size",
}

ALT_CIC_RENAME_MAP = {
    "Tot Fwd Pkts": "Total Fwd Packets",
    "Tot Bwd Pkts": "Total Backward Packets",
    "TotLen Fwd Pkts": "Total Length of Fwd Packets",
    "TotLen Bwd Pkts": "Total Length of Bwd Packets",
    "Fwd Pkt Len Max": "Fwd Packet Length Max",
    "Fwd Pkt Len Min": "Fwd Packet Length Min",
    "Fwd Pkt Len Mean": "Fwd Packet Length Mean",
    "Bwd Pkt Len Max": "Bwd Packet Length Max",
    "Bwd Pkt Len Min": "Bwd Packet Length Min",
    "Flow Byts/s": "Flow Bytes/s",
    "Flow Pkts/s": "Flow Packets/s",
    "Flow IAT Mean": "Flow IAT Mean",
    "Fwd IAT Tot": "Fwd IAT Total",
    "Bwd IAT Tot": "Bwd IAT Total",
    "Fwd PSH Flags": "Fwd PSH Flags",
    "Fwd Header Len": "Fwd Header Length",
    "Fwd Pkts/s": "Fwd Packets/s",
    "Pkt Len Min": "Min Packet Length",
    "Pkt Size Avg": "Average Packet Size",
}


def prepare_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rename CICFlowMeter columns and align them with Top 20 features."""
    working = df.copy()
    working.columns = working.columns.str.strip()
    working.drop(columns=DROP_COLS, errors="ignore", inplace=True)
    
    # Apply rename mappings
    working.rename(columns=ALT_CIC_RENAME_MAP, inplace=True)
    working.rename(columns=COLUMN_RENAME_MAP, inplace=True)
    
    # Fill missing columns with 0
    for col in TOP_FEATURES:
        if col not in working.columns:
            working[col] = 0
    
    return working.reindex(columns=TOP_FEATURES, fill_value=0)


# ---------------------------------------------------------------------------
# GLOBAL DETECTOR INSTANCE
# ---------------------------------------------------------------------------
DETECTOR: BiLSTMDetector = None


# ---------------------------------------------------------------------------
# CICFLOWMETER HELPERS
# ---------------------------------------------------------------------------

def run_cicflowmeter_cli(pcap_file: str, csv_file: str):
    """Run cicflowmeter CLI and return (success, error_message)."""
    cmd = ["cicflowmeter", "-f", pcap_file, "-c", csv_file]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    except FileNotFoundError:
        return False, "cicflowmeter CLI not found"
    except Exception as exc:
        return False, str(exc)

    if result.returncode != 0:
        err = result.stderr.strip() or result.stdout.strip() or f"CLI error {result.returncode}"
        return False, err

    if not os.path.exists(csv_file):
        base_name = os.path.splitext(pcap_file)[0]
        alt_name = f"{base_name}_Flow.csv"
        
        if os.path.exists(alt_name):
            try:
                shutil.move(alt_name, csv_file)
            except OSError:
                pass
        else:
            return False, "CSV file not found after CLI execution"

    return True, None


def run_cicflowmeter_api(pcap_file: str, csv_file: str):
    """Simplified fallback: Creates basic flow features from packets."""
    try:
        packets = rdpcap(pcap_file)
        if len(packets) == 0:
            return False, "PCAP file is empty"
        
        flows = {}
        first_timestamps = {}
        last_timestamps = {}
        
        for pkt in packets:
            if not (pkt.haslayer('IP') and (pkt.haslayer('TCP') or pkt.haslayer('UDP'))):
                continue
            
            src_ip = pkt['IP'].src
            dst_ip = pkt['IP'].dst
            src_port = pkt['TCP'].sport if pkt.haslayer('TCP') else pkt['UDP'].sport
            dst_port = pkt['TCP'].dport if pkt.haslayer('TCP') else pkt['UDP'].dport
            protocol = pkt['IP'].proto
            
            flow_key = (src_ip, dst_ip, src_port, dst_port, protocol)
            timestamp = float(pkt.time) if hasattr(pkt, 'time') else 0
            pkt_len = len(pkt)
            
            if flow_key not in flows:
                flows[flow_key] = {
                    'Src IP': src_ip,
                    'Dst IP': dst_ip,
                    'Flow Duration': 0,
                    'Total Fwd Packets': 0,
                    'Total Backward Packets': 0,
                    'Total Length of Fwd Packets': 0,
                    'Total Length of Bwd Packets': 0,
                    'Fwd Packet Length Max': 0,
                    'Fwd Packet Length Min': 999999,
                    'Fwd Packet Length Mean': 0,
                    'Bwd Packet Length Max': 0,
                    'Bwd Packet Length Min': 999999,
                    'Flow Bytes/s': 0,
                    'Flow Packets/s': 0,
                    'Flow IAT Mean': 0,
                    'Fwd IAT Total': 0,
                    'Bwd IAT Total': 0,
                    'Fwd PSH Flags': 0,
                    'Fwd Header Length': 0,
                    'Fwd Packets/s': 0,
                    'Min Packet Length': 999999,
                    'Average Packet Size': 0,
                }
                first_timestamps[flow_key] = timestamp
            
            flows[flow_key]['Total Fwd Packets'] += 1
            flows[flow_key]['Total Length of Fwd Packets'] += pkt_len
            flows[flow_key]['Fwd Packet Length Max'] = max(flows[flow_key]['Fwd Packet Length Max'], pkt_len)
            flows[flow_key]['Fwd Packet Length Min'] = min(flows[flow_key]['Fwd Packet Length Min'], pkt_len)
            flows[flow_key]['Min Packet Length'] = min(flows[flow_key]['Min Packet Length'], pkt_len)
            
            ip_header_len = pkt['IP'].ihl * 4 if hasattr(pkt['IP'], 'ihl') else 20
            tcp_header_len = pkt['TCP'].dataofs * 4 if pkt.haslayer('TCP') and hasattr(pkt['TCP'], 'dataofs') else 0
            udp_header_len = 8 if pkt.haslayer('UDP') else 0
            flows[flow_key]['Fwd Header Length'] += ip_header_len + tcp_header_len + udp_header_len
            
            if pkt.haslayer('TCP') and pkt['TCP'].flags & 0x08:  # PSH flag
                flows[flow_key]['Fwd PSH Flags'] += 1
            
            last_timestamps[flow_key] = timestamp
        
        for flow_key, flow_data in flows.items():
            duration = last_timestamps[flow_key] - first_timestamps[flow_key]
            flow_data['Flow Duration'] = int(duration * 1000000)
            
            total_pkts = flow_data['Total Fwd Packets'] + flow_data['Total Backward Packets']
            total_bytes = flow_data['Total Length of Fwd Packets'] + flow_data['Total Length of Bwd Packets']
            
            if duration > 0:
                flow_data['Flow Bytes/s'] = total_bytes / duration
                flow_data['Flow Packets/s'] = total_pkts / duration
                flow_data['Fwd Packets/s'] = flow_data['Total Fwd Packets'] / duration
            
            if total_pkts > 0:
                flow_data['Fwd Packet Length Mean'] = flow_data['Total Length of Fwd Packets'] / flow_data['Total Fwd Packets']
                flow_data['Average Packet Size'] = total_bytes / total_pkts
        
        if not flows:
            return False, "No valid flows found"
        
        df = pd.DataFrame(list(flows.values()))
        df.to_csv(csv_file, index=False)
        
        return True, None
        
    except Exception as exc:
        return False, f"Feature extraction error: {exc}"


# ---------------------------------------------------------------------------
# NETWORK INTERFACE DETECTION
# ---------------------------------------------------------------------------

def get_active_interface():
    """Auto-detect active network interface."""
    for iface in conf.ifaces.values():
        if iface.ip and iface.ip != "127.0.0.1" and iface.ip != "0.0.0.0":
            if "Wi-Fi" in iface.name or "Wireless" in iface.name or "Ethernet" in iface.name:
                return iface.name
    return conf.iface


def resolve_interface(requested: str):
    """Resolve user-provided interface string to Scapy interface name."""
    requested = (requested or "").strip()
    if not requested:
        return get_active_interface()

    req_l = requested.casefold()
    candidates = []

    for iface in conf.ifaces.values():
        name = (getattr(iface, "name", "") or "")
        desc = (getattr(iface, "description", "") or "")

        if name.casefold() == req_l or desc.casefold() == req_l:
            return name

        if req_l in name.casefold() or req_l in desc.casefold():
            candidates.append(name)

    if len(candidates) == 1:
        return candidates[0]

    return None


# ---------------------------------------------------------------------------
# MAIN INITIALIZATION
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("🛡️  AI NETWORK IPS - BiLSTM MODEL with 5-LEVEL RISK SCORING")
print("="*70)

if not os.path.exists(BILSTM_MODEL_PATH):
    print(f"❌ BiLSTM Model not found: {BILSTM_MODEL_PATH}")
    print("   Please train the BiLSTM model first.")
    sys.exit(1)

try:
    DETECTOR = BiLSTMDetector()
except Exception as exc:
    print(f"❌ BiLSTM Detector initialization failed: {exc}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

if shutil.which("cicflowmeter") is None:
    print("\n⚠️  WARNING: 'cicflowmeter' CLI not found (pip install cicflowmeter)")
    print("   Fallback API mode will be used.")

# Resolve interface
_resolved_iface = resolve_interface(INTERFACE)
if _resolved_iface is None:
    print(f"\n❌ Interface '{INTERFACE}' not found by Scapy/Npcap.")
    print("\nAvailable interfaces:")
    for iface in conf.ifaces.values():
        print(f"  - {getattr(iface, 'name', '')} | {getattr(iface, 'description', '')} | {getattr(iface, 'ip', '')}")
    sys.exit(2)

INTERFACE = _resolved_iface
print(f"\n🛡️  SYSTEM STARTED | Interface: {INTERFACE}")


# ---------------------------------------------------------------------------
# CORE PIPELINE: Extract Features & Predict with BiLSTM
# ---------------------------------------------------------------------------

def feature_extraction_and_predict():
    """Captures packets -> Extracts features -> Predicts with BiLSTM."""
    print("   ↳ ⚙️ Analyzing...", end="\r")
    
    success, err_msg = run_cicflowmeter_cli(TEMP_PCAP, TEMP_CSV)
    
    if not success:
        success, err_msg = run_cicflowmeter_api(TEMP_PCAP, TEMP_CSV)
        if not success:
            return

    try:
        df = pd.read_csv(TEMP_CSV)
    except Exception as exc:
        print(f"⚠️ CSV read error: {exc}")
        return

    if df.empty:
        return

    src_ips = df.get('Src IP') if 'Src IP' in df.columns else df.get('Source IP')
    dst_ips = df.get('Dst IP') if 'Dst IP' in df.columns else df.get('Destination IP')

    df_features = prepare_feature_frame(df)

    try:
        results, _ = DETECTOR.process_and_predict(df_features, src_ips=src_ips, dst_ips=dst_ips, original_df=df)
    except Exception as exc:
        print(f"⚠️ Prediction error: {exc}")
        import traceback
        traceback.print_exc()
        return

    if results:
        try:
            DETECTOR.log(df_features, results)
        except Exception as log_exc:
            print(f"⚠️ Logging error: {log_exc}")

    # Process high-risk detections
    for result in results or []:
        if result['action'] == 'BLOCK':
            ip_addr = result['src_ip']
            
            if ip_addr and ip_addr not in WHITELIST_IPS and ip_addr != "Unknown":
                block_ip(ip_addr)
                log_attack(ip_addr, "BLOCKED", f"{result['class_name']} Attack - Level {result['risk_level']}")
            else:
                log_attack(ip_addr, "ALLOWED", "Whitelisted")


# ---------------------------------------------------------------------------
# MAIN EXECUTION LOOP
# ---------------------------------------------------------------------------

def main_loop():
    """Main IPS loop: continuously monitors network traffic."""
    print(f"\n📡 Monitoring Network: {INTERFACE}")
    print("⏹️  Press CTRL+C to stop...\n")
    
    iteration_count = 0
    stats_interval = 10

    while True:
        try:
            print("⏳ Capturing packets...", end="\r")
            packets = sniff(iface=INTERFACE, timeout=4)
            
            if len(packets) > 0:
                wrpcap(TEMP_PCAP, packets)
                feature_extraction_and_predict()
            else:
                print(f"⚠️ 0 Packets! Check interface '{INTERFACE}'", end="\r")

            iteration_count += 1
            if iteration_count % stats_interval == 0:
                stats = DETECTOR.get_stats()
                print(f"📊 [Stats] Buffer: {stats['buffer_size']}/{HARVEST_BUFFER_SIZE} | "
                      f"Sequence: {stats['sequence_buffer_size']}/{SEQUENCE_LENGTH} | "
                      f"Total: {stats['total_rows']}")
            
        except KeyboardInterrupt:
            print("\n🛑 System stopped by user.")
            DETECTOR.shutdown()
            break
        except Exception as exc:
            print(f"\n⚠️ Loop error: {exc}")
            time.sleep(1)


if __name__ == "__main__":
    main_loop()
