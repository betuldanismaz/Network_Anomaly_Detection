#!/usr/bin/env python3
"""
Live IPS Bridge - Optimized Model with Dynamic Threshold
Captures traffic, extracts CICFlowMeter features, and detects attacks using Top 20 features.
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

import joblib
import numpy as np
import pandas as pd
from scapy.all import sniff, wrpcap, rdpcap, conf
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# NEW: Optimized Model Paths
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "rf_optimized_model.pkl")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")
THRESHOLD_PATH = os.path.join(PROJECT_ROOT, "models", "threshold.txt")

sys.path.append(os.path.join(CURRENT_DIR, "utils"))

# Import Top 20 Features from config
try:
    from config import TOP_FEATURES
    print(f"✅ TOP_FEATURES loaded from config.py ({len(TOP_FEATURES)} features)")
except ImportError:
    print("⚠️ config.py not found! Using fallback Top 20 features.")
    # Fallback Top 20 (replace with actual list if config.py missing)
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
    from firewall_manager import block_ip
    from db_manager import log_attack
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
INTERFACE = os.getenv("NETWORK_INTERFACE", "Wi-Fi")
TEMP_PCAP = "temp_live.pcap"
TEMP_CSV = "temp_live.csv"
WHITELIST_IPS = os.getenv("WHITELIST_IPS", "192.168.1.1,127.0.0.1,0.0.0.0,localhost").split(",")

# Wireshark Verification Mode
WIRESHARK_VERBOSE = True  # Set to True for detailed packet logging

DROP_COLS = [
    "Flow ID", "Source IP", "Src IP", "src_ip", "dst_ip",
    "Source Port", "Src Port", "src_port", "dst_port",
    "Destination IP", "Dest IP", "Destination Port", "Dest Port",
    "Timestamp", "timestamp", "Date", "protocol", "Flow_ID",
    "SimillarHTTP", "Label",
]

# ---------------------------------------------------------------------------
# DATA HARVEST - TRAFFIC LOGGER
# ---------------------------------------------------------------------------
HARVEST_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "live_captured_traffic.csv")
HARVEST_BUFFER_SIZE = 25  # Number of rows to buffer before writing to disk
HARVEST_FLUSH_INTERVAL = 30.0  # Force flush every N seconds even if buffer not full

# Training data schema (78 features) - must match exactly with model training columns
TRAINING_FEATURE_COLUMNS = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]


class LiveDetector:
    """
    Optimized Live IDS/IPS with Dynamic Threshold and Top 20 Feature Filtering.
    
    Features:
    - Loads optimized Random Forest model
    - Uses dynamic threshold from threshold.txt
    - Filters to TOP_FEATURES only (Top 20)
    - Wireshark-compatible verbose logging
    - Thread-safe traffic data harvesting
    Logs every prediction to CSV (live_captured_traffic.csv) for future model retraining
    """
    
    def __init__(
        self,
        csv_path: str = HARVEST_CSV_PATH,
        buffer_size: int = HARVEST_BUFFER_SIZE,
        flush_interval: float = HARVEST_FLUSH_INTERVAL,
    ):
        print("\n" + "="*70)
        print("🔧 LIVE DETECTOR INITIALIZATION")
        print("="*70)
        
        # 1. Load Optimized Model
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")
        
        self.model = joblib.load(MODEL_PATH)
        print(f"✅ Model Loaded: {MODEL_PATH}")
        print(f"   Model Type: {type(self.model).__name__}")
        
        # 2. Load Scaler
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"❌ Scaler not found: {SCALER_PATH}")
        
        self.scaler = joblib.load(SCALER_PATH)
        print(f"✅ Scaler Loaded: {SCALER_PATH}")
        
        # 3. Load Dynamic Threshold (for fallback if not found, use 0.5)
        self.threshold = 0.5  # Default
        if os.path.exists(THRESHOLD_PATH):
            try:
                with open(THRESHOLD_PATH, 'r') as f:
                    self.threshold = float(f.read().strip())
                print(f"✅ Threshold Loaded: {self.threshold:.4f} (from {THRESHOLD_PATH})")
            except Exception as e:
                print(f"⚠️ Threshold read error: {e}. Using default 0.5")
        else:
            print(f"⚠️ Threshold file not found. Using default: {self.threshold}")
        
        # 4. Store Top Features
        self.top_features = TOP_FEATURES
        print(f"✅ Top Features: {len(self.top_features)} columns")
        
        # 5. Traffic Logger Setup
        self.csv_path = csv_path
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        
        self._queue: queue.Queue = queue.Queue()
        self._buffer: list = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._last_flush_time = time.time()
        
        # CSV columns: Timestamp + Top20 Features + Predicted_Label + Confidence_Score
        self._csv_columns = ["Timestamp"] + self.top_features + ["Predicted_Label", "Confidence_Score"]
        
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
    
    def wireshark_log(self, src_ip: str, dst_ip: str, flow_duration: float, total_fwd_length: float, prediction: int, confidence: float):
        """
        Wireshark-compatible verbose logging for professor verification.
        
        Format: Timestamp | Src IP | Dst IP | Fwd Length | Flow Duration | Prediction | Confidence
        """
        if not WIRESHARK_VERBOSE:
            return
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        status = "🚨 ATTACK" if prediction == 1 else "✅ NORMAL"
        
        print(f"[{timestamp}] {status}")
        print(f"  Src: {src_ip:15s} → Dst: {dst_ip:15s}")
        print(f"  Fwd Length: {total_fwd_length:10.2f} bytes | Flow Duration: {flow_duration:10.6f} sec")
        print(f"  Prediction: {prediction} | Confidence: {confidence:.4f}")
        print("-" * 80)
    
    def process_and_predict(self, features_df: pd.DataFrame, src_ips=None, dst_ips=None, original_df=None):
        """
        Core prediction logic with dynamic threshold.
        
        Args:
            features_df: Processed features aligned to TOP_FEATURES
            src_ips: Source IP addresses (for logging)
            dst_ips: Destination IP addresses (for logging)
            original_df: Original DataFrame with raw column names (for metric extraction)
        
        Returns:
            predictions: Array of 0/1 predictions
            probabilities: Array of confidence scores
        """
        if features_df.empty:
            return None, None
        
        # 1. Filter to TOP_FEATURES only (Top 20)
        aligned_features = features_df.reindex(columns=self.top_features, fill_value=0)
        
        # Check for missing columns
        missing_cols = set(self.top_features) - set(features_df.columns)
        if missing_cols:
            print(f"⚠️ Missing features (filled with 0): {list(missing_cols)[:5]}...")
        
        # 2. Data Scaling
        try:
            scaled_array = self.scaler.transform(aligned_features)
            X_scaled = pd.DataFrame(scaled_array, columns=aligned_features.columns, index=aligned_features.index)
        except Exception as e:
            print(f"❌ Scaling error: {e}")
            return None, None
        
        # 3. Predict using predict_proba (NOT predict)
        try:
            probabilities = self.model.predict_proba(X_scaled)
            attack_probabilities = probabilities[:, 1]  # Probability of Class 1 (Attack)
            
            # Apply dynamic threshold
            predictions = (attack_probabilities >= self.threshold).astype(int)
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None, None
        
        # 4. Wireshark Verification Logging  
        for idx in range(len(predictions)):
            src_ip = src_ips.iloc[idx] if src_ips is not None and idx < len(src_ips) else "Unknown"
            dst_ip = dst_ips.iloc[idx] if dst_ips is not None and idx < len(dst_ips) else "Unknown"
            
            # Extract metrics from original DataFrame (before column renaming)
            if original_df is not None and not original_df.empty:
                # Flow Duration is in microseconds, convert to seconds
                flow_duration_us = original_df.iloc[idx].get("Flow Duration", 0.0)
                flow_duration = flow_duration_us / 1_000_000.0
                # Total Forward Length
                total_fwd_length = original_df.iloc[idx].get("TotLen Fwd Pkts", 0.0)
            else:
                flow_duration = 0.0
                total_fwd_length = 0.0
            
            self.wireshark_log(
                src_ip=src_ip,
                dst_ip=dst_ip,
                flow_duration=flow_duration,
                total_fwd_length=total_fwd_length,
                prediction=predictions[idx],
                confidence=attack_probabilities[idx]
            )
        
        # 5. Data Harvest: Log to CSV
        self.log(aligned_features, predictions, probabilities)
        
        return predictions, probabilities
    
    def log(
        self,
        features_df: pd.DataFrame,
        predictions: np.ndarray,
        probabilities: np.ndarray = None,
        debug: bool = False,
    ):
        """
        Queue feature data with predictions for async logging.
        
        Args:
            features_df: DataFrame with 78 feature columns (already scaled or raw)
            predictions: Array of 0/1 predictions
            probabilities: Optional array of confidence scores (2D array with shape [n_samples, 2])
        """
        if features_df.empty:
            return
        
        try:
            timestamp = datetime.now().isoformat()
            
            # Ensure features match Top 20 columns
            aligned_features = features_df.reindex(columns=self.top_features, fill_value=0)
            
            for idx in range(len(aligned_features)):
                row_data = {
                    "Timestamp": timestamp,
                    "Predicted_Label": int(predictions[idx]),
                    "Confidence_Score": float(probabilities[idx, 1]) if probabilities is not None else float(predictions[idx]),
                }
                # Add Top 20 features
                for col in self.top_features:
                    row_data[col] = aligned_features.iloc[idx][col]
                
                # Put row in queue (non-blocking)
                self._queue.put(row_data, block=False)
        except Exception as e:
            print(f"⚠️ [LiveDetector] Logging error: {e}")
            import traceback
            traceback.print_exc()
    
    def _writer_loop(self):
        """Background thread that processes queue and writes to CSV."""
        while not self._stop_event.is_set():
            try:
                # Try to get item from queue with timeout
                try:
                    row = self._queue.get(timeout=1.0)
                    with self._lock:
                        self._buffer.append(row)
                    self._queue.task_done()
                except queue.Empty:
                    pass
                
                # Check if we should flush
                should_flush = False
                with self._lock:
                    buffer_full = len(self._buffer) >= self.buffer_size  # buffer size is 25
                    time_elapsed = (time.time() - self._last_flush_time) >= self.flush_interval
                    should_flush = (buffer_full or time_elapsed) and len(self._buffer) > 0
                
                if should_flush:
                    self._flush_buffer()
                    
            except Exception as e:
                print(f"⚠️ [TrafficLogger] Writer thread hatası: {e}")
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
            print(f"💾 [TrafficLogger] {len(rows_to_write)} satır kaydedildi (Toplam: {self._get_total_rows()})")
        except Exception as e:
            print(f"⚠️ [TrafficLogger] Flush hatası: {e}")
            # Put rows back in queue for retry
            with self._lock:
                self._buffer.extend(rows_to_write)
    
    def _get_total_rows(self) -> int:
        """Get approximate total row count in CSV."""
        try:
            if os.path.exists(self.csv_path):
                with open(self.csv_path, 'r', encoding='utf-8') as f:
                    return sum(1 for _ in f) - 1  # Subtract header
        except Exception:
            pass
        return 0
    
    def shutdown(self):
        """Gracefully shutdown the logger, flushing remaining data."""
        print("\n🔄 [TrafficLogger] Kapanıyor, buffer temizleniyor...")
        self._stop_event.set()
        
        # Drain the queue
        while not self._queue.empty():
            try:
                row = self._queue.get_nowait()
                with self._lock:
                    self._buffer.append(row)
            except queue.Empty:
                break
        
        # Final flush
        self._flush_buffer()
        
        if self._writer_thread.is_alive():
            self._writer_thread.join(timeout=5.0)
        
        print(f"✅ [TrafficLogger] Tüm veriler kaydedildi -> {self.csv_path}")
    
    def get_stats(self) -> dict:
        """Return current logger statistics."""
        with self._lock:
            return {
                "buffer_size": len(self._buffer),
                "queue_size": self._queue.qsize(),
                "total_rows": self._get_total_rows(),
                "csv_path": self.csv_path,
            }


# Global LiveDetector instance (initialized after model loading)
DETECTOR: 'LiveDetector' = None

COLUMN_RENAME_MAP = {
    "flow_duration": "Flow Duration",
    "tot_fwd_pkts": "Total Fwd Packets",
    "tot_bwd_pkts": "Total Backward Packets",
    "totlen_fwd_pkts": "Total Length of Fwd Packets",
    "totlen_bwd_pkts": "Total Length of Bwd Packets",
    "fwd_pkt_len_max": "Fwd Packet Length Max",
    "fwd_pkt_len_min": "Fwd Packet Length Min",
    "fwd_pkt_len_mean": "Fwd Packet Length Mean",
    "fwd_pkt_len_std": "Fwd Packet Length Std",
    "bwd_pkt_len_max": "Bwd Packet Length Max",
    "bwd_pkt_len_min": "Bwd Packet Length Min",
    "bwd_pkt_len_mean": "Bwd Packet Length Mean",
    "bwd_pkt_len_std": "Bwd Packet Length Std",
    "flow_byts_s": "Flow Bytes/s",
    "flow_pkts_s": "Flow Packets/s",
    "flow_iat_mean": "Flow IAT Mean",
    "flow_iat_std": "Flow IAT Std",
    "flow_iat_max": "Flow IAT Max",
    "flow_iat_min": "Flow IAT Min",
    "fwd_iat_tot": "Fwd IAT Total",
    "fwd_iat_mean": "Fwd IAT Mean",
    "fwd_iat_std": "Fwd IAT Std",
    "fwd_iat_max": "Fwd IAT Max",
    "fwd_iat_min": "Fwd IAT Min",
    "bwd_iat_tot": "Bwd IAT Total",
    "bwd_iat_mean": "Bwd IAT Mean",
    "bwd_iat_std": "Bwd IAT Std",
    "bwd_iat_max": "Bwd IAT Max",
    "bwd_iat_min": "Bwd IAT Min",
    "fwd_psh_flags": "Fwd PSH Flags",
    "bwd_psh_flags": "Bwd PSH Flags",
    "fwd_urg_flags": "Fwd URG Flags",
    "bwd_urg_flags": "Bwd URG Flags",
    "fwd_header_len": "Fwd Header Length",
    "bwd_header_len": "Bwd Header Length",
    "fwd_pkts_s": "Fwd Packets/s",
    "bwd_pkts_s": "Bwd Packets/s",
    "pkt_len_min": "Min Packet Length",
    "pkt_len_max": "Max Packet Length",
    "pkt_len_mean": "Packet Length Mean",
    "pkt_len_std": "Packet Length Std",
    "pkt_len_var": "Packet Length Variance",
    "fin_flag_cnt": "FIN Flag Count",
    "syn_flag_cnt": "SYN Flag Count",
    "rst_flag_cnt": "RST Flag Count",
    "psh_flag_cnt": "PSH Flag Count",
    "ack_flag_cnt": "ACK Flag Count",
    "urg_flag_cnt": "URG Flag Count",
    "cwr_flag_count": "CWE Flag Count",
    "ece_flag_cnt": "ECE Flag Count",
    "down_up_ratio": "Down/Up Ratio",
    "pkt_size_avg": "Average Packet Size",
    "fwd_seg_size_avg": "Avg Fwd Segment Size",
    "bwd_seg_size_avg": "Avg Bwd Segment Size",
    "fwd_byts_b_avg": "Fwd Avg Bytes/Bulk",
    "fwd_pkts_b_avg": "Fwd Avg Packets/Bulk",
    "fwd_blk_rate_avg": "Fwd Avg Bulk Rate",
    "bwd_byts_b_avg": "Bwd Avg Bytes/Bulk",
    "bwd_pkts_b_avg": "Bwd Avg Packets/Bulk",
    "bwd_blk_rate_avg": "Bwd Avg Bulk Rate",
    "subflow_fwd_pkts": "Subflow Fwd Packets",
    "subflow_fwd_byts": "Subflow Fwd Bytes",
    "subflow_bwd_pkts": "Subflow Bwd Packets",
    "subflow_bwd_byts": "Subflow Bwd Bytes",
    "init_fwd_win_byts": "Init_Win_bytes_forward",
    "init_bwd_win_byts": "Init_Win_bytes_backward",
    "fwd_act_data_pkts": "act_data_pkt_fwd",
    "fwd_seg_size_min": "min_seg_size_forward",
    "active_mean": "Active Mean",
    "active_std": "Active Std",
    "active_max": "Active Max",
    "active_min": "Active Min",
    "idle_mean": "Idle Mean",
    "idle_std": "Idle Std",
    "idle_max": "Idle Max",
    "idle_min": "Idle Min",
}

# ---------------------------------------------------------------------------
# GOLD_STANDARD_FEATURES (78 Training Features)
EXPECTED_FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]

print("\n" + "=" * 60)

def extract_source_ips(df: pd.DataFrame):
    """Return whichever source IP column exists."""
    for candidate in ("Src IP", "Source IP", "src_ip"):
        if candidate in df.columns:
            return df[candidate]
    return None


def prepare_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rename CICFlowMeter columns and align them with training schema."""
    working = df.copy()
    working.columns = working.columns.str.strip()
    working.drop(columns=DROP_COLS, errors="ignore", inplace=True)
    working.rename(columns=COLUMN_RENAME_MAP, inplace=True)

    if "Fwd Header Length" in working.columns and "Fwd Header Length.1" not in working.columns:
        working["Fwd Header Length.1"] = working["Fwd Header Length"]

    missing_cols = [col for col in EXPECTED_FEATURES if col not in working.columns]
    for col in missing_cols:
        working[col] = 0

    return working.reindex(columns=EXPECTED_FEATURES, fill_value=0)

# ---------------------------------------------------------------------------
# MAIN INITIALIZATION
# ---------------------------------------------------------------------------
print("\n🛡️  AI NETWORK IPS - OPTIMIZED MODEL")
print("=" * 70)

if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
    print(f"❌ Model not found: {MODEL_PATH}")
    print(f"❌ Scaler not found: {SCALER_PATH}")
    print("   Please run training pipeline first.")
    sys.exit(1)

try:
    # Initialize LiveDetector (loads model, scaler, threshold, starts logger)
    DETECTOR = LiveDetector()
except Exception as exc:
    print(f"❌ LiveDetector initialization failed: {exc}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

if shutil.which("cicflowmeter") is None:
    print("\n⚠️  WARNING: 'cicflowmeter' CLI not found (pip install cicflowmeter)")
    print("   Fallback API mode will be used.")


# ---------------------------------------------------------------------------
# CICFLOWMETER HELPERS
# ---------------------------------------------------------------------------

def run_cicflowmeter_cli(pcap_file: str, csv_file: str):
    """Run cicflowmeter CLI and return (success, error_message)."""
    # CICFlowMeter CLI syntax: cicflowmeter -f input.pcap -c output.csv
    cmd = ["cicflowmeter", "-f", pcap_file, "-c", csv_file]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    except FileNotFoundError:
        # cicflowmeter not in PATH, will fall back to API mode
        return False, "cicflowmeter CLI bulunamadı, API modu kullanılacak"
    except Exception as exc:
        return False, str(exc)

    if result.returncode != 0:
        err = result.stderr.strip() or result.stdout.strip() or f"CLI hata kodu {result.returncode}"
        return False, err

    # CICFlowMeter bazen çıktı dosyasının ismini değiştirir (örn: temp_live.pcap_Flow.csv)
    # Eğer hedef dosya yoksa, olası diğer isimleri kontrol et ve düzelt.
    if not os.path.exists(csv_file):
        # Olası isim: {pcap_dosyası}_Flow.csv
        base_name = os.path.splitext(pcap_file)[0] # temp_live
        alt_name = f"{base_name}_Flow.csv"         # temp_live_Flow.csv
        
        if os.path.exists(alt_name):
            try:
                shutil.move(alt_name, csv_file)
            except OSError:
                pass # Dosya kullanımda olabilir
        else:
            return False, "CLI çalıştı fakat CSV dosyası bulunamadı (Dosya ismi farklı olabilir)."

    return True, None


def run_cicflowmeter_api(pcap_file: str, csv_file: str):
    """
    Simplified fallback: Creates basic flow features directly from packets.
    CICFlowMeter API is unreliable, so we create minimal features for prediction.
    """
    try:
        packets = rdpcap(pcap_file)
        if len(packets) == 0:
            return False, "PCAP dosyası boş"
        
        # Create basic flow statistics with timestamps
        flows = {}
        first_timestamps = {}
        last_timestamps = {}
        
        for pkt in packets:
            if not (pkt.haslayer('IP') and (pkt.haslayer('TCP') or pkt.haslayer('UDP'))):
                continue
            
            # Create flow key
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
                    'Src Port': src_port,
                    'Dst Port': dst_port,
                    'Protocol': protocol,
                    'Flow Duration': 0,
                    'Tot Fwd Pkts': 0,
                    'Tot Bwd Pkts': 0,
                    'TotLen Fwd Pkts': 0,
                    'TotLen Bwd Pkts': 0,
                    'Fwd Pkt Len Max': 0,
                    'Fwd Pkt Len Min': 999999,
                    'Fwd Pkt Len Mean': 0,
                    'Bwd Pkt Len Max': 0,
                    'Flow Byts/s': 0,
                    'Flow Pkts/s': 0,
                    'Flow IAT Mean': 0,
                    'Fwd IAT Tot': 0,
                    'Fwd IAT Mean': 0,
                    'Fwd IAT Max': 0,
                    'Fwd Header Length': 0,
                    'Bwd Header Length': 0,
                    'Fwd Pkts/s': 0,
                    'Bwd Pkts/s': 0,
                    'Pkt Len Min': 999999,
                    'Pkt Len Max': 0,
                    'Pkt Len Mean': 0,
                }
                first_timestamps[flow_key] = timestamp
            
            # Update flow statistics
            flows[flow_key]['Tot Fwd Pkts'] += 1
            flows[flow_key]['TotLen Fwd Pkts'] += pkt_len
            flows[flow_key]['Fwd Pkt Len Max'] = max(flows[flow_key]['Fwd Pkt Len Max'], pkt_len)
            flows[flow_key]['Fwd Pkt Len Min'] = min(flows[flow_key]['Fwd Pkt Len Min'], pkt_len)
            flows[flow_key]['Pkt Len Max'] = max(flows[flow_key]['Pkt Len Max'], pkt_len)
            flows[flow_key]['Pkt Len Min'] = min(flows[flow_key]['Pkt Len Min'], pkt_len)
            
            # Calculate header length (IP + TCP/UDP)
            ip_header_len = pkt['IP'].ihl * 4 if hasattr(pkt['IP'], 'ihl') else 20
            tcp_header_len = pkt['TCP'].dataofs * 4 if pkt.haslayer('TCP') and hasattr(pkt['TCP'], 'dataofs') else 0
            udp_header_len = 8 if pkt.haslayer('UDP') else 0
            flows[flow_key]['Fwd Header Length'] += ip_header_len + tcp_header_len + udp_header_len
            
            last_timestamps[flow_key] = timestamp
        
        # Calculate derived features
        for flow_key, flow_data in flows.items():
            duration = last_timestamps[flow_key] - first_timestamps[flow_key]
            flow_data['Flow Duration'] = int(duration * 1000000)  # microseconds
            
            total_pkts = flow_data['Tot Fwd Pkts'] + flow_data['Tot Bwd Pkts']
            total_bytes = flow_data['TotLen Fwd Pkts'] + flow_data['TotLen Bwd Pkts']
            
            if duration > 0:
                flow_data['Flow Byts/s'] = total_bytes / duration
                flow_data['Flow Pkts/s'] = total_pkts / duration
                flow_data['Fwd Pkts/s'] = flow_data['Tot Fwd Pkts'] / duration
            
            if total_pkts > 0:
                flow_data['Fwd Pkt Len Mean'] = flow_data['TotLen Fwd Pkts'] / flow_data['Tot Fwd Pkts']
                flow_data['Pkt Len Mean'] = total_bytes / total_pkts
        
        # Convert to DataFrame
        if not flows:
            return False, "Geçerli flow bulunamadı"
        
        df = pd.DataFrame(list(flows.values()))
        df.to_csv(csv_file, index=False)
        
        return True, None
        
    except Exception as exc:
        return False, f"Basit feature extraction hatası: {exc}"


# ---------------------------------------------------------------------------
# NETWORK INTERFACE DETECTION
# ---------------------------------------------------------------------------

def get_active_interface():
    """Auto-detect active network interface."""
    for iface in conf.ifaces.values():
        if iface.ip and iface.ip != "127.0.0.1" and iface.ip != "0.0.0.0":
            if "Wi-Fi" in iface.name or "Wireless" in iface.name or "Ethernet" in iface.name:
                return iface.name
    return conf.iface  # Fallback to default


# ---------------------------------------------------------------------------
# RUNTIME GLOBALS
# ---------------------------------------------------------------------------

INTERFACE = get_active_interface()
TEMP_PCAP = "temp_live.pcap"
TEMP_CSV = "temp_live.csv"
WHITELIST_IPS = ["192.168.1.1", "127.0.0.1", "0.0.0.0", "8.8.8.8"] # Modem vs.

print(f"\n🛡️  SİSTEM BAŞLATILDI | Arayüz: {INTERFACE}")


# ---------------------------------------------------------------------------
# CORE PIPELINE: Extract Features & Predict Attacks (NEW OPTIMIZED VERSION)
# ---------------------------------------------------------------------------

def feature_extraction_and_predict():
    """
    Captures packets → Extracts features via CICFlowMeter → Predicts attacks
    using the optimized model with dynamic threshold and Top 20 features.
    """
    print("   ↳ ⚙️ Analiz...", end="\r")
    
    # Step 1: Run CICFlowMeter CLI to extract features
    success, err_msg = run_cicflowmeter_cli(TEMP_PCAP, TEMP_CSV)
    
    if not success:
        # Fallback to simplified feature extraction if CLI fails
        success, err_msg = run_cicflowmeter_api(TEMP_PCAP, TEMP_CSV)
        if not success:
            # Silently skip - not all packets need analysis
            return

    # Step 2: Load extracted features
    try:
        df = pd.read_csv(TEMP_CSV)
    except Exception as exc:
        print(f"⚠️  CSV okuma hatası: {exc}")
        return

    if df.empty:
        return

    # Step 3: Store IP addresses for attack logging
    src_ips = df.get('Src IP') if 'Src IP' in df.columns else df.get('Source IP')
    dst_ips = df.get('Dst IP') if 'Dst IP' in df.columns else df.get('Destination IP')

    # Step 4: Align columns to training schema (78 features → Top 20)
    df_features = prepare_feature_frame(df)

    # Step 5: Use DETECTOR for prediction with optimized model
    try:
        predictions, probabilities = DETECTOR.process_and_predict(df_features, src_ips=src_ips, dst_ips=dst_ips, original_df=df)
    except Exception as exc:
        print(f"⚠️  Prediction error: {exc}")
        import traceback
        traceback.print_exc()
        return

    # Step 6: DATA HARVEST - Log all traffic to CSV for future training
    try:
        DETECTOR.log(df_features, predictions, probabilities)
    except Exception as exc:
        print(f"⚠️  Logging error: {exc}")

    # Step 7: Process attack detections
    attack_detected = False
    for idx, pred in enumerate(predictions):
        if pred != 1:  # 0 = Normal traffic
            continue
            
        attack_detected = True
        ip_addr = src_ips.iloc[idx] if src_ips is not None else "Bilinmiyor"
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Wireshark verification logging
        try:
            packet_data = {
                'src_ip': ip_addr,
                'dst_ip': dst_ips.iloc[idx] if dst_ips is not None else "Unknown",
                'fwd_length': df_features.iloc[idx].get('Total Length of Fwd Packets', 0),
                'flow_duration': df_features.iloc[idx].get('Flow Duration', 0),
                'probability': probabilities[idx] if probabilities is not None else 1.0,
            }
            DETECTOR.wireshark_log(packet_data, pred)
        except Exception:
            pass  # Don't break flow on logging errors
        
        print(f"\n🚨 [{timestamp}] TEHDİT ALGILANDI! Kaynak IP: {ip_addr}")
        print(f"   Güven Skoru: {probabilities[idx]:.2%}" if probabilities is not None else "")
        
        if ip_addr and ip_addr not in WHITELIST_IPS and ip_addr != "Bilinmiyor":
            block_ip(ip_addr)
            log_attack(ip_addr, "BLOCKED", "Attack Detected")
        else:
            print("   ✅ IP beyaz listede veya bilinmiyor, engellenmedi.")
            log_attack(ip_addr, "ALLOWED", "Whitelisted")

    # Step 8: Log clean traffic summary
    if not attack_detected:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"✅ [{timestamp}] Trafik Temiz - Güvenli ({len(predictions)} Akış)", end="\r")


# ---------------------------------------------------------------------------
# MAIN EXECUTION LOOP
# ---------------------------------------------------------------------------

def main_loop():
    """Main IPS loop: continuously monitors network traffic for attacks."""
    print(f"\n📡 Ağ Dinleniyor: {INTERFACE}")
    print("⏹️  Durdurmak için CTRL+C yapın...\n")
    
    iteration_count = 0
    stats_interval = 10  # Show logger stats every N iterations

    while True:
        try:
            print("⏳ Paket toplanıyor...", end="\r")
            packets = sniff(iface=INTERFACE, timeout=4)
            
            if len(packets) > 0:
                wrpcap(TEMP_PCAP, packets)
                feature_extraction_and_predict()
            else:
                print(f"⚠️ 0 Paket! '{INTERFACE}' ismini kontrol et.        ", end="\r")

            # Show data harvest statistics periodically
            iteration_count += 1
            if iteration_count % stats_interval == 0:
                stats = DETECTOR.get_stats()
                print(f"📊 [Data Harvest] Buffer: {stats['buffer_size']}/{HARVEST_BUFFER_SIZE} | "
                      f"Toplam Kayıt: {stats['total_rows']} | "
                      f"Son Yazma: {stats.get('last_flush_time', 'N/A')}")

            # Cleanup temporary files (optional - keep for debugging)
            # if os.path.exists(TEMP_PCAP):
            #     os.remove(TEMP_PCAP)
            # if os.path.exists(TEMP_CSV):
            #     os.remove(TEMP_CSV)
            
        except KeyboardInterrupt:
            print("\n🛑 Sistem kullanıcı tarafından durduruldu.")
            DETECTOR.shutdown()  # Ensure data is flushed before exit
            break
        except Exception as exc:
            print(f"\n⚠️  Loop error: {exc}")
            time.sleep(1)


if __name__ == "__main__":
    main_loop()

