import pandas as pd
import joblib
import os
import sys
import time
import shutil
import subprocess
import threading
import queue
import atexit
from datetime import datetime
from scapy.all import sniff, wrpcap, conf

import joblib
import numpy as np
import pandas as pd
from scapy.all import sniff, wrpcap, rdpcap
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "..", "models", "rf_model_v1.pkl")
SCALER_PATH = os.path.join(CURRENT_DIR, "..", "models", "scaler.pkl")
sys.path.append(os.path.join(CURRENT_DIR, "utils"))

# Utils Import
try:
    from firewall_manager import block_ip
    from db_manager import log_attack
except ImportError:
    def log_attack(*_args, **_kwargs):
        pass

# ---------------------------------------------------------------------------
# RUNTIME CONFIG
# ---------------------------------------------------------------------------
# Scapy interface name must match show_interfaces() output exactly.
INTERFACE = os.getenv("NETWORK_INTERFACE", "Wi-Fi")
TEMP_PCAP = "temp_live.pcap"
TEMP_CSV = "temp_live.csv"
WHITELIST_IPS = os.getenv("WHITELIST_IPS", "192.168.1.1,127.0.0.1,0.0.0.0,localhost").split(",")
DROP_COLS = [
    "Flow ID",
    "Source IP",
    "Src IP",
    "src_ip",
    "dst_ip",
    "Source Port",
    "Src Port",
    "src_port",
    "dst_port",
    "Destination IP",
    "Dest IP",
    "Destination Port",
    "Dest Port",
    "Timestamp",
    "timestamp",
    "Date",
    "protocol",
    "Flow_ID",
    "SimillarHTTP",
    "Label",
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


class TrafficLogger:
    """
    Thread-safe traffic logger with buffered writes for minimal latency impact.
    
    Features:
    - Buffers rows in memory to reduce disk I/O frequency
    - Uses a separate writer thread to avoid blocking the main sniffing loop
    - Auto-flushes on buffer full or time interval
    - Graceful shutdown with atexit hook
    """
    
    def __init__(
        self,
        csv_path: str = HARVEST_CSV_PATH,
        buffer_size: int = HARVEST_BUFFER_SIZE,
        flush_interval: float = HARVEST_FLUSH_INTERVAL,
    ):
        self.csv_path = csv_path
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        
        # Thread-safe queue for passing data to writer thread
        self._queue: queue.Queue = queue.Queue()
        self._buffer: list = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._last_flush_time = time.time()
        
        # CSV header columns: Timestamp + 78 features + Predicted_Label + Confidence_Score
        self._csv_columns = ["Timestamp"] + TRAINING_FEATURE_COLUMNS + ["Predicted_Label", "Confidence_Score"]
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        
        # Initialize CSV file with header if it doesn't exist
        self._initialize_csv()
        
        # Start the background writer thread
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True, name="TrafficLoggerWriter")
        self._writer_thread.start()
        
        # Register shutdown hook to flush remaining data
        atexit.register(self.shutdown)
        
        print(f"📊 [TrafficLogger] Data harvesting aktif -> {self.csv_path}")
    
    def _initialize_csv(self):
        """Create CSV file with header if it doesn't exist."""
        if not os.path.exists(self.csv_path):
            try:
                header_df = pd.DataFrame(columns=self._csv_columns)
                header_df.to_csv(self.csv_path, index=False)
                print(f"   ↳ Yeni CSV dosyası oluşturuldu ({len(self._csv_columns)} sütun)")
            except Exception as e:
                print(f"⚠️ [TrafficLogger] CSV başlatma hatası: {e}")
        else:
            # Validate existing file has correct columns
            try:
                existing_df = pd.read_csv(self.csv_path, nrows=0)
                if list(existing_df.columns) != self._csv_columns:
                    print(f"⚠️ [TrafficLogger] Mevcut CSV şeması uyumsuz, yedekleniyor...")
                    backup_path = self.csv_path.replace(".csv", f"_backup_{int(time.time())}.csv")
                    shutil.move(self.csv_path, backup_path)
                    self._initialize_csv()
            except Exception:
                pass  # File might be empty or corrupted, will be overwritten
    
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
            probabilities: Optional array of confidence scores (attack probability)
            debug: If True, print column alignment diagnostics
        """
        if features_df.empty:
            return
        
        try:
            timestamp = datetime.now().isoformat()
            
            # === COLUMN ALIGNMENT DEBUGGING ===
            if debug or not hasattr(self, '_alignment_checked'):
                input_cols = set(features_df.columns)
                expected_cols = set(TRAINING_FEATURE_COLUMNS)
                
                missing_cols = expected_cols - input_cols
                extra_cols = input_cols - expected_cols
                
                print(f"\n🔍 [TrafficLogger] Column Alignment Check:")
                print(f"   Input columns:    {len(features_df.columns)}")
                print(f"   Expected columns: {len(TRAINING_FEATURE_COLUMNS)}")
                
                if missing_cols:
                    print(f"   ⚠️ MISSING ({len(missing_cols)}): {list(missing_cols)[:5]}{'...' if len(missing_cols) > 5 else ''}")
                if extra_cols:
                    print(f"   ⚠️ EXTRA ({len(extra_cols)}): {list(extra_cols)[:5]}{'...' if len(extra_cols) > 5 else ''}")
                
                if not missing_cols and not extra_cols:
                    print(f"   ✅ Perfect match! All 78 columns aligned.")
                else:
                    print(f"   ℹ️ Missing columns will be filled with 0, extra columns ignored.")
                
                self._alignment_checked = True
            
            # Ensure features are in correct column order
            aligned_features = features_df.reindex(columns=TRAINING_FEATURE_COLUMNS, fill_value=0)
            
            for idx in range(len(aligned_features)):
                row_data = {
                    "Timestamp": timestamp,
                    "Predicted_Label": int(predictions[idx]),
                    "Confidence_Score": float(probabilities[idx, 1]) if probabilities is not None else float(predictions[idx]),
                }
                # Add all 78 features
                for col in TRAINING_FEATURE_COLUMNS:
                    row_data[col] = aligned_features.iloc[idx][col]
                
                # Put row in queue (non-blocking)
                self._queue.put(row_data, block=False)
        except Exception as e:
            print(f"⚠️ [TrafficLogger] Log hatası: {e}")
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


# Global TrafficLogger instance (initialized after model loading)
TRAFFIC_LOGGER: TrafficLogger = None

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

def feature_extraction_and_predict():
    GOLD_STANDARD_FEATURES = [
        "src_ip", "dst_ip", "src_port", "dst_port", "protocol", "timestamp",
        "flow_duration", "flow_byts_s", "flow_pkts_s", "fwd_pkts_s", "bwd_pkts_s",
        "tot_fwd_pkts", "tot_bwd_pkts", "totlen_fwd_pkts", "totlen_bwd_pkts",
        "fwd_pkt_len_max", "fwd_pkt_len_min", "fwd_pkt_len_mean", "fwd_pkt_len_std",
        "bwd_pkt_len_max", "bwd_pkt_len_min", "bwd_pkt_len_mean", "bwd_pkt_len_std",
        "pkt_len_max", "pkt_len_min", "pkt_len_mean", "pkt_len_std", "pkt_len_var",
        "fwd_header_len", "bwd_header_len", "fwd_seg_size_min", "fwd_act_data_pkts",
        "flow_iat_mean", "flow_iat_max", "flow_iat_min", "flow_iat_std",
        "fwd_iat_tot", "fwd_iat_max", "fwd_iat_min", "fwd_iat_mean", "fwd_iat_std",
        "bwd_iat_tot", "bwd_iat_max", "bwd_iat_min", "bwd_iat_mean", "bwd_iat_std",
        "fwd_psh_flags", "bwd_psh_flags", "fwd_urg_flags", "bwd_urg_flags",
        "fin_flag_cnt", "syn_flag_cnt", "rst_flag_cnt", "psh_flag_cnt",
        "ack_flag_cnt", "urg_flag_cnt", "ece_flag_cnt", "down_up_ratio",
        "pkt_size_avg", "init_fwd_win_byts", "init_bwd_win_byts",
        "active_max", "active_min", "active_mean", "active_std",
        "idle_max", "idle_min", "idle_mean", "idle_std",
        "fwd_byts_b_avg", "fwd_pkts_b_avg", "bwd_byts_b_avg", "bwd_pkts_b_avg",
        "fwd_blk_rate_avg", "bwd_blk_rate_avg", "fwd_seg_size_avg", "bwd_seg_size_avg",
        "cwr_flag_count", "subflow_fwd_pkts", "subflow_bwd_pkts",
        "subflow_fwd_byts", "subflow_bwd_byts",
    ]
    LOG_ONLY_COLUMNS = [
        "src_ip", "dst_ip", "src_port", "dst_port", "protocol", "timestamp",
    ]

    print("   ↳ ⚙️ Analiz ediliyor...", end="\r")

    cli_ok, cli_err = run_cicflowmeter_cli(TEMP_PCAP, TEMP_CSV)
    if not cli_ok:
        print(f"   ⚠️ CLI başarısız: {cli_err[:110]} -> API moduna geçiliyor")
        api_ok, api_err = run_cicflowmeter_api(TEMP_PCAP, TEMP_CSV)
        if not api_ok:
            print(f"   ❌ HATA: CSV oluşmadı ({api_err[:110]})       ")
            return

    if not os.path.exists(TEMP_CSV):
        print("   ❌ HATA: CSV oluşmadı (bilinmeyen neden)")
        return

    try:
        df = pd.read_csv(TEMP_CSV)
    except Exception as exc:
        print(f"   ❌ CSV okunamadı: {exc}")
        return

    if df.empty:
        print("   ⚠️ CSV boş, paket analiz edilemedi.          ")
        return

    df.columns = df.columns.str.strip()
    src_ips = extract_source_ips(df)

    features = prepare_feature_frame(df)
    features.replace([float("inf"), float("-inf")], 0, inplace=True)
    features.fillna(0, inplace=True)

    # Zorunlu sıralama ve sütun tamamlama
    features = features.reindex(columns=GOLD_STANDARD_FEATURES, fill_value=0)

    # Kimlik sütunlarını model girişinden çıkar
    model_features = features.drop(columns=LOG_ONLY_COLUMNS, errors="ignore")

    try:
        scaled_array = SCALER.transform(model_features)
        X_scaled = pd.DataFrame(
            scaled_array,
            columns=model_features.columns,
            index=model_features.index,
        )
        predictions = MODEL.predict(X_scaled)
    except ValueError as exc:
        print(f"⚠️ Ölçekleme/Tahmin hatası: {exc}")
        return

    attack_detected = False
    for idx, pred in enumerate(predictions):
        if pred != 1:
            continue
        attack_detected = True
        ip_addr = src_ips.iloc[idx] if src_ips is not None else "Bilinmiyor"
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n🚨 [{timestamp}] TEHDİT ALGILANDI! Kaynak IP: {ip_addr}")
        if ip_addr and ip_addr not in WHITELIST_IPS and ip_addr != "Bilinmiyor":
            block_ip(ip_addr)
            log_attack(ip_addr, "BLOCKED", "Attack Detected")
        else:
            print("   ✅ IP beyaz listede veya bilinmiyor, engellenmedi.")
            log_attack(ip_addr, "ALLOWED", "Whitelisted")

    if not attack_detected:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"✅ [{timestamp}] Trafik Temiz - Güvenli ({len(predictions)} Akış)")
        normal_indices = [i for i, pred in enumerate(predictions) if pred == 0][:5]
        for idx in normal_indices:
            ip_addr = (
                src_ips.iloc[idx]
                if (src_ips is not None and idx < len(src_ips))
                else "Unknown/Local"
            )
            log_attack(ip_addr, "NORMAL", "Clean Traffic")




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

print("🛡️  AI NETWORK IPS - SİBER GÜVENLİK KALKANI")
print("=" * 60)

if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
    print("❌ Model veya scaler bulunamadı. Önce eğitim pipeline'ını çalıştırın.")
    sys.exit(1)

try:
    print("⏳ Modeller yükleniyor...", end="\r")
    MODEL = joblib.load(MODEL_PATH)
    SCALER = joblib.load(SCALER_PATH)
    print("✅ Yapay Zeka Modeli (Random Forest) Aktif.        ")
    
    # Initialize TrafficLogger for data harvesting
    TRAFFIC_LOGGER = TrafficLogger()
except Exception as exc:
    print(f"❌ Model yükleme hatası: {exc}")
    sys.exit(1)

if shutil.which("cicflowmeter") is None:
    print("\n⚠️  UYARI: 'cicflowmeter' CLI bulunamadı (pip install cicflowmeter)")


# ---------------------------------------------------------------------------
# CICFLOWMETER HELPERS
# ---------------------------------------------------------------------------

def run_cicflowmeter_cli(pcap_file: str, csv_file: str):
    """Run cicflowmeter CLI and return (success, error_message)."""
    # Yöntem 1: python -m cicflowmeter (Windows için daha güvenilir)
    cmd = [sys.executable, "-m", "cicflowmeter", "-f", pcap_file, "-c", csv_file]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    except Exception as exc:
        return False, str(exc)

    if result.returncode != 0:
        # Eğer modül olarak çalışmazsa, doğrudan komut olarak dene
        if shutil.which("cicflowmeter"):
            try:
                cmd_direct = ["cicflowmeter", "-f", pcap_file, "-c", csv_file]
                result = subprocess.run(cmd_direct, capture_output=True, text=True, timeout=90)
            except Exception:
                pass
        
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
    """Fallback runner that streams packets through FlowSession manually."""
    try:
        from cicflowmeter.flow_session import FlowSession
    except ImportError as exc:  # pragma: no cover
        return False, f"FlowSession import edilemedi: {exc}"

    try:
        packets = rdpcap(pcap_file)
        if len(packets) == 0:
            return False, "PCAP dosyası boş"

# Ağ Kartı Bulucu
def get_active_interface():
    for iface in conf.ifaces.values():
        if iface.ip and iface.ip != "127.0.0.1" and iface.ip != "0.0.0.0":
            if "Wi-Fi" in iface.name or "Wireless" in iface.name or "Ethernet" in iface.name:
                return iface.name
    return conf.iface # Bulamazsa varsayılanı dön

INTERFACE = get_active_interface()
TEMP_PCAP = "temp_live.pcap"
TEMP_CSV = "temp_live.csv"
WHITELIST_IPS = ["192.168.1.1", "127.0.0.1", "0.0.0.0", "8.8.8.8"] # Modem vs.

print(f"\n🛡️  SİSTEM BAŞLATILDI | Arayüz: {INTERFACE}")

# Model Yükle
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def feature_extraction_and_predict():
    print("   ↳ ⚙️ Analiz...", end="\r")
    
    # 1. CICFlowMeter Çalıştır
    cmd = f"cicflowmeter -f {TEMP_PCAP} -c {TEMP_CSV} > nul 2>&1"
    os.system(cmd)
    
    if not os.path.exists(TEMP_CSV):
        # Fallback: Python ile CSV üret (Eğer Java çalışmazsa)
        try:
            from cicflowmeter.flow_session import FlowSession
            flow_session = FlowSession()
            sniff(offline=TEMP_PCAP, prn=flow_session.on_packet, store=False)
            flow_session.to_csv(TEMP_CSV)
        except:
            return

    try:
        df = pd.read_csv(TEMP_CSV)
    except:
        return

    if df.empty: return

    # IP Adreslerini Sakla
    src_ips = df.get('Src IP', df.get('Source IP'))
    dst_ips = df.get('Dst IP', df.get('Destination IP'))

    # 2. SÜTUN EŞİTLEME (EN KRİTİK KISIM)
    # Gelen veriyi modelin beklediği 78 sütuna zorluyoruz. Eksikleri 0 yapıyoruz.
    # Önce sütun isimlerini temizle
    df.columns = df.columns.str.strip()
    
    # Reindex ile sıralamayı ve sayıyı sabitle
    # Not: Gelen CSV'deki isimler ile GOLD_STANDARD listesindeki isimler bazen farklı olabilir.
    # Basitlik için sadece sayısal sütunları alıp, eksikleri dolduracağız.
    
    # Model için sadece sayısal veriyi hazırla
    # Eğitimdeki feature isimleri ile buradakileri eşleştirmek zor olduğu için
    # scaler'ın beklediği boyuta (78) getirmek için reindex kullanıyoruz.
    features = df.reindex(columns=GOLD_STANDARD_FEATURES, fill_value=0)
    
    # Sonsuz değerleri temizle
    features.replace([float('inf'), float('-inf')], 0, inplace=True)
    features.fillna(0, inplace=True)

    try:
        scaled_array = SCALER.transform(features)
        X_scaled = pd.DataFrame(
            scaled_array,
            columns=features.columns,
            index=features.index,
        )
        predictions = MODEL.predict(X_scaled)
        
        # Get probability scores for confidence logging
        try:
            probabilities = MODEL.predict_proba(X_scaled)
        except AttributeError:
            # Model doesn't support predict_proba, use predictions as fallback
            probabilities = None
        
        # === DATA HARVEST: Log all traffic to CSV ===
        if TRAFFIC_LOGGER is not None:
            TRAFFIC_LOGGER.log(features, predictions, probabilities)
        
    except ValueError as exc:
        print(f"⚠️ Ölçekleme/Tahmin hatası: {exc}")
        return

    attack_detected = False
    for idx, pred in enumerate(predictions):
        if pred != 1:
            continue
        attack_detected = True
        ip_addr = src_ips.iloc[idx] if src_ips is not None else "Bilinmiyor"
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n🚨 [{timestamp}] TEHDİT ALGILANDI! Kaynak IP: {ip_addr}")
        if ip_addr and ip_addr not in WHITELIST_IPS and ip_addr != "Bilinmiyor":
            block_ip(ip_addr)
            log_attack(ip_addr, "BLOCKED", "Attack Detected")
        else:
            print("   ✅ IP beyaz listede veya bilinmiyor, engellenmedi.")
            log_attack(ip_addr, "ALLOWED", "Whitelisted")

    if not attack_detected:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"✅ [{timestamp}] Trafik Temiz - Güvenli ({len(predictions)} Akış)")
        normal_indices = [i for i, pred in enumerate(predictions) if pred == 0][:5]
        for idx in normal_indices:
            if src_ips is not None and idx < len(src_ips):
                ip_addr = src_ips.iloc[idx]
            else:
                if i < 2: # Sadece ilk 2 paketi kaydet, DB şişmesin
                    log_attack(ip_src, "NORMAL", "Clean Traffic")

        if not saldirgan_var_mi:
            print(f"✅ Trafik Temiz ({len(predictions)} Akış)           ", end="\r")

    except Exception as e:
        # print(f"Hata: {e}")
        pass

def main_loop():
    global TRAFFIC_LOGGER
    
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

            # Show logger stats periodically
            iteration_count += 1
            if iteration_count % stats_interval == 0 and TRAFFIC_LOGGER is not None:
                stats = TRAFFIC_LOGGER.get_stats()
                print(f"📊 [Data Harvest] Buffer: {stats['buffer_size']}/{HARVEST_BUFFER_SIZE} | Toplam Kayıt: {stats['total_rows']}")

            # Disk temizliği geçici olarak kapatıldı (analiz için dosyalar korunsun)
            # if os.path.exists(TEMP_PCAP):
            #     os.remove(TEMP_PCAP)
            # if os.path.exists(TEMP_CSV):
            #     os.remove(TEMP_CSV)
        except KeyboardInterrupt:
            print("\n🛑 Sistem kullanıcı tarafından durduruldu.")
            # Ensure TrafficLogger flushes remaining data
            if TRAFFIC_LOGGER is not None:
                TRAFFIC_LOGGER.shutdown()
            break
        except:
            time.sleep(1)

if __name__ == "__main__":
    main_loop()
