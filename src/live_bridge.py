#!/usr/bin/env python3
"""Live IPS bridge that captures traffic, extracts CICFlowMeter features,
scales them with the training pipeline, and blocks offending IPs."""

import os
import sys
import time
import shutil
import subprocess
from datetime import datetime

import joblib
import pandas as pd
from scapy.all import sniff, wrpcap, rdpcap

# ---------------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "rf_model_v1.pkl")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")

sys.path.append(os.path.join(CURRENT_DIR, "utils"))
try:
    from firewall_manager import block_ip
except ImportError:
    print("⚠️ UYARI: firewall_manager.py bulunamadı, IP engelleme devre dışı.")

    def block_ip(ip_address):  # fallback to keep pipeline alive
        print(f"   [Simülasyon] {ip_address} engellenecekti.")

try:
    from db_manager import log_attack
except ImportError:
    def log_attack(*_args, **_kwargs):
        pass

# ---------------------------------------------------------------------------
# RUNTIME CONFIG
# ---------------------------------------------------------------------------
# Scapy interface name must match show_interfaces() output exactly.
INTERFACE = "Wi-Fi"
TEMP_PCAP = "temp_live.pcap"
TEMP_CSV = "temp_live.csv"
WHITELIST_IPS = ["192.168.1.1", "127.0.0.1", "0.0.0.0", "localhost"]
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
    if shutil.which("cicflowmeter") is None:
        return False, "cicflowmeter CLI bulunamadı"

    cmd = ["cicflowmeter", "-f", pcap_file, "-c", csv_file]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    except Exception as exc:  # pragma: no cover - runtime safety
        return False, str(exc)

    if result.returncode != 0:
        err = result.stderr.strip() or result.stdout.strip() or f"CLI hata kodu {result.returncode}"
        return False, err

    if not os.path.exists(csv_file):
        err = result.stderr.strip() or "CLI çalıştı fakat CSV bulunamadı"
        return False, err

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

        session = FlowSession(output_mode="csv", output=csv_file, fields=None, verbose=False)
        for packet in packets:
            session.process(packet)
        session.flush_flows()

        if not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0:
            return False, "FlowSession CSV üretmedi"
        return True, None
    except Exception as exc:  # pragma: no cover
        return False, str(exc)


# ---------------------------------------------------------------------------
# CORE PIPELINE
# ---------------------------------------------------------------------------

def feature_extraction_and_predict():
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

    try:
        scaled_array = SCALER.transform(features)
        X_scaled = pd.DataFrame(
            scaled_array,
            columns=features.columns,
            index=features.index,
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
            if src_ips is not None and idx < len(src_ips):
                ip_addr = src_ips.iloc[idx]
            else:
                ip_addr = "Unknown/Local"
            log_attack(ip_addr, "NORMAL", "Clean Traffic")


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------

def main_loop():
    print(f"\n📡 Ağ Dinleniyor: {INTERFACE}")
    print("⏹️  Durdurmak için CTRL+C yapın...\n")

    while True:
        try:
            print("⏳ Paket toplanıyor...", end="\r")
            packets = sniff(iface=INTERFACE, timeout=4)
            packet_count = len(packets)

            if packet_count > 0:
                print(f"📦 {packet_count} Paket Yakalandı -> İşleniyor...      ", end="\r")
                wrpcap(TEMP_PCAP, packets)
                feature_extraction_and_predict()
            else:
                print(f"⚠️ 0 Paket! '{INTERFACE}' ismini kontrol et.        ", end="\r")

            # Disk temizliği geçici olarak kapatıldı (analiz için dosyalar korunsun)
            # if os.path.exists(TEMP_PCAP):
            #     os.remove(TEMP_PCAP)
            # if os.path.exists(TEMP_CSV):
            #     os.remove(TEMP_CSV)
        except KeyboardInterrupt:
            print("\n🛑 Sistem kullanıcı tarafından durduruldu.")
            break
        except Exception as exc:
            print(f"\n❌ Beklenmedik Hata: {exc}")
            time.sleep(1)


if __name__ == "__main__":
    main_loop()
