"""NIDS PoC health check — verify all prerequisites before starting services."""

import os
import socket
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REQUIRED_MODEL_FILES = [
    "rf_3class_model.pkl",
    "dt_3class_model.pkl",
    "xgb_3class_model.pkl",
    "lstm_best.keras",
    "bilstm_best.keras",
    "scaler.pkl",
    "scaler_lstm.pkl",
    "shap_explainer.pkl",
    "top_20_features.json",
    "rf_3class_config.json",
    "xgb_3class_config.json",
    "lstm_config.json",
    "bilstm_config.json",
]

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"


def check_port(host, port, timeout=2):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, ConnectionRefusedError):
        return False


def main():
    print("\n=== NIDS PoC Health Check ===\n")
    all_ok = True

    # 1. .env
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if os.path.exists(env_path):
        print(f"  [{PASS}] .env dosyasi mevcut")
    else:
        print(f"  [{FAIL}] .env dosyasi bulunamadi — .env.example'dan kopyalayin")
        all_ok = False

    # 2. Model artifacts
    models_dir = os.path.join(PROJECT_ROOT, "models")
    missing = []
    for f in REQUIRED_MODEL_FILES:
        if not os.path.exists(os.path.join(models_dir, f)):
            missing.append(f)
    if not missing:
        print(f"  [{PASS}] Tum model artefaktlari mevcut ({len(REQUIRED_MODEL_FILES)} dosya)")
    else:
        print(f"  [{FAIL}] Eksik model dosyalari: {', '.join(missing)}")
        all_ok = False

    # 3. data/ directory
    data_dir = os.path.join(PROJECT_ROOT, "data")
    if os.path.isdir(data_dir):
        print(f"  [{PASS}] data/ dizini mevcut")
    else:
        print(f"  [{FAIL}] data/ dizini bulunamadi")
        all_ok = False

    # 4. active_model.txt
    active_model = os.path.join(data_dir, "active_model.txt")
    valid_models = {"Random Forest", "Decision Tree", "XGBoost"}
    if os.path.exists(active_model):
        with open(active_model) as f:
            model_name = f.read().strip()
        if model_name in valid_models:
            print(f"  [{PASS}] active_model.txt = {model_name}")
        else:
            print(f"  [{WARN}] active_model.txt = '{model_name}' (beklenen: {valid_models})")
    else:
        print(f"  [{WARN}] active_model.txt yok — ilk baslatmada varsayilan (Random Forest) kullanilir")

    # 5. Kafka port
    if check_port("127.0.0.1", 9092):
        print(f"  [{PASS}] Kafka broker erisilebilir (127.0.0.1:9092)")
    else:
        print(f"  [{WARN}] Kafka broker erisilemedi (127.0.0.1:9092) — docker compose up -d calistirin")

    # 6. Zookeeper port
    if check_port("127.0.0.1", 2181):
        print(f"  [{PASS}] Zookeeper erisilebilir (127.0.0.1:2181)")
    else:
        print(f"  [{WARN}] Zookeeper erisilemedi (127.0.0.1:2181)")

    # 7. Dashboard port
    if check_port("127.0.0.1", 8501):
        print(f"  [{PASS}] Dashboard erisilebilir (127.0.0.1:8501)")
    else:
        print(f"  [{WARN}] Dashboard erisilemedi (127.0.0.1:8501) — henuz baslatilmamis olabilir")

    # 8. Python packages
    pkg_issues = []
    for pkg in ["sklearn", "pandas", "streamlit", "confluent_kafka", "scapy", "joblib"]:
        try:
            __import__(pkg)
        except ImportError:
            pkg_issues.append(pkg)
    if not pkg_issues:
        print(f"  [{PASS}] Temel Python paketleri yuklu")
    else:
        print(f"  [{FAIL}] Eksik Python paketleri: {', '.join(pkg_issues)}")
        all_ok = False

    # 9. shap
    try:
        import shap  # noqa: F401
        print(f"  [{PASS}] shap paketi yuklu (v{shap.__version__})")
    except ImportError:
        print(f"  [{WARN}] shap paketi yuklu degil — XAI sekmesi calismaz (pip install 'shap<0.50')")

    # 10. requirements.txt
    req_path = os.path.join(PROJECT_ROOT, "requirements.txt")
    if os.path.exists(req_path):
        print(f"  [{PASS}] requirements.txt mevcut")
    else:
        print(f"  [{FAIL}] requirements.txt bulunamadi")
        all_ok = False

    print()
    if all_ok:
        print(f"  Sonuc: Tum zorunlu kontroller {PASS}")
    else:
        print(f"  Sonuc: Bazi kontroller {FAIL} — yukaridaki hatalari giderin")
    print()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
