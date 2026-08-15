#!/usr/bin/env bash
# NIDS PoC — Hizli Baslangic (Linux)
# Kullanim: bash scripts/start_all.sh
# Not: live_bridge root yetkisi gerektirir — ayri terminal acin.

set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo ""
echo "=== NIDS PoC Baslangic ==="

# 1. Kafka
echo ""
echo "[1/3] Kafka baslatiliyor..."
docker compose up -d
echo "Kafka OK"

# 2. Activate venv
source venv/bin/activate

# 3. Dashboard
echo ""
echo "[2/3] Dashboard baslatiliyor..."
streamlit run src/dashboard/app.py --server.port 8501 &
DASHBOARD_PID=$!
echo "Dashboard baslatildi (PID: $DASHBOARD_PID, http://localhost:8501)"

# 4. Consumer
echo ""
echo "[3/3] Consumer baslatiliyor..."
python src/kafka_consumer.py &
CONSUMER_PID=$!
echo "Consumer baslatildi (PID: $CONSUMER_PID)"

echo ""
echo "=== Tamamlandi ==="
echo "Dashboard: http://localhost:8501"
echo "Durdurmak icin: kill $DASHBOARD_PID $CONSUMER_PID && docker compose down"
echo ""
echo "live_bridge icin ayri terminal acin:"
echo "  cd $PROJECT_ROOT"
echo "  source venv/bin/activate"
echo "  sudo python src/live_bridge.py"
echo ""

wait
