# NIDS PoC — Hizli Baslangic (Windows)
# Kullanim: .\scripts\start_all.ps1
# Not: live_bridge admin yetkisi gerektirir — ayri terminal acin.

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Push-Location $ProjectRoot
try {
    Write-Host "`n=== NIDS PoC Baslangic ===" -ForegroundColor Cyan

    # 1. Kafka
    Write-Host "`n[1/3] Kafka baslatiliyor..." -ForegroundColor Yellow
    docker compose up -d
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Kafka baslatilamadi. Docker Desktop calisiyormu?" -ForegroundColor Red
        exit 1
    }
    Write-Host "Kafka OK" -ForegroundColor Green

    # 2. Dashboard
    Write-Host "`n[2/3] Dashboard baslatiliyor..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$ProjectRoot'; & '.\venv\Scripts\Activate.ps1'; streamlit run src/dashboard/app.py --server.port 8501"
    Write-Host "Dashboard baslatildi (http://localhost:8501)" -ForegroundColor Green

    # 3. Consumer
    Write-Host "`n[3/3] Consumer baslatiliyor..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$ProjectRoot'; & '.\venv\Scripts\Activate.ps1'; python src/kafka_consumer.py"
    Write-Host "Consumer baslatildi" -ForegroundColor Green

    Write-Host "`n=== Tamamlandi ===" -ForegroundColor Cyan
    Write-Host "Dashboard: http://localhost:8501"
    Write-Host ""
    Write-Host "live_bridge icin Yonetici PowerShell acin:" -ForegroundColor Yellow
    Write-Host "  cd $ProjectRoot"
    Write-Host "  .\venv\Scripts\Activate.ps1"
    Write-Host "  python src/live_bridge.py"
    Write-Host ""
}
finally {
    Pop-Location
}
