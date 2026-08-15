# PoC Runbook — Network Intrusion Detection System

Bu belge, sistemi sifirdan kurup calistirmak icin gereken tum adimlari icerir.
Hedef kitle: teknik PoC operatoru (muhendis veya guvenlik analisti).

---

## Mimari Ozet

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐
│  live_bridge │───►│ CICFlowMeter │───►│    Kafka      │───►│  Consumer  │
│  (Scapy      │    │ (feature     │    │  (Zookeeper   │    │  (predict  │
│   capture)   │    │  extraction) │    │   + Broker)   │    │  +escalate)│
└─────────────┘    └──────────────┘    └──────────────┘    └──────┬─────┘
                                                                  │
                                                            ┌─────▼──────┐
                                                            │  Dashboard │
                                                            │ (Streamlit │
                                                            │   :8501)   │
                                                            └────────────┘
```

**Servis baslama sirasi:** Kafka → Dashboard → Consumer → Producer (live_bridge)  
**Durdurma sirasi:** Producer → Consumer → Dashboard → Kafka

---

## A. Onkosullar

| Gereksinim | Windows | Linux (Ubuntu 22.04+) |
|------------|---------|------------------------|
| Python | 3.11.x | 3.11.x (`sudo apt install python3.11 python3.11-venv`) |
| Docker | Docker Desktop | docker-ce + docker-compose-plugin |
| Git | Git for Windows | `sudo apt install git` |
| Paket yakalama | **Npcap** (https://npcap.com) | `sudo apt install libpcap-dev` |
| Admin yetkisi | Scapy ag yakalama icin Yonetici olarak calistir | `sudo` veya `setcap cap_net_raw+eip` |

**Docker dogrulamasi:**

```bash
docker --version          # 20.10+ olmali
docker compose version    # v2+ olmali
```

---

## B. Sifirdan Kurulum

### B1 — Repo'yu klonla

```bash
git clone <repo-url> networkdetection
cd networkdetection
```

### B2 — Sanal ortam olustur

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux:**

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### B3 — Bagimliliklari kur

```bash
pip install -r requirements.txt
pip install "shap<0.50"
```

> **Not:** `numpy==1.26.4` kalmalidir. `shap>=0.50` numpy 2.x gerektirir ve
> tensorflow / cicflowmeter ile catisir. `shap<0.50` bunu onler.

### B4 — Ortam degiskenlerini yapilandir

```bash
cp .env.example .env
```

`.env` dosyasini duzenle — en az su degiskenleri ayarla:

| Degisken | Ne yapmali |
|----------|-----------|
| `NETWORK_INTERFACE` | Dinlenecek ag arayuzu adi (asagiya bak) |
| `TARGET_IP` | Izlenecek hedef IP |
| `WHITELIST_IPS` | Engellenmeyecek IP'ler (virgul ayirmali) |

**Ag arayuzu adini bulma:**

```python
# Windows:
python -c "from scapy.arch.windows import get_windows_if_list; [print(i['name'], '-', i['description']) for i in get_windows_if_list()]"

# Linux:
ip link show
```

### B5 — Model artefaktlarini dogrula

`models/` dizininde asagidaki dosyalar olmalidir:

```
models/
├── rf_3class_model.pkl          (Random Forest)
├── dt_3class_model.pkl          (Decision Tree)
├── xgb_3class_model.pkl         (XGBoost)
├── lstm_best.keras              (LSTM — sadece offline)
├── bilstm_best.keras            (BiLSTM — sadece offline)
├── scaler.pkl                   (tabular model scaler)
├── scaler_lstm.pkl              (sequence model scaler)
├── shap_explainer.pkl           (SHAP TreeExplainer, ~25 MB)
├── top_20_features.json         (SHAP feature listesi)
├── rf_3class_config.json        (RF hiperparametre/metrik)
├── xgb_3class_config.json       (XGBoost hiperparametre/metrik)
├── lstm_config.json             (LSTM mimari config)
└── bilstm_config.json           (BiLSTM mimari config)
```

Hizli kontrol:

```python
python scripts/health_check.py
```

### B6 — Kafka altyapisini baslat

```bash
docker compose up -d
```

Dogrula:

```bash
docker ps
# Iki container gorunmeli: network-ips-zookeeper, network-ips-kafka
```

### B7 — Varsayilan model ayarla

```bash
# Yoksa olustur:
echo "Random Forest" > data/active_model.txt
```

---

## C. Gunluk Baslatma

Dort ayri terminal ac. Her terminalde once sanal ortami aktif et.

### Terminal 1 — Kafka

```bash
docker compose up -d
```

Zaten calisiyorsa bu adimi atla.

### Terminal 2 — Dashboard

```bash
streamlit run src/dashboard/app.py --server.port 8501
```

Tarayicida ac: http://localhost:8501

### Terminal 3 — Consumer

```bash
python src/kafka_consumer.py
```

Basarili cikti:

```
🚀 Consumer is now ACTIVE and listening for messages...
   Bootstrap servers: 127.0.0.1:9092
   Topic: network-traffic
```

### Terminal 4 — Producer (live_bridge)

```powershell
# Windows (Yonetici PowerShell):
python src/live_bridge.py

# Linux:
sudo venv/bin/python src/live_bridge.py
```

> **Not:** Ag yakalama root/admin yetkisi gerektirir. Linux'ta alternatif:
> `sudo setcap cap_net_raw+eip venv/bin/python3.11` ile capability verilebilir.

### Hizli Baslangic (opsiyonel)

```powershell
# Windows:
.\scripts\start_all.ps1

# Linux:
bash scripts/start_all.sh
```

---

## D. Durdurma

Ters sirada durdur:

```
Terminal 4: Ctrl+C    → live_bridge durur, Kafka producer flush edilir
Terminal 3: Ctrl+C    → Consumer durur, CSV tamamlanir
Terminal 2: Ctrl+C    → Dashboard durur
Terminal 1: docker compose down    → Kafka + Zookeeper kapanir
```

> **Onemli:** Consumer'i durdurmadan Kafka'yi kapatmayin — veri kaybi olabilir.

---

## E. Ortam Degiskenleri Referansi

`.env` dosyasindaki tum degiskenler:

| Degisken | Varsayilan | Aciklama |
|----------|-----------|----------|
| `NETWORK_INTERFACE` | `Wi-Fi` | Scapy yakalama arayuzu. Windows'ta adapter adi veya GUID, Linux'ta `eth0`/`wlan0` |
| `TARGET_IP` | `192.168.1.1` | Izlenecek hedef IP adresi |
| `WHITELIST_IPS` | `192.168.1.1,127.0.0.1,0.0.0.0,localhost` | Asla engellenmeyecek IP'ler (virgul ayirmali) |
| `BLOCK_TTL_SECONDS` | `3600` | Otomatik IP engeli suresi (saniye). Suredolunca engel kalkar |
| `ESCALATION_WINDOW_SECONDS` | `60` | Eskalasyon penceresi — bu sure icindeki tespit sayisi sayilir |
| `ESCALATION_SUSPICIOUS_THRESHOLD` | `2` | Bu sayida tespit → SUSPICIOUS |
| `ESCALATION_BLOCK_THRESHOLD` | `4` | Bu sayida tespit → BLOCKED (otomatik engel) |
| `BUCKET_FREQUENCY` | `10s` | Dashboard zaman-serisi gruplama frekansi (pandas freq: `10s`, `30s`, `1min`) |
| `KAFKA_GROUP_ID` | `nids-consumer-group-v2` | Kafka consumer group ID |
| `KAFKA_AUTO_OFFSET_RESET` | `latest` | Yeni consumer icin baslangic: `latest` (son) veya `earliest` (bas) |

### Eskalasyon ornegi

Varsayilan degerlerle: 60 saniye icinde ayni IP'den 2 tespit → SUSPICIOUS, 4 tespit → BLOCKED.

---

## F. Sorun Giderme ve Rollback

| Sorun | Belirti | Cozum |
|-------|---------|-------|
| **Kafka baglanamadi** | Consumer/Producer baslarken hata veriyor | `docker compose down && docker compose up -d` — container'larin ayakta oldugundan emin ol |
| **Port 9092 mesgul** | Kafka baslatma hatasi | Baska Kafka instance calisiyorsa durdur: `docker ps` ile kontrol et |
| **CSV sema uyumsuzlugu** | Consumer `.invalid_*` dosyasi olusturdu | Otomatik davranis — eski CSV yedeklenir, yeni CSV dogru semayla olusturulur. Mudahale gerekmez |
| **Dashboard acilamadi** | Port 8501 mesgul | `streamlit run src/dashboard/app.py --server.port 8502` ile farkli port kullan |
| **Model yuklenemedi** | Consumer hata veriyor | `data/active_model.txt` icerigi kontrol et — `Random Forest`, `Decision Tree` veya `XGBoost` olmali |
| **LSTM/BiLSTM secildi** | Dashboard uyari gosteriyor | Normal — bu modeller canli hatta desteklenmez (sadece offline). Sistem otomatik olarak varsayilan modele doner |
| **CICFlowMeter hatasi** | Feature extraction basarisiz | `pip install cicflowmeter` dogrula. Basarisiz olursa live_bridge dummy feature fallback kullanir |
| **Scapy import hatasi** | Npcap/libpcap bulunamadi | Windows: Npcap yukle (https://npcap.com). Linux: `sudo apt install libpcap-dev` |
| **XAI sekmesi bos** | SHAP panel icerik gostermiyor | `pip install "shap<0.50"` dogrula. Consumer calisiyorken canli trafik uretilmeli |
| **numpy catismasi** | tensorflow veya scipy hatalari | `pip install "numpy==1.26.4"` — 2.x versiyonu yuklenmesin |
| **alerts.db kilit hatasi** | DB yazma hatasi | Consumer'i durdur, `src/alerts.db` sil (tekrar otomatik olusturulur), consumer'i baslat |
| **Veri akmiyorr** | Dashboard bos | (1) live_bridge calisiyormu? (2) Consumer Kafka'ya bagli mi? (3) `docker ps` ile Kafka kontrol |

### Temiz Baslangic (Rollback)

Tum calisma-zamani verisini sifirlamak icin:

```bash
# Consumer ciktisini sifirla
rm data/live_captured_traffic.csv

# Alert veritabanini sifirla
rm src/alerts.db

# Varsayilan modele don
echo "Random Forest" > data/active_model.txt
```

Servisler yeniden baslatildiginda bu dosyalar otomatik olusturulur.

---

## G. Dosya Yapisi

```
networkdetection/
├── .env                           ← Ortam degiskenleri (git'te degil)
├── .env.example                   ← Sablon
├── docker-compose.yml             ← Kafka + Zookeeper
├── requirements.txt               ← Python bagimliliklari
│
├── models/                        ← Egitilmis model artefaktlari
│   ├── rf_3class_model.pkl        ← Random Forest (varsayilan canli model)
│   ├── dt_3class_model.pkl        ← Decision Tree
│   ├── xgb_3class_model.pkl       ← XGBoost
│   ├── lstm_best.keras            ← LSTM (sadece offline)
│   ├── bilstm_best.keras          ← BiLSTM (sadece offline)
│   ├── scaler.pkl                 ← Tabular model scaler
│   ├── scaler_lstm.pkl            ← Sequence model scaler
│   ├── shap_explainer.pkl         ← SHAP TreeExplainer (~25 MB)
│   └── top_20_features.json       ← SHAP feature isimleri
│
├── data/                          ← Calisma-zamani verileri
│   ├── active_model.txt           ← Aktif model (hot-swap icin)
│   └── live_captured_traffic.csv  ← Consumer tahmin ciktisi
│
├── src/
│   ├── config.py                  ← Merkezi yapilandirma (TOP_FEATURES, eskalasyon)
│   ├── model_registry.py          ← 5 model kayit defteri
│   ├── kafka_consumer.py          ← Kafka → tahmin → eskalasyon → CSV
│   ├── live_bridge.py             ← Scapy yakalama → CICFlowMeter → Kafka
│   ├── dashboard/
│   │   └── app.py                 ← Streamlit web paneli
│   └── utils/
│       ├── db_manager.py          ← SQLite alert kayit (alerts.db)
│       ├── firewall_manager.py    ← IP engelleme
│       └── xai_engine.py          ← SHAP aciklama motoru
│
├── scripts/
│   ├── start_all.ps1              ← Windows hizli baslangic
│   ├── start_all.sh               ← Linux hizli baslangic
│   └── health_check.py            ← Kurulum dogrulama
│
├── docs/
│   └── poc_runbook.md             ← Bu belge
│
└── reports/                       ← Analiz ve denetim sonuclari
```

---

## H. Canli Model Degistirme (Hot-Swap)

Dashboard uzerinden aktif model degistirilebilir:

1. Dashboard → Admin sekmesi → Model secimi
2. **Desteklenen modeller:** Random Forest, Decision Tree, XGBoost
3. **Desteklenmeyen:** LSTM, BiLSTM (sadece offline benchmark)
4. Secim yapildiginda `data/active_model.txt` guncellenir
5. Consumer bir sonraki polling'de yeni modeli yukler (5 sn icinde)

Manuel degistirme:

```bash
echo "XGBoost" > data/active_model.txt
# Consumer otomatik algilayip model degistirir
```

---

## I. Dogrulama Kontrol Listesi

Kurulum tamamlandiktan sonra su kontrolleri yap:

- [ ] `docker ps` — zookeeper ve kafka container'lari calisiyorr
- [ ] http://localhost:8501 — Dashboard acilir
- [ ] Consumer terminalinde "Consumer is now ACTIVE" mesaji gorunur
- [ ] `.env` dosyasi mevcut ve NETWORK_INTERFACE dogru ayarli
- [ ] `models/` dizininde tum artefaktlar mevcut (`python scripts/health_check.py`)
- [ ] Dashboard XAI sekmesi hatasiz acilir
- [ ] `data/active_model.txt` icerigi gecerli bir model adi

---

*Son guncelleme: Agustos 2026*
