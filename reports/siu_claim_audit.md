# SIU 2026 (394) ↔ Repo Iddia Denetimi

**Tarih:** 2026-08-15 15:19:04
**Script:** `tools/siu_claim_audit.py`

---

## Ozet

| Durum | Sayi |
|-------|------|
| UYUMLU | 38/47 |
| SAPMA | 6/47 |
| DOGRULANAMADI | 3/47 |

---

## K1 — Dosya Bazli Katmanli Bolme

| ID | Aciklama | Durum | Detay |
|-----|---------|-------|-------|
| K1.1 | Her dosya uzerinde bagimsiz bolme | ✅ UYUMLU | process_single_file() fonksiyonu mevcut |
| K1.2 | Stratified split (stratify=y) | ✅ UYUMLU | stratify parametresi kullaniliyor |
| K1.3 | 80/10/10 bolme orani | ✅ UYUMLU | test_size=0.2 → %80/%20, sonra 0.5 → %10/%10 |
| K1.4 | Dosyalar birlesmeden ONCE boluyor | ✅ UYUMLU | process_single_file() per-file split, sonra concat |
| K1.5 | RANDOM_STATE = 42 | ✅ UYUMLU | Sabit seed |

---

## K2 — Uc Sinifli Taksonomi

| ID | Aciklama | Durum | Detay |
|-----|---------|-------|-------|
| K2.1-Benign | Tablo I Benign sayisi: paper 2,265,000 vs repo 2,273,097 | ✅ UYUMLU | Fark: 8,097 (<%2 tolerans) |
| K2.1-Volumetric | Tablo I Volumetric sayisi: paper 396,000 vs repo 380,699 | ⚠️ SAPMA | Fark: 15,301 |
| K2.1-Semantic | Tablo I Semantic sayisi: paper 170,000 vs repo 176,947 | ⚠️ SAPMA | Fark: 6,947 |
| K2.2-Benign | Oran Benign: paper ~%80 vs repo %80.3 | ✅ UYUMLU |  |
| K2.2-Volumetric | Oran Volumetric: paper ~%14 vs repo %13.4 | ✅ UYUMLU |  |
| K2.2-Semantic | Oran Semantic: paper ~%6 vs repo %6.3 | ✅ UYUMLU |  |
| K2.3 | Bot sinifi esleme: paper Botnet→Hacimsel, repo Bot→Semantic | ⚠️ SAPMA | classes_map Bot→2 (Semantic). Paper Tablo I Botnet'i Hacimsel'e koyar. JNCA'da aciklanmali. |
| K2.4 | Heartbleed esleme | ✅ UYUMLU | Heartbleed → Volumetric (paper'da Hacimsel altinda listelenmemis ama DoS ailesi) |

---

## K3 — 5 Model Karsilastirmasi

| ID | Aciklama | Durum | Detay |
|-----|---------|-------|-------|
| K3-RF-F1 | RF Macro-F1: paper 95.91% vs repo 95.91% | ✅ UYUMLU |  |
| K3-RF-Benign | Tablo III RF Benign F1: paper 98.32% vs repo 98.32% | ✅ UYUMLU |  |
| K3-RF-Volumetric | Tablo III RF Volumetric F1: paper 91.94% vs repo 91.94% | ✅ UYUMLU |  |
| K3-RF-Semantic | Tablo III RF Semantic F1: paper 97.36% vs repo 97.46% | ✅ UYUMLU |  |
| K3-XGBoost-F1 | XGBoost Macro-F1: paper 95.87% vs repo 95.87% | ✅ UYUMLU |  |
| K3-XGBoost-Benign | Tablo III XGBoost Benign F1: paper 98.56% vs repo 98.56% | ✅ UYUMLU |  |
| K3-XGBoost-Volumetric | Tablo III XGBoost Volumetric F1: paper 94.35% vs repo 94.35% | ✅ UYUMLU |  |
| K3-XGBoost-Semantic | Tablo III XGBoost Semantic F1: paper 94.71% vs repo 94.71% | ✅ UYUMLU |  |
| K3-LSTM-metrics | LSTM test metrikleri | ❓ DOGRULANAMADI | Config JSON'da test metrikleri yok; modeli yukleyip test setinde cikarim gerekli |
| K3-BiLSTM-metrics | BiLSTM test metrikleri | ❓ DOGRULANAMADI | Config JSON'da test metrikleri yok; modeli yukleyip test setinde cikarim gerekli |
| K3-DT-metrics | DT test metrikleri | ❓ DOGRULANAMADI | Config JSON'da test metrikleri yok; modeli yukleyip test setinde cikarim gerekli |
| K3-XGBoost-latency | XGBoost latency/throughput: paper 0.007ms/142,428 vs repo 0.0070ms/142,428 | ✅ UYUMLU |  |
| K3-RF-latency | RF latency/throughput: paper 0.0031ms/319,398 vs repo 0.0031ms/319,398 | ✅ UYUMLU |  |
| K3-DT-latency | DT latency/throughput: paper 0.0001ms/10,534,659 vs repo 0.0001ms/7,534,659 | ⚠️ SAPMA | Throughput farki: 3,000,000 |
| K3-LSTM-latency | LSTM latency/throughput: paper 0.0575ms/17,383 vs repo 0.0575ms/17,383 | ✅ UYUMLU |  |
| K3-BiLSTM-latency | BiLSTM latency/throughput: paper 0.1085ms/9,212 vs repo 0.1085ms/9,212 | ✅ UYUMLU |  |
| K3-protocol | 10,000 orneklemlik test protokolu | ✅ UYUMLU | benchmark_protocol = 10000_samples |
| K3-15.5x | XGBoost 15.5x daha hizli: 15.5x | ✅ UYUMLU |  |
| K3-1.88pp | Performans acigi: 1.88pp (paper 1.88pp) | ✅ UYUMLU |  |
| K3-XGB-HP | XGBoost hiperparametreleri | ✅ UYUMLU | max_depth=7, lr=0.05, sub=0.8, col=0.8 |
| K3-LSTM-arch | LSTM mimari: 128/64 units, 0.3 dropout, shape=[10, 20] | ✅ UYUMLU |  |
| K3-BiLSTM-arch | BiLSTM mimari: 128/64 units, 0.3 dropout, shape=[10, 20] | ✅ UYUMLU |  |
| K3-features | 20 oznitelik (TOP_FEATURES): 20 eleman | ✅ UYUMLU |  |
| K3-RF-grid | RF 36 kombinasyonluk grid search | ✅ UYUMLU | train_randomforest.py'da 36-combo grid search referansi mevcut |

---

## K4 — Canli Kopru Mimarisi

| ID | Aciklama | Durum | Detay |
|-----|---------|-------|-------|
| K4.1 | Scapy ile canli paket yakalama | ✅ UYUMLU |  |
| K4.2 | CICFlowMeter ile oznitelik cikarimi | ✅ UYUMLU |  |
| K4.4 | 4 saniyelik yakalama penceresi | ✅ UYUMLU | LIVE_CAPTURE_TIMEOUT_SECONDS default=4 |
| K4.3 | Apache Kafka mesaj hatti | ✅ UYUMLU |  |
| K4.5 | MinMaxScaler train-only fit | ✅ UYUMLU |  |
| K4.6 | Sifir kesintili model degisimi (ML ↔ DL) | ⚠️ SAPMA | Paper 5 model hot-swap iddia eder. Repo: LSTM/BiLSTM live_supported=False (E2-S01 karari). Hot-swap yalnizca 3 tabular model arasinda calisiyor. JNCA'da tabular-only siniri belirtilmeli. |
| K4.7 | 3 asamali kademeli yanit (ALERT → SUSPICIOUS → BLOCKED) | ✅ UYUMLU |  |
| K4.8 | Analist onayi: consumer otomatik engellemez | ✅ UYUMLU | Consumer block_ip import eder ama process_message/main'de cagirmaz. Engelleme yalnizca dashboard uzerinden analist tarafindan yapilir. |
| K4.9 | Platform bagimsiz firewall (iptables + Windows) | ✅ UYUMLU |  |
| K4.10 | 5 model canli hatta kullanilabilir | ⚠️ SAPMA | Paper 5 model canli iddia eder. Repo: yalnizca 3 tabular (RF/DT/XGB). LSTM/BiLSTM live_supported=False. JNCA'da sinir olarak belirtilmeli. |

---

## JNCA Icin Aksiyon Gereken Maddeler

### Sapmalar

1. **K2.1-Volumetric** — Tablo I Volumetric sayisi: paper 396,000 vs repo 380,699
   - Fark: 15,301
2. **K2.1-Semantic** — Tablo I Semantic sayisi: paper 170,000 vs repo 176,947
   - Fark: 6,947
3. **K2.3** — Bot sinifi esleme: paper Botnet→Hacimsel, repo Bot→Semantic
   - classes_map Bot→2 (Semantic). Paper Tablo I Botnet'i Hacimsel'e koyar. JNCA'da aciklanmali.
4. **K3-DT-latency** — DT latency/throughput: paper 0.0001ms/10,534,659 vs repo 0.0001ms/7,534,659
   - Throughput farki: 3,000,000
5. **K4.6** — Sifir kesintili model degisimi (ML ↔ DL)
   - Paper 5 model hot-swap iddia eder. Repo: LSTM/BiLSTM live_supported=False (E2-S01 karari). Hot-swap yalnizca 3 tabular model arasinda calisiyor. JNCA'da tabular-only siniri belirtilmeli.
6. **K4.10** — 5 model canli hatta kullanilabilir
   - Paper 5 model canli iddia eder. Repo: yalnizca 3 tabular (RF/DT/XGB). LSTM/BiLSTM live_supported=False. JNCA'da sinir olarak belirtilmeli.

### Dogrulanamayan Maddeler

1. **K3-LSTM-metrics** — LSTM test metrikleri
   - Config JSON'da test metrikleri yok; modeli yukleyip test setinde cikarim gerekli
2. **K3-BiLSTM-metrics** — BiLSTM test metrikleri
   - Config JSON'da test metrikleri yok; modeli yukleyip test setinde cikarim gerekli
3. **K3-DT-metrics** — DT test metrikleri
   - Config JSON'da test metrikleri yok; modeli yukleyip test setinde cikarim gerekli
