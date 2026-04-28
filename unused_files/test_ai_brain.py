import joblib
import pandas as pd
import numpy as np
import os

# Yollar
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "binarymodels", "rf_model_v1.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "..", "models", "scaler.pkl")

# 1. Modeli ve Scaler'ı Yükle
print("🧠 Yapay Zeka Beyni Yükleniyor...")
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model Yüklendi.")
except:
    print("❌ Model dosyaları bulunamadı!")
    exit()

# 2. İki Tane Senaryo Oluştur (Manuel Veri)
# Bu sütunlar modelin beklediği 78 özellik (sırası önemli değil, dataframe halledecek)
# Tüm değerleri 0 yaparsak -> Normal Trafik gibi görünmeli
# Tüm değerleri aşırı yüksek yaparsak -> DDoS Saldırısı gibi görünmeli

# Modelin beklediği özellik listesi (Eğitimdeki 78 sütun - scaler ile tam uyumlu)
expected_cols = [
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

# Boş bir DataFrame oluştur
df_test = pd.DataFrame(columns=expected_cols)

# --- SENARYO 1: NORMAL TRAFİK ---
# Her şey düşük, sakin trafik
row_normal = {col: 0 for col in expected_cols}
row_normal["Flow Duration"] = 50000  # Normal süre
row_normal["Total Fwd Packets"] = 5       # Az paket
row_normal["Packet Length Mean"] = 60      # Küçük paketler
df_test = pd.concat([df_test, pd.DataFrame([row_normal])], ignore_index=True)

# --- SENARYO 2: AGRESİF SALDIRI (DDoS) ---
# Her şey tavan yapmış, çılgın trafik
row_attack = {col: 999999 for col in expected_cols} # Anormal yüksek değerler
row_attack["Flow Duration"] = 100    # Çok kısa sürede (Saniyede binlerce istek)
row_attack["Flow Packets/s"] = 50000    # Saniyede 50.000 paket!
row_attack["FIN Flag Count"] = 1       # Bayraklar karışık
row_attack["SYN Flag Count"] = 1
df_test = pd.concat([df_test, pd.DataFrame([row_attack])], ignore_index=True)

# 3. Gereksiz Kolonları At (Tıpkı live_bridge'deki gibi)
# Hazır değerler hedef özellik setiyle eşleştiği için ayrıca kolon düşmeye gerek yok
df_final = df_test.copy()

# 4. Tahmin Et
print("\n🔍 Tahmin Yapılıyor...")
X_scaled = scaler.transform(df_final)
predictions = model.predict(X_scaled)

# 5. Sonuçları Göster
print("-" * 30)
print(f"Senaryo 1 (Sakin Trafik) Tahmini: {'🔴 SALDIRI' if predictions[0]==1 else '🟢 TEMİZ'}")
print(f"Senaryo 2 (Agresif Veri) Tahmini: {'🔴 SALDIRI' if predictions[1]==1 else '🟢 TEMİZ'}")
print("-" * 30)

if predictions[1] == 1:
    print("✅ SONUÇ: Yapay Zeka beyni sağlam! Doğru veriyi görünce tanıyor.")
    print("👉 Sorun modelde değil, bizim ürettiğimiz saldırı paketlerinin 'yetersizliğinde'.")
else:
    print("❌ SONUÇ: Model kör. Aşırı yüksek değerlere bile 'Temiz' diyor.")
    print("👉 Modelin eğitimi veya scaler hatalı olabilir.")