# src/models/stress_test.py
import pandas as pd
import joblib
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def stress_test_model():
    # 1. Yolları Ayarla
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(base_dir))
    # Allow overrides via env vars
    model_path = os.getenv('RF_MODEL_PATH') or os.path.join(root, "models", "rf_model_v1.pkl")
    data_path = os.getenv('PROCESSED_CSV_DIR') or os.path.join(root, "data", "processed_randomforest")

    print("🔥 STRES TESTİ BAŞLIYOR (Performans Analizi)...")

    # 2. Modeli ve Veriyi Yükle
    print("📂 Model ve Test verisi yükleniyor...")
    model = joblib.load(model_path)
    # Test setinden rastgele 10.000 örnek alalım
    test_df = pd.read_csv(os.path.join(data_path, "test.csv")).sample(n=10000, random_state=42)
    X_test = test_df.drop('Label', axis=1)

    print(f"⚡ {len(X_test)} adet paket üzerinde hız testi yapılıyor...")

    # 3. HIZ TESTİ (Latency Check)
    start_time = time.time()
    _ = model.predict(X_test)
    end_time = time.time()

    total_time = end_time - start_time
    pps = len(X_test) / total_time # Packet Per Second

    print("\n⏱️ PERFORMANS SONUÇLARI:")
    print("-" * 40)
    print(f"Toplam Süre:       {total_time:.4f} saniye")
    print(f"Paket Başına Süre: {total_time/len(X_test)*1000:.4f} ms")
    print(f"Saniyedeki İşlem:  {int(pps)} paket/saniye (PPS)")
    print("-" * 40)

    # DEĞERLENDİRME
    if pps > 10000:
        print("✅ HIZ DURUMU: MÜKEMMEL. Canlı akışı çok rahat kaldırır.")
    elif pps > 2000:
        print("✅ HIZ DURUMU: İYİ. Normal trafik için yeterli.")
    else:
        print("⚠️ HIZ DURUMU: KRİTİK YAVAŞLIK. Kod optimizasyonu gerekebilir.")

    # 4. GÜVEN ANALİZİ (Confidence Check)
    # Modelin ne kadar emin olduğunu görelim
    print("\n🧠 Güven Analizi (Probability Distribution)...")
    probs = model.predict_proba(X_test)
    
    # Saldırı ihtimali (Sınıf 1) olanların güven skorlarını al
    attack_probs = probs[:, 1]
    
    # Görselleştir
    plt.figure(figsize=(10, 6))
    sns.histplot(attack_probs, bins=50, kde=True, color='purple')
    plt.title("Modelin Karar Güven Dağılımı (0=Kesin Normal, 1=Kesin Saldırı)")
    plt.xlabel("Saldırı Olasılığı")
    plt.ylabel("Paket Sayısı")
    
    # Çizgiler ekle
    plt.axvline(x=0.5, color='red', linestyle='--', label='Karar Sınırı (0.5)')
    plt.legend()
    
    save_path = os.getenv('REPORTS_FIGURES_DIR') or os.path.join(root, "reports", "figures")
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, "confidence_dist.png")
    plt.savefig(save_path)
    print(f"📊 Güven grafiği kaydedildi: {save_path}")
    
    # Yorum
    uncertain_count = np.sum((attack_probs > 0.4) & (attack_probs < 0.6))
    print(f"\n🔍 Kararsız Bölge Analizi (0.4 - 0.6 arası):")
    print(f"Modelin kararsız kaldığı paket sayısı: {uncertain_count}")
    
    if uncertain_count > 100:
        print("⚠️ UYARI: Model bazı paketlerde kararsız kalıyor. Threshold ayarı gerekebilir.")
    else:
        print("✅ ONAY: Model kararlarında çok net (Ya 0 ya 1 diyor).")

if __name__ == "__main__":
    stress_test_model()