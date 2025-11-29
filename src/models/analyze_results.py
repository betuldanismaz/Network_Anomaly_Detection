# src/models/analyze_results.py
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def analyze_model():
    # 1. Yolları Ayarla
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(base_dir))
    model_path = os.path.join(root, "models", "rf_model_v1.pkl")
    data_path = os.path.join(root, "data", "processed_csv", "ready_splits")

    print("🕵️ ADLİ TIP ANALİZİ BAŞLIYOR...")

    # 2. Modeli ve Veriyi Yükle
    print("📂 Model ve Test verisi yükleniyor...")
    model = joblib.load(model_path)
    
    # Sadece validation setini yükleyelim (Hızlı olsun)
    # Veri setini yüklerken sütun isimlerini almak önemli
    val_df = pd.read_csv(os.path.join(data_path, "val.csv"))
    
    # X ve y ayır
    y_val = val_df['Label']
    X_val = val_df.drop('Label', axis=1)
    feature_names = X_val.columns

    # -------------------------------------------------------
    # ANALİZ 1: FEATURE IMPORTANCE (Model Neye Bakıyor?)
    # -------------------------------------------------------
    print("\n🔍 ANALİZ 1: Özellik Önem Düzeyleri (Feature Importance)")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # İlk 15 özelliği yazdır
    print(f"{'Sıra':<5} {'Özellik Adı':<40} {'Önem Skoru'}")
    print("-" * 60)
    top_features = []
    for f in range(15):
        idx = indices[f]
        fname = feature_names[idx]
        score = importances[idx]
        top_features.append((fname, score))
        print(f"{f+1:<5} {fname:<40} {score:.4f}")

    # Görselleştir ve Kaydet
    plt.figure(figsize=(12, 8))
    sns.barplot(x=[x[1] for x in top_features], y=[x[0] for x in top_features], palette="viridis")
    plt.title("Modelin Karar Verirken Baktığı En Önemli 15 Özellik")
    plt.xlabel("Önem Skoru")
    plt.tight_layout()
    plt.savefig(os.path.join(root, "reports", "figures", "feature_importance.png"))
    print("✅ Grafik kaydedildi: reports/figures/feature_importance.png")

    # YORUM
    top_1 = top_features[0][0]
    suspicious_keywords = ['ID', 'Id', 'id', 'Index']
    if any(s in top_1 for s in suspicious_keywords):
        print("⚠️ UYARI: En önemli özellik şüpheli görünüyor! Lütfen kontrol edin.")
    else:
        print("✅ ONAY: Model davranışsal özelliklere odaklanıyor gibi görünüyor.")

    # -------------------------------------------------------
    # ANALİZ 2: KAÇIRILAN SALDIRILAR (Error Analysis)
    # -------------------------------------------------------
    print("\n🔍 ANALİZ 2: Kaçırılan 229 Saldırının Detayı")
    
    # Tahmin yap
    y_pred = model.predict(X_val)
    
    # Hataları bul (Gerçekte Saldırı (1) ama Model Normal (0) demiş -> False Negative)
    # Pandas ile filtreleme
    mask_missed = (y_val == 1) & (y_pred == 0)
    missed_attacks = X_val[mask_missed]
    
    print(f"Toplam Kaçırılan Saldırı Sayısı: {len(missed_attacks)}")
    
    # NOT: Elimizdeki val.csv'de orijinal saldırı isimleri (DoS, Web Attack vb.) yok, sadece 0 ve 1 var.
    # Bu yüzden sadece kaçırılan paketlerin özelliklerine bakabiliriz.
    
    if len(missed_attacks) > 0:
        print("\nKaçırılan Saldırılardan Örnek Veriler (Ortalama Değerler):")
        print(missed_attacks.mean().sort_values(ascending=False).head(5))
        print("\n-> Bu saldırıların ortak özelliği ne olabilir? (Düşük paket boyutu mu? Düşük süre mi?)")

if __name__ == "__main__":
    analyze_model()