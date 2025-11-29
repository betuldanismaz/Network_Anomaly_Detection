import pandas as pd
import numpy as np
import os

def check_data_quality():
    print("🔍 Veri Kalitesi ve Sızıntı Kontrolü Başlıyor...")
    
    # Dosya yollarını belirle
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed_csv", "ready_splits"))
    train_path = os.path.join(base_path, "train.csv")
    val_path = os.path.join(base_path, "val.csv")
    test_path = os.path.join(base_path, "test.csv")
    
    # Dosyaların varlığını kontrol et
    if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        print("❌ HATA: Veri dosyaları bulunamadı! Lütfen önce preprocess.py çalıştırın.")
        return

    # Verileri yükle (bellek tasarrufu için sadece gerekli sütunları veya örneklem alabiliriz ama burada tam kontrol yapalım)
    print("📂 Veriler yükleniyor...")
    try:
        # Büyük dosyalar için chunksize veya dtype optimizasyonu yapılabilir ama şimdilik direkt okuyoruz
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
    except Exception as e:
        print(f"❌ Veri okuma hatası: {e}")
        return

    print(f"   Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

    # 1. Veri Sızıntısı Kontrolü (Data Leakage)
    # Train ve Val/Test setleri arasında ortak satır var mı?
    print("\n🕵️‍♂️ 1. Veri Sızıntısı (Data Leakage) Kontrolü:")
    
    # Label hariç sütunları alarak karşılaştırma yapalım
    cols_to_check = [c for c in train_df.columns if c != 'Label']
    
    # Train vs Val
    common_train_val = pd.merge(train_df[cols_to_check], val_df[cols_to_check], how='inner')
    if len(common_train_val) > 0:
        print(f"   ⚠️ UYARI: Train ve Validation setleri arasında {len(common_train_val)} adet ortak satır bulundu!")
    else:
        print("   ✅ Train ve Validation setleri tamamen ayrık.")

    # Train vs Test
    common_train_test = pd.merge(train_df[cols_to_check], test_df[cols_to_check], how='inner')
    if len(common_train_test) > 0:
        print(f"   ⚠️ UYARI: Train ve Test setleri arasında {len(common_train_test)} adet ortak satır bulundu!")
    else:
        print("   ✅ Train ve Test setleri tamamen ayrık.")

    # 2. Sınıf Dağılımı Kontrolü
    print("\n📊 2. Sınıf Dağılımı Kontrolü:")
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        dist = df['Label'].value_counts(normalize=True)
        print(f"   {name} Seti Dağılımı:\n{dist.to_string()}\n")

    # 3. Tek Değerli Sütun Kontrolü (Gereksiz Özellikler)
    print("🗑️ 3. Tek Değerli (Sabit) Sütun Kontrolü:")
    single_val_cols = [col for col in train_df.columns if train_df[col].nunique() <= 1]
    if single_val_cols:
        print(f"   ⚠️ Şu sütunlar sadece tek bir değer içeriyor (model için gereksiz olabilir): {single_val_cols}")
    else:
        print("   ✅ Tüm sütunlar birden fazla değer içeriyor.")

    # 4. Mükemmel Ayrıştırıcı Kontrolü (Suspiciously High Performance)
    # Eğer bir özellik tek başına hedefi %100 tahmin ediyorsa şüphelidir.
    print("\n🎯 4. Şüpheli 'Mükemmel' Özellik Kontrolü:")
    suspicious_features = []
    for col in cols_to_check:
        # Basit bir kontrol: Her özellik değeri sadece tek bir sınıfa mı ait?
        # Bu kontrol sayısal verilerde (float) çok anlamlı olmayabilir ama kategorik veya düşük kardinaliteli sayısal verilerde işe yarar.
        # Daha gelişmişi için korelasyon veya bilgi kazancı (information gain) bakılabilir.
        
        # Hızlı kontrol: Özellik ile Label arasındaki korelasyon çok yüksek mi?
        if pd.api.types.is_numeric_dtype(train_df[col]):
            corr = train_df[col].corr(train_df['Label'])
            if abs(corr) > 0.95:
                suspicious_features.append((col, corr))
    
    if suspicious_features:
        print(f"   ⚠️ Şu özellikler Label ile çok yüksek korelasyona sahip (>0.95):")
        for feat, corr in suspicious_features:
            print(f"      - {feat}: {corr:.4f}")
        print("      (Bu özellikler sızıntı veya aşırı basit bir örüntü olabilir.)")
    else:
        print("   ✅ Aşırı yüksek korelasyonlu tekil özellik bulunamadı.")

    print("\n🏁 Kontrol Tamamlandı.")

if __name__ == "__main__":
    check_data_quality()
