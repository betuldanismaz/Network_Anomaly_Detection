# src/features/preprocess.py
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def process_full_pipeline():
    # 1. AYARLAR
    base_path = "../../data/processed_csv/"
    
    # Görseldeki dosya isimlerinin tam listesi
    file_list = [
        "Monday-WorkingHours.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv",
        "Wednesday-workingHours.pcap_ISCX.csv",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv"
    ]

    print(f"🚀 DERİN ALTYAPI MODU: Toplam {len(file_list)} adet dosya işlenecek...")

    # 2. VERİ YÜKLEME VE BİRLEŞTİRME
    dfs = []
    for f in file_list:
        path = os.path.join(base_path, f)
        if os.path.exists(path):
            print(f"   Reading: {f} ...")
            try:
                # Bazı dosyalarda encoding sorunu olabilir, 'latin1' güvenlidir
                df = pd.read_csv(path, encoding='latin1') 
                df.columns = df.columns.str.strip() # Sütun isimlerindeki boşlukları temizle
                dfs.append(df)
            except Exception as e:
                print(f"   HATA: {f} okunamadı. Sebebi: {e}")
        else:
            print(f"   UYARI: {f} bulunamadı!")

    if not dfs:
        print("❌ Hiç veri yüklenemedi. İşlem iptal.")
        return

    full_data = pd.concat(dfs, ignore_index=True)
    print(f"📊 BİRLEŞTİRİLMİŞ HAM VERİ: {full_data.shape} satır/sütun")

    # 3. TEMİZLİK
    print("🧹 Temizlik yapılıyor (NaN ve Sonsuz değerler)...")
    full_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    full_data.dropna(inplace=True)
    print(f"   Temizlik sonrası: {full_data.shape}")

    # 4. ETİKET DÜZENLEME (Label Encoding)
    # Binary Classification (0: Normal, 1: Attack) yapacağız ama
    # Orijinal saldırı isimlerini kaybetmeyelim, belki ilerde lazım olur.
    
    print("🏷️ Etiketler işleniyor...")
    # 'Label' sütunundaki 'BENIGN' harici her şeye 1 (Saldırı) diyelim
    y = full_data['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    
    # X (Özellikler) -> Label sütununu çıkarıyoruz
    X = full_data.drop(['Label'], axis=1)

    # Bellek tasarrufu: Tipleri küçültelim (float64 -> float32)
    # Senin bilgisayar güçlü ama GPU eğitiminde float32 standarttır.
    for col in X.columns:
        if X[col].dtype == 'float64':
            X[col] = X[col].astype('float32')

    # 5. STRATIFIED SPLIT (%70 Train, %15 Val, %15 Test)
    print("✂️ Veri setleri bölünüyor (%70 - %15 - %15)...")
    
    # Önce Train (%70) ve Temp (%30) olarak ayır
    # stratify=y -> Saldırı oranlarını her parçada korur!
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Sonra Temp'i ikiye böl: Val (%15) ve Test (%15)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"   ✅ Train Set: {X_train.shape}")
    print(f"   ✅ Val Set:   {X_val.shape}")
    print(f"   ✅ Test Set:  {X_test.shape}")

    # 6. KAYDETME (Parçalı Kayıt)
    # Büyük veriyi tek parça kaydetmek yerine split edilmiş halde kaydedelim
    # Böylece eğitim sırasında tekrar tekrar split yapmak zorunda kalmayız.
    save_dir = "../../data/processed_csv/ready_splits/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("💾 Dosyalar diske yazılıyor...")
    # Sadece eğitim verisini kaydetsek yeterli, diğerlerini eğitim sırasında kullanacağız ama
    # "Derin Altyapı" dediğin için her şeyi fiziksel olarak ayıralım.
    
    # Train
    train_df = pd.concat([X_train, y_train], axis=1)
    train_df.to_csv(os.path.join(save_dir, "train.csv"), index=False)
    
    # Val
    val_df = pd.concat([X_val, y_val], axis=1)
    val_df.to_csv(os.path.join(save_dir, "val.csv"), index=False)
    
    # Test
    test_df = pd.concat([X_test, y_test], axis=1)
    test_df.to_csv(os.path.join(save_dir, "test.csv"), index=False)

    print(f"🏁 İŞLEM TAMAM! Dosyalar şurada hazır: {save_dir}")

if __name__ == "__main__":
    process_full_pipeline()