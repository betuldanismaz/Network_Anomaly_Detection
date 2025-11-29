# src/features/preprocess.py
import pandas as pd
import numpy as np
import os
import joblib  # Scaler'ı kaydetmek için gerekli
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def process_full_pipeline():
    # 1. DOSYA YOLLARI
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    base_path = os.path.join(project_root, "data", "processed_csv")
    
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

    # 2. YÜKLEME VE BİRLEŞTİRME
    dfs = []
    for f in file_list:
        path = os.path.join(base_path, f)
        if os.path.exists(path):
            print(f"   Reading: {f} ...")
            try:
                df = pd.read_csv(path, encoding='latin1', low_memory=False)
                df.columns = df.columns.str.strip() # Boşlukları temizle
                dfs.append(df)
            except Exception as e:
                print(f"   HATA: {f} okunamadı. Sebebi: {e}")
        else:
            print(f"   UYARI: {path} bulunamadı!")

    if not dfs:
        print("❌ Hiç veri yüklenemedi. İşlem iptal.")
        return

    full_data = pd.concat(dfs, ignore_index=True)
    print(f"📊 BİRLEŞTİRİLMİŞ HAM VERİ: {full_data.shape}")

    # 3. KİMLİK SÜTUNLARINI ATMA (Overfitting Önlemi)
    # Modelin 'Davranışı' öğrenmesi için 'Kimlikleri' siliyoruz.
    drop_cols = [
        'Flow ID', 
        'Source IP', 'Src IP', 
        'Source Port', 'Src Port', 
        'Destination IP', 'Dest IP', 
        'Destination Port', 'Dest Port', 
        'Timestamp', 'Date'
    ]
    
    # Sadece veride mevcut olan sütunları sil
    existing_drop_cols = [c for c in drop_cols if c in full_data.columns]
    print(f"🗑️ Gereksiz sütunlar siliniyor: {len(existing_drop_cols)} adet")
    full_data.drop(columns=existing_drop_cols, inplace=True)

    # 4. TEMİZLİK
    print("🧹 Temizlik yapılıyor (NaN ve Sonsuz değerler)...")
    full_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    full_data.dropna(inplace=True)

    print("🔄 Tekrarlayan veriler temizleniyor...")
    full_data.drop_duplicates(inplace=True)
    print(f"   Temizlik sonrası: {full_data.shape}")

    # 5. ETİKETLEME
    print("🏷️ Etiketler işleniyor...")
    y = full_data['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    X = full_data.drop(['Label'], axis=1)

    # 6. BÖLME (Splitting) - ÖNCE BÖL, SONRA SCALE ET!
    print("✂️ Veri setleri bölünüyor (%70 - %15 - %15)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # 7. ÖLÇEKLEME (Scaling) - KRİTİK ADIM
    # MinMaxScaler verileri 0-1 arasına sıkıştırır. Deep Learning için en iyisidir.
    print("⚖️ Veriler ölçekleniyor (MinMax Scaling)...")
    
    scaler = MinMaxScaler()
    
    # Scaler SADECE eğitim verisini görmeli (Fit)
    # Sonra diğerlerini dönüştürmeli (Transform)
    # Bunu yapmazsak 'Data Leakage' olur.
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Scaler'ı kaydet (Canlı sistemde kullanmak için şart!)
    scaler_path = os.path.join(project_root, "models", "scaler.pkl")
    if not os.path.exists(os.path.dirname(scaler_path)):
        os.makedirs(os.path.dirname(scaler_path))
    joblib.dump(scaler, scaler_path)
    print(f"💾 Scaler kaydedildi: {scaler_path}")

    # DataFrame'e geri çevir (Sütun isimlerini korumak için)
    columns = X.columns
    X_train = pd.DataFrame(X_train_scaled, columns=columns)
    X_val = pd.DataFrame(X_val_scaled, columns=columns)
    X_test = pd.DataFrame(X_test_scaled, columns=columns)

    # 8. KAYDETME
    save_dir = os.path.join(base_path, "ready_splits")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("💾 İşlenmiş veriler diske yazılıyor...")
    
    # Index resetlemek önemli, yoksa concat hata verir
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    pd.concat([X_train, y_train], axis=1).to_csv(os.path.join(save_dir, "train.csv"), index=False)
    pd.concat([X_val, y_val], axis=1).to_csv(os.path.join(save_dir, "val.csv"), index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(os.path.join(save_dir, "test.csv"), index=False)

    print(f"🏁 İŞLEM TAMAM! Dosyalar şurada hazır: {save_dir}")

if __name__ == "__main__":
    process_full_pipeline()
