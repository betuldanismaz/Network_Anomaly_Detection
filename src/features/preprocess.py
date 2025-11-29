import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Main function to process the full data pipeline
def process_full_pipeline():

    # Set the base path for processed CSV files
    base_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed_csv")
    )
    
    # List of CSV files to process
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

    # Read and concatenate all CSV files
    dfs = []
    for f in file_list:
        path = os.path.join(base_path, f)
        if os.path.exists(path):
            print(f"   Reading: {f} ...")
            try:
                # Read CSV with latin1 encoding and strip column names
                df = pd.read_csv(path, encoding='latin1') 
                df.columns = df.columns.str.strip() 
                dfs.append(df)
            except Exception as e:
                print(f"   HATA: {f} okunamadı. Sebebi: {e}")
        else:
            print(f"   UYARI: {f} bulunamadı!")

    # If no dataframes were loaded, exit
    if not dfs:
        print("❌ Hiç veri yüklenemedi. İşlem iptal.")
        return

    # Concatenate all dataframes into one
    full_data = pd.concat(dfs, ignore_index=True)
    print(f"📊 BİRLEŞTİRİLMİŞ HAM VERİ: {full_data.shape} satır/sütun")

    # Data cleaning: replace inf with NaN and drop NaNs
    print("🧹 Temizlik yapılıyor (NaN ve Sonsuz değerler)...")
    full_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    full_data.dropna(inplace=True)
    print(f"   Temizlik sonrası: {full_data.shape}")

    # Process labels: BENIGN -> 0, others -> 1
    print("🏷️ Etiketler işleniyor...")
    y = full_data['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    
    # Drop label column from features
    X = full_data.drop(['Label'], axis=1)

    # Convert float64 columns to float32 for memory efficiency
    for col in X.columns:
        if X[col].dtype == 'float64':
            X[col] = X[col].astype('float32')

    # Split data into train, validation, and test sets (70/15/15)
    print("✂️ Veri setleri bölünüyor (%70 - %15 - %15)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"   ✅ Train Set: {X_train.shape}")
    print(f"   ✅ Val Set:   {X_val.shape}")
    print(f"   ✅ Test Set:  {X_test.shape}")

    # Directory to save the split datasets
    save_dir = "../../data/processed_csv/ready_splits/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("💾 Dosyalar diske yazılıyor...")

    # Save train set
    train_df = pd.concat([X_train, y_train], axis=1)
    train_df.to_csv(os.path.join(save_dir, "train.csv"), index=False)
    
    # Save validation set
    val_df = pd.concat([X_val, y_val], axis=1)
    val_df.to_csv(os.path.join(save_dir, "val.csv"), index=False)
    
    # Save test set
    test_df = pd.concat([X_test, y_test], axis=1)
    test_df.to_csv(os.path.join(save_dir, "test.csv"), index=False)

    print(f"🏁 İŞLEM TAMAM! Dosyalar şurada hazır: {save_dir}")

# Run the pipeline if this script is executed directly
if __name__ == "__main__":
    process_full_pipeline()