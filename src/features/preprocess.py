import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def process_full_pipeline():
    print("\n🚀 STARTING DATA PREPROCESSING PIPELINE (CIC-IDS2017)")
    print("="*60)

    # 1. DYNAMIC PATHING
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

    # 2. LOAD AND CONCATENATE
    print(f"📂 Loading {len(file_list)} CSV files...")
    dfs = []
    for f in file_list:
        path = os.path.join(base_path, f)
        if os.path.exists(path):
            try:
                # Low memory=False to prevent mixed type warnings
                df = pd.read_csv(path, encoding='latin1', low_memory=False)
                df.columns = df.columns.str.strip() # Clean column names
                dfs.append(df)
                print(f"   ✅ Loaded: {f} ({df.shape})")
            except Exception as e:
                print(f"   ❌ Error loading {f}: {e}")
        else:
            print(f"   ⚠️ Warning: File not found: {f}")

    if not dfs:
        print("❌ No data loaded. Exiting.")
        return

    full_data = pd.concat(dfs, ignore_index=True)
    print(f"📊 Raw Data Shape: {full_data.shape}")

    # 3. DROP IDENTIFIERS (Step B - BEFORE removing duplicates)
    # We drop these first so that "behavioral duplicates" (same traffic pattern, different IP/Time) are caught.
    drop_cols = [
        'Flow ID', 
        'Source IP', 'Src IP', 
        'Source Port', 'Src Port', 
        'Destination IP', 'Dest IP', 
        'Destination Port', 'Dest Port', 
        'Timestamp', 'Date'
    ]
    existing_drop_cols = [c for c in drop_cols if c in full_data.columns]
    print(f"🗑️ Dropping {len(existing_drop_cols)} identifier columns to prevent overfitting...")
    full_data.drop(columns=existing_drop_cols, inplace=True)

    # 4. HANDLE MISSING/INFINITY (Step C)
    print("🧹 Cleaning NaN and Infinity values...")
    full_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    before_drop = full_data.shape[0]
    full_data.dropna(inplace=True)
    print(f"   Dropped {before_drop - full_data.shape[0]} rows containing NaN/Inf.")

    # 5. CONVERT TO FLOAT32 (Step D - Moved BEFORE Deduplication)
    # We convert to float32 BEFORE removing duplicates. 
    # This ensures that values that are distinct in float64 but identical in float32 
    # (due to precision loss) are treated as duplicates and removed.
    print("💾 Converting float64 to float32 to save memory and unify precision...")
    float_cols = full_data.select_dtypes(include=['float64']).columns
    full_data[float_cols] = full_data[float_cols].astype('float32')

    # 6. DROP DUPLICATES (Step E - CRITICAL FIX)
    print("🔄 Removing duplicates (Data Leakage Prevention)...")
    # We deduplicate based on FEATURE columns only. 
    # This removes:
    # 1. Exact duplicates (Same features, Same label)
    # 2. Conflicting duplicates (Same features, Different label) - keeping the first occurrence
    # This ensures ZERO overlap between Train and Test sets based on features.
    feature_cols = [c for c in full_data.columns if c != 'Label']
    before_dedup = full_data.shape[0]
    full_data.drop_duplicates(subset=feature_cols, keep='first', inplace=True)
    print(f"   Removed {before_dedup - full_data.shape[0]} duplicate rows.")
    print(f"   Cleaned Data Shape: {full_data.shape}")

    # 7. ENCODE LABELS (Step F)
    print("🏷️ Encoding Labels (0: BENIGN, 1: ATTACK)...")
    y = full_data['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    X = full_data.drop(['Label'], axis=1)

    # 8. STRATIFIED SPLIT
    print("✂️ Splitting Data (70% Train, 15% Val, 15% Test)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"   Train Shape: {X_train.shape}")
    print(f"   Val Shape:   {X_val.shape}")
    print(f"   Test Shape:  {X_test.shape}")

    # 9. SCALING (MinMax)
    print("⚖️ Scaling Features (MinMaxScaler)...")
    scaler = MinMaxScaler()
    
    # Fit ONLY on Train to prevent leakage
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Save Scaler
    scaler_path = os.path.join(project_root, "models", "scaler.pkl")
    if not os.path.exists(os.path.dirname(scaler_path)):
        os.makedirs(os.path.dirname(scaler_path))
    joblib.dump(scaler, scaler_path)
    print(f"   💾 Scaler saved to: {scaler_path}")

    # Reconstruct DataFrames
    columns = X.columns
    X_train = pd.DataFrame(X_train_scaled, columns=columns)
    X_val = pd.DataFrame(X_val_scaled, columns=columns)
    X_test = pd.DataFrame(X_test_scaled, columns=columns)

    # 10. SAVE TO DISK
    save_dir = os.path.join(base_path, "ready_splits")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("💾 Saving final CSV files...")
    
    # Reset indices to ensure alignment
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    pd.concat([X_train, y_train], axis=1).to_csv(os.path.join(save_dir, "train.csv"), index=False)
    pd.concat([X_val, y_val], axis=1).to_csv(os.path.join(save_dir, "val.csv"), index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(os.path.join(save_dir, "test.csv"), index=False)

    print(f"🏁 PIPELINE COMPLETE! Files saved in: {save_dir}")

if __name__ == "__main__":
    process_full_pipeline()
