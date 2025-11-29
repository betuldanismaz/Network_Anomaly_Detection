import pandas as pd
import numpy as np
import os
import sys
#DATA INTEGRITY AUDIT SCRIPT
# ANSI colors for terminal output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def check_data_health():
    # 1. Setup Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_dir = os.path.join(project_root, "data", "processed_csv", "ready_splits")
    
    files = {
        "Train": os.path.join(data_dir, "train.csv"),
        "Val": os.path.join(data_dir, "val.csv"),
        "Test": os.path.join(data_dir, "test.csv")
    }

    # Check existence
    for name, path in files.items():
        if not os.path.exists(path):
            print(f"{RED}❌ Error: {name} file not found at {path}{RESET}")
            return

    print_header("🚀 STARTING DATA HEALTH AUDIT")
    
    dfs = {}
    for name, path in files.items():
        print(f"📂 Loading {name} set from {os.path.basename(path)}...")
        try:
            # Load full data for accurate leakage check
            dfs[name] = pd.read_csv(path)
            print(f"   ✅ Loaded {dfs[name].shape[0]} rows, {dfs[name].shape[1]} columns")
        except Exception as e:
            print(f"{RED}❌ Failed to load {name}: {e}{RESET}")
            return

    train_df = dfs["Train"]
    val_df = dfs["Val"]
    test_df = dfs["Test"]

    # 2. Data Leakage Check
    print_header("🕵️‍♂️ 1. DATA LEAKAGE CHECK (Overlap)")
    
    # Check Train vs Test
    # Exclude Label from leakage check to focus on features
    feature_cols = [c for c in train_df.columns if c != 'Label']
    
    # Using merge to find exact duplicates
    overlap_train_test = pd.merge(train_df[feature_cols], test_df[feature_cols], how='inner')
    num_overlap = len(overlap_train_test)
    
    if num_overlap > 0:
        print(f"{RED}⚠️ CRITICAL WARNING: Found {num_overlap} duplicate rows between Train and Test sets!{RESET}")
        print(f"{RED}   This indicates DATA LEAKAGE. The model will memorize these examples.{RESET}")
    else:
        print(f"{GREEN}✅ No overlap found between Train and Test sets.{RESET}")

    # 3. Identifier Check
    print_header("🚫 2. FORBIDDEN IDENTIFIER CHECK")
    forbidden_cols = ['Flow ID', 'Source IP', 'Src IP', 'Source Port', 'Src Port', 
                      'Destination IP', 'Dest IP', 'Destination Port', 'Dest Port', 
                      'Timestamp', 'Date']
    
    found_forbidden = []
    for col in train_df.columns:
        if col in forbidden_cols:
            found_forbidden.append(col)
            
    if found_forbidden:
        print(f"{RED}⚠️ WARNING: Found forbidden columns that may cause overfitting:{RESET}")
        print(f"   {found_forbidden}")
    else:
        print(f"{GREEN}✅ No forbidden identifier columns found.{RESET}")

    # 4. Sanity Check (NaN/Inf)
    print_header("🧠 3. SANITY CHECK (NaNs & Infinity)")
    
    for name, df in dfs.items():
        nans = df.isna().sum().sum()
        infs = np.isinf(df.select_dtypes(include=np.number)).sum().sum()
        
        if nans > 0 or infs > 0:
            print(f"{RED}⚠️ {name} Set: Found {nans} NaNs and {infs} Infinite values.{RESET}")
        else:
            print(f"{GREEN}✅ {name} Set: Clean (No NaNs or Inf).{RESET}")

    # 5. Class Distribution
    print_header("📊 4. CLASS DISTRIBUTION (Stratification Check)")
    
    for name, df in dfs.items():
        if 'Label' not in df.columns:
            print(f"{RED}❌ 'Label' column missing in {name} set.{RESET}")
            continue
            
        counts = df['Label'].value_counts(normalize=True)
        benign_pct = counts.get(0, 0) * 100
        attack_pct = counts.get(1, 0) * 100
        
        print(f"   {name}: Normal (0): {benign_pct:.2f}% | Attack (1): {attack_pct:.2f}%")

    # 6. Scaling Verification
    print_header("⚖️ 5. SCALING VERIFICATION (MinMax Check)")
    
    # We check Train set mainly
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    # Exclude Label
    numeric_cols = [c for c in numeric_cols if c != 'Label']
    
    min_val = train_df[numeric_cols].min().min()
    max_val = train_df[numeric_cols].max().max()
    
    print(f"   Global Min Value in Train: {min_val}")
    print(f"   Global Max Value in Train: {max_val}")
    
    if min_val < -0.01 or max_val > 1.01:
        print(f"{YELLOW}⚠️ WARNING: Values are outside the expected 0-1 range for MinMaxScaler.{RESET}")
        print("   (This might be okay if you used StandardScaler, but check if you intended MinMaxScaler)")
    else:
        print(f"{GREEN}✅ Values appear to be scaled between 0 and 1.{RESET}")

    # 7. Data Type Check
    print_header("💾 6. DATA TYPE CHECK (Memory Efficiency)")
    
    dtype_counts = train_df.dtypes.value_counts()
    print(f"   Column Data Types:\n{dtype_counts}")
    
    float64_cols = train_df.select_dtypes(include=['float64']).columns
    if len(float64_cols) > 0:
        print(f"{YELLOW}⚠️ Notice: {len(float64_cols)} columns are float64. Consider using float32 to save memory.{RESET}")
    else:
        print(f"{GREEN}✅ No float64 columns found (Good for memory).{RESET}")

    print_header("🏁 AUDIT COMPLETE")

if __name__ == "__main__":
    check_data_health()
