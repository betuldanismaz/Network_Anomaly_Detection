import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ANSI colors for terminal output
GREEN = '\033[92m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def train_model():
    print(f"\n{CYAN}🚀 STARTING MODEL TRAINING PIPELINE (Random Forest){RESET}")
    print("="*60)

    # 1. DYNAMIC PATHING
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    data_path = os.path.join(project_root, "data", "processed_csv", "ready_splits")
    models_dir = os.path.join(project_root, "models")
    reports_dir = os.path.join(project_root, "reports", "figures")
    
    # Create directories if they don't exist
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 2. LOAD DATA
    print(f"{YELLOW}📂 Loading Training and Validation Data...{RESET}")
    train_file = os.path.join(data_path, "train.csv")
    val_file = os.path.join(data_path, "val.csv")

    if not os.path.exists(train_file) or not os.path.exists(val_file):
        print(f"❌ Error: Data files not found in {data_path}")
        print("   Please run 'src/features/preprocess.py' first.")
        return

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    print(f"   ✅ Train Set: {train_df.shape}")
    print(f"   ✅ Val Set:   {val_df.shape}")

    # Separate Features and Target
    X_train = train_df.drop('Label', axis=1)
    y_train = train_df['Label']
    
    X_val = val_df.drop('Label', axis=1)
    y_val = val_df['Label']

    # 3. MODEL CONFIGURATION
    print(f"\n{YELLOW}⚙️  Configuring Random Forest Model...{RESET}")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    print("   - n_estimators: 100")
    print("   - n_jobs: -1 (All Cores)")
    print("   - random_state: 42")

    # 4. TRAINING
    print(f"\n{YELLOW}⏳ Training Model (This may take a while)...{RESET}")
    rf_model.fit(X_train, y_train)
    print(f"{GREEN}✅ Training Complete!{RESET}")

    # 5. EVALUATION
    print(f"\n{YELLOW}📊 Evaluating on Validation Set...{RESET}")
    y_pred = rf_model.predict(X_val)
    
    # Accuracy
    acc = accuracy_score(y_val, y_pred)
    print(f"   🏆 Accuracy Score: {GREEN}{acc:.4f} ({acc*100:.2f}%){RESET}")

    # Classification Report
    print("\n📝 Classification Report:")
    print(classification_report(y_val, y_pred, target_names=['BENIGN (0)', 'ATTACK (1)']))

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n{CYAN}🔍 Confusion Matrix Details:{RESET}")
    print(f"   - True Negatives (Normal correctly identified): {GREEN}{tn}{RESET}")
    print(f"   - False Positives (Normal flagged as Attack):   {YELLOW}{fp}{RESET}")
    print(f"   - False Negatives (Attack missed):              {YELLOW}{fn}{RESET} ⚠️ CRITICAL")
    print(f"   - True Positives (Attack correctly identified): {GREEN}{tp}{RESET}")

    # 6. VISUALIZATION
    print(f"\n{YELLOW}🎨 Generating Confusion Matrix Plot...{RESET}")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Benign', 'Predicted Attack'],
                yticklabels=['Actual Benign', 'Actual Attack'])
    plt.title('Confusion Matrix - Random Forest')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plot_path = os.path.join(reports_dir, "confusion_matrix_rf.png")
    plt.savefig(plot_path)
    print(f"   ✅ Plot saved to: {plot_path}")
    plt.close()

    # 7. SAVE MODEL
    print(f"\n{YELLOW}💾 Saving Model...{RESET}")
    model_path = os.path.join(models_dir, "rf_model_v1.pkl")
    joblib.dump(rf_model, model_path)
    print(f"   ✅ Model saved to: {model_path}")

    print(f"\n{GREEN}🏁 Script Finished Successfully!{RESET}")

if __name__ == "__main__":
    train_model()
