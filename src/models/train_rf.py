# src/models/train_rf.py
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def train_model():
    # 1. Set Data Paths (Dynamic Path)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    # Target the ready_splits folder
    data_path = os.path.join(project_root, "data", "processed_csv", "ready_splits")
    model_save_path = os.path.join(project_root, "models")
    
    # Create the folder if it does not exist
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    print("🚀 Model Training Starting (Random Forest)...")

    # 2. Load the Data
    print("📂 Loading data...")
    try:
        train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
        val_df = pd.read_csv(os.path.join(data_path, "val.csv"))
    except FileNotFoundError:
        print("❌ ERROR: train.csv or val.csv not found! Please run preprocess.py first.")
        return

    # Separate Features (X) and Label (y)
    # The last column is 'Label':
    X_train = train_df.drop('Label', axis=1)
    y_train = train_df['Label']
    
    X_val = val_df.drop('Label', axis=1)
    y_val = val_df['Label']

    print(f"   Training Set: {X_train.shape}")
    print(f"   Validation Set: {X_val.shape}")

    # 3. Define and Train the Model
    # n_estimators=100 -> Build 100 decision trees
    # n_jobs=-1 -> Use all CPU cores (Faster)
    rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    
    print("⏳ Training started (This may take time depending on data size)...")
    rf_model.fit(X_train, y_train)
    print("✅ Training completed!")

    # 4. Measure Performance (with Validation Set)
    print("📊 Testing on validation set...")
    y_pred = rf_model.predict(X_val)
    
    acc = accuracy_score(y_val, y_pred)
    print(f"\n🏆 Accuracy: %{acc * 100:.2f}")
    print("\nDetailed Report:")
    print(classification_report(y_val, y_pred, target_names=['Normal', 'Attack']))

    # 5. Save the Model
    save_file = os.path.join(model_save_path, "rf_model_v1.pkl")
    joblib.dump(rf_model, save_file)
    print(f"💾 Model saved: {save_file}")

if __name__ == "__main__":
    train_model()