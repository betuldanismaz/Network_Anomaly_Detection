import pandas as pd
import json
import os
import sys

# CHECKING IF ALL LABELS IN TRAINING DATA ARE PRESENT IN classes_map.json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed_csv")
JSON_PATH = os.path.join(BASE_DIR, "src", "utils", "classes_map.json")


def check_mappings():
    try:
        with open(JSON_PATH, "r") as f:
            classes_map = json.load(f)
        print("Classes mapping loaded successfully.")
    except FileNotFoundError:
        print(f"Error: {JSON_PATH} not found.")
        return


    train_file = os.path.join(DATA_PATH, "ready_splits", "train.csv")
    
    if not os.path.exists(train_file):
        print(f"Error: {train_file} not found.")
        return
    
    print (f"Loading training data from {train_file}...")

    df = pd.read_csv(train_file, usecols=["Label"])

    unique_labels = df["Label"].unique()
    print(f"Unique labels in training data: {unique_labels}")


    missing_keys = []
    for label in unique_labels:

        if str(label).isdigit():
            continue

        if label not in classes_map:
            missing_keys.append(label)

    if missing_keys:
        print("The following labels are missing in classes_map.json: {missing_keys}")
    else:
        print("All labels in training data are present in classes_map.json.")
    
if __name__ == "__main__":
    check_mappings()