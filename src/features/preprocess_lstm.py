import os
import sys
import glob
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from collections import Counter


# --- CONFIG --
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Add project root to Python path to allow imports from src
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA_DIR = os.getenv('LSTM_DATA_DIR') or os.path.join(ROOT, 'data', 'original_csv')
# Set OUT_DIR from environment variable or default to 'data/processed_lstm' in project root
OUT_DIR = os.getenv('LSTM_OUT_DIR') or os.path.join(ROOT, 'data', 'processed_lstm')
# Set MODELS_DIR from environment variable or default to 'models' in project root
MODELS_DIR = os.getenv('LSTM_MODELS_DIR') or os.path.join(ROOT, 'models')
# Define the path where the scaler object will be saved
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler_lstm.pkl')
# Define the path where class weights will be saved as JSON
CLASS_WEIGHTS_PATH = os.path.join(MODELS_DIR, 'class_weights.json')

# Resolve classes_map.json from several likely locations
# Create a list of possible locations where the classes_map.json file might exist
possible_class_map_paths = [
    # First check if path is specified in environment variable
    os.getenv('CLASSES_MAP_PATH'),
    # Check in src/utils directory (listed twice, likely a duplicate)
    os.path.join(ROOT, 'src', 'utils', 'classes_map.json'),
    # Check in src/utils directory again (duplicate entry)
    os.path.join(ROOT, 'src', 'utils', 'classes_map.json'),
    # Check in data directory
    os.path.join(ROOT, 'data', 'classes_map.json'),
    # Check in project root directory
    os.path.join(ROOT, 'classes_map.json'),
]
# Initialize CLASS_MAP_PATH as None before searching
CLASS_MAP_PATH = None
# Iterate through each possible path
for p in possible_class_map_paths:
    # Skip if the path is None or empty
    if not p:
        # Continue to next iteration
        continue
    # Check if the file exists at this path
    if os.path.exists(p):
        # If found, set CLASS_MAP_PATH to this path
        CLASS_MAP_PATH = p
        # Exit the loop since we found the file
        break
# If no path was found in the list
if CLASS_MAP_PATH is None:
    # final fallback to package-relative utils
    # Set a fallback path relative to the current file's directory
    CLASS_MAP_PATH = os.path.join(os.path.dirname(__file__), '..', 'utils', 'classes_map.json')

# Create the output directory if it doesn't exist, don't raise error if it exists
os.makedirs(OUT_DIR, exist_ok=True)
# Create the models directory if it doesn't exist, don't raise error if it exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Import top features
# Import the TOP_FEATURES list from the config module
from src.config import TOP_FEATURES


# Parameters
# Define the window size for creating sequences (number of time steps)
WINDOW_SIZE = 10
# Define the stride for sliding window (1 = overlapping windows, 5 = less overlap)
STRIDE = 1  # change to 5 to reduce dataset size


# Define a function to normalize text by removing special characters and whitespace
def _normalize_text(x):
    # Check if input is None
    if x is None:
        # Return empty string for None values
        return ''
    # Convert input to string
    s = str(x)
    # Try to normalize unicode characters
    try:
        # Import unicodedata module for unicode normalization
        import unicodedata
        # Normalize unicode string to NFKC form (compatibility composition)
        s = unicodedata.normalize('NFKC', s)
    # Catch any exceptions during normalization
    except Exception:
        # If normalization fails, continue without it
        pass
    # Replace byte order mark, non-breaking space, and replacement character with regular space
    s = s.replace('\ufeff', '').replace('\u00A0', ' ').replace('\uFFFD', ' ')
    # Split string by whitespace and rejoin with single spaces (removes extra whitespace)
    s = ' '.join(s.split())
    # Remove leading and trailing whitespace and return
    return s.strip()


# Define a function to list all CSV files in a directory
def list_csv_files(data_dir):
    # Use glob to find all .csv files in the directory and return sorted list
    return sorted(glob.glob(os.path.join(data_dir, '*.csv')))


# Define a function to read only selected columns from a CSV file
def read_selected_columns(file_path, features):
    # read only needed columns to save memory
    # Read CSV header to get available columns, filter features to only those present
    usecols = [c for c in features if c in pd.read_csv(file_path, nrows=0).columns]
    # Check if 'Label' column exists in the CSV file
    if 'Label' in pd.read_csv(file_path, nrows=0).columns:
        # Add 'Label' to the list of columns to read
        usecols = usecols + ['Label']
    # If no columns to read, return None
    if not usecols:
        # Return None indicating no valid columns found
        return None
    # Read and return the CSV file with only the selected columns
    return pd.read_csv(file_path, usecols=usecols)


# Define a function to create sequences from array data using sliding window
def create_sequences(arr, labels, window_size=10, stride=1):
    # arr: numpy array (N, F), labels: 1D array-like (N,)
    # Initialize empty list to store feature sequences
    X_list = []
    # Initialize empty list to store corresponding labels
    y_list = []
    # Get the number of rows in the input array
    N = arr.shape[0]
    # Iterate through the array with sliding window using specified stride
    for start in range(0, N - window_size + 1, stride):
        # Calculate the end index of the current window
        end = start + window_size
        # Extract the window slice and add to X_list
        X_list.append(arr[start:end])
        # Take the label from the last time step in the window and add to y_list
        y_list.append(int(labels[end - 1]))
    # If no sequences were created (array too small)
    if not X_list:
        # Return empty arrays with correct shapes
        return np.zeros((0, window_size, arr.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    # Stack all sequences into a 3D numpy array and convert to float32
    X = np.stack(X_list).astype(np.float32)
    # Convert label list to numpy array with int32 type
    y = np.array(y_list, dtype=np.int32)
    # Return the sequences and labels
    return X, y


# Define a function to compute class weights for imbalanced datasets
def compute_class_weights(y):
    # y is 1D array of integer labels
    # Count occurrences of each class label
    counts = Counter(y.tolist())
    # Get sorted list of unique class labels
    classes = sorted(counts.keys())
    # Calculate total number of samples
    total = sum(counts.values())
    # Get the number of unique classes
    n_classes = len(classes)
    # Initialize empty dictionary to store weights
    weights = {}
    # Iterate through each class
    for c in classes:
        # Check if class has zero samples
        if counts[c] == 0:
            # Assign weight of 1.0 for classes with no samples
            weights[c] = 1.0
        # For classes with samples
        else:
            # Calculate inverse frequency weight: total / (n_classes * class_count)
            weights[c] = float(total) / (n_classes * counts[c])
    # Return the dictionary of class weights
    return weights


# Define the main preprocessing function with configurable window size and stride
def main(window_size=WINDOW_SIZE, stride=STRIDE):
    # 1) Load files and keep only TOP_FEATURES + Label
    # Get list of all CSV files in the data directory
    csv_files = list_csv_files(DATA_DIR)
    # Check if any CSV files were found
    if not csv_files:
        # Raise error if no CSV files exist in the directory
        raise RuntimeError(f'No CSV files found in {DATA_DIR}')

    # Initialize empty list to store dataframes from each file
    frames = []
    # Iterate through each CSV file
    for f in csv_files:
        # Try to read the file header
        try:
            # Read only the column names (first 0 rows) and convert to list
            cols = pd.read_csv(f, nrows=0).columns.str.strip().tolist()
        # Catch any exceptions during header reading
        except Exception:
            # Print warning message if file cannot be read
            print(f'[WARN] Could not read header of {f}, skipping')
            # Skip to next file
            continue
        # Filter TOP_FEATURES to only those present in this file
        keep = [c for c in TOP_FEATURES if c in cols]
        # Check if 'Label' column exists in this file
        if 'Label' in cols:
            # Add 'Label' to the list of columns to keep
            keep = keep + ['Label']
        # If no columns to keep, skip this file
        if not keep:
            # Continue to next file
            continue
        # Try to read the CSV file with selected columns
        try:
            # Read CSV file with only the columns we want to keep
            df = pd.read_csv(f, usecols=keep)
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            # Add the dataframe to our list
            frames.append(df)
        # Catch any exceptions during file reading
        except Exception as e:
            # Print warning message with the error
            print(f'[WARN] Error reading {f}: {e}')
            # Skip to next file
            continue

    # Check if any dataframes were successfully loaded
    if not frames:
        # Raise error if no data was loaded from any file
        raise RuntimeError('No data loaded after scanning CSVs')

    # Concatenate all dataframes vertically and reset index
    data = pd.concat(frames, axis=0, ignore_index=True)
    
    # Strip whitespace from column names (common issue in CSV files)
    data.columns = data.columns.str.strip()

    # 2) Map labels
    # Open the class map JSON file for reading
    with open(CLASS_MAP_PATH, 'r') as cf:
        # Load the JSON content into a dictionary
        raw_map = json.load(cf)
    # Normalize all keys in the class map dictionary
    normalized_map = { _normalize_text(k): v for k, v in raw_map.items() }

    # Check if 'Label' column exists in the merged data
    if 'Label' not in data.columns:
        # Raise error if Label column is missing
        raise RuntimeError(f'Label column not found in merged data. Available columns: {data.columns.tolist()}')

    # Convert labels to string, fill NaN with empty string, and normalize text
    labels_raw = data['Label'].astype(str).fillna('').map(_normalize_text)
    # Map normalized labels to class integers using the normalized map
    labels_mapped = labels_raw.map(normalized_map)
    # Create boolean mask for rows with valid mapped labels (not NaN)
    mask_valid = labels_mapped.notna()
    # Check if any labels were successfully mapped
    if mask_valid.sum() == 0:
        # Raise error if no labels could be mapped to classes
        raise RuntimeError('No labels mapped to classes 0/1/2; check classes_map.json')

    # Drop unknowns
    # Keep only rows with valid labels and reset index
    data = data.loc[mask_valid].reset_index(drop=True)
    # Keep only valid labels, convert to integer, and reset index
    labels = labels_mapped.loc[mask_valid].astype(int).reset_index(drop=True)

    # 3) Feature selection - ensure order of TOP_FEATURES
    # Filter TOP_FEATURES to only those present in the data
    feature_cols = [c for c in TOP_FEATURES if c in data.columns]
    # If no feature columns found
    if not feature_cols:
        # fallback numeric columns except Label
        # Use all numeric columns except 'Label' as fallback
        feature_cols = [c for c in data.select_dtypes(include=[np.number]).columns.tolist() if c != 'Label']

    # Create a copy of the dataframe with only feature columns
    X_df = data[feature_cols].copy()

    # 4) Time-based split (first 80% train, last 20% test)
    # Get the total number of rows
    N = X_df.shape[0]
    # Calculate the index to split at 80% of the data
    split_idx = int(N * 0.8)
    # Take first 80% of features for training and reset index
    X_train_df = X_df.iloc[:split_idx].reset_index(drop=True)
    # Take first 80% of labels for training and reset index
    y_train = labels.iloc[:split_idx].reset_index(drop=True)
    # Take last 20% of features for testing and reset index
    X_test_df = X_df.iloc[split_idx:].reset_index(drop=True)
    # Take last 20% of labels for testing and reset index
    y_test = labels.iloc[split_idx:].reset_index(drop=True)

    # 5) Scaling: fit on train only
    # Initialize MinMaxScaler to scale features to [0, 1] range
    scaler = MinMaxScaler()
    # Fit scaler on training data and transform it to scaled values
    X_train_scaled = scaler.fit_transform(X_train_df.astype(np.float32))
    # Transform test data using the scaler fitted on training data
    X_test_scaled = scaler.transform(X_test_df.astype(np.float32))
    # Save the fitted scaler to disk for later use
    joblib.dump(scaler, SCALER_PATH)

    # 6) Sliding window -> sequences
    # Create training sequences using sliding window on scaled training data
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train.values, window_size=window_size, stride=stride)
    # Create test sequences using sliding window on scaled test data
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test.values, window_size=window_size, stride=stride)

    # 7) Class weights (based on training sequences)
    # Check if training sequences were created
    if y_train_seq.size > 0:
        # Compute class weights based on training sequence labels
        class_weights = compute_class_weights(y_train_seq)
    # If no training sequences
    else:
        # Initialize empty class weights dictionary
        class_weights = {}
    # Ensure classes 0,1,2 exist in dict
    # Iterate through expected class labels
    for c in [0,1,2]:
        # Set default weight of 1.0 for any missing classes
        class_weights.setdefault(c, 1.0)

    # Save class weights
    # Open class weights file for writing
    with open(CLASS_WEIGHTS_PATH, 'w') as wf:
        # Write class weights dictionary to JSON file with indentation
        json.dump(class_weights, wf, indent=2)

    # 8) Save outputs
    # Save training sequences to numpy file as float32
    np.save(os.path.join(OUT_DIR, 'X_train.npy'), X_train_seq.astype(np.float32))
    # Save training labels to numpy file as int32
    np.save(os.path.join(OUT_DIR, 'y_train.npy'), y_train_seq.astype(np.int32))
    # Save test sequences to numpy file as float32
    np.save(os.path.join(OUT_DIR, 'X_test.npy'), X_test_seq.astype(np.float32))
    # Save test labels to numpy file as int32
    np.save(os.path.join(OUT_DIR, 'y_test.npy'), y_test_seq.astype(np.int32))

    # Print the shape of training sequences (samples, window_size, features)
    print(f'X_train shape: {X_train_seq.shape}')
    # Print the distribution of classes in training labels
    print(f'y_train distribution: {Counter(y_train_seq)}')
    # Print the shape of test sequences (samples, window_size, features)
    print(f'X_test shape: {X_test_seq.shape}')
    # Print the distribution of classes in test labels
    print(f'y_test distribution: {Counter(y_test_seq)}')
    # Print the path where scaler was saved
    print(f'Scaler saved to: {SCALER_PATH}')
    # Print the path where class weights were saved
    print(f'Class weights saved to: {CLASS_WEIGHTS_PATH}')


# Check if this script is being run directly (not imported)
if __name__ == '__main__':
    # Call the main function to execute preprocessing
    main()
