#!/usr/bin/env python3
"""
Kafka Consumer for Network Anomaly Detection System
Consumes network flow features from Kafka, runs ML prediction, and logs results.
"""

import os
import sys
import json
import time
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from confluent_kafka import Consumer, KafkaError

# ---------------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "rf_model_v1.pkl")  # Default, will be overridden
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")
SCALER_LSTM_PATH = os.path.join(PROJECT_ROOT, "models", "scaler_lstm.pkl")  # For LSTM models
CSV_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "live_captured_traffic.csv")
ACTIVE_MODEL_CONFIG = os.path.join(PROJECT_ROOT, "data", "active_model.txt")

# Add utils to path for database logging
sys.path.append(os.path.join(CURRENT_DIR, "utils"))

try:
    from db_manager import log_attack
    from firewall_manager import block_ip
    DB_AVAILABLE = True
except ImportError:
    print("⚠️  WARNING: db_manager or firewall_manager not found. Logging disabled.")
    DB_AVAILABLE = False
    def log_attack(*args, **kwargs):
        pass
    def block_ip(*args, **kwargs):
        pass

# ---------------------------------------------------------------------------
# ANSI COLOR CODES
# ---------------------------------------------------------------------------
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RESET = '\033[0m'
BOLD = '\033[1m'

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
KAFKA_BOOTSTRAP_SERVERS = '127.0.0.1:9092'
KAFKA_TOPIC = 'network-traffic'
KAFKA_GROUP_ID = 'nids-consumer-group-v1'
WHITELIST_IPS = os.getenv("WHITELIST_IPS", "192.168.1.1,127.0.0.1,0.0.0.0,localhost").split(",")

# Expected feature count from producer
EXPECTED_FEATURE_COUNT = 78

# ---------------------------------------------------------------------------
# GLOBAL MODEL & SCALER
# ---------------------------------------------------------------------------
MODEL = None
SCALER = None
CURRENT_MODEL_NAME = "rf_model_v1.pkl"  # Track currently loaded model
CURRENT_MODEL_TYPE = "sklearn"  # 'sklearn' or 'keras'
LAST_CONFIG_CHECK = 0  # Timestamp of last config file check
CONFIG_CHECK_INTERVAL = 5  # Check config file every 5 seconds

# Statistics tracking
STATS = {
    "total_processed": 0,
    "attacks_detected": 0,
    "clean_traffic": 0,
    "errors": 0,
    "start_time": datetime.now()
}


def initialize_csv_file():
    """Initialize CSV output file with headers if it doesn't exist."""
    os.makedirs(os.path.dirname(CSV_OUTPUT_PATH), exist_ok=True)
    
    if not os.path.exists(CSV_OUTPUT_PATH):
        # Create header row: Timestamp, Src_IP, Dst_IP, Predicted_Label, Confidence_Score, Model_Used
        header_df = pd.DataFrame(columns=[
            "Timestamp", "Src_IP", "Dst_IP", "Predicted_Label", 
            "Confidence_Score", "Model_Used", "Processing_Time_Ms"
        ])
        header_df.to_csv(CSV_OUTPUT_PATH, index=False)
        print(f"{GREEN}✅ CSV output file initialized: {CSV_OUTPUT_PATH}{RESET}")
    else:
        print(f"{CYAN}ℹ️  Using existing CSV: {CSV_OUTPUT_PATH}{RESET}")


def load_model_and_scaler(model_filename=None):
    """Load the ML model and scaler dynamically based on model type.
    
    Args:
        model_filename: Name of model file (e.g., 'xgboost_model.pkl'). If None, reads from config.
    """
    global MODEL, SCALER, CURRENT_MODEL_NAME, CURRENT_MODEL_TYPE
    
    # If no filename provided, read from config file
    if model_filename is None:
        if os.path.exists(ACTIVE_MODEL_CONFIG):
            try:
                with open(ACTIVE_MODEL_CONFIG, 'r') as f:
                    model_filename = f.read().strip()
            except Exception:
                model_filename = "rf_model_v1.pkl"  # Fallback default
        else:
            model_filename = "rf_model_v1.pkl"  # Default
            # Create default config file
            os.makedirs(os.path.dirname(ACTIVE_MODEL_CONFIG), exist_ok=True)
            try:
                with open(ACTIVE_MODEL_CONFIG, 'w') as f:
                    f.write(model_filename)
            except Exception:
                pass
    
    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}🔧 Loading ML Model & Scaler...{RESET}")
    print(f"{CYAN}{'='*60}{RESET}")
    print(f"{CYAN}   Target Model: {model_filename}{RESET}")
    
    # Build model path
    model_path = os.path.join(PROJECT_ROOT, "models", model_filename)
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"{RED}❌ CRITICAL ERROR: Model file not found!{RESET}")
        print(f"   Expected path: {model_path}")
        print(f"   Available models in models/ directory:")
        try:
            model_files = [f for f in os.listdir(os.path.join(PROJECT_ROOT, "models")) 
                          if f.endswith(('.pkl', '.keras', '.h5'))]
            for mf in model_files:
                print(f"      - {mf}")
        except Exception:
            pass
        sys.exit(1)
    
    # Determine model type based on file extension
    is_keras_model = model_filename.endswith(('.keras', '.h5'))
    
    try:
        if is_keras_model:
            # Load Keras/TensorFlow model
            try:
                from tensorflow.keras.models import load_model as keras_load_model
            except ImportError:
                print(f"{RED}❌ ERROR: TensorFlow not installed!{RESET}")
                print(f"   Install with: pip install tensorflow")
                sys.exit(1)
            
            MODEL = keras_load_model(model_path)
            CURRENT_MODEL_TYPE = "keras"
            print(f"{GREEN}✅ Keras/LSTM Model loaded successfully{RESET}")
            print(f"{CYAN}   Model type: LSTM/BiLSTM{RESET}")
            
            # Try to load LSTM scaler if available, otherwise use standard scaler
            if os.path.exists(SCALER_LSTM_PATH):
                SCALER = joblib.load(SCALER_LSTM_PATH)
                print(f"{GREEN}✅ LSTM Scaler loaded{RESET}")
            elif os.path.exists(SCALER_PATH):
                SCALER = joblib.load(SCALER_PATH)
                print(f"{GREEN}✅ Standard Scaler loaded{RESET}")
            else:
                print(f"{YELLOW}⚠️  WARNING: No scaler found, predictions may be inaccurate{RESET}")
                SCALER = None
        else:
            # Load scikit-learn model (pkl)
            MODEL = joblib.load(model_path)
            CURRENT_MODEL_TYPE = "sklearn"
            print(f"{GREEN}✅ Scikit-learn Model loaded successfully{RESET}")
            print(f"{CYAN}   Model type: {type(MODEL).__name__}{RESET}")
            
            # Load standard scaler
            if not os.path.exists(SCALER_PATH):
                print(f"{RED}❌ CRITICAL ERROR: Scaler file not found!{RESET}")
                print(f"   Expected path: {SCALER_PATH}")
                sys.exit(1)
            
            SCALER = joblib.load(SCALER_PATH)
            print(f"{GREEN}✅ StandardScaler loaded successfully{RESET}")
        
        CURRENT_MODEL_NAME = model_filename
        print(f"{CYAN}   Features expected: {EXPECTED_FEATURE_COUNT}{RESET}\n")
        
    except Exception as exc:
        print(f"{RED}❌ CRITICAL ERROR: Failed to load model/scaler!{RESET}")
        print(f"   Error: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def create_consumer():
    """Initialize and return Kafka Consumer instance."""
    print(f"{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}📡 Initializing Kafka Consumer...{RESET}")
    print(f"{CYAN}{'='*60}{RESET}")
    
    conf = {
        'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
        'group.id': KAFKA_GROUP_ID,
        'auto.offset.reset': 'latest',  # Start from latest messages
        'enable.auto.commit': True,
        'auto.commit.interval.ms': 5000,
        'session.timeout.ms': 30000,
        'max.poll.interval.ms': 300000
    }
    
    try:
        consumer = Consumer(conf)
        consumer.subscribe([KAFKA_TOPIC])
        print(f"{GREEN}✅ Consumer connected to Kafka{RESET}")
        print(f"{CYAN}   Bootstrap servers: {KAFKA_BOOTSTRAP_SERVERS}{RESET}")
        print(f"{CYAN}   Topic: {KAFKA_TOPIC}{RESET}")
        print(f"{CYAN}   Group ID: {KAFKA_GROUP_ID}{RESET}\n")
        return consumer
    except Exception as exc:
        print(f"{RED}❌ CRITICAL ERROR: Failed to create Kafka consumer!{RESET}")
        print(f"   Error: {exc}")
        print(f"\n{YELLOW}💡 Make sure Kafka is running: docker-compose up -d{RESET}")
        sys.exit(1)


def process_message(message_value):
    """
    Process a single Kafka message: parse, predict, log.
    
    Args:
        message_value: Raw message bytes from Kafka
        
    Returns:
        bool: True if processing successful, False otherwise
    """
    start_time = time.time()
    
    try:
        # 1. PARSE JSON MESSAGE
        message_data = json.loads(message_value.decode('utf-8'))
        
        timestamp = message_data.get("timestamp", datetime.now().isoformat())
        src_ip = message_data.get("src_ip", "Unknown")
        dst_ip = message_data.get("dst_ip", "Unknown")
        features_dict = message_data.get("features", {})
        producer_id = message_data.get("producer_id", "unknown")
        
        # Validate feature count
        if len(features_dict) != EXPECTED_FEATURE_COUNT:
            print(f"{YELLOW}⚠️  WARNING: Expected {EXPECTED_FEATURE_COUNT} features, got {len(features_dict)}{RESET}")
        
        # 2. CONVERT FEATURES TO DATAFRAME
        # Features dict has column names as keys, need to maintain order
        features_df = pd.DataFrame([features_dict])
        
        # Align features to match exactly what the scaler/model expects
        if hasattr(SCALER, 'feature_names_in_'):
            expected_features = SCALER.feature_names_in_
            # Reindex dataframe: keeps matching columns, adds missing ones with 0.0, drops extra ones
            features_df = features_df.reindex(columns=expected_features, fill_value=0.0)
        
        # Handle any missing or extra columns (align with scaler expectations)
        # The scaler was fitted on specific columns, so we need to match that order
        try:
            # 3. SCALE FEATURES
            features_scaled = SCALER.transform(features_df)
            features_scaled_df = pd.DataFrame(
                features_scaled,
                columns=features_df.columns,
                index=features_df.index
            )
        except Exception as e:
            print(f"{RED}⚠️  Scaling error: {e}{RESET}")
            # Try with filling missing values
            features_df = features_df.fillna(0)
            features_scaled = SCALER.transform(features_df)
            features_scaled_df = pd.DataFrame(
                features_scaled,
                columns=features_df.columns
            )
        
        # 4. MAKE PREDICTION (handle different model types)
        if CURRENT_MODEL_TYPE == "keras":
            # LSTM models expect 3D input: (samples, timesteps, features)
            # Reshape from (1, 78) to (1, 1, 78)
            features_for_prediction = features_scaled.reshape((features_scaled.shape[0], 1, features_scaled.shape[1]))
            
            # Keras model returns probabilities directly
            prediction_proba = MODEL.predict(features_for_prediction, verbose=0)[0]
            
            # For binary classification, threshold at 0.5
            if len(prediction_proba) == 1:
                # Single output neuron (sigmoid)
                confidence_score = float(prediction_proba[0])
                prediction = 1 if confidence_score > 0.5 else 0
            else:
                # Multiple output neurons (softmax) - take argmax
                prediction = int(np.argmax(prediction_proba))
                confidence_score = float(prediction_proba[prediction])
        else:
            # Scikit-learn models (RF, DT, XGB) use 2D input
            prediction = MODEL.predict(features_scaled_df)[0]
            
            # Get confidence score (probability of attack)
            try:
                probabilities = MODEL.predict_proba(features_scaled_df)[0]
                confidence_score = float(probabilities[1])  # Probability of class 1 (attack)
            except AttributeError:
                # Model doesn't support predict_proba
                confidence_score = float(prediction)
        
        # 5. DETERMINE ACTION
        is_attack = (prediction == 1)
        processing_time_ms = (time.time() - start_time) * 1000
        
        # 6. LOG TO CSV
        log_entry = {
            "Timestamp": timestamp,
            "Src_IP": src_ip,
            "Dst_IP": dst_ip,
            "Predicted_Label": int(prediction),
            "Confidence_Score": round(confidence_score, 4),
            "Model_Used": CURRENT_MODEL_NAME.replace('.pkl', '').replace('.keras', ''),
            "Processing_Time_Ms": round(processing_time_ms, 2)
        }
        
        # Append to CSV
        log_df = pd.DataFrame([log_entry])
        log_df.to_csv(CSV_OUTPUT_PATH, mode='a', header=False, index=False)
        
        # 7. LOG TO DATABASE (if available)
        if DB_AVAILABLE:
            if is_attack:
                if src_ip not in WHITELIST_IPS and src_ip != "Unknown":
                    log_attack(src_ip, "BLOCKED", f"Attack detected (confidence: {confidence_score:.2%})")
                    # Optionally block IP
                    # block_ip(src_ip)  # Uncomment to enable auto-blocking
                else:
                    log_attack(src_ip, "ALLOWED", f"Attack detected but whitelisted (confidence: {confidence_score:.2%})")
            else:
                # Optionally log normal traffic (commented to avoid DB bloat)
                # log_attack(src_ip, "NORMAL", "Clean traffic")
                pass
        
        # 8. UPDATE STATISTICS
        STATS["total_processed"] += 1
        if is_attack:
            STATS["attacks_detected"] += 1
        else:
            STATS["clean_traffic"] += 1
        
        # 9. TERMINAL OUTPUT
        current_time = datetime.now().strftime("%H:%M:%S")
        
        if is_attack:
            print(f"{RED}{BOLD}🚨 [{current_time}] ALERT: ATTACK DETECTED!{RESET}")
            print(f"{RED}   Source IP: {src_ip} → Destination: {dst_ip}{RESET}")
            print(f"{RED}   Confidence: {confidence_score:.2%} | Processing: {processing_time_ms:.2f}ms{RESET}")
            if src_ip in WHITELIST_IPS:
                print(f"{YELLOW}   ⚠️  IP is whitelisted - not blocking{RESET}")
        else:
            print(f"{GREEN}✅ [{current_time}] Clean Traffic{RESET}", end="")
            print(f"{GREEN} | {src_ip} → {dst_ip} | Confidence: {confidence_score:.2%} | {processing_time_ms:.2f}ms{RESET}")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"{RED}⚠️  JSON parsing error: {e}{RESET}")
        STATS["errors"] += 1
        return False
    except Exception as e:
        print(f"{RED}⚠️  Processing error: {e}{RESET}")
        import traceback
        traceback.print_exc()
        STATS["errors"] += 1
        return False


def print_statistics():
    """Print consumer statistics."""
    runtime = datetime.now() - STATS["start_time"]
    runtime_seconds = runtime.total_seconds()
    
    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}📊 CONSUMER STATISTICS{RESET}")
    print(f"{CYAN}{'='*60}{RESET}")
    print(f"   Runtime: {runtime}")
    print(f"   Total Processed: {STATS['total_processed']}")
    print(f"   {RED}Attacks Detected: {STATS['attacks_detected']}{RESET}")
    print(f"   {GREEN}Clean Traffic: {STATS['clean_traffic']}{RESET}")
    print(f"   {YELLOW}Errors: {STATS['errors']}{RESET}")
    if runtime_seconds > 0:
        print(f"   Throughput: {STATS['total_processed']/runtime_seconds:.2f} messages/sec")
    print(f"{CYAN}{'='*60}{RESET}\n")


def check_and_reload_model():
    """Check if model config has changed and reload if necessary."""
    global LAST_CONFIG_CHECK, CURRENT_MODEL_NAME
    
    current_time = time.time()
    
    # Only check every CONFIG_CHECK_INTERVAL seconds
    if current_time - LAST_CONFIG_CHECK < CONFIG_CHECK_INTERVAL:
        return
    
    LAST_CONFIG_CHECK = current_time
    
    # Read current config
    if not os.path.exists(ACTIVE_MODEL_CONFIG):
        return
    
    try:
        with open(ACTIVE_MODEL_CONFIG, 'r') as f:
            requested_model = f.read().strip()
        
        # Check if model changed
        if requested_model != CURRENT_MODEL_NAME:
            print(f"\n{YELLOW}{'='*60}{RESET}")
            print(f"{BOLD}{YELLOW}🔄 MODEL SWITCH DETECTED!{RESET}")
            print(f"{YELLOW}   Current: {CURRENT_MODEL_NAME}{RESET}")
            print(f"{YELLOW}   New:     {requested_model}{RESET}")
            print(f"{YELLOW}{'='*60}{RESET}")
            
            # Reload model
            load_model_and_scaler(requested_model)
            
            print(f"{GREEN}✅ Model switch complete! Now using: {CURRENT_MODEL_NAME}{RESET}\n")
    except Exception as e:
        print(f"{RED}⚠️  Error checking model config: {e}{RESET}")


def main():
    """Main consumer loop."""
    print(f"\n{BOLD}{CYAN}╔{'═'*58}╗{RESET}")
    print(f"{BOLD}{CYAN}║{' '*10}🛡️  KAFKA CONSUMER - NETWORK IPS{' '*15}║{RESET}")
    print(f"{BOLD}{CYAN}╚{'═'*58}╝{RESET}\n")
    
    # Initialize components
    load_model_and_scaler()
    initialize_csv_file()
    consumer = create_consumer()
    
    print(f"{GREEN}{BOLD}🚀 Consumer is now ACTIVE and listening for messages...{RESET}")
    print(f"{YELLOW}⏹️  Press CTRL+C to stop{RESET}")
    print(f"{CYAN}💡 Model can be changed dynamically via Dashboard{RESET}\n")
    print(f"{CYAN}{'─'*60}{RESET}\n")
    
    message_count = 0
    last_stats_print = time.time()
    
    try:
        while True:
            # Check if model config changed (every 5 seconds)
            check_and_reload_model()
            
            # Poll for messages
            msg = consumer.poll(timeout=1.0)
            
            if msg is None:
                # No message available, continue polling
                continue
            
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    # End of partition, not an error
                    continue
                else:
                    print(f"{RED}⚠️  Kafka error: {msg.error()}{RESET}")
                    continue
            
            # Process the message
            message_count += 1
            process_message(msg.value())
            
            # Print statistics every 50 messages or every 30 seconds
            if message_count % 50 == 0 or (time.time() - last_stats_print) > 30:
                print_statistics()
                last_stats_print = time.time()
    
    except KeyboardInterrupt:
        print(f"\n{YELLOW}🛑 Consumer stopped by user{RESET}")
        print_statistics()
    
    except Exception as e:
        print(f"\n{RED}❌ Fatal error in consumer loop: {e}{RESET}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean shutdown
        print(f"{CYAN}⏳ Closing consumer...{RESET}")
        consumer.close()
        print(f"{GREEN}✅ Consumer closed successfully{RESET}")
        print(f"{CYAN}📁 Results saved to: {CSV_OUTPUT_PATH}{RESET}\n")


if __name__ == "__main__":
    main()
