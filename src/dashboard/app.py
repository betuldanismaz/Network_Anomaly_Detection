#!/usr/bin/env python3
"""
Network IPS Dashboard - BiLSTM Edition
=======================================
Real-time monitoring dashboard for BiLSTM-based 3-class NIDS
with 5-level risk scoring system.

Author: NIDS Project
Date: 2026-01-01
"""

import os
import sys
import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Path setup
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(PARENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

# Try to import utilities
try:
    from utils.db_manager import fetch_logs
    from utils.firewall_manager import unblock_ip
except ImportError:
    def fetch_logs():
        return pd.DataFrame()
    def unblock_ip(ip):
        return False

# ---------------------------------------------------------------------------
# PATHS - Updated for BiLSTM
# ---------------------------------------------------------------------------
LIVE_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "live_captured_traffic_bilstm.csv")
LIVE_CSV_PATH_OLD = os.path.join(PROJECT_ROOT, "data", "live_captured_traffic.csv")  # Fallback
BILSTM_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "bilstm_best.keras")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler_lstm.pkl")
SCALER_PATH_FALLBACK = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")

# Class mapping
CLASS_NAMES = {
    0: "Benign",
    1: "Volumetric",
    2: "Semantic"
}

CLASS_COLORS = {
    "Benign": "#00CC66",      # Green
    "Volumetric": "#FF4B4B",  # Red
    "Semantic": "#FFA500"     # Orange
}

RISK_LEVELS = {
    1: {"name": "SAFE", "color": "#00CC66", "emoji": "🟢"},
    2: {"name": "LOW", "color": "#3498db", "emoji": "🔵"},
    3: {"name": "MEDIUM", "color": "#FFD700", "emoji": "🟡"},
    4: {"name": "HIGH", "color": "#FFA500", "emoji": "🟠"},
    5: {"name": "CRITICAL", "color": "#FF4B4B", "emoji": "🔴"},
}

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="🛡️ BiLSTM Network IPS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-refresh every 2 seconds
count = st_autorefresh(interval=2000, limit=None, key="auto_refresh")

# Custom CSS
st.markdown("""
<style>
    .risk-gauge {
        text-align: center;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 10px;
        padding: 15px;
        color: white;
    }
    .stMetric {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ BiLSTM Network IPS Dashboard")
st.caption("3-Class Classification: Benign | Volumetric | Semantic")


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def load_live_traffic() -> pd.DataFrame:
    """Load live captured traffic CSV with BiLSTM columns."""
    # Try new BiLSTM CSV first
    csv_path = LIVE_CSV_PATH if os.path.exists(LIVE_CSV_PATH) else LIVE_CSV_PATH_OLD
    
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip', encoding='utf-8', engine='python')
        
        # Handle different column naming conventions
        column_mapping = {
            'Predicted_Class': 'predicted_class',
            'Class_Name': 'class_name',
            'Risk_Level': 'risk_level',
            'Risk_Name': 'risk_name',
            'Prob_Benign': 'prob_benign',
            'Prob_Volumetric': 'prob_volumetric',
            'Prob_Semantic': 'prob_semantic',
            'Action': 'action',
            'Timestamp': 'timestamp',
        }
        
        df.rename(columns=column_mapping, inplace=True)
        
        # Map class IDs to names if needed
        if 'predicted_class' in df.columns and 'class_name' not in df.columns:
            df['class_name'] = df['predicted_class'].map(CLASS_NAMES)
        
        return df
    except Exception as e:
        st.error(f"CSV yükleme hatası: {e}")
        return pd.DataFrame()


def load_logs() -> pd.DataFrame:
    """Load database logs."""
    logs_df = fetch_logs()
    if logs_df.empty:
        return logs_df
    logs_df["timestamp"] = pd.to_datetime(logs_df["timestamp"], errors="coerce")
    return logs_df.sort_values("timestamp", ascending=False)


def get_system_status() -> dict:
    """
    Check system component statuses for BiLSTM.
    
    SIMPLIFIED LOGIC (Hard Links):
    - TensorFlow: If model file exists, TF must be working (you can't load .keras without TF)
    - Scapy: If CSV is being updated < 30s, Scapy must be working (live_bridge uses Scapy)
    - Live Bridge: Check if CSV was modified in last 30 seconds
    """
    # Check CSV first (used for multiple status checks)
    csv_path = LIVE_CSV_PATH if os.path.exists(LIVE_CSV_PATH) else LIVE_CSV_PATH_OLD
    csv_exists = os.path.exists(csv_path)
    csv_age = 999
    csv_rows = 0
    data_flowing = False
    
    if csv_exists:
        try:
            mtime = os.path.getmtime(csv_path)
            csv_age = time.time() - mtime
            data_flowing = csv_age < 30  # Data flowing if updated in last 30 seconds
            
            # Count rows
            try:
                with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                    csv_rows = sum(1 for _ in f) - 1
            except:
                csv_rows = 0
        except:
            csv_age = 999
    
    # Determine statuses based on hard links
    model_exists = os.path.exists(BILSTM_MODEL_PATH)
    scaler_exists = os.path.exists(SCALER_PATH) or os.path.exists(SCALER_PATH_FALLBACK)
    
    status = {
        # Model & Scaler - simple file checks
        "bilstm_model": model_exists,
        "scaler": scaler_exists,
        
        # TensorFlow - HARD LINK: If model exists, TF works (can't load .keras without TF)
        "tensorflow": model_exists,
        "tf_status": "Active" if model_exists else "Inactive",
        
        # Scapy - HARD LINK: If data is flowing, Scapy works (live_bridge uses Scapy)
        "scapy": data_flowing,
        "scapy_status": "Capturing" if data_flowing else ("Waiting" if csv_exists else "Inactive"),
        
        # Live Bridge status
        "live_bridge_status": "active" if data_flowing else ("waiting" if csv_exists and csv_age < 120 else "stopped"),
        "csv_age": csv_age,
        "csv_exists": csv_exists,
        "csv_rows": csv_rows,
        "data_flowing": data_flowing,
    }
    
    return status


def calculate_avg_confidence(df: pd.DataFrame) -> float:
    """Calculate average confidence from probabilities."""
    if df.empty:
        return 0.0
    
    max_probs = []
    for col in ['prob_benign', 'prob_volumetric', 'prob_semantic', 
                'Prob_Benign', 'Prob_Volumetric', 'Prob_Semantic']:
        if col in df.columns:
            max_probs.append(df[col])
    
    if not max_probs:
        return 0.0
    
    prob_df = pd.concat(max_probs, axis=1)
    return prob_df.max(axis=1).mean()


# ---------------------------------------------------------------------------
# RENDER FUNCTIONS
# ---------------------------------------------------------------------------

def render_system_status():
    """
    Render BiLSTM system status indicators.
    Uses HARD LINK logic for green indicators:
    - TensorFlow = Green if model exists
    - Scapy = Green if CSV is being updated
    """
    st.subheader("⚙️ Sistem Durumu")
    status = get_system_status()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Column 1: BiLSTM Model
    with col1:
        if status["bilstm_model"]:
            st.success("✅ BiLSTM Model")
        else:
            st.error("❌ BiLSTM Model")
    
    # Column 2: Scaler
    with col2:
        if status["scaler"]:
            st.success("✅ Scaler")
        else:
            st.error("❌ Scaler")
    
    # Column 3: TensorFlow - GREEN if model exists (hard link)
    with col3:
        if status["tensorflow"]:
            st.success(f"✅ TensorFlow")
        else:
            st.error("❌ TensorFlow")
    
    # Column 4: Scapy - GREEN if data is flowing (hard link)
    with col4:
        if status["scapy"]:
            st.success(f"✅ Scapy")
        elif status["csv_exists"]:
            st.warning("⏳ Scapy")
        else:
            st.error("❌ Scapy")
    
    # Column 5: Live Bridge Status
    with col5:
        if status["data_flowing"]:
            st.success(f"✅ Capturing ({status['csv_age']:.0f}s)")
        elif status["csv_exists"]:
            st.warning(f"⏳ Waiting ({status['csv_age']:.0f}s)")
        else:
            st.error("❌ No Data")
    
    # Additional info row
    if status["csv_exists"]:
        st.caption(f"📊 CSV: {status['csv_rows']:,} rows | Last update: {status['csv_age']:.0f} seconds ago")


def render_risk_gauge(df: pd.DataFrame):
    """Render current risk level gauge based on latest prediction."""
    st.subheader("🎯 Güncel Risk Seviyesi")
    
    if df.empty:
        st.info("Risk verisi bekleniyor...")
        return
    
    # Get latest risk level
    risk_col = 'risk_level' if 'risk_level' in df.columns else 'Risk_Level'
    
    if risk_col not in df.columns:
        # Calculate from class if no risk level
        class_col = 'predicted_class' if 'predicted_class' in df.columns else 'Predicted_Class'
        if class_col in df.columns:
            latest_class = df[class_col].iloc[-1]
            current_risk = 1 if latest_class == 0 else (4 if latest_class == 1 else 5)
        else:
            current_risk = 1
    else:
        current_risk = int(df[risk_col].iloc[-1]) if not pd.isna(df[risk_col].iloc[-1]) else 1
    
    risk_info = RISK_LEVELS.get(current_risk, RISK_LEVELS[1])
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_risk,
        title={'text': f"Risk: {risk_info['name']}", 'font': {'size': 24}},
        delta={'reference': 2, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge={
            'axis': {'range': [1, 5], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': risk_info['color']},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [1, 2], 'color': '#00CC66'},
                {'range': [2, 3], 'color': '#3498db'},
                {'range': [3, 4], 'color': '#FFD700'},
                {'range': [4, 5], 'color': '#FFA500'},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 4
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk level legend
    cols = st.columns(5)
    for i, (level, info) in enumerate(RISK_LEVELS.items()):
        with cols[i]:
            if level == current_risk:
                st.markdown(f"**{info['emoji']} {info['name']}**")
            else:
                st.caption(f"{info['emoji']} {info['name']}")


def render_class_distribution(df: pd.DataFrame):
    """Render 3-class distribution donut chart."""
    st.subheader("📊 Sınıf Dağılımı (3-Class)")
    
    if df.empty:
        st.info("Sınıf verisi bekleniyor...")
        return
    
    # Get class column
    class_col = 'class_name' if 'class_name' in df.columns else 'Class_Name'
    if class_col not in df.columns:
        class_col = 'predicted_class' if 'predicted_class' in df.columns else 'Predicted_Class'
        if class_col in df.columns:
            df['class_name'] = df[class_col].map(CLASS_NAMES)
            class_col = 'class_name'
    
    if class_col not in df.columns:
        st.warning("Sınıf sütunu bulunamadı.")
        return
    
    # Count classes
    class_counts = df[class_col].value_counts().reset_index()
    class_counts.columns = ['Class', 'Count']
    
    # Define colors
    colors = [CLASS_COLORS.get(c, '#808080') for c in class_counts['Class']]
    
    # Create donut chart
    fig = px.pie(
        class_counts,
        values='Count',
        names='Class',
        title='Trafik Sınıf Dağılımı',
        hole=0.6,
        color='Class',
        color_discrete_map=CLASS_COLORS
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        marker=dict(line=dict(color='white', width=2))
    )
    
    # Add center annotation
    total = class_counts['Count'].sum()
    benign_count = class_counts[class_counts['Class'] == 'Benign']['Count'].sum() if 'Benign' in class_counts['Class'].values else 0
    benign_pct = (benign_count / total * 100) if total > 0 else 0
    
    fig.add_annotation(
        text=f"<b>{total:,}</b><br>Toplam Akış",
        x=0.5, y=0.5,
        font_size=16,
        showarrow=False
    )
    
    fig.update_layout(
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_metrics(df: pd.DataFrame):
    """Render key metrics cards."""
    if df.empty:
        total, benign, volumetric, semantic, avg_conf = 0, 0, 0, 0, 0.0
    else:
        total = len(df)
        
        class_col = 'class_name' if 'class_name' in df.columns else 'Class_Name'
        pred_col = 'predicted_class' if 'predicted_class' in df.columns else 'Predicted_Class'
        
        if class_col in df.columns:
            benign = (df[class_col] == 'Benign').sum()
            volumetric = (df[class_col] == 'Volumetric').sum()
            semantic = (df[class_col] == 'Semantic').sum()
        elif pred_col in df.columns:
            benign = (df[pred_col] == 0).sum()
            volumetric = (df[pred_col] == 1).sum()
            semantic = (df[pred_col] == 2).sum()
        else:
            benign, volumetric, semantic = total, 0, 0
        
        avg_conf = calculate_avg_confidence(df)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📊 Toplam Akış", f"{total:,}")
    
    with col2:
        st.metric("🟢 Benign", f"{benign:,}", delta=f"{benign/total*100:.1f}%" if total > 0 else "0%")
    
    with col3:
        attack_count = volumetric + semantic
        st.metric("🔴 Volumetric", f"{volumetric:,}", delta=f"{volumetric/total*100:.1f}%" if total > 0 else "0%", delta_color="inverse")
    
    with col4:
        st.metric("🟠 Semantic", f"{semantic:,}", delta=f"{semantic/total*100:.1f}%" if total > 0 else "0%", delta_color="inverse")
    
    with col5:
        st.metric("🎯 Avg Confidence", f"{avg_conf*100:.1f}%")


def render_confidence_histogram(df: pd.DataFrame):
    """Render confidence score histogram."""
    st.subheader("📊 Güven Skoru Dağılımı")
    
    if df.empty:
        st.info("Güven skoru verisi bekleniyor...")
        return
    
    # Find probability columns
    prob_cols = [c for c in df.columns if 'prob' in c.lower() or 'Prob' in c]
    
    if not prob_cols:
        st.warning("Olasılık sütunları bulunamadı.")
        return
    
    # Get max probability for each row
    prob_df = df[prob_cols].apply(pd.to_numeric, errors='coerce')
    max_probs = prob_df.max(axis=1).dropna()
    
    if len(max_probs) < 5:
        st.info("Yeterli veri yok.")
        return
    
    fig = px.histogram(
        max_probs * 100,
        nbins=20,
        title="Tahmin Güven Skoru Dağılımı",
        labels={'value': 'Güven Skoru (%)', 'count': 'Frekans'},
        color_discrete_sequence=['#3498db']
    )
    
    fig.update_layout(
        showlegend=False,
        height=300
    )
    
    # Add mean line
    mean_conf = max_probs.mean() * 100
    fig.add_vline(x=mean_conf, line_dash="dash", line_color="red",
                  annotation_text=f"Ortalama: {mean_conf:.1f}%")
    
    st.plotly_chart(fig, use_container_width=True)


def render_time_series(df: pd.DataFrame):
    """Render time series of detections."""
    st.subheader("📈 Zaman Serisi Analizi")
    
    if df.empty:
        st.info("Zaman serisi verisi bekleniyor...")
        return
    
    ts_col = 'timestamp' if 'timestamp' in df.columns else 'Timestamp'
    
    if ts_col not in df.columns:
        st.warning("Zaman damgası sütunu bulunamadı.")
        return
    
    df_copy = df.copy()
    df_copy[ts_col] = pd.to_datetime(df_copy[ts_col], errors='coerce')
    df_copy = df_copy.dropna(subset=[ts_col])
    
    if len(df_copy) < 2:
        st.info("Yeterli zaman serisi verisi yok.")
        return
    
    # Resample by minute
    df_copy.set_index(ts_col, inplace=True)
    
    class_col = 'class_name' if 'class_name' in df_copy.columns else 'Class_Name'
    
    if class_col in df_copy.columns:
        # Group by class and time
        time_series = df_copy.groupby([pd.Grouper(freq='1min'), class_col]).size().unstack(fill_value=0)
    else:
        time_series = df_copy.resample('1min').size().to_frame('count')
    
    if time_series.empty:
        st.info("Zaman serisi verisi oluşturulamadı.")
        return
    
    fig = px.line(
        time_series.reset_index(),
        x=ts_col,
        y=time_series.columns.tolist() if isinstance(time_series, pd.DataFrame) else ['count'],
        title="Dakikalık Tespit Frekansı",
        color_discrete_map=CLASS_COLORS
    )
    
    fig.update_layout(
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_recent_detections(df: pd.DataFrame):
    """Render table of recent detections."""
    st.subheader("📋 Son Tespitler")
    
    if df.empty:
        st.info("Tespit verisi bekleniyor...")
        return
    
    # Select relevant columns
    display_cols = []
    col_mapping = {
        'timestamp': 'Zaman',
        'Timestamp': 'Zaman',
        'class_name': 'Sınıf',
        'Class_Name': 'Sınıf',
        'risk_level': 'Risk',
        'Risk_Level': 'Risk',
        'risk_name': 'Durum',
        'Risk_Name': 'Durum',
        'action': 'Aksiyon',
        'Action': 'Aksiyon',
    }
    
    for col in col_mapping.keys():
        if col in df.columns:
            display_cols.append(col)
    
    if not display_cols:
        display_cols = df.columns[:5].tolist()
    
    recent = df[display_cols].tail(20).iloc[::-1]  # Last 20, reversed
    recent = recent.rename(columns={c: col_mapping.get(c, c) for c in recent.columns})
    
    st.dataframe(recent, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

st.sidebar.header("🎮 Kontrol Paneli")

# ---------------------------------------------------------------------------
# MODEL SELECTION
# ---------------------------------------------------------------------------
st.sidebar.subheader("🧠 Aktif Yapay Zeka Modeli")

MODEL_MAPPING = {
    "Random Forest": "rf_model_v1.pkl",
    "Decision Tree": "dt_model.pkl",
    "XGBoost": "xgboost_model.pkl",
    "BiLSTM": "bilstm_model.keras"
}

# Read current active model
ACTIVE_MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "active_model.txt")
os.makedirs(os.path.dirname(ACTIVE_MODEL_PATH), exist_ok=True)

try:
    if os.path.exists(ACTIVE_MODEL_PATH):
        with open(ACTIVE_MODEL_PATH, "r") as f:
            current_model_file = f.read().strip()
        # Find the friendly name
        current_model = next((k for k, v in MODEL_MAPPING.items() if v == current_model_file), "BiLSTM")
    else:
        current_model = "BiLSTM"
        # Create default config
        with open(ACTIVE_MODEL_PATH, "w") as f:
            f.write(MODEL_MAPPING["BiLSTM"])
except Exception:
    current_model = "BiLSTM"

selected_model = st.sidebar.selectbox(
    "Model Seçin",
    list(MODEL_MAPPING.keys()),
    index=list(MODEL_MAPPING.keys()).index(current_model),
    key="model_selector"
)

# Write to config file when selection changes
if selected_model != current_model:
    try:
        with open(ACTIVE_MODEL_PATH, "w") as f:
            f.write(MODEL_MAPPING[selected_model])
        st.sidebar.success(f"✅ Model değiştirildi: {selected_model}")
        st.sidebar.info("⏳ Consumer yeni modeli birkaç saniye içinde yükleyecek...")
    except Exception as e:
        st.sidebar.error(f"❌ Model yazma hatası: {e}")

st.sidebar.caption(f"📊 Şu an aktif: **{selected_model}**")
st.sidebar.caption("**Accuracy:** 99.15% | **Classes:** 3")
st.sidebar.markdown("---")

# Risk level legend
st.sidebar.subheader("🎯 Risk Seviyeleri")
for level, info in RISK_LEVELS.items():
    st.sidebar.markdown(f"{info['emoji']} **Level {level}**: {info['name']}")

st.sidebar.divider()

# IP Unblock
st.sidebar.subheader("🔓 IP Engel Kaldırma")
ip_to_unblock = st.sidebar.text_input("IP Adresi")
if st.sidebar.button("Engeli Kaldır"):
    if ip_to_unblock.strip():
        success = unblock_ip(ip_to_unblock.strip())
        if success:
            st.sidebar.success(f"✅ {ip_to_unblock} engeli kaldırıldı.")
        else:
            st.sidebar.warning(f"⚠️ İşlem başarısız.")
    else:
        st.sidebar.warning("Geçerli bir IP girin.")

st.sidebar.divider()
st.sidebar.caption(f"🔄 Son yenileme: {datetime.now().strftime('%H:%M:%S')}")
st.sidebar.caption(f"📊 Refresh #{count}")


# ---------------------------------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------------------------------

# Load data
live_df = load_live_traffic()
logs_df = load_logs()

# System Status
render_system_status()
st.divider()

# Metrics Row
render_metrics(live_df)
st.divider()

# Risk Gauge & Class Distribution
col_gauge, col_dist = st.columns([1, 2])
with col_gauge:
    render_risk_gauge(live_df)
with col_dist:
    render_class_distribution(live_df)

st.divider()

# Confidence & Time Series
col_conf, col_time = st.columns(2)
with col_conf:
    render_confidence_histogram(live_df)
with col_time:
    render_time_series(live_df)

st.divider()

# Recent Detections Table
render_recent_detections(live_df)
