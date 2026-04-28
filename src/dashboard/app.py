#!/usr/bin/env python3
"""
Network IPS Dashboard - SOC Edition
=====================================
Multi-tab Security Operations Center dashboard for BiLSTM-based 3-class NIDS.
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

# ---------------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(PARENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

try:
    from utils.db_manager import fetch_logs
    from utils.firewall_manager import unblock_ip
except ImportError:
    def fetch_logs():
        return pd.DataFrame()
    def unblock_ip(ip):
        return False

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
LIVE_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "live_captured_traffic_bilstm.csv")
LIVE_CSV_PATH_OLD = os.path.join(PROJECT_ROOT, "data", "live_captured_traffic.csv")
BILSTM_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "bilstm_best.keras")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler_lstm.pkl")
SCALER_PATH_FALLBACK = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")
ACTIVE_MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "active_model.txt")

CLASS_NAMES = {0: "Benign", 1: "Volumetric", 2: "Semantic"}
CLASS_COLORS = {"Benign": "#00CC66", "Volumetric": "#FF4B4B", "Semantic": "#FFA500"}
RISK_LEVELS = {
    1: {"name": "SAFE",     "color": "#00CC66", "emoji": "🟢"},
    2: {"name": "LOW",      "color": "#3498db", "emoji": "🔵"},
    3: {"name": "MEDIUM",   "color": "#FFD700", "emoji": "🟡"},
    4: {"name": "HIGH",     "color": "#FFA500", "emoji": "🟠"},
    5: {"name": "CRITICAL", "color": "#FF4B4B", "emoji": "🔴"},
}
MODEL_MAPPING = {
    "Random Forest": "rf_model_v1.pkl",
    "Decision Tree": "dt_model.pkl",
    "XGBoost":       "xgboost_model.pkl",
    "BiLSTM":        "bilstm_model.keras",
}

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="🛡️ SOC Network IPS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Auto-refresh logic driven by session state
if "live_mode" not in st.session_state:
    st.session_state.live_mode = True
if "refresh_interval" not in st.session_state:
    st.session_state.refresh_interval = 15

if st.session_state.live_mode:
    count = st_autorefresh(interval=st.session_state.refresh_interval * 1000, limit=None, key="soc_autorefresh")
else:
    count = 0

# ---------------------------------------------------------------------------
# DARK-MODE CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  [data-testid="stAppViewContainer"] { background: #0a0e17; color: #c9d1d9; }
  [data-testid="stSidebar"] { background: #0d1220; border-right: 1px solid #1e2d45; }

  /* Metric cards */
  div[data-testid="metric-container"], .kpi-card { 
    background: rgba(255,255,255,0.04); 
    border: 1px solid rgba(255,255,255,0.08); 
    border-radius: 12px; 
    backdrop-filter: blur(12px); 
    padding: 20px; 
    transition: transform .2s, box-shadow .2s; 
  }
  div[data-testid="metric-container"]:hover, .kpi-card:hover { 
    transform: translateY(-3px); 
    box-shadow: 0 8px 32px rgba(0,200,255,0.1); 
  }

  /* Critical Alerts */
  @keyframes criticalPulse { 
    0%,100% { box-shadow: 0 0 0 0 rgba(255,75,75,0.4);} 
    50% { box-shadow: 0 0 0 12px rgba(255,75,75,0);} 
  }
  .alert-critical { 
    animation: criticalPulse 1.5s infinite; 
    border: 1px solid #ff4b4b !important; 
    border-radius: 12px;
  }

  /* Tab bar */
  div[data-testid="stTabs"] button[data-baseweb="tab"] {
    background: transparent;
    color: #8b949e;
    border-bottom: 2px solid transparent;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 8px 16px;
    transition: all 0.2s ease;
  }
  div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {
    color: #58a6ff;
  }
  div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom: 2px solid #58a6ff !important;
  }

  /* Plotly charts */
  .js-plotly-plot { border-radius: 10px; }

  /* Dataframe */
  div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

  /* Divider */
  hr { border-color: #21262d; }

  /* Status badges */
  .badge-ok   { background:#0f5132; color:#75b798; padding:3px 10px; border-radius:20px; font-size:.75rem; font-weight:600; }
  .badge-warn { background:#5c3f00; color:#e3b341; padding:3px 10px; border-radius:20px; font-size:.75rem; font-weight:600; }
  .badge-err  { background:#3d0000; color:#ff7b72; padding:3px 10px; border-radius:20px; font-size:.75rem; font-weight:600; }

  /* Header accent */
  .soc-header { font-size:1.6rem; font-weight:700; color:#58a6ff; letter-spacing:-0.5px; }
  .soc-sub    { color:#8b949e; font-size:.85rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_live_traffic() -> pd.DataFrame:
    csv_path = LIVE_CSV_PATH if os.path.exists(LIVE_CSV_PATH) else LIVE_CSV_PATH_OLD
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path, on_bad_lines="skip", encoding="utf-8", engine="python")
        col_map = {
            "Predicted_Class": "predicted_class", "Class_Name": "class_name",
            "Risk_Level": "risk_level", "Risk_Name": "risk_name",
            "Prob_Benign": "prob_benign", "Prob_Volumetric": "prob_volumetric",
            "Prob_Semantic": "prob_semantic", "Action": "action", "Timestamp": "timestamp",
        }
        df.rename(columns=col_map, inplace=True)
        if "predicted_class" in df.columns and "class_name" not in df.columns:
            df["class_name"] = df["predicted_class"].map(CLASS_NAMES)
        return df
    except Exception as e:
        st.error(f"CSV load error: {e}")
        return pd.DataFrame()


def load_logs() -> pd.DataFrame:
    df = fetch_logs()
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.sort_values("timestamp", ascending=False)


def get_system_status() -> dict:
    csv_path = LIVE_CSV_PATH if os.path.exists(LIVE_CSV_PATH) else LIVE_CSV_PATH_OLD
    csv_exists = os.path.exists(csv_path)
    csv_age, csv_rows, data_flowing = 999, 0, False
    if csv_exists:
        try:
            mtime = os.path.getmtime(csv_path)
            csv_age = time.time() - mtime
            data_flowing = csv_age < 30
            with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
                csv_rows = sum(1 for _ in f) - 1
        except Exception:
            pass
    model_exists = os.path.exists(BILSTM_MODEL_PATH)
    scaler_exists = os.path.exists(SCALER_PATH) or os.path.exists(SCALER_PATH_FALLBACK)
    return {
        "bilstm_model": model_exists, "scaler": scaler_exists,
        "tensorflow": model_exists, "scapy": data_flowing,
        "scapy_status": "Capturing" if data_flowing else ("Waiting" if csv_exists else "Inactive"),
        "live_bridge_status": "active" if data_flowing else ("waiting" if csv_exists and csv_age < 120 else "stopped"),
        "csv_age": csv_age, "csv_exists": csv_exists, "csv_rows": csv_rows, "data_flowing": data_flowing,
    }


def calculate_avg_confidence(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    prob_cols = [c for c in df.columns if "prob" in c.lower()]
    if not prob_cols:
        return 0.0
    return df[prob_cols].apply(pd.to_numeric, errors="coerce").max(axis=1).mean()

# ---------------------------------------------------------------------------
# RENDER HELPERS
# ---------------------------------------------------------------------------

def render_system_status(status: dict):
    st.markdown("#### ⚙️ System Status")
    c1, c2, c3, c4, c5 = st.columns(5)
    def badge(col, label, ok, warn=False):
        cls = "badge-ok" if ok else ("badge-warn" if warn else "badge-err")
        icon = "✅" if ok else ("⏳" if warn else "❌")
        col.markdown(f'<span class="{cls}">{icon} {label}</span>', unsafe_allow_html=True)
    badge(c1, "BiLSTM",     status["bilstm_model"])
    badge(c2, "Scaler",     status["scaler"])
    badge(c3, "TensorFlow", status["tensorflow"])
    badge(c4, "Scapy",      status["scapy"], warn=status["csv_exists"] and not status["scapy"])
    ok5 = status["data_flowing"]
    warn5 = status["csv_exists"] and not ok5
    badge(c5, f"Bridge ({status['csv_age']:.0f}s)", ok5, warn=warn5)
    if status["csv_exists"]:
        st.caption(f"📊 CSV: {status['csv_rows']:,} rows | Last update: {status['csv_age']:.0f}s ago")


def render_metrics(df: pd.DataFrame):
    total = len(df)
    if df.empty:
        benign = volumetric = semantic = 0
        avg_conf = 0.0
    else:
        c = "class_name" if "class_name" in df.columns else "Class_Name"
        p = "predicted_class" if "predicted_class" in df.columns else "Predicted_Class"
        if c in df.columns:
            benign    = int((df[c] == "Benign").sum())
            volumetric = int((df[c] == "Volumetric").sum())
            semantic  = int((df[c] == "Semantic").sum())
        elif p in df.columns:
            benign, volumetric, semantic = int((df[p]==0).sum()), int((df[p]==1).sum()), int((df[p]==2).sum())
        else:
            benign, volumetric, semantic = total, 0, 0
        avg_conf = calculate_avg_confidence(df)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📊 Total Flows", f"{total:,}")
    c2.metric("🟢 Benign",      f"{benign:,}",    delta=f"{benign/total*100:.1f}%" if total else "0%")
    c3.metric("🔴 Volumetric",  f"{volumetric:,}", delta=f"{volumetric/total*100:.1f}%" if total else "0%", delta_color="inverse")
    c4.metric("🟠 Semantic",    f"{semantic:,}",   delta=f"{semantic/total*100:.1f}%" if total else "0%",  delta_color="inverse")
    c5.metric("🎯 Avg Confidence", f"{avg_conf*100:.1f}%")


def render_risk_gauge(df: pd.DataFrame):
    st.markdown("#### 🎯 Current Risk Level")
    if df.empty:
        st.info("Waiting for data…")
        return
    risk_col = "risk_level" if "risk_level" in df.columns else "Risk_Level"
    if risk_col in df.columns:
        current_risk = int(df[risk_col].iloc[-1]) if not pd.isna(df[risk_col].iloc[-1]) else 1
    else:
        pred = "predicted_class" if "predicted_class" in df.columns else "Predicted_Class"
        lc = df[pred].iloc[-1] if pred in df.columns else 0
        current_risk = 1 if lc == 0 else (4 if lc == 1 else 5)
    info = RISK_LEVELS.get(current_risk, RISK_LEVELS[1])
    
    div_class = "alert-critical" if current_risk == 5 else ""
    if div_class:
        st.markdown(f'<div class="{div_class}">', unsafe_allow_html=True)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_risk,
        title={"text": f"Risk: {info['name']}", "font": {"size": 20, "color": "#c9d1d9"}},
        delta={"reference": 2, "increasing": {"color": "#ff4b4b"}, "decreasing": {"color": "#00cc66"}},
        gauge={
            "axis": {"range": [1, 5], "tickcolor": "#8b949e"},
            "bar":  {"color": info["color"]},
            "bgcolor": "rgba(255,255,255,0.05)",
            "steps": [
                {"range": [1, 2], "color": "#0f3d22"},
                {"range": [2, 3], "color": "#0d2b4a"},
                {"range": [3, 4], "color": "#3d3000"},
                {"range": [4, 5], "color": "#3d1f00"},
            ],
            "threshold": {"line": {"color": "#ff4b4b", "width": 4}, "thickness": 0.75, "value": 4},
        }
    ))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=50, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9")
    st.plotly_chart(fig, use_container_width=True)
    
    if div_class:
        st.markdown('</div>', unsafe_allow_html=True)


def render_class_distribution(df: pd.DataFrame):
    st.markdown("#### 📊 Class Distribution (3-Class)")
    if df.empty:
        st.info("Waiting for data…")
        return
    col = "class_name" if "class_name" in df.columns else "Class_Name"
    if col not in df.columns:
        st.warning("Class column not found.")
        return
    counts = df[col].value_counts().reset_index()
    counts.columns = ["Class", "Count"]
    fig = px.pie(counts, values="Count", names="Class", hole=0.6,
                 color="Class", color_discrete_map=CLASS_COLORS,
                 title="Traffic Class Distribution")
    fig.update_traces(textposition="inside", textinfo="percent+label",
                      marker=dict(line=dict(color="#0d1117", width=2)))
    total = counts["Count"].sum()
    fig.add_annotation(text=f"<b>{total:,}</b><br>Total", x=0.5, y=0.5,
                       font_size=14, showarrow=False, font_color="#c9d1d9")
    fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9",
                      legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"))
    st.plotly_chart(fig, use_container_width=True)


def render_time_series(df: pd.DataFrame):
    st.markdown("#### 📈 Detection Time Series")
    if df.empty:
        st.info("Waiting for data…")
        return
    ts = "timestamp" if "timestamp" in df.columns else "Timestamp"
    if ts not in df.columns:
        st.warning("Timestamp column not found.")
        return
    tmp = df.copy()
    tmp[ts] = pd.to_datetime(tmp[ts], errors="coerce")
    tmp = tmp.dropna(subset=[ts]).set_index(ts)
    if len(tmp) < 2:
        st.info("Not enough data for time series.")
        return
    c = "class_name" if "class_name" in tmp.columns else "Class_Name"
    if c in tmp.columns:
        series = tmp.groupby([pd.Grouper(freq="1min"), c]).size().unstack(fill_value=0)
    else:
        series = tmp.resample("1min").size().to_frame("count")
    fig = px.line(series.reset_index(), x=ts,
                  y=series.columns.tolist(),
                  color_discrete_map=CLASS_COLORS,
                  title="Detections per Minute")
    fig.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9",
                      legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
                      plot_bgcolor="rgba(255,255,255,0.05)")
    st.plotly_chart(fig, use_container_width=True)


def render_confidence_histogram(df: pd.DataFrame):
    st.markdown("#### 📊 Confidence Score Distribution")
    if df.empty:
        st.info("Waiting for data…")
        return
    prob_cols = [c for c in df.columns if "prob" in c.lower()]
    if not prob_cols:
        st.warning("Probability columns not found.")
        return
    max_probs = df[prob_cols].apply(pd.to_numeric, errors="coerce").max(axis=1).dropna()
    if len(max_probs) < 5:
        st.info("Not enough data.")
        return
    fig = px.histogram(max_probs * 100, nbins=20,
                       labels={"value": "Confidence (%)", "count": "Frequency"},
                       color_discrete_sequence=["#58a6ff"],
                       title="Prediction Confidence Distribution")
    mean_c = max_probs.mean() * 100
    fig.add_vline(x=mean_c, line_dash="dash", line_color="#ff7b72",
                  annotation_text=f"Mean: {mean_c:.1f}%", annotation_font_color="#ff7b72")
    fig.update_layout(showlegend=False, height=280,
                      paper_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9",
                      plot_bgcolor="rgba(255,255,255,0.05)")
    st.plotly_chart(fig, use_container_width=True)


def render_recent_detections(df: pd.DataFrame):
    st.markdown("#### 📋 Recent Detections (last 20)")
    if df.empty:
        st.info("No detections yet.")
        return
    col_map = {
        "timestamp": "Time", "Timestamp": "Time",
        "class_name": "Class", "Class_Name": "Class",
        "risk_level": "Risk", "Risk_Level": "Risk",
        "risk_name": "Status", "Risk_Name": "Status",
        "action": "Action", "Action": "Action",
    }
    display_cols = [c for c in col_map if c in df.columns]
    if not display_cols:
        display_cols = df.columns[:6].tolist()
    recent = df[display_cols].tail(20).iloc[::-1].rename(columns=col_map)
    st.dataframe(recent, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# SIDEBAR — global controls (persistent across all tabs)
# ---------------------------------------------------------------------------
st.sidebar.markdown('<p class="soc-header">🛡️ SOC Control Panel</p>', unsafe_allow_html=True)

# 1. Status LEDs
status = get_system_status()
led_data = "🟢" if status["data_flowing"] else ("🟡" if status["csv_exists"] else "🔴")
led_tf = "🟢" if status["tensorflow"] else "🔴"
st.sidebar.markdown(f"**Data Flow:** {led_data} &nbsp;|&nbsp; **Engine:** {led_tf}")
st.sidebar.markdown("---")

# Pre-load data for global filtering & banner
df_live_full = load_live_traffic()
df_logs_full = load_logs()

# 2. Threat Level Banner (Last 60s)
threat_info = RISK_LEVELS[1]
if not df_live_full.empty:
    ts_col = "timestamp" if "timestamp" in df_live_full.columns else "Timestamp"
    if ts_col in df_live_full.columns:
        df_live_full[ts_col] = pd.to_datetime(df_live_full[ts_col], errors="coerce")
        max_ts = df_live_full[ts_col].max()
        if not pd.isna(max_ts):
            last_60s = df_live_full[df_live_full[ts_col] >= (max_ts - pd.Timedelta(seconds=60))]
            if not last_60s.empty:
                risk_col = "risk_level" if "risk_level" in last_60s.columns else "Risk_Level"
                if risk_col in last_60s.columns:
                    max_r = int(last_60s[risk_col].max())
                else:
                    pred = "predicted_class" if "predicted_class" in last_60s.columns else "Predicted_Class"
                    max_c = int(last_60s[pred].max()) if pred in last_60s.columns else 0
                    max_r = 1 if max_c == 0 else (4 if max_c == 1 else 5)
                threat_info = RISK_LEVELS.get(max_r, RISK_LEVELS[1])

st.sidebar.markdown(f"""
<div style="background: {threat_info['color']}15; border: 1px solid {threat_info['color']}50; padding: 12px; border-radius: 8px; text-align: center; margin-bottom: 15px;">
    <div style="font-size: 0.8rem; color: #8b949e; margin-bottom: 5px;">THREAT LEVEL (LAST 60S)</div>
    <div style="font-size: 1.4rem; font-weight: bold; color: {threat_info['color']};">{threat_info['emoji']} {threat_info['name']}</div>
</div>
""", unsafe_allow_html=True)

# 3. Time Window
time_window = st.sidebar.selectbox("📅 Time Window", ["Last 5 min", "Last 1 hour", "Last 24h", "All Time"])
st.sidebar.markdown("---")

# 4. Live Mode
st.sidebar.subheader("🔄 Live Refresh")
live_mode = st.sidebar.toggle("⚡ Live Mode", key="live_mode")
refresh_interval = st.sidebar.slider("Interval (seconds)", 5, 60, key="refresh_interval", disabled=not live_mode)
st.sidebar.markdown("---")

# Filter dataframes for the tabs
def filter_dataframe(df: pd.DataFrame, window: str) -> pd.DataFrame:
    if df.empty or window == "All Time": return df
    c = "timestamp" if "timestamp" in df.columns else "Timestamp"
    if c not in df.columns: return df
    df[c] = pd.to_datetime(df[c], errors="coerce")
    m_ts = df[c].max()
    if pd.isna(m_ts): return df
    if window == "Last 5 min": cutoff = m_ts - pd.Timedelta(minutes=5)
    elif window == "Last 1 hour": cutoff = m_ts - pd.Timedelta(hours=1)
    elif window == "Last 24h": cutoff = m_ts - pd.Timedelta(hours=24)
    else: return df
    return df[df[c] >= cutoff].copy()

live_df = filter_dataframe(df_live_full, time_window)
logs_df = filter_dataframe(df_logs_full, time_window)

# 5. Model Status Badge
st.sidebar.subheader("🧠 Active AI Model")
os.makedirs(os.path.dirname(ACTIVE_MODEL_PATH), exist_ok=True)
try:
    if os.path.exists(ACTIVE_MODEL_PATH):
        with open(ACTIVE_MODEL_PATH) as f:
            current_model_file = f.read().strip()
        current_model = next((k for k, v in MODEL_MAPPING.items() if v == current_model_file), "BiLSTM")
    else:
        current_model = "BiLSTM"
        with open(ACTIVE_MODEL_PATH, "w") as f:
            f.write(MODEL_MAPPING["BiLSTM"])
except Exception:
    current_model = "BiLSTM"

selected_model = st.sidebar.selectbox("Select Model", list(MODEL_MAPPING.keys()), index=list(MODEL_MAPPING.keys()).index(current_model), key="model_selector", label_visibility="collapsed")
if selected_model != current_model:
    try:
        with open(ACTIVE_MODEL_PATH, "w") as f:
            f.write(MODEL_MAPPING[selected_model])
    except Exception as e:
        pass

st.sidebar.markdown(f"""
<div style="display: flex; justify-content: space-between; align-items: center; background: rgba(255,255,255,0.05); padding: 8px 12px; border-radius: 6px; margin-top: -10px;">
    <span style="font-weight: 600; font-size: 0.9rem;">{selected_model}</span>
    <span style="color: #00CC66; font-size: 0.8rem;">● Active</span>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# 6. IP Unblock
st.sidebar.subheader("🔓 IP Unblock")
ip_input = st.sidebar.text_input("IP Address", key="ip_unblock_input")
if st.sidebar.button("Unblock IP", key="unblock_btn"):
    if ip_input.strip():
        ok = unblock_ip(ip_input.strip())
        st.sidebar.success(f"✅ {ip_input} unblocked.") if ok else st.sidebar.warning("⚠️ Operation failed.")
    else:
        st.sidebar.warning("Enter a valid IP address.")

st.sidebar.markdown("---")
st.sidebar.caption(f"🕐 Last refresh: {datetime.now().strftime('%H:%M:%S')}")
st.sidebar.caption(f"🔁 Refresh #{count}")

# ---------------------------------------------------------------------------
# PAGE HEADER
# ---------------------------------------------------------------------------
st.markdown('<p class="soc-header">🛡️ AI Network IPS — Security Operations Center</p>', unsafe_allow_html=True)
st.markdown('<p class="soc-sub">3-Class BiLSTM NIDS &nbsp;|&nbsp; Benign · Volumetric · Semantic &nbsp;|&nbsp; Real-time Detection</p>', unsafe_allow_html=True)
st.markdown("---")

# ---------------------------------------------------------------------------
# FIVE-TAB LAYOUT
# ---------------------------------------------------------------------------
tab_monitor, tab_map, tab_logs, tab_xai, tab_admin = st.tabs([
    "🖥️ Live Monitor",
    "🗺️ Threat Map",
    "📋 Incident Logs",
    "🧠 XAI Explainer",
    "⚙️ Admin & Response",
])

# ── Tab 1: Live Monitor ────────────────────────────────────────────────────
with tab_monitor:
    render_system_status(status)
    st.markdown("---")
    render_metrics(live_df)
    st.markdown("---")

    col_gauge, col_dist = st.columns([1, 2])
    with col_gauge:
        render_risk_gauge(live_df)
    with col_dist:
        render_class_distribution(live_df)

    st.markdown("---")
    col_conf, col_time = st.columns(2)
    with col_conf:
        render_confidence_histogram(live_df)
    with col_time:
        render_time_series(live_df)

    st.markdown("---")
    render_recent_detections(live_df)

# ── Tab 2: Threat Map ──────────────────────────────────────────────────────
with tab_map:
    st.markdown("#### 🗺️ Threat Map")
    st.info("🚧 **Coming in Sprint 2** — Geographic threat heatmap with IP geolocation and attack-origin clustering.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Active Threat Sources", "—")
    col2.metric("Top Attack Country", "—")
    col3.metric("Blocked IPs (24h)", "—")

# ── Tab 3: Incident Logs ──────────────────────────────────────────────────
with tab_logs:
    st.markdown("#### 📋 Incident Logs")
    if logs_df.empty:
        st.info("No incident records found in the database.")
    else:
        # Quick summary metrics
        total_l   = len(logs_df)
        blocked_l = int((logs_df.get("action", pd.Series(dtype=str)) == "BLOCKED").sum())
        allowed_l = int((logs_df.get("action", pd.Series(dtype=str)) == "ALLOWED").sum())
        last_evt  = logs_df["timestamp"].max().strftime("%Y-%m-%d %H:%M:%S") if "timestamp" in logs_df.columns else "—"
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", f"{total_l:,}")
        c2.metric("Blocked", f"{blocked_l:,}")
        c3.metric("Allowed", f"{allowed_l:,}")
        c4.metric("Last Event", last_evt)
        st.markdown("---")
        st.dataframe(logs_df, use_container_width=True, hide_index=True)

# ── Tab 4: XAI Explainer ──────────────────────────────────────────────────
with tab_xai:
    st.markdown("#### 🧠 XAI Explainer")
    st.info("🚧 **Coming in Sprint 3** — SHAP / LIME feature-importance explanations for individual predictions, with attention heatmaps from the BiLSTM layer.")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Planned visualizations:**")
        st.markdown("- SHAP waterfall charts per prediction\n- Top-N feature importance bar chart\n- Attention weight heatmap (BiLSTM)\n- Counterfactual 'what-if' sliders")
    with col2:
        st.markdown("**Supported models:**")
        for m in MODEL_MAPPING:
            st.markdown(f"- {m}")

# ── Tab 5: Admin & Response ───────────────────────────────────────────────
with tab_admin:
    st.markdown("#### ⚙️ Admin & Response")
    st.info("🚧 **Coming in Sprint 4** — Automated response rules, block/allow policy management, alert thresholds, and audit trail.")

    st.markdown("##### 🔧 Quick Actions")
    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown("**Model Management**")
        st.caption(f"Active model: `{selected_model}`")
        st.caption(f"File: `{MODEL_MAPPING[selected_model]}`")
    with a2:
        st.markdown("**System Health**")
        status_q = get_system_status()
        st.caption(f"Data flowing: {'✅ Yes' if status_q['data_flowing'] else '❌ No'}")
        st.caption(f"CSV rows: {status_q['csv_rows']:,}")
    with a3:
        st.markdown("**Response Actions**")
        st.caption("IP blocking managed via firewall_manager")
        st.caption("Use sidebar → IP Unblock for immediate relief")
