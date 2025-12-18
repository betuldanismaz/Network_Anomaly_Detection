import os
import sys
import time
import subprocess
import shutil

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(PARENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from utils.db_manager import fetch_logs
from utils.firewall_manager import unblock_ip

# Paths
LIVE_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "live_captured_traffic.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "rf_optimized_model.pkl")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")
THRESHOLD_PATH = os.path.join(PROJECT_ROOT, "models", "threshold.txt")

st.set_page_config(page_title="Network IPS Dashboard", layout="wide")
st.title("🛡️ Network IPS Dashboard")


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def load_logs() -> pd.DataFrame:
    logs_df = fetch_logs()
    if logs_df.empty:
        return logs_df
    logs_df["timestamp"] = pd.to_datetime(logs_df["timestamp"], errors="coerce")
    return logs_df.sort_values("timestamp", ascending=False)


def load_live_traffic() -> pd.DataFrame:
    """Load live captured traffic CSV for confidence scores."""
    if not os.path.exists(LIVE_CSV_PATH):
        return pd.DataFrame()
    try:
        df = pd.read_csv(LIVE_CSV_PATH)
        return df
    except Exception:
        return pd.DataFrame()


def get_system_status() -> dict:
    """Check system component statuses."""
    status = {
        "model": os.path.exists(MODEL_PATH),
        "scaler": os.path.exists(SCALER_PATH),
        "threshold": os.path.exists(THRESHOLD_PATH),
        "cicflowmeter": shutil.which("cicflowmeter") is not None,
        "scapy": False,
    }
    # Check if scapy is importable
    try:
        import scapy
        status["scapy"] = True
    except ImportError:
        status["scapy"] = False
    
    # Load threshold value
    if status["threshold"]:
        try:
            with open(THRESHOLD_PATH, "r") as f:
                status["threshold_value"] = float(f.read().strip())
        except:
            status["threshold_value"] = 0.5
    else:
        status["threshold_value"] = 0.5
    
    return status


def detect_attack_type(details: str) -> str:
    """Classify attack type from details field."""
    if pd.isna(details) or not details:
        return "Unknown"
    details_lower = str(details).lower()
    if "ddos" in details_lower or "flood" in details_lower:
        return "DDoS"
    elif "port" in details_lower or "scan" in details_lower:
        return "Port Scan"
    elif "web" in details_lower or "http" in details_lower or "sql" in details_lower:
        return "Web Attack"
    elif "brute" in details_lower:
        return "Brute Force"
    elif "infilter" in details_lower:
        return "Infiltration"
    elif "whitelist" in details_lower:
        return "Whitelisted"
    elif "attack" in details_lower:
        return "Generic Attack"
    else:
        return "Other"


# ---------------------------------------------------------------------------
# RENDER FUNCTIONS
# ---------------------------------------------------------------------------

def render_system_status():
    """Render system component status indicators."""
    st.subheader("⚙️ Sistem Durumu")
    status = get_system_status()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if status["model"]:
            st.success("✅ Model")
        else:
            st.error("❌ Model")
    
    with col2:
        if status["scaler"]:
            st.success("✅ Scaler")
        else:
            st.error("❌ Scaler")
    
    with col3:
        if status["scapy"]:
            st.success("✅ Scapy")
        else:
            st.error("❌ Scapy")
    
    with col4:
        if status["cicflowmeter"]:
            st.success("✅ CICFlowMeter")
        else:
            st.warning("⚠️ CICFlowMeter")
    
    with col5:
        st.info(f"🎯 threshold: {status['threshold_value']:.4f}")


def render_kpis(logs_df: pd.DataFrame):
    total = len(logs_df)
    blocked = int((logs_df["action"] == "BLOCKED").sum())
    allowed = int((logs_df["action"] == "ALLOWED").sum())
    last_event = "-" if logs_df.empty else logs_df["timestamp"].max().strftime("%Y-%m-%d %H:%M:%S")

    col1, col2, col3 = st.columns(3)
    col1.metric("Toplam Kayıt", total)
    col2.metric("Engellenen IP", blocked)
    col3.metric("Son Olay", last_event)


def render_charts(logs_df: pd.DataFrame):
    col_line, col_pie = st.columns(2)

    if logs_df.empty:
        col_line.info("Grafik göstermek için yeterli veri yok.")
        col_pie.info("Grafik göstermek için yeterli veri yok.")
        return

    line_df = (
        logs_df.set_index("timestamp")
        .resample("1min")
        .size()
        .reset_index(name="attacks")
    )
    if line_df.empty:
        col_line.info("Saldırı frekansı için veri yok.")
    else:
        line_fig = px.line(line_df, x="timestamp", y="attacks", title="Saldırı Frekansı (1 dk)")
        col_line.plotly_chart(line_fig, use_container_width=True)

    pie_df = logs_df["action"].value_counts().reset_index()
    pie_df.columns = ["action", "count"]
    if pie_df.empty:
        col_pie.info("Eylem dağılımı için veri yok.")
    else:
        pie_fig = px.pie(pie_df, values="count", names="action", title="Blocked vs Allowed", hole=0.3)
        col_pie.plotly_chart(pie_fig, use_container_width=True)


def render_confidence_scores():
    """Render real-time confidence score metrics from live traffic."""
    st.subheader("📊 Gerçek Zamanlı Güven Skorları")
    
    live_df = load_live_traffic()
    
    if live_df.empty or "Confidence_Score" not in live_df.columns:
        st.info("Henüz canlı trafik verisi yok. live_bridge.py çalıştırıldığında veriler burada görünecek.")
        return
    
    # Filter valid confidence scores
    confidence_scores = pd.to_numeric(live_df["Confidence_Score"], errors="coerce").dropna()
    
    if len(confidence_scores) == 0:
        st.info("Güven skoru verisi bulunamadı.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_score = confidence_scores.mean() * 100
        st.metric("Ortalama Güven", f"{avg_score:.2f}%")
    
    with col2:
        max_score = confidence_scores.max() * 100
        st.metric("Maksimum Güven", f"{max_score:.2f}%")
    
    with col3:
        min_score = confidence_scores.min() * 100
        st.metric("Minimum Güven", f"{min_score:.2f}%")
    
    with col4:
        total_flows = len(live_df)
        st.metric("Toplam Akış", f"{total_flows:,}")
    
    # Confidence score distribution histogram
    if len(confidence_scores) > 10:
        fig = px.histogram(
            confidence_scores * 100, 
            nbins=20, 
            title="Güven Skoru Dağılımı (%)",
            labels={"value": "Güven Skoru (%)", "count": "Frekans"}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def render_top_attackers(logs_df: pd.DataFrame):
    """Render top 10 attacker IPs."""
    st.subheader("🎯 Top 10 Saldırgan IP")
    
    if logs_df.empty:
        st.info("Henüz saldırı kaydı yok.")
        return
    
    top_ips = logs_df["src_ip"].value_counts().head(10).reset_index()
    top_ips.columns = ["IP Adresi", "Tespit Sayısı"]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(top_ips, use_container_width=True, hide_index=True)
    
    with col2:
        if len(top_ips) > 0:
            fig = px.bar(
                top_ips, 
                x="IP Adresi", 
                y="Tespit Sayısı", 
                title="En Çok Tespit Edilen IP'ler",
                color="Tespit Sayısı",
                color_continuous_scale="Reds"
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)


def render_attack_types(logs_df: pd.DataFrame):
    """Render attack type distribution pie chart."""
    st.subheader("🔥 Saldırı Türü Dağılımı")
    
    if logs_df.empty:
        st.info("Henüz saldırı kaydı yok.")
        return
    
    # Classify attack types
    logs_df["attack_type"] = logs_df["details"].apply(detect_attack_type)
    
    attack_counts = logs_df["attack_type"].value_counts().reset_index()
    attack_counts.columns = ["Saldırı Türü", "Sayı"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            attack_counts, 
            values="Sayı", 
            names="Saldırı Türü", 
            title="Saldırı Türleri",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(attack_counts, use_container_width=True, hide_index=True)


def render_live_traffic_counter():
    """Render live traffic statistics from CSV."""
    st.subheader("📡 Canlı Trafik Sayacı")
    
    live_df = load_live_traffic()
    
    if live_df.empty:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Toplam Akış", "0")
        col2.metric("Saldırı", "0")
        col3.metric("Normal", "0")
        col4.metric("Dosya Boyutu", "0 KB")
        st.info("Canlı trafik verisi bekleniyor... live_bridge.py çalıştırın.")
        return
    
    # Calculate metrics
    total_flows = len(live_df)
    
    if "Predicted_Label" in live_df.columns:
        attacks = int((live_df["Predicted_Label"] == 1).sum())
        normal = int((live_df["Predicted_Label"] == 0).sum())
    else:
        attacks = 0
        normal = total_flows
    
    # File size
    try:
        file_size_kb = os.path.getsize(LIVE_CSV_PATH) / 1024
    except:
        file_size_kb = 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Toplam Akış", f"{total_flows:,}")
    col2.metric("🚨 Saldırı", f"{attacks:,}", delta=f"{attacks/total_flows*100:.1f}%" if total_flows > 0 else "0%")
    col3.metric("✅ Normal", f"{normal:,}")
    col4.metric("Dosya Boyutu", f"{file_size_kb:.1f} KB")
    
    # Attack rate visualization - Pie Chart + Progress Bar
    if total_flows > 0:
        attack_rate = attacks / total_flows * 100
        normal_rate = 100 - attack_rate
        
        col_chart, col_progress = st.columns([2, 1])
        
        with col_chart:
            # Donut chart for attack vs normal
            fig = px.pie(
                values=[attacks, normal],
                names=["🚨 Saldırı", "✅ Normal"],
                title="Trafik Dağılımı",
                hole=0.6,
                color_discrete_sequence=["#FF4B4B", "#00CC66"]
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                height=300,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            # Add center text
            fig.add_annotation(
                text=f"<b>{attack_rate:.1f}%</b><br>Saldırı",
                x=0.5, y=0.5,
                font_size=16,
                showarrow=False,
                font_color="#FF4B4B" if attack_rate > 20 else "#00CC66"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col_progress:
            st.markdown("### 📊 Saldırı Oranı")
            
            # Color based on severity
            if attack_rate > 50:
                color = "🔴"
                status = "KRİTİK"
                bar_color = "#FF4B4B"
            elif attack_rate > 20:
                color = "🟠"
                status = "YÜKSEK"
                bar_color = "#FFA500"
            elif attack_rate > 5:
                color = "🟡"
                status = "ORTA"
                bar_color = "#FFD700"
            else:
                color = "🟢"
                status = "DÜŞÜK"
                bar_color = "#00CC66"
            
            st.markdown(f"## {color} {attack_rate:.1f}%")
            st.markdown(f"**Durum:** {status}")
            
            # Progress bar using streamlit
            st.progress(min(attack_rate / 100, 1.0))
            
            st.markdown(f"""
            ---
            - **Saldırı:** {attacks:,} akış
            - **Normal:** {normal:,} akış
            - **Toplam:** {total_flows:,} akış
            """)


# ---------------------------------------------------------------------------
# MAIN DASHBOARD LAYOUT
# ---------------------------------------------------------------------------

logs = load_logs()

# System Status (Top)
render_system_status()
st.divider()

# KPIs
render_kpis(logs)
st.divider()

# Live Traffic Counter & Confidence Scores
col_left, col_right = st.columns(2)
with col_left:
    render_live_traffic_counter()
with col_right:
    render_confidence_scores()
st.divider()

# Charts Row 1
render_charts(logs)
st.divider()

# Charts Row 2: Top Attackers & Attack Types
col_attackers, col_types = st.columns(2)
with col_attackers:
    render_top_attackers(logs)
with col_types:
    render_attack_types(logs)
st.divider()

# Sidebar Controls
st.sidebar.header("Kontroller")
auto_refresh = st.sidebar.checkbox("Otomatik Yenile", value=True, key="auto_refresh_checkbox")
refresh_interval = st.sidebar.slider("Yenileme Aralığı (sn)", min_value=5, max_value=60, value=15, key="refresh_interval_slider")

st.subheader("📋 Detaylı Loglar")
if logs.empty:
    st.info("Henüz kayıt bulunamadı.")
else:
    st.dataframe(logs, use_container_width=True)

ip_to_unblock = st.sidebar.text_input("Engeli Kaldırılacak IP")
if st.sidebar.button("Unblock IP"):
    if ip_to_unblock.strip():
        success = unblock_ip(ip_to_unblock.strip())
        if success:
            st.sidebar.success(f"{ip_to_unblock} engeli kaldırıldı.")
        else:
            st.sidebar.warning(f"{ip_to_unblock} için kural bulunamadı veya işlem başarısız.")
    else:
        st.sidebar.warning("Lütfen geçerli bir IP girin.")

if auto_refresh:
    st.sidebar.caption(f"Sayfa {refresh_interval} sn'de bir yenileniyor.")
    time.sleep(refresh_interval)
    st.rerun()
