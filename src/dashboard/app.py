import os
import sys
import time

import pandas as pd
import plotly.express as px
import streamlit as st

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from utils.db_manager import fetch_logs
from utils.firewall_manager import unblock_ip

st.set_page_config(page_title="AI Network IPS Dashboard", layout="wide")
st.title("🛡️ AI Network IPS Dashboard")


def load_logs() -> pd.DataFrame:
    logs_df = fetch_logs()
    if logs_df.empty:
        return logs_df
    logs_df["timestamp"] = pd.to_datetime(logs_df["timestamp"], errors="coerce")
    return logs_df.sort_values("timestamp", ascending=False)


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


logs = load_logs()
render_kpis(logs)
render_charts(logs)

st.subheader("Detaylı Loglar")
if logs.empty:
    st.info("Henüz kayıt bulunamadı.")
else:
    def _highlight_blocked(row: pd.Series):
        color = "background-color: #ffe6e6" if row.get("action") == "BLOCKED" else ""
        return [color] * len(row)

    styled_logs = logs.style.apply(_highlight_blocked, axis=1)

    st.dataframe(
        styled_logs,
        use_container_width=True,
        height=360,
        hide_index=True,
        column_config={
            "details": st.column_config.TextColumn(
                "🧠 Reasoning",
                help="SHAP tabanlı açıklama",
                width="large",
            )
        },
    )

st.sidebar.header("Kontroller")
auto_refresh = st.sidebar.checkbox("Otomatik Yenile", value=True)
refresh_interval = st.sidebar.slider("Yenileme Aralığı (sn)", min_value=5, max_value=60, value=15)

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
