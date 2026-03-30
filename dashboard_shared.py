"""Shared constants, CSS, and helper functions for all dashboard pages."""
import glob
import os
import sqlite3
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = os.getenv("DATA_DIR", "./data")
DB_PATH = os.path.join(DATA_DIR, "trades.db")
SIM_DB_PATH = os.path.join(DATA_DIR, "simulation.db")
HISTORY_DB_PATH = os.path.join(DATA_DIR, "simLastCharts.db")
LOG_DIR = os.path.join(DATA_DIR, "logs")
BOT_API_URL = os.getenv("BOT_API_URL", "http://localhost:8502")


# ── Cyber Design CSS ─────────────────────────────────────────────────────────
CYBER_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

    html, body, [class*="css"] {
        font-family: 'Share Tech Mono', monospace;
        background-color: #0a0a0a;
        color: #00ff41;
    }
    .stApp { background-color: #0a0a0a; }

    /* Header */
    h1, h2, h3 { color: #00ff41 !important; letter-spacing: 2px; }
    h1 { border-bottom: 1px solid #00ff41; padding-bottom: 8px; }

    /* Metrics */
    [data-testid="metric-container"] {
        background: #0d0d0d;
        border: 1px solid #00ff41;
        border-radius: 4px;
        padding: 12px;
    }
    [data-testid="metric-container"] label { color: #007a1f !important; font-size: 0.75rem; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #00ff41 !important;
        font-size: 1.6rem !important;
    }
    [data-testid="stMetricDelta"] { color: #39ff14 !important; }

    /* Tables */
    .stDataFrame { border: 1px solid #00ff41; }
    thead tr th {
        background-color: #001a00 !important;
        color: #00ff41 !important;
        font-family: 'Share Tech Mono', monospace !important;
    }
    tbody tr:hover { background-color: #001a00 !important; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #050505; border-right: 1px solid #00ff41; }

    /* Divider */
    hr { border-color: #00ff41; opacity: 0.3; }

    /* Log box */
    .log-box {
        background: #050505;
        border: 1px solid #007a1f;
        border-radius: 4px;
        padding: 12px;
        font-size: 0.75rem;
        color: #00cc33;
        height: 300px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-break: break-all;
    }
    .log-error   { color: #ff4444; }
    .log-warning { color: #ffaa00; }
    .log-info    { color: #00cc33; }

    /* Status badges */
    .badge-open     { color: #00ff41; }
    .badge-tp       { color: #39ff14; }
    .badge-sl       { color: #ff4444; }
    .badge-wait     { color: #ffaa00; }

    /* Chart */
    [data-testid="stArrowVegaLiteChart"] { border: 1px solid #007a1f; border-radius: 4px; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #0d0d0d;
        border: 1px solid #00ff41;
        border-radius: 4px 4px 0 0;
        color: #00ff41;
        font-family: 'Share Tech Mono', monospace;
        padding: 8px 24px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #001a00 !important;
        border-bottom: 2px solid #00ff41;
    }

    /* Bot status switch */
    .bot-status-on {
        background: linear-gradient(135deg, #001a00, #003300);
        border: 2px solid #00ff41;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        animation: pulse-green 2s infinite;
    }
    .bot-status-off {
        background: linear-gradient(135deg, #1a0000, #330000);
        border: 2px solid #ff4444;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    @keyframes pulse-green {
        0%, 100% { box-shadow: 0 0 5px #00ff41; }
        50% { box-shadow: 0 0 20px #00ff41; }
    }

    /* Signal cards */
    .signal-card {
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        border: 1px solid #00ff41;
        background: #0d0d0d;
    }
    .signal-buy  { border-color: #00ff41; background: rgba(0, 255, 65, 0.05); }
    .signal-sell { border-color: #ff4444; background: rgba(255, 68, 68, 0.05); }
    .signal-hold { border-color: #007a1f; background: #0d0d0d; }

    /* Trade position cards */
    .trade-card {
        background: #0d0d0d;
        border: 1px solid #00ff41;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
</style>
"""


def apply_css():
    """Inject the Cyber CSS into the current page."""
    st.markdown(CYBER_CSS, unsafe_allow_html=True)


# ── Plotly theme ──────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0a0a0a",
    plot_bgcolor="#0d0d0d",
    font=dict(color="#00ff41", family="Share Tech Mono"),
    xaxis=dict(gridcolor="#001a00", showgrid=True),
    yaxis=dict(gridcolor="#001a00", showgrid=True),
    margin=dict(l=10, r=10, t=10, b=10),
)


# ── DB helpers ────────────────────────────────────────────────────────────────
@st.cache_resource
def get_connection():
    if not Path(DB_PATH).exists():
        return None
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def get_sim_connection():
    if not Path(SIM_DB_PATH).exists():
        return None
    return sqlite3.connect(SIM_DB_PATH, check_same_thread=False)


def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    conn = get_connection()
    if conn is None:
        return pd.DataFrame()
    try:
        return pd.read_sql_query(sql, conn, params=params)
    except Exception:
        return pd.DataFrame()


def sim_query(sql: str, params: tuple = ()) -> pd.DataFrame:
    conn = get_sim_connection()
    if conn is None:
        return pd.DataFrame()
    try:
        return pd.read_sql_query(sql, conn, params=params)
    except Exception:
        return pd.DataFrame()


def load_trades() -> pd.DataFrame:
    df = query("SELECT * FROM trades ORDER BY timestamp DESC")
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["exit_timestamp"] = pd.to_datetime(df["exit_timestamp"], errors="coerce")
    return df


def load_snapshots() -> pd.DataFrame:
    df = query("SELECT * FROM account_snapshots ORDER BY timestamp ASC")
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_log_lines(n: int = 100) -> list[str]:
    log_files = sorted(glob.glob(os.path.join(LOG_DIR, "bot_*.log")), reverse=True)
    if not log_files:
        return ["Keine Log-Dateien gefunden."]
    lines = []
    with open(log_files[0], encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return lines[-n:]
