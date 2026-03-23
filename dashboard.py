"""Streamlit Dashboard – Commodities Trading Bot."""
import glob
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.getenv("DATA_DIR", "./data")
DB_PATH = os.path.join(DATA_DIR, "trades.db")
LOG_DIR = os.path.join(DATA_DIR, "logs")

# ── Page setup ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trading Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Cyber Design ───────────────────────────────────────────────────────────────
st.markdown("""
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
</style>
""", unsafe_allow_html=True)


# ── DB helpers ─────────────────────────────────────────────────────────────────
@st.cache_resource
def get_connection():
    if not Path(DB_PATH).exists():
        return None
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    conn = get_connection()
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


def load_analyses() -> pd.DataFrame:
    df = query("SELECT * FROM daily_analyses ORDER BY timestamp DESC LIMIT 50")
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"])
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


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# ◈ COMMODITIES TRADING BOT")
st.markdown(f"*{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
st.markdown("---")

# ── Auto-refresh ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙ EINSTELLUNGEN")
    refresh = st.slider("Auto-Refresh (Sek)", 0, 300, 60, 10)
    if refresh > 0:
        st.markdown(f"*Aktualisierung alle {refresh}s*")
    log_lines = st.slider("Log-Zeilen", 20, 500, 100, 20)
    st.markdown("---")
    if st.button("⟳ Jetzt aktualisieren"):
        st.cache_resource.clear()
        st.rerun()

if refresh > 0:
    import time
    st.empty()

# Load data
trades = load_trades()
snapshots = load_snapshots()
analyses = load_analyses()

# ── KPI Row ────────────────────────────────────────────────────────────────────
open_trades = trades[trades["status"] == "OPEN"] if not trades.empty else pd.DataFrame()
closed_trades = trades[trades["status"].isin(["TAKE_PROFIT", "STOPPED_OUT", "CLOSED"])] if not trades.empty else pd.DataFrame()

latest_balance = snapshots["balance"].iloc[-1] if not snapshots.empty else 0.0
prev_balance = snapshots["balance"].iloc[-2] if len(snapshots) > 1 else latest_balance

total_pl = closed_trades["profit_loss"].sum() if not closed_trades.empty else 0.0
win_trades = closed_trades[closed_trades["profit_loss"] > 0] if not closed_trades.empty else pd.DataFrame()
win_rate = len(win_trades) / len(closed_trades) * 100 if not closed_trades.empty else 0.0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("KONTOSTAND", f"€{latest_balance:.2f}", f"{latest_balance - prev_balance:+.2f}")
col2.metric("OFFENE POSITIONEN", len(open_trades))
col3.metric("GESAMT P/L", f"€{total_pl:+.2f}")
col4.metric("WIN RATE", f"{win_rate:.0f}%", f"{len(win_trades)}/{len(closed_trades)} Trades")
col5.metric("ANALYSEN GESAMT", len(analyses))

st.markdown("---")

# ── Charts Row ─────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("### KONTOSTAND VERLAUF")
    if not snapshots.empty:
        chart_data = snapshots.set_index("timestamp")[["balance"]]
        st.line_chart(chart_data, color="#00ff41")
    else:
        st.info("Noch keine Snapshot-Daten.")

with col_right:
    st.markdown("### P/L PRO TRADE")
    if not closed_trades.empty:
        pl_data = closed_trades[["timestamp", "asset", "profit_loss"]].copy()
        pl_data = pl_data.sort_values("timestamp").tail(20)
        pl_data["color"] = pl_data["profit_loss"].apply(lambda x: "#00ff41" if x >= 0 else "#ff4444")
        st.bar_chart(pl_data.set_index("timestamp")[["profit_loss"]], color="#00ff41")
    else:
        st.info("Noch keine geschlossenen Trades.")

st.markdown("---")

# ── Open Positions ─────────────────────────────────────────────────────────────
st.markdown("### ◈ OFFENE POSITIONEN")
if not open_trades.empty:
    display_open = open_trades[[
        "timestamp", "asset", "direction", "entry_price",
        "stop_loss", "take_profit", "position_size", "confidence", "deal_id"
    ]].copy()

    # Duration since position opened
    now_utc = pd.Timestamp.now(tz="UTC")
    ts_aware = display_open["timestamp"].dt.tz_localize("UTC") if display_open["timestamp"].dt.tz is None else display_open["timestamp"].dt.tz_convert("UTC")
    delta = now_utc - ts_aware
    display_open["offen_seit"] = delta.apply(
        lambda d: f"{int(d.total_seconds()//3600)}h {int((d.total_seconds()%3600)//60)}m"
    )

    # Distance to SL / TP in %
    display_open["zu_sl_pct"] = (
        (display_open["entry_price"] - display_open["stop_loss"]).abs()
        / display_open["entry_price"] * 100
    ).apply(lambda x: f"{x:.2f}%")
    display_open["zu_tp_pct"] = (
        (display_open["take_profit"] - display_open["entry_price"]).abs()
        / display_open["entry_price"] * 100
    ).apply(lambda x: f"{x:.2f}%")

    display_open["timestamp"] = display_open["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    display_open = display_open[[
        "timestamp", "offen_seit", "asset", "direction", "entry_price",
        "zu_sl_pct", "zu_tp_pct", "stop_loss", "take_profit",
        "position_size", "confidence", "deal_id"
    ]]
    display_open.columns = ["ZEIT", "DAUER", "ASSET", "DIR", "ENTRY", "▼SL %", "▲TP %", "SL", "TP", "SIZE", "CONF", "DEAL ID"]
    st.dataframe(display_open, use_container_width=True, hide_index=True)
    st.caption("Aktueller Kurs nicht in DB verfügbar – ▼SL % / ▲TP % zeigen Abstand vom Entry.")
else:
    st.markdown("*Keine offenen Positionen.*")

st.markdown("---")

# ── Trade History ──────────────────────────────────────────────────────────────
st.markdown("### ◈ TRADE HISTORY")

col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    assets = ["Alle"] + (sorted(trades["asset"].unique().tolist()) if not trades.empty else [])
    asset_filter = st.selectbox("Asset", assets)
with col_f2:
    statuses = ["Alle", "OPEN", "TAKE_PROFIT", "STOPPED_OUT", "CLOSED"]
    status_filter = st.selectbox("Status", statuses)
with col_f3:
    days = st.selectbox("Zeitraum", ["7 Tage", "30 Tage", "90 Tage", "Alle"])

if not trades.empty:
    filtered = trades.copy()
    if asset_filter != "Alle":
        filtered = filtered[filtered["asset"] == asset_filter]
    if status_filter != "Alle":
        filtered = filtered[filtered["status"] == status_filter]
    if days != "Alle":
        d = int(days.split()[0])
        cutoff = pd.Timestamp.now(tz="UTC") - timedelta(days=d)
        ts_col = filtered["timestamp"].dt.tz_localize("UTC") if filtered["timestamp"].dt.tz is None else filtered["timestamp"].dt.tz_convert("UTC")
        filtered = filtered[ts_col >= cutoff]

    display = filtered[[
        "timestamp", "asset", "direction", "entry_price", "exit_price",
        "stop_loss", "take_profit", "position_size", "profit_loss", "profit_loss_pct", "status", "confidence"
    ]].copy()
    display["timestamp"] = display["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    display["profit_loss"] = display["profit_loss"].apply(
        lambda x: f"+{x:.2f}" if pd.notna(x) and x >= 0 else (f"{x:.2f}" if pd.notna(x) else "–")
    )
    display["profit_loss_pct"] = display["profit_loss_pct"].apply(
        lambda x: f"{x:+.2f}%" if pd.notna(x) else "–"
    )
    display.columns = ["ZEIT", "ASSET", "DIR", "ENTRY", "EXIT", "SL", "TP", "SIZE", "P/L €", "P/L %", "STATUS", "CONF"]
    st.dataframe(display, use_container_width=True, hide_index=True)
else:
    st.info("Noch keine Trades in der Datenbank.")

st.markdown("---")

# ── Analyses ───────────────────────────────────────────────────────────────────
st.markdown("### ◈ CLAUDE ANALYSEN")
if not analyses.empty:
    for _, row in analyses.head(5).iterrows():
        rec = row["recommendation"]
        color = "#00ff41" if rec == "TRADE" else "#ffaa00"
        ts = row["timestamp"].strftime("%Y-%m-%d %H:%M")
        with st.expander(f"{ts}  |  {rec}  |  {row['tokens_used']} tokens  |  ${row['cost_usd']:.4f}"):
            st.markdown(f"**Zusammenfassung:** {row['market_summary']}")
            if row["raw_analysis"]:
                try:
                    raw = json.loads(row["raw_analysis"])
                    opp = raw.get("best_opportunity", {})
                    if opp:
                        st.markdown(f"**Bestes Setup:** {opp.get('asset')} {opp.get('direction')} | Confidence: {opp.get('confidence')}/10 | RR: {opp.get('risk_reward_ratio', 0):.2f}")
                        st.markdown(f"**Begründung:** {opp.get('reasoning', '')}")
                    if raw.get("wait_reason"):
                        st.markdown(f"**WAIT Grund:** {raw['wait_reason']}")
                except Exception:
                    pass
else:
    st.info("Noch keine Analysen gespeichert.")

st.markdown("---")

# ── Log Viewer ─────────────────────────────────────────────────────────────────
st.markdown("### ◈ LOG")
lines = load_log_lines(log_lines)

colored = []
for line in lines:
    line = line.rstrip()
    if "[ERROR]" in line or "ERROR" in line:
        colored.append(f'<span class="log-error">{line}</span>')
    elif "[WARNING]" in line or "WARNING" in line:
        colored.append(f'<span class="log-warning">{line}</span>')
    else:
        colored.append(f'<span class="log-info">{line}</span>')

log_html = "<br>".join(colored)
st.markdown(f'<div class="log-box">{log_html}</div>', unsafe_allow_html=True)

# ── Auto-refresh script ────────────────────────────────────────────────────────
if refresh > 0:
    st.markdown(f"""
    <script>
        setTimeout(function() {{ window.location.reload(); }}, {refresh * 1000});
    </script>
    """, unsafe_allow_html=True)
