"""Streamlit Dashboard – Commodities Trading Bot."""
import glob
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.getenv("DATA_DIR", "./data")
DB_PATH = os.path.join(DATA_DIR, "trades.db")
LOG_DIR = os.path.join(DATA_DIR, "logs")
BOT_API_URL = os.getenv("BOT_API_URL", "http://localhost:8502")

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

    # ── Bot Settings (from .env) ──────────────────────────────────────
    st.markdown("---")
    st.markdown("### ◈ BOT EINSTELLUNGEN")

    # Load current settings from API
    _settings_data = None
    try:
        _resp = httpx.get(f"{BOT_API_URL}/api/settings", timeout=5)
        if _resp.status_code == 200:
            _settings_data = _resp.json().get("settings", [])
    except Exception:
        pass

    if _settings_data is None:
        st.warning("Bot-API nicht erreichbar – Settings nicht verfuegbar.")
    else:
        # Group settings
        _groups: dict[str, list] = {}
        for _s in _settings_data:
            _groups.setdefault(_s["group"], []).append(_s)

        _changed: dict[str, object] = {}

        for _group_name, _group_settings in _groups.items():
            st.markdown(f"**{_group_name}**")
            for _s in _group_settings:
                _key = _s["key"]
                _label = _s["label"]
                _stype = _s["type"]
                _val = _s["value"]
                _desc = _s.get("description", "")
                _widget_key = f"setting_{_key}"

                if _stype == "bool":
                    _new = st.toggle(_label, value=bool(_val), key=_widget_key, help=_desc)
                    if _new != bool(_val):
                        _changed[_key] = _new
                elif _stype == "int":
                    _new = st.number_input(
                        _label, value=int(_val) if _val is not None else 0,
                        min_value=_s.get("min", 0), max_value=_s.get("max", 999),
                        step=1, key=_widget_key, help=_desc,
                    )
                    if int(_new) != int(_val or 0):
                        _changed[_key] = int(_new)
                elif _stype == "float":
                    _new = st.number_input(
                        _label, value=float(_val) if _val is not None else 0.0,
                        min_value=float(_s.get("min", 0.0)),
                        max_value=float(_s.get("max", 999.0)),
                        step=float(_s.get("step", 0.1)),
                        format="%.2f", key=_widget_key, help=_desc,
                    )
                    if abs(float(_new) - float(_val or 0.0)) > 0.001:
                        _changed[_key] = float(_new)
                elif _stype == "select":
                    _options = _s.get("options", [])
                    _idx = _options.index(_val) if _val in _options else 0
                    _new = st.selectbox(_label, _options, index=_idx, key=_widget_key, help=_desc)
                    if _new != _val:
                        _changed[_key] = _new
                else:  # str
                    _new = st.text_input(_label, value=str(_val or ""), key=_widget_key, help=_desc)
                    if _new != str(_val or ""):
                        _changed[_key] = _new

        # Save button
        st.markdown("---")
        if _changed:
            st.info(f"{len(_changed)} Aenderung(en) ausstehend")
        if st.button("SPEICHERN", use_container_width=True, disabled=not _changed):
            try:
                _save_resp = httpx.post(
                    f"{BOT_API_URL}/api/settings",
                    json={"updates": _changed},
                    timeout=10,
                )
                if _save_resp.status_code == 200:
                    _result = _save_resp.json()
                    st.success(f"{_result.get('changed', 0)} Einstellung(en) gespeichert!")
                    st.rerun()
                else:
                    _err = _save_resp.json() if _save_resp.headers.get("content-type", "").startswith("application/json") else {}
                    st.error(f"Fehler: {_err.get('detail', _save_resp.text[:200])}")
            except Exception as e:
                st.error(f"Speichern fehlgeschlagen: {e}")

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

# ── Action Buttons ────────────────────────────────────────────────────────
st.markdown("### ◈ AKTIONEN")
col_b1, col_b2, col_b3, col_b4 = st.columns(4)

with col_b1:
    if st.button("ANALYSE STARTEN", use_container_width=True):
        with st.spinner("Claude analysiert den Markt..."):
            try:
                resp = httpx.post(f"{BOT_API_URL}/api/analyze", timeout=120)
                if resp.status_code == 200:
                    result = resp.json()
                    opp = result.get("best_opportunity", {})
                    st.success(
                        f"**{result.get('recommendation')}** | "
                        f"{opp.get('asset')} {opp.get('direction')} "
                        f"(Confidence: {opp.get('confidence')}/10, "
                        f"RR: {opp.get('risk_reward_ratio', 0):.2f})"
                    )
                    st.info(result.get("market_summary", ""))
                else:
                    st.error(f"Fehler: {resp.status_code} – {resp.text[:200]}")
            except Exception as e:
                st.error(f"API nicht erreichbar: {e}")

with col_b2:
    if st.button("TAGESBILANZ", use_container_width=True):
        with st.spinner("Erstelle Tagesbilanz..."):
            try:
                resp = httpx.post(f"{BOT_API_URL}/api/daily-summary", timeout=120)
                if resp.status_code == 200:
                    result = resp.json()
                    st.success(
                        f"Trades heute: {result.get('trades_count', 0)} | "
                        f"P/L: {result.get('total_pl', 0):+.2f} EUR | "
                        f"Balance: {result.get('balance', 0):.2f} EUR"
                    )
                    cs = result.get("claude_summary", {})
                    if cs.get("summary"):
                        st.info(cs["summary"])
                    for rec in cs.get("recommendations", []):
                        st.markdown(f"- {rec}")
                else:
                    st.error(f"Fehler: {resp.status_code}")
            except Exception as e:
                st.error(f"API nicht erreichbar: {e}")

with col_b3:
    if st.button("WOCHENREPORT", use_container_width=True):
        with st.spinner("Erstelle Wochenreport..."):
            try:
                resp = httpx.post(f"{BOT_API_URL}/api/weekly-report", timeout=120)
                if resp.status_code == 200:
                    result = resp.json()
                    st.success(
                        f"Woche: {result.get('trades_count', 0)} Trades | "
                        f"P/L: {result.get('total_pl', 0):+.2f} EUR"
                    )
                    cs = result.get("claude_summary", {})
                    if cs.get("summary"):
                        st.info(cs["summary"])
                    if cs.get("highlights"):
                        st.markdown("**Highlights:** " + " | ".join(cs["highlights"]))
                    if cs.get("issues"):
                        st.markdown("**Probleme:** " + " | ".join(cs["issues"]))
                    for rec in cs.get("recommendations", []):
                        st.markdown(f"- {rec}")
                else:
                    st.error(f"Fehler: {resp.status_code}")
            except Exception as e:
                st.error(f"API nicht erreichbar: {e}")

with col_b4:
    if st.button("BOT STATUS", use_container_width=True):
        try:
            resp = httpx.get(f"{BOT_API_URL}/api/status", timeout=10)
            if resp.status_code == 200:
                status = resp.json()
                st.markdown(
                    f"**Balance:** {status.get('balance', 0):.2f} EUR | "
                    f"**Offen:** {status.get('open_trades', 0)} | "
                    f"**Heute:** {status.get('trades_today', 0)} Trades, "
                    f"{status.get('analyses_today', 0)} Analysen"
                )
                perf = status.get("performance", {})
                if perf.get("total", 0) > 0:
                    st.markdown(
                        f"Win-Rate: {perf.get('win_rate', 0):.0f}% | "
                        f"Verlustserie: {perf.get('current_loss_streak', 0)}"
                    )
            else:
                st.error(f"Fehler: {resp.status_code}")
        except Exception as e:
            st.error(f"API nicht erreichbar: {e}")

st.markdown("---")

# ── Pending Rechecks ──────────────────────────────────────────────────────
rechecks_df = query(
    "SELECT * FROM pending_rechecks WHERE status = 'PENDING' ORDER BY recheck_at ASC"
)
if not rechecks_df.empty:
    st.markdown("### ◈ PENDING RECHECKS")
    for _, rc in rechecks_df.iterrows():
        recheck_at = str(rc["recheck_at"])[:16] if rc["recheck_at"] else "?"
        st.markdown(
            f'**{rc["asset"]}** {rc["direction"]} | '
            f'Trigger: *{rc["trigger_condition"]}* | '
            f'Confidence: {rc["current_confidence"]} | '
            f'Check #{int(rc["recheck_count"])+1}/{int(rc["max_rechecks"])} | '
            f'Naechster: {recheck_at}'
        )
        col_rc1, col_rc2, _ = st.columns([1, 1, 4])
        with col_rc1:
            if st.button("CANCEL", key=f"cancel_rc_{int(rc['id'])}"):
                try:
                    resp = httpx.post(f"{BOT_API_URL}/api/recheck/{int(rc['id'])}/cancel", timeout=10)
                    if resp.status_code == 200:
                        st.success("Recheck abgebrochen")
                        st.rerun()
                    else:
                        st.error(f"Fehler: {resp.status_code}")
                except Exception as e:
                    st.error(f"API: {e}")
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

# ── Trade Review ──────────────────────────────────────────────────────────
st.markdown("### ◈ TRADE REVIEW")
if not closed_trades.empty:
    col_rev1, col_rev2 = st.columns([3, 1])
    with col_rev1:
        review_options = []
        for _, row in closed_trades.head(20).iterrows():
            pl = row["profit_loss"] if pd.notna(row["profit_loss"]) else 0
            label = f"#{int(row['id'])} {row['asset']} {row['direction']} | P/L: {pl:+.2f} | {row['status']}"
            review_options.append((int(row["id"]), label))
        if review_options:
            selected_id = st.selectbox(
                "Trade auswaehlen",
                [x[0] for x in review_options],
                format_func=lambda x: next(l for i, l in review_options if i == x),
            )
    with col_rev2:
        if review_options and st.button("ANALYSIEREN", use_container_width=True):
            with st.spinner("Claude analysiert den Trade..."):
                try:
                    resp = httpx.post(
                        f"{BOT_API_URL}/api/trade-review/{selected_id}", timeout=120,
                    )
                    if resp.status_code == 200:
                        result = resp.json()
                        rev = result.get("review", {})
                        st.success(f"Review fuer Trade #{selected_id} gespeichert")
                        st.markdown(f"**Einstieg:** {rev.get('entry_quality', '?')} – {rev.get('entry_quality_explanation', '')}")
                        st.markdown(f"**Stop-Loss:** {rev.get('sl_quality', '?')} – {rev.get('sl_quality_explanation', '')}")
                        st.markdown(f"**Marktlage:** {rev.get('market_condition', '?')}")
                        st.markdown(f"**Lesson Learned:** {rev.get('lesson_learned', '?')}")
                        for sug in rev.get("improvement_suggestions", []):
                            st.markdown(f"- {sug}")
                    else:
                        st.error(f"Fehler: {resp.status_code} – {resp.text[:200]}")
                except Exception as e:
                    st.error(f"API nicht erreichbar: {e}")
else:
    st.markdown("*Keine geschlossenen Trades zum Analysieren.*")

st.markdown("---")

# ── Lernhistorie ──────────────────────────────────────────────────────────
st.markdown("### ◈ WAS HAT DER BOT GELERNT?")
reviews_df = query(
    """SELECT r.review_timestamp, r.entry_quality, r.sl_quality,
              r.market_condition, r.lesson_learned,
              t.asset, t.direction, t.status, t.profit_loss
    FROM trade_reviews r JOIN trades t ON r.trade_id = t.id
    ORDER BY r.review_timestamp DESC LIMIT 20"""
)
if not reviews_df.empty:
    for _, r in reviews_df.iterrows():
        pl = r["profit_loss"] if pd.notna(r["profit_loss"]) else 0
        color = "#00ff41" if pl >= 0 else "#ff4444"
        st.markdown(
            f'<span style="color:{color}">{r["asset"]} {r["direction"]} '
            f'({r["status"]}, P/L: {pl:+.2f})</span><br>'
            f'Entry: {r["entry_quality"]} | SL: {r["sl_quality"]} | Markt: {r["market_condition"]}<br>'
            f'<b>Lesson:</b> {r["lesson_learned"]}',
            unsafe_allow_html=True,
        )
        st.markdown("")
else:
    st.info("Noch keine Trade Reviews. Nutze den ANALYSIEREN Button oben.")

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
