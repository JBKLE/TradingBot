"""Streamlit Dashboard – DQN Trading Bot."""
import glob
import json
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.getenv("DATA_DIR", "./data")
DB_PATH = os.path.join(DATA_DIR, "trades.db")
SIM_DB_PATH = os.path.join(DATA_DIR, "simulation.db")
HISTORY_DB_PATH = os.path.join(DATA_DIR, "simLastCharts.db")
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
</style>
""", unsafe_allow_html=True)


# ── DB helpers ─────────────────────────────────────────────────────────────────
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


# ── Backtest Helpers ─────────────────────────────────────────────────────────

def _render_single_backtest(bt_trades: list[dict], bt_source: str, bt_with_pos: bool,
                            fin_capital: float, fin_risk: float, fin_leverage: int,
                            fin_eur_usd: float, fin_enabled: bool, conf_threshold: int = 8):
    """Einzelnen Trade per Dropdown auswaehlen und backtesten."""
    options = {
        f"#{t['id']} {t['asset']} {t['direction']} "
        f"{'%.4f' % t['entry_price']} | P/L: {'%.2f' % (t['pnl'] or 0)} "
        f"({t['status']}) {(t.get('entry_timestamp') or '')[:16]}": t
        for t in bt_trades
    }
    selected = st.selectbox("Trade auswaehlen", list(options.keys()), key="bt_trade_select")
    sel_trade = options[selected]

    if st.button("ANALYSIEREN", width="stretch", key="bt_run"):
        with st.spinner("DQN analysiert Trade..."):
            r = _run_single_backtest(sel_trade["id"], bt_source, bt_with_pos,
                                     fin_capital if fin_enabled else None,
                                     fin_risk if fin_enabled else None,
                                     fin_leverage if fin_enabled else None,
                                     fin_eur_usd)
            if r:
                _display_single_result(r)


def _render_batch_backtest(bt_trades: list[dict], bt_source: str, bt_with_pos: bool,
                           fin_capital: float, fin_risk: float, fin_leverage: int,
                           fin_eur_usd: float, fin_enabled: bool, conf_threshold: int = 8):
    """Alle geladenen Trades backtesten mit Fortschritt und Abbruch."""
    # Filter-Optionen
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        filter_asset = st.selectbox(
            "Asset filtern",
            ["Alle"] + sorted(set(t["asset"] for t in bt_trades)),
            key="bt_filter_asset",
        )
    with fc2:
        filter_dir = st.selectbox("Richtung filtern", ["Alle", "BUY", "SELL"], key="bt_filter_dir")
    with fc3:
        st.markdown(f"**{len(bt_trades)} Trades geladen**")

    # Filter anwenden
    filtered = bt_trades
    if filter_asset != "Alle":
        filtered = [t for t in filtered if t["asset"] == filter_asset]
    if filter_dir != "Alle":
        filtered = [t for t in filtered if t["direction"] == filter_dir]

    st.markdown(f"**{len(filtered)} Trades nach Filter**")

    # Session-State fuer Batch
    if "batch_results" not in st.session_state:
        st.session_state.batch_results = []
    if "batch_running" not in st.session_state:
        st.session_state.batch_running = False
    if "batch_equity" not in st.session_state:
        st.session_state.batch_equity = []

    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        start_batch = st.button("BATCH STARTEN", width="stretch", key="bt_batch_start")
    with bc2:
        stop_batch = st.button("ABBRECHEN", width="stretch", key="bt_batch_stop")
    with bc3:
        if st.session_state.batch_results:
            csv = _results_to_csv(st.session_state.batch_results, fin_enabled)
            st.download_button(
                "CSV DOWNLOAD",
                csv,
                file_name=f"backtest_{bt_source}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                width="stretch",
            )

    if stop_batch:
        st.session_state.batch_running = False

    if start_batch and filtered:
        st.session_state.batch_results = []
        st.session_state.batch_equity = []
        st.session_state.batch_running = True

        # Laufendes Kapital fuer Equity-Kurve
        running_capital = fin_capital if fin_enabled else 0

        progress_bar = st.progress(0, text="Starte Batch-Backtest...")
        status_text = st.empty()
        results_container = st.empty()

        for i, trade in enumerate(filtered):
            if not st.session_state.batch_running:
                status_text.warning(f"Abgebrochen nach {i}/{len(filtered)} Trades")
                break

            pct = (i + 1) / len(filtered)
            progress_bar.progress(pct, text=f"Trade {i+1}/{len(filtered)}: #{trade['id']} {trade['asset']} {trade['direction']}")

            r = _run_single_backtest(
                trade["id"], bt_source, bt_with_pos,
                running_capital if fin_enabled else None,
                fin_risk if fin_enabled else None,
                fin_leverage if fin_enabled else None,
                fin_eur_usd,
            )
            if r and "error" not in r:
                # DQN-Entscheidung: Haette die KI hier gehandelt?
                dqn_action = r.get("dqn_action", "HOLD")
                dqn_conf = r.get("dqn_confidence", 0)
                dqn_would_trade = (
                    dqn_action in ("BUY", "SELL")
                    and dqn_conf >= conf_threshold
                )
                r["dqn_would_trade"] = dqn_would_trade

                # Equity tracking: nur P&L anrechnen wenn DQN gehandelt haette
                if fin_enabled and "financial" in r:
                    fin = r["financial"]
                    if not dqn_would_trade:
                        # DQN haette NICHT gehandelt → P&L = 0
                        fin["netto_pnl_eur"] = 0.0
                        fin["brutto_pnl_eur"] = 0.0
                        fin["spread_cost_eur"] = 0.0
                        fin["skipped"] = True
                        fin["kapital_danach"] = round(running_capital, 2)
                        st.session_state.batch_equity.append(running_capital)
                    else:
                        fin["skipped"] = False
                        if fin.get("margin_call"):
                            status_text.error(
                                f"MARGIN CALL bei Trade #{trade['id']}! "
                                f"Kapital: {running_capital:.2f} EUR < "
                                f"Margin: {fin['margin_eur']:.2f} EUR"
                            )
                            fin["margin_call_stop"] = True
                            st.session_state.batch_results.append(r)
                            st.session_state.batch_equity.append(running_capital)
                            break
                        running_capital += fin["netto_pnl_eur"]
                        fin["kapital_danach"] = round(running_capital, 2)
                        st.session_state.batch_equity.append(running_capital)

                st.session_state.batch_results.append(r)

            # Zwischenergebnis alle 10 Trades aktualisieren
            if (i + 1) % 10 == 0 or i == len(filtered) - 1:
                _display_batch_summary(
                    results_container, st.session_state.batch_results,
                    i + 1, len(filtered), fin_enabled, fin_capital,
                    st.session_state.batch_equity,
                )

        st.session_state.batch_running = False
        progress_bar.progress(1.0, text="Fertig!")

    # Vorherige Ergebnisse anzeigen (auch nach Abbruch)
    if st.session_state.batch_results and not start_batch:
        _display_batch_summary(
            st, st.session_state.batch_results,
            len(st.session_state.batch_results), len(filtered),
            fin_enabled, fin_capital, st.session_state.batch_equity,
        )


def _run_single_backtest(trade_id: int, source: str, with_pos: bool,
                         capital: float | None = None, risk_pct: float | None = None,
                         leverage: int | None = None, eur_usd: float = 1.08) -> dict | None:
    """Einzelnen Backtest via API ausfuehren."""
    try:
        params = {
            "source": source,
            "with_position": str(with_pos).lower(),
        }
        if capital is not None:
            params["capital"] = capital
            params["risk_pct"] = risk_pct
            params["leverage"] = leverage
            params["eur_usd"] = eur_usd
        resp = httpx.post(
            f"{BOT_API_URL}/api/backtest/{trade_id}",
            params=params,
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception:
        return None


def _display_single_result(r: dict):
    """Einzelnes Backtest-Ergebnis anzeigen."""
    verdict = r.get("verdict", "?")
    v_color = {"MATCH": "green", "BESSER": "blue", "MISS": "red"}.get(verdict, "gray")
    st.markdown(f"### Ergebnis: :{v_color}[**{verdict}**]")

    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown("**Echter Trade**")
        st.markdown(
            f"- Richtung: **{r['trade_direction']}**\n"
            f"- Entry: {r['trade_entry_price']:.4f}\n"
            f"- P/L: **{r['trade_pnl']:+.2f}** "
            f"({'Gewinn' if r['trade_won'] else 'Verlust'})"
        )
    with rc2:
        st.markdown("**DQN-Entscheidung**")
        st.markdown(
            f"- Aktion: **{r['dqn_action']}** "
            f"(Confidence: {r['dqn_confidence']}/10)\n"
            f"- SL: {r['dqn_sl'] or '-'}\n"
            f"- TP: {r['dqn_tp'] or '-'}"
        )
    st.markdown(
        f"**Q-Werte:** {r['dqn_q_values']} | "
        f"Kerzen: {r['candle_count']}/500 | "
        f"Position-Info: {'Ja' if r['with_position'] else 'Nein'}"
    )

    # Finanzrechnung anzeigen
    if "financial" in r:
        fin = r["financial"]
        st.markdown("---")
        st.markdown("**Finanzrechnung**")
        fc1, fc2, fc3, fc4 = st.columns(4)
        fc1.metric("Lot-Groesse", f"{fin['lot_size']:.4f}")
        fc2.metric("Positionswert", f"{fin['position_value_eur']:.2f} EUR")
        fc3.metric("Margin", f"{fin['margin_eur']:.2f} EUR")
        pnl_color = "normal" if fin["netto_pnl_eur"] >= 0 else "inverse"
        fc4.metric("Netto P/L", f"{fin['netto_pnl_eur']:+.2f} EUR",
                   f"Spread: -{fin['spread_cost_eur']:.2f} EUR", delta_color=pnl_color)


def _display_batch_summary(container, results: list[dict], done: int, total: int,
                           fin_enabled: bool = False, start_capital: float = 0,
                           equity_curve: list[float] | None = None):
    """Batch-Zusammenfassung mit Statistiken und Tabelle anzeigen."""
    with container.container():
        n = len(results)
        match = sum(1 for r in results if r["verdict"] == "MATCH")
        besser = sum(1 for r in results if r["verdict"] == "BESSER")
        miss = sum(1 for r in results if r["verdict"] == "MISS")
        avg_conf = sum(r["dqn_confidence"] for r in results) / n if n else 0

        # Wie viele haette DQN gehandelt?
        traded = sum(1 for r in results if r.get("dqn_would_trade"))
        skipped = n - traded

        # Statistik-Kacheln
        sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
        sc1.metric("Analysiert", f"{done}/{total}")
        sc2.metric("MATCH", match, f"{match/n*100:.0f}%" if n else "")
        sc3.metric("BESSER", besser, f"{besser/n*100:.0f}%" if n else "")
        sc4.metric("MISS", miss, f"{miss/n*100:.0f}%" if n else "")
        sc5.metric("Avg Confidence", f"{avg_conf:.1f}/10")
        sc6.metric("DQN handelt", f"{traded}", f"{skipped} uebersprungen")

        # ── Finanz-Kennzahlen ──
        if fin_enabled and equity_curve:
            end_capital = equity_curve[-1] if equity_curve else start_capital
            rendite_pct = ((end_capital - start_capital) / start_capital * 100) if start_capital > 0 else 0

            # Max Drawdown
            peak = start_capital
            max_dd = 0
            for eq in equity_curve:
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak * 100 if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd

            # Profit Factor
            fin_results = [r for r in results if "financial" in r]
            gewinne = sum(r["financial"]["netto_pnl_eur"] for r in fin_results if r["financial"]["netto_pnl_eur"] > 0)
            verluste = abs(sum(r["financial"]["netto_pnl_eur"] for r in fin_results if r["financial"]["netto_pnl_eur"] < 0))
            profit_factor = gewinne / verluste if verluste > 0 else float("inf") if gewinne > 0 else 0

            # Groesster Gewinn / Verlust
            all_pnl = [r["financial"]["netto_pnl_eur"] for r in fin_results]
            max_win = max(all_pnl) if all_pnl else 0
            max_loss = min(all_pnl) if all_pnl else 0

            st.markdown("---")
            st.markdown("**Finanz-Kennzahlen**")
            fk1, fk2, fk3, fk4, fk5 = st.columns(5)
            fk1.metric("Startkapital", f"{start_capital:.2f} EUR")
            fk2.metric("Endkapital", f"{end_capital:.2f} EUR",
                       f"{rendite_pct:+.1f}%")
            fk3.metric("Max Drawdown", f"{max_dd:.1f}%")
            fk4.metric("Profit Factor", f"{profit_factor:.2f}")
            fk5.metric("Max Win / Loss",
                       f"+{max_win:.2f} / {max_loss:.2f} EUR")

            # Equity-Kurve
            if len(equity_curve) > 1:
                st.markdown("**Equity-Kurve**")
                eq_df = pd.DataFrame({
                    "Trade": range(1, len(equity_curve) + 1),
                    "Kapital (EUR)": equity_curve,
                })
                st.line_chart(eq_df, x="Trade", y="Kapital (EUR)")

        # Aufschluesselung nach Asset
        assets = sorted(set(r["asset"] for r in results))
        if len(assets) > 1:
            asset_rows = []
            for asset in assets:
                ar = [r for r in results if r["asset"] == asset]
                an = len(ar)
                a_match = sum(1 for r in ar if r["verdict"] == "MATCH")
                a_besser = sum(1 for r in ar if r["verdict"] == "BESSER")
                a_miss = sum(1 for r in ar if r["verdict"] == "MISS")
                row = {
                    "Asset": asset,
                    "Trades": an,
                    "MATCH": a_match,
                    "BESSER": a_besser,
                    "MISS": a_miss,
                    "Match%": f"{a_match/an*100:.0f}%" if an else "-",
                }
                # EUR P&L pro Asset
                if fin_enabled:
                    ar_fin = [r for r in ar if "financial" in r]
                    asset_pnl = sum(r["financial"]["netto_pnl_eur"] for r in ar_fin)
                    row["P/L EUR"] = f"{asset_pnl:+.2f}"
                asset_rows.append(row)
            st.dataframe(pd.DataFrame(asset_rows), width="stretch", hide_index=True)

        # Detail-Tabelle
        with st.expander(f"Detail-Tabelle ({n} Trades)", expanded=False):
            rows = []
            for r in results:
                row = {
                    "Asset": r["asset"],
                    "Trade": r["trade_direction"],
                    "Entry": f"{r['trade_entry_price']:.4f}",
                    "P/L": f"{r['trade_pnl']:+.4f}",
                    "Gewinn": r["trade_won"],
                    "DQN": r["dqn_action"],
                    "Conf": r["dqn_confidence"],
                    "Handelt": "JA" if r.get("dqn_would_trade") else "NEIN",
                    "Verdict": r["verdict"],
                    "Kerzen": r["candle_count"],
                }
                if fin_enabled and "financial" in r:
                    fin = r["financial"]
                    if fin.get("skipped"):
                        row["Netto EUR"] = "-"
                    else:
                        row["Netto EUR"] = f"{fin['netto_pnl_eur']:+.2f}"
                    row["Kapital"] = f"{fin.get('kapital_danach', '-')}"
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _results_to_csv(results: list[dict], fin_enabled: bool = False) -> str:
    """Batch-Ergebnisse als CSV-String."""
    import io
    rows = []
    for r in results:
        row = {
            "Asset": r["asset"],
            "Trade": r["trade_direction"],
            "Entry": r["trade_entry_price"],
            "P/L": r["trade_pnl"],
            "Gewinn": r["trade_won"],
            "DQN": r["dqn_action"],
            "Conf": r["dqn_confidence"],
            "Handelt": r.get("dqn_would_trade", ""),
            "Verdict": r["verdict"],
            "Kerzen": r["candle_count"],
        }
        if fin_enabled and "financial" in r:
            fin = r["financial"]
            row["Skipped"] = fin.get("skipped", False)
            row["Lot"] = fin["lot_size"] if not fin.get("skipped") else 0
            row["Positionswert_EUR"] = fin["position_value_eur"] if not fin.get("skipped") else 0
            row["Margin_EUR"] = fin["margin_eur"] if not fin.get("skipped") else 0
            row["Brutto_EUR"] = fin["brutto_pnl_eur"]
            row["Spread_EUR"] = fin["spread_cost_eur"]
            row["Netto_EUR"] = fin["netto_pnl_eur"]
            row["Kapital_danach"] = fin.get("kapital_danach", "")
        rows.append(row)
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    return buf.getvalue()


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# ◈ DQN TRADING BOT")
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
        if st.button("SPEICHERN", width="stretch", disabled=not _changed):
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
    st_autorefresh(interval=refresh * 1000, key="auto_refresh")


# ══════════════════════════════════════════════════════════════════════════════
# TAB NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
tab_trading, tab_simulation, tab_history = st.tabs(["◈ TRADING", "◈ SIMULATION", "◈ HISTORISCHE DATEN"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: TRADING (existing view)
# ══════════════════════════════════════════════════════════════════════════════
with tab_trading:
    # Load data
    trades = load_trades()
    snapshots = load_snapshots()
    analyses = load_analyses()

    # ── KPI Row ────────────────────────────────────────────────────────────────
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

    # ── Action Buttons ────────────────────────────────────────────────────
    st.markdown("### ◈ AKTIONEN")
    col_b1, col_b2, col_b3, col_b4 = st.columns(4)

    with col_b1:
        if st.button("DQN BACKTEST", width="stretch"):
            st.session_state["show_backtest"] = True

    # ── Backtest-Panel (aufklappbar) ─────────────────────────────────
    if st.session_state.get("show_backtest"):
        with st.container():
            st.markdown("#### DQN Backtest")
            bt_col1, bt_col2, bt_col3, bt_col4 = st.columns([2, 1, 1, 1])
            with bt_col1:
                bt_source = st.selectbox("Quelle", ["sim", "real"], key="bt_source")
            with bt_col2:
                bt_with_pos = st.checkbox("Mit Position-Info", key="bt_with_pos")
            with bt_col3:
                bt_limit = st.number_input("Trades laden", value=50, min_value=5, max_value=5000, key="bt_limit")
            with bt_col4:
                bt_mode = st.selectbox("Modus", ["Einzeln", "Batch (alle)"], key="bt_mode")

            # Confidence-Schwelle
            bt_conf_threshold = st.slider(
                "Min. Confidence (DQN handelt nur ab diesem Wert)",
                min_value=1, max_value=10, value=8, step=1, key="bt_conf_threshold",
            )

            # Finanz-Einstellungen
            fin_enabled = st.checkbox("Finanzrechnung aktivieren", key="bt_fin_enabled")
            if fin_enabled:
                fin_c1, fin_c2, fin_c3, fin_c4 = st.columns(4)
                with fin_c1:
                    fin_capital = st.number_input(
                        "Startkapital (EUR)", value=1000.0,
                        min_value=100.0, step=100.0, key="bt_fin_capital")
                with fin_c2:
                    fin_risk = st.number_input(
                        "Risiko pro Trade (%)", value=1.0,
                        min_value=0.1, max_value=10.0, step=0.1, key="bt_fin_risk") / 100.0
                with fin_c3:
                    fin_leverage = st.number_input(
                        "Hebel", value=20,
                        min_value=1, max_value=100, step=1, key="bt_fin_leverage")
                with fin_c4:
                    fin_eur_usd = st.number_input(
                        "EUR/USD Kurs", value=1.08,
                        min_value=0.80, max_value=1.50, step=0.01, key="bt_fin_eurusd")
            else:
                fin_capital, fin_risk, fin_leverage, fin_eur_usd = 1000.0, 0.01, 20, 1.08

            # Trades laden
            try:
                resp = httpx.get(
                    f"{BOT_API_URL}/api/backtest/trades",
                    params={"source": bt_source, "limit": bt_limit},
                    timeout=10,
                )
                if resp.status_code != 200:
                    st.error(f"Trades laden fehlgeschlagen: {resp.status_code}")
                else:
                    bt_trades = resp.json().get("trades", [])
                    if not bt_trades:
                        st.warning("Keine abgeschlossenen Trades gefunden")
                    elif bt_mode == "Einzeln":
                        _render_single_backtest(bt_trades, bt_source, bt_with_pos,
                                                fin_capital, fin_risk, fin_leverage,
                                                fin_eur_usd, fin_enabled, bt_conf_threshold)
                    else:
                        _render_batch_backtest(bt_trades, bt_source, bt_with_pos,
                                               fin_capital, fin_risk, fin_leverage,
                                               fin_eur_usd, fin_enabled, bt_conf_threshold)
            except Exception as e:
                st.error(f"API nicht erreichbar: {e}")

    with col_b2:
        if st.button("TAGESBILANZ", width="stretch"):
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
                        cs = result.get("ai_summary", {})
                        if cs.get("summary"):
                            st.info(cs["summary"])
                        for rec in cs.get("recommendations", []):
                            st.markdown(f"- {rec}")
                    else:
                        st.error(f"Fehler: {resp.status_code}")
                except Exception as e:
                    st.error(f"API nicht erreichbar: {e}")

    with col_b3:
        if st.button("WOCHENREPORT", width="stretch"):
            with st.spinner("Erstelle Wochenreport..."):
                try:
                    resp = httpx.post(f"{BOT_API_URL}/api/weekly-report", timeout=120)
                    if resp.status_code == 200:
                        result = resp.json()
                        st.success(
                            f"Woche: {result.get('trades_count', 0)} Trades | "
                            f"P/L: {result.get('total_pl', 0):+.2f} EUR"
                        )
                        cs = result.get("ai_summary", {})
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
        if st.button("BOT STATUS", width="stretch"):
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

    # ── API Tests ─────────────────────────────────────────────────────────────
    st.markdown("### ◈ API TESTS")

    # Persist test results in session state
    if "api_test_results" not in st.session_state:
        st.session_state.api_test_results = {}

    _api_tests = [
        ("CAPITAL.COM", "test_capital", "/api/test/capital", 30),
        ("DQN MODELL", "test_dqn", "/api/test/dqn", 15),
        ("NTFY.SH", "test_ntfy", "/api/test/ntfy", 15),
    ]

    col_t1, col_t2, col_t3 = st.columns(3)
    for _col, (_label, _key, _endpoint, _timeout) in zip(
        [col_t1, col_t2, col_t3], _api_tests
    ):
        with _col:
            # Show colored status from previous test
            _prev = st.session_state.api_test_results.get(_key)
            if _prev is not None:
                if _prev["status"] == "ok":
                    st.markdown(
                        f'<div style="background:#002200;border:1px solid #00ff41;border-radius:4px;'
                        f'padding:8px;margin-bottom:8px;font-size:0.8rem;color:#00ff41;">'
                        f'{_label}: OK ({_prev["latency_ms"]}ms)</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div style="background:#220000;border:1px solid #ff4444;border-radius:4px;'
                        f'padding:8px;margin-bottom:8px;font-size:0.8rem;color:#ff4444;">'
                        f'{_label}: FEHLER</div>',
                        unsafe_allow_html=True,
                    )

            if st.button(_label, width="stretch", key=_key):
                with st.spinner(f"Teste {_label}..."):
                    try:
                        _resp = httpx.post(f"{BOT_API_URL}{_endpoint}", timeout=_timeout)
                        _r = _resp.json()
                        st.session_state.api_test_results[_key] = _r
                        if _r.get("status") == "ok":
                            st.success(f"{_r['message']} ({_r['latency_ms']}ms)")
                        else:
                            st.error(_r.get("message", "Unbekannter Fehler"))
                    except Exception as _e:
                        st.session_state.api_test_results[_key] = {
                            "status": "error", "latency_ms": 0,
                            "message": f"Bot-API nicht erreichbar: {_e}",
                        }
                        st.error(f"Bot-API nicht erreichbar: {_e}")
                    st.rerun()

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

    # ── Charts Row ─────────────────────────────────────────────────────────────
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

    # ── Open Positions ─────────────────────────────────────────────────────────
    st.markdown("### ◈ OFFENE POSITIONEN")
    if not open_trades.empty:
        display_open = open_trades[[
            "timestamp", "asset", "direction", "entry_price",
            "stop_loss", "take_profit", "position_size", "confidence", "deal_id"
        ]].copy()

        now_utc = pd.Timestamp.now(tz="UTC")
        ts_aware = display_open["timestamp"].dt.tz_localize("UTC") if display_open["timestamp"].dt.tz is None else display_open["timestamp"].dt.tz_convert("UTC")
        delta = now_utc - ts_aware
        display_open["offen_seit"] = delta.apply(
            lambda d: f"{int(d.total_seconds()//3600)}h {int((d.total_seconds()%3600)//60)}m"
        )

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
        st.dataframe(display_open, width="stretch", hide_index=True)
        st.caption("Aktueller Kurs nicht in DB verfügbar – ▼SL % / ▲TP % zeigen Abstand vom Entry.")
    else:
        st.markdown("*Keine offenen Positionen.*")

    st.markdown("---")

    # ── Trade History ──────────────────────────────────────────────────────────
    st.markdown("### ◈ TRADE HISTORY")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        assets = ["Alle"] + (sorted(trades["asset"].unique().tolist()) if not trades.empty else [])
        asset_filter = st.selectbox("Asset", assets, key="trading_asset")
    with col_f2:
        statuses = ["Alle", "OPEN", "TAKE_PROFIT", "STOPPED_OUT", "CLOSED"]
        status_filter = st.selectbox("Status", statuses, key="trading_status")
    with col_f3:
        days = st.selectbox("Zeitraum", ["7 Tage", "30 Tage", "90 Tage", "Alle"], key="trading_days")

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
        st.dataframe(display, width="stretch", hide_index=True)
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
            if review_options and st.button("ANALYSIEREN", width="stretch"):
                with st.spinner("Trade wird analysiert..."):
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

    # ── Analyses ───────────────────────────────────────────────────────────────
    st.markdown("### ◈ DQN ANALYSEN")
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

    # ── Log Viewer ─────────────────────────────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_simulation:
    sim_conn = get_sim_connection()

    if sim_conn is None:
        st.info("Simulation-Datenbank nicht gefunden. Starte den Bot mit SIM_ENABLED=true.")
    else:
        # ── Load sim data ──────────────────────────────────────────────────────
        sim_trades_all = sim_query("SELECT * FROM sim_trades ORDER BY entry_timestamp DESC")
        sim_prices = sim_query("SELECT COUNT(*) as cnt FROM price_history")

        if sim_trades_all.empty:
            st.info("Noch keine Simulationsdaten vorhanden. Die Simulation sammelt jede Minute Daten.")
        else:
            sim_trades_all["entry_timestamp"] = pd.to_datetime(sim_trades_all["entry_timestamp"], utc=True).dt.tz_localize(None)
            sim_trades_all["exit_timestamp"] = pd.to_datetime(sim_trades_all["exit_timestamp"], utc=True, errors="coerce").dt.tz_localize(None)

            # ── KPI Row ────────────────────────────────────────────────────────
            sim_open = sim_trades_all[sim_trades_all["status"] == "open"]
            sim_closed = sim_trades_all[sim_trades_all["status"].isin(["closed_tp", "closed_sl"])]
            sim_tp = sim_trades_all[sim_trades_all["status"] == "closed_tp"]
            sim_sl = sim_trades_all[sim_trades_all["status"] == "closed_sl"]

            total_closed = len(sim_closed)
            win_rate_sim = len(sim_tp) / total_closed * 100 if total_closed > 0 else 0.0
            avg_pnl = sim_closed["pnl"].mean() if not sim_closed.empty else 0.0
            avg_r = sim_closed["r_multiple"].mean() if not sim_closed.empty else 0.0
            price_count = int(sim_prices["cnt"].iloc[0]) if not sim_prices.empty else 0

            col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
            col1.metric("GESAMT TRADES", f"{len(sim_trades_all):,}")
            col2.metric("OFFENE SIM-TRADES", f"{len(sim_open):,}")
            col3.metric("ABGESCHLOSSEN", f"{total_closed:,}")
            col4.metric("WIN (TP)", f"{len(sim_tp):,}")
            col5.metric("LOSS (SL)", f"{len(sim_sl):,}")
            col6.metric("WIN RATE", f"{win_rate_sim:.1f}%")
            col7.metric("PREISDATEN", f"{price_count:,}")

            st.markdown("---")

            # ── Price Charts ───────────────────────────────────────────────────
            st.markdown("### ◈ PREISVERLAUF (LIVE)")

            sim_price_data = sim_query(
                "SELECT timestamp, asset, close FROM price_history ORDER BY timestamp ASC"
            )
            if not sim_price_data.empty:
                sim_price_data["timestamp"] = pd.to_datetime(sim_price_data["timestamp"], utc=True).dt.tz_localize(None)

                # Asset selector for chart
                chart_assets = sorted(sim_price_data["asset"].unique().tolist())
                selected_assets = st.multiselect(
                    "Assets anzeigen", chart_assets, default=chart_assets, key="sim_chart_assets"
                )

                if selected_assets:
                    # Pivot: one column per asset
                    chart_filtered = sim_price_data[sim_price_data["asset"].isin(selected_assets)]
                    chart_pivot = chart_filtered.pivot_table(
                        index="timestamp", columns="asset", values="close", aggfunc="last"
                    )

                    # Separate charts per asset (different price scales)
                    cols = st.columns(min(len(selected_assets), 2))
                    for i, asset in enumerate(selected_assets):
                        if asset in chart_pivot.columns:
                            with cols[i % 2]:
                                st.markdown(f"**{asset}**")
                                asset_data = chart_pivot[[asset]].dropna()
                                if not asset_data.empty:
                                    current = asset_data.iloc[-1, 0]
                                    first = asset_data.iloc[0, 0]
                                    change = ((current - first) / first) * 100
                                    st.metric(
                                        "Aktuell",
                                        f"{current:.2f}",
                                        f"{change:+.2f}%",
                                    )
                                    st.line_chart(asset_data, color="#00ff41")
            else:
                st.info("Noch keine Preisdaten gespeichert.")

            st.markdown("---")

            # ── Win/Loss per variant ───────────────────────────────────────────
            st.markdown("### ◈ PERFORMANCE PRO VARIANTE")

            if not sim_closed.empty:
                variant_stats = sim_closed.groupby("sl_variant").agg(
                    total=("id", "count"),
                    wins=("status", lambda x: (x == "closed_tp").sum()),
                    losses=("status", lambda x: (x == "closed_sl").sum()),
                    avg_pnl=("pnl", "mean"),
                    avg_r=("r_multiple", "mean"),
                ).reset_index()
                variant_stats["win_rate"] = (variant_stats["wins"] / variant_stats["total"] * 100).round(1)
                variant_stats.columns = ["VARIANTE", "TOTAL", "WINS", "LOSSES", "AVG P/L", "AVG R", "WIN RATE %"]
                variant_stats["AVG P/L"] = variant_stats["AVG P/L"].apply(lambda x: f"{x:+.6f}")
                variant_stats["AVG R"] = variant_stats["AVG R"].apply(lambda x: f"{x:+.2f}")
                st.dataframe(variant_stats, width="stretch", hide_index=True)

            # ── Win/Loss per asset ─────────────────────────────────────────────
            st.markdown("### ◈ PERFORMANCE PRO ASSET")

            if not sim_closed.empty:
                asset_stats = sim_closed.groupby("asset").agg(
                    total=("id", "count"),
                    wins=("status", lambda x: (x == "closed_tp").sum()),
                    losses=("status", lambda x: (x == "closed_sl").sum()),
                    avg_pnl=("pnl", "mean"),
                    avg_r=("r_multiple", "mean"),
                ).reset_index()
                asset_stats["win_rate"] = (asset_stats["wins"] / asset_stats["total"] * 100).round(1)
                asset_stats.columns = ["ASSET", "TOTAL", "WINS", "LOSSES", "AVG P/L", "AVG R", "WIN RATE %"]
                asset_stats["AVG P/L"] = asset_stats["AVG P/L"].apply(lambda x: f"{x:+.6f}")
                asset_stats["AVG R"] = asset_stats["AVG R"].apply(lambda x: f"{x:+.2f}")
                st.dataframe(asset_stats, width="stretch", hide_index=True)

            # ── Win/Loss per direction ─────────────────────────────────────────
            st.markdown("### ◈ PERFORMANCE PRO RICHTUNG")

            if not sim_closed.empty:
                dir_stats = sim_closed.groupby("direction").agg(
                    total=("id", "count"),
                    wins=("status", lambda x: (x == "closed_tp").sum()),
                    losses=("status", lambda x: (x == "closed_sl").sum()),
                    avg_pnl=("pnl", "mean"),
                    avg_r=("r_multiple", "mean"),
                ).reset_index()
                dir_stats["win_rate"] = (dir_stats["wins"] / dir_stats["total"] * 100).round(1)
                dir_stats.columns = ["RICHTUNG", "TOTAL", "WINS", "LOSSES", "AVG P/L", "AVG R", "WIN RATE %"]
                dir_stats["AVG P/L"] = dir_stats["AVG P/L"].apply(lambda x: f"{x:+.6f}")
                dir_stats["AVG R"] = dir_stats["AVG R"].apply(lambda x: f"{x:+.2f}")
                st.dataframe(dir_stats, width="stretch", hide_index=True)

            st.markdown("---")

            # ── DB Walker ──────────────────────────────────────────────────────
            st.markdown("### ◈ DB WALKER – SIM TRADES")

            # Filters
            col_f1, col_f2, col_f3, col_f4, col_f5 = st.columns(5)

            with col_f1:
                sim_assets = ["Alle"] + sorted(sim_trades_all["asset"].unique().tolist())
                sim_asset_filter = st.selectbox("Asset", sim_assets, key="sim_asset")

            with col_f2:
                sim_directions = ["Alle", "BUY", "SELL"]
                sim_dir_filter = st.selectbox("Richtung", sim_directions, key="sim_dir")

            with col_f3:
                sim_variants = ["Alle", "tight", "medium", "wide"]
                sim_variant_filter = st.selectbox("Variante", sim_variants, key="sim_variant")

            with col_f4:
                sim_statuses = ["Alle", "open", "closed_tp", "closed_sl"]
                sim_status_filter = st.selectbox("Status", sim_statuses, key="sim_status")

            with col_f5:
                sim_days = st.selectbox("Zeitraum", ["1 Tag", "3 Tage", "7 Tage", "30 Tage", "Alle"], key="sim_days")

            # Date/time range picker
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                min_date = sim_trades_all["entry_timestamp"].min().date()
                max_date = sim_trades_all["entry_timestamp"].max().date()
                date_from = st.date_input("Von Datum", value=min_date, min_value=min_date, max_value=max_date, key="sim_date_from")
            with col_d2:
                date_to = st.date_input("Bis Datum", value=max_date, min_value=min_date, max_value=max_date, key="sim_date_to")

            col_t1, col_t2 = st.columns(2)
            with col_t1:
                time_from = st.time_input("Von Uhrzeit", value=datetime.min.time(), key="sim_time_from")
            with col_t2:
                time_to = st.time_input("Bis Uhrzeit", value=datetime.max.replace(microsecond=0).time(), key="sim_time_to")

            # Apply filters
            filtered_sim = sim_trades_all.copy()

            if sim_asset_filter != "Alle":
                filtered_sim = filtered_sim[filtered_sim["asset"] == sim_asset_filter]
            if sim_dir_filter != "Alle":
                filtered_sim = filtered_sim[filtered_sim["direction"] == sim_dir_filter]
            if sim_variant_filter != "Alle":
                filtered_sim = filtered_sim[filtered_sim["sl_variant"] == sim_variant_filter]
            if sim_status_filter != "Alle":
                filtered_sim = filtered_sim[filtered_sim["status"] == sim_status_filter]

            if sim_days != "Alle":
                d = int(sim_days.split()[0])
                cutoff = pd.Timestamp.now() - timedelta(days=d)
                filtered_sim = filtered_sim[filtered_sim["entry_timestamp"] >= cutoff]

            # Date/time filter
            dt_from = pd.Timestamp(datetime.combine(date_from, time_from))
            dt_to = pd.Timestamp(datetime.combine(date_to, time_to))
            filtered_sim = filtered_sim[
                (filtered_sim["entry_timestamp"] >= dt_from) &
                (filtered_sim["entry_timestamp"] <= dt_to)
            ]

            # Display count
            st.markdown(f"**{len(filtered_sim):,} Trades gefunden**")

            # Show filtered data
            if not filtered_sim.empty:
                display_sim = filtered_sim[[
                    "entry_timestamp", "asset", "direction", "sl_variant",
                    "entry_price", "sl_price", "tp_price",
                    "exit_timestamp", "exit_price", "status", "pnl", "r_multiple"
                ]].copy()

                display_sim["entry_timestamp"] = display_sim["entry_timestamp"].dt.strftime("%Y-%m-%d %H:%M")
                display_sim["exit_timestamp"] = display_sim["exit_timestamp"].dt.strftime("%Y-%m-%d %H:%M").fillna("–")
                display_sim["pnl"] = display_sim["pnl"].apply(
                    lambda x: f"{x:+.6f}" if pd.notna(x) else "–"
                )
                display_sim["r_multiple"] = display_sim["r_multiple"].apply(
                    lambda x: f"{x:+.2f}" if pd.notna(x) else "–"
                )
                display_sim["status"] = display_sim["status"].map({
                    "open": "OFFEN",
                    "closed_tp": "WIN (TP)",
                    "closed_sl": "LOSS (SL)",
                })
                display_sim.columns = [
                    "ENTRY ZEIT", "ASSET", "DIR", "VARIANTE",
                    "ENTRY", "SL", "TP",
                    "EXIT ZEIT", "EXIT", "STATUS", "P/L", "R-MULT"
                ]

                # Limit display for performance
                page_size = 100
                total_pages = max(1, (len(display_sim) + page_size - 1) // page_size)
                page = st.number_input("Seite", min_value=1, max_value=total_pages, value=1, key="sim_page")
                start_idx = (page - 1) * page_size
                end_idx = start_idx + page_size

                st.dataframe(
                    display_sim.iloc[start_idx:end_idx],
                    width="stretch",
                    hide_index=True,
                )
                st.caption(f"Seite {page} von {total_pages} ({page_size} Trades pro Seite)")
            else:
                st.info("Keine Trades fuer die gewaehlten Filter gefunden.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: HISTORISCHE DATEN
# ══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown("### ◈ HISTORISCHE KURSDATEN")
    st.markdown("Lade vergangene 1-Min-Kerzen von Capital.com und lasse die KI im Turbo-Modus darueber laufen.")

    # ── Section 1: Daten-Status ───────────────────────────────────────────
    st.markdown("#### 1. Daten-Status")

    def _load_history_status():
        """Check existing data in simLastCharts.db."""
        db_path = HISTORY_DB_PATH
        if not os.path.exists(db_path):
            return {}
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.execute(
                "SELECT asset, COUNT(*), MIN(timestamp), MAX(timestamp) "
                "FROM price_history GROUP BY asset"
            )
            result = {}
            for row in cursor.fetchall():
                result[row[0]] = {"count": row[1], "min_ts": row[2], "max_ts": row[3]}
            conn.close()
            return result
        except Exception:
            return {}

    history_status = _load_history_status()

    if history_status:
        cols_hs = st.columns(len(history_status))
        for i, (asset, info) in enumerate(history_status.items()):
            with cols_hs[i]:
                st.metric(asset, f"{info['count']:,} Kerzen")
                if info["min_ts"] and info["max_ts"]:
                    st.caption(f"{info['min_ts'][:10]} bis {info['max_ts'][:10]}")
    else:
        st.info("Noch keine historischen Daten vorhanden. Lade Daten ueber das Formular unten.")

    st.markdown("---")

    # ── Section 2: Daten laden ────────────────────────────────────────────
    st.markdown("#### 2. Historische Daten laden")

    col_date1, col_date2 = st.columns(2)
    with col_date1:
        hist_start = st.date_input(
            "Von",
            value=datetime.now() - timedelta(days=28),
            key="hist_start",
        )
    with col_date2:
        hist_end = st.date_input(
            "Bis",
            value=datetime.now() - timedelta(days=1),
            key="hist_end",
        )

    hist_assets = st.multiselect(
        "Assets (leer = alle)",
        options=["GOLD", "SILVER", "OIL_CRUDE", "NATURALGAS"],
        default=[],
        key="hist_assets",
    )

    # Estimated chunks
    if hist_start and hist_end:
        days = (hist_end - hist_start).days
        est_candles = days * 22 * 60  # ~22 trading hours per day
        est_requests = max(1, est_candles // 1000)
        n_assets = len(hist_assets) if hist_assets else 4
        st.caption(
            f"Geschaetzt: ~{est_candles * n_assets:,} Kerzen, "
            f"~{est_requests * n_assets} API-Requests, "
            f"~{est_requests * n_assets * 0.15 / 60:.1f} Minuten"
        )

    col_fetch1, col_fetch2 = st.columns([1, 3])
    with col_fetch1:
        fetch_btn = st.button("Daten laden", key="fetch_history_btn", type="primary")
    with col_fetch2:
        if "fetch_running" not in st.session_state:
            st.session_state.fetch_running = False

    if fetch_btn and not st.session_state.fetch_running:
        st.session_state.fetch_running = True
        try:
            resp = httpx.post(
                f"{BOT_API_URL}/api/fetch-history",
                json={
                    "start_date": hist_start.isoformat(),
                    "end_date": hist_end.isoformat(),
                    "assets": hist_assets if hist_assets else None,
                },
                timeout=10.0,
            )
            data = resp.json()
            if data.get("status") == "started":
                st.success("Download gestartet! Fortschritt wird unten angezeigt.")
            elif data.get("status") == "already_running":
                st.warning("Download laeuft bereits.")
            else:
                st.error(f"Fehler: {data}")
        except Exception as exc:
            st.error(f"API-Fehler: {exc}")
            st.session_state.fetch_running = False

    # Show fetch progress
    if st.session_state.get("fetch_running"):
        try:
            resp = httpx.get(f"{BOT_API_URL}/api/fetch-history/progress", timeout=5.0)
            prog = resp.json()
            if prog.get("running"):
                p = prog.get("progress", {})
                st.info(
                    f"Lade: {p.get('asset', '...')} — "
                    f"Block {p.get('chunk', 0)}/{p.get('total_chunks', '?')} "
                    f"({p.get('new_bars', 0)} neue Kerzen)"
                )
            elif prog.get("result") is not None:
                result = prog["result"]
                if isinstance(result, list):
                    for r in result:
                        st.success(
                            f"{r['asset']}: {r['fetched']} geladen, "
                            f"{r['skipped']} uebersprungen, "
                            f"{r['errors']} Fehler"
                        )
                elif isinstance(result, dict) and "error" in result:
                    st.error(f"Fehler: {result['error']}")
                st.session_state.fetch_running = False
        except Exception:
            pass

    st.markdown("---")

    # ── Section 3: Zeitstrahl-Simulation ──────────────────────────────────
    st.markdown("#### 3. Zeitstrahl-Simulation (Turbo-Modus)")
    st.markdown("DQN laeuft offline ueber alle geladenen Kerzen — so schnell wie CPU/GPU es erlauben.")

    col_sim1, col_sim2, col_sim3 = st.columns(3)
    with col_sim1:
        sim_start = st.date_input(
            "Sim Von",
            value=hist_start if hist_start else datetime.now() - timedelta(days=28),
            key="sim_start",
        )
    with col_sim2:
        sim_end = st.date_input(
            "Sim Bis",
            value=hist_end if hist_end else datetime.now() - timedelta(days=1),
            key="sim_end",
        )
    with col_sim3:
        sim_confidence = st.slider(
            "Confidence Schwelle",
            min_value=1,
            max_value=10,
            value=8,
            key="sim_confidence_threshold",
            help="DQN handelt nur bei Confidence >= diesem Wert",
        )

    sim_assets = st.multiselect(
        "Sim Assets (leer = alle)",
        options=["GOLD", "SILVER", "OIL_CRUDE", "NATURALGAS"],
        default=[],
        key="sim_assets",
    )

    # ── Financial inputs ──────────────────────────────────────────────────
    sim_fin_enabled = st.checkbox("Finanzrechnung aktivieren", key="sim_fin_enabled",
                                  help="Rechnet mit echtem Kapital, Hebel und Spread-Kosten")
    if sim_fin_enabled:
        sf1, sf2, sf3, sf4 = st.columns(4)
        with sf1:
            sim_capital = st.number_input("Startkapital (EUR)", value=1000.0,
                                          min_value=100.0, step=100.0, key="sim_fin_capital")
        with sf2:
            sim_risk_pct = st.number_input("Risiko pro Trade (%)", value=1.0,
                                           min_value=0.1, max_value=10.0, step=0.1,
                                           key="sim_fin_risk") / 100.0
        with sf3:
            sim_leverage = st.number_input("Hebel", value=20, min_value=1,
                                           max_value=100, step=1, key="sim_fin_leverage")
        with sf4:
            sim_eur_usd = st.number_input("EUR/USD", value=1.08, min_value=0.80,
                                          max_value=1.50, step=0.01, key="sim_fin_eurusd")
    else:
        sim_capital, sim_risk_pct, sim_leverage, sim_eur_usd = None, None, None, 1.08

    # ── Session state init ────────────────────────────────────────────────
    if "sim_running" not in st.session_state:
        st.session_state.sim_running = False

    col_simbtn1, col_simbtn2 = st.columns([1, 1])
    with col_simbtn1:
        sim_btn = st.button(
            "Simulation starten",
            key="timeline_sim_btn",
            type="primary",
            disabled=st.session_state.sim_running,
        )
    with col_simbtn2:
        cancel_btn = st.button(
            "Abbrechen",
            key="timeline_cancel_btn",
            disabled=not st.session_state.sim_running,
        )

    if cancel_btn:
        try:
            httpx.post(f"{BOT_API_URL}/api/timeline-sim/cancel", timeout=5.0)
        except Exception:
            pass
        st.session_state.sim_running = False
        st.warning("Abbruch angefordert...")

    if sim_btn and not st.session_state.sim_running:
        st.session_state.pop("sim_result", None)
        try:
            resp = httpx.post(
                f"{BOT_API_URL}/api/run-timeline-sim",
                json={
                    "start_date": sim_start.isoformat() if sim_start else None,
                    "end_date": sim_end.isoformat() if sim_end else None,
                    "assets": sim_assets if sim_assets else None,
                    "confidence_threshold": sim_confidence,
                    "capital":  sim_capital,
                    "risk_pct": sim_risk_pct,
                    "leverage": sim_leverage,
                    "eur_usd":  sim_eur_usd,
                },
                timeout=10.0,
            )
            data = resp.json()
            if data.get("status") in ("started", "already_running"):
                st.session_state.sim_running = True
        except Exception as exc:
            st.error(f"API-Fehler: {exc}")

    # ── Live progress loop (kein Page-Reload, wie Batch-Backtest) ─────────
    if st.session_state.sim_running:
        st.markdown("---")
        st.markdown("#### Simulation laeuft...")
        progress_bar  = st.progress(0.0, text="Startet...")
        metrics_area  = st.empty()

        while st.session_state.sim_running:
            try:
                _resp = httpx.get(f"{BOT_API_URL}/api/timeline-sim/progress", timeout=3.0)
                _prog = _resp.json()
            except Exception:
                _prog = {}

            running = _prog.get("running", False)
            p       = _prog.get("progress", {})
            current = p.get("current_minute", 0)
            total   = p.get("total_minutes", 1) or 1
            pct     = p.get("pct", 0.0)

            progress_bar.progress(
                min(pct / 100.0, 1.0),
                text=f"{pct:.1f}% | Minute {current:,} / {total:,}",
            )

            with metrics_area.container():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Minute",              f"{current:,} / {total:,}")
                c2.metric("Fortschritt",          f"{pct:.1f}%")
                c3.metric("Offene Trades",        p.get("open_trades", 0))
                c4.metric("Geschlossene Trades",  p.get("closed_trades", 0))
                if sim_fin_enabled and sim_capital:
                    cur_cap = p.get("current_capital")
                    if cur_cap is not None:
                        cap_delta = cur_cap - sim_capital
                        st.metric("Aktuelles Kapital", f"€{cur_cap:,.2f}",
                                  f"{cap_delta:+.2f} EUR")

            if not running:
                result = _prog.get("result")
                if result:
                    st.session_state.sim_result = result
                st.session_state.sim_running = False
                progress_bar.progress(1.0, text="Fertig!")
                break

            time.sleep(2)

    # ── Display simulation results ─────────────────────────────────────────
    sim_result = st.session_state.get("sim_result")
    if sim_result and "error" not in sim_result:
        st.markdown("---")
        st.markdown("#### Ergebnis")

        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        col_r1.metric("TRADES",       sim_result.get("trades", 0))
        col_r2.metric(
            "WIN RATE",
            f"{sim_result.get('win_rate', 0):.1f}%",
            f"{sim_result.get('wins', 0)}W / {sim_result.get('losses', 0)}L",
        )
        col_r3.metric("P/L (Punkte)", f"{sim_result.get('total_pnl_points', 0):+.4f}")
        col_r4.metric("Avg R-Mult",   f"{sim_result.get('avg_r_multiple', 0):+.3f}")

        st.caption(
            f"Zeitraum: {sim_result.get('start_ts', '')[:16]} bis "
            f"{sim_result.get('end_ts', '')[:16]} "
            f"({sim_result.get('total_minutes', 0):,} Minuten simuliert)"
        )

        # Financial summary
        fin = sim_result.get("financial", {})
        if fin:
            st.markdown("##### Finanzrechnung")
            fc1, fc2, fc3, fc4 = st.columns(4)
            start_cap = fin.get("start_capital", 0)
            end_cap   = fin.get("end_capital", 0)
            ret_pct   = fin.get("total_return_pct", 0)
            dd_pct    = fin.get("max_drawdown_pct", 0)
            fc1.metric("Startkapital",  f"€{start_cap:,.2f}")
            fc2.metric("Endkapital",    f"€{end_cap:,.2f}", f"{ret_pct:+.2f}%")
            fc3.metric("Max Drawdown",  f"{dd_pct:.2f}%")
            fc4.metric("Rendite",       f"{ret_pct:+.2f}%",
                       "MARGIN CALL" if fin.get("margin_call") else "OK")
            if fin.get("margin_call"):
                st.error("MARGIN CALL — Simulation wurde vorzeitig beendet (Kapital aufgebraucht)")

            # Equity curve in EUR
            eq_curve = fin.get("equity_curve", [])
            if len(eq_curve) > 1:
                eq_df = pd.DataFrame({"Kapital (EUR)": eq_curve})
                eq_df.index.name = "Trade #"
                st.markdown("##### Equity-Kurve (EUR)")
                st.line_chart(eq_df)

        st.info("Trades gespeichert in simLastCharts.db → nutzbar fuer TradeAI Training (`--db history`)")

        # Per-asset breakdown
        per_asset = sim_result.get("per_asset", {})
        if per_asset:
            st.markdown("##### Pro Asset")
            pa_cols = st.columns(len(per_asset))
            for i, (asset, stats) in enumerate(per_asset.items()):
                with pa_cols[i]:
                    st.metric(
                        asset,
                        f"{stats.get('trades', 0)} Trades",
                        f"WR: {stats.get('win_rate', 0):.0f}% | "
                        f"P/L: {stats.get('pnl', 0):+.4f}",
                    )

        # P/L Equity-Kurve (kumulativ)
        trade_list = sim_result.get("trade_list", [])
        if trade_list:
            df_trades = pd.DataFrame(trade_list)
            df_trades["cum_pnl"] = df_trades["pnl"].cumsum()

            st.markdown("##### Equity-Kurve (kumulativer P/L in Punkten)")
            st.line_chart(df_trades[["cum_pnl"]].rename(columns={"cum_pnl": "P/L kumulativ"}))

            # Trade-Tabelle
            st.markdown("##### Alle Trades")
            drop_cols = ["cum_pnl"] + [c for c in ["lot_size", "margin_eur"] if c in df_trades.columns and not sim_fin_enabled]
            display_df = df_trades.drop(columns=drop_cols, errors="ignore").rename(columns={
                "asset": "Asset", "direction": "Dir",
                "entry_ts": "Entry", "exit_ts": "Exit",
                "entry_price": "Entry Preis", "exit_price": "Exit Preis",
                "pnl": "P/L (Pkt)", "r_multiple": "R-Mult",
                "status": "Status", "confidence": "Conf",
                "netto_pnl_eur": "Netto P/L (€)",
                "capital_after": "Kapital nach (€)",
                "lot_size": "Lot",
                "margin_eur": "Margin (€)",
            })
            st.dataframe(display_df, width="stretch", hide_index=True)

            csv_data = display_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "CSV Download",
                data=csv_data,
                file_name=f"timeline_sim_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                width="stretch",
            )

    elif sim_result and "error" in sim_result:
        st.error(f"Simulation fehlgeschlagen: {sim_result['error']}")
