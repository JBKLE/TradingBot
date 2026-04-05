"""Page 1: Trading — Bot Status, Market Analysis, Open/Closed Trades."""
import time
from datetime import datetime

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard_shared import (
    BOT_API_URL, apply_css, load_log_lines, load_snapshots, load_trades,
    PLOTLY_LAYOUT,
)

st.set_page_config(page_title="Trading", page_icon="📈", layout="wide")
apply_css()

st.markdown("# ◈ TRADING")


# ══════════════════════════════════════════════════════════════════════════════
# BOT ON/OFF SWITCH
# ══════════════════════════════════════════════════════════════════════════════

def _get_bot_status() -> dict:
    try:
        r = httpx.get(f"{BOT_API_URL}/api/bot/status", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {"running": False, "error": True}


bot_status = _get_bot_status()
_is_running = bot_status.get("running", False)
_status_error = bot_status.get("error", False)

if _status_error:
    st.warning("Bot-API nicht erreichbar — Status unbekannt")
else:
    _sw_col1, _sw_col2 = st.columns([1, 3])
    with _sw_col1:
        if _is_running:
            st.markdown(
                '<div class="bot-status-on">'
                '<span style="font-size:1.4rem;color:#00ff41;">&#9673; BOT AKTIV</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="bot-status-off">'
                '<span style="font-size:1.4rem;color:#ff4444;">&#9675; BOT GESTOPPT</span>'
                '</div>',
                unsafe_allow_html=True,
            )

    with _sw_col2:
        _toggle = st.toggle(
            "Bot aktiv",
            value=_is_running,
            key="bot_toggle",
        )
        if _toggle != _is_running:
            endpoint = "/api/bot/start" if _toggle else "/api/bot/stop"
            try:
                r = httpx.post(f"{BOT_API_URL}{endpoint}", timeout=10)
                if r.status_code == 200:
                    st.success("Bot gestartet!" if _toggle else "Bot gestoppt!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(f"Fehler: {r.status_code}")
            except Exception as e:
                st.error(f"API nicht erreichbar: {e}")

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# MODELL-AUSWAHL
# ══════════════════════════════════════════════════════════════════════════════

with st.expander("◈ DQN-MODELL", expanded=False):
    # Verfuegbare Modelle laden
    _models_data = {}
    try:
        _mr = httpx.get(f"{BOT_API_URL}/api/models", timeout=5)
        if _mr.status_code == 200:
            _models_data = _mr.json()
    except Exception:
        pass

    _model_list = _models_data.get("models", [])
    _versions_info = _models_data.get("versions", {})
    _assets_list = _models_data.get("assets", [])

    if not _model_list:
        st.warning("Keine Modelle gefunden.")
    else:
        # Aktuelle Modellinfo laden
        _current_info = {}
        try:
            _ci = httpx.get(f"{BOT_API_URL}/api/models/current", timeout=5)
            if _ci.status_code == 200:
                _current_info = _ci.json()
        except Exception:
            pass

        _current_file = _current_info.get("model_file", "")
        _current_version = _current_info.get("version", "?")
        _current_asset = _current_info.get("asset", "?")

        st.caption(
            f"Aktuell: **{_current_file}** "
            f"(Version: `{_current_version}`, Asset: `{_current_asset}`)"
        )

        # Modell-Dropdown
        _model_names = [m["filename"] for m in _model_list]
        _default_idx = _model_names.index(_current_file) if _current_file in _model_names else 0
        _selected_model = st.selectbox(
            "Modell:", _model_names,
            index=_default_idx,
            key="model_select",
        )

        # Info zum ausgewaehlten Modell
        _sel_info = next((m for m in _model_list if m["filename"] == _selected_model), {})
        _auto_version = _sel_info.get("version")
        _auto_asset = _sel_info.get("asset")
        _parsed = _sel_info.get("parsed", False)

        _mc1, _mc2 = st.columns(2)

        with _mc1:
            if _parsed and _auto_version:
                st.success(f"Auto-erkannt: Version `{_auto_version}`")
                _version_options = list(_versions_info.keys())
                _ver_idx = _version_options.index(_auto_version) if _auto_version in _version_options else 0
                _sel_version = st.selectbox(
                    "Version:", _version_options,
                    index=_ver_idx,
                    key="model_version",
                    help="Automatisch erkannt. Manuell änderbar falls nötig.",
                )
            else:
                st.warning("Version nicht erkannt — bitte manuell wählen")
                _version_options = list(_versions_info.keys())
                _sel_version = st.selectbox(
                    "Version:", _version_options,
                    key="model_version",
                )

        with _mc2:
            if _parsed and _auto_asset:
                st.success(f"Auto-erkannt: Asset `{_auto_asset}`")
                _asset_idx = _assets_list.index(_auto_asset) if _auto_asset in _assets_list else 0
                _sel_asset = st.selectbox(
                    "Asset:", _assets_list,
                    index=_asset_idx,
                    key="model_asset",
                    help="Automatisch erkannt. Manuell änderbar falls nötig.",
                )
            else:
                st.warning("Asset nicht erkannt — bitte manuell wählen")
                _sel_asset = st.selectbox(
                    "Asset:", _assets_list,
                    key="model_asset",
                )

        # Version-Details anzeigen
        _v_info = _versions_info.get(_sel_version, {})
        if _v_info:
            _dc1, _dc2, _dc3, _dc4 = st.columns(4)
            _dc1.metric("Candles", _v_info.get("max_window", "?"))
            _dc2.metric("Indikatoren", _v_info.get("n_indicators", "?"))
            _dc3.metric("Actions", _v_info.get("action_size", "?"))
            _dc4.metric("State-Size", _v_info.get("state_size", "?"))
            st.caption(
                f"Actions: {_v_info.get('actions', {})} · "
                f"SL: {_v_info.get('sl_pct', 0)*100:.1f}% · "
                f"TP: {_v_info.get('tp_pct', 0)*100:.1f}%"
            )

        # Aktivieren-Button
        _needs_override = not _parsed or _sel_version != _auto_version or _sel_asset != _auto_asset
        if st.button("◈ MODELL AKTIVIEREN", type="primary", use_container_width=True):
            with st.spinner("Modell wird geladen..."):
                try:
                    _payload = {"filename": _selected_model}
                    if _needs_override or not _parsed:
                        _payload["version"] = _sel_version
                        _payload["asset"] = _sel_asset
                    _resp = httpx.post(
                        f"{BOT_API_URL}/api/models/select",
                        json=_payload, timeout=30,
                    )
                    if _resp.status_code == 200:
                        _result = _resp.json()
                        st.success(
                            f"Modell aktiviert: {_result.get('model_file')} "
                            f"(v{_result.get('version', '?')}, {_result.get('asset', '?')})"
                        )
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"Fehler: {_resp.text}")
                except Exception as e:
                    st.error(f"API-Fehler: {e}")

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# BOT-EINSTELLUNGEN
# ══════════════════════════════════════════════════════════════════════════════

with st.expander("◈ BOT-EINSTELLUNGEN", expanded=False):
    # Aktuelle Settings laden
    _cur_settings = {}
    try:
        _sr = httpx.get(f"{BOT_API_URL}/api/bot/settings", timeout=5)
        if _sr.status_code == 200:
            _cur_settings = _sr.json()
    except Exception:
        pass

    _sc1, _sc2, _sc3, _sc4 = st.columns(4)
    with _sc1:
        _set_conf = st.slider(
            "Min. Confidence (Entry):",
            1, 10, int(_cur_settings.get("min_confidence", 8)),
            key="set_min_conf",
        )
    with _sc2:
        _set_close_conf = st.slider(
            "Min. Close-Confidence:",
            1, 10, int(_cur_settings.get("min_close_confidence", 1)),
            key="set_min_close_conf",
        )
    with _sc3:
        _cur_assets = _cur_settings.get("assets", ["GOLD", "SILVER", "OIL_CRUDE", "NATURALGAS"])
        _set_assets = st.multiselect(
            "Aktive Assets:",
            ["GOLD", "SILVER", "OIL_CRUDE", "NATURALGAS"],
            default=_cur_assets,
            key="set_assets",
        )
    with _sc4:
        _set_risk = st.number_input(
            "Risiko (%):", value=float(_cur_settings.get("risk_pct", 1.0)),
            min_value=0.1, max_value=10.0, step=0.1, key="set_risk",
        )

    if st.button("◈ EINSTELLUNGEN SPEICHERN", use_container_width=True):
        try:
            _save_resp = httpx.post(
                f"{BOT_API_URL}/api/bot/settings",
                json={
                    "min_confidence": _set_conf,
                    "min_close_confidence": _set_close_conf,
                    "assets": _set_assets,
                    "risk_pct": _set_risk,
                },
                timeout=5,
            )
            if _save_resp.status_code == 200:
                st.success("Einstellungen gespeichert!")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error(f"Fehler: {_save_resp.status_code}")
        except Exception as _e:
            st.error(f"API-Fehler: {_e}")

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# KPI ROW
# ══════════════════════════════════════════════════════════════════════════════

trades = load_trades()
snapshots = load_snapshots()

open_trades = trades[trades["status"] == "OPEN"] if not trades.empty else pd.DataFrame()
closed_trades = (
    trades[trades["status"].isin(["TAKE_PROFIT", "STOPPED_OUT", "CLOSED"])]
    if not trades.empty else pd.DataFrame()
)

latest_balance = snapshots["balance"].iloc[-1] if not snapshots.empty else 0.0
prev_balance = snapshots["balance"].iloc[-2] if len(snapshots) > 1 else latest_balance
total_pl = closed_trades["profit_loss"].sum() if not closed_trades.empty else 0.0
win_trades = (
    closed_trades[closed_trades["profit_loss"] > 0]
    if not closed_trades.empty else pd.DataFrame()
)
win_rate = len(win_trades) / len(closed_trades) * 100 if not closed_trades.empty else 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("KONTOSTAND", f"€{latest_balance:.2f}", f"{latest_balance - prev_balance:+.2f}")
col2.metric("OFFENE POSITIONEN", len(open_trades))
col3.metric("GESAMT P/L", f"€{total_pl:+.2f}")
col4.metric("WIN RATE", f"{win_rate:.0f}%", f"{len(win_trades)}/{len(closed_trades)} Trades")

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# MARKT JETZT ANALYSIEREN
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### ◈ MARKT-ANALYSE")

if st.button("◈ MARKT JETZT ANALYSIEREN", type="primary", use_container_width=True):
    with st.spinner("DQN analysiert alle Assets..."):
        try:
            resp = httpx.post(f"{BOT_API_URL}/api/analyze", timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                signals = data.get("signals", [])
                st.session_state["analysis_signals"] = signals
                st.session_state["analysis_ts"] = data.get("timestamp", "")
            else:
                st.error(f"Analyse fehlgeschlagen: {resp.status_code}")
        except Exception as e:
            st.error(f"API nicht erreichbar: {e}")

# Display signals
_signals = st.session_state.get("analysis_signals")
if _signals:
    _ts = st.session_state.get("analysis_ts", "")
    st.caption(f"Letzte Analyse: {_ts}")

    cols = st.columns(min(len(_signals), 4))
    for col, sig in zip(cols, _signals):
        action = sig.get("action", "HOLD")
        conf = sig.get("confidence", 0)
        asset = sig.get("asset", "?")

        if action == "BUY":
            css_class = "signal-buy"
            icon = "▲"
            color = "#00ff41"
        elif action == "SELL":
            css_class = "signal-sell"
            icon = "▼"
            color = "#ff4444"
        else:
            css_class = "signal-hold"
            icon = "●"
            color = "#007a1f"

        # Confidence bar as HTML
        bar_width = conf * 10
        bar_color = color

        with col:
            st.markdown(
                f'<div class="signal-card {css_class}">'
                f'<div style="font-size:1.1rem;font-weight:bold;margin-bottom:8px;">{asset}</div>'
                f'<div style="font-size:1.5rem;color:{color};margin-bottom:4px;">{icon} {action}</div>'
                f'<div style="margin-bottom:8px;">Conf: {conf}/10</div>'
                f'<div style="background:#1a1a1a;border-radius:4px;height:12px;width:100%;">'
                f'<div style="background:{bar_color};height:100%;width:{bar_width}%;'
                f'border-radius:4px;transition:width 0.5s;"></div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Q-values detail
    with st.expander("Q-Werte Detail"):
        for sig in _signals:
            qv = sig.get("q_values", [])
            st.markdown(
                f"**{sig['asset']}**: {sig['action']} (Conf: {sig['confidence']}/10) "
                f"— Q: `{qv}`"
            )

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# OFFENE POSITIONEN — Card-Visualisierung
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### ◈ OFFENE POSITIONEN")

if open_trades.empty:
    st.info("Keine offenen Positionen.")
else:
    for _, t in open_trades.iterrows():
        asset = t.get("asset", "?")
        direction = t.get("direction", "?")
        entry_price = t.get("entry_price", 0)
        sl_price = t.get("stop_loss", 0)
        tp_price = t.get("take_profit", 0)
        ts = t.get("timestamp", "")

        # Calculate percentages
        sl_pct = abs(entry_price - sl_price) / entry_price * 100 if entry_price > 0 else 0
        tp_pct = abs(tp_price - entry_price) / entry_price * 100 if entry_price > 0 else 0

        # Time since open
        try:
            dt = pd.to_datetime(ts)
            delta = datetime.now() - dt.replace(tzinfo=None)
            hours = int(delta.total_seconds() // 3600)
            minutes = int((delta.total_seconds() % 3600) // 60)
            time_str = f"{hours}h {minutes}m"
        except Exception:
            time_str = "?"

        dir_icon = "▲ BUY" if direction == "BUY" else "▼ SELL"
        dir_color = "#00ff41" if direction == "BUY" else "#ff4444"

        # SL/TP visual bar
        total_range = sl_pct + tp_pct
        sl_frac = (sl_pct / total_range * 100) if total_range > 0 else 50
        tp_frac = (tp_pct / total_range * 100) if total_range > 0 else 50

        st.markdown(
            f'<div class="trade-card">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">'
            f'<span style="font-size:1.2rem;font-weight:bold;">{asset} '
            f'<span style="color:{dir_color};">{dir_icon}</span></span>'
            f'<span style="color:#007a1f;">● OFFEN seit {time_str}</span>'
            f'</div>'
            f'<div style="display:flex;gap:24px;margin-bottom:12px;">'
            f'<span>Entry: <b>{entry_price:.5f}</b></span>'
            f'<span style="color:#ff4444;">SL: {sl_price:.5f} (▼ {sl_pct:.2f}%)</span>'
            f'<span style="color:#00ff41;">TP: {tp_price:.5f} (▲ {tp_pct:.2f}%)</span>'
            f'</div>'
            # SL/TP bar
            f'<div style="display:flex;height:8px;border-radius:4px;overflow:hidden;margin-bottom:8px;">'
            f'<div style="background:#ff4444;width:{sl_frac}%;"></div>'
            f'<div style="background:#ffffff;width:3px;"></div>'
            f'<div style="background:#00ff41;width:{tp_frac}%;"></div>'
            f'</div>'
            f'<div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#007a1f;">'
            f'<span>SL ◄ {sl_pct:.2f}%</span>'
            f'<span>ENTRY</span>'
            f'<span>{tp_pct:.2f}% ► TP</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# GESCHLOSSENE TRADES — Charts + Tabelle
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### ◈ GESCHLOSSENE TRADES")

if closed_trades.empty:
    st.info("Keine abgeschlossenen Trades.")
else:
    # ── KPI Row ───────────────────────────────────────────────────────────
    best_trade = closed_trades["profit_loss"].max()
    worst_trade = closed_trades["profit_loss"].min()

    kc1, kc2, kc3, kc4, kc5 = st.columns(5)
    kc1.metric("TRADES", len(closed_trades))
    kc2.metric("WIN RATE", f"{win_rate:.1f}%")
    kc3.metric("GESAMT P/L", f"€{total_pl:+.2f}")
    kc4.metric("BESTER", f"€{best_trade:+.2f}")
    kc5.metric("SCHLECHTESTER", f"€{worst_trade:+.2f}")

    # ── Equity-Kurve ──────────────────────────────────────────────────────
    sorted_trades = closed_trades.sort_values("exit_timestamp")
    sorted_trades["cum_pl"] = sorted_trades["profit_loss"].cumsum()

    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=list(range(1, len(sorted_trades) + 1)),
        y=sorted_trades["cum_pl"].values,
        mode="lines",
        fill="tozeroy",
        line=dict(color="#00ff41", width=1.5),
        fillcolor="rgba(0,255,65,0.08)",
        name="Kum. P/L",
        hovertemplate=(
            "Trade #%{x}<br>"
            "Kum. P/L: €%{y:.2f}<extra></extra>"
        ),
    ))
    fig_equity.add_hline(y=0, line_color="#007a1f", line_dash="dash", line_width=1)
    fig_equity.update_layout(
        **PLOTLY_LAYOUT,
        height=280,
        showlegend=False,
        xaxis_title="Trade #",
        yaxis_title="Kumuliertes P/L (€)",
    )
    st.markdown("##### Equity-Kurve")
    st.plotly_chart(fig_equity, use_container_width=True)

    # ── P/L Balkendiagramm (letzte 30) ────────────────────────────────────
    last_30 = sorted_trades.tail(30)
    colors = ["#00ff41" if pl > 0 else "#ff4444" for pl in last_30["profit_loss"]]

    fig_bars = go.Figure()
    fig_bars.add_trace(go.Bar(
        x=last_30["exit_timestamp"].dt.strftime("%d.%m %H:%M"),
        y=last_30["profit_loss"],
        marker_color=colors,
        name="P/L",
        hovertemplate="P/L: €%{y:.2f}<extra></extra>",
    ))
    fig_bars.add_hline(y=0, line_color="#007a1f", line_dash="dash", line_width=1)
    fig_bars.update_layout(
        **PLOTLY_LAYOUT,
        height=250,
        showlegend=False,
        xaxis=dict(gridcolor="#001a00", showgrid=True, tickangle=-30),
        yaxis=dict(gridcolor="#001a00", showgrid=True),
    )
    st.markdown("##### P/L letzte 30 Trades")
    st.plotly_chart(fig_bars, use_container_width=True)

    # ── Gefilterte Tabelle ────────────────────────────────────────────────
    st.markdown("##### Trade-Tabelle")
    _ft_col1, _ft_col2, _ft_col3 = st.columns(3)
    with _ft_col1:
        _assets = ["Alle"] + sorted(closed_trades["asset"].dropna().unique().tolist())
        _f_asset = st.selectbox("Asset:", _assets, key="ct_asset_filter")
    with _ft_col2:
        _statuses = ["Alle", "TAKE_PROFIT", "STOPPED_OUT", "CLOSED"]
        _f_status = st.selectbox("Status:", _statuses, key="ct_status_filter")
    with _ft_col3:
        _f_limit = st.select_slider("Zeilen:", [25, 50, 100, 250], value=50, key="ct_limit")

    display = closed_trades.copy()
    if _f_asset != "Alle":
        display = display[display["asset"] == _f_asset]
    if _f_status != "Alle":
        display = display[display["status"] == _f_status]
    display = display.head(_f_limit)

    # Color status badges
    def _style_status(val):
        if val == "TAKE_PROFIT":
            return "color: #39ff14"
        if val == "STOPPED_OUT":
            return "color: #ff4444"
        return "color: #888"

    styled = display.style.applymap(_style_status, subset=["status"])
    st.dataframe(styled, hide_index=True, use_container_width=True, height=400)


# ══════════════════════════════════════════════════════════════════════════════
# LOG VIEWER
# ══════════════════════════════════════════════════════════════════════════════

with st.expander("◈ BOT LOG", expanded=False):
    log_lines = load_log_lines(100)
    log_html = ""
    for line in log_lines:
        css = "log-info"
        if "ERROR" in line:
            css = "log-error"
        elif "WARNING" in line:
            css = "log-warning"
        log_html += f'<span class="{css}">{line}</span>'
    st.markdown(f'<div class="log-box">{log_html}</div>', unsafe_allow_html=True)
