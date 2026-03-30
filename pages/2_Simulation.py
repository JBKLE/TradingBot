"""Page 2: Simulation — Timeline-Simulation mit Live-Fortschritt."""
import time
from datetime import datetime

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard_shared import BOT_API_URL, PLOTLY_LAYOUT, apply_css

st.set_page_config(page_title="Simulation", page_icon="🔬", layout="wide")
apply_css()

st.markdown("# ◈ SIMULATION")


# ══════════════════════════════════════════════════════════════════════════════
# 1. KONFIG & START
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### ◈ TIMELINE-SIMULATION STARTEN")

# ── Modell-Auswahl ────────────────────────────────────────────────────────
try:
    _models_resp = httpx.get(f"{BOT_API_URL}/api/models", timeout=5)
    _models = _models_resp.json().get("models", []) if _models_resp.status_code == 200 else []
except Exception:
    _models = []

_cfg_col1, _cfg_col2 = st.columns(2)
with _cfg_col1:
    if _models:
        _model_options = [m.get("name", m.get("path", "?")) for m in _models]
        _model_choice = st.selectbox("Modell:", _model_options, key="sim_model")
        _model_path = _models[_model_options.index(_model_choice)].get("path", "")
    else:
        _model_path = st.text_input("Modell-Pfad:", key="sim_model_path")

with _cfg_col2:
    # DB-Auswahl fuer Ausgabe
    try:
        _db_resp = httpx.get(f"{BOT_API_URL}/api/db-viewer/databases", timeout=5)
        _dbs = _db_resp.json().get("databases", []) if _db_resp.status_code == 200 else []
        _db_names = [d["name"] for d in _dbs if "price_history" in (d.get("tables") or {})]
    except Exception:
        _db_names = []

    if _db_names:
        _output_db = st.selectbox("Ausgabe-DB:", _db_names, key="sim_output_db")
    else:
        _output_db = st.text_input("Ausgabe-DB:", value="simLastCharts.db", key="sim_output_db_manual")

# ── Asset & Zeitraum ──────────────────────────────────────────────────────
_ac1, _ac2, _ac3, _ac4 = st.columns(4)
with _ac1:
    _assets_all = ["GOLD", "SILVER", "OIL_CRUDE", "NATURALGAS"]
    _sim_assets = st.multiselect("Assets:", _assets_all, default=_assets_all, key="sim_assets")
with _ac2:
    _sim_start = st.date_input("Von:", value=None, key="sim_start")
with _ac3:
    _sim_end = st.date_input("Bis:", value=None, key="sim_end")
with _ac4:
    _sim_conf = st.slider("Min. Confidence:", 1, 10, 7, key="sim_conf")

# ── Finanz-Einstellungen ─────────────────────────────────────────────────
sim_fin_enabled = st.checkbox("Finanzrechnung aktivieren", key="sim_fin_enabled")
if sim_fin_enabled:
    _fc1, _fc2, _fc3, _fc4 = st.columns(4)
    with _fc1:
        _sim_capital = st.number_input("Startkapital (€):", value=1000.0, min_value=100.0,
                                       step=100.0, key="sim_capital")
    with _fc2:
        _sim_risk = st.number_input("Risiko (%):", value=1.0, min_value=0.1, max_value=10.0,
                                     step=0.1, key="sim_risk") / 100.0
    with _fc3:
        _sim_leverage = st.number_input("Hebel:", value=20, min_value=1, max_value=100,
                                         step=1, key="sim_leverage")
    with _fc4:
        _sim_eur_usd = st.number_input("EUR/USD:", value=1.08, min_value=0.80, max_value=1.50,
                                        step=0.01, key="sim_eur_usd")
else:
    _sim_capital, _sim_risk, _sim_leverage, _sim_eur_usd = 1000.0, 0.01, 20, 1.08

# ── Start-Button ─────────────────────────────────────────────────────────
if st.button("◈ SIMULATION STARTEN", type="primary", use_container_width=True):
    _payload = {
        "model_path": _model_path,
        "output_db": _output_db,
        "assets": _sim_assets,
        "confidence_threshold": _sim_conf,
    }
    if _sim_start:
        _payload["start_date"] = str(_sim_start)
    if _sim_end:
        _payload["end_date"] = str(_sim_end)
    if sim_fin_enabled:
        _payload["capital"] = _sim_capital
        _payload["risk_pct"] = _sim_risk
        _payload["leverage"] = _sim_leverage
        _payload["eur_usd"] = _sim_eur_usd

    try:
        r = httpx.post(f"{BOT_API_URL}/api/simulation/start", json=_payload, timeout=10)
        if r.status_code == 200:
            st.session_state.sim_running = True
            st.session_state.sim_result = None
            st.rerun()
        else:
            st.error(f"Start fehlgeschlagen: {r.text[:300]}")
    except Exception as e:
        st.error(f"API-Fehler: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. LIVE-FORTSCHRITT
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.get("sim_running"):
    st.markdown("---")
    st.markdown("### ◈ LAUFENDE SIMULATION")

    progress_bar = st.progress(0, text="Simulation läuft...")
    kpi_area = st.empty()
    equity_live_area = st.empty()
    open_trades_area = st.empty()
    closed_trades_area = st.empty()

    eq_snap: list[float] = []

    while True:
        try:
            _prog = httpx.get(f"{BOT_API_URL}/api/simulation/progress", timeout=5).json()
        except Exception:
            time.sleep(2)
            continue

        running = _prog.get("running", False)
        pct = _prog.get("progress", 0.0)
        progress_bar.progress(min(pct, 1.0), text=f"Fortschritt: {pct*100:.1f}%")

        # KPIs
        with kpi_area.container():
            _kc1, _kc2, _kc3, _kc4 = st.columns(4)
            _kc1.metric("Trades", _prog.get("trades", 0))
            _kc2.metric("Win-Rate", f"{_prog.get('win_rate', 0):.1f}%")
            _kc3.metric("P/L (Pkt)", f"{_prog.get('total_pnl', 0):+.4f}")
            _fin_prog = _prog.get("financial", {})
            if _fin_prog:
                _kc4.metric("Kapital", f"€{_fin_prog.get('current_capital', 0):,.2f}")
                if _fin_prog.get("current_capital"):
                    eq_snap.append(_fin_prog["current_capital"])

        # Live equity curve
        if sim_fin_enabled and len(eq_snap) > 1:
            with equity_live_area.container():
                st.markdown("##### Equity-Kurve (live)")
                eq_df = pd.DataFrame({"Kapital (EUR)": eq_snap})
                eq_df.index.name = "Trade #"
                st.line_chart(eq_df)

        # Open trades snapshot
        open_snap = _prog.get("open_trades_snap", [])
        with open_trades_area.container():
            st.markdown(f"##### Offene Trades ({len(open_snap)})")
            if open_snap:
                open_df = pd.DataFrame([{
                    "Asset":       t.get("asset", ""),
                    "Dir":         t.get("direction", ""),
                    "Conf":        t.get("confidence", 0),
                    "Entry":       str(t.get("entry_timestamp", ""))[:16],
                    "Entry Preis": t.get("entry_price", 0),
                    "SL":          round(t.get("sl_price", 0), 5),
                    "TP":          round(t.get("tp_price", 0), 5),
                } for t in open_snap])
                st.dataframe(open_df, hide_index=True, use_container_width=True)
            else:
                st.caption("Keine offenen Trades")

        # Closed trades snapshot
        closed_snap = _prog.get("closed_trades_snap", [])
        with closed_trades_area.container():
            st.markdown(f"##### Abgeschlossene Trades ({_prog.get('closed_trades', 0)})")
            if closed_snap:
                closed_rows = []
                for t in reversed(closed_snap):
                    row = {
                        "Asset":       t.get("asset", ""),
                        "Dir":         t.get("direction", ""),
                        "Conf":        t.get("confidence", 0),
                        "Status":      t.get("status", ""),
                        "Entry":       str(t.get("entry_timestamp", ""))[:16],
                        "Exit":        str(t.get("exit_timestamp", ""))[:16],
                        "P/L (Pkt)":  round(t.get("pnl") or 0, 4),
                        "R-Mult":     round(t.get("r_multiple") or 0, 3),
                    }
                    if t.get("capital_after") is not None:
                        row["Kapital (€)"] = round(t["capital_after"], 2)
                    closed_rows.append(row)
                closed_df = pd.DataFrame(closed_rows)

                def _style_status(val):
                    if val == "closed_tp":
                        return "color: #39ff14"
                    if val == "closed_sl":
                        return "color: #ff4444"
                    return ""

                styled = closed_df.style.applymap(_style_status, subset=["Status"])
                st.dataframe(styled, hide_index=True, use_container_width=True)
            else:
                st.caption("Noch keine abgeschlossenen Trades")

        if not running:
            result = _prog.get("result")
            if result:
                st.session_state.sim_result = result
            st.session_state.sim_running = False
            progress_bar.progress(1.0, text="Fertig!")
            break

        time.sleep(2)


# ══════════════════════════════════════════════════════════════════════════════
# 3. ERGEBNIS-ANZEIGE
# ══════════════════════════════════════════════════════════════════════════════

sim_result = st.session_state.get("sim_result")
if sim_result and "error" not in sim_result:
    st.markdown("---")

    # Banner wenn aus Historie geladen
    _hmeta = sim_result.get("_history_meta")
    if _hmeta:
        _h_assets = ", ".join(_hmeta.get("assets") or []) or "alle Assets"
        st.info(
            f"Geladen aus Historie — "
            f"**Run #{_hmeta['run_id']}** | "
            f"{str(_hmeta.get('run_at',''))[:16]} | "
            f"Modell: `{_hmeta.get('model_name','—')}` | "
            f"DB: `{_hmeta.get('output_db','—')}` | "
            f"Assets: {_h_assets} | "
            f"Conf ≥ {_hmeta.get('confidence','—')}"
        )
        if st.button("Ergebnis verwerfen", key="clear_history_result"):
            st.session_state.pop("sim_result", None)
            st.rerun()

    st.markdown("#### Ergebnis")

    # ── KI-Analyse ────────────────────────────────────────────────────────
    _ai_col1, _ai_col2 = st.columns([1, 4])
    with _ai_col1:
        _run_analysis = st.button(
            "KI-Analyse starten",
            key="run_sim_analysis",
            type="primary",
            help="Claude Opus 4.6 analysiert diesen Run im Vergleich mit der Historie",
        )
    with _ai_col2:
        if st.session_state.get("sim_analysis"):
            if st.button("Analyse verwerfen", key="clear_sim_analysis"):
                st.session_state.pop("sim_analysis", None)
                st.rerun()

    if _run_analysis:
        st.session_state.pop("sim_analysis", None)
        _analysis_box = st.empty()
        _analysis_text = ""
        try:
            with httpx.stream(
                "POST",
                f"{BOT_API_URL}/api/sim-analysis",
                json={"current_result": sim_result, "history_limit": 10},
                timeout=120.0,
            ) as _stream:
                for _chunk in _stream.iter_text():
                    _analysis_text += _chunk
                    _analysis_box.markdown(_analysis_text + " ▌")
            _analysis_box.markdown(_analysis_text)
            st.session_state["sim_analysis"] = _analysis_text
        except Exception as _ae:
            st.error(f"Analyse-Fehler: {_ae}")

    if st.session_state.get("sim_analysis") and not _run_analysis:
        st.markdown("##### KI-Analyse")
        st.markdown(st.session_state["sim_analysis"])

    st.markdown("---")

    # ── Ergebnis-KPIs ─────────────────────────────────────────────────────
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    col_r1.metric("TRADES", sim_result.get("trades", 0))
    col_r2.metric(
        "WIN RATE",
        f"{sim_result.get('win_rate', 0):.1f}%",
        f"{sim_result.get('wins', 0)}W / {sim_result.get('losses', 0)}L",
    )
    col_r3.metric("P/L (Punkte)", f"{sim_result.get('total_pnl_points', 0):+.4f}")
    col_r4.metric("Avg R-Mult", f"{sim_result.get('avg_r_multiple', 0):+.3f}")

    st.caption(
        f"Zeitraum: {sim_result.get('start_ts', '')[:16]} bis "
        f"{sim_result.get('end_ts', '')[:16]} "
        f"({sim_result.get('total_minutes', 0):,} Minuten simuliert)"
    )

    # ── Financial summary ─────────────────────────────────────────────────
    fin = sim_result.get("financial", {})
    if fin:
        st.markdown("##### Finanzrechnung")
        fc1, fc2, fc3, fc4 = st.columns(4)
        start_cap = fin.get("start_capital", 0)
        end_cap = fin.get("end_capital", 0)
        ret_pct = fin.get("total_return_pct", 0)
        dd_pct = fin.get("max_drawdown_pct", 0)
        fc1.metric("Startkapital", f"€{start_cap:,.2f}")
        fc2.metric("Endkapital", f"€{end_cap:,.2f}", f"{ret_pct:+.2f}%")
        fc3.metric("Max Drawdown", f"{dd_pct:.2f}%")
        fc4.metric("Rendite", f"{ret_pct:+.2f}%",
                   "MARGIN CALL" if fin.get("margin_call") else "OK")
        if fin.get("margin_call"):
            st.error("MARGIN CALL — Simulation wurde vorzeitig beendet (Kapital aufgebraucht)")

        # Equity curve EUR
        eq_curve = fin.get("equity_curve", [])
        _tl_for_eur = sim_result.get("trade_list", [])
        if len(eq_curve) > 1 or _tl_for_eur:
            st.markdown("##### Equity-Kurve (EUR)")
            _tl_eur_sorted = sorted(_tl_for_eur, key=lambda t: t.get("exit_ts") or "")
            _all_assets_eur = sorted({t["asset"] for t in _tl_eur_sorted}) if _tl_eur_sorted else []
            _eur_cum2: dict[str, float] = {}
            _per_asset_eur: list[dict] = []
            for _t in _tl_eur_sorted:
                _a = _t["asset"]
                _v = _t.get("netto_pnl_eur") or 0.0
                _eur_cum2[_a] = _eur_cum2.get(_a, 0.0) + _v
                _per_asset_eur.append(dict(_eur_cum2))

            if _per_asset_eur and eq_curve:
                _combined: list[dict] = []
                for i, cap in enumerate(eq_curve):
                    row: dict = {"Gesamt (Kapital)": cap}
                    if i < len(_per_asset_eur):
                        for _a2 in _all_assets_eur:
                            row[_a2] = _per_asset_eur[i].get(_a2, None)
                    _combined.append(row)
                eq_combined_df = pd.DataFrame(_combined)
                eq_combined_df.index.name = "Trade #"
                st.line_chart(eq_combined_df)
            elif eq_curve:
                eq_df = pd.DataFrame({"Gesamt (Kapital)": eq_curve})
                eq_df.index.name = "Trade #"
                st.line_chart(eq_df)

    st.info("Trades gespeichert in simLastCharts.db -> nutzbar fuer TradeAI Training (`--db history`)")

    # ── Per-asset breakdown ───────────────────────────────────────────────
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

    # ── P/L Equity curve (points) ─────────────────────────────────────────
    trade_list = sim_result.get("trade_list", [])
    if trade_list:
        df_trades = pd.DataFrame(trade_list)
        df_trades["cum_pnl"] = df_trades["pnl"].cumsum()

        st.markdown("##### Equity-Kurve (kumulativer P/L in Punkten)")
        _tl_sorted = sorted(trade_list, key=lambda t: t.get("exit_ts") or "")
        _all_assets_pnl = sorted({t["asset"] for t in _tl_sorted})
        _cum_pnl: dict[str, float] = {a: 0.0 for a in _all_assets_pnl}
        _cum_total = 0.0
        _pnl_curve_rows: list[dict] = []
        for _t in _tl_sorted:
            _a = _t["asset"]
            _p = _t.get("pnl") or 0.0
            _cum_pnl[_a] += _p
            _cum_total += _p
            row = {"Gesamt": round(_cum_total, 4)}
            row.update({a: round(_cum_pnl[a], 4) for a in _all_assets_pnl})
            _pnl_curve_rows.append(row)
        if _pnl_curve_rows:
            pnl_curve_df = pd.DataFrame(_pnl_curve_rows)
            pnl_curve_df.index.name = "Trade #"
            st.line_chart(pnl_curve_df)

        # Trade table + CSV download
        st.markdown("##### Alle Trades")
        drop_cols = ["cum_pnl"] + [
            c for c in ["lot_size", "margin_eur"]
            if c in df_trades.columns and not sim_fin_enabled
        ]
        display_df = df_trades.drop(columns=drop_cols, errors="ignore").rename(columns={
            "asset": "Asset", "direction": "Dir",
            "entry_ts": "Entry", "exit_ts": "Exit",
            "entry_price": "Entry Preis", "exit_price": "Exit Preis",
            "pnl": "P/L (Pkt)", "r_multiple": "R-Mult",
            "status": "Status", "confidence": "Conf",
            "netto_pnl_eur": "Netto P/L (€)",
            "capital_after": "Kapital nach (€)",
            "lot_size": "Lot", "margin_eur": "Margin (€)",
        })
        st.dataframe(display_df, width="stretch", hide_index=True)

        csv_data = display_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "CSV Download",
            data=csv_data,
            file_name=f"timeline_sim_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

elif sim_result and "error" in sim_result:
    st.error(f"Simulation fehlgeschlagen: {sim_result['error']}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. SIMULATIONSHISTORIE
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("### ◈ SIMULATIONSHISTORIE")


def _load_sim_history():
    try:
        r = httpx.get(f"{BOT_API_URL}/api/sim-history?limit=30", timeout=5.0)
        return r.json().get("runs", [])
    except Exception:
        return []


if st.button("Historie aktualisieren", key="sim_history_refresh"):
    st.session_state["sim_history"] = _load_sim_history()
if "sim_history" not in st.session_state:
    st.session_state["sim_history"] = _load_sim_history()

_runs = st.session_state.get("sim_history", [])
if not _runs:
    st.info("Noch keine Simulation-Runs protokolliert.")
else:
    # Compact overview table
    _hist_rows = []
    for _r in _runs:
        _assets_str = ", ".join(_r.get("assets") or []) or "alle"
        _fin_str = (
            f"€{_r['start_capital']:,.0f} -> €{_r['end_capital']:,.0f} "
            f"({_r['total_return_pct']:+.1f}%)"
            if _r.get("start_capital") is not None
            else "—"
        )
        _mc = " MARGIN CALL" if _r.get("margin_call") else ""
        _status_icon = {"completed": "OK", "cancelled": "STOP", "error": "ERR"}.get(
            _r.get("status", ""), "?"
        )
        _hist_rows.append({
            "#": _r["id"],
            "Status": f"{_status_icon} {_r.get('status', '')}",
            "Start": str(_r.get("run_at", ""))[:16],
            "Dauer (s)": _r.get("duration_sec", ""),
            "Modell": _r.get("model_name", ""),
            "Assets": _assets_str,
            "Conf": _r.get("confidence_threshold", ""),
            "Output-DB": _r.get("output_db", ""),
            "Trades": _r.get("trades", 0),
            "WR %": f"{_r.get('win_rate', 0):.1f}",
            "P/L (Pkt)": f"{_r.get('total_pnl_points', 0):+.4f}",
            "Avg R": f"{_r.get('avg_r_multiple', 0):+.3f}",
            "Finanzen": f"{_fin_str}{_mc}",
        })
    hist_df = pd.DataFrame(_hist_rows)
    st.dataframe(hist_df, hide_index=True, use_container_width=True)

    # Detail expanders
    for _r in _runs:
        _label = (
            f"#{_r['id']} | {str(_r.get('run_at',''))[:16]} | "
            f"{_r.get('model_name','')} | "
            f"{', '.join(_r.get('assets') or []) or 'alle Assets'} | "
            f"{_r.get('trades', 0)} Trades"
        )
        with st.expander(_label, expanded=False):
            dc1, dc2, dc3 = st.columns(3)
            with dc1:
                st.markdown("**Modell**")
                st.code(_r.get("model_path", "—"))
                st.caption(f"Geaendert: {_r.get('model_modified_at','—')}")
            with dc2:
                st.markdown("**Einstellungen**")
                st.markdown(
                    f"- Assets: `{', '.join(_r.get('assets') or []) or 'alle'}`\n"
                    f"- Zeitraum: `{_r.get('start_date','—')}` – `{_r.get('end_date','—')}`\n"
                    f"- Confidence: `{_r.get('confidence_threshold','—')}`\n"
                    f"- Output-DB: `{_r.get('output_db','—')}`\n"
                    f"- Kapital: `{_r.get('capital','—')}` EUR | "
                    f"Risiko: `{(_r.get('risk_pct') or 0)*100:.1f}%` | "
                    f"Hebel: `{_r.get('leverage','—')}`"
                )
            with dc3:
                st.markdown("**Ergebnisse**")
                st.markdown(
                    f"- Trades: `{_r.get('trades',0)}` "
                    f"({_r.get('wins',0)}W / {_r.get('losses',0)}L)\n"
                    f"- Win-Rate: `{_r.get('win_rate',0):.1f}%`\n"
                    f"- P/L: `{_r.get('total_pnl_points',0):+.4f}` Pkt\n"
                    f"- Avg R: `{_r.get('avg_r_multiple',0):+.4f}`\n"
                    f"- Dauer: `{_r.get('duration_sec','—')}s`"
                )
                if _r.get("start_capital") is not None:
                    st.markdown(
                        f"- Startkapital: `€{_r['start_capital']:,.2f}`\n"
                        f"- Endkapital: `€{_r.get('end_capital',0):,.2f}`\n"
                        f"- Rendite: `{_r.get('total_return_pct',0):+.2f}%`\n"
                        f"- Max DD: `{_r.get('max_drawdown_pct',0):.2f}%`"
                    )
                    if _r.get("margin_call"):
                        st.error("MARGIN CALL")

            # Per-asset breakdown
            _pa = _r.get("per_asset") or {}
            if _pa:
                st.markdown("**Pro Asset**")
                pa_cols = st.columns(len(_pa))
                for _i, (_asset, _stats) in enumerate(_pa.items()):
                    with pa_cols[_i]:
                        st.metric(
                            _asset,
                            f"{_stats.get('trades', 0)} Trades",
                            f"WR: {_stats.get('win_rate', 0):.0f}% | "
                            f"P/L: {_stats.get('pnl', 0):+.4f}",
                        )
            if _r.get("error_message"):
                st.error(f"Fehler: {_r['error_message']}")

            # Load button
            if _r.get("status") == "completed" and _r.get("trades", 0) > 0:
                if st.button(
                    "Ergebnis laden & anzeigen",
                    key=f"load_run_{_r['id']}",
                    type="primary",
                ):
                    try:
                        _load_resp = httpx.get(
                            f"{BOT_API_URL}/api/sim-history/{_r['id']}/load",
                            timeout=15.0,
                        )
                        _loaded = _load_resp.json()
                        if "error" in _loaded:
                            st.error(f"Fehler: {_loaded['error']}")
                        else:
                            st.session_state["sim_result"] = _loaded
                            st.success("Geladen! Ergebnis wird oben angezeigt.")
                            st.rerun()
                    except Exception as _e:
                        st.error(f"API-Fehler: {_e}")
