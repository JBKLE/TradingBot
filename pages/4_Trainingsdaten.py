"""Page 4: Trainingsdaten — DB-Viewer + Trainingsdaten-Manager."""
import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard_shared import BOT_API_URL, PLOTLY_LAYOUT, apply_css

st.set_page_config(page_title="Trainingsdaten", page_icon="🗄️", layout="wide")
apply_css()

st.markdown("# ◈ TRAININGSDATEN")

_subtab_viewer, _subtab_manager = st.tabs(["◈ DB-Viewer", "◈ Trainingsdaten-Manager"])


# ══════════════════════════════════════════════════════════════════════════════
# SUB-TAB: DB-VIEWER
# ══════════════════════════════════════════════════════════════════════════════

with _subtab_viewer:
    st.subheader("Datenbank-Viewer")

    try:
        _dv_resp = httpx.get(f"{BOT_API_URL}/api/db-viewer/databases", timeout=10.0)
        _dv_dbs = _dv_resp.json().get("databases", [])
    except Exception:
        _dv_dbs = []

    if not _dv_dbs:
        st.info("Keine Datenbanken gefunden.")
    else:
        # ── DB selection ──────────────────────────────────────────────────
        _dv_db_options = [f"{d['name']}  ({d['size_kb']} KB)" for d in _dv_dbs]
        _dv_db_choice = st.selectbox("Datenbank:", _dv_db_options, key="dv_db_select")
        _dv_selected = _dv_dbs[_dv_db_options.index(_dv_db_choice)]

        # ── Table overview ────────────────────────────────────────────────
        _dv_tables = _dv_selected.get("tables", {})
        if _dv_tables:
            st.caption(f"Tabellen in **{_dv_selected['name']}**:")
            _dv_tbl_cols = st.columns(min(len(_dv_tables), 4))
            for _i, (_tname, _tcnt) in enumerate(_dv_tables.items()):
                _dv_tbl_cols[_i % 4].metric(_tname, f"{_tcnt:,} Zeilen")
        else:
            st.warning("Keine Tabellen gefunden.")
            _dv_tables = {}

        if _dv_tables:
            st.divider()

            # ── Table selection ───────────────────────────────────────────
            _dv_tbl_choice = st.selectbox(
                "Tabelle:", list(_dv_tables.keys()), key="dv_tbl_select"
            )

            # ── Summary ───────────────────────────────────────────────────
            _dv_sum: dict = {}
            if _dv_tbl_choice in ("price_history", "sim_trades", "training_trades"):
                try:
                    _dv_sum_resp = httpx.get(
                        f"{BOT_API_URL}/api/db-viewer"
                        f"/{_dv_selected['name']}/summary/{_dv_tbl_choice}",
                        timeout=10.0,
                    )
                    _dv_sum = _dv_sum_resp.json()
                except Exception:
                    _dv_sum = {}

                if _dv_tbl_choice == "price_history" and "per_asset" in _dv_sum:
                    st.markdown("**Kerzen pro Asset:**")
                    _dv_ph_rows = []
                    for _pa in _dv_sum["per_asset"]:
                        _dv_ph_rows.append({
                            "Asset": _pa["asset"],
                            "Kerzen": f"{_pa['count']:,}",
                            "Von": _pa["from"],
                            "Bis": _pa["to"],
                            "Tage": f"{_pa['count'] / (60*24):.1f}",
                        })
                    st.dataframe(
                        pd.DataFrame(_dv_ph_rows),
                        hide_index=True, use_container_width=True,
                    )

                elif _dv_tbl_choice in ("sim_trades", "training_trades") \
                        and "per_asset_dir" in _dv_sum:
                    _tot = _dv_sum.get("totals", {})
                    if _tot:
                        _dv_c1, _dv_c2, _dv_c3, _dv_c4 = st.columns(4)
                        _dv_c1.metric("Trades", _tot.get("trades", 0))
                        _dv_c2.metric("Wins", _tot.get("wins", 0))
                        _dv_c3.metric("Win-Rate", f"{_tot.get('win_rate',0):.1f}%")
                        _dv_c4.metric("Avg R", f"{_tot.get('avg_r',0):+.3f}")
                    st.markdown("**Pro Asset / Richtung:**")
                    _dv_tr_rows = []
                    for _r in _dv_sum["per_asset_dir"]:
                        _dv_tr_rows.append({
                            "Asset": _r["asset"],
                            "Richtung": _r["direction"],
                            "Trades": _r["trades"],
                            "Wins": _r["wins"],
                            "Win-Rate": f"{_r['win_rate']:.1f}%",
                            "Avg R": f"{_r['avg_r']:+.3f}",
                            "Gesamt P/L": f"{_r['total_pnl']:+.4f}",
                        })
                    _dv_df = pd.DataFrame(_dv_tr_rows)
                    st.dataframe(_dv_df, hide_index=True, use_container_width=True)

            # ── Chart ─────────────────────────────────────────────────────
            if _dv_tbl_choice in ("price_history", "sim_trades", "training_trades"):
                st.divider()
                _chart_col1, _chart_col2 = st.columns([3, 1])

                _chart_asset = None
                if _dv_tbl_choice == "price_history" and "per_asset" in _dv_sum:
                    _chart_assets = [p["asset"] for p in _dv_sum["per_asset"]]
                    if _chart_assets:
                        _chart_asset = _chart_col2.selectbox(
                            "Asset (Chart):", _chart_assets, key="dv_chart_asset"
                        )
                elif _dv_tbl_choice in ("sim_trades", "training_trades") \
                        and "per_asset_dir" in _dv_sum:
                    _chart_assets = sorted({r["asset"] for r in _dv_sum["per_asset_dir"]})
                    _chart_asset = _chart_col2.selectbox(
                        "Asset (Chart):", ["alle"] + _chart_assets, key="dv_chart_asset"
                    )
                    if _chart_asset == "alle":
                        _chart_asset = None

                try:
                    _chart_params = {}
                    if _chart_asset:
                        _chart_params["asset"] = _chart_asset
                    _chart_resp = httpx.get(
                        f"{BOT_API_URL}/api/db-viewer"
                        f"/{_dv_selected['name']}/chart/{_dv_tbl_choice}",
                        params=_chart_params,
                        timeout=20.0,
                    )
                    _chart_data = _chart_resp.json()
                except Exception:
                    _chart_data = {"type": "none"}

                _ct = _chart_data.get("type")

                if _ct == "ohlc" and _chart_data.get("timestamps"):
                    _stride = _chart_data.get("stride", 1)
                    _total = _chart_data.get("total", 0)
                    with _chart_col1:
                        st.caption(
                            f"**{_chart_data['asset']}** — Close-Preis  "
                            f"({_total:,} Kerzen, jede {_stride}. Kerze dargestellt)"
                        )
                        _fig = go.Figure()
                        _fig.add_trace(go.Candlestick(
                            x=_chart_data["timestamps"],
                            open=_chart_data["open"],
                            high=_chart_data["high"],
                            low=_chart_data["low"],
                            close=_chart_data["close"],
                            name=_chart_data["asset"],
                            increasing_line_color="#00ff41",
                            decreasing_line_color="#ff4444",
                        ))
                        _fig.update_layout(
                            **PLOTLY_LAYOUT,
                            height=350,
                            xaxis=dict(
                                gridcolor="#001a00", showgrid=True,
                                rangeslider=dict(visible=False),
                            ),
                        )
                        st.plotly_chart(_fig, use_container_width=True)

                elif _ct == "trades" and _chart_data.get("timestamps"):
                    with _chart_col1:
                        st.caption("**Kumulatives PnL** ueber Zeit")
                        _cum = _chart_data["cumulative"]
                        _fig_pnl = go.Figure()
                        _fig_pnl.add_trace(go.Scatter(
                            x=_chart_data["timestamps"],
                            y=_cum,
                            mode="lines",
                            line=dict(color="#00ff41", width=1.5),
                            fill="tozeroy",
                            fillcolor="rgba(0,255,65,0.08)",
                            name="Kum. PnL",
                        ))
                        _fig_pnl.add_hline(
                            y=0, line_color="#007a1f", line_dash="dash", line_width=1
                        )
                        _fig_pnl.update_layout(**PLOTLY_LAYOUT, height=230, showlegend=False)
                        st.plotly_chart(_fig_pnl, use_container_width=True)

                    if _chart_data.get("bar_labels"):
                        with _chart_col1:
                            st.caption("**Win-Rate** pro Asset / Richtung")
                            _bar_colors = [
                                "#00ff41" if wr >= 50 else "#ff4444"
                                for wr in _chart_data["bar_wr"]
                            ]
                            _fig_bar = go.Figure()
                            _fig_bar.add_trace(go.Bar(
                                x=_chart_data["bar_labels"],
                                y=_chart_data["bar_wr"],
                                marker_color=_bar_colors,
                                text=[
                                    f"{w:.1f}%<br>({n})" for w, n in zip(
                                        _chart_data["bar_wr"], _chart_data["bar_trades"]
                                    )
                                ],
                                textposition="outside",
                                textfont=dict(color="#00ff41", size=10),
                                name="Win-Rate",
                            ))
                            _fig_bar.add_hline(
                                y=50, line_color="#ffaa00", line_dash="dash", line_width=1
                            )
                            _fig_bar.update_layout(
                                **PLOTLY_LAYOUT,
                                height=280,
                                showlegend=False,
                                xaxis=dict(gridcolor="#001a00", tickangle=-30),
                                yaxis=dict(
                                    gridcolor="#001a00", showgrid=True,
                                    range=[0, 110], ticksuffix="%",
                                ),
                            )
                            st.plotly_chart(_fig_bar, use_container_width=True)

            st.divider()

            # ── Raw data table ────────────────────────────────────────────
            _dv_filter_cols = st.columns([2, 2, 1])
            _dv_asset_filter = None
            if "asset" in [c.lower() for c in (_dv_sum.get("columns") or [])]:
                _dv_asset_filter = _dv_filter_cols[0].text_input(
                    "Asset-Filter (leer = alle):", key="dv_asset_filter"
                ) or None
            _dv_limit = _dv_filter_cols[1].select_slider(
                "Zeilen anzeigen:", options=[50, 100, 250, 500, 1000],
                value=250, key="dv_limit",
            )
            _dv_page = _dv_filter_cols[2].number_input(
                "Seite:", min_value=1, value=1, step=1, key="dv_page"
            )
            _dv_offset = (_dv_page - 1) * _dv_limit

            try:
                _dv_params = {"limit": _dv_limit, "offset": _dv_offset}
                if _dv_asset_filter:
                    _dv_params["asset"] = _dv_asset_filter
                _dv_data_resp = httpx.get(
                    f"{BOT_API_URL}/api/db-viewer"
                    f"/{_dv_selected['name']}/table/{_dv_tbl_choice}",
                    params=_dv_params,
                    timeout=20.0,
                )
                _dv_data = _dv_data_resp.json()
                _dv_total = _dv_data.get("total", 0)
                _dv_pages = max(1, (_dv_total + _dv_limit - 1) // _dv_limit)

                st.caption(
                    f"{_dv_total:,} Zeilen gesamt  |  "
                    f"Seite {_dv_page}/{_dv_pages}  |  "
                    f"Zeige {_dv_offset+1}–{min(_dv_offset+_dv_limit, _dv_total)}"
                )
                if _dv_data.get("rows"):
                    _dv_df_raw = pd.DataFrame(
                        _dv_data["rows"],
                        columns=_dv_data["columns"],
                    )
                    st.dataframe(
                        _dv_df_raw,
                        hide_index=True,
                        use_container_width=True,
                        height=500,
                    )
                else:
                    st.info("Keine Daten fuer diese Auswahl.")
            except Exception as _dv_e:
                st.error(f"Fehler beim Laden: {_dv_e}")


# ══════════════════════════════════════════════════════════════════════════════
# SUB-TAB: TRAININGSDATEN-MANAGER
# ══════════════════════════════════════════════════════════════════════════════

with _subtab_manager:
    st.subheader("Trainingsdaten-Manager")
    st.caption("Filtere simulierte Trades und exportiere gute Daten in eine Training-DB.")

    # ── 1. Source databases ───────────────────────────────────────────────
    try:
        _td_resp = httpx.get(f"{BOT_API_URL}/api/training-data/databases", timeout=10.0)
        _td_dbs = _td_resp.json().get("databases", [])
    except Exception:
        _td_dbs = []

    if not _td_dbs:
        st.info("Keine Datenbanken mit sim_trades-Tabelle gefunden.")
    else:
        _td_with_sim = [d for d in _td_dbs if d.get("has_sim_trades")]
        if not _td_with_sim:
            st.info("Keine Datenbanken mit sim_trades-Tabelle gefunden.")
        else:
            st.subheader("1. Quell-Datenbanken")
            _td_options = [
                f"{d['name']}  ({d['trade_count']} Trades, {d['size_kb']} KB)"
                for d in _td_with_sim
            ]
            _td_selected_labels = st.multiselect(
                "Datenbanken mit sim_trades auswaehlen:",
                options=_td_options,
                default=_td_options[:1],
                key="td_source_dbs",
            )
            _td_selected_dbs = [
                _td_with_sim[i]["name"]
                for i, lbl in enumerate(_td_options)
                if lbl in _td_selected_labels
            ]

            if _td_selected_dbs:
                # ── Filter options ────────────────────────────────────────
                try:
                    _fo_resp = httpx.post(
                        f"{BOT_API_URL}/api/training-data/filter-options",
                        json={"source_dbs": _td_selected_dbs},
                        timeout=10.0,
                    )
                    _fo = _fo_resp.json()
                except Exception:
                    _fo = {}

                # ── 2. Filters ────────────────────────────────────────────
                st.subheader("2. Filter")
                _fc1, _fc2, _fc3 = st.columns(3)
                with _fc1:
                    _f_assets = st.multiselect(
                        "Assets", options=_fo.get("assets", []),
                        default=_fo.get("assets", []), key="td_assets",
                    )
                    _f_directions = st.multiselect(
                        "Richtung", options=_fo.get("directions", []),
                        default=_fo.get("directions", []), key="td_directions",
                    )
                with _fc2:
                    _f_statuses = st.multiselect(
                        "Status", options=_fo.get("statuses", []),
                        default=_fo.get("statuses", []), key="td_statuses",
                    )
                    _f_sl_variants = st.multiselect(
                        "SL-Variante", options=_fo.get("sl_variants", []),
                        default=_fo.get("sl_variants", []), key="td_sl_variants",
                    )
                with _fc3:
                    _f_date_from = st.date_input("Von", value=None, key="td_date_from")
                    _f_date_to = st.date_input("Bis", value=None, key="td_date_to")

                with st.expander("Erweiterte Filter"):
                    _ef1, _ef2 = st.columns(2)
                    with _ef1:
                        _f_r_min = st.number_input(
                            "R-Multiple min", value=None, step=0.1,
                            key="td_r_min", format="%.2f",
                        )
                        _f_r_max = st.number_input(
                            "R-Multiple max", value=None, step=0.1,
                            key="td_r_max", format="%.2f",
                        )
                    with _ef2:
                        _f_pnl_min = st.number_input(
                            "P/L min (Punkte)", value=None, step=0.001,
                            key="td_pnl_min", format="%.4f",
                        )

                _td_filters: dict = {}
                if _f_assets and len(_f_assets) < len(_fo.get("assets", [])):
                    _td_filters["assets"] = _f_assets
                if _f_directions and len(_f_directions) < len(_fo.get("directions", [])):
                    _td_filters["directions"] = _f_directions
                if _f_statuses and len(_f_statuses) < len(_fo.get("statuses", [])):
                    _td_filters["statuses"] = _f_statuses
                if _f_sl_variants and len(_f_sl_variants) < len(_fo.get("sl_variants", [])):
                    _td_filters["sl_variants"] = _f_sl_variants
                if _f_date_from:
                    _td_filters["date_from"] = str(_f_date_from)
                if _f_date_to:
                    _td_filters["date_to"] = str(_f_date_to)
                if _f_r_min is not None:
                    _td_filters["r_multiple_min"] = _f_r_min
                if _f_r_max is not None:
                    _td_filters["r_multiple_max"] = _f_r_max
                if _f_pnl_min is not None:
                    _td_filters["pnl_min"] = _f_pnl_min

                # ── 3. Preview ────────────────────────────────────────────
                st.subheader("3. Vorschau")
                if st.button("Vorschau aktualisieren", key="td_preview_btn"):
                    with st.spinner("Lade Vorschau..."):
                        try:
                            _pv_resp = httpx.post(
                                f"{BOT_API_URL}/api/training-data/preview",
                                json={"source_dbs": _td_selected_dbs, "filters": _td_filters},
                                timeout=30.0,
                            )
                            st.session_state["td_preview"] = _pv_resp.json()
                        except Exception as _e:
                            st.error(f"Vorschau-Fehler: {_e}")

                _pv = st.session_state.get("td_preview")
                if _pv:
                    _pk1, _pk2, _pk3, _pk4, _pk5 = st.columns(5)
                    _pk1.metric("Trades", _pv.get("total", 0))
                    _pk2.metric("Wins", _pv.get("wins", 0))
                    _pk3.metric("Losses", _pv.get("losses", 0))
                    _pk4.metric("Win-Rate", f"{_pv.get('win_rate', 0):.1f}%")
                    _pk5.metric("Gesamt P/L", f"{_pv.get('total_pnl', 0):+.4f}")

                    _pa = _pv.get("per_asset", {})
                    if _pa:
                        st.markdown("**Pro Asset:**")
                        _pa_rows = []
                        for _asset, _stats in sorted(_pa.items()):
                            _pa_rows.append({
                                "Asset": _asset,
                                "Trades": _stats.get("trades", 0),
                                "Wins": _stats.get("wins", 0),
                                "Losses": _stats.get("losses", 0),
                                "Win-Rate": f"{_stats.get('win_rate', 0):.1f}%",
                                "P/L": f"{_stats.get('pnl', 0):+.4f}",
                            })
                        st.dataframe(
                            pd.DataFrame(_pa_rows), hide_index=True, use_container_width=True
                        )

                    _pd_dir = _pv.get("per_direction", {})
                    if _pd_dir:
                        _dc1, _dc2 = st.columns(2)
                        _dc1.metric("BUY-Trades", _pd_dir.get("BUY", 0))
                        _dc2.metric("SELL-Trades", _pd_dir.get("SELL", 0))

                # ── 4. Export ─────────────────────────────────────────────
                st.subheader("4. Export in Training-DB")
                _ex1, _ex2 = st.columns(2)
                with _ex1:
                    _existing_training = [
                        d["name"] for d in _td_dbs if d.get("has_training_trades")
                    ]
                    _target_options = _existing_training + ["-- Neue Datenbank anlegen --"]
                    _target_choice = st.selectbox(
                        "Ziel-Datenbank:",
                        options=_target_options,
                        index=0 if _existing_training else len(_target_options) - 1,
                        key="td_target_db",
                    )
                    if _target_choice == "-- Neue Datenbank anlegen --":
                        _target_db_name = st.text_input(
                            "Neuer DB-Name:", value="training.db", key="td_new_db",
                        )
                    else:
                        _target_db_name = _target_choice

                with _ex2:
                    _export_mode = st.radio(
                        "Modus:", ["append", "replace"],
                        format_func=lambda x: "Anhaengen" if x == "append" else "Ersetzen (loescht vorhandene)",
                        key="td_export_mode",
                    )

                _total_preview = (_pv or {}).get("total", 0)
                if _total_preview > 0:
                    st.info(
                        f"**{_total_preview}** Trades werden exportiert "
                        f"nach **{_target_db_name}** (Modus: {_export_mode})"
                    )

                if st.button(
                    f"Export starten ({_total_preview} Trades)",
                    key="td_export_btn",
                    disabled=_total_preview == 0,
                    type="primary",
                ):
                    with st.spinner("Exportiere..."):
                        try:
                            _ex_resp = httpx.post(
                                f"{BOT_API_URL}/api/training-data/export",
                                json={
                                    "source_dbs": _td_selected_dbs,
                                    "filters": _td_filters,
                                    "target_db": _target_db_name,
                                    "mode": _export_mode,
                                },
                                timeout=60.0,
                            )
                            _ex_result = _ex_resp.json()
                            st.success(
                                f"Erfolgreich! **{_ex_result.get('exported', 0)}** Trades "
                                f"exportiert nach **{_ex_result.get('target_db', '?')}** "
                                f"(gesamt in DB: {_ex_result.get('total_in_target', 0)})"
                            )
                            _ps = _ex_result.get("per_source", {})
                            if _ps:
                                for _src, _cnt in _ps.items():
                                    st.caption(f"  {_src}: {_cnt} Trades")
                        except Exception as _e:
                            st.error(f"Export-Fehler: {_e}")
