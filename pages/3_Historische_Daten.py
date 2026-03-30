"""Page 3: Historische Daten — Datenabruf, Status, Kerzen-Uebersicht."""
from datetime import datetime

import httpx
import pandas as pd
import streamlit as st

from dashboard_shared import BOT_API_URL, apply_css

st.set_page_config(page_title="Historische Daten", page_icon="📊", layout="wide")
apply_css()

st.markdown("# ◈ HISTORISCHE DATEN")


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATEN-STATUS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### ◈ DATEN-STATUS")

try:
    _status_resp = httpx.get(f"{BOT_API_URL}/api/history/status", timeout=10)
    _status = _status_resp.json() if _status_resp.status_code == 200 else {}
except Exception:
    _status = {}

if _status:
    _assets_status = _status.get("assets", {})
    if _assets_status:
        _cols = st.columns(len(_assets_status))
        for i, (asset, info) in enumerate(_assets_status.items()):
            with _cols[i]:
                st.metric(
                    asset,
                    f"{info.get('candles', 0):,} Kerzen",
                    f"{info.get('from', '?')[:10]} — {info.get('to', '?')[:10]}",
                )
    _total = _status.get("total_candles", 0)
    st.caption(f"Gesamt: {_total:,} Kerzen in der Datenbank")
else:
    st.info("Keine Daten verfuegbar oder API nicht erreichbar.")


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATEN ABRUFEN
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("### ◈ DATEN ABRUFEN")

_fc1, _fc2, _fc3 = st.columns(3)
with _fc1:
    _fetch_assets = st.multiselect(
        "Assets:",
        ["GOLD", "SILVER", "OIL_CRUDE", "NATURALGAS"],
        default=["GOLD", "SILVER", "OIL_CRUDE", "NATURALGAS"],
        key="fetch_assets",
    )
with _fc2:
    _fetch_days = st.number_input("Tage zurueck:", value=30, min_value=1, max_value=365, key="fetch_days")
with _fc3:
    _fetch_resolution = st.selectbox("Resolution:", ["MINUTE", "MINUTE_5", "HOUR"], key="fetch_res")

if st.button("◈ DATEN ABRUFEN", type="primary", use_container_width=True):
    with st.spinner("Rufe historische Daten ab..."):
        try:
            r = httpx.post(
                f"{BOT_API_URL}/api/history/fetch",
                json={
                    "assets": _fetch_assets,
                    "days_back": _fetch_days,
                    "resolution": _fetch_resolution,
                },
                timeout=120,
            )
            if r.status_code == 200:
                result = r.json()
                st.success(
                    f"Erfolgreich! {result.get('total_new', 0)} neue Kerzen abgerufen "
                    f"({result.get('total_fetched', 0)} gesamt)"
                )
                for asset, info in result.get("per_asset", {}).items():
                    st.caption(f"  {asset}: +{info.get('new', 0)} neue ({info.get('total', 0)} gesamt)")
            else:
                st.error(f"Fehler: {r.text[:300]}")
        except Exception as e:
            st.error(f"API-Fehler: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. ABRUF-HISTORIE
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("### ◈ ABRUF-HISTORIE")

try:
    _hist_resp = httpx.get(f"{BOT_API_URL}/api/history/log?limit=20", timeout=5)
    _hist_log = _hist_resp.json().get("log", []) if _hist_resp.status_code == 200 else []
except Exception:
    _hist_log = []

if _hist_log:
    _log_rows = []
    for entry in _hist_log:
        _log_rows.append({
            "Zeitpunkt": str(entry.get("timestamp", ""))[:19],
            "Assets": ", ".join(entry.get("assets", [])),
            "Neue Kerzen": entry.get("total_new", 0),
            "Gesamt": entry.get("total_fetched", 0),
            "Status": entry.get("status", ""),
        })
    st.dataframe(pd.DataFrame(_log_rows), hide_index=True, use_container_width=True)
else:
    st.info("Keine Abruf-Historie vorhanden.")
