"""Streamlit Dashboard – DQN Trading Bot (Entry Point)."""
import httpx
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from dashboard_shared import BOT_API_URL, apply_css

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trading Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_css()

st.markdown("# ◈ DQN TRADING BOT")
st.markdown("---")
st.info("Bitte waehle eine Seite in der Sidebar (links).")

# ── Auto-refresh ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### EINSTELLUNGEN")
    refresh = st.slider("Auto-Refresh (Sek)", 0, 300, 60, 10)
    if refresh > 0:
        st.markdown(f"*Aktualisierung alle {refresh}s*")
    st.markdown("---")
    if st.button("Jetzt aktualisieren"):
        st.cache_resource.clear()
        st.rerun()

    # ── Bot Settings (from .env) ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### BOT EINSTELLUNGEN")

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
                else:
                    _new = st.text_input(_label, value=str(_val or ""), key=_widget_key, help=_desc)
                    if _new != str(_val or ""):
                        _changed[_key] = _new

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
                    _err = (
                        _save_resp.json()
                        if _save_resp.headers.get("content-type", "").startswith("application/json")
                        else {}
                    )
                    st.error(f"Fehler: {_err.get('detail', _save_resp.text[:200])}")
            except Exception as e:
                st.error(f"Speichern fehlgeschlagen: {e}")

if refresh > 0:
    st_autorefresh(interval=refresh * 1000, key="auto_refresh")
