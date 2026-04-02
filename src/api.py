"""FastAPI-Server fuer Dashboard-Buttons und Bot-Steuerung."""
import asyncio
import glob as _glob
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, database
from .ai_analyzer import DQNAnalyzer, list_available_models, MODEL_VERSIONS, ASSETS
from .broker import CapitalComBroker, CapitalComError
from .env_writer import read_env_file, update_env_file

logger = logging.getLogger(__name__)

# SSE-Event: wird von unified_tick gesetzt wenn neue Signale vorliegen
_signal_event: asyncio.Event = asyncio.Event()

# Bot-State und Settings (Modul-Level für Import aus main.py)
_bot_state: dict = {"running": False}
_bot_settings: dict = {
    "min_confidence": 8,
    "assets": ["GOLD", "SILVER", "OIL_CRUDE", "NATURALGAS"],
    "risk_pct": 1.0,
    "leverage": 20,
    "sl_pct": 999,
    "tp_pct": 999,
}


def create_api() -> FastAPI:
    """Erstellt die FastAPI-App mit allen Endpunkten."""
    app = FastAPI(title="TradingBot API", version="1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── GET /api/status ─────────────────────────────────────────────────
    @app.get("/api/status")
    async def get_status():
        """Bot-Status: Kontostand live von Capital.com, offene Positionen, Performance."""
        try:
            open_trades = await database.get_open_trades()
            trades_today = await database.get_trades_today()
            analyses_today = await database.get_analyses_today()
            stats = await database.get_performance_stats()

            # Kontostand live von Capital.com holen
            balance: float | None = None
            equity: float | None = None
            currency: str = "EUR"
            demo: bool = config.CAPITAL_DEMO
            try:
                async with CapitalComBroker() as broker:
                    account = await broker.get_account_balance()
                    balance = account.balance
                    equity = getattr(account, "equity", account.balance)
                    currency = getattr(account, "currency", "EUR")
            except Exception as broker_exc:
                logger.warning("Capital.com Kontostand nicht abrufbar: %s – nutze Snapshot", broker_exc)
                balance = await database.get_latest_balance()

            return {
                "status": "running",
                "balance": balance,
                "equity": equity,
                "currency": currency,
                "demo": demo,
                "open_trades": len(open_trades),
                "trades_today": len(trades_today),
                "analyses_today": len(analyses_today),
                "performance": stats,
                "timestamp": datetime.now(tz=config.TZ).isoformat(),
            }
        except Exception as exc:
            raise HTTPException(500, str(exc))

    # ── POST /api/analyze ───────────────────────────────────────────────
    @app.post("/api/analyze")
    async def trigger_analysis():
        """Sofort-DQN-Analyse aus DB (0 Broker-Calls fuer Inferenz)."""
        try:
            async with CapitalComBroker() as broker:
                open_positions = await broker.get_open_positions()

                # DQN-Inferenz fuer alle 4 Assets (State aus price_history DB)
                analyzer = DQNAnalyzer()
                signals = await analyzer.get_all_signals(open_positions)

                logger.info(
                    "API-Analyse: %s",
                    " | ".join(f"{s['asset']}={s['action']}({s['confidence']}/10)" for s in signals),
                )
                return {
                    "signals": signals,
                    "open_positions": len(open_positions),
                    "timestamp": datetime.now(tz=config.TZ).isoformat(),
                }

        except CapitalComError as exc:
            raise HTTPException(502, f"Capital.com API error: {exc}")
        except Exception as exc:
            logger.exception("API analysis failed: %s", exc)
            raise HTTPException(500, str(exc))

    # ── POST /api/backtest/{trade_id} ─────────────────────────────────────
    @app.post("/api/backtest/{trade_id}")
    async def backtest_trade(
        trade_id: int,
        source: str = "sim",
        with_position: bool = False,
        capital: float | None = None,
        risk_pct: float | None = None,
        leverage: int | None = None,
    ):
        """Backtest: DQN-Inferenz zum Zeitpunkt eines historischen Trades.

        Args:
            trade_id: ID des Trades
            source: "sim" (simulation.db) oder "real" (trades.db)
            with_position: Wenn true, wird Position-Info in den State eingebaut
            capital: Kapital in EUR (optional, fuer Finanzrechnung)
            risk_pct: Risiko pro Trade in Dezimal (z.B. 0.01 = 1%)
            leverage: Hebel (z.B. 20)
        """
        try:
            if source == "sim":
                from .sim_database import get_sim_trade_by_id
                trade = await get_sim_trade_by_id(trade_id)
                if not trade:
                    raise HTTPException(404, f"Sim-Trade {trade_id} nicht gefunden")
                if trade.status.value == "open":
                    raise HTTPException(400, "Trade ist noch offen")
                asset = trade.asset
                entry_ts = trade.entry_timestamp.isoformat()
                direction = trade.direction.value
                entry_price = trade.entry_price
                pnl = trade.pnl or 0.0
            elif source == "real":
                trade = await database.get_trade_by_id(trade_id)
                if not trade:
                    raise HTTPException(404, f"Trade {trade_id} nicht gefunden")
                if trade.status.value == "OPEN":
                    raise HTTPException(400, "Trade ist noch offen")
                asset = trade.asset
                entry_ts = trade.timestamp.isoformat()
                direction = trade.direction.value
                entry_price = trade.entry_price
                pnl = trade.profit_loss or 0.0
            else:
                raise HTTPException(400, "source muss 'sim' oder 'real' sein")

            analyzer = DQNAnalyzer()
            result = await analyzer.backtest_trade(
                asset=asset,
                entry_timestamp=entry_ts,
                trade_direction=direction,
                trade_entry_price=entry_price,
                trade_result_pnl=pnl,
                with_position=with_position,
                position_direction=direction if with_position else None,
                position_entry_price=entry_price if with_position else None,
                capital=capital,
                risk_pct=risk_pct,
                leverage=leverage,
            )

            if "error" in result:
                raise HTTPException(422, result["error"])

            return result

        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Backtest failed: %s", exc)
            raise HTTPException(500, str(exc))

    # ── GET /api/backtest/trades ────────────────────────────────────────
    @app.get("/api/backtest/trades")
    async def get_backtest_trades(source: str = "sim", limit: int = 50):
        """Liste abgeschlossener Trades fuer Backtest-Auswahl."""
        try:
            if source == "sim":
                from .sim_database import get_closed_sim_trades
                trades = await get_closed_sim_trades(limit=limit)
                return {
                    "source": "sim",
                    "trades": [
                        {
                            "id": t.id,
                            "asset": t.asset,
                            "direction": t.direction.value,
                            "variant": t.sl_variant.value,
                            "entry_price": t.entry_price,
                            "exit_price": t.exit_price,
                            "pnl": t.pnl,
                            "r_multiple": t.r_multiple,
                            "status": t.status.value,
                            "entry_timestamp": t.entry_timestamp.isoformat(),
                            "exit_timestamp": t.exit_timestamp.isoformat() if t.exit_timestamp else None,
                        }
                        for t in trades
                    ],
                }
            elif source == "real":
                trades = await database.get_recent_trades(days=30)
                closed = [t for t in trades if t.profit_loss is not None]
                return {
                    "source": "real",
                    "trades": [
                        {
                            "id": t.id,
                            "asset": t.asset,
                            "direction": t.direction.value,
                            "entry_price": t.entry_price,
                            "exit_price": t.exit_price,
                            "pnl": t.profit_loss,
                            "status": t.status.value,
                            "entry_timestamp": t.timestamp.isoformat(),
                            "exit_timestamp": t.exit_timestamp.isoformat() if t.exit_timestamp else None,
                        }
                        for t in closed
                    ],
                }
            else:
                raise HTTPException(400, "source muss 'sim' oder 'real' sein")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(500, str(exc))

    # ── POST /api/daily-summary ─────────────────────────────────────────
    @app.post("/api/daily-summary")
    async def trigger_daily_summary():
        """Tagesbilanz mit DQN-Zusammenfassung."""
        try:
            trades_today = await database.get_trades_today()
            closed = [t for t in trades_today if t.profit_loss is not None]
            total_pl = sum(t.profit_loss for t in closed if t.profit_loss)
            balance = await database.get_latest_balance()
            stats = await database.get_performance_stats()

            # DQN-Zusammenfassung
            analyzer = DQNAnalyzer()
            ai_summary = await analyzer.generate_summary(
                trades=closed,
                balance=balance,
                performance_stats=stats,
                period="Tages",
            )

            return {
                "trades_count": len(closed),
                "total_pl": total_pl,
                "balance": balance,
                "trades": [
                    {
                        "id": t.id, "asset": t.asset,
                        "direction": t.direction.value,
                        "profit_loss": t.profit_loss,
                        "status": t.status.value,
                    }
                    for t in closed
                ],
                "ai_summary": ai_summary,
            }
        except Exception as exc:
            logger.exception("Daily summary failed: %s", exc)
            raise HTTPException(500, str(exc))

    # ── POST /api/trade-review/{trade_id} ───────────────────────────────
    @app.post("/api/trade-review/{trade_id}")
    async def trigger_trade_review(trade_id: int):
        """Post-Trade Review."""
        try:
            trade = await database.get_trade_by_id(trade_id)
            if not trade:
                raise HTTPException(404, f"Trade {trade_id} not found")
            if trade.status.value == "OPEN":
                raise HTTPException(400, "Trade is still open")

            # Preisverlauf nach dem Trade laden
            price_bars_after = None
            try:
                async with CapitalComBroker() as broker:
                    price_bars_after = await broker.get_price_history(
                        trade.epic, resolution="HOUR", max_bars=48,
                    )
            except Exception:
                pass

            analyzer = DQNAnalyzer()
            review = await analyzer.review_trade(trade, price_bars_after)

            # Review in DB speichern
            await database.save_trade_review(trade_id, review)

            return {
                "trade_id": trade_id,
                "asset": trade.asset,
                "direction": trade.direction.value,
                "profit_loss": trade.profit_loss,
                "review": review,
            }
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Trade review failed: %s", exc)
            raise HTTPException(500, str(exc))

    # ── POST /api/weekly-report ─────────────────────────────────────────
    @app.post("/api/weekly-report")
    async def trigger_weekly_report():
        """Wochenreport mit DQN-Analyse."""
        try:
            trades = await database.get_recent_trades(days=7)
            closed = [t for t in trades if t.profit_loss is not None]
            balance = await database.get_latest_balance()
            stats = await database.get_performance_stats()

            analyzer = DQNAnalyzer()
            ai_summary = await analyzer.generate_summary(
                trades=closed,
                balance=balance,
                performance_stats=stats,
                period="Wochen",
            )

            return {
                "trades_count": len(closed),
                "total_pl": sum(t.profit_loss for t in closed if t.profit_loss),
                "balance": balance,
                "ai_summary": ai_summary,
            }
        except Exception as exc:
            logger.exception("Weekly report failed: %s", exc)
            raise HTTPException(500, str(exc))

    # ── GET /api/pending-rechecks ──────────────────────────────────────
    @app.get("/api/pending-rechecks")
    async def get_pending_rechecks():
        """Alle ausstehenden Rechecks."""
        try:
            rechecks = await database.get_pending_rechecks()
            return {
                "rechecks": [
                    {
                        "id": rc.id,
                        "asset": rc.asset,
                        "direction": rc.direction.value,
                        "trigger_condition": rc.trigger_condition,
                        "recheck_at": rc.recheck_at.isoformat(),
                        "recheck_count": rc.recheck_count,
                        "max_rechecks": rc.max_rechecks,
                        "confidence": rc.current_confidence,
                        "status": rc.status,
                    }
                    for rc in rechecks
                ],
            }
        except Exception as exc:
            raise HTTPException(500, str(exc))

    # ── POST /api/recheck/{id}/cancel ───────────────────────────────────
    @app.post("/api/recheck/{recheck_id}/cancel")
    async def cancel_recheck(recheck_id: int):
        """Recheck manuell abbrechen."""
        try:
            await database.update_recheck_status(recheck_id, "CANCELLED")
            return {"status": "cancelled", "id": recheck_id}
        except Exception as exc:
            raise HTTPException(500, str(exc))

    # ── GET /api/learning-history ───────────────────────────────────────
    @app.get("/api/learning-history")
    async def get_learning_history():
        """Alle Lessons Learned aus Trade-Reviews."""
        try:
            lessons = await database.get_recent_lessons(limit=50)
            stats = await database.get_performance_stats()
            unreviewed = await database.get_unreviewed_trades()
            return {
                "lessons": lessons,
                "performance": stats,
                "unreviewed_trade_ids": [t.id for t in unreviewed],
            }
        except Exception as exc:
            raise HTTPException(500, str(exc))

    # ── POST /api/test/capital ───────────────────────────────────────
    @app.post("/api/test/capital")
    async def test_capital_api():
        """Testet die Capital.com API-Verbindung (Session + Account-Abfrage)."""
        t0 = time.time()
        try:
            async with CapitalComBroker() as broker:
                account = await broker.get_account_balance()
                latency_ms = int((time.time() - t0) * 1000)
                return {
                    "status": "ok",
                    "latency_ms": latency_ms,
                    "demo": config.CAPITAL_DEMO,
                    "balance": account.balance,
                    "currency": account.currency,
                    "message": f"Verbunden – Balance: {account.balance:.2f} {account.currency}",
                }
        except CapitalComError as exc:
            latency_ms = int((time.time() - t0) * 1000)
            return {
                "status": "error",
                "latency_ms": latency_ms,
                "message": f"Capital.com Fehler: {exc}",
            }
        except Exception as exc:
            latency_ms = int((time.time() - t0) * 1000)
            return {
                "status": "error",
                "latency_ms": latency_ms,
                "message": f"Verbindung fehlgeschlagen: {exc}",
            }

    # ── POST /api/test/dqn ──────────────────────────────────────────
    @app.post("/api/test/dqn")
    async def test_dqn_model():
        """Testet ob das DQN-Modell geladen werden kann (versionsbewusst)."""
        t0 = time.time()
        try:
            analyzer = DQNAnalyzer()
            analyzer._load_model()
            info = analyzer.get_current_model_info()
            latency_ms = int((time.time() - t0) * 1000)
            return {
                "status": "ok",
                "latency_ms": latency_ms,
                "model": info["model_file"],
                "version": info["version"],
                "device": str(analyzer._device),
                "config": info["config"],
                "message": f"DQN-Modell geladen: {info['model_file']} ({info['version']}, {analyzer._device})",
            }
        except Exception as exc:
            latency_ms = int((time.time() - t0) * 1000)
            return {
                "status": "error",
                "latency_ms": latency_ms,
                "message": f"DQN Fehler: {exc}",
            }

    # ── POST /api/test/ntfy ──────────────────────────────────────────
    @app.post("/api/test/ntfy")
    async def test_ntfy_api():
        """Testet die ntfy.sh Notification API (sendet Test-Nachricht)."""
        t0 = time.time()
        if not config.NTFY_TOPIC:
            return {
                "status": "error",
                "latency_ms": 0,
                "message": "NTFY_TOPIC nicht konfiguriert",
            }
        try:
            url = f"{config.NTFY_SERVER.rstrip('/')}/{config.NTFY_TOPIC}"
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    url,
                    content="API-Test vom Dashboard".encode("utf-8"),
                    headers={
                        "Title": "TradingBot API-Test",
                        "Priority": "low",
                        "Tags": "white_check_mark",
                    },
                )
                resp.raise_for_status()
            latency_ms = int((time.time() - t0) * 1000)
            return {
                "status": "ok",
                "latency_ms": latency_ms,
                "topic": config.NTFY_TOPIC,
                "message": f"Test-Nachricht gesendet an {config.NTFY_TOPIC}",
            }
        except Exception as exc:
            latency_ms = int((time.time() - t0) * 1000)
            return {
                "status": "error",
                "latency_ms": latency_ms,
                "message": f"ntfy Fehler: {exc}",
            }

    # ── GET /api/settings ────────────────────────────────────────────
    @app.get("/api/settings")
    async def get_settings():
        """Aktuelle Einstellungen mit Schema fuer Dashboard-UI."""
        try:
            env_values = read_env_file()
            settings: list[dict] = []
            for s in config.SETTINGS_SCHEMA:
                key = s["key"]
                # Current live value from config module
                live_value = getattr(config, key, None)
                # Value from .env file (may differ from live if not yet reloaded)
                env_value = env_values.get(key)

                entry = {**s, "value": live_value}
                if env_value is not None:
                    entry["env_value"] = env_value
                settings.append(entry)

            return {"settings": settings}
        except Exception as exc:
            logger.exception("Failed to read settings: %s", exc)
            raise HTTPException(500, str(exc))

    # ── POST /api/settings ───────────────────────────────────────────
    class SettingsUpdate(BaseModel):
        updates: dict[str, str | int | float | bool]

    @app.post("/api/settings")
    async def update_settings(body: SettingsUpdate):
        """Einstellungen aendern, in .env speichern und Config neu laden."""
        try:
            # Build set of allowed keys from schema
            schema_keys = {s["key"] for s in config.SETTINGS_SCHEMA}
            schema_map = {s["key"]: s for s in config.SETTINGS_SCHEMA}

            errors: list[str] = []
            env_updates: dict[str, str] = {}

            for key, value in body.updates.items():
                if key not in schema_keys:
                    errors.append(f"Unbekannter Key: {key}")
                    continue

                schema = schema_map[key]
                stype = schema["type"]

                # Validate and convert to string for .env
                try:
                    if stype == "bool":
                        str_val = "true" if value else "false"
                    elif stype == "int":
                        int_val = int(value)
                        if "min" in schema and int_val < schema["min"]:
                            errors.append(f"{key}: min {schema['min']}")
                            continue
                        if "max" in schema and int_val > schema["max"]:
                            errors.append(f"{key}: max {schema['max']}")
                            continue
                        str_val = str(int_val)
                    elif stype == "float":
                        float_val = float(value)
                        if "min" in schema and float_val < schema["min"]:
                            errors.append(f"{key}: min {schema['min']}")
                            continue
                        if "max" in schema and float_val > schema["max"]:
                            errors.append(f"{key}: max {schema['max']}")
                            continue
                        str_val = str(float_val)
                    elif stype == "select":
                        str_val = str(value)
                        if str_val not in schema.get("options", []):
                            errors.append(f"{key}: muss einer von {schema['options']} sein")
                            continue
                    else:
                        str_val = str(value)
                except (ValueError, TypeError) as exc:
                    errors.append(f"{key}: ungueltig ({exc})")
                    continue

                env_updates[key] = str_val

            if errors:
                raise HTTPException(400, detail={"errors": errors})

            if not env_updates:
                return {"status": "ok", "changed": 0}

            # Write to .env and reload config
            update_env_file(env_updates)
            config.reload()

            logger.info("Settings updated: %s", list(env_updates.keys()))
            return {
                "status": "ok",
                "changed": len(env_updates),
                "keys": list(env_updates.keys()),
            }

        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Failed to update settings: %s", exc)
            raise HTTPException(500, str(exc))

    # ── GET /api/history-status ─────────────────────────────────────────
    @app.get("/api/history-status")
    async def get_history_status():
        """Status der historischen Daten in simLastCharts.db."""
        try:
            from .fetch_history import get_all_existing_ranges, HISTORY_DB_PATH, init_history_db
            await init_history_db()
            ranges = await get_all_existing_ranges()
            return {
                "db_path": HISTORY_DB_PATH,
                "assets": ranges,
            }
        except Exception as exc:
            logger.exception("History status failed: %s", exc)
            raise HTTPException(500, str(exc))

    # ── POST /api/fetch-history ──────────────────────────────────────────
    class FetchHistoryRequest(BaseModel):
        start_date: str  # "2026-01-01"
        end_date: str    # "2026-03-28"
        assets: list[str] | None = None  # None = all from WATCHLIST

    # Track running fetch task
    _fetch_task_state: dict = {"running": False, "progress": {}, "result": None}

    @app.post("/api/fetch-history")
    async def fetch_history(body: FetchHistoryRequest):
        """Historische 1-Min-Kerzen von Capital.com laden."""
        if _fetch_task_state["running"]:
            return {"status": "already_running", "progress": _fetch_task_state["progress"]}

        from .fetch_history import fetch_all_assets

        def on_progress(asset, chunk_idx, total_chunks, new_bars):
            _fetch_task_state["progress"] = {
                "asset": asset,
                "chunk": chunk_idx,
                "total_chunks": total_chunks,
                "new_bars": new_bars,
            }

        async def _run():
            _fetch_task_state["running"] = True
            _fetch_task_state["result"] = None
            try:
                results = await fetch_all_assets(
                    start_date=body.start_date,
                    end_date=body.end_date,
                    assets=body.assets,
                    progress_callback=on_progress,
                )
                _fetch_task_state["result"] = results
            except Exception as exc:
                _fetch_task_state["result"] = {"error": str(exc)}
                logger.exception("Fetch history failed: %s", exc)
            finally:
                _fetch_task_state["running"] = False

        asyncio.create_task(_run())
        return {"status": "started", "message": "Download gestartet"}

    @app.get("/api/fetch-history/progress")
    async def fetch_history_progress():
        """Fortschritt des laufenden History-Downloads."""
        return {
            "running": _fetch_task_state["running"],
            "progress": _fetch_task_state["progress"],
            "result": _fetch_task_state["result"],
        }

    # ── GET /api/financial-defaults ────────────────────────────────────────
    @app.get("/api/financial-defaults")
    async def get_financial_defaults():
        """Aktuelle Capital.com Kontoinfos als Vorbelegung für Finanzrechnung."""
        try:
            async with CapitalComBroker() as broker:
                account = await broker.get_account_balance()
                return {
                    "balance": account.balance,
                    "currency": account.currency,
                    "demo": config.CAPITAL_DEMO,
                    "default_risk_pct": config.MAX_RISK_PER_TRADE_PCT,
                    "default_leverage": config.BACKTEST_LEVERAGE,
                }
        except Exception as exc:
            return {
                "balance": config.BACKTEST_CAPITAL,
                "currency": "EUR",
                "demo": config.CAPITAL_DEMO,
                "default_risk_pct": config.MAX_RISK_PER_TRADE_PCT,
                "default_leverage": config.BACKTEST_LEVERAGE,
                "error": str(exc),
            }

    # ── POST /api/run-timeline-sim ───────────────────────────────────────
    class TimelineSimRequest(BaseModel):
        start_date: str | None = None
        end_date: str | None = None
        assets: list[str] | None = None
        confidence_threshold: int = 8
        # Explicit model path (None = auto-select latest)
        model_path: str | None = None
        # Financial params (None = no financial tracking)
        capital:  float | None = None
        risk_pct: float | None = None
        leverage: int   | None = None
        # SL/TP percentages
        sl_pct:   float = 0.003
        tp_pct:   float = 0.005
        # Input DB (price data source, basename only)
        input_db:  str | None = None
        # Output DB filename (basename only, resolved inside DATA_DIR)
        output_db: str | None = None

    _sim_task_state: dict = {"running": False, "progress": {}, "result": None, "simulator": None}

    @app.post("/api/run-timeline-sim")
    async def run_timeline_sim(body: TimelineSimRequest):
        """Zeitstrahl-Simulation im Turbo-Modus starten."""
        if _sim_task_state["running"]:
            return {"status": "already_running", "progress": _sim_task_state["progress"]}

        from .timeline_sim import TimelineSimulator
        from .fetch_history import HISTORY_DB_PATH

        # Resolve output_db safely: basename only, always inside DATA_DIR
        data_dir = os.path.dirname(HISTORY_DB_PATH)

        def _resolve_db(name: str | None, default: str) -> str:
            if not name:
                return default
            bn = os.path.basename(name)
            if not bn.endswith(".db"):
                bn += ".db"
            return os.path.join(data_dir, bn)

        input_db_path  = _resolve_db(body.input_db,  HISTORY_DB_PATH)
        output_db_path = _resolve_db(body.output_db, HISTORY_DB_PATH)

        # Validate explicit model path if given
        explicit_model: str | None = None
        if body.model_path:
            mp = body.model_path
            if not os.path.isabs(mp):
                mp = os.path.join(config.AI_MODELS_DIR, mp)
            if os.path.exists(mp):
                explicit_model = mp
            else:
                logger.warning("Angegebenes Modell nicht gefunden: %s – nehme neuestes", mp)

        sim = TimelineSimulator(
            db_path=input_db_path,
            confidence_threshold=body.confidence_threshold,
            model_path=explicit_model,
            capital=body.capital,
            risk_pct=body.risk_pct,
            leverage=body.leverage,
            sl_pct=body.sl_pct,
            tp_pct=body.tp_pct,
            output_db_path=output_db_path,
        )
        _sim_task_state["simulator"] = sim

        def on_progress(current, total, open_trades, closed_trades):
            _sim_task_state["progress"] = {
                "current_minute": current,
                "total_minutes": total,
                "open_trades": open_trades,
                "closed_trades": closed_trades,
                "pct": round(current / total * 100, 1) if total > 0 else 0,
            }

        import time as _time

        _run_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
        _run_start_ts = _time.monotonic()

        # Config merken fuer manuelles Speichern
        _sim_task_state["config"] = {
            "run_at": _run_at,
            "model_path": sim.model_path,
            "assets": body.assets or [],
            "start_date": body.start_date,
            "end_date": body.end_date,
            "confidence_threshold": body.confidence_threshold,
            "capital": body.capital,
            "risk_pct": body.risk_pct,
            "leverage": body.leverage,
            "sl_pct": body.sl_pct,
            "tp_pct": body.tp_pct,
        }

        async def _run():
            _sim_task_state["running"] = True
            _sim_task_state["result"] = None
            try:
                result = await sim.run(
                    assets=body.assets,
                    start_date=body.start_date,
                    end_date=body.end_date,
                    progress_callback=on_progress,
                )
                _sim_task_state["result"] = result
            except Exception as exc:
                _sim_task_state["result"] = {"error": str(exc)}
                logger.exception("Timeline sim failed: %s", exc)
            finally:
                _sim_task_state["running"] = False
                _sim_task_state["simulator"] = None
                _sim_task_state["config"]["finished_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
                _sim_task_state["config"]["duration_sec"] = _time.monotonic() - _run_start_ts
                _sim_task_state["config"]["status"] = (
                    "cancelled" if sim._cancelled else
                    "error" if _sim_task_state["result"].get("error") else
                    "completed"
                )

        asyncio.create_task(_run())
        return {"status": "started", "message": "Zeitstrahl-Simulation gestartet"}

    @app.get("/api/timeline-sim/progress")
    async def timeline_sim_progress():
        """Fortschritt der laufenden Zeitstrahl-Simulation (live vom Simulator)."""
        sim: "TimelineSimulator | None" = _sim_task_state.get("simulator")
        progress = dict(_sim_task_state["progress"])
        # Merge live counters directly from simulator object (updated every iteration)
        if sim and _sim_task_state["running"]:
            total = sim.total_minutes or 1
            current = sim.current_minute
            progress = {
                "current_minute":    current,
                "total_minutes":     total,
                "open_trades":       sim.open_trade_count,
                "closed_trades":     sim.closed_trade_count,
                "pct":               round(current / total * 100, 1),
                "current_capital":   sim.current_capital if sim.fin_enabled else None,
                "equity_snap":       sim.equity_snap[-300:],
                "open_trades_snap":  sim.open_trades_snap,
                "closed_trades_snap": sim.closed_trades_snap[-50:],
            }
        return {
            "running":  _sim_task_state["running"],
            "progress": progress,
            "result":   _sim_task_state["result"],
        }

    @app.post("/api/timeline-sim/cancel")
    async def cancel_timeline_sim():
        """Laufende Zeitstrahl-Simulation abbrechen."""
        sim = _sim_task_state.get("simulator")
        if sim and _sim_task_state["running"]:
            sim.cancel()
            return {"status": "cancelling"}
        return {"status": "not_running"}

    @app.get("/api/sim-history")
    async def get_sim_history(limit: int = 50):
        """Letzte N Simulation-Runs aus sim_history.db laden."""
        from .sim_log import load_runs
        return {"runs": load_runs(limit=limit)}

    @app.post("/api/sim-history/save")
    async def save_sim_run():
        """Aktuelle Simulation manuell speichern (Speichern-Button)."""
        import json as _json
        from .sim_log import save_run

        result = _sim_task_state.get("result")
        cfg = _sim_task_state.get("config")
        if not result or result.get("error") or not cfg:
            return {"error": "Kein gueltiges Simulationsergebnis vorhanden"}

        fin = result.get("financial", {})
        mp = cfg.get("model_path") or ""
        run_id = save_run(
            run_at=cfg["run_at"],
            finished_at=cfg.get("finished_at", ""),
            duration_sec=cfg.get("duration_sec", 0),
            status=cfg.get("status", "completed"),
            model_name=os.path.basename(mp) if mp else "",
            model_path=mp,
            assets=cfg.get("assets", []),
            start_date=cfg.get("start_date"),
            end_date=cfg.get("end_date"),
            confidence_threshold=cfg.get("confidence_threshold", 1),
            capital=cfg.get("capital"),
            risk_pct=cfg.get("risk_pct"),
            leverage=cfg.get("leverage"),
            sl_pct=cfg.get("sl_pct"),
            tp_pct=cfg.get("tp_pct"),
            trades=result.get("trades", 0),
            win_rate=result.get("win_rate", 0.0),
            total_pnl_points=result.get("total_pnl_points", 0.0),
            start_capital=fin.get("start_capital"),
            end_capital=fin.get("end_capital"),
            total_return_pct=fin.get("total_return_pct"),
            max_drawdown_pct=fin.get("max_drawdown_pct"),
            result_json=_json.dumps(result, ensure_ascii=False),
        )
        return {"saved": True, "id": run_id}

    @app.delete("/api/sim-history/{run_id}")
    async def delete_sim_run(run_id: int):
        """Einen Simulation-Run aus der Historie loeschen."""
        from .sim_log import delete_run
        ok = delete_run(run_id)
        if not ok:
            return {"error": f"Run #{run_id} nicht gefunden oder Loeschfehler"}
        return {"deleted": True, "id": run_id}

    @app.get("/api/sim-history/{run_id}/load")
    async def load_sim_run(run_id: int):
        """Gespeichertes Simulationsergebnis 1:1 laden."""
        from .sim_log import load_run_result
        result = load_run_result(run_id)
        if result is None:
            return {"error": f"Run #{run_id} nicht gefunden oder keine Daten"}
        return result

    @app.get("/api/sim/candles")
    async def sim_get_candles(
        db: str,
        assets: str,
        start: str | None = None,
        end: str | None = None,
        max_points: int = 400,
    ):
        """Downsampled close-price timeseries for given assets from a sim/history DB."""
        from .fetch_history import HISTORY_DB_PATH
        data_dir = os.path.dirname(HISTORY_DB_PATH)
        db_path = os.path.join(data_dir, os.path.basename(db))
        if not os.path.exists(db_path):
            raise HTTPException(404, f"DB nicht gefunden: {db}")
        asset_list = [a.strip() for a in assets.split(",") if a.strip()]

        def _query() -> dict:
            import sqlite3
            result: dict = {}
            con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            try:
                for asset in asset_list:
                    conds, params = ["asset = ?"], [asset]
                    if start:
                        conds.append("timestamp >= ?"); params.append(start)
                    if end:
                        conds.append("timestamp <= ?"); params.append(end)
                    rows = con.execute(
                        f"SELECT timestamp, close FROM price_history "
                        f"WHERE {' AND '.join(conds)} ORDER BY timestamp ASC",
                        params,
                    ).fetchall()
                    if len(rows) > max_points:
                        step = max(1, len(rows) // max_points)
                        last = rows[-1]
                        rows = rows[::step]
                        if rows[-1] != last:
                            rows.append(last)
                    result[asset] = {
                        "timestamps": [r[0] for r in rows],
                        "prices":     [float(r[1]) for r in rows],
                    }
            finally:
                con.close()
            return result

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _query)

    # ── POST /api/sim-analysis ───────────────────────────────────────────
    class SimAnalysisRequest(BaseModel):
        current_result: dict
        history_limit: int = 10

    @app.post("/api/sim-analysis")
    async def sim_analysis(body: SimAnalysisRequest):
        """Startet Claude-Analyse und streamt das Ergebnis als text/plain."""
        from .sim_log import load_runs
        from .sim_analyzer import stream_analysis

        history_runs = load_runs(limit=body.history_limit + 1)
        # Aktuellen Run (falls aus Historie geladen) aus dem Verlauf ausschließen
        current_run_id = (body.current_result.get("_history_meta") or {}).get("run_id")
        if current_run_id:
            history_runs = [r for r in history_runs if r["id"] != current_run_id]
        history_runs = history_runs[: body.history_limit]

        async def _generate():
            async for chunk in stream_analysis(body.current_result, history_runs):
                yield chunk

        return StreamingResponse(_generate(), media_type="text/plain; charset=utf-8")

    @app.get("/api/sim-databases")
    async def list_sim_databases():
        """Alle .db-Dateien im DATA_DIR auflisten (fuer DB-Auswahl im Dashboard)."""
        from .fetch_history import HISTORY_DB_PATH
        data_dir = os.path.dirname(HISTORY_DB_PATH)
        default_name = os.path.basename(HISTORY_DB_PATH)
        dbs = []
        try:
            for fname in sorted(os.listdir(data_dir)):
                if fname.endswith(".db"):
                    fpath = os.path.join(data_dir, fname)
                    size_kb = round(os.path.getsize(fpath) / 1024, 1)
                    dbs.append({"name": fname, "size_kb": size_kb})
        except Exception:
            pass
        return {"databases": dbs, "default": default_name, "data_dir": data_dir}

    # ── Training Data Endpoints ──────────────────────────────────────────

    @app.get("/api/training-data/databases")
    async def training_data_databases():
        """Alle DBs mit sim_trades-Tabelle auflisten."""
        from .training_data import list_trade_databases
        return {"databases": list_trade_databases()}

    class TrainingFilterRequest(BaseModel):
        source_dbs: list[str]
        filters: dict = {}

    @app.post("/api/training-data/filter-options")
    async def training_filter_options(body: TrainingFilterRequest):
        """Distinct-Werte für Filter-Dropdowns laden."""
        from .training_data import get_filter_options
        return get_filter_options(body.source_dbs)

    @app.post("/api/training-data/preview")
    async def training_data_preview(body: TrainingFilterRequest):
        """Vorschau: Statistik der gefilterten Trades."""
        from .training_data import preview_filtered
        return preview_filtered(body.source_dbs, body.filters)

    class TrainingExportRequest(BaseModel):
        source_dbs: list[str]
        filters: dict = {}
        target_db: str = "training.db"
        mode: str = "append"  # "append" oder "replace"

    @app.post("/api/training-data/export")
    async def training_data_export(body: TrainingExportRequest):
        """Gefilterte Trades in Training-DB exportieren."""
        from .training_data import export_to_training_db
        if body.mode not in ("append", "replace"):
            raise HTTPException(400, "mode muss 'append' oder 'replace' sein")
        return export_to_training_db(
            source_dbs=body.source_dbs,
            filters=body.filters,
            target_db=body.target_db,
            mode=body.mode,
        )

    # ── DB-Viewer Endpoints ──────────────────────────────────────────────

    @app.get("/api/db-viewer/databases")
    async def db_viewer_databases():
        """Alle .db-Dateien im DATA_DIR mit Tabellen-Übersicht."""
        import sqlite3 as _sq
        data_dir = config.DATA_DIR
        result = []
        try:
            for fname in sorted(os.listdir(data_dir)):
                if not fname.endswith(".db"):
                    continue
                fpath = os.path.join(data_dir, fname)
                try:
                    conn = _sq.connect(fpath, timeout=3)
                    tables = {}
                    for (tname,) in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                    ).fetchall():
                        try:
                            cnt = conn.execute(f"SELECT COUNT(*) FROM \"{tname}\"").fetchone()[0]
                        except Exception:
                            cnt = -1
                        tables[tname] = cnt
                    conn.close()
                    result.append({
                        "name": fname,
                        "size_kb": round(os.path.getsize(fpath) / 1024, 1),
                        "tables": tables,
                    })
                except Exception:
                    continue
        except Exception:
            pass
        return {"databases": result}

    @app.get("/api/db-viewer/{db_name}/table/{table_name}")
    async def db_viewer_table(
        db_name: str,
        table_name: str,
        limit: int = 500,
        offset: int = 0,
        asset: str | None = None,
    ):
        """Tabelleninhalt mit optionalem Asset-Filter und Paginierung."""
        import sqlite3 as _sq
        db_name = os.path.basename(db_name)
        fpath   = os.path.join(config.DATA_DIR, db_name)
        if not os.path.exists(fpath):
            raise HTTPException(404, f"DB '{db_name}' nicht gefunden")

        try:
            conn    = _sq.connect(fpath, timeout=5)
            # Schema
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info(\"{table_name}\")").fetchall()]
            if not cols:
                raise HTTPException(404, f"Tabelle '{table_name}' nicht gefunden")

            # Zeilen mit optionalem Filter
            where  = "WHERE asset = ?" if (asset and "asset" in cols) else ""
            params = [asset] if (asset and "asset" in cols) else []

            total = conn.execute(
                f"SELECT COUNT(*) FROM \"{table_name}\" {where}", params
            ).fetchone()[0]

            rows = conn.execute(
                f"SELECT * FROM \"{table_name}\" {where} ORDER BY rowid DESC LIMIT ? OFFSET ?",
                params + [limit, offset],
            ).fetchall()
            conn.close()

            return {
                "columns": cols,
                "rows":    [list(r) for r in rows],
                "total":   total,
                "limit":   limit,
                "offset":  offset,
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(500, str(exc))

    @app.get("/api/db-viewer/{db_name}/summary/{table_name}")
    async def db_viewer_summary(db_name: str, table_name: str):
        """Spezielle Zusammenfassung für bekannte Tabellen (price_history, sim_trades, training_trades)."""
        import sqlite3 as _sq
        db_name = os.path.basename(db_name)
        fpath   = os.path.join(config.DATA_DIR, db_name)
        if not os.path.exists(fpath):
            raise HTTPException(404, f"DB '{db_name}' nicht gefunden")

        try:
            conn = _sq.connect(fpath, timeout=5)
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info(\"{table_name}\")").fetchall()]

            summary: dict = {"table": table_name, "columns": cols}

            if table_name == "price_history" and "asset" in cols:
                rows = conn.execute(
                    """SELECT asset, COUNT(*),
                       MIN(timestamp), MAX(timestamp)
                       FROM price_history GROUP BY asset ORDER BY asset"""
                ).fetchall()
                summary["per_asset"] = [
                    {"asset": r[0], "count": r[1], "from": r[2][:16], "to": r[3][:16]}
                    for r in rows
                ]

            elif table_name in ("sim_trades", "training_trades") and "asset" in cols:
                src_col = "source_db, " if table_name == "training_trades" else ""
                rows = conn.execute(
                    f"""SELECT asset, direction, COUNT(*),
                        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END),
                        ROUND(AVG(r_multiple), 3),
                        ROUND(SUM(pnl), 4)
                        FROM {table_name}
                        WHERE status != 'open'
                        GROUP BY asset, direction ORDER BY asset, direction"""
                ).fetchall()
                summary["per_asset_dir"] = [
                    {"asset": r[0], "direction": r[1], "trades": r[2],
                     "wins": r[3], "win_rate": round(r[3]/r[2]*100,1) if r[2] else 0,
                     "avg_r": r[4], "total_pnl": r[5]}
                    for r in rows
                ]
                total = conn.execute(
                    f"SELECT COUNT(*), SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END), "
                    f"ROUND(AVG(r_multiple),3) FROM {table_name} WHERE status != 'open'"
                ).fetchone()
                summary["totals"] = {
                    "trades": total[0], "wins": total[1],
                    "win_rate": round(total[1]/total[0]*100,1) if total[0] else 0,
                    "avg_r": total[2],
                }

            conn.close()
            return summary
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(500, str(exc))

    @app.get("/api/db-viewer/{db_name}/chart/{table_name}")
    async def db_viewer_chart(
        db_name: str,
        table_name: str,
        asset: str | None = None,
        points: int = 600,
    ):
        """Chart-Daten: OHLC für price_history, kumulatives PnL für Trades."""
        import sqlite3 as _sq
        db_name = os.path.basename(db_name)
        fpath   = os.path.join(config.DATA_DIR, db_name)
        if not os.path.exists(fpath):
            raise HTTPException(404, f"DB '{db_name}' nicht gefunden")

        try:
            conn = _sq.connect(fpath, timeout=10)

            if table_name == "price_history":
                if not asset:
                    # Erster verfügbarer Asset
                    row = conn.execute("SELECT asset FROM price_history LIMIT 1").fetchone()
                    asset = row[0] if row else None
                if not asset:
                    conn.close()
                    return {"type": "none"}

                total = conn.execute(
                    "SELECT COUNT(*) FROM price_history WHERE asset=?", [asset]
                ).fetchone()[0]
                stride = max(1, total // points)

                rows = conn.execute(
                    """SELECT timestamp, open, high, low, close
                       FROM price_history WHERE asset=?
                       ORDER BY timestamp""",
                    [asset],
                ).fetchall()
                conn.close()

                # Downsample
                sampled = rows[::stride]
                return {
                    "type":       "ohlc",
                    "asset":      asset,
                    "timestamps": [r[0] for r in sampled],
                    "open":       [r[1] for r in sampled],
                    "high":       [r[2] for r in sampled],
                    "low":        [r[3] for r in sampled],
                    "close":      [r[4] for r in sampled],
                    "total":      total,
                    "stride":     stride,
                }

            elif table_name in ("sim_trades", "training_trades"):
                asset_where = "AND asset = ?" if asset else ""
                params_a    = [asset] if asset else []

                rows = conn.execute(
                    f"""SELECT entry_timestamp, pnl, asset, direction
                        FROM {table_name}
                        WHERE status != 'open' {asset_where}
                        ORDER BY entry_timestamp""",
                    params_a,
                ).fetchall()

                # Per-asset win-rate
                wr_rows = conn.execute(
                    f"""SELECT asset, direction,
                           COUNT(*),
                           SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)
                        FROM {table_name}
                        WHERE status != 'open' {asset_where}
                        GROUP BY asset, direction ORDER BY asset, direction""",
                    params_a,
                ).fetchall()
                conn.close()

                # Cumulative PnL
                cum = 0.0
                timestamps, cumulative = [], []
                for ts, pnl, *_ in rows:
                    cum += pnl or 0.0
                    timestamps.append(ts)
                    cumulative.append(round(cum, 4))

                # Bar-Daten
                bar_labels = [f"{r[0]} {r[1]}" for r in wr_rows]
                bar_wr     = [round(r[3] / r[2] * 100, 1) if r[2] else 0.0 for r in wr_rows]
                bar_trades = [r[2] for r in wr_rows]

                return {
                    "type":        "trades",
                    "timestamps":  timestamps,
                    "cumulative":  cumulative,
                    "bar_labels":  bar_labels,
                    "bar_wr":      bar_wr,
                    "bar_trades":  bar_trades,
                }

            else:
                conn.close()
                return {"type": "none"}

        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(500, str(exc))

    # ── Bot-Steuerung ──────────────────────────────────────────────────────

    @app.get("/api/bot/status")
    async def get_bot_status():
        return {"running": _bot_state["running"], "settings": _bot_settings}

    @app.get("/api/bot/signals")
    async def get_bot_signals():
        """Letzte Signale aus dem unified_tick (gecacht)."""
        return {"signals": config.BOT_LAST_SIGNALS}

    @app.get("/api/bot/signals/stream")
    async def stream_bot_signals():
        """SSE-Stream: pusht neue Signale sofort wenn verfügbar."""
        import json as _json

        async def event_generator():
            last_ts = None
            # Send current state immediately
            if config.BOT_LAST_SIGNALS:
                data = _json.dumps({"signals": config.BOT_LAST_SIGNALS})
                yield f"data: {data}\n\n"
                last_ts = config.BOT_LAST_SIGNALS[0].get("checked_at") if config.BOT_LAST_SIGNALS else None

            while True:
                try:
                    _signal_event.clear()
                    await asyncio.wait_for(_signal_event.wait(), timeout=30)
                except asyncio.TimeoutError:
                    # Send heartbeat to keep connection alive
                    yield ": heartbeat\n\n"
                    continue

                if config.BOT_LAST_SIGNALS:
                    new_ts = config.BOT_LAST_SIGNALS[0].get("checked_at")
                    if new_ts != last_ts:
                        last_ts = new_ts
                        data = _json.dumps({"signals": config.BOT_LAST_SIGNALS})
                        yield f"data: {data}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/bot/settings")
    async def get_bot_settings():
        return _bot_settings

    @app.post("/api/bot/settings")
    async def update_bot_settings(body: dict):
        """Bot-Einstellungen aktualisieren."""
        allowed = {"min_confidence", "assets", "risk_pct", "leverage", "sl_pct", "tp_pct"}
        for k, v in body.items():
            if k in allowed:
                _bot_settings[k] = v
        # Sync to config for live bot
        if "min_confidence" in body:
            config.MIN_CONFIDENCE_SCORE = int(body["min_confidence"])
        if "assets" in body:
            config.BOT_ACTIVE_ASSETS = body["assets"] if len(body["assets"]) < 4 else None
        return _bot_settings

    @app.post("/api/bot/start")
    async def start_bot():
        _bot_state["running"] = True
        config.TRADING_ENABLED = True
        # Apply current settings to config
        config.MIN_CONFIDENCE_SCORE = int(_bot_settings.get("min_confidence", 8))
        assets = _bot_settings.get("assets", [])
        config.BOT_ACTIVE_ASSETS = assets if len(assets) < 4 else None
        logger.info(
            "Bot gestartet (via Dashboard) – Konfidenz: %s, Assets: %s, SL: %s%%, TP: %s%%",
            _bot_settings.get("min_confidence"),
            _bot_settings.get("assets"),
            _bot_settings.get("sl_pct"),
            _bot_settings.get("tp_pct"),
        )
        return {"status": "started", "running": True, "settings": _bot_settings}

    @app.post("/api/bot/stop")
    async def stop_bot():
        _bot_state["running"] = False
        config.TRADING_ENABLED = False
        logger.info("Bot gestoppt (via Dashboard)")
        # Offene Positionen zurueckgeben damit Frontend fragen kann
        import aiosqlite
        open_pos = []
        try:
            async with aiosqlite.connect(config.DB_PATH) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT deal_id, asset, direction FROM trades WHERE status = 'OPEN'"
                ) as cur:
                    async for row in cur:
                        open_pos.append(dict(row))
        except Exception:
            pass
        return {"status": "stopped", "running": False, "open_positions": open_pos}

    @app.post("/api/positions/{deal_id}/close")
    async def close_position(deal_id: str):
        """Eine einzelne offene Position schliessen."""
        try:
            result = await broker.close_position(deal_id)
            logger.info("Position manuell geschlossen: %s", deal_id)
            return {"closed": True, "deal_id": deal_id, "result": result}
        except Exception as exc:
            logger.error("Position close failed: %s – %s", deal_id, exc)
            raise HTTPException(500, str(exc))

    @app.post("/api/positions/close-all")
    async def close_all_positions():
        """Alle offenen Positionen schliessen."""
        import aiosqlite
        closed = []
        errors = []
        try:
            async with aiosqlite.connect(config.DB_PATH) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT deal_id, asset FROM trades WHERE status = 'OPEN'"
                ) as cur:
                    rows = [dict(r) async for r in cur]
            for row in rows:
                try:
                    await broker.close_position(row["deal_id"])
                    closed.append(row["deal_id"])
                    logger.info("Position geschlossen: %s (%s)", row["deal_id"], row["asset"])
                except Exception as exc:
                    errors.append({"deal_id": row["deal_id"], "error": str(exc)})
        except Exception as exc:
            raise HTTPException(500, str(exc))
        return {"closed": closed, "errors": errors, "total": len(closed)}

    # ── GET /api/snapshots ─────────────────────────────────────────────────
    @app.get("/api/snapshots")
    async def get_snapshots(days: int = 7):
        """Kontostand-Verlauf (account_snapshots) fuer Equity-Kurve."""
        try:
            import aiosqlite
            since = (datetime.now(tz=config.TZ) - timedelta(days=days)).isoformat()
            rows: list[dict] = []
            async with aiosqlite.connect(config.DB_PATH) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT timestamp, balance, equity, open_positions "
                    "FROM account_snapshots WHERE timestamp >= ? ORDER BY timestamp ASC",
                    (since,),
                ) as cur:
                    async for row in cur:
                        rows.append(dict(row))
            return {"snapshots": rows, "days": days}
        except Exception as exc:
            raise HTTPException(500, str(exc))

    # ── GET /api/trades ────────────────────────────────────────────────────
    @app.get("/api/trades")
    async def get_trades(limit: int = 50, status: str | None = None):
        """Letzte N Trades (offen + geschlossen) aus trades.db."""
        try:
            import aiosqlite
            where = f"WHERE status = '{status}'" if status else ""
            rows: list[dict] = []
            async with aiosqlite.connect(config.DB_PATH) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    f"SELECT id, timestamp, asset, direction, entry_price, stop_loss, "
                    f"take_profit, position_size, confidence, deal_id, status, exit_price, "
                    f"exit_timestamp, profit_loss, profit_loss_pct, model "
                    f"FROM trades {where} ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ) as cur:
                    async for row in cur:
                        rows.append(dict(row))
            return {"trades": rows, "total": len(rows)}
        except Exception as exc:
            raise HTTPException(500, str(exc))

    # ── GET /api/models ────────────────────────────────────────────────────
    @app.get("/api/models")
    async def get_models():
        """Alle .pt-Modelle mit geparsten Version/Asset-Infos."""
        try:
            models = list_available_models()
            return {
                "models": models,
                "versions": {
                    k: {
                        "max_window": v.max_window,
                        "n_indicators": v.n_indicators,
                        "action_size": v.action_size,
                        "actions": v.action_map,
                        "state_size": v.state_size,
                        "sl_pct": v.sl_pct,
                        "tp_pct": v.tp_pct,
                    }
                    for k, v in MODEL_VERSIONS.items()
                },
                "assets": ASSETS,
            }
        except Exception as exc:
            raise HTTPException(500, str(exc))

    # ── POST /api/models/select ───────────────────────────────────────────
    class ModelSelectRequest(BaseModel):
        filename: str | None = None  # z.B. "GOLD_v1_run1.pt" (None = neuestes)
        version: str | None = None   # z.B. "v1", "v2" (None = auto-detect)
        asset: str | None = None     # z.B. "GOLD" (None = auto-detect)

    @app.post("/api/models/select")
    async def select_model(body: ModelSelectRequest):
        """Modell auswaehlen und Architektur-Einstellungen auto-konfigurieren."""
        try:
            analyzer = DQNAnalyzer()
            result = analyzer.select_model(
                filename=body.filename,
                version=body.version,
                asset=body.asset,
            )
            return result
        except Exception as exc:
            logger.exception("Model selection failed: %s", exc)
            raise HTTPException(500, str(exc))

    # ── GET /api/models/current ──────────────────────────────────────────
    @app.get("/api/models/current")
    async def get_current_model():
        """Infos zum aktuell geladenen Modell."""
        try:
            analyzer = DQNAnalyzer()
            return analyzer.get_current_model_info()
        except Exception as exc:
            raise HTTPException(500, str(exc))

    # ── Static Files (SPA Dashboard) ───────────────────────────────────────
    _static_dir = Path(__file__).parent.parent / "static"
    if _static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

        @app.get("/")
        async def serve_dashboard():
            return FileResponse(str(_static_dir / "index.html"))

    return app
