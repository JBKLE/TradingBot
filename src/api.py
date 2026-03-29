"""FastAPI-Server fuer Dashboard-Buttons und Bot-Steuerung."""
import asyncio
import logging
import os
import time
from datetime import datetime

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import config, database
from .ai_analyzer import DQNAnalyzer
from .broker import CapitalComBroker, CapitalComError
from .env_writer import read_env_file, update_env_file

logger = logging.getLogger(__name__)


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
        """Bot-Status: Kontostand, offene Positionen, naechste Analyse."""
        try:
            balance = await database.get_latest_balance()
            open_trades = await database.get_open_trades()
            trades_today = await database.get_trades_today()
            analyses_today = await database.get_analyses_today()
            stats = await database.get_performance_stats()
            return {
                "status": "running",
                "balance": balance,
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
        eur_usd: float = 1.08,
    ):
        """Backtest: DQN-Inferenz zum Zeitpunkt eines historischen Trades.

        Args:
            trade_id: ID des Trades
            source: "sim" (simulation.db) oder "real" (trades.db)
            with_position: Wenn true, wird Position-Info in den State eingebaut
            capital: Kapital in EUR (optional, fuer Finanzrechnung)
            risk_pct: Risiko pro Trade in Dezimal (z.B. 0.01 = 1%)
            leverage: Hebel (z.B. 20)
            eur_usd: EUR/USD Kurs
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
                eur_usd=eur_usd,
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
        """Testet ob das DQN-Modell geladen werden kann."""
        t0 = time.time()
        try:
            from .ai_analyzer import _get_latest_model_path, DuelingDQN, ACTION_SIZE
            import torch
            model_path = _get_latest_model_path(config.AI_MODELS_DIR)
            model_name = os.path.basename(model_path)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            net = DuelingDQN(ACTION_SIZE).to(device)
            ckpt = torch.load(model_path, map_location=device, weights_only=True)
            net.load_state_dict(ckpt["policy_net"])
            latency_ms = int((time.time() - t0) * 1000)
            return {
                "status": "ok",
                "latency_ms": latency_ms,
                "model": model_name,
                "device": str(device),
                "message": f"DQN-Modell geladen: {model_name} ({device})",
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

    # ── POST /api/run-timeline-sim ───────────────────────────────────────
    class TimelineSimRequest(BaseModel):
        start_date: str | None = None
        end_date: str | None = None
        assets: list[str] | None = None
        confidence_threshold: int = 8
        # Financial params (None = no financial tracking)
        capital:  float | None = None
        risk_pct: float | None = None
        leverage: int   | None = None
        eur_usd:  float = 1.08
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
        if body.output_db:
            basename = os.path.basename(body.output_db)
            if not basename.endswith(".db"):
                basename += ".db"
            output_db_path = os.path.join(data_dir, basename)
        else:
            output_db_path = HISTORY_DB_PATH

        sim = TimelineSimulator(
            confidence_threshold=body.confidence_threshold,
            capital=body.capital,
            risk_pct=body.risk_pct,
            leverage=body.leverage,
            eur_usd=body.eur_usd,
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

        from .sim_log import save_run
        import time as _time

        _run_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
        _run_start_ts = _time.monotonic()

        async def _run():
            _sim_task_state["running"] = True
            _sim_task_state["result"] = None
            _status = "error"
            _error_msg = ""
            result: dict = {}
            try:
                result = await sim.run(
                    assets=body.assets,
                    start_date=body.start_date,
                    end_date=body.end_date,
                    progress_callback=on_progress,
                )
                _sim_task_state["result"] = result
                _status = "cancelled" if sim._cancelled else "completed"
            except Exception as exc:
                _error_msg = str(exc)
                _sim_task_state["result"] = {"error": _error_msg}
                logger.exception("Timeline sim failed: %s", exc)
            finally:
                _sim_task_state["running"] = False
                _sim_task_state["simulator"] = None

            # ── Protokoll in sim_history.db speichern ────────────────────
            try:
                _duration = _time.monotonic() - _run_start_ts
                _fin = result.get("financial", {})
                _mp = sim.model_path
                _mtime = ""
                if _mp and os.path.exists(_mp):
                    import datetime as _dt
                    _mtime = _dt.datetime.fromtimestamp(
                        os.path.getmtime(_mp), tz=_dt.timezone.utc
                    ).strftime("%Y-%m-%dT%H:%M:%S")
                save_run(
                    run_at=_run_at,
                    finished_at=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
                    duration_sec=_duration,
                    status=_status,
                    model_name=os.path.basename(_mp) if _mp else "",
                    model_path=_mp,
                    model_modified_at=_mtime,
                    assets=body.assets or [],
                    start_date=body.start_date,
                    end_date=body.end_date,
                    confidence_threshold=body.confidence_threshold,
                    output_db=os.path.basename(output_db_path),
                    capital=body.capital,
                    risk_pct=body.risk_pct,
                    leverage=body.leverage,
                    eur_usd=body.eur_usd,
                    total_minutes=result.get("total_minutes", 0),
                    trades=result.get("trades", 0),
                    wins=result.get("wins", 0),
                    losses=result.get("losses", 0),
                    win_rate=result.get("win_rate", 0.0),
                    total_pnl_points=result.get("total_pnl_points", 0.0),
                    avg_r_multiple=result.get("avg_r_multiple", 0.0),
                    start_capital=_fin.get("start_capital"),
                    end_capital=_fin.get("end_capital"),
                    total_return_pct=_fin.get("total_return_pct"),
                    max_drawdown_pct=_fin.get("max_drawdown_pct"),
                    margin_call=bool(_fin.get("margin_call", False)),
                    per_asset=result.get("per_asset", {}),
                    error_message=_error_msg,
                )
            except Exception as _log_exc:
                logger.warning("sim_log save_run failed: %s", _log_exc)

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

    @app.get("/api/sim-history/{run_id}/load")
    async def load_sim_run(run_id: int):
        """Einen gespeicherten Run inkl. Trades aus der Output-DB rekonstruieren."""
        import sqlite3 as _sq
        from .sim_log import load_runs
        from .fetch_history import HISTORY_DB_PATH

        runs = load_runs(limit=10000)
        run = next((r for r in runs if r["id"] == run_id), None)
        if not run:
            return {"error": f"Run #{run_id} nicht gefunden"}

        # Trades aus der damaligen Output-DB laden
        data_dir = os.path.dirname(HISTORY_DB_PATH)
        output_db_name = run.get("output_db") or "simLastCharts.db"
        output_db_path = os.path.join(data_dir, output_db_name)

        trade_list: list[dict] = []
        start_ts = end_ts = ""
        if os.path.exists(output_db_path):
            try:
                conn = _sq.connect(output_db_path, timeout=5)
                rows = conn.execute(
                    """SELECT asset, direction, entry_timestamp, exit_timestamp,
                              entry_price, exit_price, pnl, r_multiple, status
                       FROM sim_trades
                       WHERE sl_variant = 'dqn_timeline'
                       ORDER BY entry_timestamp ASC"""
                ).fetchall()
                conn.close()
                for row in rows:
                    trade_list.append({
                        "asset":       row[0],
                        "direction":   row[1],
                        "entry_ts":    row[2],
                        "exit_ts":     row[3],
                        "entry_price": row[4],
                        "exit_price":  row[5],
                        "pnl":         row[6],
                        "r_multiple":  row[7],
                        "status":      row[8],
                        "confidence":  None,
                        "netto_pnl_eur": None,
                        "capital_after": None,
                    })
                if trade_list:
                    start_ts = trade_list[0]["entry_ts"] or ""
                    end_ts   = trade_list[-1]["exit_ts"]  or ""
            except Exception as exc:
                return {"error": f"Trades konnten nicht geladen werden: {exc}"}
        else:
            return {"error": f"Output-DB '{output_db_name}' nicht gefunden"}

        # Finanzdaten aus gespeichertem Run
        fin: dict = {}
        if run.get("start_capital") is not None:
            fin = {
                "start_capital":    run["start_capital"],
                "end_capital":      run["end_capital"],
                "total_return_pct": run["total_return_pct"],
                "max_drawdown_pct": run["max_drawdown_pct"],
                "margin_call":      bool(run.get("margin_call")),
                "equity_curve":     [],   # nicht in DB gespeichert
            }

        return {
            "total_minutes":    run.get("total_minutes", 0),
            "trades":           run.get("trades", 0),
            "wins":             run.get("wins", 0),
            "losses":           run.get("losses", 0),
            "win_rate":         run.get("win_rate", 0.0),
            "total_pnl_points": run.get("total_pnl_points", 0.0),
            "avg_r_multiple":   run.get("avg_r_multiple", 0.0),
            "start_ts":         start_ts,
            "end_ts":           end_ts,
            "per_asset":        run.get("per_asset", {}),
            "financial":        fin,
            "trade_list":       trade_list,
            # Metadaten für das Dashboard-Banner
            "_history_meta": {
                "run_id":     run_id,
                "run_at":     run.get("run_at", ""),
                "model_name": run.get("model_name", ""),
                "output_db":  output_db_name,
                "confidence": run.get("confidence_threshold"),
                "assets":     run.get("assets", []),
            },
        }

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

    return app
