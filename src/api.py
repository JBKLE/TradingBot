"""FastAPI-Server fuer Dashboard-Buttons und Bot-Steuerung."""
import logging
import os
import time
from datetime import datetime

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

    return app
