"""Entry point – unified 1-minute DQN loop.

Every minute (during market hours):
  1. collect_prices() -> 4 API calls, writes to price_history DB
  2. get_open_positions() -> 1 API call
  3. Sim-engine: open + evaluate sim trades (0 API calls, uses prices)
  4. DQN inference for all 4 assets (0 API calls, reads DB)
  5. Process signals: validate -> execute / close (0-1 API calls)

Total: 5 API calls/min (300/h, well under the 1000/h limit).
"""
import asyncio
import logging
import os
import sys
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from . import config, database
from .sim_database import init_sim_db
from .sim_engine import (
    collect_prices,
    evaluate_open_trades,
    is_market_open,
    open_sim_trades,
    shutdown_broker as sim_shutdown_broker,
    _get_broker,
)
from .ai_analyzer import DQNAnalyzer
from .broker import CapitalComBroker, CapitalComError
from .executor import TradeExecutor
from .models import (
    AnalysisResult,
    BestOpportunity,
    Direction,
    Recommendation,
    TradeStatus,
)
from .notifier import Notifier
from .strategy import TradingStrategy

# ── Logging setup ──────────────────────────────────────────────────────────────
os.makedirs(config.LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(config.LOG_DIR, f"bot_{datetime.now().strftime('%Y%m%d')}.log"),
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger(__name__)


async def unified_tick() -> None:
    """Einziger 1-Min-Job – ersetzt sim_tick + daily_routine + monitor_positions.

    Ablauf:
      1. collect_prices(broker)          → 4 API-Calls, Kerzen → DB
      2. broker.get_open_positions()     → 1 API-Call
      3. Sim-Engine (open + evaluate)    → 0 API-Calls (nutzt prices)
      4. DQN-Inferenz (alle 4 Assets)    → 0 API-Calls (liest DB)
      5. process_signals()               → 0-1 API-Calls (nur bei Trade)
    """
    if not is_market_open():
        return

    notifier = Notifier()

    try:
        broker = await _get_broker()

        # == 1. Preise holen (EINMAL, fuer alles) =============================
        prices = await collect_prices(broker)  # 4 API-Calls → DB
        if not prices:
            logger.warning("Unified tick: keine Preise erhalten – skip")
            return

        # == 2. Offene Positionen (EINMAL, fuer alles) ========================
        open_positions = await broker.get_open_positions()  # 1 API-Call

        # == 3. Sim-Engine (nutzt prices aus Schritt 1) =======================
        if config.SIM_ENABLED:
            opened = await open_sim_trades(prices)
            closed_tp, closed_sl = await evaluate_open_trades(prices)
            logger.debug(
                "Sim: opened=%d | closed_tp=%d | closed_sl=%d",
                opened, closed_tp, closed_sl,
            )

        # == 4. Offene DB-Trades gegen Broker synchronisieren =================
        executor = TradeExecutor(broker)
        closed_events = await executor.monitor_open_positions()
        for event in closed_events:
            await notifier.notify_trade_closed(
                trade=event["trade"],
                exit_price=event["exit_price"],
                profit_loss=event["profit_loss"],
                profit_loss_pct=event["profit_loss_pct"],
            )

        # == 5. DQN-Inferenz fuer alle 4 Assets (State aus DB) ================
        analyzer = DQNAnalyzer()
        signals = await analyzer.get_all_signals(open_positions)

        # Signale mit Timestamp cachen fuer Dashboard + SSE-Event triggern
        ts_now = datetime.now().isoformat()
        for sig in signals:
            sig["checked_at"] = ts_now
        config.BOT_LAST_SIGNALS = signals
        try:
            from .api import _signal_event
            _signal_event.set()
        except Exception:
            pass

        # == 6. Signale verarbeiten ============================================
        await _process_dqn_signals(signals, open_positions, broker, executor, notifier)

    except CapitalComError as exc:
        logger.error("Unified tick – Capital.com error: %s", exc)
    except Exception as exc:
        logger.exception("Unified tick – unexpected error: %s", exc)


async def _process_dqn_signals(
    signals: list[dict],
    open_positions: list,
    broker: CapitalComBroker,
    executor: TradeExecutor,
    notifier: Notifier,
) -> None:
    """Verarbeitet DQN-Signale: BUY/SELL → validate → execute, CLOSE → schliessen."""
    strategy = TradingStrategy()
    trades_today = await database.get_trades_today()
    last_closed_trade = await database.get_last_closed_trade()
    account = await broker.get_account_balance()

    # Offene Positionen nach Epic indexieren
    pos_by_epic: dict[str, object] = {}
    for p in open_positions:
        pos_by_epic[p.epic] = p

    for sig in signals:
        asset = sig["asset"]
        action = sig["action"]
        epic = config.WATCHLIST.get(asset, {}).get("epic", asset)
        has_position = epic in pos_by_epic

        # Asset-Filter: nur aktive Assets traden (CLOSE immer erlaubt)
        if config.BOT_ACTIVE_ASSETS and asset not in config.BOT_ACTIVE_ASSETS:
            if action != "CLOSE":
                logger.debug("Skip %s %s – Asset nicht aktiv", asset, action)
                continue

        # ── CLOSE: Position schliessen (auch bei Kill-Switch!) ───────────
        if action == "CLOSE" and has_position:
            pos = pos_by_epic[epic]
            await _close_position(broker, pos, asset, notifier)
            continue

        # ── BUY/SELL: Gegenrichtung = erst CLOSE, dann neuer Trade ──────
        if action in ("BUY", "SELL") and has_position:
            pos = pos_by_epic[epic]
            if pos.direction.value != action:
                # Gegenrichtung → schliessen
                logger.info(
                    "DQN Richtungswechsel %s: %s → %s – schliesse Position",
                    asset, pos.direction.value, action,
                )
                await _close_position(broker, pos, asset, notifier)
                # Position ist jetzt zu – weiter mit neuem Trade
                del pos_by_epic[epic]
            else:
                # Gleiche Richtung, Position schon offen → HOLD
                continue

        # ── BUY/SELL ohne bestehende Position → neuen Trade pruefen ──────
        if action in ("BUY", "SELL") and epic not in pos_by_epic:
            direction = Direction.BUY if action == "BUY" else Direction.SELL
            confidence = sig["confidence"]

            # AnalysisResult bauen fuer strategy.validate_signal()
            opp = BestOpportunity(
                asset=asset,
                direction=direction,
                confidence=confidence,
                reasoning=f"DQN-Signal: {action} | Q: {sig['q_values']}",
                entry_price=sig["current_price"],
                stop_loss=sig["sl"] or 0.0,
                take_profit=sig["tp"] or 0.0,
                risk_reward_ratio=sig["risk_reward_ratio"] or 0.0,
            )
            analysis = AnalysisResult(
                date=datetime.now().strftime("%Y-%m-%d"),
                market_summary=f"DQN unified_tick: {asset}={action}({confidence}/10)",
                best_opportunity=opp,
                recommendation=Recommendation.TRADE if confidence >= config.MIN_CONFIDENCE_SCORE else Recommendation.WAIT,
                wait_reason=f"Confidence {confidence} < {config.MIN_CONFIDENCE_SCORE}" if confidence < config.MIN_CONFIDENCE_SCORE else None,
                tokens_used=0,
                cost_usd=0.0,
            )

            validation = strategy.validate_signal(
                analysis=analysis,
                open_positions_count=len(open_positions),
                account_balance=account.balance,
                open_positions=open_positions,
                trades_today=trades_today,
                last_closed_trade=last_closed_trade,
            )

            if not validation.valid:
                logger.debug("Signal %s %s rejected: %s", asset, action, validation.reason)
                continue

            signal = strategy.build_signal(analysis, account.balance)
            if signal is None:
                logger.debug("Signal %s %s rejected by build_signal (RR)", asset, action)
                continue

            # Bot-Settings SL/TP Override: 999 = deaktiviert → None
            from .api import _bot_settings
            if _bot_settings.get("sl_pct", 0) >= 999:
                signal.stop_loss = None
            if _bot_settings.get("tp_pct", 0) >= 999:
                signal.take_profit = None

            logger.info(
                "DQN-Trade: %s %s size=%.2f entry=%.4f SL=%s TP=%s conf=%d",
                signal.asset, signal.direction.value, signal.position_size,
                signal.entry_price, signal.stop_loss, signal.take_profit,
                signal.confidence,
            )

            result = await executor.execute_trade(signal)
            if result.success and result.trade:
                await notifier.notify_trade_opened(result.trade)
                logger.info("Trade opened: %s deal_id=%s", asset, result.deal_id)
            else:
                logger.error("Trade failed %s: %s", asset, result.error)

        # ── HOLD: nichts tun ─────────────────────────────────────────────


async def _close_position(broker, position, asset: str, notifier: Notifier) -> None:
    """Schliesst eine Broker-Position und aktualisiert die DB."""
    try:
        # P/L berechnen
        current_price = position.current_price
        if position.direction == Direction.BUY:
            pl = (current_price - position.entry_price) * position.size
        else:
            pl = (position.entry_price - current_price) * position.size
        denom = position.entry_price * position.size
        pl_pct = (pl / denom * 100) if denom else 0.0

        await broker.close_position(position.deal_id)

        # DB-Trade aktualisieren (wenn vorhanden)
        db_trades = await database.get_open_trades()
        for t in db_trades:
            if t.deal_id == position.deal_id:
                await database.update_trade_closed(
                    trade_id=t.id,
                    exit_price=current_price,
                    profit_loss=pl,
                    profit_loss_pct=pl_pct,
                    status=TradeStatus.CLOSED,
                )
                await notifier.notify_trade_closed(
                    trade=t,
                    exit_price=current_price,
                    profit_loss=pl,
                    profit_loss_pct=pl_pct,
                )
                break

        logger.info(
            "DQN-CLOSE: %s (%s) geschlossen – P/L=%.2f (%.2f%%)",
            asset, position.deal_id, pl, pl_pct,
        )
    except Exception as exc:
        logger.error("Close fehlgeschlagen fuer %s: %s", asset, exc)
        await notifier.notify_error(f"Close fehlgeschlagen: {asset}: {exc}")


async def daily_summary() -> None:
    """20:00 Uhr: Tagesübersicht per Push-Notification."""
    notifier = Notifier()
    try:
        trades_today = await database.get_trades_today()
        closed_today = [t for t in trades_today if t.profit_loss is not None]
        total_pl = sum(t.profit_loss for t in closed_today if t.profit_loss)

        balance = await database.get_latest_balance()

        await notifier.notify_daily_summary_report(
            balance=balance,
            trades_today=len(closed_today),
            profit_loss_total=total_pl,
        )
        logger.info("Daily summary sent: %d trades, P/L=%.2f", len(closed_today), total_pl)
    except Exception as exc:
        logger.error("Daily summary failed: %s", exc)


async def run_once() -> None:
    """Run a single unified tick immediately (useful for testing)."""
    await database.init_db()
    await init_sim_db()
    await unified_tick()


async def run_scheduler() -> None:
    """Start the APScheduler with a single unified 1-minute tick."""
    await database.init_db()
    await init_sim_db()
    logger.info("Initialising scheduler (unified 1-min tick, TZ: %s)", config.TIMEZONE)

    scheduler = AsyncIOScheduler(timezone=config.TIMEZONE)

    # ── Einziger Job: unified_tick jede Minute ───────────────────────────
    scheduler.add_job(
        unified_tick,
        IntervalTrigger(minutes=1),
        id="unified_tick",
        name="Unified DQN Tick (1 min)",
        misfire_grace_time=30,
        coalesce=True,
        max_instances=1,
    )

    # ── Tages-Summary bleibt als separater Job ───────────────────────────
    scheduler.add_job(
        daily_summary,
        CronTrigger(hour=20, minute=0, day_of_week="mon-fri", timezone=config.TIMEZONE),
        id="daily_summary",
        name="Daily Summary (20:00)",
        misfire_grace_time=300,
        coalesce=True,
    )

    scheduler.start()
    logger.info(
        "Scheduler started: unified_tick jede Minute | Summary: 20:00 | "
        "Sim=%s | 5 API-Calls/Min",
        config.SIM_ENABLED,
    )

    # ── FastAPI-Server starten (fuer Dashboard-Buttons) ──────────────────
    try:
        import uvicorn
        from .api import create_api

        api_app = create_api()
        api_config = uvicorn.Config(
            api_app, host="0.0.0.0", port=8502, log_level="info",
        )
        api_server = uvicorn.Server(api_config)
        asyncio.create_task(api_server.serve())
        logger.info("FastAPI-Server gestartet auf Port 8502")
    except ImportError:
        logger.warning("uvicorn nicht installiert – API-Server deaktiviert")
    except Exception as exc:
        logger.error("FastAPI-Server konnte nicht gestartet werden: %s", exc)

    # Keep running until interrupted
    try:
        while True:
            await asyncio.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopping...")
        scheduler.shutdown()
        await sim_shutdown_broker()


def main() -> None:
    """Entry point – honours RUN_ONCE env variable for one-shot execution."""
    try:
        if os.getenv("RUN_ONCE", "false").lower() == "true":
            asyncio.run(run_once())
        else:
            asyncio.run(run_scheduler())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
