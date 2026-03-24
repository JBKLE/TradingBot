"""Entry point – orchestrates the daily trading routine with APScheduler."""
import asyncio
import logging
import os
import sys
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from . import config, database
from .analyzer import MarketAnalyzer
from .indicators import calculate_all
from .news_analyzer import NewsAnalyzer
from .broker import CapitalComBroker, CapitalComError
from .executor import TradeExecutor
from .models import Direction, MarketData, Recommendation, TradeStatus
from .monitor import ActionType, PositionMonitor
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


async def daily_routine() -> None:
    """
    Main trading cycle (runs 3× per day: 08:00, 12:00, 16:00 CET):

    1.  Connect to Capital.com and renew session
    2.  Fetch account balance + open positions
    3.  Monitor open positions (auto-close detection)
    4.  Fetch price data for all watchlist assets
    5.  If no trade today: request Claude analysis
    6.  Validate signal (strategy.py)
    7.  Execute trade if signal is valid (executor.py)
    8.  Save account snapshot
    9.  Send notifications
    """
    logger.info("=" * 60)
    logger.info("Daily routine started at %s", datetime.now(tz=config.TZ).strftime("%Y-%m-%d %H:%M:%S %Z"))
    logger.info("=" * 60)

    notifier = Notifier()

    try:
        async with CapitalComBroker() as broker:
            # ── 1. Account state ──────────────────────────────────────────────
            account = await broker.get_account_balance()
            open_broker_positions = await broker.get_open_positions()
            logger.info(
                "Account: balance=%.2f %s | equity=%.2f | open_positions=%d",
                account.balance,
                account.currency,
                account.equity,
                len(open_broker_positions),
            )

            # ── 2. Monitor open positions ─────────────────────────────────────
            executor = TradeExecutor(broker)
            closed_events = await executor.monitor_open_positions()
            for event in closed_events:
                await notifier.notify_trade_closed(
                    trade=event["trade"],
                    exit_price=event["exit_price"],
                    profit_loss=event["profit_loss"],
                    profit_loss_pct=event["profit_loss_pct"],
                )

            # ── 3. Fetch market data ──────────────────────────────────────────
            market_data: dict[str, MarketData] = {}
            for asset_key, asset_info in config.WATCHLIST.items():
                epic = asset_info["epic"]
                try:
                    data = await broker.get_market_prices(epic)
                    history = await broker.get_price_history(epic)
                    data.price_history = history
                    # Tages-Kerzen zusätzlich laden (für übergeordneten Trend + Indikatoren)
                    if config.TRADING_STYLE == "intraday":
                        try:
                            daily_bars = await broker.get_price_history(epic, resolution="DAY", max_bars=30)
                            data.daily_price_history = daily_bars
                            logger.debug("Loaded %d daily bars for %s", len(daily_bars), asset_key)
                        except CapitalComError as exc:
                            logger.warning("Could not fetch daily bars for %s: %s", asset_key, exc)
                    market_data[asset_key] = data
                    logger.debug("Fetched data for %s: %s", asset_key, data.current_price)
                except CapitalComError as exc:
                    logger.warning("Could not fetch data for %s: %s", asset_key, exc)

            if not market_data:
                logger.error("No market data available – aborting routine")
                await notifier.notify_error("No market data available")
                return

            # ── 4b. Indikatoren berechnen ─────────────────────────────────────
            indicators: dict[str, dict] = {}
            for asset_key, data in market_data.items():
                bars = data.daily_price_history if data.daily_price_history else data.price_history
                if bars:
                    indicators[asset_key] = calculate_all(bars)
                    logger.debug(
                        "Indikatoren für %s: ATR=%.4f RSI=%.1f",
                        asset_key,
                        indicators[asset_key].get("atr") or 0,
                        indicators[asset_key].get("rsi") or 0,
                    )

            # ── 5. Claude analysis ────────────────────────────────────────────
            news = NewsAnalyzer()
            market_context = await news.get_market_context()

            # ── Lern-Kontext laden ───────────────────────────────────────────
            performance_stats = await database.get_performance_stats()
            recent_lessons = await database.get_recent_lessons(limit=10)

            analyzer = MarketAnalyzer()
            analysis = await analyzer.analyze_market(
                market_data=market_data,
                account_balance=account.balance,
                open_positions=open_broker_positions,
                market_context=market_context,
                indicators=indicators,
                performance_stats=performance_stats,
                recent_lessons=recent_lessons,
            )
            await database.save_analysis(analysis)

            logger.info(
                "Analysis: recommendation=%s | best=%s %s confidence=%d RR=%.2f",
                analysis.recommendation.value,
                analysis.best_opportunity.asset,
                analysis.best_opportunity.direction.value,
                analysis.best_opportunity.confidence,
                analysis.best_opportunity.risk_reward_ratio,
            )

            # ── 6. Validate signal ────────────────────────────────────────────
            trades_today = await database.get_trades_today()
            last_closed_trade = await database.get_last_closed_trade()

            # Price-Bars des besten Assets für ATR-Gate holen
            best_asset = analysis.best_opportunity.asset
            best_bars = None
            if best_asset in market_data:
                d = market_data[best_asset]
                best_bars = d.daily_price_history if d.daily_price_history else d.price_history

            strategy = TradingStrategy()
            validation = strategy.validate_signal(
                analysis=analysis,
                open_positions_count=len(open_broker_positions),
                account_balance=account.balance,
                open_positions=open_broker_positions,
                price_bars=best_bars,
                trades_today=trades_today,
                last_closed_trade=last_closed_trade,
            )

            if not validation.valid:
                logger.info("Signal rejected: %s", validation.reason)

                # ── Recheck planen wenn Setup vielversprechend ──────
                if analysis.recheck and analysis.recheck.worthy:
                    rc = analysis.recheck
                    epic = config.WATCHLIST.get(rc.asset, {}).get("epic", rc.asset)
                    recheck_mins = max(
                        config.RECHECK_MIN_MINUTES,
                        min(rc.recheck_in_minutes, config.RECHECK_MAX_MINUTES),
                    )
                    await database.save_pending_recheck(
                        asset=rc.asset,
                        epic=epic,
                        direction=rc.direction.value,
                        trigger_condition=rc.trigger_condition,
                        recheck_in_minutes=recheck_mins,
                        confidence=rc.current_confidence,
                        original_analysis=analysis.model_dump_json(),
                    )
                    await notifier.notify_daily_summary(
                        recommendation="WAIT",
                        reason=(
                            f"{validation.reason or analysis.wait_reason} – "
                            f"Recheck geplant: {rc.asset} in {recheck_mins} Min "
                            f"(Trigger: {rc.trigger_condition})"
                        ),
                    )
                else:
                    await notifier.notify_daily_summary(
                        recommendation="WAIT",
                        reason=validation.reason or analysis.wait_reason,
                    )

                await database.save_account_snapshot(
                    account.balance, account.equity, len(open_broker_positions)
                )
                return

            # ── 7. Execute trade ──────────────────────────────────────────────
            signal = strategy.build_signal(analysis, account.balance, price_bars=best_bars)
            if signal is None:
                logger.info("Signal nach build_signal abgelehnt (RR unter Minimum nach ATR-Override)")
                await notifier.notify_daily_summary(
                    recommendation="WAIT",
                    reason="RR-Ratio nach ATR-Override unter Minimum – kein Trade",
                )
                await database.save_account_snapshot(
                    account.balance, account.equity, len(open_broker_positions),
                )
                return

            logger.info(
                "Executing signal: %s %s size=%.2f entry=%.4f SL=%.4f TP=%.4f",
                signal.asset,
                signal.direction.value,
                signal.position_size,
                signal.entry_price,
                signal.stop_loss,
                signal.take_profit,
            )

            result = await executor.execute_trade(signal)

            if result.success and result.trade:
                await notifier.notify_trade_opened(result.trade)
                logger.info("Trade successfully opened: deal_id=%s", result.deal_id)
            else:
                logger.error("Trade execution failed: %s", result.error)
                await notifier.notify_error(f"Trade execution failed: {result.error}")

            # ── 8. Snapshot ───────────────────────────────────────────────────
            new_positions = await broker.get_open_positions()
            await database.save_account_snapshot(
                account.balance, account.equity, len(new_positions)
            )

    except CapitalComError as exc:
        logger.error("Capital.com API error: %s", exc)
        await notifier.notify_error(f"Capital.com API error: {exc}")
    except Exception as exc:
        logger.exception("Unexpected error in daily routine: %s", exc)
        await notifier.notify_error(f"Unexpected error: {exc}")

    logger.info("Daily routine completed")


async def _emergency_close(broker, trade, notifier) -> None:
    """Schliesst eine Position als Notfall wenn SL-Update fehlschlaegt."""
    if not trade.deal_id:
        return
    try:
        market = await broker.get_market_prices(trade.epic)
        close_price = market.current_price.mid
        if trade.direction == Direction.BUY:
            pl = (close_price - trade.entry_price) * trade.position_size
        else:
            pl = (trade.entry_price - close_price) * trade.position_size
        denom = trade.entry_price * trade.position_size
        pl_pct = (pl / denom * 100) if denom else 0.0

        await broker.close_position(trade.deal_id)
        await database.update_trade_closed(
            trade_id=trade.id,
            exit_price=close_price,
            profit_loss=pl,
            profit_loss_pct=pl_pct,
            status=TradeStatus.CLOSED,
        )
        logger.warning(
            "NOTFALL-CLOSE: Trade %s (%s) geschlossen – SL-Update fehlgeschlagen. P/L=%.2f",
            trade.id, trade.asset, pl,
        )
        await notifier.notify_error(
            f"NOTFALL: Trade {trade.id} ({trade.asset}) geschlossen – "
            f"SL-Update fehlgeschlagen. P/L: {pl:.2f} EUR"
        )
    except Exception as close_exc:
        logger.critical(
            "KRITISCH: Trade %s kann weder SL-Update noch Close durchfuehren: %s",
            trade.id, close_exc,
        )
        await notifier.notify_error(
            f"KRITISCH: Trade {trade.id} ({trade.asset}) – SL-Update UND Close fehlgeschlagen! "
            f"Manuelles Eingreifen erforderlich!"
        )


async def monitor_positions() -> None:
    """
    Alle 5 Minuten: Regelbasierter Check offener Positionen.
    Wird übersprungen wenn keine Position offen ist.
    """
    notifier = Notifier()
    analyzer = MarketAnalyzer()

    try:
        async with CapitalComBroker() as broker:
            monitor = PositionMonitor(broker)
            actions = await monitor.check_positions()

            for action in actions:
                trade = action.trade

                if action.action_type in (ActionType.TRAIL_STOP, ActionType.BREAK_EVEN):
                    if action.new_stop_loss and trade.deal_id:
                        try:
                            await broker.update_position(trade.deal_id, action.new_stop_loss)
                            old_sl = trade.stop_loss
                            await database.update_trade_stop_loss(trade.id, action.new_stop_loss)
                            if action.action_type == ActionType.BREAK_EVEN:
                                await notifier.notify_break_even(trade)
                            else:
                                await notifier.notify_trailing_stop(trade, old_sl, action.new_stop_loss)
                            logger.info(action.reason)
                        except Exception as exc:
                            logger.error("SL update failed for trade %s: %s", trade.id, exc)
                            # ── NOTFALL: Position schliessen wenn SL nicht gesetzt werden kann
                            await _emergency_close(broker, trade, notifier)

                elif action.action_type == ActionType.CLOSE:
                    # Intraday-Close oder regelbasierter Close
                    if trade.deal_id:
                        try:
                            market = await broker.get_market_prices(trade.epic)
                            close_price = market.current_price.mid
                            if trade.direction == Direction.BUY:
                                pl = (close_price - trade.entry_price) * trade.position_size
                            else:
                                pl = (trade.entry_price - close_price) * trade.position_size
                            denom = trade.entry_price * trade.position_size
                            pl_pct = (pl / denom * 100) if denom else 0.0

                            await broker.close_position(trade.deal_id)
                            await database.update_trade_closed(
                                trade_id=trade.id,
                                exit_price=close_price,
                                profit_loss=pl,
                                profit_loss_pct=pl_pct,
                                status=TradeStatus.CLOSED,
                            )
                            logger.info(
                                "Position %s geschlossen: %s | P/L=%.2f (%.2f%%)",
                                trade.id, action.reason, pl, pl_pct,
                            )
                            await notifier.notify_trade_closed(
                                trade=trade,
                                exit_price=close_price,
                                profit_loss=pl,
                                profit_loss_pct=pl_pct,
                            )
                        except Exception as exc:
                            logger.error("Close failed for trade %s: %s", trade.id, exc)
                            await notifier.notify_error(
                                f"Close fehlgeschlagen: Trade {trade.id} ({trade.asset}): {exc}"
                            )

                elif action.action_type == ActionType.ALERT:
                    logger.info("Monitor alert [%s]: %s", action.urgency, action.reason)
                    await notifier.notify_monitor_alert(trade, action.reason, action.urgency)

                elif action.action_type == ActionType.ESCALATE_TO_CLAUDE:
                    if not monitor.can_escalate():
                        logger.info("Escalation skipped (daily limit reached): %s", action.reason)
                        continue
                    try:
                        snapshot = monitor._price_tracker if hasattr(monitor, "_price_tracker") else None
                        latest = None
                        from .monitor import _price_tracker
                        latest = _price_tracker.get_latest(trade.epic)
                        current_price = latest.mid if latest else trade.entry_price
                        if trade.direction.value == "BUY":
                            profit_loss = (current_price - trade.entry_price) * trade.position_size
                        else:
                            profit_loss = (trade.entry_price - current_price) * trade.position_size
                        profit_loss_pct = profit_loss / (trade.entry_price * trade.position_size) * 100

                        result = await analyzer.escalate_position(
                            trade=trade,
                            escalation_reason=action.reason,
                            current_price=current_price,
                            profit_loss=profit_loss,
                            profit_loss_pct=profit_loss_pct,
                        )
                        monitor.record_escalation()
                        msg = f"Claude-Empfehlung: {result.action} – {result.reasoning}"
                        logger.info(msg)
                        await notifier.notify_monitor_alert(trade, msg, result.urgency)

                        if result.action == "ADJUST_SL" and result.new_stop_loss and trade.deal_id:
                            await broker.update_position(trade.deal_id, result.new_stop_loss, result.new_take_profit)
                            await database.update_trade_stop_loss(trade.id, result.new_stop_loss)

                    except Exception as exc:
                        logger.error("Escalation failed for trade %s: %s", trade.id, exc)

            # ── Pending Rechecks pruefen ─────────────────────────────────
            await _process_rechecks(broker, notifier, analyzer)

    except CapitalComError as exc:
        logger.error("Monitor – Capital.com error: %s", exc)
    except Exception as exc:
        logger.exception("Monitor – unexpected error: %s", exc)


async def _process_rechecks(broker, notifier, analyzer) -> None:
    """Prueft faellige Rechecks und fuehrt reife Trades aus."""
    # 1. Alte Rechecks expiren
    await database.expire_overnight_rechecks()

    # 2. Faellige Rechecks holen
    due_rechecks = await database.get_due_rechecks()
    if not due_rechecks:
        return

    logger.info("Pruefe %d faellige Recheck(s)...", len(due_rechecks))

    # 3. Marktdaten fuer alle Recheck-Assets laden
    from .indicators import calculate_all as calc_indicators
    market_data = {}
    indicators_data = {}
    for rc in due_rechecks:
        if rc.asset in market_data:
            continue
        try:
            data = await broker.get_market_prices(rc.epic)
            history = await broker.get_price_history(rc.epic)
            data.price_history = history
            if config.TRADING_STYLE == "intraday":
                try:
                    daily_bars = await broker.get_price_history(rc.epic, resolution="DAY", max_bars=30)
                    data.daily_price_history = daily_bars
                except Exception:
                    pass
            market_data[rc.asset] = data
            bars = data.daily_price_history if data.daily_price_history else data.price_history
            if bars:
                indicators_data[rc.asset] = calc_indicators(bars)
        except Exception as exc:
            logger.warning("Marktdaten fuer Recheck %s nicht verfuegbar: %s", rc.asset, exc)

    if not market_data:
        return

    # 4. Batched Claude-Recheck (alle Setups in einem Call)
    results = await analyzer.recheck_opportunities(due_rechecks, market_data, indicators_data)

    # 5. Ergebnisse verarbeiten
    account = await broker.get_account_balance()
    open_positions = await broker.get_open_positions()
    trades_today = await database.get_trades_today()
    last_closed = await database.get_last_closed_trade()

    for rc, result in zip(due_rechecks, results):
        confidence = result.get("confidence", 0)
        is_ready = result.get("is_ready", False)
        retry_worthy = result.get("retry_worthy", False)

        if is_ready and confidence >= config.MIN_CONFIDENCE_SCORE:
            # ── Trade ausfuehren ────────────────────────────────────
            logger.info(
                "Recheck REIF: %s %s (Confidence: %d, Recheck #%d)",
                rc.asset, rc.direction.value, confidence, rc.recheck_count + 1,
            )
            from .models import (
                AnalysisResult as AR, BestOpportunity as BO, Recommendation as Rec,
            )
            opp = BO(
                asset=rc.asset,
                direction=rc.direction,
                confidence=confidence,
                reasoning=result.get("reasoning", f"Recheck #{rc.recheck_count + 1}"),
                entry_price=float(result.get("entry_price", 0)),
                stop_loss=float(result.get("stop_loss", 0)),
                take_profit=float(result.get("take_profit", 0)),
                risk_reward_ratio=float(result.get("risk_reward_ratio", 0)),
            )
            analysis = AR(
                date=datetime.now().strftime("%Y-%m-%d"),
                market_summary=f"Recheck #{rc.recheck_count + 1}: {rc.trigger_condition}",
                best_opportunity=opp,
                recommendation=Rec.TRADE,
            )

            best_data = market_data.get(rc.asset)
            best_bars = None
            if best_data:
                best_bars = best_data.daily_price_history or best_data.price_history

            strategy = TradingStrategy()
            validation = strategy.validate_signal(
                analysis=analysis,
                open_positions_count=len(open_positions),
                account_balance=account.balance,
                open_positions=open_positions,
                price_bars=best_bars,
                trades_today=trades_today,
                last_closed_trade=last_closed,
            )

            if not validation.valid:
                logger.info("Recheck-Signal abgelehnt: %s", validation.reason)
                if retry_worthy and rc.recheck_count + 1 < rc.max_rechecks:
                    next_mins = _clamp_recheck_minutes(result.get("retry_in_minutes"))
                    await database.increment_recheck(rc.id, next_mins)
                else:
                    await database.update_recheck_status(rc.id, "EXPIRED")
                continue

            signal = strategy.build_signal(analysis, account.balance, price_bars=best_bars)
            if signal is None:
                await database.update_recheck_status(rc.id, "EXPIRED")
                continue

            executor = TradeExecutor(broker)
            trade_result = await executor.execute_trade(signal)

            if trade_result.success and trade_result.trade:
                await database.update_recheck_status(rc.id, "EXECUTED")
                await database.save_analysis(analysis)
                await notifier.notify_trade_opened(trade_result.trade)
                logger.info(
                    "Recheck-Trade ausgefuehrt: %s %s deal_id=%s",
                    rc.asset, rc.direction.value, trade_result.deal_id,
                )
            else:
                logger.error("Recheck-Trade fehlgeschlagen: %s", trade_result.error)
                await database.update_recheck_status(rc.id, "EXPIRED")

        elif retry_worthy and rc.recheck_count + 1 < rc.max_rechecks:
            # ── Naechster Recheck ───────────────────────────────────
            next_mins = _clamp_recheck_minutes(result.get("retry_in_minutes"))
            await database.increment_recheck(rc.id, next_mins)
            logger.info(
                "Recheck #%d %s: noch nicht reif – naechster in %d Min (%s)",
                rc.recheck_count + 1, rc.asset, next_mins,
                result.get("reasoning", "")[:80],
            )
        else:
            # ── Verworfen ───────────────────────────────────────────
            await database.update_recheck_status(rc.id, "EXPIRED")
            logger.info(
                "Recheck %s verworfen: %s",
                rc.asset, result.get("reasoning", "max Rechecks oder nicht mehr vielversprechend")[:100],
            )


def _clamp_recheck_minutes(minutes) -> int:
    """Begrenzt Recheck-Intervall auf konfigurierte Min/Max-Werte."""
    val = int(minutes or config.RECHECK_DEFAULT_MINUTES)
    return max(config.RECHECK_MIN_MINUTES, min(val, config.RECHECK_MAX_MINUTES))


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
    """Run a single analysis cycle immediately (useful for testing)."""
    await database.init_db()
    await daily_routine()


async def run_scheduler() -> None:
    """Start the APScheduler cron job and keep the event loop alive."""
    await database.init_db()
    logger.info("Initialising scheduler (schedule: %s, TZ: %s)", config.ANALYSIS_SCHEDULE, config.TIMEZONE)

    scheduler = AsyncIOScheduler(timezone=config.TIMEZONE)

    scheduler.add_job(
        daily_routine,
        CronTrigger.from_crontab(config.ANALYSIS_SCHEDULE, timezone=config.TIMEZONE),
        id="daily_routine",
        name="Daily Trading Routine",
        misfire_grace_time=300,
        coalesce=True,
    )

    scheduler.add_job(
        monitor_positions,
        IntervalTrigger(minutes=5),
        id="position_monitor",
        name="Position Monitor (5 min)",
        misfire_grace_time=60,
        coalesce=True,
    )

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
        "Scheduler started. Next analysis: %s | Monitor: alle 5 Min | Summary: 20:00",
        scheduler.get_job("daily_routine").next_run_time,
    )

    # ── Startup: nur Initialisierung, kein automatischer Durchlauf ────────
    # Analysen und Summaries werden per Schedule oder Dashboard-Buttons getriggert
    logger.info(
        "Bot gestartet – warte auf Schedule oder Dashboard-Trigger. "
        "Naechste geplante Analyse: %s",
        scheduler.get_job("daily_routine").next_run_time,
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
