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
from .news_analyzer import NewsAnalyzer
from .broker import CapitalComBroker, CapitalComError
from .executor import TradeExecutor
from .models import MarketData, Recommendation
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
                    market_data[asset_key] = data
                    logger.debug("Fetched data for %s: %s", asset_key, data.current_price)
                except CapitalComError as exc:
                    logger.warning("Could not fetch data for %s: %s", asset_key, exc)

            if not market_data:
                logger.error("No market data available – aborting routine")
                await notifier.notify_error("No market data available")
                return

            # ── 5. Claude analysis ────────────────────────────────────────────
            news = NewsAnalyzer()
            market_context = await news.get_market_context()

            analyzer = MarketAnalyzer()
            analysis = await analyzer.analyze_market(
                market_data=market_data,
                account_balance=account.balance,
                open_positions=open_broker_positions,
                market_context=market_context,
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
            strategy = TradingStrategy()
            validation = strategy.validate_signal(
                analysis=analysis,
                open_positions_count=len(open_broker_positions),
                account_balance=account.balance,
            )

            if not validation.valid:
                logger.info("Signal rejected: %s", validation.reason)
                await notifier.notify_daily_summary(
                    recommendation="WAIT",
                    reason=validation.reason or analysis.wait_reason,
                )
                await database.save_account_snapshot(
                    account.balance, account.equity, len(open_broker_positions)
                )
                return

            # ── 6b. Request user confirmation (if enabled) ────────────────────
            if config.TRADE_CONFIRMATION_ENABLED:
                import time as _time
                since = _time.time()
                await notifier.request_trade_confirmation(analysis.best_opportunity)
                approved = await notifier.wait_for_confirmation(since)
                if not approved:
                    logger.info("Trade not confirmed – skipping execution")
                    await database.save_account_snapshot(
                        account.balance, account.equity, len(open_broker_positions)
                    )
                    return

            # ── 7. Execute trade ──────────────────────────────────────────────
            signal = strategy.build_signal(analysis, account.balance)
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

    except CapitalComError as exc:
        logger.error("Monitor – Capital.com error: %s", exc)
    except Exception as exc:
        logger.exception("Monitor – unexpected error: %s", exc)


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

    # ── Startup-Sequenz: einmal direkt beim Start ──────────────────────────
    logger.info("Running startup checks...")
    await daily_routine()
    await monitor_positions()
    await daily_summary()

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
