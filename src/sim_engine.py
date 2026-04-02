"""Simulation engine – collects 1-min OHLCV and opens/evaluates paper trades.

Every minute (during market hours only):
  1. Fetch latest 1-minute candle for all 4 assets → price_history
  2. Open 24 sim trades (4 assets × 2 directions × 3 SL/TP variants)
  3. Evaluate all open sim trades against current high/low

Market hours (Capital.com, CET/CEST):
  - Gold/Silver:  Sun 23:00 → Fri 22:00, daily break 23:00–00:00
  - Oil/Gas:      Mon 00:00 → Fri 23:00, daily break 23:00–00:00
  - Weekend:      Sat 00:00 → Sun 23:00 fully closed
"""
import logging
from datetime import datetime
from typing import Optional

from . import config
from .broker import CapitalComBroker, CapitalComError
from .sim_config import DIRECTIONS, SL_TP_VARIANTS
from .sim_database import (
    batch_close_sim_trades,
    batch_insert_prices,
    batch_insert_sim_trades,
    get_assets_with_open_trades,
    get_open_sim_trades,
    get_sim_trade_stats,
)
from .sim_models import SimDirection, SimTradeStatus

logger = logging.getLogger(__name__)


# ── Market hours ──────────────────────────────────────────────────────────────
# Capital.com commodity market hours (CET/CEST = Europe/Berlin)
# All commodities share the same daily maintenance break: 23:00–00:00 CET
# Weekend: Saturday full day closed, Sunday closed until 23:00
#
# Asset-specific schedules (CET):
#   Gold/Silver:    Sunday 23:00 → Friday 22:00
#   Oil/NaturalGas: Monday 00:00 → Friday 23:00

def is_market_open() -> bool:
    """Check if commodity markets are currently open on Capital.com.

    Returns True if at least some assets should be tradeable.
    Uses Europe/Berlin (CET/CEST) as the reference timezone.
    """
    now = datetime.now(tz=config.TZ)
    weekday = now.weekday()  # 0=Mon … 6=Sun
    hour = now.hour

    # Saturday: fully closed
    if weekday == 5:
        return False

    # Sunday: only open from 23:00 (Gold/Silver open)
    if weekday == 6:
        return hour >= 23

    # Friday: closes at 23:00
    if weekday == 4 and hour >= 23:
        return False

    # Mon–Fri: daily maintenance break 23:00–00:00
    if hour == 23:
        return False

    return True


# ── Shared broker singleton (avoids creating a new session every minute) ──────

_broker: Optional[CapitalComBroker] = None


async def _get_broker() -> CapitalComBroker:
    """Return a long-lived broker instance, creating it on first call."""
    global _broker
    if _broker is None:
        _broker = CapitalComBroker()
        await _broker.__aenter__()
        logger.info("Simulation broker session created")
    else:
        await _broker._ensure_session()
    return _broker


async def shutdown_broker() -> None:
    """Gracefully close the shared broker (call on app shutdown)."""
    global _broker
    if _broker is not None:
        await _broker.__aexit__(None, None, None)
        _broker = None
        logger.info("Simulation broker session closed")


# ── Step A: Collect prices ────────────────────────────────────────────────────

async def collect_prices(broker: CapitalComBroker) -> dict[str, dict]:
    """Fetch latest 1-min candle for each asset and store in price_history.

    Returns a dict mapping asset_key → {bid, ask, mid, high, low} for use
    by the trade opening and evaluation steps.
    """
    now = datetime.now(tz=config.TZ)
    timestamp_iso = now.isoformat()
    price_rows: list[tuple] = []
    current_prices: dict[str, dict] = {}

    for asset_key, asset_info in config.WATCHLIST.items():
        epic = asset_info["epic"]
        try:
            bars = await broker.get_price_history(epic, resolution="MINUTE", max_bars=1)
            if not bars:
                logger.warning("No 1-min candle returned for %s", asset_key)
                continue

            bar = bars[-1]
            price_rows.append((
                timestamp_iso,
                asset_key,
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                0.0,  # Capital.com MINUTE candles don't include volume
            ))
            current_prices[asset_key] = {
                "mid": bar.close,
                "high": bar.high,
                "low": bar.low,
            }
        except CapitalComError as exc:
            logger.warning("Price fetch failed for %s: %s", asset_key, exc)

    if price_rows:
        await batch_insert_prices(price_rows)
        # Also write to trades.db for trading page charts
        try:
            from . import database
            await database.batch_insert_prices(price_rows)
        except Exception as exc:
            logger.debug("trades.db price insert failed: %s", exc)

    return current_prices


# ── Step B: Open sim trades ───────────────────────────────────────────────────

async def open_sim_trades(prices: dict[str, dict]) -> int:
    """Open simulated trades (4 assets × 2 directions × 3 variants).

    Skips assets that already have open trades to avoid duplicates.
    Returns the number of trades opened.
    """
    now_iso = datetime.now(tz=config.TZ).isoformat()
    trade_rows: list[tuple] = []

    # Only open new trades for assets without existing open trades
    busy_assets = await get_assets_with_open_trades()

    for asset_key, price_info in prices.items():
        if asset_key in busy_assets:
            continue

        entry_price = price_info["mid"]

        for direction in DIRECTIONS:
            for variant_cfg in SL_TP_VARIANTS:
                if direction == SimDirection.BUY:
                    sl_price = entry_price * (1 - variant_cfg.sl_pct)
                    tp_price = entry_price * (1 + variant_cfg.tp_pct)
                else:  # SELL
                    sl_price = entry_price * (1 + variant_cfg.sl_pct)
                    tp_price = entry_price * (1 - variant_cfg.tp_pct)

                trade_rows.append((
                    asset_key,
                    direction.value,
                    variant_cfg.variant.value,
                    now_iso,
                    entry_price,
                    sl_price,
                    tp_price,
                ))

    if trade_rows:
        await batch_insert_sim_trades(trade_rows)

    return len(trade_rows)


# ── Step C: Evaluate open trades ──────────────────────────────────────────────

async def evaluate_open_trades(prices: dict[str, dict]) -> tuple[int, int]:
    """Check all open sim trades against current high/low.

    Uses high/low of the latest candle for more realistic SL/TP detection
    (a close-only check would miss intra-candle hits).

    Returns (closed_tp_count, closed_sl_count).
    """
    open_trades = await get_open_sim_trades()
    if not open_trades:
        return 0, 0

    now_iso = datetime.now(tz=config.TZ).isoformat()
    updates: list[tuple] = []
    closed_tp = 0
    closed_sl = 0

    for trade in open_trades:
        price_info = prices.get(trade.asset)
        if not price_info:
            continue

        candle_high = price_info["high"]
        candle_low = price_info["low"]
        mid_price = price_info["mid"]

        status: Optional[SimTradeStatus] = None
        exit_price: Optional[float] = None

        if trade.direction == SimDirection.BUY:
            # BUY: SL hit if price dropped to sl_price, TP hit if rose to tp_price
            if candle_low <= trade.sl_price:
                status = SimTradeStatus.CLOSED_SL
                exit_price = trade.sl_price  # assume filled at SL level
            elif candle_high >= trade.tp_price:
                status = SimTradeStatus.CLOSED_TP
                exit_price = trade.tp_price
        else:  # SELL
            # SELL: SL hit if price rose to sl_price, TP hit if dropped to tp_price
            if candle_high >= trade.sl_price:
                status = SimTradeStatus.CLOSED_SL
                exit_price = trade.sl_price
            elif candle_low <= trade.tp_price:
                status = SimTradeStatus.CLOSED_TP
                exit_price = trade.tp_price

        if status and exit_price is not None:
            # Calculate P/L
            if trade.direction == SimDirection.BUY:
                pnl = exit_price - trade.entry_price
            else:
                pnl = trade.entry_price - exit_price

            risk = abs(trade.entry_price - trade.sl_price)
            r_multiple = pnl / risk if risk > 0 else 0.0

            updates.append((
                status.value,
                now_iso,
                exit_price,
                pnl,
                r_multiple,
                trade.id,
            ))

            if status == SimTradeStatus.CLOSED_TP:
                closed_tp += 1
            else:
                closed_sl += 1

    if updates:
        await batch_close_sim_trades(updates)

    return closed_tp, closed_sl


# ── Combined tick (called by APScheduler every minute) ────────────────────────

async def sim_tick() -> None:
    """Single simulation tick – collect prices, open trades, evaluate."""
    if not is_market_open():
        return

    try:
        broker = await _get_broker()

        # Step A: collect 1-min OHLCV for all assets
        prices = await collect_prices(broker)
        if not prices:
            logger.warning("Sim tick: no prices received – skipping")
            return

        # Step B: open 24 new sim trades
        opened = await open_sim_trades(prices)

        # Step C: evaluate all open sim trades
        closed_tp, closed_sl = await evaluate_open_trades(prices)

        logger.info(
            "Sim tick: assets=%d | opened=%d | closed_tp=%d | closed_sl=%d",
            len(prices),
            opened,
            closed_tp,
            closed_sl,
        )

    except CapitalComError as exc:
        logger.error("Sim tick – Capital.com error: %s", exc)
    except Exception as exc:
        logger.exception("Sim tick – unexpected error: %s", exc)
