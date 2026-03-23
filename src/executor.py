"""Trade execution – opens positions and monitors them."""
import logging
from datetime import datetime

from . import config, database
from .broker import CapitalComBroker
from .models import Direction, Trade, TradeResult, TradeSignal, TradeStatus

logger = logging.getLogger(__name__)

# Maximum acceptable price slippage before aborting the trade
MAX_SLIPPAGE_PCT = 0.5


class TradeExecutor:
    """Executes validated trade signals via the Capital.com API."""

    def __init__(self, broker: CapitalComBroker) -> None:
        self._broker = broker

    async def execute_trade(self, signal: TradeSignal) -> TradeResult:
        """
        1. Slippage check – verify current price is close to expected entry
        2. Open the position with Stop-Loss and Take-Profit
        3. Persist trade to database
        """
        # ── 1. Pre-trade price check ───────────────────────────────────────────
        try:
            market = await self._broker.get_market_prices(signal.epic)
        except Exception as exc:
            msg = f"Could not fetch current price for slippage check: {exc}"
            logger.error(msg)
            return TradeResult(success=False, error=msg)

        current_price = market.current_price.ask if signal.direction == Direction.BUY else market.current_price.bid

        if signal.entry_price > 0:
            slippage_pct = abs(current_price - signal.entry_price) / signal.entry_price * 100
            if slippage_pct > MAX_SLIPPAGE_PCT:
                msg = (
                    f"Slippage too high: expected {signal.entry_price:.4f}, "
                    f"current {current_price:.4f} ({slippage_pct:.2f}%)"
                )
                logger.warning(msg)
                return TradeResult(success=False, error=msg)

        # Use current market price as actual entry
        actual_entry = current_price

        # ── 2. Open the position ──────────────────────────────────────────────
        try:
            result = await self._broker.open_position(
                epic=signal.epic,
                direction=signal.direction.value,
                size=signal.position_size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
            )
        except Exception as exc:
            msg = f"Failed to open position: {exc}"
            logger.error(msg)
            return TradeResult(success=False, error=msg)

        deal_reference = result.get("dealReference", "")
        deal_id = result.get("dealId", "")
        if not deal_id and deal_reference:
            try:
                confirm = await self._broker.confirm_trade(deal_reference)
                deal_id = confirm.get("dealId", deal_reference)
            except Exception as exc:
                logger.warning("Could not confirm trade deal_id: %s – using dealReference", exc)
                deal_id = deal_reference

        # ── 3. Persist to database ────────────────────────────────────────────
        trade = Trade(
            timestamp=datetime.now(tz=config.TZ),
            asset=signal.asset,
            epic=signal.epic,
            direction=signal.direction,
            entry_price=actual_entry,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            position_size=signal.position_size,
            confidence=signal.confidence,
            reasoning=signal.reasoning,
            deal_id=deal_id,
            status=TradeStatus.OPEN,
        )
        trade.id = await database.save_trade(trade)

        logger.info(
            "Trade executed: %s %s %.2f units @ %.4f | SL=%.4f TP=%.4f | deal_id=%s",
            signal.asset,
            signal.direction.value,
            signal.position_size,
            actual_entry,
            signal.stop_loss,
            signal.take_profit,
            deal_id,
        )
        return TradeResult(success=True, deal_id=deal_id, trade=trade)

    async def monitor_open_positions(self) -> list[dict]:
        """
        Check open positions against current prices.
        Returns a list of events (closed trades, approaching TP etc.)
        """
        events: list[dict] = []
        db_trades = await database.get_open_trades()
        if not db_trades:
            return events

        broker_positions = {p.deal_id: p for p in await self._broker.get_open_positions()}

        for trade in db_trades:
            if not trade.deal_id:
                continue

            if trade.deal_id not in broker_positions:
                # Position no longer open at broker – determine close reason
                try:
                    market = await self._broker.get_market_prices(trade.epic)
                    current = market.current_price.mid
                except Exception:
                    current = trade.entry_price

                profit_loss = _calc_pnl(trade, current)
                profit_loss_pct = (profit_loss / (trade.entry_price * trade.position_size)) * 100

                # Guess status based on price proximity to SL/TP
                if trade.direction == Direction.BUY:
                    status = (
                        TradeStatus.TAKE_PROFIT
                        if current >= trade.take_profit * 0.99
                        else TradeStatus.STOPPED_OUT
                    )
                else:
                    status = (
                        TradeStatus.TAKE_PROFIT
                        if current <= trade.take_profit * 1.01
                        else TradeStatus.STOPPED_OUT
                    )

                await database.update_trade_closed(
                    trade_id=trade.id,  # type: ignore[arg-type]
                    exit_price=current,
                    profit_loss=profit_loss,
                    profit_loss_pct=profit_loss_pct,
                    status=status,
                )
                events.append(
                    {
                        "type": "trade_closed",
                        "trade": trade,
                        "exit_price": current,
                        "profit_loss": profit_loss,
                        "profit_loss_pct": profit_loss_pct,
                        "status": status,
                    }
                )
                logger.info(
                    "Trade %s auto-closed: %s | P/L=%.2f (%.2f%%)",
                    trade.id,
                    status.value,
                    profit_loss,
                    profit_loss_pct,
                )

        return events


def _calc_pnl(trade: Trade, current_price: float) -> float:
    if trade.direction == Direction.BUY:
        return (current_price - trade.entry_price) * trade.position_size
    else:
        return (trade.entry_price - current_price) * trade.position_size
