"""Trade execution – opens positions and monitors them."""
import asyncio
import logging
from datetime import datetime

from . import config, database
from .broker import CapitalComBroker
from .models import Direction, Trade, TradeResult, TradeSignal, TradeStatus

logger = logging.getLogger(__name__)

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
            slippage_abs = abs(current_price - signal.entry_price)
            max_slip_abs = config.MAX_SLIPPAGE_ABS.get(signal.asset)
            if max_slip_abs is not None:
                # Asset-spezifischer absoluter Slippage-Check
                if slippage_abs > max_slip_abs:
                    msg = (
                        f"Slippage too high: expected {signal.entry_price:.4f}, "
                        f"current {current_price:.4f} "
                        f"(diff={slippage_abs:.4f} > max={max_slip_abs} for {signal.asset})"
                    )
                    logger.warning(msg)
                    return TradeResult(success=False, error=msg)
            else:
                # Fallback: prozentualer Check
                slippage_pct = slippage_abs / signal.entry_price * 100
                if slippage_pct > config.MAX_SLIPPAGE_PCT_DEFAULT:
                    msg = (
                        f"Slippage too high: expected {signal.entry_price:.4f}, "
                        f"current {current_price:.4f} ({slippage_pct:.2f}%)"
                    )
                    logger.warning(msg)
                    return TradeResult(success=False, error=msg)

        # ── 1b. Spread-Check: illiquide Maerkte meiden ────────────────────────
        spread = market.current_price.ask - market.current_price.bid
        sl_distance = abs(signal.entry_price - signal.stop_loss)
        if sl_distance > 0 and spread > 0:
            spread_to_sl = spread / sl_distance
            if spread_to_sl > 0.3:
                msg = (
                    f"Spread zu hoch relativ zum SL: Spread={spread:.4f}, "
                    f"SL-Distanz={sl_distance:.4f} ({spread_to_sl:.0%}) – illiquider Markt"
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

        # ── Deal-ID Verifizierung: echte Broker-Position-ID ermitteln ─────────
        # Capital.com vergibt oft eine andere dealId in der Positionsliste als im Confirm.
        # Retry-Loop: Capital.com braucht manchmal >1s bis die Position sichtbar ist.
        actual_deal_id = deal_id
        verified = False
        for verify_attempt in range(3):
            try:
                await asyncio.sleep(1 + verify_attempt)  # 1s, 2s, 3s
                positions = await self._broker.get_open_positions()
                for pos in positions:
                    if pos.epic == signal.epic and pos.direction.value == signal.direction.value:
                        price_diff = abs(pos.entry_price - actual_entry) / actual_entry if actual_entry else 0
                        if price_diff < 0.02:  # 2% Toleranz
                            if pos.deal_id != actual_deal_id:
                                logger.info(
                                    "Deal-ID korrigiert (Versuch %d): confirm=%s → broker=%s",
                                    verify_attempt + 1, actual_deal_id, pos.deal_id,
                                )
                            actual_deal_id = pos.deal_id
                            verified = True
                            break
                if verified:
                    break
            except Exception as exc:
                logger.warning(
                    "Deal-ID Verifikation Versuch %d fehlgeschlagen: %s",
                    verify_attempt + 1, exc,
                )

        if not verified:
            logger.warning(
                "Deal-ID konnte nach 3 Versuchen nicht verifiziert werden – "
                "verwende ID aus Confirm: %s (wird beim naechsten Monitor-Lauf korrigiert)",
                actual_deal_id,
            )

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
            deal_id=actual_deal_id,
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
            actual_deal_id,
        )
        return TradeResult(success=True, deal_id=actual_deal_id, trade=trade)

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
