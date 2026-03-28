"""Regelbasiertes Position-Monitoring – alle 5 Minuten."""
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

import logging

from . import config, database
from .broker import CapitalComBroker
from .models import Direction, Trade, TradeStatus

logger = logging.getLogger(__name__)


class ActionType(str, Enum):
    HOLD = "HOLD"
    TRAIL_STOP = "TRAIL_STOP"
    BREAK_EVEN = "BREAK_EVEN"
    ALERT = "ALERT"
    ESCALATE_TO_DQN = "ESCALATE_TO_DQN"
    CLOSE = "CLOSE"


@dataclass
class MonitorAction:
    action_type: ActionType
    trade: Trade
    reason: str
    new_stop_loss: Optional[float] = None
    urgency: str = "low"


@dataclass
class PriceSnapshot:
    timestamp: datetime
    bid: float
    ask: float

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2


class PriceTracker:
    """In-Memory Ring-Buffer für Kursdaten (288 × 5 Min = 24h)."""

    def __init__(self, max_snapshots: int = 288) -> None:
        self._prices: dict[str, deque[PriceSnapshot]] = {}
        self._max = max_snapshots

    def add_snapshot(self, epic: str, snapshot: PriceSnapshot) -> None:
        if epic not in self._prices:
            self._prices[epic] = deque(maxlen=self._max)
        self._prices[epic].append(snapshot)

    def get_change_pct(self, epic: str, minutes: int) -> float:
        snaps = self._prices.get(epic)
        if not snaps or len(snaps) < 2:
            return 0.0
        lookback = max(1, minutes // 5)
        entries = list(snaps)
        oldest = entries[max(0, len(entries) - lookback - 1)]
        newest = entries[-1]
        if oldest.mid == 0:
            return 0.0
        return (newest.mid - oldest.mid) / oldest.mid * 100

    def get_average_spread(self, epic: str, hours: int = 24) -> float:
        snaps = self._prices.get(epic)
        if not snaps:
            return 0.0
        lookback = min(len(snaps), hours * 12)
        recent = list(snaps)[-lookback:]
        return sum(s.spread for s in recent) / len(recent)

    def get_latest(self, epic: str) -> Optional[PriceSnapshot]:
        snaps = self._prices.get(epic)
        return snaps[-1] if snaps else None


# ── Module-level singletons – persists across scheduler calls ──────────────────
_price_tracker = PriceTracker()
_break_even_done: set[int] = set()   # trade IDs where break-even was applied
_escalations: dict[str, int] = {}    # date → count


class PositionMonitor:
    """
    Überwacht offene Positionen alle 5 Minuten.
    Rein regelbasiert mit DQN-Eskalation bei Bedarf.
    """

    def __init__(self, broker: CapitalComBroker) -> None:
        self._broker = broker

    async def check_positions(self) -> list[MonitorAction]:
        """Alle offenen Trades pruefen, synchronisieren und Aktionen zurueckgeben."""
        actions: list[MonitorAction] = []

        broker_positions = await self._broker.get_open_positions()
        open_trades = await database.get_open_trades()

        if not broker_positions and not open_trades:
            return actions

        broker_deal_ids = {p.deal_id for p in broker_positions}
        trades_by_deal_id = {t.deal_id: t for t in open_trades if t.deal_id}

        # ── 1. Deal-ID Korrektur (Epic+Direction Matching) ──────────────
        # MUSS vor allem anderen laufen, damit SL/TP Updates die richtige ID nutzen
        for trade in open_trades:
            if not trade.deal_id or trade.deal_id in broker_deal_ids:
                continue
            for pos in broker_positions:
                if (pos.epic == trade.epic
                        and pos.direction == trade.direction
                        and pos.deal_id not in trades_by_deal_id):
                    logger.info(
                        "Deal-ID korrigiert: Trade %s DB=%s → Broker=%s (%s)",
                        trade.id, trade.deal_id, pos.deal_id, trade.epic,
                    )
                    old_id = trade.deal_id
                    await database.update_trade_deal_id(trade.id, pos.deal_id)
                    trade.deal_id = pos.deal_id
                    trades_by_deal_id.pop(old_id, None)
                    trades_by_deal_id[pos.deal_id] = trade
                    break

        # ── 2. Orphan-Sync: Broker → DB (Positionen ohne DB-Eintrag) ───
        for pos in broker_positions:
            if pos.deal_id not in trades_by_deal_id:
                logger.warning(
                    "Verwaiste Broker-Position: %s (%s %s %.2f @ %.4f) – sync to DB",
                    pos.deal_id, pos.epic, pos.direction.value, pos.size, pos.entry_price,
                )
                try:
                    trade = await database.save_orphan_trade(pos)
                    trades_by_deal_id[pos.deal_id] = trade
                    open_trades.append(trade)
                except Exception as exc:
                    logger.error("Orphan-Sync fehlgeschlagen fuer %s: %s", pos.deal_id, exc)

        # ── 3. Stale-Sync: DB → Broker (DB-Trades ohne Broker-Position) ─
        now = datetime.now(tz=config.TZ)
        stale_trades: list[Trade] = []
        for trade in open_trades:
            if not trade.deal_id or trade.deal_id in broker_deal_ids:
                continue
            # Nur als stale markieren wenn aelter als 5 Min (neue Trades brauchen Zeit)
            ts = trade.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=config.TZ)
            if (now - ts).total_seconds() < 300:
                continue
            stale_trades.append(trade)

        for trade in stale_trades:
            try:
                market = await self._broker.get_market_prices(trade.epic)
                current = market.current_price.mid
            except Exception:
                current = trade.entry_price

            if trade.direction == Direction.BUY:
                profit_loss = (current - trade.entry_price) * trade.position_size
            else:
                profit_loss = (trade.entry_price - current) * trade.position_size
            denom = trade.entry_price * trade.position_size
            profit_loss_pct = (profit_loss / denom * 100) if denom else 0.0

            if trade.direction == Direction.BUY:
                status = (
                    TradeStatus.TAKE_PROFIT
                    if trade.take_profit and current >= trade.take_profit * 0.99
                    else TradeStatus.STOPPED_OUT
                )
            else:
                status = (
                    TradeStatus.TAKE_PROFIT
                    if trade.take_profit and current <= trade.take_profit * 1.01
                    else TradeStatus.STOPPED_OUT
                )

            await database.update_trade_closed(
                trade_id=trade.id,
                exit_price=current,
                profit_loss=profit_loss,
                profit_loss_pct=profit_loss_pct,
                status=status,
            )
            open_trades.remove(trade)
            logger.info(
                "Trade %s (%s) beim Broker geschlossen: %s | P/L=%.2f (%.2f%%)",
                trade.id, trade.asset, status.value, profit_loss, profit_loss_pct,
            )

        # ── 4. Regelbasierte Evaluation der verbleibenden Positionen ────
        for trade in open_trades:
            if trade.deal_id not in broker_deal_ids:
                continue
            try:
                market = await self._broker.get_market_prices(trade.epic)
                snapshot = PriceSnapshot(
                    timestamp=datetime.now(tz=config.TZ),
                    bid=market.current_price.bid,
                    ask=market.current_price.ask,
                )
                _price_tracker.add_snapshot(trade.epic, snapshot)
                actions.extend(self._evaluate_trade(trade, snapshot))
            except Exception as exc:
                logger.warning("Fehler beim Monitoring von Position %s: %s", trade.id, exc)

        return actions

    def can_escalate(self) -> bool:
        today = datetime.now().date().isoformat()
        return _escalations.get(today, 0) < config.ESCALATION_MAX_PER_DAY

    def record_escalation(self) -> None:
        today = datetime.now().date().isoformat()
        _escalations[today] = _escalations.get(today, 0) + 1

    def _evaluate_trade(self, trade: Trade, snapshot: PriceSnapshot) -> list[MonitorAction]:
        current = snapshot.bid if trade.direction == Direction.BUY else snapshot.ask

        results = [
            self._check_trailing_stop(trade, current, snapshot),
            self._check_position_age(trade, current),
            self._check_volatility(trade),
            self._check_tp_proximity(trade, current),
            self._check_spread(trade, snapshot),
        ]
        return [a for a in results if a is not None]

    # ── Regel 1 & 2: Trailing Stop / Break-Even ────────────────────────────────

    def _check_trailing_stop(
        self, trade: Trade, current: float, snapshot: PriceSnapshot
    ) -> Optional[MonitorAction]:
        if trade.direction == Direction.BUY:
            return self._trail_long(trade, current, snapshot)
        return self._trail_short(trade, current, snapshot)

    def _trail_long(
        self, trade: Trade, current: float, snapshot: PriceSnapshot
    ) -> Optional[MonitorAction]:
        # Break-even (einmalig)
        if (
            trade.id not in _break_even_done
            and trade.stop_loss < trade.entry_price
            and current >= trade.entry_price + 2 * snapshot.spread
        ):
            _break_even_done.add(trade.id)
            return MonitorAction(
                action_type=ActionType.BREAK_EVEN,
                trade=trade,
                reason=f"Break-Even erreicht bei {current:.4f}",
                new_stop_loss=trade.entry_price,
            )

        # Trailing Stop
        profit_pct = (current - trade.entry_price) / trade.entry_price * 100
        if profit_pct >= config.TRAIL_TRIGGER_PCT:
            profit_amount = current - trade.entry_price
            new_sl = trade.entry_price + profit_amount * config.TRAIL_DISTANCE_PCT
            if trade.stop_loss and (new_sl - trade.stop_loss) >= config.TRAIL_MIN_STEP:
                return MonitorAction(
                    action_type=ActionType.TRAIL_STOP,
                    trade=trade,
                    reason=f"Trailing Stop: SL {trade.stop_loss:.4f} → {new_sl:.4f}",
                    new_stop_loss=new_sl,
                )
        return None

    def _trail_short(
        self, trade: Trade, current: float, snapshot: PriceSnapshot
    ) -> Optional[MonitorAction]:
        # Break-even (einmalig)
        if (
            trade.id not in _break_even_done
            and trade.stop_loss > trade.entry_price
            and current <= trade.entry_price - 2 * snapshot.spread
        ):
            _break_even_done.add(trade.id)
            return MonitorAction(
                action_type=ActionType.BREAK_EVEN,
                trade=trade,
                reason=f"Break-Even erreicht bei {current:.4f}",
                new_stop_loss=trade.entry_price,
            )

        # Trailing Stop
        profit_pct = (trade.entry_price - current) / trade.entry_price * 100
        if profit_pct >= config.TRAIL_TRIGGER_PCT:
            profit_amount = trade.entry_price - current
            new_sl = trade.entry_price - profit_amount * config.TRAIL_DISTANCE_PCT
            if trade.stop_loss and (trade.stop_loss - new_sl) >= config.TRAIL_MIN_STEP:
                return MonitorAction(
                    action_type=ActionType.TRAIL_STOP,
                    trade=trade,
                    reason=f"Trailing Stop: SL {trade.stop_loss:.4f} → {new_sl:.4f}",
                    new_stop_loss=new_sl,
                )
        return None

    # ── Regel 3: Zeitbasierter Check ──────────────────────────────────────────

    def _check_position_age(self, trade: Trade, current: float) -> Optional[MonitorAction]:
        now = datetime.now(tz=config.TZ)
        ts = trade.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=config.TZ)
        age = now - ts

        if trade.direction == Direction.BUY:
            profit_pct = (current - trade.entry_price) / trade.entry_price * 100
        else:
            profit_pct = (trade.entry_price - current) / trade.entry_price * 100

        hours = age.total_seconds() / 3600

        if age >= timedelta(hours=48):
            return MonitorAction(
                action_type=ActionType.ESCALATE_TO_DQN,
                trade=trade,
                reason=f"Position seit {hours:.0f}h offen – überfällig (Swap-Kosten!)",
                urgency="high",
            )
        if age >= timedelta(hours=24) and profit_pct < 0.5:
            return MonitorAction(
                action_type=ActionType.ESCALATE_TO_DQN,
                trade=trade,
                reason=f"Position seit {hours:.0f}h offen mit nur {profit_pct:.2f}% Gewinn",
                urgency="medium",
            )
        return None

    # ── Regel 4: Volatilitäts-Alarm ───────────────────────────────────────────

    def _check_volatility(self, trade: Trade) -> Optional[MonitorAction]:
        change_30m = _price_tracker.get_change_pct(trade.epic, 30)
        abs_change = abs(change_30m)

        if abs_change > 3.0:
            return MonitorAction(
                action_type=ActionType.ESCALATE_TO_DQN,
                trade=trade,
                reason=f"Extreme Volatilität: {trade.epic} {change_30m:+.2f}% in 30 Min",
                urgency="high",
            )
        if abs_change > 2.0:
            return MonitorAction(
                action_type=ActionType.ALERT,
                trade=trade,
                reason=f"Hohe Volatilität: {trade.epic} {change_30m:+.2f}% in 30 Min",
                urgency="medium",
            )
        return None

    # ── Regel 5: Take-Profit Proximity ────────────────────────────────────────

    def _check_tp_proximity(self, trade: Trade, current: float) -> Optional[MonitorAction]:
        if not trade.take_profit:
            return None

        if trade.direction == Direction.BUY:
            total = trade.take_profit - trade.entry_price
            progress = current - trade.entry_price
        else:
            total = trade.entry_price - trade.take_profit
            progress = trade.entry_price - current

        if total <= 0:
            return None

        pct = progress / total
        if pct >= 0.8:
            return MonitorAction(
                action_type=ActionType.ALERT,
                trade=trade,
                reason=(
                    f"Take-Profit fast erreicht: {current:.4f} "
                    f"({pct*100:.0f}% der Strecke, TP={trade.take_profit:.4f})"
                ),
                urgency="low",
            )
        return None

    # ── Regel 6: Spread-Anomalie ──────────────────────────────────────────────

    def _check_spread(self, trade: Trade, snapshot: PriceSnapshot) -> Optional[MonitorAction]:
        avg = _price_tracker.get_average_spread(trade.epic)
        if avg <= 0:
            return None
        if snapshot.spread > avg * 3:
            return MonitorAction(
                action_type=ActionType.ALERT,
                trade=trade,
                reason=(
                    f"Ungewöhnlich hoher Spread bei {trade.epic}: "
                    f"{snapshot.spread:.4f} (Durchschnitt: {avg:.4f})"
                ),
                urgency="low",
            )
        return None
