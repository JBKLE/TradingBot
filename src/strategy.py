"""Trading strategy – validates signals and calculates position sizing."""
import logging
from datetime import datetime, time as dt_time, timedelta
from typing import Optional

from . import config
from .models import (
    AnalysisResult,
    Direction,
    PriceBar,
    Recommendation,
    Trade,
    TradeStatus,
    TradeSignal,
    ValidationResult,
)

logger = logging.getLogger(__name__)


class TradingStrategy:
    """Validates Claude's recommendation and decides whether to execute."""

    def validate_signal(
        self,
        analysis: AnalysisResult,
        open_positions_count: int,
        account_balance: float,
        open_positions: list | None = None,
        price_bars: list[PriceBar] | None = None,
        trades_today: list[Trade] | None = None,
        last_closed_trade: Optional[Trade] = None,
    ) -> ValidationResult:
        """
        Returns (valid=True) if ALL conditions pass:

        1. Kill-switch not engaged
        2. Recommendation is TRADE
        3. Direction is BUY or SELL (not NONE)
        4. Confidence >= MIN_CONFIDENCE_SCORE
        5. Risk/Reward >= MIN_RISK_REWARD_RATIO
        6. Open positions < MAX_OPEN_POSITIONS
        7. ATR-Volatilitäts-Gate (kein Trade bei extremer Volatilität)
        8. Max Trades per Day nicht überschritten
        9. Cooldown nach Verlust einhalten
        10. Inside the trade window (TRADE_WINDOW_START – TRADE_WINDOW_END)
        11. Stop-loss distance <= MAX_RISK_PER_TRADE_PCT of account balance
        12. Prices are non-zero (sanity check)
        """
        opp = analysis.best_opportunity

        if not config.TRADING_ENABLED:
            return ValidationResult(valid=False, reason="Kill-switch active (TRADING_ENABLED=false)")

        # ── Max-Drawdown Schutz (Tages-Limit) ───────────────────────────────
        if trades_today:
            daily_pl = sum(t.profit_loss for t in trades_today if t.profit_loss)
            if daily_pl < 0 and account_balance > 0:
                daily_drawdown_pct = abs(daily_pl) / account_balance * 100
                if daily_drawdown_pct >= config.MAX_DAILY_DRAWDOWN_PCT:
                    return ValidationResult(
                        valid=False,
                        reason=(
                            f"Max Tages-Drawdown erreicht: {daily_drawdown_pct:.1f}% "
                            f"(Limit: {config.MAX_DAILY_DRAWDOWN_PCT}%, "
                            f"Verlust: {daily_pl:+.2f} EUR)"
                        ),
                    )

        # ── Verlustserie-Erkennung ──────────────────────────────────────────
        if trades_today:
            consecutive_losses = 0
            for t in sorted(trades_today, key=lambda x: x.timestamp, reverse=True):
                if t.status == TradeStatus.STOPPED_OUT:
                    consecutive_losses += 1
                elif t.profit_loss is not None:
                    break
            if consecutive_losses >= 3:
                return ValidationResult(
                    valid=False,
                    reason=(
                        f"Verlustserie: {consecutive_losses} Stop-Losses in Folge – "
                        f"Trading fuer heute pausiert"
                    ),
                )

        if analysis.recommendation != Recommendation.TRADE:
            return ValidationResult(valid=False, reason=f"Recommendation is WAIT: {analysis.wait_reason}")

        if opp.direction == Direction.NONE:
            return ValidationResult(valid=False, reason="Direction is NONE")

        if opp.confidence < config.MIN_CONFIDENCE_SCORE:
            return ValidationResult(
                valid=False,
                reason=f"Confidence {opp.confidence} < minimum {config.MIN_CONFIDENCE_SCORE}",
            )

        if opp.risk_reward_ratio < config.MIN_RISK_REWARD_RATIO:
            return ValidationResult(
                valid=False,
                reason=f"Risk/Reward {opp.risk_reward_ratio:.2f} < minimum {config.MIN_RISK_REWARD_RATIO}",
            )

        if open_positions_count >= config.MAX_OPEN_POSITIONS:
            return ValidationResult(
                valid=False,
                reason=f"Max open positions reached ({open_positions_count}/{config.MAX_OPEN_POSITIONS})",
            )

        # Kein zweiter Trade auf dasselbe Asset
        if open_positions:
            epic = config.WATCHLIST.get(opp.asset, {}).get("epic", opp.asset)
            already_open = [p for p in open_positions if p.epic == epic]
            if already_open:
                return ValidationResult(
                    valid=False,
                    reason=f"{opp.asset} hat bereits eine offene Position",
                )

        # ── ATR-Volatilitäts-Gate ─────────────────────────────────────────────
        if price_bars:
            from .indicators import calculate_atr, calculate_average_atr
            current_atr = calculate_atr(price_bars, period=config.ATR_LOOKBACK_PERIOD)
            avg_atr = calculate_average_atr(price_bars, period=config.ATR_LOOKBACK_PERIOD)
            if current_atr and avg_atr and avg_atr > 0:
                atr_ratio = current_atr / avg_atr
                if atr_ratio > config.MAX_ATR_MULTIPLIER:
                    return ValidationResult(
                        valid=False,
                        reason=(
                            f"Volatilität zu hoch: ATR={current_atr:.4f} "
                            f"({atr_ratio:.2f}x Durchschnitt={avg_atr:.4f}, "
                            f"Limit={config.MAX_ATR_MULTIPLIER}x)"
                        ),
                    )
                logger.info(
                    "ATR-Gate OK: ATR=%.4f (%.2fx Durchschnitt=%.4f)",
                    current_atr, atr_ratio, avg_atr,
                )

            # ── ATR-basierte Konto-Validierung ──────────────────────────────
            if current_atr and current_atr > 0:
                min_sl_distance = current_atr * config.SL_ATR_MULTIPLIER
                max_risk_amount = account_balance * (config.MAX_RISK_PER_TRADE_PCT / 100.0)
                if min_sl_distance > 0:
                    implied_size = max_risk_amount / min_sl_distance
                    if implied_size < 0.01:
                        return ValidationResult(
                            valid=False,
                            reason=(
                                f"Konto zu klein fuer aktuelle Volatilitaet: {opp.asset} "
                                f"ATR-SL={min_sl_distance:.2f}, max Risiko={max_risk_amount:.2f} EUR, "
                                f"Positionsgroesse waere {implied_size:.4f} (min: 0.01)"
                            ),
                        )

        # ── Max Trades per Day ────────────────────────────────────────────────
        if trades_today is not None and len(trades_today) >= config.MAX_TRADES_PER_DAY:
            return ValidationResult(
                valid=False,
                reason=(
                    f"Max Trades/Tag erreicht ({len(trades_today)}/{config.MAX_TRADES_PER_DAY})"
                ),
            )

        # ── Cooldown nach Verlust ─────────────────────────────────────────────
        if last_closed_trade and last_closed_trade.status == TradeStatus.STOPPED_OUT:
            if last_closed_trade.exit_timestamp:
                now = datetime.now(tz=config.TZ)
                exit_ts = last_closed_trade.exit_timestamp
                if exit_ts.tzinfo is None:
                    exit_ts = exit_ts.replace(tzinfo=config.TZ)
                minutes_since = (now - exit_ts).total_seconds() / 60
                if minutes_since < config.COOLDOWN_AFTER_LOSS_MINUTES:
                    remaining = int(config.COOLDOWN_AFTER_LOSS_MINUTES - minutes_since)
                    return ValidationResult(
                        valid=False,
                        reason=(
                            f"Cooldown nach Stop-Loss aktiv: noch {remaining} Min "
                            f"(Cooldown={config.COOLDOWN_AFTER_LOSS_MINUTES} Min)"
                        ),
                    )

        # ── Kein Richtungswechsel auf dasselbe Asset am selben Tag ───────────
        if trades_today:
            for t in trades_today:
                if t.asset == opp.asset and t.direction != opp.direction:
                    return ValidationResult(
                        valid=False,
                        reason=(
                            f"Richtungswechsel auf {opp.asset} am selben Tag nicht erlaubt "
                            f"(vorher: {t.direction.value}, jetzt: {opp.direction.value})"
                        ),
                    )

        # ── Asset-Cooldown nach Stop-Loss ────────────────────────────────────
        if trades_today:
            for t in trades_today:
                if t.asset == opp.asset and t.status == TradeStatus.STOPPED_OUT:
                    return ValidationResult(
                        valid=False,
                        reason=f"Asset-Cooldown: {opp.asset} wurde heute bereits ausgestoppt",
                    )

        if not self._within_trade_window():
            return ValidationResult(
                valid=False,
                reason=f"Outside trade window {config.TRADE_WINDOW_START}–{config.TRADE_WINDOW_END}",
            )

        if opp.entry_price <= 0 or opp.stop_loss <= 0 or opp.take_profit <= 0:
            return ValidationResult(valid=False, reason="Invalid price levels (zero or negative)")

        # Verify stop-loss risk does not exceed account limit
        sl_distance = abs(opp.entry_price - opp.stop_loss)
        if sl_distance > 0 and account_balance > 0:
            # Here we just verify the percentage-distance is plausible
            sl_pct_of_entry = (sl_distance / opp.entry_price) * 100
            if sl_pct_of_entry > 20:
                return ValidationResult(
                    valid=False,
                    reason=f"Stop-loss distance {sl_pct_of_entry:.1f}% of entry price seems too wide",
                )

        logger.info(
            "Signal validated: %s %s confidence=%d RR=%.2f",
            opp.asset,
            opp.direction.value,
            opp.confidence,
            opp.risk_reward_ratio,
        )
        return ValidationResult(valid=True)

    def build_signal(
        self,
        analysis: AnalysisResult,
        account_balance: float,
        price_bars: list[PriceBar] | None = None,
    ) -> Optional[TradeSignal]:
        """Build a TradeSignal from a validated AnalysisResult. Returns None if final RR is too low."""
        opp = analysis.best_opportunity
        epic = config.WATCHLIST.get(opp.asset, {}).get("epic", opp.asset)

        # ── ATR-basierter SL/TP ───────────────────────────────────────────────
        stop_loss = opp.stop_loss
        take_profit = opp.take_profit

        if config.USE_ATR_BASED_SL and price_bars:
            from .indicators import calculate_atr
            atr = calculate_atr(price_bars)
            if atr and atr > 0:
                if opp.direction == Direction.BUY:
                    stop_loss = opp.entry_price - (atr * config.SL_ATR_MULTIPLIER)
                    take_profit = opp.entry_price + (atr * config.TP_ATR_MULTIPLIER)
                else:
                    stop_loss = opp.entry_price + (atr * config.SL_ATR_MULTIPLIER)
                    take_profit = opp.entry_price - (atr * config.TP_ATR_MULTIPLIER)
                logger.info(
                    "SL: Claude=%.4f → ATR-basiert=%.4f (ATR=%.4f × %.1f)",
                    opp.stop_loss, stop_loss, atr, config.SL_ATR_MULTIPLIER,
                )
                logger.info(
                    "TP: Claude=%.4f → ATR-basiert=%.4f (ATR=%.4f × %.1f)",
                    opp.take_profit, take_profit, atr, config.TP_ATR_MULTIPLIER,
                )

        sl_distance = abs(opp.entry_price - stop_loss)
        size = self.calculate_position_size(account_balance, sl_distance, config.MAX_RISK_PER_TRADE_PCT)

        # Neu berechnetes RR-Verhaeltnis nach ATR-Override
        tp_distance = abs(opp.entry_price - take_profit)
        rr_ratio = (tp_distance / sl_distance) if sl_distance > 0 else opp.risk_reward_ratio

        # ── Finaler RR-Check: ATR-Override kann RR verschlechtern ────────
        if rr_ratio < config.MIN_RISK_REWARD_RATIO:
            logger.warning(
                "RR nach ATR-Override zu niedrig: %.2f < %.2f – fallback auf Claude's SL/TP",
                rr_ratio, config.MIN_RISK_REWARD_RATIO,
            )
            stop_loss = opp.stop_loss
            take_profit = opp.take_profit
            sl_distance = abs(opp.entry_price - stop_loss)
            tp_distance = abs(opp.entry_price - take_profit)
            rr_ratio = (tp_distance / sl_distance) if sl_distance > 0 else opp.risk_reward_ratio
            size = self.calculate_position_size(
                account_balance, sl_distance, config.MAX_RISK_PER_TRADE_PCT,
            )

            if rr_ratio < config.MIN_RISK_REWARD_RATIO:
                logger.error(
                    "Auch Claude's RR=%.2f unter Minimum %.2f – Trade abgelehnt",
                    rr_ratio, config.MIN_RISK_REWARD_RATIO,
                )
                return None

        return TradeSignal(
            asset=opp.asset,
            epic=epic,
            direction=opp.direction,
            entry_price=opp.entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=size,
            confidence=opp.confidence,
            reasoning=opp.reasoning,
            risk_reward_ratio=rr_ratio,
        )

    @staticmethod
    def calculate_position_size(
        balance: float,
        stop_loss_distance: float,
        max_risk_pct: float,
    ) -> float:
        """
        Position size = (balance × max_risk_pct%) / stop_loss_distance

        Minimum size is 0.01 (1 micro-lot equivalent).
        Maximum size is capped so we never risk more than max_risk_pct%.
        """
        if stop_loss_distance <= 0 or balance <= 0:
            return 0.01

        max_risk_amount = balance * (max_risk_pct / 100.0)
        size = max_risk_amount / stop_loss_distance

        # Round down to 2 decimal places and enforce a sensible minimum
        size = max(0.01, round(size, 2))
        logger.debug(
            "Position size: balance=%.2f sl_dist=%.4f risk=%.2f%% -> size=%.2f",
            balance,
            stop_loss_distance,
            max_risk_pct,
            size,
        )
        return size

    @staticmethod
    def _within_trade_window() -> bool:
        """Return True if current time is inside the configured trade window."""
        now = datetime.now(tz=config.TZ).time()
        start_h, start_m = map(int, config.TRADE_WINDOW_START.split(":"))
        end_h, end_m = map(int, config.TRADE_WINDOW_END.split(":"))
        window_start = dt_time(start_h, start_m)
        window_end = dt_time(end_h, end_m)
        return window_start <= now <= window_end
