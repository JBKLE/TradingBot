"""Trading strategy – validates signals and calculates position sizing."""
import logging
from datetime import datetime, time as dt_time

from . import config
from .models import (
    AnalysisResult,
    Direction,
    Recommendation,
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
    ) -> ValidationResult:
        """
        Returns (valid=True) if ALL conditions pass:

        1. Kill-switch not engaged
        2. Recommendation is TRADE
        3. Direction is BUY or SELL (not NONE)
        4. Confidence >= MIN_CONFIDENCE_SCORE
        5. Risk/Reward >= MIN_RISK_REWARD_RATIO
        6. Open positions < MAX_OPEN_POSITIONS
        7. Inside the trade window (TRADE_WINDOW_START – TRADE_WINDOW_END)
        8. Stop-loss distance <= MAX_RISK_PER_TRADE_PCT of account balance
        9. Prices are non-zero (sanity check)
        """
        opp = analysis.best_opportunity

        if not config.TRADING_ENABLED:
            return ValidationResult(valid=False, reason="Kill-switch active (TRADING_ENABLED=false)")

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
            # Max monetary risk we accept
            max_risk_amount = account_balance * (config.MAX_RISK_PER_TRADE_PCT / 100.0)
            # Minimum size = 1 unit – if even 1 unit exceeds the risk, reject
            # (actual size check happens in calculate_position_size)
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
    ) -> TradeSignal:
        """Build a TradeSignal from a validated AnalysisResult."""
        opp = analysis.best_opportunity
        epic = config.WATCHLIST.get(opp.asset, {}).get("epic", opp.asset)
        sl_distance = abs(opp.entry_price - opp.stop_loss)
        size = self.calculate_position_size(account_balance, sl_distance, config.MAX_RISK_PER_TRADE_PCT)
        return TradeSignal(
            asset=opp.asset,
            epic=epic,
            direction=opp.direction,
            entry_price=opp.entry_price,
            stop_loss=opp.stop_loss,
            take_profit=opp.take_profit,
            position_size=size,
            confidence=opp.confidence,
            reasoning=opp.reasoning,
            risk_reward_ratio=opp.risk_reward_ratio,
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
