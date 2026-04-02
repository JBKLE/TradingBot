"""Trading strategy – validates signals and calculates position sizing."""
import logging
from typing import Optional

from . import config
from .models import (
    AnalysisResult,
    Direction,
    PriceBar,
    Recommendation,
    TradeSignal,
    ValidationResult,
)

logger = logging.getLogger(__name__)


class TradingStrategy:
    """Validates DQN's recommendation and decides whether to execute."""

    def validate_signal(
        self,
        analysis: AnalysisResult,
        open_positions_count: int = 0,
        account_balance: float = 0.0,
        **kwargs,
    ) -> ValidationResult:
        """
        Returns (valid=True) if ALL conditions pass:

        1. Kill-switch not engaged
        2. Recommendation is TRADE
        3. Direction is BUY or SELL (not NONE)
        4. Confidence >= MIN_CONFIDENCE_SCORE
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

        logger.info(
            "Signal validated: %s %s confidence=%d",
            opp.asset,
            opp.direction.value,
            opp.confidence,
        )
        return ValidationResult(valid=True)

    def build_signal(
        self,
        analysis: AnalysisResult,
        account_balance: float,
        price_bars: list[PriceBar] | None = None,
    ) -> Optional[TradeSignal]:
        """Build a TradeSignal from a validated AnalysisResult."""
        opp = analysis.best_opportunity
        epic = config.WATCHLIST.get(opp.asset, {}).get("epic", opp.asset)

        stop_loss = opp.stop_loss
        take_profit = opp.take_profit

        sl_distance = abs(opp.entry_price - stop_loss) if stop_loss else 0
        size = self.calculate_position_size(account_balance, sl_distance, config.MAX_RISK_PER_TRADE_PCT)

        tp_distance = abs(opp.entry_price - take_profit) if take_profit else 0
        rr_ratio = (tp_distance / sl_distance) if sl_distance > 0 else opp.risk_reward_ratio

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
        """
        if stop_loss_distance <= 0 or balance <= 0:
            return 0.01

        max_risk_amount = balance * (max_risk_pct / 100.0)
        size = max_risk_amount / stop_loss_distance

        size = max(0.01, round(size, 2))
        logger.debug(
            "Position size: balance=%.2f sl_dist=%.4f risk=%.2f%% -> size=%.2f",
            balance,
            stop_loss_distance,
            max_risk_pct,
            size,
        )
        return size
