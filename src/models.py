"""Pydantic data models for the DQN trading bot."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Direction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    NONE = "NONE"


class Recommendation(str, Enum):
    TRADE = "TRADE"
    WAIT = "WAIT"


class TradeStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    STOPPED_OUT = "STOPPED_OUT"
    TAKE_PROFIT = "TAKE_PROFIT"
    CANCELLED = "CANCELLED"


class MarketPrice(BaseModel):
    epic: str
    bid: float
    ask: float
    high: float
    low: float
    change_pct: float = 0.0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2


class PriceBar(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float


class MarketData(BaseModel):
    epic: str
    name: str
    current_price: MarketPrice
    price_history: list[PriceBar] = Field(default_factory=list)
    daily_price_history: list[PriceBar] = Field(default_factory=list)  # Tages-Kerzen für übergeordneten Trend

    @property
    def change_24h_pct(self) -> float:
        return self.current_price.change_pct


class AccountInfo(BaseModel):
    balance: float
    equity: float
    available: float
    currency: str = "EUR"


class PositionInfo(BaseModel):
    deal_id: str
    epic: str
    direction: Direction
    size: float
    entry_price: float
    current_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    profit_loss: float = 0.0
    profit_loss_pct: float = 0.0


class AssetOutlook(BaseModel):
    asset: str
    outlook: str
    confidence: int = Field(ge=1, le=10)
    note: str = ""


class RecheckInfo(BaseModel):
    worthy: bool = False
    asset: str = ""
    direction: Direction = Direction.NONE
    trigger_condition: str = ""
    recheck_in_minutes: int = 60
    current_confidence: int = 0
    expected_confidence_if_trigger: int = 0


class BestOpportunity(BaseModel):
    asset: str
    direction: Direction
    confidence: int = Field(ge=1, le=10)
    reasoning: str
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    key_events: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    date: str
    market_summary: str
    best_opportunity: BestOpportunity
    other_assets: list[AssetOutlook] = Field(default_factory=list)
    recommendation: Recommendation
    wait_reason: Optional[str] = None
    recheck: Optional[RecheckInfo] = None
    tokens_used: int = 0
    cost_usd: float = 0.0


class TradeSignal(BaseModel):
    asset: str
    epic: str
    direction: Direction
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    confidence: int
    reasoning: str
    risk_reward_ratio: float


class Trade(BaseModel):
    id: Optional[int] = None
    timestamp: datetime
    asset: str
    epic: str = ""
    direction: Direction
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    confidence: int
    reasoning: str
    status: TradeStatus = TradeStatus.OPEN
    deal_id: Optional[str] = None
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    profit_loss: Optional[float] = None
    profit_loss_pct: Optional[float] = None


class TradeResult(BaseModel):
    success: bool
    deal_id: Optional[str] = None
    trade: Optional[Trade] = None
    error: Optional[str] = None


class ValidationResult(BaseModel):
    valid: bool
    reason: Optional[str] = None


class EscalationResult(BaseModel):
    action: str  # HOLD | CLOSE | ADJUST_SL | ADJUST_TP
    reasoning: str
    new_stop_loss: Optional[float] = None
    new_take_profit: Optional[float] = None
    urgency: str = "low"


class PendingRecheck(BaseModel):
    id: Optional[int] = None
    created_at: datetime
    asset: str
    epic: str
    direction: Direction
    trigger_condition: str
    recheck_at: datetime
    recheck_count: int = 0
    max_rechecks: int = 3
    current_confidence: int = 0
    original_analysis: str = ""
    status: str = "PENDING"
    resolved_at: Optional[datetime] = None


class TradeReview(BaseModel):
    id: Optional[int] = None
    trade_id: int
    review_timestamp: datetime
    entry_quality: str = ""
    sl_quality: str = ""
    market_condition: str = ""
    what_happened_after: str = ""
    lesson_learned: str = ""
    raw_review: str = ""
