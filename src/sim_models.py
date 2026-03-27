"""Pydantic data models for the simulation engine."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class SlVariant(str, Enum):
    TIGHT = "tight"
    MEDIUM = "medium"
    WIDE = "wide"


class SimDirection(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class SimTradeStatus(str, Enum):
    OPEN = "open"
    CLOSED_TP = "closed_tp"
    CLOSED_SL = "closed_sl"


class SlTpConfig(BaseModel):
    variant: SlVariant
    sl_pct: float  # e.g. 0.003 for 0.3%
    tp_pct: float  # e.g. 0.005 for 0.5%


class PriceRecord(BaseModel):
    id: Optional[int] = None
    timestamp: datetime
    asset: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class SimTrade(BaseModel):
    id: Optional[int] = None
    asset: str
    direction: SimDirection
    sl_variant: SlVariant
    entry_timestamp: datetime
    entry_price: float
    sl_price: float
    tp_price: float
    exit_timestamp: Optional[datetime] = None
    exit_price: Optional[float] = None
    status: SimTradeStatus = SimTradeStatus.OPEN
    pnl: Optional[float] = None
    r_multiple: Optional[float] = None
