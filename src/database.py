"""SQLite database layer for trade history and performance tracking."""
import json
import logging
from datetime import datetime
from typing import Optional

import aiosqlite

from . import config
from .models import AnalysisResult, Trade, TradeStatus

logger = logging.getLogger(__name__)

CREATE_TRADES_TABLE = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    asset TEXT NOT NULL,
    epic TEXT NOT NULL DEFAULT '',
    direction TEXT NOT NULL,
    entry_price REAL NOT NULL,
    stop_loss REAL NOT NULL,
    take_profit REAL NOT NULL,
    position_size REAL NOT NULL,
    confidence INTEGER NOT NULL,
    reasoning TEXT,
    deal_id TEXT,
    status TEXT DEFAULT 'OPEN',
    exit_price REAL,
    exit_timestamp TEXT,
    profit_loss REAL,
    profit_loss_pct REAL
);
"""

CREATE_ANALYSES_TABLE = """
CREATE TABLE IF NOT EXISTS daily_analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    market_summary TEXT,
    recommendation TEXT,
    raw_analysis TEXT,
    tokens_used INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0.0
);
"""

CREATE_SNAPSHOTS_TABLE = """
CREATE TABLE IF NOT EXISTS account_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    balance REAL NOT NULL,
    equity REAL NOT NULL,
    open_positions INTEGER DEFAULT 0
);
"""


async def init_db() -> None:
    """Create all tables if they don't exist."""
    async with aiosqlite.connect(config.DB_PATH) as db:
        await db.execute(CREATE_TRADES_TABLE)
        await db.execute(CREATE_ANALYSES_TABLE)
        await db.execute(CREATE_SNAPSHOTS_TABLE)
        await db.commit()
    logger.info("Database initialised at %s", config.DB_PATH)


async def save_trade(trade: Trade) -> int:
    """Insert a new trade and return its database ID."""
    async with aiosqlite.connect(config.DB_PATH) as db:
        cursor = await db.execute(
            """
            INSERT INTO trades
                (timestamp, asset, epic, direction, entry_price, stop_loss,
                 take_profit, position_size, confidence, reasoning, deal_id, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade.timestamp.isoformat(),
                trade.asset,
                trade.epic,
                trade.direction.value,
                trade.entry_price,
                trade.stop_loss,
                trade.take_profit,
                trade.position_size,
                trade.confidence,
                trade.reasoning,
                trade.deal_id,
                trade.status.value,
            ),
        )
        await db.commit()
        row_id = cursor.lastrowid
        logger.info("Trade saved with ID %s (asset=%s, direction=%s)", row_id, trade.asset, trade.direction)
        return row_id


async def update_trade_closed(
    trade_id: int,
    exit_price: float,
    profit_loss: float,
    profit_loss_pct: float,
    status: TradeStatus,
) -> None:
    """Update a trade record when it closes."""
    async with aiosqlite.connect(config.DB_PATH) as db:
        await db.execute(
            """
            UPDATE trades
            SET status = ?, exit_price = ?, exit_timestamp = ?,
                profit_loss = ?, profit_loss_pct = ?
            WHERE id = ?
            """,
            (
                status.value,
                exit_price,
                datetime.now().isoformat(),
                profit_loss,
                profit_loss_pct,
                trade_id,
            ),
        )
        await db.commit()
    logger.info("Trade %s closed – P/L: %.2f (%.2f%%)", trade_id, profit_loss, profit_loss_pct)


async def get_trades_today() -> list[Trade]:
    """Return all trades opened today."""
    today = datetime.now().date().isoformat()
    async with aiosqlite.connect(config.DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM trades WHERE timestamp LIKE ?",
            (f"{today}%",),
        ) as cursor:
            rows = await cursor.fetchall()
    return [_row_to_trade(r) for r in rows]


async def get_open_trades() -> list[Trade]:
    """Return all currently open trades."""
    async with aiosqlite.connect(config.DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM trades WHERE status = 'OPEN'",
        ) as cursor:
            rows = await cursor.fetchall()
    return [_row_to_trade(r) for r in rows]


async def save_analysis(analysis: AnalysisResult) -> None:
    """Persist a Claude market analysis."""
    async with aiosqlite.connect(config.DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO daily_analyses
                (timestamp, market_summary, recommendation, raw_analysis,
                 tokens_used, cost_usd)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(),
                analysis.market_summary,
                analysis.recommendation.value,
                analysis.model_dump_json(),
                analysis.tokens_used,
                analysis.cost_usd,
            ),
        )
        await db.commit()
    logger.debug("Analysis saved (recommendation=%s)", analysis.recommendation)


async def update_trade_stop_loss(trade_id: int, stop_loss: float) -> None:
    """Update the stop_loss of an open trade (e.g. after trailing)."""
    async with aiosqlite.connect(config.DB_PATH) as db:
        await db.execute(
            "UPDATE trades SET stop_loss = ? WHERE id = ?",
            (stop_loss, trade_id),
        )
        await db.commit()
    logger.debug("Trade %s stop_loss updated to %.4f", trade_id, stop_loss)


async def get_latest_balance() -> float:
    """Return the most recent account balance from snapshots, or 0.0 if none."""
    today = datetime.now().date().isoformat()
    async with aiosqlite.connect(config.DB_PATH) as db:
        async with db.execute(
            "SELECT balance FROM account_snapshots WHERE timestamp LIKE ? ORDER BY timestamp DESC LIMIT 1",
            (f"{today}%",),
        ) as cursor:
            row = await cursor.fetchone()
    return float(row[0]) if row else 0.0


async def save_account_snapshot(balance: float, equity: float, open_positions: int) -> None:
    """Save a point-in-time account snapshot."""
    async with aiosqlite.connect(config.DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO account_snapshots (timestamp, balance, equity, open_positions)
            VALUES (?, ?, ?, ?)
            """,
            (datetime.now().isoformat(), balance, equity, open_positions),
        )
        await db.commit()


def _row_to_trade(row: aiosqlite.Row) -> Trade:
    from .models import Direction, TradeStatus

    return Trade(
        id=row["id"],
        timestamp=datetime.fromisoformat(row["timestamp"]),
        asset=row["asset"],
        epic=row["epic"] or "",
        direction=Direction(row["direction"]),
        entry_price=row["entry_price"],
        stop_loss=row["stop_loss"],
        take_profit=row["take_profit"],
        position_size=row["position_size"],
        confidence=row["confidence"],
        reasoning=row["reasoning"] or "",
        deal_id=row["deal_id"],
        status=TradeStatus(row["status"]),
        exit_price=row["exit_price"],
        exit_timestamp=(
            datetime.fromisoformat(row["exit_timestamp"])
            if row["exit_timestamp"]
            else None
        ),
        profit_loss=row["profit_loss"],
        profit_loss_pct=row["profit_loss_pct"],
    )
