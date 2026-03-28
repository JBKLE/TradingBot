"""SQLite database layer for the simulation engine (price history + sim trades)."""
import logging
from datetime import datetime
from typing import Optional

import aiosqlite

from . import config
from .sim_models import PriceRecord, SimDirection, SimTrade, SimTradeStatus, SlVariant

logger = logging.getLogger(__name__)

# ── DDL ───────────────────────────────────────────────────────────────────────

CREATE_PRICE_HISTORY_TABLE = """
CREATE TABLE IF NOT EXISTS price_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    asset TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL DEFAULT 0.0
);
"""

CREATE_PRICE_HISTORY_INDEX = """
CREATE INDEX IF NOT EXISTS idx_price_history_asset_ts
ON price_history (asset, timestamp DESC);
"""

CREATE_SIM_TRADES_TABLE = """
CREATE TABLE IF NOT EXISTS sim_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset TEXT NOT NULL,
    direction TEXT NOT NULL,
    sl_variant TEXT NOT NULL,
    entry_timestamp TEXT NOT NULL,
    entry_price REAL NOT NULL,
    sl_price REAL NOT NULL,
    tp_price REAL NOT NULL,
    exit_timestamp TEXT,
    exit_price REAL,
    status TEXT NOT NULL DEFAULT 'open',
    pnl REAL,
    r_multiple REAL
);
"""

CREATE_SIM_TRADES_STATUS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_sim_trades_status
ON sim_trades (status);
"""

CREATE_SIM_TRADES_ASSET_STATUS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_sim_trades_asset_status
ON sim_trades (asset, status);
"""


# ── Initialisation ────────────────────────────────────────────────────────────

async def init_sim_db() -> None:
    """Create simulation tables and set WAL mode for concurrent access."""
    async with aiosqlite.connect(config.SIM_DB_PATH) as db:
        await db.execute(CREATE_PRICE_HISTORY_TABLE)
        await db.execute(CREATE_PRICE_HISTORY_INDEX)
        await db.execute(CREATE_SIM_TRADES_TABLE)
        await db.execute(CREATE_SIM_TRADES_STATUS_INDEX)
        await db.execute(CREATE_SIM_TRADES_ASSET_STATUS_INDEX)
        await db.execute("PRAGMA journal_mode=WAL")
        await db.commit()
    logger.info("Simulation database initialised at %s", config.SIM_DB_PATH)


# ── Price history ─────────────────────────────────────────────────────────────

async def batch_insert_prices(records: list[tuple]) -> None:
    """Insert multiple price_history rows in one transaction.

    Each tuple: (timestamp_iso, asset, open, high, low, close, volume)
    """
    async with aiosqlite.connect(config.SIM_DB_PATH) as db:
        await db.executemany(
            """INSERT INTO price_history
               (timestamp, asset, open, high, low, close, volume)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            records,
        )
        await db.commit()


async def get_latest_price(asset: str) -> Optional[PriceRecord]:
    """Return the most recent price_history row for an asset."""
    async with aiosqlite.connect(config.SIM_DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM price_history WHERE asset = ? ORDER BY timestamp DESC LIMIT 1",
            (asset,),
        ) as cursor:
            row = await cursor.fetchone()
    if not row:
        return None
    return PriceRecord(
        id=row["id"],
        timestamp=datetime.fromisoformat(row["timestamp"]),
        asset=row["asset"],
        open=row["open"],
        high=row["high"],
        low=row["low"],
        close=row["close"],
        volume=row["volume"],
    )


# ── Sim trades ────────────────────────────────────────────────────────────────

async def batch_insert_sim_trades(trades: list[tuple]) -> None:
    """Insert multiple sim_trades in one transaction.

    Each tuple: (asset, direction, sl_variant, entry_timestamp_iso,
                 entry_price, sl_price, tp_price)
    """
    async with aiosqlite.connect(config.SIM_DB_PATH) as db:
        await db.executemany(
            """INSERT INTO sim_trades
               (asset, direction, sl_variant, entry_timestamp,
                entry_price, sl_price, tp_price, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'open')""",
            trades,
        )
        await db.commit()


async def get_open_sim_trades() -> list[SimTrade]:
    """Fetch all sim_trades with status='open'."""
    async with aiosqlite.connect(config.SIM_DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM sim_trades WHERE status = 'open'"
        ) as cursor:
            rows = await cursor.fetchall()
    return [_row_to_sim_trade(r) for r in rows]


async def get_assets_with_open_trades() -> set[str]:
    """Return the set of asset names that have at least one open sim trade."""
    async with aiosqlite.connect(config.SIM_DB_PATH) as db:
        async with db.execute(
            "SELECT DISTINCT asset FROM sim_trades WHERE status = 'open'"
        ) as cursor:
            rows = await cursor.fetchall()
    return {row[0] for row in rows}


async def batch_close_sim_trades(updates: list[tuple]) -> None:
    """Batch-update closed sim trades.

    Each tuple: (status, exit_timestamp_iso, exit_price, pnl, r_multiple, id)
    """
    if not updates:
        return
    async with aiosqlite.connect(config.SIM_DB_PATH) as db:
        await db.executemany(
            """UPDATE sim_trades
               SET status = ?, exit_timestamp = ?,
                   exit_price = ?, pnl = ?, r_multiple = ?
               WHERE id = ?""",
            updates,
        )
        await db.commit()


async def get_sim_trade_stats() -> dict:
    """Return basic statistics about sim trades (for logging)."""
    async with aiosqlite.connect(config.SIM_DB_PATH) as db:
        async with db.execute(
            "SELECT status, COUNT(*) FROM sim_trades GROUP BY status"
        ) as cursor:
            rows = await cursor.fetchall()
    return {row[0]: row[1] for row in rows}


async def get_sim_trade_by_id(trade_id: int) -> Optional[SimTrade]:
    """Fetch a single sim trade by ID."""
    async with aiosqlite.connect(config.SIM_DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM sim_trades WHERE id = ?", (trade_id,)
        ) as cursor:
            row = await cursor.fetchone()
    return _row_to_sim_trade(row) if row else None


async def get_closed_sim_trades(limit: int = 100) -> list[SimTrade]:
    """Fetch recently closed sim trades (for backtest selection)."""
    async with aiosqlite.connect(config.SIM_DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM sim_trades WHERE status != 'open' "
            "ORDER BY exit_timestamp DESC LIMIT ?",
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
    return [_row_to_sim_trade(r) for r in rows]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _row_to_sim_trade(row: aiosqlite.Row) -> SimTrade:
    return SimTrade(
        id=row["id"],
        asset=row["asset"],
        direction=SimDirection(row["direction"]),
        sl_variant=SlVariant(row["sl_variant"]),
        entry_timestamp=datetime.fromisoformat(row["entry_timestamp"]),
        entry_price=row["entry_price"],
        sl_price=row["sl_price"],
        tp_price=row["tp_price"],
        exit_timestamp=(
            datetime.fromisoformat(row["exit_timestamp"])
            if row["exit_timestamp"]
            else None
        ),
        exit_price=row["exit_price"],
        status=SimTradeStatus(row["status"]),
        pnl=row["pnl"],
        r_multiple=row["r_multiple"],
    )
