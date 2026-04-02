"""SQLite database layer for trade history and performance tracking."""
import json
import logging
from datetime import datetime, timedelta
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
    profit_loss_pct REAL,
    model TEXT
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

CREATE_RECHECKS_TABLE = """
CREATE TABLE IF NOT EXISTS pending_rechecks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    asset TEXT NOT NULL,
    epic TEXT NOT NULL,
    direction TEXT NOT NULL,
    trigger_condition TEXT NOT NULL,
    recheck_at TEXT NOT NULL,
    recheck_count INTEGER DEFAULT 0,
    max_rechecks INTEGER DEFAULT 3,
    current_confidence INTEGER DEFAULT 0,
    original_analysis TEXT DEFAULT '',
    status TEXT DEFAULT 'PENDING',
    resolved_at TEXT
);
"""

CREATE_REVIEWS_TABLE = """
CREATE TABLE IF NOT EXISTS trade_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id INTEGER NOT NULL UNIQUE,
    review_timestamp TEXT NOT NULL,
    entry_quality TEXT DEFAULT '',
    sl_quality TEXT DEFAULT '',
    market_condition TEXT DEFAULT '',
    what_happened_after TEXT DEFAULT '',
    lesson_learned TEXT DEFAULT '',
    raw_review TEXT DEFAULT ''
);
"""


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


async def init_db() -> None:
    """Create all tables if they don't exist."""
    async with aiosqlite.connect(config.DB_PATH) as db:
        await db.execute(CREATE_TRADES_TABLE)
        await db.execute(CREATE_ANALYSES_TABLE)
        await db.execute(CREATE_SNAPSHOTS_TABLE)
        await db.execute(CREATE_REVIEWS_TABLE)
        await db.execute(CREATE_RECHECKS_TABLE)
        await db.execute(CREATE_PRICE_HISTORY_TABLE)
        await db.execute(CREATE_PRICE_HISTORY_INDEX)
        # Migration: model-Spalte hinzufuegen falls fehlend
        try:
            await db.execute("ALTER TABLE trades ADD COLUMN model TEXT")
        except Exception:
            pass  # Spalte existiert bereits
        await db.commit()


async def batch_insert_prices(records: list[tuple]) -> None:
    """Insert price_history rows into trades.db."""
    async with aiosqlite.connect(config.DB_PATH) as db:
        await db.executemany(
            """INSERT INTO price_history
               (timestamp, asset, open, high, low, close, volume)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            records,
        )
        await db.commit()
    logger.info("Database initialised at %s", config.DB_PATH)


async def save_trade(trade: Trade) -> int:
    """Insert a new trade and return its database ID."""
    async with aiosqlite.connect(config.DB_PATH) as db:
        cursor = await db.execute(
            """
            INSERT INTO trades
                (timestamp, asset, epic, direction, entry_price, stop_loss,
                 take_profit, position_size, confidence, reasoning, deal_id, status, model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                trade.model,
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
    """Persist a market analysis."""
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


async def get_analyses_today() -> list:
    """Gibt alle Analysen von heute zurück."""
    today = datetime.now().date().isoformat()
    async with aiosqlite.connect(config.DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM daily_analyses WHERE timestamp LIKE ?",
            (f"{today}%",),
        ) as cursor:
            rows = await cursor.fetchall()
    return list(rows)


async def get_trade_by_id(trade_id: int) -> Optional[Trade]:
    """Gibt einen Trade anhand seiner ID zurueck."""
    async with aiosqlite.connect(config.DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM trades WHERE id = ?", (trade_id,),
        ) as cursor:
            row = await cursor.fetchone()
    return _row_to_trade(row) if row else None


async def get_recent_trades(days: int = 7) -> list[Trade]:
    """Gibt alle Trades der letzten N Tage zurueck."""
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    async with aiosqlite.connect(config.DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM trades WHERE timestamp >= ? ORDER BY timestamp DESC",
            (cutoff,),
        ) as cursor:
            rows = await cursor.fetchall()
    return [_row_to_trade(r) for r in rows]


async def get_last_closed_trade() -> Optional[Trade]:
    """Gibt den zuletzt geschlossenen Trade zurück (für Cooldown-Check)."""
    async with aiosqlite.connect(config.DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM trades WHERE status != 'OPEN' ORDER BY exit_timestamp DESC LIMIT 1",
        ) as cursor:
            row = await cursor.fetchone()
    return _row_to_trade(row) if row else None


async def update_trade_deal_id(trade_id: int, new_deal_id: str) -> None:
    """Korrigiert die Deal-ID eines Trades (für Capital.com ID-Mismatch Fix)."""
    async with aiosqlite.connect(config.DB_PATH) as db:
        await db.execute(
            "UPDATE trades SET deal_id = ? WHERE id = ?",
            (new_deal_id, trade_id),
        )
        await db.commit()
    logger.info("Trade %s deal_id korrigiert → %s", trade_id, new_deal_id)


async def save_orphan_trade(pos) -> "Trade":
    """Erstellt einen DB-Eintrag fuer eine verwaiste Broker-Position."""
    from .models import Direction, TradeStatus

    # Asset-Name aus Watchlist ermitteln
    asset_name = pos.epic
    for key, info in config.WATCHLIST.items():
        if info["epic"] == pos.epic:
            asset_name = key
            break

    trade = Trade(
        timestamp=datetime.now(tz=config.TZ),
        asset=asset_name,
        epic=pos.epic,
        direction=pos.direction,
        entry_price=pos.entry_price,
        stop_loss=pos.stop_loss or 0.0,
        take_profit=pos.take_profit or 0.0,
        position_size=pos.size,
        confidence=0,
        reasoning="Orphan: automatisch vom Monitor synchronisiert",
        deal_id=pos.deal_id,
        status=TradeStatus.OPEN,
    )
    trade.id = await save_trade(trade)
    logger.info(
        "Orphan-Trade gespeichert: ID=%s %s %s @ %.4f (deal_id=%s)",
        trade.id, trade.asset, trade.direction.value, trade.entry_price, trade.deal_id,
    )
    return trade


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


async def save_trade_review(trade_id: int, review_data: dict) -> int:
    """Speichert ein Post-Trade Review."""
    async with aiosqlite.connect(config.DB_PATH) as db:
        cursor = await db.execute(
            """INSERT OR REPLACE INTO trade_reviews
                (trade_id, review_timestamp, entry_quality, sl_quality,
                 market_condition, what_happened_after, lesson_learned, raw_review)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trade_id,
                datetime.now().isoformat(),
                review_data.get("entry_quality", ""),
                review_data.get("sl_quality", ""),
                review_data.get("market_condition", ""),
                review_data.get("what_happened_after", ""),
                review_data.get("lesson_learned", ""),
                json.dumps(review_data, ensure_ascii=False),
            ),
        )
        await db.commit()
        logger.info("Trade-Review gespeichert fuer Trade %s", trade_id)
        return cursor.lastrowid


async def get_recent_lessons(limit: int = 10) -> list[dict]:
    """Gibt die letzten N Lessons Learned zurueck (fuer Lern-Kontext)."""
    async with aiosqlite.connect(config.DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT r.lesson_learned, r.entry_quality, r.sl_quality,
                      r.market_condition, t.asset, t.direction, t.status,
                      t.profit_loss, t.profit_loss_pct
            FROM trade_reviews r JOIN trades t ON r.trade_id = t.id
            ORDER BY r.review_timestamp DESC LIMIT ?""",
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def get_performance_stats() -> dict:
    """Aggregierte Performance-Statistiken fuer den Lern-Kontext."""
    stats: dict = {
        "total": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
        "by_asset": {}, "by_direction": {},
        "avg_win": 0.0, "avg_loss": 0.0,
        "current_loss_streak": 0,
    }
    async with aiosqlite.connect(config.DB_PATH) as db:
        # Gesamt-Statistik
        async with db.execute(
            """SELECT COUNT(*),
                SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END),
                SUM(CASE WHEN profit_loss <= 0 THEN 1 ELSE 0 END),
                AVG(CASE WHEN profit_loss > 0 THEN profit_loss END),
                AVG(CASE WHEN profit_loss <= 0 THEN profit_loss END)
            FROM trades WHERE status != 'OPEN' AND profit_loss IS NOT NULL"""
        ) as cursor:
            row = await cursor.fetchone()
            if row and row[0]:
                stats["total"] = row[0]
                stats["wins"] = row[1] or 0
                stats["losses"] = row[2] or 0
                stats["win_rate"] = (stats["wins"] / stats["total"] * 100) if stats["total"] else 0
                stats["avg_win"] = row[3] or 0.0
                stats["avg_loss"] = row[4] or 0.0

        # Per Asset
        async with db.execute(
            """SELECT asset, COUNT(*),
                SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END),
                SUM(profit_loss)
            FROM trades WHERE status != 'OPEN' AND profit_loss IS NOT NULL
            GROUP BY asset"""
        ) as cursor:
            async for row in cursor:
                total = row[1]
                wins = row[2] or 0
                stats["by_asset"][row[0]] = {
                    "total": total, "wins": wins,
                    "total_pl": row[3] or 0.0,
                    "win_rate": (wins / total * 100) if total else 0,
                }

        # Per Direction
        async with db.execute(
            """SELECT direction, COUNT(*),
                SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END),
                SUM(profit_loss)
            FROM trades WHERE status != 'OPEN' AND profit_loss IS NOT NULL
            GROUP BY direction"""
        ) as cursor:
            async for row in cursor:
                total = row[1]
                wins = row[2] or 0
                stats["by_direction"][row[0]] = {
                    "total": total, "wins": wins,
                    "total_pl": row[3] or 0.0,
                    "win_rate": (wins / total * 100) if total else 0,
                }

        # Aktuelle Verlustserie
        async with db.execute(
            """SELECT profit_loss FROM trades
            WHERE status != 'OPEN' AND profit_loss IS NOT NULL
            ORDER BY exit_timestamp DESC LIMIT 20"""
        ) as cursor:
            streak = 0
            async for row in cursor:
                if row[0] is not None and row[0] <= 0:
                    streak += 1
                else:
                    break
            stats["current_loss_streak"] = streak

    return stats


async def save_pending_recheck(
    asset: str, epic: str, direction: str,
    trigger_condition: str, recheck_in_minutes: int,
    confidence: int, original_analysis: str = "",
) -> int:
    """Erstellt einen neuen Pending Recheck."""
    now = datetime.now(tz=config.TZ)
    recheck_at = (now + timedelta(minutes=recheck_in_minutes)).isoformat()
    async with aiosqlite.connect(config.DB_PATH) as db:
        cursor = await db.execute(
            """INSERT INTO pending_rechecks
                (created_at, asset, epic, direction, trigger_condition,
                 recheck_at, recheck_count, max_rechecks, current_confidence,
                 original_analysis, status)
            VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?, 'PENDING')""",
            (
                now.isoformat(), asset, epic, direction, trigger_condition,
                recheck_at, config.RECHECK_MAX_PER_IDEA, confidence,
                original_analysis,
            ),
        )
        await db.commit()
        logger.info("Recheck geplant: %s in %d Min (ID=%s)", asset, recheck_in_minutes, cursor.lastrowid)
        return cursor.lastrowid


async def get_due_rechecks() -> list:
    """Gibt alle faelligen Pending Rechecks zurueck."""
    now = datetime.now(tz=config.TZ).isoformat()
    async with aiosqlite.connect(config.DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT * FROM pending_rechecks
            WHERE status = 'PENDING' AND recheck_at <= ?
            ORDER BY recheck_at ASC""",
            (now,),
        ) as cursor:
            rows = await cursor.fetchall()
    return [_row_to_recheck(r) for r in rows]


async def get_pending_rechecks() -> list:
    """Gibt alle ausstehenden Rechecks zurueck (fuer API/Dashboard)."""
    async with aiosqlite.connect(config.DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM pending_rechecks WHERE status = 'PENDING' ORDER BY recheck_at ASC",
        ) as cursor:
            rows = await cursor.fetchall()
    return [_row_to_recheck(r) for r in rows]


async def update_recheck_status(recheck_id: int, status: str) -> None:
    """Aendert den Status eines Rechecks."""
    async with aiosqlite.connect(config.DB_PATH) as db:
        await db.execute(
            "UPDATE pending_rechecks SET status = ?, resolved_at = ? WHERE id = ?",
            (status, datetime.now(tz=config.TZ).isoformat(), recheck_id),
        )
        await db.commit()
    logger.info("Recheck %s → %s", recheck_id, status)


async def increment_recheck(recheck_id: int, next_in_minutes: int) -> None:
    """Zaehlt Recheck hoch und plant den naechsten Zeitpunkt."""
    next_at = (datetime.now(tz=config.TZ) + timedelta(minutes=next_in_minutes)).isoformat()
    async with aiosqlite.connect(config.DB_PATH) as db:
        await db.execute(
            "UPDATE pending_rechecks SET recheck_count = recheck_count + 1, recheck_at = ? WHERE id = ?",
            (next_at, recheck_id),
        )
        await db.commit()
    logger.debug("Recheck %s: count+1, naechster in %d Min", recheck_id, next_in_minutes)


async def expire_overnight_rechecks() -> int:
    """Markiert alte und uebernacht-Rechecks als EXPIRED."""
    now = datetime.now(tz=config.TZ)
    expire_h, expire_m = map(int, config.RECHECK_EXPIRE_TIME.split(":"))
    from datetime import time as dt_time

    count = 0
    async with aiosqlite.connect(config.DB_PATH) as db:
        # Rechecks von vorherigen Tagen immer expiren
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        cursor = await db.execute(
            "UPDATE pending_rechecks SET status = 'EXPIRED', resolved_at = ? WHERE status = 'PENDING' AND created_at < ?",
            (now.isoformat(), today_start),
        )
        count += cursor.rowcount

        # Heutige Rechecks nach Expire-Zeit expiren
        if now.time() >= dt_time(expire_h, expire_m):
            cursor = await db.execute(
                "UPDATE pending_rechecks SET status = 'EXPIRED', resolved_at = ? WHERE status = 'PENDING'",
                (now.isoformat(),),
            )
            count += cursor.rowcount

        await db.commit()

    if count > 0:
        logger.info("%d Recheck(s) verfallen", count)
    return count


def _row_to_recheck(row):
    """Konvertiert eine DB-Row in ein PendingRecheck-Objekt."""
    from .models import Direction, PendingRecheck
    return PendingRecheck(
        id=row["id"],
        created_at=datetime.fromisoformat(row["created_at"]),
        asset=row["asset"],
        epic=row["epic"],
        direction=Direction(row["direction"]),
        trigger_condition=row["trigger_condition"],
        recheck_at=datetime.fromisoformat(row["recheck_at"]),
        recheck_count=row["recheck_count"],
        max_rechecks=row["max_rechecks"],
        current_confidence=row["current_confidence"],
        original_analysis=row["original_analysis"] or "",
        status=row["status"],
    )


async def get_unreviewed_trades() -> list[Trade]:
    """Gibt geschlossene Trades zurueck die noch kein Review haben."""
    async with aiosqlite.connect(config.DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT t.* FROM trades t
            LEFT JOIN trade_reviews r ON t.id = r.trade_id
            WHERE t.status != 'OPEN' AND r.id IS NULL
            ORDER BY t.exit_timestamp DESC""",
        ) as cursor:
            rows = await cursor.fetchall()
    return [_row_to_trade(r) for r in rows]


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
        model=row["model"] if "model" in row.keys() else None,
    )
