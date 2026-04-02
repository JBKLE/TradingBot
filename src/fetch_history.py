"""Fetch historical 1-minute candles from Capital.com and store in simLastCharts.db.

Usage from code:
    await fetch_all_assets("2026-01-01", "2026-03-28")

The module checks which data already exists in the DB before making API calls,
so re-running for the same period is safe and only downloads missing gaps.
"""
import asyncio
import logging
import os
from datetime import datetime, timedelta

import aiosqlite

from . import config
from .broker import CapitalComBroker, CapitalComError, get_shared_broker

logger = logging.getLogger(__name__)

# ── DB path ──────────────────────────────────────────────────────────────────

HISTORY_DB_PATH = os.path.join(config.DATA_DIR, "simLastCharts.db")

# ── DDL (identical schema to simulation.db) ──────────────────────────────────

_DDL = [
    """CREATE TABLE IF NOT EXISTS price_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        asset TEXT NOT NULL,
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        volume REAL DEFAULT 0.0
    )""",
    """CREATE UNIQUE INDEX IF NOT EXISTS idx_ph_asset_ts
       ON price_history (asset, timestamp)""",
    """CREATE INDEX IF NOT EXISTS idx_ph_asset_ts_desc
       ON price_history (asset, timestamp DESC)""",
    """CREATE TABLE IF NOT EXISTS sim_trades (
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
    )""",
    """CREATE INDEX IF NOT EXISTS idx_st_status
       ON sim_trades (status)""",
    """CREATE INDEX IF NOT EXISTS idx_st_asset_status
       ON sim_trades (asset, status)""",
]


async def init_history_db(db_path: str = HISTORY_DB_PATH) -> None:
    """Create tables if they don't exist."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    async with aiosqlite.connect(db_path) as db:
        for stmt in _DDL:
            await db.execute(stmt)
        await db.execute("PRAGMA journal_mode=WAL")
        await db.commit()
    logger.info("History DB initialised at %s", db_path)


# ── Check existing data ─────────────────────────────────────────────────────

async def get_existing_range(
    asset: str, db_path: str = HISTORY_DB_PATH,
) -> dict:
    """Return info about what data already exists for an asset.

    Returns:
        {"count": int, "min_ts": str|None, "max_ts": str|None}
    """
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) "
            "FROM price_history WHERE asset = ?",
            (asset,),
        ) as cur:
            row = await cur.fetchone()
    return {
        "count": row[0] or 0,
        "min_ts": row[1],
        "max_ts": row[2],
    }


async def get_all_existing_ranges(
    db_path: str = HISTORY_DB_PATH,
) -> dict[str, dict]:
    """Return existing data ranges for all assets."""
    result = {}
    for asset_key in config.WATCHLIST:
        result[asset_key] = await get_existing_range(asset_key, db_path)
    return result


# ── Fetch logic ──────────────────────────────────────────────────────────────

CHUNK_MINUTES = 800  # Capital.com MINUTE resolution limit: ~880 bars max per request


def _generate_chunks(
    start: datetime, end: datetime,
) -> list[tuple[datetime, datetime]]:
    """Split a time range into chunks of CHUNK_MINUTES."""
    chunks = []
    current = start
    while current < end:
        chunk_end = min(current + timedelta(minutes=CHUNK_MINUTES), end)
        chunks.append((current, chunk_end))
        current = chunk_end
    return chunks


async def fetch_asset_history(
    broker: CapitalComBroker,
    asset_key: str,
    epic: str,
    start_date: str,
    end_date: str,
    db_path: str = HISTORY_DB_PATH,
    progress_callback=None,
) -> dict:
    """Fetch historical 1-min candles for one asset and store in DB.

    Args:
        broker: Authenticated broker instance
        asset_key: e.g. "GOLD"
        epic: Capital.com epic code
        start_date: ISO date "2026-01-01"
        end_date: ISO date "2026-03-28"
        db_path: Path to history DB
        progress_callback: Optional fn(asset, chunk_idx, total_chunks, new_bars)

    Returns:
        {"asset": str, "fetched": int, "skipped": int, "errors": int}
    """
    start_dt = datetime.fromisoformat(start_date + "T00:00:00")
    end_dt = datetime.fromisoformat(end_date + "T23:59:00")

    # Check what we already have
    existing = await get_existing_range(asset_key, db_path)
    existing_count = existing["count"]

    # Generate time chunks
    chunks = _generate_chunks(start_dt, end_dt)
    total_chunks = len(chunks)

    fetched_total = 0
    skipped_total = 0
    error_count = 0

    for idx, (chunk_start, chunk_end) in enumerate(chunks):
        from_iso = chunk_start.strftime("%Y-%m-%dT%H:%M:%S")
        to_iso = chunk_end.strftime("%Y-%m-%dT%H:%M:%S")

        # Skip chunks that fall entirely on a weekend (Sa=5, So=6)
        # A chunk is skipped when both start AND end land on Sat/Sun.
        # Mid-week chunks that merely *touch* a weekend are kept so we
        # don't miss Friday-evening or Sunday-night bars.
        chunk_mid = chunk_start + (chunk_end - chunk_start) / 2
        if chunk_start.weekday() >= 5 and chunk_end.weekday() >= 5 and chunk_mid.weekday() >= 5:
            logger.debug("Wochenende übersprungen: %s–%s", from_iso, to_iso)
            if progress_callback:
                progress_callback(asset_key, idx + 1, total_chunks, 0)
            continue

        # Check if this chunk is fully covered in DB
        async with aiosqlite.connect(db_path) as db:
            async with db.execute(
                "SELECT COUNT(*) FROM price_history "
                "WHERE asset = ? AND timestamp >= ? AND timestamp <= ?",
                (asset_key, from_iso, to_iso),
            ) as cur:
                row = await cur.fetchone()
                existing_in_chunk = row[0] or 0

        # If chunk seems complete (>90% of expected), skip it
        expected_bars = (chunk_end - chunk_start).total_seconds() / 60
        if existing_in_chunk > expected_bars * 0.9:
            skipped_total += existing_in_chunk
            if progress_callback:
                progress_callback(asset_key, idx + 1, total_chunks, 0)
            continue

        # Fetch from API
        try:
            bars = await broker.get_price_history(
                epic,
                resolution="MINUTE",
                max_bars=CHUNK_MINUTES,
                from_date=from_iso,
                to_date=to_iso,
            )
        except CapitalComError as exc:
            exc_str = str(exc)
            if "404" in exc_str or "not-found" in exc_str or "Not Found" in exc_str:
                # Data not available for this period (demo API history limit)
                logger.debug("No data for %s chunk %d (%s–%s) – skipping", asset_key, idx, from_iso, to_iso)
                if progress_callback:
                    progress_callback(asset_key, idx + 1, total_chunks, 0)
                continue
            if "daterange" in exc_str or "400 Date range" in exc_str:
                # Date range rejected – skip this chunk
                logger.debug("Date range rejected for %s chunk %d – skipping", asset_key, idx)
                if progress_callback:
                    progress_callback(asset_key, idx + 1, total_chunks, 0)
                continue
            logger.warning("Fetch failed for %s chunk %d: %s", asset_key, idx, exc)
            error_count += 1
            await asyncio.sleep(1)
            continue

        if not bars:
            if progress_callback:
                progress_callback(asset_key, idx + 1, total_chunks, 0)
            continue

        # Insert into DB (INSERT OR IGNORE handles duplicates)
        rows = [
            (bar.timestamp, asset_key, bar.open, bar.high, bar.low, bar.close, 0.0)
            for bar in bars
        ]

        async with aiosqlite.connect(db_path) as db:
            await db.executemany(
                """INSERT OR IGNORE INTO price_history
                   (timestamp, asset, open, high, low, close, volume)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            await db.commit()

        fetched_total += len(bars)

        if progress_callback:
            progress_callback(asset_key, idx + 1, total_chunks, len(bars))

        # Rate limiting: small delay between requests
        await asyncio.sleep(0.15)

    return {
        "asset": asset_key,
        "fetched": fetched_total,
        "skipped": skipped_total,
        "existing_before": existing_count,
        "errors": error_count,
    }


async def fetch_all_assets(
    start_date: str,
    end_date: str,
    db_path: str = HISTORY_DB_PATH,
    assets: list[str] | None = None,
    progress_callback=None,
) -> list[dict]:
    """Fetch historical candles for all (or selected) assets.

    Args:
        start_date: "2026-01-01"
        end_date: "2026-03-28"
        db_path: Path to history DB
        assets: List of asset keys, None = all from WATCHLIST
        progress_callback: fn(asset, chunk_idx, total_chunks, new_bars)

    Returns:
        List of per-asset result dicts
    """
    await init_history_db(db_path)

    target_assets = assets or list(config.WATCHLIST.keys())
    results = []

    broker = await get_shared_broker()
    for asset_key in target_assets:
        asset_info = config.WATCHLIST.get(asset_key)
        if not asset_info:
            logger.warning("Asset %s not in WATCHLIST, skipping", asset_key)
            continue

        epic = asset_info["epic"]
        logger.info("Fetching %s (%s) from %s to %s",
                    asset_key, epic, start_date, end_date)

        result = await fetch_asset_history(
            broker, asset_key, epic,
            start_date, end_date, db_path,
            progress_callback,
        )
        results.append(result)
        logger.info("  %s: fetched=%d, skipped=%d, errors=%d",
                    asset_key, result["fetched"],
                    result["skipped"], result["errors"])

    return results
