"""Trainingsdaten-Manager — Filtert sim_trades aus Quell-DBs und exportiert in Training-DB."""
import logging
import os
import sqlite3
from typing import Any

from . import config

logger = logging.getLogger(__name__)

TRAINING_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS training_trades (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    source_db        TEXT,
    asset            TEXT NOT NULL,
    direction        TEXT NOT NULL,
    sl_variant       TEXT NOT NULL,
    entry_timestamp  TEXT NOT NULL,
    entry_price      REAL NOT NULL,
    sl_price         REAL NOT NULL,
    tp_price         REAL NOT NULL,
    exit_timestamp   TEXT,
    exit_price       REAL,
    status           TEXT NOT NULL,
    pnl              REAL,
    r_multiple       REAL
);
"""


# ── DB-Discovery ──────────────────────────────────────────────────────────────

def list_trade_databases() -> list[dict]:
    """Alle .db-Dateien im DATA_DIR auflisten die eine sim_trades-Tabelle haben."""
    data_dir = config.DATA_DIR
    results = []
    try:
        for fname in sorted(os.listdir(data_dir)):
            if not fname.endswith(".db"):
                continue
            fpath = os.path.join(data_dir, fname)
            try:
                conn = sqlite3.connect(fpath, timeout=3)
                tables = [r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()]
                has_sim = "sim_trades" in tables
                has_training = "training_trades" in tables
                trade_count = 0
                if has_sim:
                    trade_count = conn.execute(
                        "SELECT COUNT(*) FROM sim_trades WHERE status != 'open'"
                    ).fetchone()[0]
                conn.close()
                results.append({
                    "name": fname,
                    "size_kb": round(os.path.getsize(fpath) / 1024, 1),
                    "has_sim_trades": has_sim,
                    "has_training_trades": has_training,
                    "trade_count": trade_count,
                })
            except Exception:
                continue
    except Exception:
        pass
    return results


# ── Filteroptionen aus DB laden ───────────────────────────────────────────────

def get_filter_options(source_dbs: list[str]) -> dict:
    """Distinct-Werte für Filter-Dropdowns aus den Quell-DBs laden."""
    data_dir = config.DATA_DIR
    assets: set[str] = set()
    directions: set[str] = set()
    statuses: set[str] = set()
    sl_variants: set[str] = set()
    min_date = ""
    max_date = ""

    for db_name in source_dbs:
        fpath = os.path.join(data_dir, db_name)
        if not os.path.exists(fpath):
            continue
        try:
            conn = sqlite3.connect(fpath, timeout=3)
            for r in conn.execute("SELECT DISTINCT asset FROM sim_trades").fetchall():
                assets.add(r[0])
            for r in conn.execute("SELECT DISTINCT direction FROM sim_trades").fetchall():
                directions.add(r[0])
            for r in conn.execute("SELECT DISTINCT status FROM sim_trades WHERE status != 'open'").fetchall():
                statuses.add(r[0])
            for r in conn.execute("SELECT DISTINCT sl_variant FROM sim_trades").fetchall():
                sl_variants.add(r[0])
            r = conn.execute("SELECT MIN(entry_timestamp), MAX(entry_timestamp) FROM sim_trades").fetchone()
            if r[0]:
                if not min_date or r[0] < min_date:
                    min_date = r[0]
                if not max_date or r[1] > max_date:
                    max_date = r[1]
            conn.close()
        except Exception:
            continue

    return {
        "assets": sorted(assets),
        "directions": sorted(directions),
        "statuses": sorted(statuses),
        "sl_variants": sorted(sl_variants),
        "min_date": min_date[:10] if min_date else "",
        "max_date": max_date[:10] if max_date else "",
    }


# ── Gefilterter Query ────────────────────────────────────────────────────────

def _build_where(filters: dict) -> tuple[str, list]:
    """WHERE-Klausel + Params aus Filter-Dict bauen."""
    conditions = ["status != 'open'"]
    params: list[Any] = []

    if filters.get("assets"):
        placeholders = ",".join("?" for _ in filters["assets"])
        conditions.append(f"asset IN ({placeholders})")
        params.extend(filters["assets"])

    if filters.get("directions"):
        placeholders = ",".join("?" for _ in filters["directions"])
        conditions.append(f"direction IN ({placeholders})")
        params.extend(filters["directions"])

    if filters.get("statuses"):
        placeholders = ",".join("?" for _ in filters["statuses"])
        conditions.append(f"status IN ({placeholders})")
        params.extend(filters["statuses"])

    if filters.get("sl_variants"):
        placeholders = ",".join("?" for _ in filters["sl_variants"])
        conditions.append(f"sl_variant IN ({placeholders})")
        params.extend(filters["sl_variants"])

    if filters.get("date_from"):
        conditions.append("entry_timestamp >= ?")
        params.append(filters["date_from"] + "T00:00:00")

    if filters.get("date_to"):
        conditions.append("entry_timestamp <= ?")
        params.append(filters["date_to"] + "T23:59:59")

    if filters.get("r_multiple_min") is not None:
        conditions.append("r_multiple >= ?")
        params.append(filters["r_multiple_min"])

    if filters.get("r_multiple_max") is not None:
        conditions.append("r_multiple <= ?")
        params.append(filters["r_multiple_max"])

    if filters.get("pnl_min") is not None:
        conditions.append("pnl >= ?")
        params.append(filters["pnl_min"])

    return " AND ".join(conditions), params


def preview_filtered(source_dbs: list[str], filters: dict) -> dict:
    """Vorschau: Statistik der gefilterten Trades (ohne alle Trades zu dumpen)."""
    data_dir = config.DATA_DIR
    where_clause, params = _build_where(filters)

    total = 0
    wins = 0
    losses = 0
    total_pnl = 0.0
    per_asset: dict[str, dict] = {}
    per_direction: dict[str, int] = {"BUY": 0, "SELL": 0}

    for db_name in source_dbs:
        fpath = os.path.join(data_dir, db_name)
        if not os.path.exists(fpath):
            continue
        try:
            conn = sqlite3.connect(fpath, timeout=5)
            rows = conn.execute(
                f"""SELECT asset, direction, status, pnl, r_multiple
                    FROM sim_trades WHERE {where_clause}""",
                params,
            ).fetchall()
            conn.close()

            for asset, direction, status, pnl, r_mult in rows:
                total += 1
                pnl_val = pnl or 0.0
                total_pnl += pnl_val
                is_win = pnl_val > 0
                if is_win:
                    wins += 1
                else:
                    losses += 1

                per_direction[direction] = per_direction.get(direction, 0) + 1

                if asset not in per_asset:
                    per_asset[asset] = {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0}
                per_asset[asset]["trades"] += 1
                per_asset[asset]["pnl"] += pnl_val
                if is_win:
                    per_asset[asset]["wins"] += 1
                else:
                    per_asset[asset]["losses"] += 1
        except Exception as exc:
            logger.warning("preview_filtered: %s → %s", db_name, exc)

    for stats in per_asset.values():
        stats["win_rate"] = round(
            stats["wins"] / stats["trades"] * 100, 1
        ) if stats["trades"] > 0 else 0.0
        stats["pnl"] = round(stats["pnl"], 4)

    return {
        "total": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / total * 100, 1) if total > 0 else 0.0,
        "total_pnl": round(total_pnl, 4),
        "per_asset": per_asset,
        "per_direction": per_direction,
    }


# ── Export in Training-DB ─────────────────────────────────────────────────────

def export_to_training_db(
    source_dbs: list[str],
    filters: dict,
    target_db: str,
    mode: str = "append",     # "append" oder "replace"
) -> dict:
    """Gefilterte Trades in eine Training-DB schreiben."""
    data_dir = config.DATA_DIR
    target_path = os.path.join(data_dir, os.path.basename(target_db))
    if not target_path.endswith(".db"):
        target_path += ".db"

    where_clause, params = _build_where(filters)

    # Ziel-DB vorbereiten
    target_conn = sqlite3.connect(target_path, timeout=10)
    target_conn.execute("PRAGMA journal_mode=WAL")
    target_conn.execute(TRAINING_TABLE_DDL)
    if mode == "replace":
        target_conn.execute("DELETE FROM training_trades")
    target_conn.commit()

    total_exported = 0
    per_source: dict[str, int] = {}

    for db_name in source_dbs:
        fpath = os.path.join(data_dir, db_name)
        if not os.path.exists(fpath):
            continue
        try:
            conn = sqlite3.connect(fpath, timeout=5)
            rows = conn.execute(
                f"""SELECT asset, direction, sl_variant,
                           entry_timestamp, entry_price, sl_price, tp_price,
                           exit_timestamp, exit_price, status, pnl, r_multiple
                    FROM sim_trades WHERE {where_clause}""",
                params,
            ).fetchall()
            conn.close()

            if rows:
                insert_rows = [
                    (db_name, *row) for row in rows
                ]
                target_conn.executemany(
                    """INSERT INTO training_trades
                       (source_db, asset, direction, sl_variant,
                        entry_timestamp, entry_price, sl_price, tp_price,
                        exit_timestamp, exit_price, status, pnl, r_multiple)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    insert_rows,
                )
                per_source[db_name] = len(rows)
                total_exported += len(rows)
        except Exception as exc:
            logger.warning("export: %s → %s", db_name, exc)
            per_source[db_name] = 0

    target_conn.commit()

    # Gesamtzahl in Ziel-DB
    total_in_target = target_conn.execute(
        "SELECT COUNT(*) FROM training_trades"
    ).fetchone()[0]
    target_conn.close()

    return {
        "exported": total_exported,
        "per_source": per_source,
        "total_in_target": total_in_target,
        "target_db": os.path.basename(target_path),
        "mode": mode,
    }
