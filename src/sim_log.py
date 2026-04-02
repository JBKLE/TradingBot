"""Simulation-Protokoll – speichert Timeline-Sim-Runs in sim_history.db.

Jeder Run wird manuell gespeichert ("Speichern"-Button) und enthält das
komplette Ergebnis-JSON (trade_list, financial, equity, per_asset) damit
es 1:1 wieder geladen werden kann.
"""
import json
import os
import sqlite3

from . import config

SIM_HISTORY_DB = os.path.join(config.DATA_DIR, "sim_history.db")

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS sim_runs (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    -- Zeitstempel & Status
    run_at               TEXT NOT NULL,
    finished_at          TEXT,
    duration_sec         REAL,
    status               TEXT,          -- completed | cancelled | error
    -- Modell
    model_name           TEXT,
    model_path           TEXT,
    -- Einstellungen
    assets               TEXT,          -- JSON-Array
    start_date           TEXT,
    end_date             TEXT,
    confidence_threshold INTEGER,
    -- Finanzparameter (NULL = nicht aktiv)
    capital              REAL,
    risk_pct             REAL,
    leverage             INTEGER,
    sl_pct               REAL,
    tp_pct               REAL,
    -- Kurzuebersicht (fuer Historie-Liste)
    trades               INTEGER,
    win_rate             REAL,
    total_pnl_points     REAL,
    start_capital        REAL,
    end_capital          REAL,
    total_return_pct     REAL,
    max_drawdown_pct     REAL,
    -- Vollstaendiges Ergebnis
    result_json          TEXT
)
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(SIM_HISTORY_DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(_CREATE_SQL)
    # Migration: result_json hinzufuegen falls fehlend (alte DB)
    existing = {row[1] for row in conn.execute("PRAGMA table_info(sim_runs)").fetchall()}
    if "result_json" not in existing:
        conn.execute("ALTER TABLE sim_runs ADD COLUMN result_json TEXT")
        conn.commit()
    return conn


def save_run(
    *,
    run_at: str,
    finished_at: str,
    duration_sec: float,
    status: str,
    model_name: str = "",
    model_path: str = "",
    assets: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    confidence_threshold: int = 1,
    capital: float | None = None,
    risk_pct: float | None = None,
    leverage: int | None = None,
    sl_pct: float | None = None,
    tp_pct: float | None = None,
    trades: int = 0,
    win_rate: float = 0.0,
    total_pnl_points: float = 0.0,
    start_capital: float | None = None,
    end_capital: float | None = None,
    total_return_pct: float | None = None,
    max_drawdown_pct: float | None = None,
    result_json: str = "",
) -> int:
    """Einen Simulation-Run speichern. Gibt die neue ID zurueck."""
    conn = _connect()
    cur = conn.execute(
        """INSERT INTO sim_runs (
            run_at, finished_at, duration_sec, status,
            model_name, model_path,
            assets, start_date, end_date, confidence_threshold,
            capital, risk_pct, leverage, sl_pct, tp_pct,
            trades, win_rate, total_pnl_points,
            start_capital, end_capital, total_return_pct, max_drawdown_pct,
            result_json
        ) VALUES (
            ?,?,?,?, ?,?, ?,?,?,?, ?,?,?,?,?, ?,?,?, ?,?,?,?, ?
        )""",
        (
            run_at, finished_at, round(duration_sec, 1), status,
            model_name, model_path,
            json.dumps(assets or []), start_date, end_date, confidence_threshold,
            capital, risk_pct, leverage, sl_pct, tp_pct,
            trades, round(win_rate, 2), round(total_pnl_points, 4),
            start_capital, end_capital,
            round(total_return_pct, 2) if total_return_pct is not None else None,
            round(max_drawdown_pct, 2) if max_drawdown_pct is not None else None,
            result_json,
        ),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def delete_run(run_id: int) -> bool:
    """Einen Run anhand seiner ID loeschen."""
    try:
        conn = _connect()
        cur = conn.execute("DELETE FROM sim_runs WHERE id = ?", (run_id,))
        conn.commit()
        deleted = cur.rowcount > 0
        conn.close()
        return deleted
    except Exception:
        return False


def load_runs(limit: int = 50) -> list[dict]:
    """Letzte N Runs laden (neueste zuerst). Ohne result_json (zu gross)."""
    try:
        conn = _connect()
        rows = conn.execute(
            """SELECT id, run_at, finished_at, duration_sec, status,
                      model_name, assets, start_date, end_date,
                      confidence_threshold, capital, sl_pct, tp_pct,
                      trades, win_rate, total_pnl_points,
                      start_capital, end_capital, total_return_pct, max_drawdown_pct
               FROM sim_runs ORDER BY id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        conn.close()
        cols = [
            "id", "run_at", "finished_at", "duration_sec", "status",
            "model_name", "assets", "start_date", "end_date",
            "confidence_threshold", "capital", "sl_pct", "tp_pct",
            "trades", "win_rate", "total_pnl_points",
            "start_capital", "end_capital", "total_return_pct", "max_drawdown_pct",
        ]
        result = []
        for row in rows:
            d = dict(zip(cols, row))
            try:
                d["assets"] = json.loads(d["assets"] or "[]")
            except Exception:
                d["assets"] = []
            result.append(d)
        return result
    except Exception:
        return []


def load_run_result(run_id: int) -> dict | None:
    """Vollstaendiges Ergebnis-JSON eines Runs laden."""
    try:
        conn = _connect()
        row = conn.execute(
            "SELECT result_json FROM sim_runs WHERE id = ?", (run_id,)
        ).fetchone()
        conn.close()
        if row and row[0]:
            return json.loads(row[0])
        return None
    except Exception:
        return None
