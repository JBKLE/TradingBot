"""Simulation-Protokoll – speichert jeden Timeline-Sim-Run in sim_history.db.

Schema sim_runs:
  - Metadaten des Runs (Zeitstempel, Status, Dauer)
  - Modell-Infos (Name, Pfad, Änderungsdatum)
  - Einstellungen (Assets, Confidence, Kapital, Hebel …)
  - Ergebnisse (Trades, WR, P/L, Drawdown …)
  - Per-Asset-Breakdown (JSON)
"""
import json
import os
import sqlite3
from datetime import datetime, timezone

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
    model_modified_at    TEXT,
    -- Einstellungen
    assets               TEXT,          -- JSON-Array
    start_date           TEXT,
    end_date             TEXT,
    confidence_threshold INTEGER,
    output_db            TEXT,
    -- Finanzparameter (NULL = nicht aktiv)
    capital              REAL,
    risk_pct             REAL,
    leverage             INTEGER,
    eur_usd              REAL,
    -- Simulationsergebnisse
    total_minutes        INTEGER,
    trades               INTEGER,
    wins                 INTEGER,
    losses               INTEGER,
    win_rate             REAL,
    total_pnl_points     REAL,
    avg_r_multiple       REAL,
    -- Finanzergebnisse (NULL = nicht aktiv)
    start_capital        REAL,
    end_capital          REAL,
    total_return_pct     REAL,
    max_drawdown_pct     REAL,
    margin_call          INTEGER,
    -- Per-Asset-Breakdown (JSON)
    per_asset            TEXT,
    -- Fehlermeldung
    error_message        TEXT
)
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(SIM_HISTORY_DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(_CREATE_SQL)
    conn.commit()
    return conn


def save_run(
    *,
    run_at: str,
    finished_at: str,
    duration_sec: float,
    status: str,
    # Modell
    model_name: str = "",
    model_path: str = "",
    model_modified_at: str = "",
    # Einstellungen
    assets: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    confidence_threshold: int = 8,
    output_db: str = "",
    capital: float | None = None,
    risk_pct: float | None = None,
    leverage: int | None = None,
    eur_usd: float = 1.08,
    # Ergebnisse
    total_minutes: int = 0,
    trades: int = 0,
    wins: int = 0,
    losses: int = 0,
    win_rate: float = 0.0,
    total_pnl_points: float = 0.0,
    avg_r_multiple: float = 0.0,
    start_capital: float | None = None,
    end_capital: float | None = None,
    total_return_pct: float | None = None,
    max_drawdown_pct: float | None = None,
    margin_call: bool = False,
    per_asset: dict | None = None,
    error_message: str = "",
) -> int:
    """Einen Simulation-Run in sim_history.db speichern. Gibt die neue ID zurück."""
    conn = _connect()
    cur = conn.execute(
        """INSERT INTO sim_runs (
            run_at, finished_at, duration_sec, status,
            model_name, model_path, model_modified_at,
            assets, start_date, end_date, confidence_threshold, output_db,
            capital, risk_pct, leverage, eur_usd,
            total_minutes, trades, wins, losses, win_rate,
            total_pnl_points, avg_r_multiple,
            start_capital, end_capital, total_return_pct, max_drawdown_pct,
            margin_call, per_asset, error_message
        ) VALUES (
            ?,?,?,?,  ?,?,?,  ?,?,?,?,?,  ?,?,?,?,
            ?,?,?,?,?,  ?,?,  ?,?,?,?,  ?,?,?
        )""",
        (
            run_at, finished_at, round(duration_sec, 1), status,
            model_name, model_path, model_modified_at,
            json.dumps(assets or []), start_date, end_date,
            confidence_threshold, output_db,
            capital, risk_pct, leverage, eur_usd,
            total_minutes, trades, wins, losses, round(win_rate, 2),
            round(total_pnl_points, 4), round(avg_r_multiple, 4),
            start_capital, end_capital,
            round(total_return_pct, 2) if total_return_pct is not None else None,
            round(max_drawdown_pct, 2) if max_drawdown_pct is not None else None,
            1 if margin_call else 0,
            json.dumps(per_asset or {}),
            error_message,
        ),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def load_runs(limit: int = 50) -> list[dict]:
    """Letzte N Runs laden (neueste zuerst)."""
    try:
        conn = _connect()
        rows = conn.execute(
            "SELECT * FROM sim_runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        cols = [d[0] for d in conn.execute("SELECT * FROM sim_runs LIMIT 0").description]
        conn.close()
        result = []
        for row in rows:
            d = dict(zip(cols, row))
            try:
                d["assets"] = json.loads(d["assets"] or "[]")
            except Exception:
                d["assets"] = []
            try:
                d["per_asset"] = json.loads(d["per_asset"] or "{}")
            except Exception:
                d["per_asset"] = {}
            result.append(d)
        return result
    except Exception:
        return []
