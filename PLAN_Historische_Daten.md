# Plan: Historische Kursdaten laden & DQN auf Zeitstrahl testen

## Ziel

1. **Historische 1-Min-Kerzen** ueber die Capital.com API laden (Wochen/Monate in die Vergangenheit)
2. In einer eigenen DB **`simLastCharts.db`** speichern (gleiches Schema wie `simulation.db`)
3. **DQN auf dem historischen Zeitstrahl laufen lassen** — die KI entscheidet Minute fuer Minute BUY/SELL/HOLD
4. Die Ergebnisse (simulierte Trades) ebenfalls in `simLastCharts.db` speichern
5. **TradeAI kann mit diesen Daten trainieren** — statt 10h sofort Wochen/Monate an Daten

---

## Capital.com API: Was geht?

Die API `GET /api/v1/prices/{epic}` unterstuetzt:

| Parameter | Beschreibung |
|---|---|
| `resolution` | MINUTE, MINUTE_5, MINUTE_15, MINUTE_30, HOUR, HOUR_4, DAY, WEEK |
| `max` | Max. Kerzen pro Request (bis 1000) |
| `from` | Start-Zeitpunkt (ISO-8601) |
| `to` | End-Zeitpunkt (ISO-8601) |

### Rechenbeispiel: 4 Wochen 1-Min-Daten

```
4 Wochen × 5 Tage × 22h × 60 min = ~26.400 Kerzen pro Asset
1000 Kerzen pro Request → 27 Requests pro Asset
4 Assets → 108 Requests gesamt
API-Limit: 1000 req/Stunde → locker machbar in ~10 Minuten
```

### Rechenbeispiel: 6 Monate 1-Min-Daten

```
26 Wochen × 5 Tage × 22h × 60 min = ~171.600 Kerzen pro Asset
172 Requests pro Asset × 4 Assets = 688 Requests
→ ~45 Minuten (mit Rate-Limiting)
```

---

## Technische Umsetzung

### Schritt 1: Historische Daten laden (neues Script/Endpoint)

**Neues Script: `fetch_history.py`** (oder als API-Endpoint im Bot)

```python
async def fetch_historical_candles(
    broker, epic, asset_key,
    start_date,           # z.B. "2026-01-01"
    end_date,             # z.B. "2026-03-28"
    resolution="MINUTE",
    db_path="data/simLastCharts.db"
):
    """Laedt historische Kerzen in 1000er-Bloecken und speichert in DB."""
    # 0. DB pruefen: Welche Zeitraeume sind schon vorhanden?
    #    SELECT MIN(timestamp), MAX(timestamp) FROM price_history WHERE asset=?
    #    → Nur fehlende Luecken nachladenr
    # 1. Zeitraum in 1000-Minuten-Fenster aufteilen
    # 2. Pro Fenster: GET /api/v1/prices/{epic}?from=...&to=...&resolution=MINUTE&max=1000
    # 3. INSERT OR IGNORE (Duplikate ueberspringen dank UNIQUE-Index)
```

**Ablauf:**
1. **VOR dem Download**: DB abfragen welche Timestamps fuer dieses Asset bereits existieren
   - Bereits vorhandene Zeitraeume ueberspringen
   - Nur Luecken und neue Zeitraeume laden
   - Status anzeigen: "GOLD: 15.200/26.400 schon vorhanden, lade 11.200 nach"
2. Fehlende Zeitraeume in ~16h-Bloecke (1000 Minuten) aufteilen
3. Pro Block einen API-Call
4. `INSERT OR IGNORE` — UNIQUE-Index auf `(asset, timestamp)` verhindert Duplikate
5. Rate-Limiting: max 10 Requests/Sekunde, Pause wenn noetig
6. Fortschrittsanzeige im Terminal oder Dashboard

### Schritt 2: DB-Schema (simLastCharts.db)

Exakt gleiches Schema wie `simulation.db`:

```sql
-- Kerzen (identisch)
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
CREATE UNIQUE INDEX IF NOT EXISTS idx_ph_unique
ON price_history (asset, timestamp);

-- Simulierte Trades (identisch)
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
```

Weil das Schema identisch ist, kann TradeAI direkt `simLastCharts.db` als Datenquelle nutzen —
es muss nur `DB_PATH` in der TradeAI-Config umgestellt werden.

### Schritt 3: DQN auf historischem Zeitstrahl laufen lassen (Turbo-Modus)

**Neues Feature: Offline-Zeitstrahl-Simulation**

Die Simulation laeuft **so schnell wie CPU/GPU es erlauben** — kein Warten auf echte Minuten.
Alle Daten liegen bereits in der DB, es werden keine API-Calls benoetigt.

```
Geschwindigkeit: ~500–2000 Minuten/Sekunde (GPU)
26.400 Minuten (4 Wochen) → ~15–50 Sekunden
171.600 Minuten (6 Monate) → ~2–6 Minuten
```

**Ablauf (pro simulierte Minute):**
```
  1. Naechste Kerze aus price_history lesen (kein API-Call, nur DB-Read)
  2. Offene Trades pruefen: SL/TP getroffen?
     → Ja: Trade schliessen, P/L berechnen
  3. Ab Kerze 500: DQN-Inferenz (State aus den letzten 500 Kerzen)
     → Aktion + Confidence
  4. Wenn BUY/SELL + Confidence >= Schwelle + kein offener Trade fuer Asset:
     → Sim-Trade oeffnen (entry in sim_trades)
  5. Weiter zur naechsten Kerze (kein sleep, sofort)
```

**Optimierung fuer Speed:**
- State-Vektor wird inkrementell aktualisiert (sliding window, nicht jedes Mal 500 Kerzen neu laden)
- Batch-Inferenz: mehrere Assets gleichzeitig durch das Netz
- GPU-Tensor bleibt im VRAM, nur neue Kerze wird angehaengt

Das ist im Prinzip der gleiche `unified_tick()` wie im Live-Betrieb,
aber offline auf historischen Daten — ohne API-Calls und ohne Wartezeiten.

### Schritt 4: Dashboard-Integration

**Neuer Tab oder Bereich im Dashboard:**

1. **Daten laden**
   - Datumsbereich waehlen (von/bis)
   - Asset(s) waehlen
   - **Status-Anzeige VOR dem Download**: "GOLD: 15.200 Kerzen vorhanden, 11.200 fehlen"
   - Button "Fehlende Daten laden" (nur Luecken werden geladen)
   - Fortschrittsanzeige (X/Y Bloecke geladen)

2. **Zeitstrahl-Simulation**
   - Confidence-Schwelle waehlen
   - Button "Simulation starten"
   - Live-Fortschritt: "Minute 1500/26400 — 3 Trades offen, 12 geschlossen"
   - Ergebnis: Equity-Kurve, Trade-Statistik, CSV-Download

3. **Training starten**
   - Button "TradeAI mit diesen Daten trainieren"
   - Startet `train_rl.py` mit `--db simLastCharts.db`

### Schritt 5: TradeAI-Anbindung

TradeAI `config.py` anpassen: DB-Pfad konfigurierbar machen.

```python
# config.py
DB_PATHS = [
    r"Z:\TradingBot\data\simulation.db",       # Live-Daten
    r"Z:\TradingBot\data\simLastCharts.db",     # Historische Daten
]
```

`train_rl.py` neues Flag:
```bash
python train_rl.py --db simLastCharts.db --resume --episodes 2000
```

Das Environment laedt dann Kerzen aus beiden oder einer spezifischen DB.

---

## Dateien & Aenderungen

| Datei | Aenderung |
|---|---|
| `src/broker.py` | `get_price_history()` erweitern um `from_date`/`to_date` Parameter |
| `src/fetch_history.py` | **NEU** — Script zum Laden historischer Daten |
| `src/sim_database.py` | Funktionen fuer simLastCharts.db (oder parametrisch machen) |
| `src/api.py` | Endpoint `/api/fetch-history` + `/api/run-timeline-sim` |
| `dashboard.py` | UI fuer Datumsauswahl, Laden, Simulation |
| `TradeAI/config.py` | Mehrere DB-Pfade unterstuetzen |
| `TradeAI/train_rl.py` | `--db` Flag fuer alternative Datenquelle |

---

## Reihenfolge

1. `broker.py`: `from_date`/`to_date` Parameter in `get_price_history()`
2. `fetch_history.py`: Historische Daten laden + in DB schreiben
3. Testen: Manuell 4 Wochen Daten laden
4. Zeitstrahl-Simulation implementieren (offline unified_tick)
5. Dashboard-UI
6. TradeAI-Integration

---

## Schaetzung: Datenvolumen

| Zeitraum | Kerzen/Asset | Gesamt (4 Assets) | DB-Groesse (ca.) | API-Calls |
|---|---|---|---|---|
| 1 Woche | 6.600 | 26.400 | ~5 MB | 28 |
| 4 Wochen | 26.400 | 105.600 | ~20 MB | 108 |
| 3 Monate | 85.800 | 343.200 | ~65 MB | 344 |
| 6 Monate | 171.600 | 686.400 | ~130 MB | 688 |

Alles locker im API-Limit von 1000 req/Stunde machbar.
