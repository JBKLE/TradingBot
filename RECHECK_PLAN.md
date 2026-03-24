# Recheck-Feature: "Nicht reif" → Selbst-Aufgabe → Nachtrade

## Idee

Wenn Claude bei der Analyse ein Setup sieht, das **vielversprechend aber noch nicht reif** ist
(z.B. Confidence 6-7, Setup formt sich, wartet auf Breakout), soll der Bot sich selbst
eine **Recheck-Aufgabe** erstellen. Beim Recheck wird nur dieses eine Asset gezielt
nachanalysiert. Ist es dann reif → traden. Immer noch nicht → nächster Recheck (max 3).

---

## Wann gilt ein Trade als "nicht reif"?

Claude meldet `WAIT`, aber mit Hinweisen die auf ein baldiges Setup deuten:

| Signal | Beispiel |
|--------|----------|
| Confidence 5-7 (knapp unter Minimum 8) | "Gold zeigt starken Aufwärtstrend, aber RSI ist noch nicht überverkauft genug" |
| Setup formt sich | "Silver nähert sich Support bei 64.00, Bounce wahrscheinlich aber noch nicht bestätigt" |
| Warten auf Event | "Vor der Fed-Entscheidung um 14:30 kein Trade, danach prüfen" |
| Preis nähert sich Level | "Oil bei 72.50, Breakout über 73.00 wäre ein klares Kaufsignal" |

## Neues JSON-Feld in Claude's Analyse

Erweiterung des bestehenden Analyse-Schemas:

```json
{
    "recommendation": "WAIT",
    "wait_reason": "Gold nähert sich Breakout-Level...",
    "recheck": {
        "worthy": true,
        "asset": "GOLD",
        "direction": "BUY",
        "trigger_condition": "Breakout über 2350.00 bestätigt",
        "recheck_in_minutes": 60,
        "current_confidence": 6,
        "expected_confidence_if_trigger": 9
    }
}
```

Wenn `recheck.worthy = false` oder `recheck` fehlt → kein Recheck, normales WAIT.

---

## Umsetzungsplan

### 1. Datenbank: Neue Tabelle `pending_rechecks`

**Datei:** `src/database.py`

```sql
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
    original_analysis TEXT,
    status TEXT DEFAULT 'PENDING',  -- PENDING, EXECUTED, EXPIRED, CANCELLED
    resolved_at TEXT
);
```

Funktionen:
- `save_pending_recheck(...)` → neuen Recheck eintragen
- `get_due_rechecks()` → alle fälligen Rechecks (WHERE status='PENDING' AND recheck_at <= NOW)
- `update_recheck_status(id, status)` → Status ändern
- `increment_recheck(id, next_recheck_at)` → Count hochzählen + neuen Zeitpunkt

### 2. Modelle erweitern

**Datei:** `src/models.py`

```python
class RecheckInfo(BaseModel):
    worthy: bool = False
    asset: str = ""
    direction: Direction = Direction.NONE
    trigger_condition: str = ""
    recheck_in_minutes: int = 60
    current_confidence: int = 0
    expected_confidence_if_trigger: int = 0

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
    status: str = "PENDING"
```

`AnalysisResult` erweitern:
```python
class AnalysisResult(BaseModel):
    ...
    recheck: Optional[RecheckInfo] = None  # NEU
```

### 3. Claude-Prompt erweitern

**Datei:** `src/analyzer.py`

Im `_SYSTEM_PROMPT` das JSON-Schema um `recheck` erweitern:

```
"recheck": {{
    "worthy": true/false,
    "asset": "GOLD|SILVER|OIL_CRUDE|NATURALGAS",
    "direction": "BUY|SELL",
    "trigger_condition": "Beschreibung wann Setup reif wäre",
    "recheck_in_minutes": 30-120,
    "current_confidence": 1-10,
    "expected_confidence_if_trigger": 1-10
}}
```

Zusätzliche Regel im Prompt:
```
- Wenn du WAIT empfiehlst, aber ein Setup sich bildet (Confidence 5-7), markiere es als
  recheck-worthy mit einer konkreten Trigger-Bedingung und Zeitrahmen.
- recheck_in_minutes: wie lange der Bot warten soll bevor er erneut prüft (30-120 Min)
```

### 4. Recheck-Analyse Prompt (neuer fokussierter Prompt)

**Datei:** `src/analyzer.py` (neue Methode `recheck_opportunity()`)

Ein **schlankerer Prompt** der nur das eine Asset prüft:

```
Du bist ein Trading-Analyst. Du hattest zuvor ein Setup identifiziert, das noch nicht reif war.
Prüfe ob es JETZT reif ist.

URSPRÜNGLICHES SETUP:
- Asset: {asset}
- Richtung: {direction}
- Trigger-Bedingung: {trigger_condition}
- Damalige Confidence: {original_confidence}
- Erwartete Confidence wenn Trigger: {expected_confidence}

AKTUELLE MARKTDATEN:
{current_market_data}

AKTUELLE INDIKATOREN:
{current_indicators}

Antworte als JSON:
{
    "is_ready": true/false,
    "confidence": 1-10,
    "entry_price": 0.0,
    "stop_loss": 0.0,
    "take_profit": 0.0,
    "risk_reward_ratio": 0.0,
    "reasoning": "Warum jetzt reif / noch nicht reif",
    "retry_worthy": true/false,
    "retry_in_minutes": 0-120
}
```

- `is_ready = true` + confidence >= 8 → Trade ausführen
- `is_ready = false` + `retry_worthy = true` → nächster Recheck (wenn < max)
- `is_ready = false` + `retry_worthy = false` → Setup verworfen

### 5. Integration in den 5-Minuten-Monitor

**Datei:** `src/main.py` (in `monitor_positions()` oder eigener Job)

Alle 5 Minuten (zusammen mit Position-Monitoring):
1. `get_due_rechecks()` → fällige Rechecks holen
2. Für jeden fälligen Recheck:
   a. Frische Marktdaten für das Asset laden
   b. `analyzer.recheck_opportunity()` aufrufen
   c. Wenn `is_ready` → normalen Trade-Flow (validate → build_signal → execute)
   d. Wenn `retry_worthy` + count < max → `increment_recheck()`
   e. Sonst → `update_recheck_status(EXPIRED)`
3. Notification bei jedem Statuswechsel

### 6. Integration in `daily_routine()`

**Datei:** `src/main.py`

Nach der Claude-Analyse, wenn Ergebnis = WAIT:
```python
if analysis.recheck and analysis.recheck.worthy:
    await database.save_pending_recheck(
        asset=analysis.recheck.asset,
        epic=config.WATCHLIST[analysis.recheck.asset]["epic"],
        direction=analysis.recheck.direction,
        trigger_condition=analysis.recheck.trigger_condition,
        recheck_in_minutes=analysis.recheck.recheck_in_minutes,
        confidence=analysis.recheck.current_confidence,
        original_analysis=analysis.model_dump_json(),
    )
    logger.info("Recheck geplant: %s in %d Min",
                analysis.recheck.asset, analysis.recheck.recheck_in_minutes)
    await notifier.notify(f"Recheck geplant: {analysis.recheck.asset} in {analysis.recheck.recheck_in_minutes} Min")
```

### 7. API-Endpunkt

**Datei:** `src/api.py`

- `GET /api/pending-rechecks` → alle ausstehenden Rechecks anzeigen
- `POST /api/recheck/{id}/cancel` → Recheck manuell abbrechen
- `POST /api/recheck/{id}/execute-now` → Recheck sofort ausführen

### 8. Dashboard

**Datei:** `dashboard.py`

Neuer Abschnitt "PENDING RECHECKS":
- Tabelle mit ausstehenden Rechecks (Asset, Trigger, nächster Check, Anzahl)
- "Cancel" Button pro Recheck
- "Jetzt prüfen" Button

---

## Ablauf-Diagramm

```
daily_routine() → Claude-Analyse
    │
    ├── TRADE (confidence >= 8)
    │   └── validate → execute → fertig
    │
    └── WAIT
        ├── recheck.worthy = true
        │   └── save_pending_recheck(recheck_at = now + X min)
        │       │
        │       ▼ (5-Min-Monitor oder dedicated Job)
        │   check_pending_rechecks()
        │       │
        │       ├── recheck_opportunity() → is_ready = true
        │       │   └── validate → execute → status = EXECUTED
        │       │
        │       ├── retry_worthy = true, count < 3
        │       │   └── increment_recheck(count++, neuer Zeitpunkt)
        │       │
        │       └── retry_worthy = false ODER count >= 3
        │           └── status = EXPIRED
        │
        └── recheck.worthy = false
            └── normales WAIT (kein Recheck)
```

---

## Config-Parameter

**Datei:** `src/config.py`

```python
RECHECK_MAX_PER_IDEA = 3          # Max Rechecks pro Setup
RECHECK_DEFAULT_MINUTES = 60      # Fallback wenn Claude keine Zeit angibt
RECHECK_MIN_MINUTES = 15          # Minimum zwischen Rechecks
RECHECK_MAX_MINUTES = 180         # Maximum
```

---

## Offene Fragen

1. **Recheck-Kosten:** Jeder Recheck = 1 Claude-API-Call (~$0.01-0.05). Bei 3 WAITs × 3 Rechecks = 9 Extra-Calls/Tag. Akzeptabel? <-- ja
2. **Recheck-Timing:** Soll der 5-Min-Monitor die Rechecks prüfen, oder ein eigener Scheduler-Job? (5-Min-Monitor ist einfacher, eigener Job flexibler) <-- für das haben wir den Monitor :)
3. **Übernacht:** Sollen Rechecks über Nacht (nach 20:00) automatisch verfallen oder bis zum nächsten Tag weiterlaufen? <-- einstellbar, aber default verfallen
4. **Parallel:** Können mehrere Rechecks gleichzeitig ausstehen (z.B. Gold-Recheck + Silver-Recheck)? <-- ja, diese können evtl. mit der gleichen Anfrage an Claude gesendet werdenn? Spart Tokens.
