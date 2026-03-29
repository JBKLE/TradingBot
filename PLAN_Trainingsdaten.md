# Plan: Trainingsdaten-Manager

## Macht das Sinn? (Konzeptuelle Antwort)

**Kurz: Ja — aber mit Bedacht.**

In einem DQN-Bot lernt das Modell aus **(State → Action → Reward)**-Tripeln.
Das bedeutet:

| Datenkategorie | Im Training sinnvoll? | Warum |
|---|---|---|
| TP-Trades mit hoher Confidence | ✅ Ja | Klares Signal: KI war richtig, Muster war gut |
| SL-Trades mit hoher Confidence | ✅ Ja | Negatives Lernsignal ist genauso wichtig |
| Trades mit sehr niedriger Confidence | ⚠️ Bedingt | Rauschen — KI war selbst unsicher |
| OIL_CRUDE wenn dauerhaft verlustbringend | ⚠️ Bedingt | Kann schlechte Muster einschleifen |
| Trades aus bestimmten Zeitfenstern (z.B. Donnerstag) | ⚠️ Bedingt | Strukturelle Marktschwäche, nicht lernbar |
| Alle Daten gemischt | ✅ Für Robustheit | Verhindert Overfitting auf "gute" Szenarien |

**Empfehlung:**
- Für initiales / korrigierendes Training: gefilterte Daten sinnvoll
  (z.B. nur Assets mit WR > 40%, nur Conf ≥ 7)
- Für Robustheitstraining: alle Daten + Gewichtung nach R-Multiple
- Niemals: nur TP-Trades — das Modell würde nie lernen wann es FALSCH liegt

---

## Ziel des Features

Ein UI in dem man:
1. Eine oder mehrere Quell-DBs wählt (mit `sim_trades`-Tabelle)
2. Filter definiert (Asset, Richtung, Status, Confidence, Datum, R-Multiple, sl_variant)
3. Eine Vorschau mit Live-Statistik sieht
4. Gefilterte Trades in eine dedizierte Training-DB schreibt

---

## UI-Aufbau (neuer Tab "◈ TRAININGSDATEN")

```
┌─────────────────────────────────────────────────────────┐
│  1. QUELL-DATENBANK(EN)                                  │
│  [simLastCharts.db ▼] [sim_test_v4.db ▼]  [+ Hinzufügen]│
│  Gesamt: 1.847 Trades gefunden                           │
├─────────────────────────────────────────────────────────┤
│  2. FILTER                                               │
│  Assets:    [GOLD ✓] [SILVER ✓] [OIL_CRUDE] [NATURALGAS]│
│  Richtung:  [BUY ✓] [SELL ✓]                            │
│  Status:    [closed_tp ✓] [closed_sl ✓] [closed_end]    │
│  Confidence: Min [0] Max [10]   (falls gespeichert)      │
│  R-Multiple: Min [-∞] Max [+∞]  Slider oder Input        │
│  Datum:     Von [____] Bis [____]                        │
│  sl_variant: [dqn_timeline ✓] [dqn_sim ✓] [alle]        │
├─────────────────────────────────────────────────────────┤
│  3. VORSCHAU (live beim Filtern)                         │
│  Treffer: 842 Trades | WR: 47.3% | TP: 398 | SL: 444   │
│  Pro Asset: [Tabelle]                                    │
│  [Bar-Chart: Trades pro Asset]                           │
├─────────────────────────────────────────────────────────┤
│  4. IN TRAINING-DB SCHREIBEN                             │
│  Ziel-DB: [training_data.db ▼] oder [Neu: ___________]  │
│  Modus:   (●) Anhängen  ( ) Ersetzen                    │
│  [✓ In Training-DB schreiben]                            │
└─────────────────────────────────────────────────────────┘
```

---

## Filter-Felder im Detail

| Filter | Typ | Spalte in sim_trades |
|---|---|---|
| Assets | Multiselect | `asset` |
| Richtung | Multiselect | `direction` (BUY/SELL) |
| Status | Multiselect | `status` (closed_tp/closed_sl/closed_end) |
| R-Multiple | Range Slider | `r_multiple` |
| Datum Von/Bis | Date Input | `entry_timestamp` |
| sl_variant | Multiselect | `sl_variant` |
| Min. Trades pro Asset | Number | (Aggregat-Filter nach DB-Query) |

> **Confidence** ist in `sim_trades` aktuell nicht gespeichert.
> Für zukünftige Nutzung sollte sie in das Schema aufgenommen werden (späteres TODO).

---

## Training-DB Schema

Identisch mit `sim_trades` — kein eigenes Schema nötig.
Das Training-Skript kann dieselbe DB direkt verwenden.
Einzige Ergänzung: Spalte `source_db` (welche Quell-DB der Trade stammt).

```sql
CREATE TABLE IF NOT EXISTS training_trades (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    source_db        TEXT,           -- Herkunft
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
```

---

## Dateien die geändert / neu erstellt werden

| Datei | Änderung |
|---|---|
| `src/training_data.py` | NEU — Filter-Logik, DB-Query, Schreiben in Training-DB |
| `src/api.py` | NEU — `POST /api/training-data/preview` + `POST /api/training-data/export` |
| `dashboard.py` | NEU — 4. Tab `◈ TRAININGSDATEN` |

---

## API-Endpoints

### `POST /api/training-data/preview`
Body: `{ source_dbs, filters }` → gibt Statistik zurück (kein Trade-Dump)

### `POST /api/training-data/export`
Body: `{ source_dbs, filters, target_db, mode }` → schreibt Trades

### `GET /api/training-data/databases`
→ Alle DBs im DATA_DIR mit Spalten-Check (hat `sim_trades`?)

---

## Nicht im Scope (v1)
- Confidence-Filter (Spalte fehlt in sim_trades aktuell)
- Duplikat-Erkennung über DBs hinweg
- Automatischer Export nach Training-Run
- Gewichtung der Trades nach R-Multiple beim Export
