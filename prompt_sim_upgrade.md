# Prompt für Claude Code: Simulation-Upgrade im TradingBot

## Kontext

Du arbeitest im Projekt `Z:\TradingBot`. Es ist ein Trading-Bot mit einem Streamlit-Dashboard.
Die **Timeline-Simulation** (`src/timeline_sim.py`) führt ein DQN-Modell auf historischen Minutenkerzen aus und erzeugt simulierte Trades. Das Dashboard (`pages/2_Simulation.py`) steuert die Sim und zeigt Ergebnisse.

Die Simulation funktioniert bereits. Es gibt aber zu wenig Daten pro Trade für eine gute Analyse. Außerdem soll die Filterung erweitert werden.

## Aufgabe: 3 Änderungsbereiche

### 1. Zusätzliche Felder pro Trade loggen

In `_infer_batch()` (timeline_sim.py:227) werden aktuell nur `(action_id, softmax_conf)` zurückgegeben. Ändere das so, dass die **rohen Q-Values** (numpy array) pro Inference ebenfalls zurückgegeben werden.

Dann speichere folgende **neue Felder** pro Trade:

**Bei Trade-Eröffnung (BUY/SELL):**
- `q_buy`, `q_sell`, `q_close` — die rohen Q-Values bei Entry
- `q_spread` — Differenz zwischen höchstem und zweithöchstem Q-Value (= Entscheidungssicherheit)
- `rsi_entry` — RSI-Wert bei Entry (bereits in `_build_state` berechnet, einfach mitgeben)
- `atr_pct_entry` — ATR% bei Entry (ebenfalls bereits berechnet)

**Bei Trade-Schließung (CLOSE/SL/TP/End):**
- `q_buy_exit`, `q_sell_exit`, `q_close_exit` — Q-Values bei Exit (nur bei `closed_dqn`, sonst NULL)
- `peak_pnl` — höchster unrealisierter PnL (in Punkten) während der Laufzeit des Trades
- `peak_timestamp` — Zeitpunkt des Peaks
- `steps` — Anzahl Minutenkerzen die der Trade offen war

**Dazu nötig:**
- Peak-Tracking: Während ein Trade offen ist, bei jedem Tick `high` (für BUY) bzw. `low` (für SELL) gegen den bisherigen Peak prüfen und updaten. Speichere `peak_pnl` und `peak_timestamp` im Trade-Dict.
- Steps-Tracking: Zähler pro Trade hochzählen bei jedem Tick.
- Die `_build_state()`-Methode berechnet bereits RSI und ATR — gib diese als zusätzliche Return-Values zurück (oder berechne sie separat im Sim-Loop, was einfacher sein könnte).

**DB-Schema erweitern:**
- `sim_trades`-Tabelle: neue Spalten hinzufügen (mit Migration für bestehende Tabellen via `ALTER TABLE ADD COLUMN` wie bereits für `confidence`/`close_confidence` gemacht, siehe Zeile 276-281)
- `_save_trades_sync()` anpassen
- `_build_summary()` → `trade_list` um die neuen Felder erweitern

### 2. Confidence-Filter: Checkboxen statt Slider

Aktuell (`pages/2_Simulation.py`, Zeile 64):
```python
_sim_conf = st.slider("Min. Confidence:", 1, 10, 7, key="sim_conf")
```

Ersetze das durch **Checkboxen für jede Confidence-Stufe (1-10)**. Der User soll einzelne Stufen an/abwählen können. Default: alle an.

**Umsetzung:**
- Im Dashboard: `st.multiselect` oder eine Reihe von `st.checkbox` (1-10) in einer Zeile
- An die API: statt `confidence_threshold: int` ein `confidence_levels: list[int]` übergeben (z.B. `[2, 3, 5]`)
- In `TimelineSimulator`: `confidence_threshold` durch `confidence_levels: list[int]` ersetzen
- Trade-Öffnung (timeline_sim.py:535): `confidence >= self.confidence_threshold` ändern zu `confidence in self.confidence_levels`
- **Abwärtskompatibel**: wenn die API einen `confidence_threshold` bekommt (alter Client), daraus eine Level-Liste erzeugen

### 3. Q-Spread-Filter (neue Einstellung)

Neuer optionaler Filter: **Min Q-Spread**. Der Q-Spread ist die Differenz zwischen dem höchsten Q-Value und dem zweithöchsten. Ein größerer Spread bedeutet, das Modell ist sicherer.

**Umsetzung:**
- Dashboard: `st.slider("Min. Q-Spread:", 0.0, 0.5, 0.0, step=0.01)` — Default 0.0 (= kein Filter)
- An die API als `min_q_spread: float`
- In `TimelineSimulator.__init__`: neuer Parameter `min_q_spread: float = 0.0`
- Trade-Öffnung: zusätzlich zum Confidence-Check prüfen ob `q_spread >= self.min_q_spread`

## Wichtige Hinweise

- **Keine Änderungen an Dateien außerhalb von `src/` und `pages/2_Simulation.py`** — der Rest des Bots darf nicht brechen
- Die API-Endpunkte für die Simulation sind in `src/api.py` definiert — dort auch die neuen Parameter entgegennehmen
- Die CSV-Export-Funktion (falls vorhanden) sollte die neuen Felder ebenfalls enthalten
- Bestehende gespeicherte Sim-Runs (`sim_log.py`) sollen weiterhin ladbar sein (keine Breaking Changes)
- Teste ob die Simulation nach den Änderungen noch startet (kein Syntax-Error etc.)
- Es gibt bereits eine Migration-Pattern für neue Spalten (ALTER TABLE ... ADD COLUMN, wrapped in try/except), nutze dasselbe Pattern

---

## Implementierungsplan (Claude Code)

> **Status: WARTE AUF FREIGABE**

### Erkenntnisse aus Code-Analyse

- **API-Endpunkte**: Dashboard nutzt `/api/simulation/start` + `/api/simulation/progress`, API definiert `/api/run-timeline-sim` + `/api/timeline-sim/progress`. Das ist ein bestehendes Mismatch — ich fasse es **nicht** an, da es außerhalb des Auftrags liegt.
- **`_build_state()`** berechnet RSI/ATR intern, gibt aber nur den flachen State-Array zurück. Statt den Return-Type zu ändern (würde auch andere Aufrufe brechen), berechne ich RSI/ATR **separat** im Sim-Loop — wie im Prompt vorgeschlagen.
- **Q-Values**: `_infer_batch()` hat die Q-Values schon lokal (`qs`), gibt aber nur `(action_id, softmax_conf)` zurück. Einfache Erweiterung auf `(action_id, softmax_conf, q_raw)`.
- **Action-Map**: v1 hat `{0: "HOLD", 1: "BUY", 2: "SELL", 3: "CLOSE"}`. Q-Values-Indizes entsprechen diesen Actions. Für `q_buy/q_sell/q_close` muss ich die Indizes aus `action_map` ableiten (reverse lookup), um versionssicher zu bleiben.

### Schritt 1 — `_infer_batch()` erweitern

**Datei:** `src/timeline_sim.py`, Zeile 227

```python
# Vorher: return list[tuple[int, float]]
# Nachher: return list[tuple[int, float, np.ndarray]]
def _infer_batch(self, states: list[np.ndarray]) -> list[tuple[int, float, np.ndarray]]:
    ...
    for q in qs:
        action = int(q.argmax())
        probs = np.exp(q - q.max())
        probs /= probs.sum()
        out.append((action, float(probs.max()), q.copy()))  # q_raw dazu
    return out
```

### Schritt 2 — Q-Value-Helper + RSI/ATR-Extraktion

**Datei:** `src/timeline_sim.py`

Neue Helper-Methode auf der Klasse:

```python
def _extract_q_fields(self, q: np.ndarray) -> dict:
    """Extrahiert q_buy, q_sell, q_close, q_spread aus rohen Q-Values."""
    rev = {v: k for k, v in self._vcfg.action_map.items()}
    q_buy   = float(q[rev["BUY"]])   if "BUY"   in rev else None
    q_sell  = float(q[rev["SELL"]])   if "SELL"  in rev else None
    q_close = float(q[rev["CLOSE"]]) if "CLOSE" in rev else None
    sorted_q = np.sort(q)[::-1]
    q_spread = float(sorted_q[0] - sorted_q[1]) if len(sorted_q) >= 2 else 0.0
    return {"q_buy": q_buy, "q_sell": q_sell, "q_close": q_close, "q_spread": q_spread}
```

Für RSI/ATR im Sim-Loop: Die Funktionen `_rsi()` und `_atr()` sind bereits importiert. Ich berechne sie direkt aus dem `window`-Array, das sowieso pro Asset/Tick vorliegt:

```python
closes = window[:, 3]
highs  = window[:, 1]
lows   = window[:, 2]
ref    = float(closes[-1]) or 1.0
rsi_val = round(float(_rsi(closes)), 4)
atr_pct_val = round(float(np.clip(_atr(highs, lows, closes) / (ref + 1e-8), 0, 0.05) / 0.05), 4)
```

### Schritt 3 — Peak-Tracking + Steps im Sim-Loop

**Datei:** `src/timeline_sim.py`, im `_simulate_sync()`-Loop

**Bei Trade-Eröffnung** (BUY/SELL, ~Zeile 540): Neue Felder im Trade-Dict:

```python
open_trades[asset] = {
    ...,
    "q_buy": qf["q_buy"], "q_sell": qf["q_sell"], "q_close": qf["q_close"],
    "q_spread": qf["q_spread"],
    "rsi_entry": rsi_val, "atr_pct_entry": atr_pct_val,
    "peak_pnl": 0.0, "peak_timestamp": current_ts, "steps": 0,
}
```

**Bei jedem Tick mit offenem Trade** (vor SL/TP-Check, ~Zeile 399): Peak + Steps updaten:

```python
tr["steps"] += 1
# Peak-PnL tracking
if buy:
    current_peak = c_high - tr["entry_price"]
else:
    current_peak = tr["entry_price"] - c_low
if current_peak > tr.get("peak_pnl", 0.0):
    tr["peak_pnl"] = current_peak
    tr["peak_timestamp"] = current_ts
```

**Bei Trade-Schließung (CLOSE via DQN)**: Q-Values bei Exit speichern:

```python
tr.update({
    ...,
    "q_buy_exit": qf_exit["q_buy"], "q_sell_exit": qf_exit["q_sell"],
    "q_close_exit": qf_exit["q_close"],
})
```

Bei SL/TP/End-Schließung: `q_*_exit` bleiben `None` (werden nicht gesetzt).

### Schritt 4 — DB-Schema erweitern

**Datei:** `src/timeline_sim.py`, Zeile 276-281 (Migration-Block)

Neue Spalten hinzufügen (gleiches try/except-Pattern):

```python
for col, ctype in [
    ("confidence", "REAL"), ("close_confidence", "REAL"),
    # NEU:
    ("q_buy", "REAL"), ("q_sell", "REAL"), ("q_close", "REAL"), ("q_spread", "REAL"),
    ("rsi_entry", "REAL"), ("atr_pct_entry", "REAL"),
    ("q_buy_exit", "REAL"), ("q_sell_exit", "REAL"), ("q_close_exit", "REAL"),
    ("peak_pnl", "REAL"), ("peak_timestamp", "TEXT"),
    ("steps", "INTEGER"),
]:
```

### Schritt 5 — `_save_trades_sync()` erweitern

**Datei:** `src/timeline_sim.py`, Zeile 642

Neue Felder in die INSERT-Query + Tuple aufnehmen.

### Schritt 6 — `_build_summary()` erweitern

**Datei:** `src/timeline_sim.py`, Zeile 713

Neue Felder in `trade_list`-Dict aufnehmen:

```python
"q_buy": t.get("q_buy"), "q_sell": t.get("q_sell"), ...
"peak_pnl": t.get("peak_pnl"), "steps": t.get("steps"), ...
```

### Schritt 7 — Confidence-Filter: Slider → Multiselect

**Datei:** `pages/2_Simulation.py`, Zeile 64

```python
# Vorher:
_sim_conf = st.slider("Min. Confidence:", 1, 10, 7, key="sim_conf")

# Nachher:
_sim_conf_levels = st.multiselect(
    "Confidence-Stufen:", list(range(1, 11)), default=list(range(1, 11)),
    key="sim_conf_levels",
)
```

Payload anpassen (Zeile 88):
```python
"confidence_levels": _sim_conf_levels,
```

**Datei:** `src/api.py`, `TimelineSimRequest` (Zeile 706):

```python
confidence_levels: list[int] | None = None  # NEU
confidence_threshold: int = 8               # bleibt für Rückwärtskompatibilität
```

Im Sim-Aufruf: `confidence_levels` berechnen:
```python
levels = body.confidence_levels or list(range(body.confidence_threshold, 11))
sim = TimelineSimulator(..., confidence_levels=levels, ...)
```

**Datei:** `src/timeline_sim.py`, `__init__`:

```python
# Vorher: confidence_threshold: int
# Nachher: confidence_levels: list[int] | None = None
# Intern: self.confidence_levels = confidence_levels or list(range(DEFAULT_CONFIDENCE_THRESHOLD, 11))
```

Trade-Eröffnung (Zeile 535):
```python
# Vorher: confidence >= self.confidence_threshold
# Nachher: confidence in self.confidence_levels
```

### Schritt 8 — Q-Spread-Filter

**Datei:** `pages/2_Simulation.py`

Neuer Slider neben dem Confidence-Multiselect:

```python
_sim_q_spread = st.slider("Min. Q-Spread:", 0.0, 0.5, 0.0, step=0.01, key="sim_q_spread")
```

Payload: `"min_q_spread": _sim_q_spread`

**Datei:** `src/api.py`, `TimelineSimRequest`:
```python
min_q_spread: float = 0.0
```

Weitergabe an `TimelineSimulator(min_q_spread=body.min_q_spread)`.

**Datei:** `src/timeline_sim.py`:
- `__init__`: `self.min_q_spread = min_q_spread`
- Trade-Eröffnung: `... and qf["q_spread"] >= self.min_q_spread`

### Schritt 9 — Dashboard: Neue Felder in Trade-Tabelle + CSV

**Datei:** `pages/2_Simulation.py`, Trade-Table (~Zeile 400)

Spalten-Rename erweitern um die neuen Felder. CSV-Export enthält automatisch alles was in `df_trades` ist.

### Zusammenfassung der Dateien & Änderungen

| Datei | Änderungen |
|---|---|
| `src/timeline_sim.py` | `_infer_batch` erweitert, `_extract_q_fields` neu, `__init__` Params, Sim-Loop (Peak/Steps/RSI/ATR/Q-Fields), DB-Schema, `_save_trades_sync`, `_build_summary` |
| `src/api.py` | `TimelineSimRequest` + Sim-Aufruf (confidence_levels, min_q_spread) |
| `pages/2_Simulation.py` | Multiselect statt Slider, Q-Spread-Slider, Payload, Trade-Tabelle |

Keine anderen Dateien werden angefasst.
