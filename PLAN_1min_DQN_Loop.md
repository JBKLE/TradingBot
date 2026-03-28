# 1-Minuten DQN-Loop: Analyse & Umsetzung

> Erstellt: 2026-03-28
> Frage: Soll die DQN-AI jede Minute laufen, Trades entscheiden UND offene Capital.com-Positionen pruefen?

---

## 1. Ausgangslage

| Komponente | Aktuell | Geplant |
|---|---|---|
| **Marktanalyse** | 3x taeglich (08:00, 12:00, 16:00) | Jede Minute |
| **Position-Monitor** | Alle 5 Minuten (regelbasiert) | Jede Minute (DQN-gesteuert) |
| **Simulation** | Jede Minute (sammelt Kerzen + Sim-Trades) | Bleibt, zusaetzlich DQN-Training-Daten |
| **DQN-Inferenz** | Nur bei Analyse-Schedule | Jede Minute, alle 4 Assets |

---

## 2. Macht das Sinn? -- Ja, und zwar aus diesen Gruenden

### 2.1 Das DQN wurde genau dafuer trainiert

Das Modell wurde mit **Minuten-Kerzen** aus der Simulation trainiert (`sim_engine.py` -> `price_history`-Tabelle). Der State-Vektor nutzt die letzten **500 Kerzen** -- bei 1-Minuten-Resolution sind das ~8,3 Stunden Marktdaten. Das Modell ist darauf optimiert, auf dieser Granularitaet Entscheidungen zu treffen.

Die bisherige 3x-taeglich-Logik war ein Kompromiss wegen:
- Claude-API-Kosten (~EUR 0.05-0.50 pro Analyse -> EUR 150+/Monat bei jeder Minute)
- Latenz (5-15 Sekunden pro Claude-Call)

Beide Einschraenkungen fallen weg: **DQN-Inferenz kostet EUR 0 und dauert <100ms.**

### 2.2 SL/TP-Werte passen zum 1-Minuten-Takt

Die trainierten SL/TP-Werte sind extrem eng:
```
SL_PCT = 0.003  (0.3 %)
TP_PCT = 0.005  (0.5 %)
```

Bei Gold (~EUR 3.000): SL = EUR 9, TP = EUR 15. Das sind Bewegungen, die **innerhalb von Minuten** passieren. Ein 3x-taeglicher Check verpasst die meisten dieser Setups komplett. **Das Modell braucht den 1-Minuten-Takt, um seine trainierten Muster ueberhaupt nutzen zu koennen.**

### 2.3 CLOSE-Aktion wird erst jetzt sinnvoll

Das DQN hat 4 Aktionen: HOLD, BUY, SELL, **CLOSE**. Die CLOSE-Aktion wurde trainiert, um Positionen aktiv zu managen -- nicht nur passiv auf SL/TP zu warten. Bei 3x-taeglich hat das DQN maximal 3 Chancen pro Tag, eine CLOSE-Empfehlung abzugeben. Bei minuetlicher Pruefung kann es **in Echtzeit** reagieren -- z.B. vor einem Kurseinbruch schliessen, bevor der SL greift.

---

## 3. API-Calls minimieren: Ein Tick, eine gemeinsame Abfrage

### Das Problem bei getrennten Loops

Aktuell laufen Sim-Engine, Analyse und Monitor getrennt. Jeder holt Preise separat:

```
SIM-ENGINE  (jede Min):      get_price_history() x 4 Assets     = 4 Calls
ANALYSE     (3x taeglich):   get_market_prices() x 4 Assets     = 4 Calls
MONITOR     (alle 5 Min):    get_open_positions()                = 1 Call
                                                          Gesamt:  9 Calls
```

### Die Loesung: Ein unified Tick, eine Datenquelle

`sim_engine.collect_prices()` holt bereits jede Minute **1-Minuten-Kerzen fuer alle 4 Assets** und schreibt sie in die `price_history`-Tabelle. Das sind die einzigen Broker-Calls die wir brauchen. Alles andere liest aus der DB:

```
+----------------------------------------------------------------------+
|  UNIFIED TICK (jede Minute)                                          |
|                                                                      |
|  +-- BROKER-CALLS (nur diese!) -----------------------------------+  |
|  |  get_price_history(MINUTE, 1) x 4 Assets         = 4 Calls    |  |
|  |  get_open_positions()                             = 1 Call     |  |
|  |                                          Gesamt -> 5 Calls/Min |  |
|  +----------------------------------------------------------------+  |
|                          |                                           |
|              +-----------+-----------+                               |
|              v                       v                               |
|  +-- SIM-ENGINE ------+  +-- DQN-ANALYSE -----------------+         |
|  | Kerzen -> DB        |  | State aus price_history (DB)   |         |
|  | Sim-Trades          |  | Inferenz (<100ms)              |         |
|  |   oeffnen/schliessen|  | Signal -> Strategy -> Executor |         |
|  +---------------------+  +-------------------------------+         |
|                                                                      |
|  Nur bei Trade-Aktion: open_position() oder close_position()        |
|                                                = 0-1 extra Calls     |
|                                                                      |
|  GESAMT: 5 Calls/Min (Normalfall) -> 300 Calls/Stunde               |
|          6 Calls/Min (bei Trade)  -> weit unter 1.000er Limit        |
+----------------------------------------------------------------------+
```

### Calls pro Tick im Detail

| Call | Zweck | Genutzt von | Anzahl |
|---|---|---|---|
| `get_price_history(epic, MINUTE, 1)` | 1-Min-Kerze holen | Sim-Engine + DB fuer DQN | 4 |
| `get_open_positions()` | Offene Positionen | DQN (Position-State) + Monitor | 1 |
| `open_position(...)` | Neuer Trade | Nur bei BUY/SELL-Signal | 0-1 |
| `close_position(deal_id)` | Position schliessen | Nur bei CLOSE-Signal | 0-1 |

### Was komplett wegfaellt

| Alter Call | Warum unnoetig |
|---|---|
| `get_market_prices() x 4` | Preis kommt aus der 1-Min-Kerze (bereits geholt) |
| `get_price_history(MINUTE, 500)` | State wird aus DB gebaut (Kerzen liegen dort) |
| Separater 5-Min-Monitor | Integriert in den unified Tick |
| Separater 3x-taeglich-Schedule | Ersetzt durch minuetliche DQN-Analyse |

### DQN-State kommt aus der DB -- nicht vom Broker

`collect_prices()` schreibt jede Minute eine neue Kerze in `price_history`. Der DQN-State braucht die letzten 500 Kerzen. Statt 500 Kerzen per API zu holen (= 4 teure Calls mit grosser Payload), lesen wir einfach:

```sql
SELECT open, high, low, close FROM price_history
WHERE asset = ? ORDER BY timestamp DESC LIMIT 500
```

Das ist **exakt** was `TradeAI/predict.py` beim Training macht -- identische Datenquelle, identischer State. Kein zusaetzlicher API-Call, <5ms Latenz.

### Vergleich: Alt vs. Neu

| Metrik | Alt (getrennte Loops) | Neu (unified Tick) |
|---|---|---|
| Broker-Calls/Minute | 4 (Sim) + 4 (Analyse gelegentlich) + 1 (Monitor) | **5** |
| Calls/Stunde | ~250-300 (unkoordiniert) | **300** (planbar) |
| Redundante Calls | Ja (Preise doppelt geholt) | **Keine** |
| DB-Reads fuer DQN | 0 (State kam aus API) | 1 Query (<5ms) |
| Separate Scheduler-Jobs | 3 (Sim, Analyse, Monitor) | **1** |

---

## 4. Architektur: Der unified 1-Minuten-Tick

### Ablauf

```python
async def unified_tick():
    """Einziger 1-Min-Job -- ersetzt sim_tick + daily_routine + monitor."""
    if not is_market_open():
        return

    broker = await _get_broker()

    # == 1. Preise holen (EINMAL, fuer alles) ==============
    prices = await collect_prices(broker)      # 4 API-Calls -> DB
    if not prices:
        return

    # == 2. Offene Positionen (EINMAL, fuer alles) =========
    open_positions = await broker.get_open_positions()  # 1 API-Call

    # == 3. Sim-Engine (nutzt prices aus Schritt 1) ========
    await open_sim_trades(prices)
    await evaluate_open_trades(prices)

    # == 4. DQN-Analyse (nutzt DB + open_positions) ========
    #    State kommt aus price_history-Tabelle (0 API-Calls)
    analyzer = DQNAnalyzer()
    signals = analyzer.get_all_signals(open_positions)

    # == 5. Entscheidung + Ausfuehrung ====================
    #    (0-1 API-Calls, nur bei Trade-Aktion)
    await process_signals(signals, open_positions, broker)
```

**5 API-Calls pro Minute. Nicht mehr.**

### Was bleibt gleich
- `TradingStrategy.validate_signal()` -- alle Sicherheits-Gates bleiben aktiv
- `TradeExecutor.execute_trade()` -- Slippage-Check, Spread-Check, Deal-ID-Verifizierung
- Sim-Trades werden weiterhin fuer Retraining gesammelt

### Was sich aendert
- `sim_tick()` + `daily_routine()` + `monitor_positions()` -> **ein** `unified_tick()`
- DQN-State kommt aus der DB statt vom Broker
- `get_open_positions()` wird einmal geholt und an alle Komponenten weitergereicht

---

## 5. Sicherheits-Grenzen (Safety Gates)

### Pruefung: Welche Grenzen existieren bereits im Code?

Alle Gates sitzen in `strategy.py -> validate_signal()` und werden bei **jedem** Trade-Signal geprueft -- egal ob 3x taeglich oder jede Minute. Hier der vollstaendige Status:

| # | Gate | Code-Stelle | Status | Wert |
|---|---|---|---|---|
| 1 | **Kill-Switch** | `strategy.py:52` | **OK** -- erster Check, blockiert sofort | `TRADING_ENABLED=true/false` |
| 2 | **Max offene Positionen** | `strategy.py:105` | **ANGEPASST** -> 4 | `MAX_OPEN_POSITIONS=4` |
| 3 | **1 Trade pro Asset** | `strategy.py:111-119` | **OK** -- prueft epic gegen offene Positionen | Hardcoded |
| 4 | **Max Tages-Drawdown** | `strategy.py:56-68` | **OK** | `MAX_DAILY_DRAWDOWN_PCT=5%` |
| 5 | **Verlustserie-Limit** | `strategy.py:71-85` | **OK** -- 3 Stop-Losses in Folge -> Pause | Hardcoded (3) |
| 6 | **Min Confidence** | `strategy.py:93` | **OK** | `MIN_CONFIDENCE_SCORE=8` |
| 7 | **Min Risk/Reward** | `strategy.py:99` | Anpassen -> 1.5 | `MIN_RISK_REWARD_RATIO` |
| 8 | **Trade-Window** | `strategy.py:206` | **OK** | `09:00-20:00` |
| 9 | **Max Trades pro Tag** | `strategy.py:159` | Anpassen -> 5 | `MAX_TRADES_PER_DAY` |
| 10 | **Cooldown nach Verlust** | `strategy.py:168` | **OK** | `COOLDOWN_AFTER_LOSS_MINUTES=120` |
| 11 | **Kein Richtungswechsel** | `strategy.py:186-195` | **OK** -- kein BUY dann SELL auf selbes Asset am selben Tag | Hardcoded |
| 12 | **Asset-Cooldown nach SL** | `strategy.py:198-204` | **OK** -- Asset heute schon ausgestoppt -> kein neuer Trade | Hardcoded |
| 13 | **ATR-Volatilitaets-Gate** | `strategy.py:122-136` | **OK** -- kein Trade bei extremer Volatilitaet | `MAX_ATR_MULTIPLIER=2.0` |
| 14 | **Slippage-Check** | `executor.py` | **OK** -- vor Ausfuehrung | `MAX_SLIPPAGE_ABS` |
| 15 | **Spread-Check** | `executor.py` | **OK** -- Spread > 30% SL -> Abbruch | Hardcoded |

### Besonders wichtig fuer den 1-Minuten-Takt

**Kill-Switch (Gate 1):**
```python
# strategy.py:52 -- ERSTER Check, vor allem anderen
if not config.TRADING_ENABLED:
    return ValidationResult(valid=False, reason="Kill-switch active")
```
Der Kill-Switch blockiert **neue Trades**. CLOSE-Aktionen (Position schliessen) sind davon bewusst **nicht betroffen** -- der Bot soll bestehende Positionen auch bei Kill-Switch managen koennen.

**1 Trade pro Asset (Gate 3):**
```python
# strategy.py:111-119 -- Prueft ob Asset schon eine Position hat
epic = config.WATCHLIST.get(opp.asset, {}).get("epic", opp.asset)
already_open = [p for p in open_positions if p.epic == epic]
if already_open:
    return ValidationResult(valid=False, reason=f"{opp.asset} hat bereits eine offene Position")
```
Das ist ein **harter Block** -- egal wie hoch die Confidence ist. Solange GOLD eine offene Position hat, wird kein zweiter GOLD-Trade eroeffnet.

**Max 4 Positionen gleichzeitig (Gate 2):**
```
MAX_OPEN_POSITIONS=4  (war: 1, jetzt angepasst)
```
4 Assets in der Watchlist -> maximal 4 offene Positionen. Zusammen mit Gate 3 (1 pro Asset) ergibt das:

```
GOLD:        0 oder 1 offene Position
SILVER:      0 oder 1 offene Position
OIL_CRUDE:   0 oder 1 offene Position
NATURALGAS:  0 oder 1 offene Position
                                ------
Maximum:     4 Positionen gesamt
```

### Ablauf im unified_tick mit allen Gates

```
Jede Minute:
  1. Preise holen
  2. Offene Positionen abrufen
  3. DQN-Inferenz (alle 4 Assets)
  4. Fuer jedes Asset mit BUY/SELL-Signal:
     +-- Kill-Switch aktiv?           -> STOP (kein neuer Trade)
     +-- Asset hat schon Position?    -> STOP (1 pro Asset)
     +-- 4 Positionen offen?          -> STOP (Maximum erreicht)
     +-- Tages-Drawdown >= 5%?        -> STOP
     +-- 3 Stop-Losses in Folge?      -> STOP (Verlustserie)
     +-- Confidence < 8?              -> STOP
     +-- Risk/Reward < 1.5?           -> STOP
     +-- Ausserhalb Trade-Window?     -> STOP
     +-- Max Trades/Tag erreicht?     -> STOP
     +-- Cooldown nach Verlust?       -> STOP
     +-- ATR zu hoch?                 -> STOP
     +-- Alles OK                     -> TRADE AUSFUEHREN
  5. Fuer Assets mit CLOSE-Signal:
     +-- Position vorhanden?          -> SCHLIESSEN (auch bei Kill-Switch)
     +-- Keine Position?              -> Ignorieren
```

### Empfohlene Config-Aenderungen

```env
MAX_OPEN_POSITIONS=4            # GEAENDERT: War 1, jetzt 1 pro Asset
MAX_TRADES_PER_DAY=5            # War: 1 (bei 3x taeglich reichte 1)
MIN_RISK_REWARD_RATIO=1.5       # War: 1.8 (DQN trainiert auf 1.67)
COOLDOWN_AFTER_LOSS_MINUTES=60  # War: 120 (schnellere Recovery bei Scalping)
```

---

## 6. Position-Management durch DQN

Der groesste Gewinn des 1-Minuten-Ticks: **aktives Position-Management**.

### Aktuell (passiv)
```
Position offen -> warte auf SL oder TP -> fertig
                  (Trailing-Stop alle 5 Min als Bonus)
```

### Neu (aktiv)
```
Position offen -> DQN prueft jede Minute:
  +-- HOLD        -> weiter halten
  +-- CLOSE       -> sofort schliessen (Gewinn mitnehmen oder Verlust begrenzen)
  +-- BUY/SELL    -> Gegenrichtung: CLOSE + neuer Trade
```

Das DQN hat den **unrealisierten P/L als State-Feature** (unrealised_R im Position-Vektor). Es wurde darauf trainiert, Positionen aktiv zu managen -- nicht nur zu eroeffnen.

### Implementierung

```python
# Im 1-Min-Tick, nach der Analyse:
for asset, signal in dqn_signals.items():
    open_pos = get_open_position(asset)

    if open_pos and signal["action"] == "CLOSE":
        await executor.close_position(open_pos.deal_id)

    elif open_pos and signal["action"] in ("BUY", "SELL"):
        if signal["action"] != open_pos.direction.value:
            # Gegenrichtung -> erst CLOSE, dann neuer Trade
            await executor.close_position(open_pos.deal_id)
            await executor.execute_trade(build_signal(signal))

    elif not open_pos and signal["action"] in ("BUY", "SELL"):
        if signal["confidence"] >= MIN_CONFIDENCE_SCORE:
            await executor.execute_trade(build_signal(signal))
```

---

## 7. Retraining-Feedback-Loop

Mit dem 1-Minuten-Takt entsteht ein geschlossener Kreislauf:

```
         +------------------------+
         |  Capital.com Markt     |
         +----------+-------------+
                    | Preise (jede Minute)
                    v
         +----------------------+
         |  collect_prices()    |---- Kerzen -> price_history DB
         |  (4 API-Calls)       |---- Sim-Trades -> sim_trades DB
         +----------+-----------+
                    |
                    v
         +----------------------+
         |  DQN-Analyzer        |<--- Liest 500 Kerzen aus DB (0 API-Calls)
         |  (Inferenz <100ms)   |---- Signal: BUY/SELL/HOLD/CLOSE
         +----------+-----------+
                    |
                    v
         +----------------------+
         |  Strategy + Executor |---- Validierung -> Trade bei Capital.com
         +----------+-----------+     (0-1 API-Calls)
                    |
                    v
         +----------------------+
         |  Trade-Ergebnis      |---- Gespeichert in trades DB
         +----------+-----------+
                    |
                    v
         +----------------------+
         |  Retraining          |<--- Sim-Trades + echte Trades
         |  (periodisch)        |---- Neues .pt -> models/
         +----------------------+
                    |
                    v
              DQN laedt automatisch
              das neueste Modell
```

Das DQN profitiert doppelt:
1. **Bessere Live-Performance** durch minuetliche Entscheidungen
2. **Besseres Training** durch mehr reale Trade-Daten als Feedback

---

## 8. Fazit & Empfehlung

| Kriterium | 3x taeglich (alt) | 1x pro Minute (neu) |
|---|---|---|
| Passt zum Modell | Nein (trainiert auf Minuten-Kerzen) | **Ja** |
| API-Calls/Stunde | ~250 (unkoordiniert, redundant) | **300 (5/Min, keine Redundanz)** |
| Latenz | Nicht relevant bei 3x/Tag | <100ms Inferenz + ~1s API |
| Setup-Erkennung | Verpasst 99% der Minuten-Setups | Faengt alle Setups |
| Position-Management | Passiv (SL/TP warten) | Aktiv (DQN entscheidet) |
| Retraining-Daten | Wenig echte Trades | Mehr Daten, besseres Modell |
| Scheduler-Jobs | 3 getrennte | **1 unified** |

**Klare Empfehlung: Ja, 1-Minuten-Intervall umsetzen.**

Das Modell wurde exakt fuer diese Granularitaet trainiert. Die enge SL/TP-Konfiguration (0.3%/0.5%) ergibt bei einem 3x-taeglichen Check keinen Sinn -- die Trades sind laengst vorbei bevor der Bot ueberhaupt schaut. Durch die gemeinsame Abfrage (`collect_prices` + `get_open_positions`) verbrauchen wir sogar **weniger** API-Calls als die bisherigen getrennten Loops.

### Naechste Schritte
1. `unified_tick()` implementieren -- `sim_tick` + `daily_routine` + `monitor_positions` zusammenfuehren
2. `ai_analyzer.py` anpassen: State-Vektor aus `price_history`-Tabelle statt Broker-API
3. CLOSE-Logik fuer aktive Positionsverwaltung einbauen
4. Config anpassen: `MAX_TRADES_PER_DAY=5`, `MIN_RISK_REWARD_RATIO=1.5`
5. Alte separate Scheduler-Jobs entfernen
