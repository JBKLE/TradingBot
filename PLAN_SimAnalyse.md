# Plan: KI-gestützte Simulationsanalyse

## Ziel
Nach einer Simulation (live oder aus Historie geladen) kann per Knopfdruck eine
tiefgehende Analyse durch Claude Opus 4.6 gestartet werden. Die Antwort wird live
gestreamt und direkt im Dashboard angezeigt.

---

## Was die Analyse liefert

1. **Asset-Fortschritt / Rückschritt**
   - Vergleich aktueller Run mit den letzten N Runs aus `sim_history.db`
   - Pro Asset: Win-Rate-Trend, P/L-Trend, R-Multiple-Trend

2. **Trainingsempfehlungen**
   - Welche Assets / Richtungen (BUY/SELL) mit schlechten Daten das Training vergiften
   - Vorschlag: "Nächstes Training nur mit Trades wo Conf ≥ X und Asset = Y"
   - Hinweis auf Overfitting-Risiken

3. **Konkrete nächste Schritte**
   - Welcher Asset soll aus der Watchlist raus / stärker trainiert werden
   - Zeitfenster-Filter die sich lohnen
   - Confidence-Schwellen-Empfehlung

---

## Datenbasis pro Analyse-Aufruf

| Datenquelle         | Inhalt                                                      |
|---------------------|-------------------------------------------------------------|
| `sim_result`        | Aktueller Run: alle Metriken, per-asset, trade_list (sample)|
| `sim_history.db`    | Letzte 10 Runs: Zeitstempel, Modell, Metriken, per-asset   |
| Request-Kontext     | Confidence-Schwelle, Assets, Zeitraum, Modellname           |

---

## Technischer Aufbau

### Backend: `src/sim_analyzer.py`
- Funktion `build_analysis_prompt(current_result, history_runs)` → strukturierter Prompt
- Async-Generator `stream_analysis(prompt)` → streamt Chunks via Claude Opus 4.6

### API: `POST /api/sim-analysis`
- Nimmt `current_result` + `run_id` (optional) als Body
- Lädt History aus `sim_history.db`
- Gibt `StreamingResponse` (text/event-stream) zurück
- Claude: `claude-opus-4-6`, `thinking: adaptive`, streaming

### Dashboard
- Button **"KI-Analyse starten"** erscheint wenn `sim_result` vorhanden
- `httpx.stream()` → chunks werden live in `st.write_stream()` angezeigt
- Analyse-Text bleibt in `st.session_state["sim_analysis"]` gespeichert

---

## Dateien die geändert / neu erstellt werden

| Datei                   | Änderung                                          |
|-------------------------|---------------------------------------------------|
| `src/sim_analyzer.py`   | NEU — Prompt-Builder + Claude-Streaming-Generator |
| `src/api.py`            | NEU — `POST /api/sim-analysis` Endpoint           |
| `dashboard.py`          | Analyse-Button + Streaming-Anzeige im Result-Block|
| `.env`                  | `ANTHROPIC_API_KEY=...` (muss vom User gesetzt werden)|
| `requirements.txt`      | `anthropic>=0.40.0` hinzufügen                   |

---

## Prompt-Strategie

```
System:
  Du bist Analyse-KI für einen DQN-Trading-Bot. Antworte auf Deutsch.
  Sei präzise, zahlenbasiert und direkt. Kein Bullshit.

User:
  ## Aktueller Run
  Modell: shared_dqn.pt | Zeitraum: 2026-03-01 bis 2026-03-26
  Confidence ≥ 8 | Output-DB: sim_test_v4.db

  Gesamt: 338 Trades | WR: 38.5% | P/L: +89.41 Pkt | Avg R: +0.026

  Pro Asset:
  | Asset      | Trades | WR    | P/L      | Avg R  |
  | GOLD       | 20     | 50.0% | +90.84   | +0.333 |
  | SILVER     | 91     | 42.9% | +2.85    | +0.143 |
  | NATURALGAS | 52     | 40.4% | +0.04    | +0.077 |
  | OIL_CRUDE  | 175    | 34.3% | -4.32    | -0.086 |

  ## Letzte 5 Runs (Verlauf)
  [Run #1..#5 Tabelle]

  ---
  Analysiere:
  1. Macht das Modell Fortschritte oder Rückschritte? Pro Asset begründet.
  2. Welche Trades / Assets schaden dem Training? Was konkret tun?
  3. Was sind die 3 wichtigsten nächsten Schritte?
```

---

## Sicherheit / API-Key
- `ANTHROPIC_API_KEY` wird aus `.env` gelesen (wie alle anderen Keys)
- Key wird NIE geloggt oder in der DB gespeichert
- Fehler bei fehlendem Key → klare Meldung im Dashboard

---

## Nicht im Scope (dieses Features)
- Automatisches Speichern der Analyse in DB (kann später ergänzt werden)
- Multi-Modell-Vergleich (würde separate Modell-Registry brauchen)
- Chart-Generierung durch Claude (Streamlit macht das bereits)
