# Dashboard Refactor — Implementierungsplan

## Übersicht

Vier Ziele:
1. Dashboard in separate Page-Dateien aufteilen (Multi-Page-App)
2. Überbleibsel der alten Struktur entfernen, an neue Bot-Struktur anpassen
3. "Markt jetzt analysieren" Button auf der Trading-Seite
4. Neue Visualisierungen für offene und geschlossene Trades

---

## 1. Dateistruktur — Multi-Page-App

### Empfehlung: Ja, aufteilen — aber mit Bedacht

Streamlit unterstützt Multi-Page-Apps nativ über das `pages/`-Verzeichnis.
Jede Datei in `pages/` wird automatisch als eigene Seite in der Sidebar angezeigt.

**Neue Struktur:**

```
TadingBot/
├── dashboard.py                  # Einstiegspunkt (nur Page-Config + Redirect)
├── pages/
│   ├── 1_Trading.py              # Tab 1: Live Trading, offene Positionen, Analyse
│   ├── 2_Simulation.py           # Tab 2: Simulationshistorie, Ergebnisse
│   ├── 3_Historische_Daten.py    # Tab 3: Datenabruf, Kerzen-Übersicht
│   └── 4_Trainingsdaten.py       # Tab 4: DB-Viewer + Trainingsdaten-Manager
├── dashboard_shared.py           # Gemeinsame Konstanten, CSS, Helper-Funktionen
└── src/
    └── ...
```

**Was in `dashboard_shared.py` kommt:**
- `BOT_API_URL`, `DATA_DIR`, `DB_PATH`, `SIM_DB_PATH` etc.
- Der gesamte CSS-Block (Cyber-Design)
- `get_connection()`, `query()`, `sim_query()`
- `load_trades()`, `load_snapshots()`, `load_analyses()`
- `load_log_lines()`

**Warum nicht alles trennen:**
- Die Backtest-Hilfsfunktionen (`_render_single_backtest`, `_render_batch_backtest`,
  `_run_single_backtest`, `_display_single_result`, `_display_batch_summary`, `_results_to_csv`)
  sind groß und eng an die Trading-Seite gekoppelt — bleiben in `pages/1_Trading.py`.

---

## 2. Überbleibsel der alten Struktur entfernen

Nach Analyse von `dashboard.py` und `src/api.py` sind folgende Elemente **nicht mehr
passend zur aktuellen Bot-Struktur** und sollen entfernt oder ersetzt werden:

### Zu entfernen aus Trading-Seite:

| Element | Warum entfernen |
|---|---|
| `### ◈ API TESTS` (Capital.com / DQN / Ntfy) | Gehört in eine Admin/Settings-Seite, nicht in Trading |
| `### ◈ PENDING RECHECKS` | `pending_rechecks`-Tabelle existiert nur in der alten trades.db — DQN-Bot nutzt das nicht mehr |
| `### ◈ TRADE REVIEW` + `### ◈ WAS HAT DER BOT GELERNT?` | Greift auf `trade_reviews`-Tabelle zu, die im neuen Bot nicht mehr existiert |
| `### ◈ DQN ANALYSEN` (Expander mit `daily_analyses`) | `daily_analyses`-Tabelle nicht mehr in neuer Struktur |
| Button `TAGESBILANZ` → `/api/daily-summary` | Endpoint ruft `DQNAnalyzer.generate_summary()` auf (LLM-Kosten), nicht mehr nötig |
| Button `WOCHENREPORT` → `/api/weekly-report` | Gleich wie oben, Legacy-Endpoint |
| Button `DQN BACKTEST` (großer Backtest-Block) | Komplex, veraltet — ersetzt durch neuen Analyse-Button (Punkt 3) |
| `load_analyses()` | Greift auf `daily_analyses` zu → weg |
| `KONTOSTAND VERLAUF` chart | Greift auf `account_snapshots` zu — prüfen ob Tabelle noch existiert |
| Sidebar: `Bot-Einstellungen` aus `.env` | Kann bleiben aber vereinfachen |

### Was bleibt / angepasst wird:

| Element | Anpassung |
|---|---|
| KPI-Metriken (Kontostand, Positionen, P/L, Win-Rate) | Bleibt, Datenquelle prüfen |
| Offene Positionen Tabelle | Bleibt, wird visuell überarbeitet (Punkt 4) |
| Trade History Tabelle | Bleibt, wird visuell überarbeitet (Punkt 4) |
| Sidebar Auto-Refresh + Settings | Bleibt, vereinfachen |
| Simulation-Tab | Prüfen welche Endpunkte noch aktiv sind |

### Zu prüfen (unklar ob noch aktiv):
- `trades.db` — existiert diese DB noch? Welche Tabellen?
- `/api/status` — gibt es diesen Endpoint noch?
- `account_snapshots`-Tabelle — noch vorhanden?

---

## 3a. Bot On/Off Switch

Ein prominenter Toggle oben auf der Trading-Seite zum Starten/Stoppen des Bots.

**Design — großer Status-Switch:**
```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   BOT STATUS         ◉ AKTIV       ○ GESTOPPT      │
│                  [  ████  AN  ]                     │
│                                                     │
└─────────────────────────────────────────────────────┘
```

- Grüner Hintergrund + Pulsieren-Animation wenn AN
- Roter Hintergrund wenn AUS
- Streamlit `st.toggle()` mit custom CSS-Styling
- Zustand wird beim Umschalten sofort via API gesetzt:
  - AN → `POST /api/bot/start`
  - AUS → `POST /api/bot/stop`
- Aktueller Zustand beim Laden der Seite von `GET /api/status` abgerufen

**API-Endpoints (neu in `src/api.py`):**
```
POST /api/bot/start   → startet den Trading-Loop
POST /api/bot/stop    → stoppt den Trading-Loop (laufende Positionen bleiben offen)
GET  /api/bot/status  → gibt {"running": true/false} zurück
```

**Platzierung:** Ganz oben auf der Trading-Seite, vor den KPI-Metriken — gut sichtbar.

---

## 3b. "Markt jetzt analysieren" Button

### Neues Konzept

Ersetzt den alten komplexen Backtest-Block. Ein einzelner prominenter Button
der den DQN-Analyzer für alle Assets aufruft und die Signale live anzeigt.

**Button:** `◈ MARKT JETZT ANALYSIEREN`

**Vorhandener API-Endpoint:** `POST /api/analyze` — gibt bereits zurück:
```json
{
  "signals": [
    {"asset": "GOLD", "action": "SELL", "confidence": 8, "q_values": [...], ...},
    ...
  ],
  "open_positions": 2,
  "timestamp": "..."
}
```

**UI nach Button-Klick — Signal-Cards:**

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   GOLD          │  │   SILVER        │  │   OIL_CRUDE     │  │  NATURALGAS     │
│                 │  │                 │  │                 │  │                 │
│  ▼ SELL         │  │  ● HOLD         │  │  ▲ BUY          │  │  ● HOLD         │
│  Conf: 8/10     │  │  Conf: 4/10     │  │  Conf: 7/10     │  │  Conf: 3/10     │
│  ████████░░     │  │  ████░░░░░░     │  │  ███████░░░     │  │  ███░░░░░░░     │
└─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘
   Hintergrund rot       Hintergrund grau     Hintergrund grün     Hintergrund grau
```

**Farbcodierung:**
- BUY → grüner Rahmen / leicht grüner Hintergrund
- SELL → roter Rahmen / leicht roter Hintergrund
- HOLD → normaler Rahmen (#00ff41)

**Confidence-Bar:** Plotly horizontal gauge oder einfache HTML-Progressbar

**Implementierung in `pages/1_Trading.py`:**
```python
if st.button("◈ MARKT JETZT ANALYSIEREN", type="primary"):
    with st.spinner("DQN analysiert alle Assets..."):
        resp = httpx.post(f"{BOT_API_URL}/api/analyze", timeout=30)
        signals = resp.json()["signals"]
    # 4 Spalten mit Signal-Cards
    for col, signal in zip(st.columns(4), signals):
        with col:
            _render_signal_card(signal)
```

---

## 4. Visualisierungen für Trades

### 4a. Offene Positionen

Statt einfacher Tabelle: **Cards pro Position** + kompakte Übersichtstabelle

**Card-Layout pro offener Position:**
```
┌────────────────────────────────────────────────┐
│  GOLD  ▲ BUY          ● OFFEN seit 2h 34m      │
├────────────────────────────────────────────────┤
│  Entry:  4,892.50                              │
│  SL:     4,877.80  (▼ 0.30%)                   │
│  TP:     4,916.90  (▲ 0.50%)                   │
├────────────────────────────────────────────────┤
│  [SL]──────────[ENTRY]──────────[TP]           │
│   ◄──── 0.30% ────►◄──── 0.50% ────►          │
│  Conf: 8/10  |  Size: 0.42  |  Deal: ABC123   │
└────────────────────────────────────────────────┘
```

Der horizontale Balken zeigt visuell den SL/TP-Abstand —
grün rechts (TP), rot links (SL), weißer Punkt = aktuell bei Entry.

**Implementierung:**
- Plotly `go.Bar` horizontal oder pure HTML/CSS
- Für jede offene Position eine Card via `st.container()` + custom HTML

### 4b. Geschlossene Trades

**Zwei-Ebenen-Design:**

**Ebene 1 — KPI-Row (immer sichtbar):**
```
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│  Trades  │ │  Win-Rate│ │ Gesamt   │ │  Bester  │ │Schlechst.│
│   142    │ │  41.3%   │ │+514.68€  │ │ +27.03€  │ │ -16.44€  │
└──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
```

**Ebene 2 — Charts (Plotly):**

1. **Equity-Kurve** (Linie, grün/rot fill):
   - X: Trade-Nummer, Y: kumuliertes P/L
   - Fill `tozeroy`, grün wenn positiv, rot wenn negativ
   - Hover zeigt: Asset, Richtung, P/L, Datum

2. **P/L Balkendiagramm** (letzte 30 Trades):
   - Grüne Balken = TP, rote Balken = SL
   - X: Trade-Datum, Y: P/L in Punkten
   - Referenzlinie bei 0

3. **Win-Rate Heatmap nach Stunde/Wochentag** (optional, wenn genug Daten):
   - X: Wochentag (Mo–Fr), Y: Stunde (0–23)
   - Farbe: Win-Rate in dem Slot

**Ebene 3 — Gefilterte Tabelle:**
- Gleiche Filter wie jetzt (Asset, Status, Zeitraum)
- Farbige Status-Badges statt reinem Text:
  - `closed_tp` → grüner Badge
  - `closed_sl` → roter Badge
  - `closed_end` → grauer Badge

---

## Implementierungsreihenfolge

```
Schritt 1:  dashboard_shared.py erstellen (CSS + Helpers)
Schritt 2:  pages/ Verzeichnis + 4 Seiten-Stubs anlegen
Schritt 3:  dashboard.py auf Redirect reduzieren
Schritt 4:  pages/1_Trading.py — Überbleibsel entfernen, Struktur aufräumen
Schritt 5:  "Markt analysieren" Button implementieren (Signal-Cards)
Schritt 6:  Offene Positionen — Card-Visualisierung
Schritt 7:  Geschlossene Trades — Equity-Kurve + P/L Chart (Plotly)
Schritt 8:  pages/2_Simulation.py — bestehenden Code migrieren
Schritt 9:  pages/3_Historische_Daten.py — bestehenden Code migrieren
Schritt 10: pages/4_Trainingsdaten.py — DB-Viewer + Manager migrieren
```

---

## Offene Fragen (vor Implementierung klären)

1. **`trades.db` noch aktiv?** — Die Trading-Seite liest aus `trades.db`
   (Tabellen: `trades`, `account_snapshots`, `daily_analyses`, `trade_reviews`,
   `pending_rechecks`). Welche davon schreibt der aktuelle Bot noch?

2. **`/api/status` Endpoint** — Ist der noch deployed und aktuell?

3. **Simulation-Tab** — Welche Endpunkte sind da noch relevant?
   (`/api/simulation/*`?)

4. **Sidebar Settings** — Sollen die Bot-Einstellungen (`.env`-Editor) bleiben
   oder in eine eigene Seite?
