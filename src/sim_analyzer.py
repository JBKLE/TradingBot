"""KI-gestützte Simulationsanalyse via Claude Opus 4.6.

Streamt die Analyse direkt zurück — kein Warten auf vollständige Antwort.
"""
import json
import logging
import os
from typing import AsyncIterator

logger = logging.getLogger(__name__)

MODEL = "claude-opus-4-6"
MAX_TOKENS = 16000

SYSTEM_PROMPT = """Du bist ein spezialisierter Analyse-Assistent für einen DQN-Trading-Bot.

Deine Aufgabe: Simulationsergebnisse auswerten und dem Entwickler klare,
zahlenbasierte Handlungsempfehlungen geben.

Regeln:
- Antworte auf Deutsch
- Sei direkt und konkret — keine Floskeln
- Nenne immer konkrete Zahlen aus den Daten
- Strukturiere deine Antwort mit den vorgegebenen Abschnitten
- Wenn Vergleichsdaten fehlen, sage es kurz und arbeite mit dem was da ist"""


def _fmt_asset_table(per_asset: dict) -> str:
    if not per_asset:
        return "  (keine Asset-Daten)"
    lines = ["  | Asset       | Trades | WR %  | P/L (Pkt) | Avg R  |",
             "  |-------------|--------|-------|-----------|--------|"]
    for asset, s in per_asset.items():
        lines.append(
            f"  | {asset:<11s} | {s.get('trades',0):6d} | "
            f"{s.get('win_rate',0):5.1f} | {s.get('pnl',0):+9.4f} | "
            f"{s.get('win_rate',0)/100*(s.get('pnl',0) or 1):.3f} |"
        )
    return "\n".join(lines)


def _fmt_history_table(runs: list[dict]) -> str:
    if not runs:
        return "  (keine früheren Runs vorhanden)"
    lines = [
        "  | # | Datum            | Modell             | Assets | Trades | WR %  | P/L (Pkt) | Avg R  |",
        "  |---|------------------|--------------------|--------|--------|-------|-----------|--------|",
    ]
    for r in runs:
        assets_str = ",".join(r.get("assets") or []) or "alle"
        lines.append(
            f"  | {r['id']:3d} | {str(r.get('run_at',''))[:16]} | "
            f"{str(r.get('model_name',''))[:18]:<18s} | "
            f"{assets_str[:6]:<6s} | {r.get('trades',0):6d} | "
            f"{r.get('win_rate',0):5.1f} | {r.get('total_pnl_points',0):+9.4f} | "
            f"{r.get('avg_r_multiple',0):+.4f} |"
        )
    return "\n".join(lines)


def build_prompt(current: dict, history_runs: list[dict]) -> str:
    """Strukturierten Analyse-Prompt aus Simulationsdaten bauen."""
    meta = current.get("_history_meta") or {}
    fin  = current.get("financial") or {}

    # Aktuelle Run-Infos
    model_name  = meta.get("model_name") or "unbekannt"
    confidence  = meta.get("confidence") or current.get("confidence_threshold", "?")
    output_db   = meta.get("output_db") or "?"
    assets_used = ", ".join(meta.get("assets") or []) or "alle"
    start_ts    = current.get("start_ts", "?")[:10]
    end_ts      = current.get("end_ts",   "?")[:10]

    trades   = current.get("trades", 0)
    wins     = current.get("wins", 0)
    losses   = current.get("losses", 0)
    win_rate = current.get("win_rate", 0.0)
    pnl      = current.get("total_pnl_points", 0.0)
    avg_r    = current.get("avg_r_multiple", 0.0)
    per_asset = current.get("per_asset") or {}

    # Finanz-Zusammenfassung (falls vorhanden)
    fin_section = ""
    if fin.get("start_capital") is not None:
        fin_section = (
            f"\n  Finanzen: €{fin['start_capital']:,.0f} → "
            f"€{fin.get('end_capital',0):,.0f} "
            f"({fin.get('total_return_pct',0):+.2f}%) | "
            f"Max DD: {fin.get('max_drawdown_pct',0):.2f}%"
            + (" | ⚠ MARGIN CALL" if fin.get("margin_call") else "")
        )

    # Trade-Stichprobe (letzte 20, nur relevante Felder)
    trade_list = current.get("trade_list") or []
    sample_trades = trade_list[-20:] if len(trade_list) > 20 else trade_list
    if sample_trades:
        sl_count = sum(1 for t in trade_list if t.get("status") == "closed_sl")
        tp_count = sum(1 for t in trade_list if t.get("status") == "closed_tp")
        buy_count  = sum(1 for t in trade_list if t.get("direction") == "BUY")
        sell_count = sum(1 for t in trade_list if t.get("direction") == "SELL")
        trade_summary = (
            f"\n  TP-Treffer: {tp_count} | SL-Treffer: {sl_count} | "
            f"BUY: {buy_count} ({buy_count/max(trades,1)*100:.0f}%) | "
            f"SELL: {sell_count} ({sell_count/max(trades,1)*100:.0f}%)"
        )
    else:
        trade_summary = ""

    prompt = f"""## AKTUELLER SIMULATIONS-RUN

  Modell:      {model_name}
  Zeitraum:    {start_ts} bis {end_ts}
  Confidence:  ≥ {confidence}
  Assets:      {assets_used}
  Output-DB:   {output_db}

  GESAMT:  {trades} Trades | WR: {win_rate:.1f}% ({wins}W/{losses}L) | P/L: {pnl:+.4f} Pkt | Avg R: {avg_r:+.4f}{fin_section}{trade_summary}

  PRO ASSET:
{_fmt_asset_table(per_asset)}

---

## LETZTE {len(history_runs)} RUNS (VERLAUF)

{_fmt_history_table(history_runs)}

---

## DEINE ANALYSE (bitte genau diese Struktur verwenden):

### 1. Fortschritt / Rückschritt pro Asset
Vergleiche den aktuellen Run mit dem Verlauf. Was verbessert sich, was verschlechtert sich?
Begründe mit konkreten Zahlen.

### 2. Trainingsqualität & -empfehlungen
Welche Trades / Assets schaden dem Training? Was sollte beim nächsten Trainingslauf anders gemacht werden?
Sei sehr konkret (z.B. "Filtere OIL_CRUDE-SELL-Trades unter Conf 7 heraus, da...").

### 3. Die 3 wichtigsten nächsten Schritte
Was bringt den größten Fortschritt? Sortiert nach Priorität."""

    return prompt


async def stream_analysis(current: dict, history_runs: list[dict]) -> AsyncIterator[str]:
    """Streamt die Claude-Analyse als Text-Chunks."""
    try:
        import anthropic
    except ImportError:
        yield "**Fehler:** `anthropic`-Paket nicht installiert. Bitte `pip install anthropic` ausführen."
        return

    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        yield "**Fehler:** `ANTHROPIC_API_KEY` nicht gesetzt. Bitte in `.env` eintragen."
        return

    client   = anthropic.AsyncAnthropic(api_key=api_key)
    prompt   = build_prompt(current, history_runs)

    logger.info("sim_analyzer: starting Claude stream (model=%s)", MODEL)
    try:
        async with client.messages.stream(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for text in stream.text_stream:
                yield text
    except Exception as exc:
        logger.exception("sim_analyzer: Claude API error: %s", exc)
        yield f"\n\n**API-Fehler:** {exc}"
