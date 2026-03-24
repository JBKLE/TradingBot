"""Claude-based market analysis module."""
import json
import logging
from datetime import datetime
from typing import Optional

import anthropic

from . import config
from .models import (
    AnalysisResult,
    AssetOutlook,
    BestOpportunity,
    Direction,
    EscalationResult,
    MarketData,
    PositionInfo,
    Recommendation,
)
from .news_analyzer import MarketContext

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
Du bist ein erfahrener Rohstoff-Analyst und Trading-Berater.
Deine Aufgabe ist es, täglich die beste Trading-Gelegenheit aus einer
Watchlist von Rohstoffen (Gold, Silber, Öl, Gas) zu identifizieren.

REGELN:
- Antworte IMMER als valides JSON (kein Markdown, kein Text davor/danach)
- Confidence-Score von 1-10 (10 = absolut sicher)
- Empfehle "WAIT" wenn kein Asset einen Confidence-Score >= 7 hat
- Risk/Reward-Ratio muss mindestens 1:1.5 sein
- Berücksichtige: Geopolitik, Zentralbank-Politik, Angebot/Nachfrage, Dollar-Stärke, Ölpreis
- Stop-Loss so setzen, dass maximal {max_risk_pct}% des Kontostands riskiert werden
- Keine Trades vor/während wichtiger Wirtschaftsdaten (Fed, EZB, NFP etc.)
- Bevorzuge klare Trends, meide Seitwärtsphasen
- Wenn du WAIT empfiehlst aber ein vielversprechendes Setup siehst (Confidence 5-7),
  setze recheck.worthy = true. Beschreibe eine KONKRETE Trigger-Bedingung (z.B.
  "Breakout ueber 2350" oder "RSI faellt unter 30") und schaetze wann der Recheck
  sinnvoll waere (30-120 Minuten). Der Bot wird dann automatisch pruefen.
- Wenn mehrere Assets gleich gut aussehen, wähle das mit dem besten Risk/Reward
- Dir werden berechnete technische Indikatoren mitgegeben. Nutze diese als primäre
  Entscheidungsgrundlage statt eigene Berechnungen aus den Kerzen abzuleiten.
- Wenn der ATR als "erhöht" markiert ist (>1.5x Durchschnitt), senke automatisch
  deinen Confidence-Score um mindestens 2 Punkte.
- Stop-Loss und Take-Profit werden regelbasiert berechnet – gib trotzdem deine
  Einschätzung für Entry-Preis, SL und TP an, aber die endgültige Berechnung
  übernimmt das System.
- Antworte AUSSCHLIESSLICH mit dem folgenden JSON-Schema, ohne zusätzlichen Text:

{{
    "date": "YYYY-MM-DD",
    "market_summary": "Kurze Zusammenfassung der Marktlage",
    "best_opportunity": {{
        "asset": "GOLD|SILVER|OIL_CRUDE|NATURALGAS",
        "direction": "BUY|SELL|NONE",
        "confidence": 1-10,
        "reasoning": "Ausführliche Begründung",
        "entry_price": 0.0,
        "stop_loss": 0.0,
        "take_profit": 0.0,
        "risk_reward_ratio": 0.0,
        "key_events": ["Event 1", "Event 2"],
        "risks": ["Risiko 1", "Risiko 2"]
    }},
    "other_assets": [
        {{"asset": "SILVER", "outlook": "bullish|bearish|neutral", "confidence": 1-10, "note": "..."}}
    ],
    "recommendation": "TRADE|WAIT",
    "wait_reason": "Falls WAIT: Warum kein Trade heute?",
    "recheck": {{
        "worthy": true/false,
        "asset": "GOLD|SILVER|OIL_CRUDE|NATURALGAS",
        "direction": "BUY|SELL",
        "trigger_condition": "Konkrete Bedingung wann das Setup reif waere",
        "recheck_in_minutes": 30-120,
        "current_confidence": 1-10,
        "expected_confidence_if_trigger": 1-10
    }}
}}

HANDELSSTIL: {trading_style}

AKTUELLES DATUM: {current_date}
KONTOSTAND: {account_balance} EUR
OFFENE POSITIONEN: {open_positions}
MAX RISIKO PRO TRADE: {max_risk_pct}% des Kontostands
"""

_STYLE_HINTS = {
    "swing": (
        "SWING-TRADING (Haltedauer 1-5 Tage) – "
        "Fokus auf klare Trendrichtung anhand von Tageskerzen. "
        "Stop-Loss und Take-Profit auf Basis markanter Tages-Levels."
    ),
    "intraday": (
        "INTRADAY-TRADING (bevorzugt < 24h Haltedauer) – "
        "Fokus auf kurzfristige Impulse anhand von Stundenkerzen. "
        "Enge Stop-Losses, kleinere Take-Profits, schnelle Setups bevorzugen. "
        "Trades die abends noch offen sind können über Nacht laufen – "
        "Swap-Kosten sind vertretbar wenn das Setup weiterhin intakt ist. "
        "Kein Trade mehr nach {close_time} Uhr eröffnen."
    ),
}


class MarketAnalyzer:
    """Uses Claude to analyse market data and generate trade signals."""

    def __init__(self) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)

    async def analyze_market(
        self,
        market_data: dict[str, MarketData],
        account_balance: float,
        open_positions: list[PositionInfo],
        market_context: Optional[MarketContext] = None,
        indicators: dict[str, dict] | None = None,
        performance_stats: dict | None = None,
        recent_lessons: list[dict] | None = None,
    ) -> AnalysisResult:
        """
        Send market data to Claude and receive a structured analysis.

        Uses adaptive thinking for deeper reasoning on complex market conditions.
        Streams the response to avoid HTTP timeouts.
        """
        user_message = _build_user_message(market_data, market_context, indicators)
        system_prompt = _SYSTEM_PROMPT.format(
            current_date=datetime.now().strftime("%Y-%m-%d"),
            account_balance=f"{account_balance:.2f}",
            open_positions=_format_open_positions(open_positions),
            max_risk_pct=config.MAX_RISK_PER_TRADE_PCT,
            trading_style=_STYLE_HINTS.get(
                config.TRADING_STYLE, _STYLE_HINTS["swing"]
            ).replace("{close_time}", config.TRADE_WINDOW_END),
        )

        # ── Lern-Kontext anfuegen (Performance + Lessons Learned) ────────
        learning_ctx = _build_learning_context(performance_stats, recent_lessons)
        if learning_ctx:
            system_prompt += learning_ctx

        logger.info("Requesting Claude market analysis (%d assets)...", len(market_data))

        raw_text = ""
        usage_input = 0
        usage_output = 0

        async with self._client.messages.stream(
            model=config.CLAUDE_MODEL,
            max_tokens=8192,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            async for event in stream:
                pass  # drain stream events
            final = await stream.get_final_message()

        # Extract text content (thinking blocks are skipped)
        for block in final.content:
            if block.type == "text":
                raw_text = block.text.strip()
                break

        usage_input = final.usage.input_tokens
        usage_output = final.usage.output_tokens

        # Estimate cost (claude-sonnet-4-6: $3/1M input, $15/1M output)
        cost_usd = (usage_input * 3.0 + usage_output * 15.0) / 1_000_000

        logger.info(
            "Claude analysis received (tokens: %d in / %d out, cost: $%.4f)",
            usage_input,
            usage_output,
            cost_usd,
        )

        analysis = _parse_analysis(raw_text)
        analysis.tokens_used = usage_input + usage_output
        analysis.cost_usd = cost_usd
        return analysis

    async def escalate_position(
        self,
        trade: "Trade",  # type: ignore[name-defined]
        escalation_reason: str,
        current_price: float,
        profit_loss: float,
        profit_loss_pct: float,
    ) -> EscalationResult:
        """Call Claude to evaluate an open position that triggered an escalation rule."""
        from datetime import datetime as _dt
        now = _dt.now(tz=config.TZ)
        ts = trade.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=config.TZ)
        duration_h = (now - ts).total_seconds() / 3600

        prompt = _ESCALATION_PROMPT.format(
            asset=trade.asset,
            direction=trade.direction.value,
            entry_price=trade.entry_price,
            entry_time=ts.strftime("%Y-%m-%d %H:%M"),
            current_price=current_price,
            stop_loss=trade.stop_loss,
            take_profit=trade.take_profit,
            profit_loss=profit_loss,
            profit_loss_pct=profit_loss_pct,
            duration=f"{duration_h:.1f}h",
            escalation_reason=escalation_reason,
        )
        logger.info("Escalating position %s to Claude: %s", trade.id, escalation_reason)

        raw_text = ""
        async with self._client.messages.stream(
            model=config.CLAUDE_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for _ in stream:
                pass
            final = await stream.get_final_message()

        for block in final.content:
            if block.type == "text":
                raw_text = block.text.strip()
                break

        try:
            import json as _json
            data = _json.loads(raw_text)
            return EscalationResult(
                action=data.get("action", "HOLD"),
                reasoning=data.get("reasoning", ""),
                new_stop_loss=data.get("new_stop_loss"),
                new_take_profit=data.get("new_take_profit"),
                urgency=data.get("urgency", "low"),
            )
        except Exception as exc:
            logger.error("Failed to parse escalation response: %s", exc)
            return EscalationResult(action="HOLD", reasoning=f"Parse error: {exc}", urgency="low")


    async def review_trade(
        self,
        trade,
        price_bars_after: list | None = None,
    ) -> dict:
        """Post-Trade Analyse: Claude bewertet einen geschlossenen Trade."""
        ts = trade.timestamp
        exit_ts = trade.exit_timestamp or datetime.now(tz=config.TZ)

        history_str = "Keine Daten verfuegbar"
        if price_bars_after:
            lines = []
            for bar in price_bars_after[-15:]:
                lines.append(
                    f"  {bar.timestamp}: O={bar.open:.4f} H={bar.high:.4f} "
                    f"L={bar.low:.4f} C={bar.close:.4f}"
                )
            history_str = "\n".join(lines)

        prompt = _REVIEW_PROMPT.format(
            asset=trade.asset,
            direction=trade.direction.value,
            entry_price=trade.entry_price,
            entry_time=ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "strftime") else str(ts),
            exit_price=trade.exit_price or trade.entry_price,
            exit_time=exit_ts.strftime("%Y-%m-%d %H:%M") if hasattr(exit_ts, "strftime") else str(exit_ts),
            stop_loss=trade.stop_loss,
            take_profit=trade.take_profit,
            profit_loss=trade.profit_loss or 0,
            profit_loss_pct=trade.profit_loss_pct or 0,
            status=trade.status.value,
            confidence=trade.confidence,
            reasoning=trade.reasoning[:500] if trade.reasoning else "",
            price_history_after=history_str,
        )

        logger.info("Requesting trade review for Trade %s (%s)...", trade.id, trade.asset)

        raw_text = ""
        async with self._client.messages.stream(
            model=config.CLAUDE_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for _ in stream:
                pass
            final = await stream.get_final_message()

        for block in final.content:
            if block.type == "text":
                raw_text = block.text.strip()
                break

        try:
            text = raw_text
            if text.startswith("```"):
                text = text.split("```", 2)[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.rsplit("```", 1)[0]
            result = json.loads(text.strip())
            logger.info(
                "Trade review erhalten: quality=%s, lesson=%s",
                result.get("entry_quality", "?"),
                result.get("lesson_learned", "?")[:80],
            )
            return result
        except Exception as exc:
            logger.error("Failed to parse trade review: %s", exc)
            return {
                "entry_quality": "unknown",
                "lesson_learned": f"Review konnte nicht geparsed werden: {exc}",
            }


    async def recheck_opportunities(
        self,
        rechecks: list,
        market_data: dict,
        indicators: dict,
    ) -> list[dict]:
        """Batched recheck: prueft mehrere Setups in einem Claude-Call."""
        setup_lines = []
        for i, rc in enumerate(rechecks, 1):
            setup_lines.append(f"SETUP {i}:")
            setup_lines.append(f"- Asset: {rc.asset}")
            setup_lines.append(f"- Richtung: {rc.direction.value}")
            setup_lines.append(f"- Trigger-Bedingung: {rc.trigger_condition}")
            setup_lines.append(f"- Bisherige Confidence: {rc.current_confidence}")
            setup_lines.append(f"- Recheck #{rc.recheck_count + 1} von max {rc.max_rechecks}")
            setup_lines.append("")

        market_lines = []
        for asset_key, data in market_data.items():
            price = data.current_price
            market_lines.append(f"### {asset_key}")
            market_lines.append(f"- Bid={price.bid:.4f} / Ask={price.ask:.4f}")
            market_lines.append(f"- Tageshoch/-tief: {price.high:.4f} / {price.low:.4f}")
            market_lines.append(f"- Veraenderung: {price.change_pct:+.2f}%")
            if asset_key in indicators:
                ind = indicators[asset_key]
                if ind.get("rsi") is not None:
                    market_lines.append(f"- RSI: {ind['rsi']:.1f}")
                if ind.get("atr") is not None:
                    market_lines.append(f"- ATR: {ind['atr']:.4f}")
                macd = ind.get("macd")
                if macd:
                    market_lines.append(f"- MACD Histogram: {macd['histogram']:.4f}")
                sr = ind.get("support_resistance")
                if sr:
                    sup = ", ".join(f"{s:.4f}" for s in sr.get("support", []))
                    res = ", ".join(f"{r:.4f}" for r in sr.get("resistance", []))
                    if sup or res:
                        market_lines.append(f"- Support: {sup or 'n/a'} | Resistance: {res or 'n/a'}")
            market_lines.append("")

        prompt = _RECHECK_PROMPT.format(
            setups_text="\n".join(setup_lines),
            market_data_text="\n".join(market_lines),
        )

        logger.info("Sende %d Recheck(s) an Claude...", len(rechecks))

        raw_text = ""
        async with self._client.messages.stream(
            model=config.CLAUDE_MODEL,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for _ in stream:
                pass
            final = await stream.get_final_message()

        for block in final.content:
            if block.type == "text":
                raw_text = block.text.strip()
                break

        try:
            text = raw_text
            if text.startswith("```"):
                text = text.split("```", 2)[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.rsplit("```", 1)[0]
            results = json.loads(text.strip())
            if isinstance(results, dict):
                results = [results]
            while len(results) < len(rechecks):
                results.append({"is_ready": False, "retry_worthy": False, "reasoning": "Keine Antwort"})
            return results[: len(rechecks)]
        except Exception as exc:
            logger.error("Failed to parse recheck response: %s", exc)
            return [{"is_ready": False, "retry_worthy": True, "reasoning": f"Parse error: {exc}"}] * len(rechecks)

    async def generate_summary(
        self,
        trades: list,
        balance: float,
        performance_stats: dict,
        period: str = "Tages",
    ) -> dict:
        """Generiert eine Claude-basierte Zusammenfassung."""
        trades_text = "Keine Trades im Zeitraum."
        if trades:
            lines = []
            for t in trades:
                pl = t.profit_loss or 0
                exit_p = f"{t.exit_price:.4f}" if t.exit_price else "offen"
                lines.append(
                    f"- {t.asset} {t.direction.value}: Entry={t.entry_price:.4f}, "
                    f"Exit={exit_p}, P/L={pl:+.2f} EUR, Status={t.status.value}"
                )
            trades_text = "\n".join(lines)

        stats_parts = [f"Win-Rate: {performance_stats.get('win_rate', 0):.0f}%"]
        streak = performance_stats.get("current_loss_streak", 0)
        if streak > 0:
            stats_parts.append(f"Aktuelle Verlustserie: {streak}")
        for asset, data in performance_stats.get("by_asset", {}).items():
            stats_parts.append(
                f"{asset}: {data['total']} Trades, WR={data['win_rate']:.0f}%, "
                f"P/L={data['total_pl']:+.2f}"
            )
        stats_text = "\n".join(stats_parts)

        prompt = _SUMMARY_PROMPT.format(
            period=period,
            date_range=datetime.now().strftime("%Y-%m-%d"),
            balance=balance,
            trades_summary=trades_text,
            stats_summary=stats_text,
        )

        logger.info("Requesting %s-Zusammenfassung...", period)
        raw_text = ""
        async with self._client.messages.stream(
            model=config.CLAUDE_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for _ in stream:
                pass
            final = await stream.get_final_message()

        for block in final.content:
            if block.type == "text":
                raw_text = block.text.strip()
                break

        try:
            text = raw_text
            if text.startswith("```"):
                text = text.split("```", 2)[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.rsplit("```", 1)[0]
            return json.loads(text.strip())
        except Exception:
            return {"summary": raw_text[:500], "highlights": [], "issues": [], "recommendations": []}


_SUMMARY_PROMPT = """\
Du bist ein Trading-Analyst. Erstelle eine {period}-Zusammenfassung.

ZEITRAUM: {date_range}
KONTOSTAND: {balance:.2f} EUR

TRADES:
{trades_summary}

STATISTIKEN:
{stats_summary}

Antworte AUSSCHLIESSLICH als valides JSON:
{{
    "summary": "Zusammenfassung (2-3 Saetze)",
    "highlights": ["Positiv 1", "Positiv 2"],
    "issues": ["Problem 1", "Problem 2"],
    "recommendations": ["Empfehlung 1", "Empfehlung 2"],
    "overall_rating": "good|neutral|poor"
}}
"""


_RECHECK_PROMPT = """\
Du bist ein Trading-Analyst. Du hattest zuvor vielversprechende Setups identifiziert,
die noch nicht reif waren. Pruefe ob sie JETZT reif sind.

{setups_text}

AKTUELLE MARKTDATEN:
{market_data_text}

Antworte AUSSCHLIESSLICH als valides JSON-Array (ein Objekt pro Setup, gleiche Reihenfolge):
[
    {{
        "asset": "GOLD|SILVER|OIL_CRUDE|NATURALGAS",
        "is_ready": true/false,
        "confidence": 1-10,
        "entry_price": 0.0,
        "stop_loss": 0.0,
        "take_profit": 0.0,
        "risk_reward_ratio": 0.0,
        "reasoning": "Warum jetzt reif / noch nicht reif",
        "retry_worthy": true/false,
        "retry_in_minutes": 30-120
    }}
]

Regeln:
- is_ready = true NUR wenn Confidence >= 8 und klares Setup
- retry_worthy = true wenn Setup noch intakt aber Trigger nicht erreicht
- retry_worthy = false wenn Setup nicht mehr gueltig (Trend gebrochen etc.)
- Bei mehreren Setups: jedes einzeln bewerten
"""


_REVIEW_PROMPT = """\
Du bist ein Trading-Analyst. Analysiere diesen abgeschlossenen Trade und ziehe Lehren daraus.

TRADE-DETAILS:
- Asset: {asset}
- Richtung: {direction}
- Einstieg: {entry_price:.4f} @ {entry_time}
- Ausstieg: {exit_price:.4f} @ {exit_time}
- Stop-Loss: {stop_loss:.4f}
- Take-Profit: {take_profit:.4f}
- P/L: {profit_loss:+.2f} EUR ({profit_loss_pct:+.2f}%)
- Status: {status}
- Confidence bei Einstieg: {confidence}
- Begruendung: {reasoning}

KURSVERLAUF NACH EINSTIEG (letzte Kerzen):
{price_history_after}

Antworte AUSSCHLIESSLICH als valides JSON:
{{
    "entry_quality": "good|bad|neutral",
    "entry_quality_explanation": "Warum war der Einstieg gut/schlecht?",
    "sl_quality": "too_tight|too_wide|good",
    "sl_quality_explanation": "War der SL angemessen?",
    "market_condition": "trend|range|breakout|reversal|volatile",
    "what_happened_after": "Zusammenfassung des Kursverlaufs nach dem Trade",
    "lesson_learned": "Wichtigste Erkenntnis fuer zukuenftige Trades (1-2 Saetze)",
    "would_trade_again": true,
    "improvement_suggestions": ["Vorschlag 1", "Vorschlag 2"]
}}
"""


_ESCALATION_PROMPT = """\
Du bist ein Rohstoff-Trading-Analyst. Eine offene Position benötigt deine Einschätzung.

POSITION:
- Asset: {asset}
- Richtung: {direction}
- Einstieg: {entry_price:.4f} @ {entry_time}
- Aktueller Kurs: {current_price:.4f}
- Stop-Loss: {stop_loss:.4f}
- Take-Profit: {take_profit:.4f}
- Aktueller P/L: {profit_loss:+.2f} EUR ({profit_loss_pct:+.2f}%)
- Laufzeit: {duration}

ESKALATIONSGRUND: {escalation_reason}

Antworte AUSSCHLIESSLICH als valides JSON:
{{
    "action": "HOLD" | "CLOSE" | "ADJUST_SL" | "ADJUST_TP",
    "reasoning": "Begründung",
    "new_stop_loss": null | Preis,
    "new_take_profit": null | Preis,
    "urgency": "low" | "medium" | "high"
}}
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_user_message(
    market_data: dict[str, MarketData],
    market_context: Optional[MarketContext] = None,
    indicators: dict[str, dict] | None = None,
) -> str:
    lines = ["## Aktuelle Marktdaten\n"]
    for key, data in market_data.items():
        price = data.current_price
        lines.append(f"### {data.name} ({key})")
        lines.append(f"- Aktueller Kurs: Bid={price.bid:.4f} / Ask={price.ask:.4f}")
        lines.append(f"- Tageshoch/-tief: {price.high:.4f} / {price.low:.4f}")
        lines.append(f"- Tagesveränderung: {price.change_pct:+.2f}%")
        if data.price_history:
            lines.append(f"- Kursverlauf (letzte {len(data.price_history)} Kerzen):")
            for bar in data.price_history[-7:]:  # last 7 bars
                lines.append(
                    f"  {bar.timestamp[:10]}: O={bar.open:.4f} H={bar.high:.4f} "
                    f"L={bar.low:.4f} C={bar.close:.4f}"
                )

        # ── Technische Indikatoren ────────────────────────────────────────────
        if indicators and key in indicators:
            ind = indicators[key]
            lines.append("- Technische Indikatoren:")

            rsi = ind.get("rsi")
            if rsi is not None:
                rsi_label = "überverkauft" if rsi < 30 else ("überkauft" if rsi > 70 else "neutral")
                lines.append(f"  - RSI(14): {rsi:.1f} ({rsi_label})")

            atr = ind.get("atr")
            avg_atr = ind.get("avg_atr")
            atr_ratio = ind.get("atr_ratio")
            if atr is not None:
                if avg_atr and atr_ratio:
                    flag = " → ERHÖHT!" if atr_ratio > 1.5 else ""
                    lines.append(
                        f"  - ATR(14): {atr:.4f} (Durchschnitt: {avg_atr:.4f} → {atr_ratio:.2f}x{flag})"
                    )
                else:
                    lines.append(f"  - ATR(14): {atr:.4f}")

            ema_20 = ind.get("ema_20")
            ema_50 = ind.get("ema_50")
            if ema_20 is not None or ema_50 is not None:
                ema_parts = []
                if ema_20 is not None:
                    ema_parts.append(f"EMA(20): {ema_20:.4f}")
                if ema_50 is not None:
                    ema_parts.append(f"EMA(50): {ema_50:.4f}")
                ema_str = " | ".join(ema_parts)
                if ema_20 and ema_50:
                    if price.bid < ema_20 and price.bid < ema_50:
                        trend_hint = " (Kurs unter beiden EMAs = bärisch)"
                    elif price.bid > ema_20 and price.bid > ema_50:
                        trend_hint = " (Kurs über beiden EMAs = bullisch)"
                    else:
                        trend_hint = ""
                    lines.append(f"  - {ema_str}{trend_hint}")
                else:
                    lines.append(f"  - {ema_str}")

            macd = ind.get("macd")
            if macd:
                macd_dir = "bullisch" if macd["histogram"] > 0 else "bärisch"
                lines.append(
                    f"  - MACD: {macd['macd']:.4f} | Signal: {macd['signal']:.4f} "
                    f"| Histogramm: {macd['histogram']:.4f} ({macd_dir})"
                )

            bb = ind.get("bollinger")
            if bb:
                mid = price.bid
                if mid <= bb["lower"] * 1.001:
                    bb_pos = "Kurs am unteren Band"
                elif mid >= bb["upper"] * 0.999:
                    bb_pos = "Kurs am oberen Band"
                else:
                    bb_pos = f"Kurs bei {mid:.4f}"
                lines.append(
                    f"  - Bollinger: {bb_pos} (unteres Band: {bb['lower']:.4f}, "
                    f"oberes Band: {bb['upper']:.4f}, Breite: {bb['width']:.4f})"
                )

            sr = ind.get("support_resistance")
            if sr:
                support_str = ", ".join(f"{s:.4f}" for s in sr.get("support", []))
                resistance_str = ", ".join(f"{r:.4f}" for r in sr.get("resistance", []))
                if support_str or resistance_str:
                    lines.append(
                        f"  - Support: {support_str or 'n/a'} | Resistance: {resistance_str or 'n/a'}"
                    )

        lines.append("")
    if market_context and not market_context.is_empty():
        lines.append("\n" + market_context.to_prompt_text())

    return "\n".join(lines)


def _format_open_positions(positions: list[PositionInfo]) -> str:
    if not positions:
        return "Keine offenen Positionen"
    parts = []
    for pos in positions:
        parts.append(
            f"{pos.epic} {pos.direction.value} {pos.size} @ {pos.entry_price:.4f} "
            f"(P/L: {pos.profit_loss:+.2f})"
        )
    return ", ".join(parts)


def _parse_analysis(raw_text: str) -> AnalysisResult:
    """Parse Claude's JSON response into an AnalysisResult."""
    # Strip potential markdown code fences
    text = raw_text
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0]

    try:
        data = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse Claude response as JSON: %s\nRaw: %.500s", exc, raw_text)
        return _fallback_wait_result("JSON parse error")

    try:
        opp_raw = data.get("best_opportunity", {})
        direction = Direction(opp_raw.get("direction", "NONE"))
        opportunity = BestOpportunity(
            asset=opp_raw.get("asset", "GOLD"),
            direction=direction,
            confidence=int(opp_raw.get("confidence", 1)),
            reasoning=opp_raw.get("reasoning", ""),
            entry_price=float(opp_raw.get("entry_price", 0)),
            stop_loss=float(opp_raw.get("stop_loss", 0)),
            take_profit=float(opp_raw.get("take_profit", 0)),
            risk_reward_ratio=float(opp_raw.get("risk_reward_ratio", 0)),
            key_events=opp_raw.get("key_events", []),
            risks=opp_raw.get("risks", []),
        )
        other_assets = [
            AssetOutlook(
                asset=a.get("asset", ""),
                outlook=a.get("outlook", "neutral"),
                confidence=int(a.get("confidence", 5)),
                note=a.get("note", ""),
            )
            for a in data.get("other_assets", [])
        ]
        # Parse recheck info
        recheck = None
        recheck_raw = data.get("recheck")
        if recheck_raw and recheck_raw.get("worthy"):
            from .models import RecheckInfo
            try:
                recheck = RecheckInfo(
                    worthy=True,
                    asset=recheck_raw.get("asset", ""),
                    direction=Direction(recheck_raw.get("direction", "NONE")),
                    trigger_condition=recheck_raw.get("trigger_condition", ""),
                    recheck_in_minutes=int(recheck_raw.get("recheck_in_minutes", 60)),
                    current_confidence=int(recheck_raw.get("current_confidence", 0)),
                    expected_confidence_if_trigger=int(recheck_raw.get("expected_confidence_if_trigger", 0)),
                )
            except Exception as exc:
                logger.warning("Failed to parse recheck info: %s", exc)

        return AnalysisResult(
            date=data.get("date", datetime.now().strftime("%Y-%m-%d")),
            market_summary=data.get("market_summary", ""),
            best_opportunity=opportunity,
            other_assets=other_assets,
            recommendation=Recommendation(data.get("recommendation", "WAIT")),
            wait_reason=data.get("wait_reason"),
            recheck=recheck,
        )
    except (KeyError, ValueError, TypeError) as exc:
        logger.error("Failed to build AnalysisResult: %s", exc)
        return _fallback_wait_result(f"Data error: {exc}")


def _build_learning_context(
    performance_stats: dict | None,
    recent_lessons: list[dict] | None,
) -> str:
    """Baut den Lern-Kontext fuer den Analyse-Prompt zusammen."""
    if not performance_stats and not recent_lessons:
        return ""

    lines = ["\n\n## Bisherige Trade-Erfahrungen\n"]

    if performance_stats:
        s = performance_stats
        if s.get("total", 0) > 0:
            lines.append(f"### Performance (Gesamt: {s['total']} Trades)")
            lines.append(
                f"- Win-Rate: {s['win_rate']:.0f}% "
                f"({s['wins']} Gewinne, {s['losses']} Verluste)"
            )
            if s.get("avg_win"):
                lines.append(
                    f"- Avg Gewinn: {s['avg_win']:+.2f} EUR | "
                    f"Avg Verlust: {s['avg_loss']:+.2f} EUR"
                )
            if s.get("current_loss_streak", 0) > 0:
                lines.append(
                    f"- ACHTUNG: Aktuelle Verlustserie: {s['current_loss_streak']} Trades in Folge"
                )

            if s.get("by_asset"):
                lines.append("\n### Performance per Asset")
                for asset, data in s["by_asset"].items():
                    lines.append(
                        f"- {asset}: {data['total']} Trades, "
                        f"Win-Rate {data['win_rate']:.0f}%, "
                        f"P/L: {data['total_pl']:+.2f} EUR"
                    )

            if s.get("by_direction"):
                lines.append("\n### Performance per Richtung")
                for direction, data in s["by_direction"].items():
                    lines.append(
                        f"- {direction}: {data['total']} Trades, "
                        f"Win-Rate {data['win_rate']:.0f}%, "
                        f"P/L: {data['total_pl']:+.2f} EUR"
                    )

    if recent_lessons:
        lines.append("\n### Gelernte Lektionen (neueste zuerst)")
        for i, lesson in enumerate(recent_lessons[:10], 1):
            asset = lesson.get("asset", "?")
            direction = lesson.get("direction", "?")
            status = lesson.get("status", "?")
            pl = lesson.get("profit_loss") or 0
            learned = lesson.get("lesson_learned", "")
            if learned:
                lines.append(
                    f"{i}. [{asset} {direction}, {status}, P/L={pl:+.2f}]: {learned}"
                )

    lines.append(
        "\nBeziehe diese Erfahrungen in deine Analyse ein. "
        "Wiederhole keine Fehler aus vergangenen Trades."
    )
    return "\n".join(lines)


def _fallback_wait_result(reason: str) -> AnalysisResult:
    return AnalysisResult(
        date=datetime.now().strftime("%Y-%m-%d"),
        market_summary="Analysis failed – defaulting to WAIT",
        best_opportunity=BestOpportunity(
            asset="GOLD",
            direction=Direction.NONE,
            confidence=1,
            reasoning=reason,
            entry_price=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            risk_reward_ratio=0.0,
        ),
        recommendation=Recommendation.WAIT,
        wait_reason=reason,
    )
