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
- Wenn mehrere Assets gleich gut aussehen, wähle das mit dem besten Risk/Reward
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
    "wait_reason": "Falls WAIT: Warum kein Trade heute?"
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
        "INTRADAY-TRADING (Haltedauer < 24h, kein Overnight) – "
        "Fokus auf kurzfristige Impulse anhand von Stundenkerzen. "
        "Enge Stop-Losses, kleinere Take-Profits, schnelle Setups bevorzugen. "
        "Keine Positionen über Nacht halten – lieber WAIT als zu spät einsteigen."
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
    ) -> AnalysisResult:
        """
        Send market data to Claude and receive a structured analysis.

        Uses adaptive thinking for deeper reasoning on complex market conditions.
        Streams the response to avoid HTTP timeouts.
        """
        user_message = _build_user_message(market_data, market_context)
        system_prompt = _SYSTEM_PROMPT.format(
            current_date=datetime.now().strftime("%Y-%m-%d"),
            account_balance=f"{account_balance:.2f}",
            open_positions=_format_open_positions(open_positions),
            max_risk_pct=config.MAX_RISK_PER_TRADE_PCT,
            trading_style=_STYLE_HINTS.get(config.TRADING_STYLE, _STYLE_HINTS["swing"]),
        )

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
        return AnalysisResult(
            date=data.get("date", datetime.now().strftime("%Y-%m-%d")),
            market_summary=data.get("market_summary", ""),
            best_opportunity=opportunity,
            other_assets=other_assets,
            recommendation=Recommendation(data.get("recommendation", "WAIT")),
            wait_reason=data.get("wait_reason"),
        )
    except (KeyError, ValueError, TypeError) as exc:
        logger.error("Failed to build AnalysisResult: %s", exc)
        return _fallback_wait_result(f"Data error: {exc}")


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
