"""Push notifications via ntfy.sh."""
import asyncio
import logging
import time
from typing import Optional

import httpx

from . import config
from .models import BestOpportunity, Trade, TradeStatus

logger = logging.getLogger(__name__)


class Notifier:
    """Sends push notifications via ntfy.sh (free, self-hosted optional)."""

    def __init__(self) -> None:
        self._enabled = bool(config.NTFY_TOPIC)
        if not self._enabled:
            logger.info("Notifications disabled (NTFY_TOPIC not set)")

    async def notify_trade_opened(self, trade: Trade) -> None:
        emoji = "🟢"
        title = f"{emoji} Trade eröffnet: {trade.asset} {trade.direction.value}"
        body = (
            f"@ {trade.entry_price:.4f} | "
            f"SL: {trade.stop_loss:.4f} | "
            f"TP: {trade.take_profit:.4f} | "
            f"Size: {trade.position_size:.2f} | "
            f"Konfidenz: {trade.confidence}/10"
        )
        await self._send(title, body, priority="high")

    async def notify_trade_closed(self, trade: Trade, exit_price: float, profit_loss: float, profit_loss_pct: float) -> None:
        status = trade.status
        if status == TradeStatus.TAKE_PROFIT:
            emoji = "✅"
        elif status == TradeStatus.STOPPED_OUT:
            emoji = "🛑"
        else:
            emoji = "🔴"

        sign = "+" if profit_loss >= 0 else ""
        title = f"{emoji} Trade geschlossen: {trade.asset} {sign}{profit_loss:.2f} EUR"
        body = (
            f"Exit: {exit_price:.4f} | "
            f"P/L: {sign}{profit_loss:.2f} EUR ({sign}{profit_loss_pct:.2f}%) | "
            f"Status: {status.value}"
        )
        await self._send(title, body, priority="default")

    async def notify_daily_summary(self, recommendation: str, reason: Optional[str] = None) -> None:
        if recommendation == "WAIT":
            title = "Tagesanalyse: Kein Trade heute"
            body = reason or "Kein Setup mit ausreichend hoher Konfidenz gefunden."
        else:
            title = "Tagesanalyse: Trade-Signal aktiv"
            body = "Ein Trade-Setup wurde identifiziert – Ausführung läuft."
        await self._send(title, body, priority="low")

    async def notify_trailing_stop(self, trade: Trade, old_sl: float, new_sl: float) -> None:
        direction = "nachgezogen" if trade.direction.value == "BUY" else "nachgezogen"
        title = f"Trailing Stop {trade.asset}: SL {direction}"
        body = f"SL: {old_sl:.4f} → {new_sl:.4f} | Kurs: {trade.asset}"
        await self._send(title, body, priority="default")

    async def notify_break_even(self, trade: Trade) -> None:
        title = f"Break-Even gesichert: {trade.asset}"
        body = f"Stop-Loss auf Einstieg {trade.entry_price:.4f} gesetzt – kein Verlust mehr möglich"
        await self._send(title, body, priority="default")

    async def notify_monitor_alert(self, trade: Trade, reason: str, urgency: str = "low") -> None:
        priority_map = {"low": "default", "medium": "high", "high": "urgent"}
        title = f"Monitor-Alert: {trade.asset} {trade.direction.value}"
        body = reason[:500]
        await self._send(title, body, priority=priority_map.get(urgency, "default"))

    async def notify_daily_summary_report(
        self,
        balance: float,
        trades_today: int,
        profit_loss_total: float,
    ) -> None:
        sign = "+" if profit_loss_total >= 0 else ""
        title = f"Tagesuebersicht: {trades_today} Trade(s), {sign}{profit_loss_total:.2f} EUR"
        body = f"Kontostand: {balance:.2f} EUR"
        await self._send(title, body, priority="default")

    async def notify_error(self, error: str) -> None:
        title = "Trading-Bot Fehler"
        body = error[:500]  # ntfy has message limits
        await self._send(title, body, priority="urgent")

    async def request_trade_confirmation(self, opportunity: BestOpportunity) -> None:
        """Send a confirmation request notification with trade details and action buttons."""
        if not config.NTFY_CONFIRM_TOPIC:
            logger.warning("NTFY_CONFIRM_TOPIC not set – cannot request confirmation")
            return

        title = (
            f"Trade-Bestätigung: {opportunity.asset} {opportunity.direction.value} "
            f"(Confidence {opportunity.confidence}/10)"
        )
        lines = [
            f"Begründung: {opportunity.reasoning}",
            "",
            f"Entry: {opportunity.entry_price:.4f} | SL: {opportunity.stop_loss:.4f} | "
            f"TP: {opportunity.take_profit:.4f} | RR: {opportunity.risk_reward_ratio:.2f}",
        ]
        if opportunity.key_events:
            lines.append("Events: " + ", ".join(opportunity.key_events))
        if opportunity.risks:
            lines.append("Risiken: " + ", ".join(opportunity.risks))
        body = "\n".join(lines)

        confirm_url = f"{config.NTFY_SERVER.rstrip('/')}/{config.NTFY_CONFIRM_TOPIC}"
        actions = (
            f"http, Approve, {confirm_url}, method=POST, body=APPROVE; "
            f"http, Reject, {confirm_url}, method=POST, body=REJECT"
        )
        await self._send(title, body, priority="high", extra_headers={"Actions": actions})

    async def wait_for_confirmation(self, since: float) -> bool:
        """
        Poll the confirm topic for an APPROVE/REJECT response.
        Returns True if approved, False if rejected or timed out.
        """
        if not config.NTFY_CONFIRM_TOPIC:
            return False

        timeout_seconds = config.TRADE_CONFIRMATION_TIMEOUT_MINUTES * 60
        deadline = time.time() + timeout_seconds
        poll_url = (
            f"{config.NTFY_SERVER.rstrip('/')}/{config.NTFY_CONFIRM_TOPIC}/json"
            f"?poll=1&since={int(since)}"
        )
        logger.info(
            "Waiting for trade confirmation (timeout: %d min)...",
            config.TRADE_CONFIRMATION_TIMEOUT_MINUTES,
        )
        async with httpx.AsyncClient(timeout=10.0) as client:
            while time.time() < deadline:
                try:
                    response = await client.get(poll_url)
                    for line in response.text.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        import json as _json
                        try:
                            msg = _json.loads(line)
                        except _json.JSONDecodeError:
                            continue
                        if msg.get("event") != "message":
                            continue
                        body = msg.get("message", "").strip().upper()
                        if body == "APPROVE":
                            logger.info("Trade confirmed by user")
                            return True
                        if body == "REJECT":
                            logger.info("Trade rejected by user")
                            return False
                except Exception as exc:
                    logger.warning("Confirmation poll error: %s", exc)
                await asyncio.sleep(15)

        logger.info("Confirmation timeout – skipping trade")
        return False

    async def _send(self, title: str, body: str, priority: str = "default", extra_headers: Optional[dict] = None) -> None:
        if not self._enabled:
            logger.debug("Notification (skipped): %s – %s", title, body)
            return
        url = f"{config.NTFY_SERVER.rstrip('/')}/{config.NTFY_TOPIC}"
        try:
            headers = {
                "Title": _ascii_safe(title),
                "Priority": priority,
                "Tags": "chart_with_upwards_trend",
            }
            if extra_headers:
                headers.update(extra_headers)
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    url,
                    content=body.encode("utf-8"),
                    headers=headers,
                )
            logger.debug("Notification sent: %s", title)
        except Exception as exc:
            logger.warning("Failed to send notification: %s", exc)


def _ascii_safe(text: str) -> str:
    """ntfy title header must be ASCII-compatible."""
    return text.encode("ascii", errors="replace").decode("ascii")
