"""Capital.com REST API wrapper."""
import asyncio
import logging
import time
from typing import Any, Optional

import httpx

from . import config
from .models import AccountInfo, MarketData, MarketPrice, PositionInfo, PriceBar

logger = logging.getLogger(__name__)

_RETRY_ATTEMPTS = 3
_RETRY_BASE_DELAY = 1.0  # seconds


class CapitalComError(Exception):
    """Raised when the Capital.com API returns an error."""


class CapitalComBroker:
    """Async wrapper for the Capital.com REST API."""

    def __init__(self) -> None:
        self._cst: Optional[str] = None
        self._security_token: Optional[str] = None
        self._session_created_at: float = 0.0
        self._client: Optional[httpx.AsyncClient] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def __aenter__(self) -> "CapitalComBroker":
        self._client = httpx.AsyncClient(
            base_url=config.CAPITAL_BASE_URL,
            timeout=30.0,
            headers={"X-CAP-API-KEY": config.CAPITAL_API_KEY, "Content-Type": "application/json"},
        )
        await self.create_session()
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._client:
            await self._client.aclose()

    # ── Session management ─────────────────────────────────────────────────────

    async def create_session(self) -> None:
        """POST /api/v1/session – obtain CST and X-SECURITY-TOKEN."""
        payload = {
            "identifier": config.CAPITAL_EMAIL,
            "password": config.CAPITAL_PASSWORD,
        }
        response = await self._request_raw("POST", "/api/v1/session", json=payload, auth=False)
        self._cst = response.headers.get("CST")
        self._security_token = response.headers.get("X-SECURITY-TOKEN")
        self._session_created_at = time.monotonic()
        if not self._cst or not self._security_token:
            raise CapitalComError("Session creation failed: missing CST or X-SECURITY-TOKEN headers")
        logger.info("Capital.com session created (demo=%s)", config.CAPITAL_DEMO)

    async def _ensure_session(self) -> None:
        """Renew session if it is older than 9 minutes."""
        age = time.monotonic() - self._session_created_at
        if age > (config.CAPITAL_SESSION_TTL_SECONDS - 60):
            logger.info("Renewing Capital.com session (age=%.0fs)", age)
            await self.create_session()

    # ── Account ────────────────────────────────────────────────────────────────

    async def get_account_balance(self) -> AccountInfo:
        """GET /api/v1/accounts – return balance information."""
        await self._ensure_session()
        data = await self._request("GET", "/api/v1/accounts")
        logger.debug("Raw account response: %s", data)
        accounts = data.get("accounts", [])
        if not accounts:
            raise CapitalComError("No accounts found")
        acc = accounts[0]
        balance_data = acc.get("balance", {})

        balance = float(balance_data.get("balance", 0.0) or 0.0)
        available = float(balance_data.get("available", 0.0) or 0.0)

        # Equity kann unter verschiedenen Keys liegen – mehrere Fallbacks prüfen
        equity = float(balance_data.get("equity", 0.0) or 0.0)
        if equity == 0.0:
            # Fallback 1: direkt im Account-Objekt
            equity = float(acc.get("equity", 0.0) or 0.0)
        if equity == 0.0:
            # Fallback 2: profitLoss-basiert (balance + unrealisierter P/L)
            profit_loss = float(balance_data.get("profitLoss", 0.0) or 0.0)
            if profit_loss != 0.0:
                equity = balance + profit_loss
        if equity == 0.0:
            # Fallback 3: balance als Equity (keine offenen Positionen)
            equity = balance

        return AccountInfo(
            balance=balance,
            equity=equity,
            available=available,
            currency=acc.get("currency", "EUR"),
        )

    # ── Market data ────────────────────────────────────────────────────────────

    async def get_market_prices(self, epic: str) -> MarketData:
        """GET /api/v1/markets/{epic} – current bid/ask and instrument details."""
        await self._ensure_session()
        data = await self._request("GET", f"/api/v1/markets/{epic}")
        instrument = data.get("instrument", {})
        snapshot = data.get("snapshot", {})
        return MarketData(
            epic=epic,
            name=instrument.get("name", epic),
            current_price=MarketPrice(
                epic=epic,
                bid=float(snapshot.get("bid", 0)),
                ask=float(snapshot.get("offer", 0)),
                high=float(snapshot.get("high", 0)),
                low=float(snapshot.get("low", 0)),
                change_pct=float(snapshot.get("percentageChange", 0)),
            ),
        )

    async def get_price_history(
        self,
        epic: str,
        resolution: str = config.PRICE_HISTORY_RESOLUTION,
        max_bars: int = config.PRICE_HISTORY_MAX_BARS,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> list[PriceBar]:
        """GET /api/v1/prices/{epic} – OHLC candlestick history.

        Args:
            from_date: ISO-8601 start (e.g. "2026-01-01T00:00:00")
            to_date:   ISO-8601 end   (e.g. "2026-03-28T23:59:00")
        """
        await self._ensure_session()
        params: dict[str, Any] = {"resolution": resolution, "max": max_bars}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        data = await self._request(
            "GET",
            f"/api/v1/prices/{epic}",
            params=params,
        )
        bars: list[PriceBar] = []
        for price in data.get("prices", []):
            mid = price.get("closePrice", {})
            open_mid = price.get("openPrice", {})
            high_mid = price.get("highPrice", {})
            low_mid = price.get("lowPrice", {})
            bars.append(
                PriceBar(
                    timestamp=price.get("snapshotTimeUTC", ""),
                    open=_mid_price(open_mid),
                    high=_mid_price(high_mid),
                    low=_mid_price(low_mid),
                    close=_mid_price(mid),
                )
            )
        return bars

    # ── Positions ──────────────────────────────────────────────────────────────

    async def get_open_positions(self) -> list[PositionInfo]:
        """GET /api/v1/positions – list all open positions."""
        await self._ensure_session()
        data = await self._request("GET", "/api/v1/positions")
        positions: list[PositionInfo] = []
        for pos in data.get("positions", []):
            position = pos.get("position", {})
            market = pos.get("market", {})
            direction_raw = position.get("direction", "BUY")
            from .models import Direction
            try:
                direction = Direction(direction_raw)
            except ValueError:
                direction = Direction.BUY
            positions.append(
                PositionInfo(
                    deal_id=position.get("dealId", ""),
                    epic=market.get("epic", ""),
                    direction=direction,
                    size=float(position.get("size", 0)),
                    entry_price=float(position.get("level", 0)),
                    current_price=float(market.get("bid", 0)),
                    stop_loss=position.get("stopLevel"),
                    take_profit=position.get("limitLevel"),
                    profit_loss=float(pos.get("position", {}).get("upl", 0)),
                )
            )
        return positions

    async def open_position(
        self,
        epic: str,
        direction: str,
        size: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> dict[str, Any]:
        """POST /api/v1/positions – open a new position."""
        await self._ensure_session()
        payload: dict[str, Any] = {
            "epic": epic,
            "direction": direction,
            "size": str(size),
            "guaranteedStop": False,
        }
        if stop_loss is not None:
            payload["stopLevel"] = stop_loss
        if take_profit is not None:
            payload["profitLevel"] = take_profit
        data = await self._request("POST", "/api/v1/positions", json=payload)
        logger.info(
            "Position opened: epic=%s direction=%s size=%s deal_reference=%s",
            epic,
            direction,
            size,
            data.get("dealReference", ""),
        )
        return data

    async def update_position(self, deal_id: str, stop_loss: float, take_profit: Optional[float] = None) -> dict[str, Any]:
        """PUT /api/v1/positions/{dealId} – update stop-loss (and optionally take-profit)."""
        await self._ensure_session()
        payload: dict[str, Any] = {"stopLevel": stop_loss}
        if take_profit is not None:
            payload["profitLevel"] = take_profit
        data = await self._request("PUT", f"/api/v1/positions/{deal_id}", json=payload)
        logger.info("Position %s updated: SL=%.4f", deal_id, stop_loss)
        return data

    async def confirm_trade(self, deal_reference: str) -> dict[str, Any]:
        """GET /api/v1/confirms/{dealReference} – resolve dealReference → dealId."""
        await self._ensure_session()
        return await self._request("GET", f"/api/v1/confirms/{deal_reference}")

    async def close_position(self, deal_id: str) -> dict[str, Any]:
        """DELETE /api/v1/positions/{dealId} – close an open position."""
        await self._ensure_session()
        data = await self._request("DELETE", f"/api/v1/positions/{deal_id}")
        logger.info("Position closed: deal_id=%s", deal_id)
        return data

    # ── HTTP helpers ───────────────────────────────────────────────────────────

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> dict[str, Any]:
        response = await self._request_raw(method, path, json=json, params=params)
        return response.json()

    async def _request_raw(
        self,
        method: str,
        path: str,
        *,
        json: Optional[dict] = None,
        params: Optional[dict] = None,
        auth: bool = True,
    ) -> httpx.Response:
        if self._client is None:
            raise RuntimeError("Broker not initialised – use 'async with CapitalComBroker()'")

        headers: dict[str, str] = {}
        if auth and self._cst and self._security_token:
            headers["CST"] = self._cst
            headers["X-SECURITY-TOKEN"] = self._security_token

        last_exc: Optional[Exception] = None
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                response = await self._client.request(
                    method, path, json=json, params=params, headers=headers
                )
                if response.status_code == 401:
                    logger.warning("401 Unauthorised – renewing session")
                    await self.create_session()
                    headers["CST"] = self._cst  # type: ignore[assignment]
                    headers["X-SECURITY-TOKEN"] = self._security_token  # type: ignore[assignment]
                    continue
                if response.status_code == 429:
                    delay = _RETRY_BASE_DELAY * (2**attempt)
                    logger.warning("Rate limited – waiting %.1fs", delay)
                    await asyncio.sleep(delay)
                    continue
                if response.status_code == 404:
                    # Data not found – no point retrying
                    raise CapitalComError(f"404 Not Found: {response.text[:100]}")
                if response.status_code == 400:
                    body = response.text[:200]
                    if "daterange" in body or "max.date" in body:
                        # Date range too large or out of bounds – no point retrying
                        raise CapitalComError(f"400 Date range error: {body}")
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                body = exc.response.text[:200] if exc.response else ""
                logger.error(
                    "HTTP %s on %s %s: %s",
                    exc.response.status_code if exc.response else "?",
                    method,
                    path,
                    body,
                )
                delay = _RETRY_BASE_DELAY * (2**attempt)
                await asyncio.sleep(delay)
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                last_exc = exc
                logger.warning("Network error on %s %s: %s", method, path, exc)
                delay = _RETRY_BASE_DELAY * (2**attempt)
                await asyncio.sleep(delay)

        raise CapitalComError(f"Request failed after {_RETRY_ATTEMPTS} attempts: {last_exc}") from last_exc


# ── Global singleton ──────────────────────────────────────────────────────────
_shared_broker: Optional[CapitalComBroker] = None
_shared_lock: asyncio.Lock = asyncio.Lock()


async def get_shared_broker() -> CapitalComBroker:
    """Return a long-lived shared broker instance (created on first call).

    All API endpoints and the unified_tick share this single session,
    avoiding repeated POST /session calls that trigger rate limits.
    """
    global _shared_broker
    async with _shared_lock:
        if _shared_broker is None:
            _shared_broker = CapitalComBroker()
            await _shared_broker.__aenter__()
            logger.info("Shared broker session created")
        else:
            await _shared_broker._ensure_session()
    return _shared_broker


async def shutdown_shared_broker() -> None:
    """Gracefully close the shared broker (call on app shutdown)."""
    global _shared_broker
    async with _shared_lock:
        if _shared_broker is not None:
            await _shared_broker.__aexit__(None, None, None)
            _shared_broker = None
            logger.info("Shared broker session closed")


def _mid_price(price_dict: dict) -> float:
    """Return mid price from a Capital.com price dict (bid+ask average)."""
    bid = float(price_dict.get("bid", 0) or 0)
    ask = float(price_dict.get("ask", 0) or 0)
    if bid and ask:
        return (bid + ask) / 2
    return bid or ask or float(price_dict.get("mid", 0) or 0)
