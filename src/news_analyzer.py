"""News-Analyse-Modul – ruft aktuelle Wirtschaftsnachrichten ab und bewertet Sentiment."""
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import httpx

from . import config

logger = logging.getLogger(__name__)

# ── Sentiment-Keywords ─────────────────────────────────────────────────────────
_BULLISH = {
    "rally", "surge", "rise", "gain", "gains", "up", "high", "strong", "buy",
    "positive", "growth", "boom", "jump", "soar", "rebound", "recover", "demand",
    "boost", "optimism", "bullish", "breakout", "outperform",
}
_BEARISH = {
    "fall", "drop", "crash", "decline", "down", "low", "weak", "sell", "negative",
    "loss", "tariff", "sanction", "recession", "slump", "plunge", "bearish",
    "cut", "risk", "fear", "concern", "uncertainty", "slowdown", "contraction",
    "war", "conflict", "tension", "invasion", "attack", "ban", "shortage",
}

# ── Asset-Keywords für Relevanz-Filterung ─────────────────────────────────────
_ASSET_KEYWORDS: dict[str, set[str]] = {
    "GOLD":       {"gold", "xau", "precious metal", "safe haven", "bullion"},
    "SILVER":     {"silver", "xag", "precious metal"},
    "OIL_CRUDE":  {"oil", "crude", "wti", "brent", "opec", "petroleum", "barrel"},
    "NATURALGAS": {"gas", "natural gas", "lng", "pipeline", "ttf"},
}

# ── Makro-Event-Muster ────────────────────────────────────────────────────────
_EVENT_PATTERNS: list[dict] = [
    {
        "keywords": {"fed", "federal reserve", "fomc", "powell", "rate hike", "rate cut", "interest rate"},
        "type": "geldpolitik",
        "description_prefix": "Fed/Geldpolitik",
    },
    {
        "keywords": {"ecb", "lagarde", "european central bank", "boe", "bank of england"},
        "type": "geldpolitik",
        "description_prefix": "Zentralbank-Entscheidung",
    },
    {
        "keywords": {"opec", "production cut", "output cut", "supply cut", "quota"},
        "type": "angebot_nachfrage",
        "description_prefix": "OPEC-Entscheidung",
    },
    {
        "keywords": {"tariff", "trade war", "import duty", "sanction", "embargo"},
        "type": "geopolitisch",
        "description_prefix": "Handelspolitik/Sanktionen",
    },
    {
        "keywords": {"war", "invasion", "conflict", "military", "attack", "strike"},
        "type": "geopolitisch",
        "description_prefix": "Geopolitisches Ereignis",
    },
    {
        "keywords": {"inflation", "cpi", "pce", "consumer price"},
        "type": "makro_wirtschaft",
        "description_prefix": "Inflationsdaten",
    },
    {
        "keywords": {"recession", "gdp", "economic slowdown", "contraction"},
        "type": "makro_wirtschaft",
        "description_prefix": "Konjunkturdaten",
    },
    {
        "keywords": {"nfp", "jobs report", "unemployment", "payroll"},
        "type": "makro_wirtschaft",
        "description_prefix": "Arbeitsmarktdaten",
    },
]


@dataclass
class MarketContext:
    """Ergebnis der News-Analyse."""
    event_identified: bool = False
    event_description: str = ""
    event_type: str = ""
    affected_assets: list[str] = field(default_factory=list)
    sentiment_scores: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    headlines: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.event_identified and not self.sentiment_scores

    def to_prompt_text(self) -> str:
        """Formatiert den Kontext für den Claude-Prompt."""
        if self.is_empty():
            return ""
        lines = ["## Aktuelle Nachrichtenlage\n"]
        if self.event_identified:
            lines.append(f"**Makroereignis:** {self.event_description} (Typ: {self.event_type}, Konfidenz: {self.confidence:.0%})")
            if self.affected_assets:
                lines.append(f"**Betroffene Assets:** {', '.join(self.affected_assets)}")
        if self.sentiment_scores:
            scores = ", ".join(
                f"{asset}: {'bullish' if s > 0 else 'bearish' if s < 0 else 'neutral'} ({s:+.2f})"
                for asset, s in self.sentiment_scores.items()
            )
            lines.append(f"**Sentiment:** {scores}")
        if self.headlines:
            lines.append("\n**Top-Schlagzeilen:**")
            for h in self.headlines[:5]:
                lines.append(f"- {h}")
        return "\n".join(lines)


# ── In-Memory Cache ────────────────────────────────────────────────────────────
_cache: Optional[MarketContext] = None
_cache_timestamp: float = 0.0


class NewsAnalyzer:
    """
    Ruft Wirtschaftsnachrichten ab und bewertet Sentiment für Rohstoffe.
    Graceful Degradation: funktioniert auch ohne API-Key.
    """

    def __init__(self) -> None:
        self._enabled = bool(config.NEWS_API_KEY)
        if not self._enabled:
            logger.info("News-Analyse deaktiviert (NEWS_API_KEY nicht gesetzt)")

    async def get_market_context(self) -> MarketContext:
        """
        Hauptmethode – gibt aktuellen Marktkontext zurück.
        Nutzt Cache um API-Limits zu schonen (Standard: 4h).
        """
        if not self._enabled:
            return MarketContext()

        global _cache, _cache_timestamp
        cache_age = time.time() - _cache_timestamp
        if _cache is not None and cache_age < config.NEWS_FETCH_INTERVAL_SECONDS:
            logger.debug("News-Cache gültig (Alter: %.0f min)", cache_age / 60)
            return _cache

        articles = await self.fetch_news()
        if not articles:
            return MarketContext()

        context = self._build_context(articles)
        _cache = context
        _cache_timestamp = time.time()
        logger.info(
            "News-Analyse: %d Artikel, Event=%s, Konfidenz=%.0f%%",
            len(articles),
            context.event_description or "keins",
            context.confidence * 100,
        )
        return context

    async def fetch_news(self) -> list[dict]:
        """Ruft aktuelle Nachrichten von NewsAPI.org ab."""
        since = (datetime.utcnow() - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")
        params = {
            "q": " OR ".join(config.NEWS_KEYWORDS),
            "from": since,
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": 50,
            "apiKey": config.NEWS_API_KEY,
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get("https://newsapi.org/v2/everything", params=params)
                response.raise_for_status()
                data = response.json()
                articles = data.get("articles", [])
                logger.debug("NewsAPI: %d Artikel abgerufen", len(articles))
                return articles
        except Exception as exc:
            logger.warning("News-Abruf fehlgeschlagen: %s – Bot läuft ohne News-Kontext", exc)
            return []

    def classify_sentiment(self, headline: str, description: str) -> dict[str, float]:
        """
        Bewertet Sentiment pro Asset basierend auf Keywords.
        Gibt Score zwischen -1.0 (bearish) und +1.0 (bullish) zurück.
        """
        text = (headline + " " + (description or "")).lower()
        scores: dict[str, float] = {}

        for asset, asset_keys in _ASSET_KEYWORDS.items():
            if not any(kw in text for kw in asset_keys):
                continue

            words = set(text.split())
            bullish_hits = len(words & _BULLISH)
            bearish_hits = len(words & _BEARISH)
            total = bullish_hits + bearish_hits
            if total == 0:
                scores[asset] = 0.0
            else:
                scores[asset] = round((bullish_hits - bearish_hits) / total, 2)

        return scores

    def identify_macro_event(self, articles: list[dict]) -> tuple[str, str, float]:
        """
        Identifiziert das dominante Makroereignis.
        Gibt (description, event_type, confidence) zurück.
        """
        pattern_hits: dict[int, int] = {}
        all_text = " ".join(
            (a.get("title", "") + " " + (a.get("description") or "")).lower()
            for a in articles
        )

        for i, pattern in enumerate(_EVENT_PATTERNS):
            hits = sum(1 for kw in pattern["keywords"] if kw in all_text)
            if hits > 0:
                pattern_hits[i] = hits

        if not pattern_hits:
            return "", "", 0.0

        best_idx = max(pattern_hits, key=lambda k: pattern_hits[k])
        best_pattern = _EVENT_PATTERNS[best_idx]
        hits = pattern_hits[best_idx]
        confidence = min(1.0, hits / 5)

        # Versuche eine spezifischere Beschreibung aus den Schlagzeilen zu extrahieren
        description = best_pattern["description_prefix"]
        for article in articles[:10]:
            title = article.get("title", "").lower()
            if any(kw in title for kw in best_pattern["keywords"]):
                description = article.get("title", description)
                break

        return description, best_pattern["type"], round(confidence, 2)

    def _build_context(self, articles: list[dict]) -> MarketContext:
        """Erstellt MarketContext aus einer Liste von Artikeln."""
        # Sentiment aggregieren
        combined_scores: dict[str, list[float]] = {}
        headlines: list[str] = []

        for article in articles:
            title = article.get("title", "")
            desc = article.get("description", "")
            if not title:
                continue
            headlines.append(title)
            scores = self.classify_sentiment(title, desc)
            for asset, score in scores.items():
                combined_scores.setdefault(asset, []).append(score)

        sentiment_scores = {
            asset: round(sum(scores) / len(scores), 2)
            for asset, scores in combined_scores.items()
        }

        affected_assets = [
            asset for asset, score in sentiment_scores.items()
            if abs(score) > 0.2
        ]

        event_desc, event_type, confidence = self.identify_macro_event(articles)

        return MarketContext(
            event_identified=bool(event_desc),
            event_description=event_desc,
            event_type=event_type,
            affected_assets=affected_assets,
            sentiment_scores=sentiment_scores,
            confidence=confidence,
            headlines=headlines[:10],
        )
