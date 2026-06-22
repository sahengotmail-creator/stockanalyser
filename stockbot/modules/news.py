"""
News Analysis Module.

Fetches latest news articles for a stock ticker and generates
a sentiment score plus bullish/bearish/risk/catalyst summaries.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import requests
import yfinance as yf
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Simple keyword lists for rule-based sentiment
BULLISH_KEYWORDS = [
    "beat", "beats", "surpass", "record", "growth", "gain", "rally", "upgrade",
    "outperform", "profit", "revenue", "strong", "positive", "bullish", "buy",
    "raise", "raised", "increase", "accelerat", "expand", "exceed", "top",
    "deliver", "robust", "momentum", "opportunity", "partnership", "launch",
    "approval", "approved", "win", "won", "optimistic", "breakthrough",
]

BEARISH_KEYWORDS = [
    "miss", "misses", "decline", "drop", "fall", "loss", "layoff", "cut",
    "downgrade", "underperform", "weak", "negative", "bearish", "sell",
    "reduce", "warn", "warning", "concern", "risk", "lawsuit", "investigation",
    "recession", "slowdown", "disappointing", "below", "missed", "struggled",
    "deficit", "debt", "bankruptcy", "delay", "withdrawal", "recall",
]

RISK_KEYWORDS = [
    "risk", "uncertainty", "volatile", "competition", "regulatory", "lawsuit",
    "investigation", "tariff", "inflation", "rate hike", "slowdown",
    "geopolit", "cyber", "breach", "data", "recall", "liability",
]


@dataclass
class NewsArticle:
    """A single news article."""

    title: str = ""
    publisher: str = ""
    link: str = ""
    published: str = ""
    sentiment: str = "neutral"  # bullish / bearish / neutral


@dataclass
class NewsAnalysis:
    """Container for news analysis output."""

    articles: list[NewsArticle] = field(default_factory=list)
    bullish_developments: list[str] = field(default_factory=list)
    bearish_developments: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    catalysts: list[str] = field(default_factory=list)
    sentiment_score: float = 5.0
    summary: str = ""


def analyze(ticker: yf.Ticker, symbol: str) -> NewsAnalysis:
    """
    Fetch and analyze news for a stock.

    Parameters
    ----------
    ticker : yf.Ticker
        A yfinance Ticker object.
    symbol : str
        Ticker symbol string (e.g., "AAPL").

    Returns
    -------
    NewsAnalysis
        Structured news analysis with sentiment score.
    """
    result = NewsAnalysis()
    articles = _fetch_news(ticker)

    if not articles:
        result.summary = "No news articles found."
        result.sentiment_score = 5.0
        return result

    result.articles = articles

    bullish: list[str] = []
    bearish: list[str] = []
    risks: list[str] = []
    catalysts: list[str] = []
    sentiment_scores: list[float] = []

    for article in articles:
        text = (article.title + " ").lower()
        bull_count = sum(1 for kw in BULLISH_KEYWORDS if kw in text)
        bear_count = sum(1 for kw in BEARISH_KEYWORDS if kw in text)
        risk_count = sum(1 for kw in RISK_KEYWORDS if kw in text)

        if bull_count > bear_count:
            article.sentiment = "bullish"
            bullish.append(article.title)
            sentiment_scores.append(7.0 + min(bull_count, 3) * 0.5)
        elif bear_count > bull_count:
            article.sentiment = "bearish"
            bearish.append(article.title)
            sentiment_scores.append(3.0 - min(bear_count, 3) * 0.5)
        else:
            article.sentiment = "neutral"
            sentiment_scores.append(5.0)

        if risk_count > 0:
            risks.append(article.title)

        # Catalysts: positive news about earnings, products, partnerships
        catalyst_kws = ["beat", "approval", "launch", "partnership", "contract", "win", "expand", "record"]
        if any(kw in text for kw in catalyst_kws):
            catalysts.append(article.title)

    result.bullish_developments = bullish[:5]
    result.bearish_developments = bearish[:5]
    result.risks = risks[:5]
    result.catalysts = catalysts[:5]

    if sentiment_scores:
        result.sentiment_score = round(sum(sentiment_scores) / len(sentiment_scores), 2)
    else:
        result.sentiment_score = 5.0

    result.sentiment_score = max(0.0, min(10.0, result.sentiment_score))
    result.summary = _build_summary(result)
    logger.info("News analysis complete. %d articles. Sentiment: %.2f", len(articles), result.sentiment_score)
    return result


def _fetch_news(ticker: yf.Ticker) -> list[NewsArticle]:
    """Fetch news from yfinance ticker."""
    articles: list[NewsArticle] = []
    try:
        news_items = ticker.news or []
        for item in news_items[:20]:
            if isinstance(item, dict):
                content = item.get("content", {}) or {}
                title = (
                    item.get("title")
                    or content.get("title")
                    or ""
                )
                publisher = (
                    item.get("publisher")
                    or (content.get("provider") or {}).get("displayName", "")
                    or ""
                )
                link = (
                    item.get("link")
                    or content.get("canonicalUrl", {}).get("url", "")
                    or ""
                )
                pub_date = (
                    item.get("providerPublishTime")
                    or content.get("pubDate")
                    or ""
                )
                if pub_date and isinstance(pub_date, (int, float)):
                    import datetime
                    pub_date = datetime.datetime.fromtimestamp(pub_date).strftime("%Y-%m-%d")
                if title:
                    articles.append(
                        NewsArticle(
                            title=str(title),
                            publisher=str(publisher),
                            link=str(link),
                            published=str(pub_date),
                        )
                    )
    except Exception as exc:
        logger.warning("Failed to fetch news from yfinance: %s", exc)

    return articles


def _build_summary(n: NewsAnalysis) -> str:
    """Build a human-readable news summary."""
    total = len(n.articles)
    bull_count = sum(1 for a in n.articles if a.sentiment == "bullish")
    bear_count = sum(1 for a in n.articles if a.sentiment == "bearish")
    neutral_count = total - bull_count - bear_count

    lines: list[str] = [
        f"--- News Analysis ({total} articles) ---",
        f"Bullish: {bull_count} | Bearish: {bear_count} | Neutral: {neutral_count}",
        "",
    ]

    if n.bullish_developments:
        lines.append("Bullish Developments:")
        for item in n.bullish_developments:
            lines.append(f"  + {item}")
        lines.append("")

    if n.bearish_developments:
        lines.append("Bearish Developments:")
        for item in n.bearish_developments:
            lines.append(f"  - {item}")
        lines.append("")

    if n.risks:
        lines.append("Key Risks Mentioned:")
        for item in n.risks[:3]:
            lines.append(f"  ! {item}")
        lines.append("")

    if n.catalysts:
        lines.append("Potential Catalysts:")
        for item in n.catalysts[:3]:
            lines.append(f"  * {item}")
        lines.append("")

    sentiment_label = (
        "Very Bullish" if n.sentiment_score >= 7.5
        else "Bullish" if n.sentiment_score >= 6.0
        else "Bearish" if n.sentiment_score <= 3.5
        else "Very Bearish" if n.sentiment_score <= 2.0
        else "Neutral"
    )
    lines.append(f"Overall News Sentiment: {sentiment_label} ({n.sentiment_score:.1f}/10)")
    return "\n".join(lines)
