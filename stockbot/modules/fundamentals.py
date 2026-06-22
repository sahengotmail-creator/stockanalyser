"""
Fundamental Analysis Module.

Collects financial metrics from Yahoo Finance via yfinance and
computes a fundamental quality score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class FundamentalMetrics:
    """Container for fundamental financial metrics."""

    symbol: str = ""
    company_name: str = ""
    sector: str = ""
    industry: str = ""
    market_cap: Optional[float] = None
    revenue: Optional[float] = None
    net_income: Optional[float] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    peg_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    roe: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    free_cash_flow: Optional[float] = None
    current_price: Optional[float] = None
    book_value: Optional[float] = None
    price_to_book: Optional[float] = None
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None
    score: float = 0.0
    summary: str = ""
    raw_info: dict = field(default_factory=dict)


def analyze(ticker: yf.Ticker) -> FundamentalMetrics:
    """
    Compute fundamental metrics and quality score for a ticker.

    Parameters
    ----------
    ticker : yf.Ticker
        A yfinance Ticker object.

    Returns
    -------
    FundamentalMetrics
        Populated metrics container with score and narrative summary.
    """
    metrics = FundamentalMetrics()

    try:
        info = ticker.info or {}
    except Exception as exc:
        logger.error("Failed to fetch ticker info: %s", exc)
        info = {}

    metrics.raw_info = info
    metrics.symbol = info.get("symbol", "")
    metrics.company_name = info.get("longName", info.get("shortName", "N/A"))
    metrics.sector = info.get("sector", "N/A")
    metrics.industry = info.get("industry", "N/A")
    metrics.market_cap = _safe_float(info.get("marketCap"))
    metrics.revenue = _safe_float(info.get("totalRevenue"))
    metrics.net_income = _safe_float(info.get("netIncomeToCommon"))
    metrics.pe_ratio = _safe_float(info.get("trailingPE"))
    metrics.forward_pe = _safe_float(info.get("forwardPE"))
    metrics.peg_ratio = _safe_float(info.get("pegRatio"))
    metrics.debt_to_equity = _safe_float(info.get("debtToEquity"))
    metrics.roe = _safe_float(info.get("returnOnEquity"))
    metrics.gross_margin = _safe_float(info.get("grossMargins"))
    metrics.operating_margin = _safe_float(info.get("operatingMargins"))
    metrics.free_cash_flow = _safe_float(info.get("freeCashflow"))
    metrics.current_price = _safe_float(
        info.get("currentPrice") or info.get("regularMarketPrice")
    )
    metrics.book_value = _safe_float(info.get("bookValue"))
    metrics.price_to_book = _safe_float(info.get("priceToBook"))
    metrics.dividend_yield = _safe_float(info.get("dividendYield"))
    metrics.beta = _safe_float(info.get("beta"))

    metrics.score = _compute_score(metrics)
    metrics.summary = _build_summary(metrics)
    logger.info("Fundamental analysis complete for %s. Score: %.2f", metrics.symbol, metrics.score)
    return metrics


def _safe_float(value: object) -> Optional[float]:
    """Convert a value to float, returning None on failure."""
    if value is None:
        return None
    try:
        f = float(value)
        return None if (f != f) else f  # NaN check
    except (TypeError, ValueError):
        return None


def _compute_score(m: FundamentalMetrics) -> float:
    """
    Score fundamental quality on a 0-10 scale.

    Scoring rubric (each component contributes up to a fixed weight):
      • Profitability (ROE, net income)        — 20%
      • Valuation (PE, PEG, P/B)              — 20%
      • Margins (gross, operating)             — 20%
      • Leverage (debt-to-equity)              — 15%
      • Cash generation (FCF)                 — 15%
      • Growth proxy (revenue vs net income)   — 10%
    """
    components: list[tuple[float, float]] = []  # (value, weight)

    # --- Profitability: ROE ---
    if m.roe is not None:
        roe_pct = m.roe * 100
        if roe_pct >= 20:
            components.append((1.0, 2.0))
        elif roe_pct >= 10:
            components.append((0.6, 2.0))
        elif roe_pct >= 0:
            components.append((0.2, 2.0))
        else:
            components.append((-0.5, 2.0))

    # --- Valuation: Trailing PE ---
    if m.pe_ratio is not None and m.pe_ratio > 0:
        pe = m.pe_ratio
        if pe < 15:
            components.append((1.0, 1.5))
        elif pe < 25:
            components.append((0.6, 1.5))
        elif pe < 40:
            components.append((0.2, 1.5))
        else:
            components.append((-0.3, 1.5))

    # --- Valuation: PEG ---
    if m.peg_ratio is not None and m.peg_ratio > 0:
        peg = m.peg_ratio
        if peg < 1:
            components.append((1.0, 1.0))
        elif peg < 2:
            components.append((0.5, 1.0))
        else:
            components.append((-0.25, 1.0))

    # --- Margins: Gross Margin ---
    if m.gross_margin is not None:
        gm = m.gross_margin * 100
        if gm >= 50:
            components.append((1.0, 1.5))
        elif gm >= 30:
            components.append((0.6, 1.5))
        elif gm >= 10:
            components.append((0.2, 1.5))
        else:
            components.append((-0.3, 1.5))

    # --- Margins: Operating Margin ---
    if m.operating_margin is not None:
        om = m.operating_margin * 100
        if om >= 20:
            components.append((1.0, 1.5))
        elif om >= 10:
            components.append((0.5, 1.5))
        elif om >= 0:
            components.append((0.1, 1.5))
        else:
            components.append((-0.5, 1.5))

    # --- Leverage: Debt-to-Equity ---
    if m.debt_to_equity is not None:
        de = m.debt_to_equity
        if de < 50:
            components.append((1.0, 1.5))
        elif de < 100:
            components.append((0.5, 1.5))
        elif de < 200:
            components.append((0.0, 1.5))
        else:
            components.append((-0.5, 1.5))

    # --- Cash: Free Cash Flow ---
    if m.free_cash_flow is not None:
        if m.free_cash_flow > 0:
            components.append((1.0, 1.5))
        else:
            components.append((-0.5, 1.5))

    if not components:
        return 5.0  # neutral default when no data

    weighted_sum = sum(v * w for v, w in components)
    total_weight = sum(w for _, w in components)
    normalized = weighted_sum / total_weight  # -1 to 1
    score = round((normalized + 1) / 2 * 10, 2)  # 0 to 10
    return max(0.0, min(10.0, score))


def _fmt(value: Optional[float], prefix: str = "", suffix: str = "", scale: float = 1.0, decimals: int = 2) -> str:
    """Format a numeric value for display."""
    if value is None:
        return "N/A"
    scaled = value * scale
    return f"{prefix}{scaled:,.{decimals}f}{suffix}"


def _fmt_large(value: Optional[float]) -> str:
    """Format large numbers with B/M suffix."""
    if value is None:
        return "N/A"
    if abs(value) >= 1e9:
        return f"${value / 1e9:.2f}B"
    if abs(value) >= 1e6:
        return f"${value / 1e6:.2f}M"
    return f"${value:,.0f}"


def _build_summary(m: FundamentalMetrics) -> str:
    """Build a human-readable fundamental analysis narrative."""
    lines: list[str] = [
        f"Company: {m.company_name} ({m.symbol})",
        f"Sector: {m.sector} | Industry: {m.industry}",
        f"Market Cap: {_fmt_large(m.market_cap)}",
        "",
        "--- Income & Cash Flow ---",
        f"Revenue: {_fmt_large(m.revenue)}",
        f"Net Income: {_fmt_large(m.net_income)}",
        f"Free Cash Flow: {_fmt_large(m.free_cash_flow)}",
        "",
        "--- Valuation ---",
        f"Current Price: {_fmt(m.current_price, '$')}",
        f"Trailing P/E: {_fmt(m.pe_ratio)}",
        f"Forward P/E: {_fmt(m.forward_pe)}",
        f"PEG Ratio: {_fmt(m.peg_ratio)}",
        f"Price/Book: {_fmt(m.price_to_book)}",
        "",
        "--- Profitability ---",
        f"Gross Margin: {_fmt(m.gross_margin, suffix='%', scale=100)}",
        f"Operating Margin: {_fmt(m.operating_margin, suffix='%', scale=100)}",
        f"ROE: {_fmt(m.roe, suffix='%', scale=100)}",
        "",
        "--- Financial Health ---",
        f"Debt/Equity: {_fmt(m.debt_to_equity)}",
        f"Beta: {_fmt(m.beta)}",
        f"Dividend Yield: {_fmt(m.dividend_yield, suffix='%', scale=100)}",
        "",
        f"Fundamental Score: {m.score}/10",
    ]
    return "\n".join(lines)
