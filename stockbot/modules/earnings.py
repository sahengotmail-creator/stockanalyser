"""
Earnings Analysis Module.

Analyzes last 4 quarters of earnings: revenue, EPS growth,
earnings surprises, and generates trend summaries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class QuarterlyEarnings:
    """Data for a single quarter."""

    period: str = ""
    reported_eps: Optional[float] = None
    estimated_eps: Optional[float] = None
    surprise_pct: Optional[float] = None
    revenue: Optional[float] = None


@dataclass
class EarningsAnalysis:
    """Container for earnings analysis output."""

    quarters: list[QuarterlyEarnings] = field(default_factory=list)
    revenue_growth_yoy: Optional[float] = None
    eps_growth_yoy: Optional[float] = None
    beat_rate: Optional[float] = None  # fraction of quarters that beat estimates
    score: float = 5.0
    summary: str = ""


def analyze(ticker: yf.Ticker) -> EarningsAnalysis:
    """
    Analyze earnings history for the given ticker.

    Parameters
    ----------
    ticker : yf.Ticker
        A yfinance Ticker object.

    Returns
    -------
    EarningsAnalysis
        Structured earnings data with score and narrative.
    """
    result = EarningsAnalysis()

    try:
        # Quarterly earnings data (EPS)
        earnings_df = _fetch_quarterly_earnings(ticker)
        # Quarterly financials (revenue)
        financials_df = _fetch_quarterly_financials(ticker)
    except Exception as exc:
        logger.error("Error fetching earnings data: %s", exc)
        result.summary = "Unable to retrieve earnings data."
        return result

    # Merge EPS and revenue
    quarters = _build_quarters(earnings_df, financials_df)
    result.quarters = quarters[-4:] if len(quarters) >= 4 else quarters

    if not result.quarters:
        result.summary = "No quarterly earnings data available."
        return result

    result.revenue_growth_yoy = _compute_revenue_growth_yoy(financials_df)
    result.eps_growth_yoy = _compute_eps_growth_yoy(earnings_df)
    result.beat_rate = _compute_beat_rate(result.quarters)
    result.score = _compute_score(result)
    result.summary = _build_summary(result)

    logger.info("Earnings analysis complete. Score: %.2f", result.score)
    return result


def _fetch_quarterly_earnings(ticker: yf.Ticker) -> pd.DataFrame:
    """Fetch quarterly EPS data."""
    try:
        df = ticker.earnings_history
        if df is not None and not df.empty:
            return df
    except Exception:
        pass
    try:
        df = ticker.quarterly_earnings
        if df is not None and not df.empty:
            return df
    except Exception:
        pass
    return pd.DataFrame()


def _fetch_quarterly_financials(ticker: yf.Ticker) -> pd.DataFrame:
    """Fetch quarterly income statement (revenue)."""
    try:
        df = ticker.quarterly_financials
        if df is not None and not df.empty:
            return df
    except Exception:
        pass
    return pd.DataFrame()


def _build_quarters(
    earnings_df: pd.DataFrame,
    financials_df: pd.DataFrame,
) -> list[QuarterlyEarnings]:
    """Combine EPS and revenue data into QuarterlyEarnings list."""
    quarters: list[QuarterlyEarnings] = []

    # Build revenue map: period → revenue
    rev_map: dict[str, float] = {}
    if not financials_df.empty:
        rev_row = None
        for label in ["Total Revenue", "Revenue"]:
            if label in financials_df.index:
                rev_row = financials_df.loc[label]
                break
        if rev_row is not None:
            for col in rev_row.index:
                val = rev_row[col]
                if pd.notna(val):
                    period_str = str(col)[:10]
                    rev_map[period_str] = float(val)

    if not earnings_df.empty:
        # Handle earnings_history format
        if "epsEstimate" in earnings_df.columns or "epsActual" in earnings_df.columns:
            for _, row in earnings_df.iterrows():
                q = QuarterlyEarnings()
                period = str(row.name)[:10] if hasattr(row, "name") else ""
                q.period = period
                q.reported_eps = _safe_float(row.get("epsActual"))
                q.estimated_eps = _safe_float(row.get("epsEstimate"))
                if q.reported_eps is not None and q.estimated_eps is not None and q.estimated_eps != 0:
                    q.surprise_pct = round((q.reported_eps - q.estimated_eps) / abs(q.estimated_eps) * 100, 1)
                # Look up revenue
                for key in [period, period[:7]]:
                    if key in rev_map:
                        q.revenue = rev_map[key]
                        break
                quarters.append(q)
        # Handle quarterly_earnings format (index = years, columns include EPS)
        elif "EPS" in earnings_df.columns or "Earnings" in earnings_df.columns:
            eps_col = "EPS" if "EPS" in earnings_df.columns else "Earnings"
            for period, row in earnings_df.iterrows():
                q = QuarterlyEarnings()
                q.period = str(period)
                q.reported_eps = _safe_float(row.get(eps_col))
                rev_str = str(period)[:10]
                q.revenue = rev_map.get(rev_str)
                quarters.append(q)

    if not quarters and not financials_df.empty:
        # Fallback: build quarters from financials only
        rev_row = None
        for label in ["Total Revenue", "Revenue"]:
            if label in financials_df.index:
                rev_row = financials_df.loc[label]
                break
        if rev_row is not None:
            for col in sorted(rev_row.index, reverse=True):
                val = rev_row[col]
                if pd.notna(val):
                    q = QuarterlyEarnings()
                    q.period = str(col)[:10]
                    q.revenue = float(val)
                    quarters.append(q)

    # Sort chronologically (most recent last)
    quarters.sort(key=lambda x: x.period)
    return quarters


def _compute_revenue_growth_yoy(financials_df: pd.DataFrame) -> Optional[float]:
    """Estimate YoY revenue growth from quarterly financials."""
    if financials_df.empty:
        return None
    for label in ["Total Revenue", "Revenue"]:
        if label in financials_df.index:
            rev_row = financials_df.loc[label].dropna().sort_index(ascending=False)
            if len(rev_row) >= 5:
                recent = float(rev_row.iloc[0])
                year_ago = float(rev_row.iloc[4])
                if year_ago and year_ago != 0:
                    return round((recent / year_ago - 1) * 100, 1)
    return None


def _compute_eps_growth_yoy(earnings_df: pd.DataFrame) -> Optional[float]:
    """Estimate YoY EPS growth."""
    if earnings_df.empty:
        return None

    eps_col = None
    for col in ["epsActual", "EPS", "Earnings"]:
        if col in earnings_df.columns:
            eps_col = col
            break

    if eps_col is None:
        return None

    series = earnings_df[eps_col].dropna()
    if len(series) >= 5:
        recent = float(series.iloc[-1])
        year_ago = float(series.iloc[-5])
        if year_ago and year_ago != 0:
            return round((recent / year_ago - 1) * 100, 1)
    return None


def _compute_beat_rate(quarters: list[QuarterlyEarnings]) -> Optional[float]:
    """Fraction of quarters where reported EPS ≥ estimated EPS."""
    eligible = [q for q in quarters if q.reported_eps is not None and q.estimated_eps is not None]
    if not eligible:
        return None
    beats = sum(1 for q in eligible if q.reported_eps >= q.estimated_eps)
    return round(beats / len(eligible) * 100, 1)


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
        return None if f != f else f
    except (TypeError, ValueError):
        return None


def _compute_score(e: EarningsAnalysis) -> float:
    """Score earnings quality on 0-10 scale."""
    score = 5.0

    if e.revenue_growth_yoy is not None:
        if e.revenue_growth_yoy >= 20:
            score += 1.5
        elif e.revenue_growth_yoy >= 10:
            score += 0.75
        elif e.revenue_growth_yoy < 0:
            score -= 1.0

    if e.eps_growth_yoy is not None:
        if e.eps_growth_yoy >= 20:
            score += 1.5
        elif e.eps_growth_yoy >= 10:
            score += 0.75
        elif e.eps_growth_yoy < 0:
            score -= 1.0

    if e.beat_rate is not None:
        if e.beat_rate >= 75:
            score += 1.0
        elif e.beat_rate >= 50:
            score += 0.5
        else:
            score -= 0.5

    # Consistent surprise magnitude
    surprises = [q.surprise_pct for q in e.quarters if q.surprise_pct is not None]
    if surprises:
        avg_surprise = sum(surprises) / len(surprises)
        if avg_surprise >= 5:
            score += 0.5
        elif avg_surprise < 0:
            score -= 0.5

    return round(max(0.0, min(10.0, score)), 2)


def _fmt_large(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    if abs(value) >= 1e9:
        return f"${value / 1e9:.2f}B"
    if abs(value) >= 1e6:
        return f"${value / 1e6:.2f}M"
    return f"${value:,.0f}"


def _build_summary(e: EarningsAnalysis) -> str:
    lines: list[str] = ["--- Earnings Analysis ---"]

    for q in e.quarters:
        parts = [f"Q:{q.period}"]
        if q.revenue is not None:
            parts.append(f"Rev={_fmt_large(q.revenue)}")
        if q.reported_eps is not None:
            parts.append(f"EPS={q.reported_eps:.2f}")
        if q.estimated_eps is not None:
            parts.append(f"Est={q.estimated_eps:.2f}")
        if q.surprise_pct is not None:
            sign = "+" if q.surprise_pct >= 0 else ""
            parts.append(f"Surprise={sign}{q.surprise_pct:.1f}%")
        lines.append("  " + " | ".join(parts))

    lines.append("")
    if e.revenue_growth_yoy is not None:
        lines.append(f"Revenue Growth YoY: {e.revenue_growth_yoy:+.1f}%")
    if e.eps_growth_yoy is not None:
        lines.append(f"EPS Growth YoY: {e.eps_growth_yoy:+.1f}%")
    if e.beat_rate is not None:
        lines.append(f"Earnings Beat Rate: {e.beat_rate:.0f}%")

    trend = "Accelerating" if (e.revenue_growth_yoy or 0) >= 10 else ("Decelerating" if (e.revenue_growth_yoy or 0) < 0 else "Stable")
    lines.append(f"\nEarnings Trend: {trend} (Score: {e.score}/10)")
    return "\n".join(lines)
