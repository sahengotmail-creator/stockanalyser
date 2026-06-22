"""
AI Analysis Module.

Generates an investment thesis using an OpenAI-compatible LLM.
Falls back to a rule-based thesis when no API key is configured.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from .fundamentals import FundamentalMetrics
from .technical import TechnicalIndicators
from .valuation import ValuationResult
from .earnings import EarningsAnalysis
from .news import NewsAnalysis

logger = logging.getLogger(__name__)


@dataclass
class InvestmentThesis:
    """Container for the AI-generated investment thesis."""

    business_overview: str = ""
    bull_case: str = ""
    bear_case: str = ""
    risk_factors: str = ""
    competitive_advantage: str = ""
    five_year_outlook: str = ""
    full_thesis: str = ""


def analyze(
    fundamentals: FundamentalMetrics,
    technicals: TechnicalIndicators,
    valuation: ValuationResult,
    earnings: EarningsAnalysis,
    news: NewsAnalysis,
) -> InvestmentThesis:
    """
    Generate an investment thesis.

    Tries OpenAI API first (if OPENAI_API_KEY env var is set),
    then falls back to a structured rule-based thesis.

    Parameters
    ----------
    fundamentals : FundamentalMetrics
    technicals : TechnicalIndicators
    valuation : ValuationResult
    earnings : EarningsAnalysis
    news : NewsAnalysis

    Returns
    -------
    InvestmentThesis
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if api_key:
        try:
            return _llm_thesis(api_key, fundamentals, technicals, valuation, earnings, news)
        except Exception as exc:
            logger.warning("LLM thesis generation failed (%s); using rule-based fallback.", exc)

    return _rule_based_thesis(fundamentals, technicals, valuation, earnings, news)


def _build_context(
    fundamentals: FundamentalMetrics,
    technicals: TechnicalIndicators,
    valuation: ValuationResult,
    earnings: EarningsAnalysis,
    news: NewsAnalysis,
) -> str:
    """Build a compact context string for the LLM prompt."""
    parts = [
        f"Symbol: {fundamentals.symbol}",
        f"Company: {fundamentals.company_name}",
        f"Sector: {fundamentals.sector} / Industry: {fundamentals.industry}",
        f"Market Cap: ${fundamentals.market_cap/1e9:.1f}B" if fundamentals.market_cap else "",
        f"Current Price: ${fundamentals.current_price:.2f}" if fundamentals.current_price else "",
        f"Fair Value (DCF): ${valuation.fair_value:.2f}" if valuation.fair_value else "",
        f"Upside: {valuation.upside_pct:.1f}%" if valuation.upside_pct else "",
        f"P/E: {fundamentals.pe_ratio:.1f}" if fundamentals.pe_ratio else "",
        f"ROE: {fundamentals.roe*100:.1f}%" if fundamentals.roe else "",
        f"Gross Margin: {fundamentals.gross_margin*100:.1f}%" if fundamentals.gross_margin else "",
        f"Revenue Growth YoY: {earnings.revenue_growth_yoy:.1f}%" if earnings.revenue_growth_yoy else "",
        f"Technical Score: {technicals.score}/10",
        f"Fundamental Score: {fundamentals.score}/10",
        f"RSI: {technicals.rsi:.1f}" if technicals.rsi else "",
        f"News Sentiment: {news.sentiment_score:.1f}/10",
        f"Recent bullish news: {'; '.join(news.bullish_developments[:3])}" if news.bullish_developments else "",
        f"Recent bearish news: {'; '.join(news.bearish_developments[:3])}" if news.bearish_developments else "",
    ]
    return "\n".join(p for p in parts if p)


def _llm_thesis(
    api_key: str,
    fundamentals: FundamentalMetrics,
    technicals: TechnicalIndicators,
    valuation: ValuationResult,
    earnings: EarningsAnalysis,
    news: NewsAnalysis,
) -> InvestmentThesis:
    """Call OpenAI API to generate investment thesis."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package not installed") from exc

    client = OpenAI(api_key=api_key)
    context = _build_context(fundamentals, technicals, valuation, earnings, news)

    system_prompt = (
        "You are an expert equity research analyst. "
        "Given the following stock data, write a concise investment thesis. "
        "Structure your response with these exact headings:\n"
        "BUSINESS OVERVIEW:\nBULL CASE:\nBEAR CASE:\nRISK FACTORS:\n"
        "COMPETITIVE ADVANTAGE:\n5-YEAR OUTLOOK:\n"
        "Keep each section to 2-3 sentences."
    )

    user_prompt = f"Analyze this stock:\n\n{context}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1000,
        temperature=0.3,
    )

    raw_text = response.choices[0].message.content or ""
    return _parse_thesis(raw_text)


def _parse_thesis(text: str) -> InvestmentThesis:
    """Parse structured thesis text from LLM output."""
    thesis = InvestmentThesis()
    thesis.full_thesis = text

    sections = {
        "business_overview": ["BUSINESS OVERVIEW:", "Business Overview:"],
        "bull_case": ["BULL CASE:", "Bull Case:"],
        "bear_case": ["BEAR CASE:", "Bear Case:"],
        "risk_factors": ["RISK FACTORS:", "Risk Factors:"],
        "competitive_advantage": ["COMPETITIVE ADVANTAGE:", "Competitive Advantage:"],
        "five_year_outlook": ["5-YEAR OUTLOOK:", "5-Year Outlook:", "FIVE-YEAR OUTLOOK:"],
    }

    lines = text.split("\n")
    current_section = None
    section_content: dict[str, list[str]] = {k: [] for k in sections}

    for line in lines:
        stripped = line.strip()
        matched = False
        for attr, headers in sections.items():
            if any(stripped.startswith(h) for h in headers):
                current_section = attr
                # Inline content after the header
                for h in headers:
                    if stripped.startswith(h):
                        inline = stripped[len(h):].strip()
                        if inline:
                            section_content[attr].append(inline)
                        break
                matched = True
                break
        if not matched and current_section and stripped:
            section_content[current_section].append(stripped)

    for attr, content_lines in section_content.items():
        setattr(thesis, attr, " ".join(content_lines).strip())

    return thesis


def _rule_based_thesis(
    fundamentals: FundamentalMetrics,
    technicals: TechnicalIndicators,
    valuation: ValuationResult,
    earnings: EarningsAnalysis,
    news: NewsAnalysis,
) -> InvestmentThesis:
    """Generate a structured rule-based investment thesis."""
    thesis = InvestmentThesis()
    sym = fundamentals.symbol
    name = fundamentals.company_name
    sector = fundamentals.sector

    # Business Overview
    market_cap_str = f"${fundamentals.market_cap/1e9:.1f}B" if fundamentals.market_cap else "undisclosed market cap"
    thesis.business_overview = (
        f"{name} ({sym}) is a {sector} company with a market capitalization of {market_cap_str}. "
        f"The company operates in the {fundamentals.industry} industry. "
        f"It generates revenue of {_fmt_large(fundamentals.revenue)} with a gross margin of "
        f"{fundamentals.gross_margin*100:.1f}%." if fundamentals.gross_margin else
        f"{name} ({sym}) is a {sector} company with a market capitalization of {market_cap_str}. "
        f"It operates in the {fundamentals.industry} industry."
    )

    # Bull Case
    bull_points: list[str] = []
    if technicals.golden_cross:
        bull_points.append("a recent Golden Cross on the price chart indicating bullish momentum")
    if (earnings.revenue_growth_yoy or 0) >= 10:
        bull_points.append(f"strong revenue growth of {earnings.revenue_growth_yoy:.1f}% year-over-year")
    if (valuation.upside_pct or 0) >= 15:
        bull_points.append(f"the stock appears undervalued with {valuation.upside_pct:.1f}% upside to fair value")
    if (fundamentals.roe or 0) >= 0.15:
        bull_points.append(f"high return on equity of {fundamentals.roe*100:.1f}%")
    if (fundamentals.gross_margin or 0) >= 0.40:
        bull_points.append(f"strong gross margins of {fundamentals.gross_margin*100:.1f}%")
    if news.sentiment_score >= 6.5:
        bull_points.append("positive recent news sentiment")
    if not bull_points:
        bull_points.append("stable business model with consistent operations")
    thesis.bull_case = "Bull case is supported by: " + "; ".join(bull_points) + "."

    # Bear Case
    bear_points: list[str] = []
    if technicals.death_cross:
        bear_points.append("a recent Death Cross suggesting bearish price trend")
    if (earnings.revenue_growth_yoy or 0) < 0:
        bear_points.append(f"declining revenue growth of {earnings.revenue_growth_yoy:.1f}%")
    if (valuation.upside_pct or 0) <= -15:
        bear_points.append(f"the stock appears overvalued by {abs(valuation.upside_pct or 0):.1f}% vs fair value")
    if (fundamentals.debt_to_equity or 0) > 150:
        bear_points.append(f"elevated debt-to-equity ratio of {fundamentals.debt_to_equity:.0f}%")
    if technicals.overbought:
        bear_points.append(f"technically overbought RSI of {technicals.rsi:.0f}")
    if news.sentiment_score <= 4.0:
        bear_points.append("negative recent news flow")
    if not bear_points:
        bear_points.append("potential valuation compression or sector rotation risk")
    thesis.bear_case = "Bear case includes: " + "; ".join(bear_points) + "."

    # Risk Factors
    risks: list[str] = []
    if (fundamentals.beta or 1.0) > 1.5:
        risks.append(f"high market sensitivity (beta {fundamentals.beta:.1f})")
    if (fundamentals.debt_to_equity or 0) > 100:
        risks.append("significant financial leverage")
    if news.risks:
        risks.append("regulatory or competitive risks noted in recent news")
    risks.append("macroeconomic headwinds including interest rate changes")
    risks.append("competitive dynamics in the " + fundamentals.industry + " industry")
    thesis.risk_factors = "Key risks: " + "; ".join(risks) + "."

    # Competitive Advantage
    moat_points: list[str] = []
    if (fundamentals.gross_margin or 0) >= 0.50:
        moat_points.append("exceptional pricing power reflected in >50% gross margins")
    elif (fundamentals.gross_margin or 0) >= 0.30:
        moat_points.append("solid gross margins suggesting some competitive moat")
    if (fundamentals.roe or 0) >= 0.20:
        moat_points.append("high returns on equity indicating an efficient business model")
    if (fundamentals.operating_margin or 0) >= 0.15:
        moat_points.append("strong operating leverage")
    if not moat_points:
        moat_points.append("the company's competitive position requires further analysis")
    thesis.competitive_advantage = " ".join(moat_points) + "."

    # 5-Year Outlook
    combined_score = (
        technicals.score * 0.25 +
        fundamentals.score * 0.35 +
        (valuation.score if valuation else 5.0) * 0.25 +
        news.sentiment_score * 0.15
    )
    if combined_score >= 7.0:
        outlook_tone = "favorable"
        outlook_action = "outperform the broader market"
    elif combined_score >= 5.5:
        outlook_tone = "moderate"
        outlook_action = "track closely with market returns"
    else:
        outlook_tone = "cautious"
        outlook_action = "face headwinds in the medium term"

    thesis.five_year_outlook = (
        f"The 5-year outlook for {name} appears {outlook_tone}. "
        f"Based on current trends in revenue growth, margins, and valuation, the stock is expected to "
        f"{outlook_action}. Investors should monitor earnings execution, competitive dynamics, "
        f"and macroeconomic conditions over the investment horizon."
    )

    thesis.full_thesis = "\n\n".join([
        f"BUSINESS OVERVIEW:\n{thesis.business_overview}",
        f"BULL CASE:\n{thesis.bull_case}",
        f"BEAR CASE:\n{thesis.bear_case}",
        f"RISK FACTORS:\n{thesis.risk_factors}",
        f"COMPETITIVE ADVANTAGE:\n{thesis.competitive_advantage}",
        f"5-YEAR OUTLOOK:\n{thesis.five_year_outlook}",
    ])

    logger.info("Rule-based investment thesis generated for %s.", fundamentals.symbol)
    return thesis


def _fmt_large(value: object) -> str:
    if value is None:
        return "N/A"
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "N/A"
    if abs(v) >= 1e9:
        return f"${v/1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v/1e6:.2f}M"
    return f"${v:,.0f}"
