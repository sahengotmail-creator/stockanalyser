"""
Valuation Module.

Implements a simplified Discounted Cash Flow (DCF) model to estimate
fair value, upside/downside potential, and margin of safety.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .fundamentals import FundamentalMetrics

logger = logging.getLogger(__name__)


@dataclass
class ValuationResult:
    """Container for DCF valuation output."""

    current_price: Optional[float] = None
    fair_value: Optional[float] = None
    upside_pct: Optional[float] = None
    downside_pct: Optional[float] = None
    margin_of_safety: Optional[float] = None
    dcf_assumptions: dict = None  # type: ignore[assignment]
    score: float = 5.0
    summary: str = ""

    def __post_init__(self) -> None:
        if self.dcf_assumptions is None:
            self.dcf_assumptions = {}


def analyze(metrics: FundamentalMetrics) -> ValuationResult:
    """
    Perform a simplified DCF valuation.

    The model uses:
      1. Free Cash Flow as the base cash flow.
      2. A two-stage growth model (near-term + terminal).
      3. WACC derived from beta and market risk premium.

    Parameters
    ----------
    metrics : FundamentalMetrics
        Fundamental metrics including FCF, revenue, margins, beta.

    Returns
    -------
    ValuationResult
        Fair value estimate, upside/downside, and margin of safety.
    """
    result = ValuationResult()
    result.current_price = metrics.current_price

    if metrics.current_price is None:
        result.summary = "No current price available; valuation skipped."
        return result

    # --- DCF Inputs ---
    fcf = metrics.free_cash_flow
    shares = metrics.raw_info.get("sharesOutstanding")

    if fcf is None or fcf <= 0 or shares is None or shares <= 0:
        # Fall back to earnings-based estimate
        result = _earnings_based_valuation(metrics)
        return result

    # Growth rate assumptions
    revenue_growth = _safe_float(metrics.raw_info.get("revenueGrowth")) or 0.08
    earnings_growth = _safe_float(metrics.raw_info.get("earningsGrowth")) or revenue_growth

    # Near-term growth (5 years): use analyst estimate, capped at reasonable bounds
    growth_rate_5y = max(-0.20, min(0.40, earnings_growth))
    # Long-term terminal growth (years 6-10)
    terminal_growth_rate = min(growth_rate_5y * 0.5, 0.04)  # revert toward GDP growth
    # Perpetuity growth rate
    perpetuity_growth = 0.025

    # Discount rate (WACC proxy)
    beta = metrics.beta or 1.0
    risk_free_rate = 0.045  # approximate 10-yr treasury
    equity_risk_premium = 0.055
    cost_of_equity = risk_free_rate + beta * equity_risk_premium
    wacc = cost_of_equity  # simplified: assume all-equity

    # --- Project FCF ---
    projected_fcf: list[float] = []
    cf = float(fcf)
    for year in range(1, 6):
        cf *= (1 + growth_rate_5y)
        projected_fcf.append(cf)

    for year in range(6, 11):
        cf *= (1 + terminal_growth_rate)
        projected_fcf.append(cf)

    # --- Discount to PV ---
    pv_fcf = 0.0
    for i, cashflow in enumerate(projected_fcf, start=1):
        pv_fcf += cashflow / (1 + wacc) ** i

    # --- Terminal Value (Gordon Growth Model) ---
    terminal_fcf = projected_fcf[-1] * (1 + perpetuity_growth)
    terminal_value = terminal_fcf / (wacc - perpetuity_growth)
    pv_terminal = terminal_value / (1 + wacc) ** 10

    # --- Enterprise Value → Equity Value ---
    enterprise_value = pv_fcf + pv_terminal
    net_debt = _safe_float(metrics.raw_info.get("totalDebt", 0)) or 0.0
    cash = _safe_float(metrics.raw_info.get("totalCash", 0)) or 0.0
    equity_value = enterprise_value - net_debt + cash
    fair_value_per_share = equity_value / float(shares)

    # Apply a conservative 15% discount to account for DCF model uncertainty:
    # DCF models are highly sensitive to growth rate and WACC assumptions, and
    # a margin-of-safety discount compensates for optimistic analyst estimates.
    fair_value_per_share *= 0.85

    result.fair_value = round(fair_value_per_share, 2)
    result.upside_pct = round((result.fair_value / result.current_price - 1) * 100, 1)
    result.downside_pct = result.upside_pct  # same metric, sign indicates direction
    result.margin_of_safety = round((1 - result.current_price / result.fair_value) * 100, 1) if result.fair_value > 0 else None

    result.dcf_assumptions = {
        "base_fcf": round(fcf / 1e9, 2),
        "growth_rate_5y": f"{growth_rate_5y*100:.1f}%",
        "terminal_growth_rate": f"{terminal_growth_rate*100:.1f}%",
        "perpetuity_growth": f"{perpetuity_growth*100:.1f}%",
        "wacc": f"{wacc*100:.1f}%",
        "beta": beta,
    }

    result.score = _compute_score(result)
    result.summary = _build_summary(result)
    logger.info("DCF valuation complete. Fair value: $%.2f, Upside: %.1f%%", result.fair_value, result.upside_pct or 0)
    return result


def _earnings_based_valuation(metrics: FundamentalMetrics) -> ValuationResult:
    """
    Fallback Graham Number / PE-based valuation when FCF is unavailable.
    """
    result = ValuationResult()
    result.current_price = metrics.current_price

    eps = _safe_float(metrics.raw_info.get("trailingEps"))
    book_value = metrics.book_value
    forward_pe = metrics.forward_pe
    pe_ratio = metrics.pe_ratio

    fair_value: Optional[float] = None

    # Graham Number: sqrt(22.5 * EPS * Book Value)
    if eps and eps > 0 and book_value and book_value > 0:
        import math
        graham = math.sqrt(22.5 * eps * book_value)
        fair_value = round(graham, 2)
        method = "Graham Number"
    elif eps and eps > 0:
        # Simple: apply sector median PE of 18
        target_pe = min(forward_pe or pe_ratio or 18, 30)
        fair_value = round(eps * target_pe, 2)
        method = f"EPS × PE({target_pe:.0f})"
    else:
        result.summary = "Insufficient data for any valuation model."
        result.score = 5.0
        return result

    result.fair_value = fair_value
    if result.current_price and result.fair_value:
        result.upside_pct = round((result.fair_value / result.current_price - 1) * 100, 1)
        result.downside_pct = result.upside_pct
        result.margin_of_safety = round((1 - result.current_price / result.fair_value) * 100, 1) if result.fair_value > 0 else None
    result.dcf_assumptions = {"method": method}
    result.score = _compute_score(result)
    result.summary = _build_summary(result)
    return result


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
        return None if f != f else f
    except (TypeError, ValueError):
        return None


def _compute_score(r: ValuationResult) -> float:
    """Score valuation attractiveness on 0-10 scale."""
    if r.upside_pct is None or r.fair_value is None:
        return 5.0

    upside = r.upside_pct
    if upside >= 50:
        return 9.5
    elif upside >= 30:
        return 8.5
    elif upside >= 15:
        return 7.0
    elif upside >= 5:
        return 6.0
    elif upside >= -5:
        return 5.0
    elif upside >= -15:
        return 4.0
    elif upside >= -30:
        return 2.5
    else:
        return 1.0


def _build_summary(r: ValuationResult) -> str:
    """Build a human-readable valuation summary."""
    lines: list[str] = ["--- Valuation Model ---"]

    price_str = f"${r.current_price:.2f}" if r.current_price else "N/A"
    fv_str = f"${r.fair_value:.2f}" if r.fair_value else "N/A"

    lines.append(f"Current Price:      {price_str}")
    lines.append(f"Fair Value (DCF):   {fv_str}")

    if r.upside_pct is not None:
        direction = "Upside" if r.upside_pct >= 0 else "Downside"
        lines.append(f"{direction}:            {abs(r.upside_pct):.1f}%")

    if r.margin_of_safety is not None:
        mos = r.margin_of_safety
        if mos > 0:
            lines.append(f"Margin of Safety:  {mos:.1f}% (price below fair value)")
        else:
            lines.append(f"Premium to FV:     {abs(mos):.1f}% (price above fair value)")

    if r.dcf_assumptions:
        lines.append("")
        lines.append("DCF Assumptions:")
        for k, v in r.dcf_assumptions.items():
            lines.append(f"  {k}: {v}")

    rating = "Undervalued" if (r.upside_pct or 0) >= 15 else ("Overvalued" if (r.upside_pct or 0) <= -10 else "Fairly Valued")
    lines.append(f"\nValuation: {rating} (Score: {r.score}/10)")
    return "\n".join(lines)
