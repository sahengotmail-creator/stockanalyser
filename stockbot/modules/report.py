"""
Report Generation Module.

Creates Markdown, HTML, and PDF reports with embedded charts.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from .technical import TechnicalIndicators
from .fundamentals import FundamentalMetrics
from .valuation import ValuationResult
from .earnings import EarningsAnalysis
from .news import NewsAnalysis
from .ai_analysis import InvestmentThesis

logger = logging.getLogger(__name__)


@dataclass
class ReportPaths:
    """Paths of generated report files."""

    markdown: str = ""
    html: str = ""
    pdf: str = ""
    chart_price: str = ""
    chart_indicators: str = ""
    chart_fundamentals: str = ""


RATING_LABELS = {
    "Strong Buy": ("strong_buy", "🟢"),
    "Buy": ("buy", "🟩"),
    "Hold": ("hold", "🟡"),
    "Sell": ("sell", "🟠"),
    "Strong Sell": ("strong_sell", "🔴"),
}


def compute_final_rating(
    technical_score: float,
    fundamental_score: float,
    valuation_score: float,
    news_score: float,
) -> tuple[str, float]:
    """
    Compute weighted final rating.

    Weights:
      Technical:   25%
      Fundamental: 35%
      Valuation:   25%
      News:        15%

    Returns
    -------
    tuple[str, float]
        (rating_label, composite_score_0_to_10)
    """
    composite = (
        technical_score * 0.25 +
        fundamental_score * 0.35 +
        valuation_score * 0.25 +
        news_score * 0.15
    )
    composite = round(composite, 2)

    if composite >= 7.5:
        label = "Strong Buy"
    elif composite >= 6.5:
        label = "Buy"
    elif composite >= 4.5:
        label = "Hold"
    elif composite >= 3.0:
        label = "Sell"
    else:
        label = "Strong Sell"

    return label, composite


def generate(
    symbol: str,
    df_daily: pd.DataFrame,
    technicals: TechnicalIndicators,
    fundamentals: FundamentalMetrics,
    valuation: ValuationResult,
    earnings: EarningsAnalysis,
    news: NewsAnalysis,
    thesis: InvestmentThesis,
    output_dir: str = "reports",
    charts_dir: str = "charts",
) -> ReportPaths:
    """
    Generate Markdown, HTML, and PDF reports for a stock.

    Parameters
    ----------
    symbol : str
        Stock ticker symbol.
    df_daily : pd.DataFrame
        Daily OHLCV data used for charting.
    technicals, fundamentals, valuation, earnings, news, thesis :
        Analysis results from each module.
    output_dir : str
        Directory to write report files.
    charts_dir : str
        Directory to write chart image files.

    Returns
    -------
    ReportPaths
        Paths to the generated files.
    """
    paths = ReportPaths()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(charts_dir).mkdir(parents=True, exist_ok=True)

    rating, composite_score = compute_final_rating(
        technicals.score,
        fundamentals.score,
        valuation.score if valuation else 5.0,
        news.sentiment_score,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{symbol}_{timestamp}"

    # --- Charts ---
    paths.chart_price = _plot_price_chart(symbol, df_daily, technicals, charts_dir, base_name)
    paths.chart_indicators = _plot_indicators_chart(symbol, df_daily, technicals, charts_dir, base_name)
    paths.chart_fundamentals = _plot_fundamentals_chart(symbol, fundamentals, earnings, charts_dir, base_name)

    # --- Markdown ---
    md_path = os.path.join(output_dir, f"{base_name}.md")
    md_content = _build_markdown(
        symbol, rating, composite_score,
        technicals, fundamentals, valuation, earnings, news, thesis,
        paths,
    )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    paths.markdown = md_path

    # --- HTML ---
    html_path = os.path.join(output_dir, f"{base_name}.html")
    html_content = _build_html(md_content, symbol, rating, composite_score, paths)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    paths.html = html_path

    # --- PDF ---
    pdf_path = os.path.join(output_dir, f"{base_name}.pdf")
    try:
        _build_pdf(
            pdf_path, symbol, rating, composite_score,
            technicals, fundamentals, valuation, earnings, news, thesis,
            paths,
        )
        paths.pdf = pdf_path
    except Exception as exc:
        logger.warning("PDF generation failed: %s", exc)

    logger.info("Reports generated: %s", output_dir)
    return paths


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _plot_price_chart(
    symbol: str,
    df: pd.DataFrame,
    ind: TechnicalIndicators,
    charts_dir: str,
    base_name: str,
) -> str:
    """Plot price with SMA/EMA overlays."""
    if df.empty or "Close" not in df.columns:
        return ""

    fig, ax = plt.subplots(figsize=(14, 6))
    df_plot = df.tail(252)  # ~1 year

    close = df_plot["Close"]
    dates = df_plot.index

    ax.plot(dates, close, label="Price", color="black", linewidth=1.5, zorder=5)

    # Moving averages
    if len(close) >= 20:
        sma20 = close.rolling(20).mean()
        ax.plot(dates, sma20, label="SMA 20", color="blue", linewidth=1, alpha=0.8)
    if len(close) >= 50:
        sma50 = close.rolling(50).mean()
        ax.plot(dates, sma50, label="SMA 50", color="orange", linewidth=1, alpha=0.8)
    if len(close) >= 200:
        sma200 = close.rolling(200).mean()
        ax.plot(dates, sma200, label="SMA 200", color="red", linewidth=1, alpha=0.8)

    # Bollinger Bands
    if len(close) >= 20:
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        ax.fill_between(dates, bb_upper, bb_lower, alpha=0.1, color="purple", label="Bollinger Bands")

    ax.set_title(f"{symbol} — Price Chart with Indicators (1Y)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(charts_dir, f"{base_name}_price.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_indicators_chart(
    symbol: str,
    df: pd.DataFrame,
    ind: TechnicalIndicators,
    charts_dir: str,
    base_name: str,
) -> str:
    """Plot RSI and MACD indicators."""
    if df.empty or "Close" not in df.columns:
        return ""

    close = df["Close"].dropna().tail(252)
    dates = close.index

    if len(close) < 26:
        return ""

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(com=13, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # RSI
    ax1.plot(dates, rsi, color="purple", linewidth=1.5, label="RSI(14)")
    ax1.axhline(70, color="red", linestyle="--", linewidth=1, alpha=0.7, label="Overbought (70)")
    ax1.axhline(30, color="green", linestyle="--", linewidth=1, alpha=0.7, label="Oversold (30)")
    ax1.fill_between(dates, rsi, 70, where=(rsi >= 70), alpha=0.2, color="red")
    ax1.fill_between(dates, rsi, 30, where=(rsi <= 30), alpha=0.2, color="green")
    ax1.set_ylabel("RSI")
    ax1.set_ylim(0, 100)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_title(f"{symbol} — Technical Indicators (1Y)", fontsize=13, fontweight="bold")

    # MACD
    colors = ["green" if h >= 0 else "red" for h in histogram]
    ax2.bar(dates, histogram, color=colors, alpha=0.6, label="MACD Histogram")
    ax2.plot(dates, macd_line, color="blue", linewidth=1.2, label="MACD")
    ax2.plot(dates, signal_line, color="orange", linewidth=1.2, label="Signal")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("MACD")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30)
    plt.tight_layout()

    path = os.path.join(charts_dir, f"{base_name}_indicators.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_fundamentals_chart(
    symbol: str,
    fundamentals: FundamentalMetrics,
    earnings: EarningsAnalysis,
    charts_dir: str,
    base_name: str,
) -> str:
    """Plot fundamental metrics as a bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Score gauge
    scores = {
        "Technical": 0.0,  # placeholder; will be passed from caller or omitted
        "Fundamental": fundamentals.score,
        "Valuation": 5.0,  # default
        "News Sentiment": 5.0,
    }
    bars = axes[0].barh(
        list(scores.keys()),
        list(scores.values()),
        color=["#4CAF50" if v >= 6 else "#FF9800" if v >= 4 else "#F44336" for v in scores.values()],
    )
    axes[0].set_xlim(0, 10)
    axes[0].set_xlabel("Score (0-10)")
    axes[0].set_title("Score Summary", fontweight="bold")
    axes[0].axvline(5, color="gray", linestyle="--", alpha=0.5)
    for bar, val in zip(bars, scores.values()):
        axes[0].text(val + 0.1, bar.get_y() + bar.get_height() / 2, f"{val:.1f}", va="center", fontsize=10)

    # Quarterly revenue
    quarters = earnings.quarters
    if quarters and any(q.revenue is not None for q in quarters):
        rev_labels = [q.period[-7:] for q in quarters]
        rev_vals = [q.revenue / 1e9 if q.revenue else 0 for q in quarters]
        axes[1].bar(rev_labels, rev_vals, color="#2196F3", alpha=0.8)
        axes[1].set_title("Quarterly Revenue ($B)", fontweight="bold")
        axes[1].set_ylabel("Revenue ($B)")
        axes[1].tick_params(axis="x", rotation=30)
        axes[1].grid(axis="y", alpha=0.3)
    else:
        # Margin bars instead
        margin_labels, margin_vals, colors = [], [], []
        if fundamentals.gross_margin:
            margin_labels.append("Gross Margin")
            margin_vals.append(fundamentals.gross_margin * 100)
            colors.append("#4CAF50")
        if fundamentals.operating_margin:
            margin_labels.append("Operating Margin")
            margin_vals.append(fundamentals.operating_margin * 100)
            colors.append("#2196F3")
        if fundamentals.roe:
            margin_labels.append("ROE")
            margin_vals.append(fundamentals.roe * 100)
            colors.append("#9C27B0")
        if margin_labels:
            axes[1].bar(margin_labels, margin_vals, color=colors, alpha=0.8)
            axes[1].set_title(f"{symbol} — Key Margins & ROE (%)", fontweight="bold")
            axes[1].set_ylabel("Percentage (%)")
            axes[1].grid(axis="y", alpha=0.3)
        else:
            axes[1].set_visible(False)

    fig.suptitle(f"{symbol} — Fundamental Overview", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(charts_dir, f"{base_name}_fundamentals.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _build_markdown(
    symbol: str,
    rating: str,
    composite_score: float,
    technicals: TechnicalIndicators,
    fundamentals: FundamentalMetrics,
    valuation: ValuationResult,
    earnings: EarningsAnalysis,
    news: NewsAnalysis,
    thesis: InvestmentThesis,
    paths: ReportPaths,
) -> str:
    """Build the full Markdown report string."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    emoji = RATING_LABELS.get(rating, ("", ""))[1]

    price_str = f"${fundamentals.current_price:.2f}" if fundamentals.current_price else "N/A"
    fv_str = f"${valuation.fair_value:.2f}" if valuation and valuation.fair_value else "N/A"
    upside_str = f"{valuation.upside_pct:+.1f}%" if valuation and valuation.upside_pct is not None else "N/A"

    lines: list[str] = [
        f"# {symbol} Investment Report",
        f"*Generated: {now}*",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| **Symbol** | {symbol} |",
        f"| **Company** | {fundamentals.company_name} |",
        f"| **Current Price** | {price_str} |",
        f"| **Fair Value** | {fv_str} |",
        f"| **Upside / Downside** | {upside_str} |",
        f"| **Final Rating** | {emoji} **{rating}** |",
        f"| **Composite Score** | {composite_score}/10 |",
        "",
        "---",
        "",
        "## Score Breakdown",
        "",
        f"| Component | Score | Weight |",
        f"|-----------|-------|--------|",
        f"| Technical Analysis | {technicals.score}/10 | 25% |",
        f"| Fundamental Analysis | {fundamentals.score}/10 | 35% |",
        f"| Valuation | {valuation.score if valuation else 5.0}/10 | 25% |",
        f"| News Sentiment | {news.sentiment_score}/10 | 15% |",
        f"| **Composite** | **{composite_score}/10** | 100% |",
        "",
        "---",
        "",
    ]

    # Charts
    report_dir = os.path.dirname(paths.markdown) if paths.markdown else "."

    def _rel(chart_path: str) -> str:
        return os.path.relpath(chart_path, start=report_dir)

    if paths.chart_price:
        lines += ["## Price Chart", "", f"![Price Chart]({_rel(paths.chart_price)})", ""]
    if paths.chart_indicators:
        lines += ["## Technical Indicators", "", f"![Indicators]({_rel(paths.chart_indicators)})", ""]
    if paths.chart_fundamentals:
        lines += ["## Fundamental Overview", "", f"![Fundamentals]({_rel(paths.chart_fundamentals)})", ""]

    # Technical Analysis
    lines += [
        "---",
        "",
        "## Technical Analysis",
        "",
        f"```",
        technicals.summary or "No technical summary.",
        "```",
        "",
    ]

    # Fundamental Analysis
    lines += [
        "---",
        "",
        "## Fundamental Analysis",
        "",
        f"```",
        fundamentals.summary or "No fundamental summary.",
        "```",
        "",
    ]

    # Valuation
    lines += [
        "---",
        "",
        "## Valuation",
        "",
        f"```",
        valuation.summary if valuation else "No valuation data.",
        "```",
        "",
    ]

    # Earnings
    lines += [
        "---",
        "",
        "## Earnings Analysis",
        "",
        f"```",
        earnings.summary or "No earnings data.",
        "```",
        "",
    ]

    # News
    lines += [
        "---",
        "",
        "## News & Sentiment",
        "",
        f"```",
        news.summary or "No news available.",
        "```",
        "",
    ]

    # AI Thesis
    lines += [
        "---",
        "",
        "## AI Investment Thesis",
        "",
        thesis.full_thesis or "No investment thesis generated.",
        "",
        "---",
        "",
        f"*Disclaimer: This report is for informational purposes only and does not constitute financial advice.*",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _build_html(
    markdown_content: str,
    symbol: str,
    rating: str,
    composite_score: float,
    paths: ReportPaths,
) -> str:
    """Convert Markdown report to a styled HTML page."""
    try:
        import re

        def md_to_html_basic(md: str) -> str:
            """Minimal Markdown → HTML converter."""
            html = md
            # Headings
            html = re.sub(r"^### (.+)$", r"<h3>\1</h3>", html, flags=re.MULTILINE)
            html = re.sub(r"^## (.+)$", r"<h2>\1</h2>", html, flags=re.MULTILINE)
            html = re.sub(r"^# (.+)$", r"<h1>\1</h1>", html, flags=re.MULTILINE)
            # Bold
            html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
            # Italic
            html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html)
            # Code blocks
            html = re.sub(r"```\n?(.*?)```", r"<pre><code>\1</code></pre>", html, flags=re.DOTALL)
            # Inline code
            html = re.sub(r"`(.+?)`", r"<code>\1</code>", html)
            # Tables: basic
            lines_out = []
            in_table = False
            for line in html.split("\n"):
                if line.startswith("|") and line.endswith("|"):
                    if not in_table:
                        lines_out.append("<table border='1' cellpadding='4' cellspacing='0'>")
                        in_table = True
                    if re.match(r"^\|[-|: ]+\|$", line):
                        continue  # skip separator rows
                    cells = [c.strip() for c in line.strip("|").split("|")]
                    lines_out.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
                else:
                    if in_table:
                        lines_out.append("</table>")
                        in_table = False
                    lines_out.append(line)
            if in_table:
                lines_out.append("</table>")
            html = "\n".join(lines_out)
            # Images
            html = re.sub(r"!\[(.+?)\]\((.+?)\)", r'<img src="\2" alt="\1" style="max-width:100%;margin:10px 0;">', html)
            # Horizontal rules: only convert bare `---` lines (not `--- Title ---` inside code blocks)
            html = re.sub(r"^---$", "<hr>", html, flags=re.MULTILINE)
            # Paragraphs
            html = re.sub(r"\n\n", "</p><p>", html)
            html = "<p>" + html + "</p>"
            return html

        body = md_to_html_basic(markdown_content)
    except Exception:
        body = f"<pre>{markdown_content}</pre>"

    rating_color = {
        "Strong Buy": "#27ae60",
        "Buy": "#2ecc71",
        "Hold": "#f39c12",
        "Sell": "#e67e22",
        "Strong Sell": "#c0392b",
    }.get(rating, "#7f8c8d")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{symbol} Investment Report</title>
  <style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; background: #f8f9fa; color: #212529; }}
    h1 {{ color: #1a1a2e; border-bottom: 3px solid #0d6efd; padding-bottom: 10px; }}
    h2 {{ color: #16213e; border-left: 4px solid #0d6efd; padding-left: 12px; }}
    h3 {{ color: #0f3460; }}
    table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
    th, td {{ padding: 8px 12px; text-align: left; border: 1px solid #dee2e6; }}
    th {{ background: #e9ecef; }}
    tr:nth-child(even) {{ background: #f8f9fa; }}
    pre, code {{ background: #1e1e2e; color: #cdd6f4; padding: 12px; border-radius: 6px; overflow-x: auto; white-space: pre-wrap; }}
    code {{ padding: 2px 6px; font-size: 0.9em; }}
    img {{ border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,.15); }}
    hr {{ border: none; border-top: 1px solid #dee2e6; margin: 24px 0; }}
    .rating-badge {{ display: inline-block; background: {rating_color}; color: white; padding: 6px 18px; border-radius: 20px; font-weight: bold; font-size: 1.2em; }}
    .score-bar {{ height: 16px; background: linear-gradient(to right, #dc3545, #ffc107, #28a745); border-radius: 8px; position: relative; }}
    .disclaimer {{ font-size: 0.85em; color: #6c757d; font-style: italic; }}
  </style>
</head>
<body>
  <div style="background:white; padding:20px; border-radius:12px; box-shadow:0 4px 16px rgba(0,0,0,.1); margin-bottom:24px;">
    <h1 style="margin:0 0 8px 0;">{symbol} Investment Report</h1>
    <span class="rating-badge">{rating}</span>
    <span style="margin-left:16px; font-size:1.1em;">Composite Score: <strong>{composite_score}/10</strong></span>
  </div>
  {body}
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------

def _build_pdf(
    pdf_path: str,
    symbol: str,
    rating: str,
    composite_score: float,
    technicals: TechnicalIndicators,
    fundamentals: FundamentalMetrics,
    valuation: ValuationResult,
    earnings: EarningsAnalysis,
    news: NewsAnalysis,
    thesis: InvestmentThesis,
    paths: ReportPaths,
) -> None:
    """Build a PDF report using ReportLab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image, HRFlowable,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    doc = SimpleDocTemplate(pdf_path, pagesize=A4, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle("Title", parent=styles["Title"], fontSize=22, spaceAfter=6)
    h2_style = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=14, spaceBefore=12, spaceAfter=6, textColor=colors.HexColor("#16213e"))
    body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=14)
    code_style = ParagraphStyle("Code", parent=styles["Code"], fontSize=8, leading=12, backColor=colors.HexColor("#1e1e2e"), textColor=colors.HexColor("#cdd6f4"))

    rating_color = {
        "Strong Buy": colors.HexColor("#27ae60"),
        "Buy": colors.HexColor("#2ecc71"),
        "Hold": colors.HexColor("#f39c12"),
        "Sell": colors.HexColor("#e67e22"),
        "Strong Sell": colors.HexColor("#c0392b"),
    }.get(rating, colors.grey)

    now = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    price_str = f"${fundamentals.current_price:.2f}" if fundamentals.current_price else "N/A"
    fv_str = f"${valuation.fair_value:.2f}" if valuation and valuation.fair_value else "N/A"
    upside_str = f"{valuation.upside_pct:+.1f}%" if valuation and valuation.upside_pct is not None else "N/A"

    story.append(Paragraph(f"{symbol} Investment Report", title_style))
    story.append(Paragraph(f"Generated: {now}", styles["Normal"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0d6efd")))
    story.append(Spacer(1, 0.3 * cm))

    # Summary table
    summary_data = [
        ["Symbol", symbol],
        ["Company", fundamentals.company_name],
        ["Current Price", price_str],
        ["Fair Value", fv_str],
        ["Upside / Downside", upside_str],
        ["Rating", rating],
        ["Composite Score", f"{composite_score}/10"],
    ]
    t = Table(summary_data, colWidths=[5 * cm, 11 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e9ecef")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
        ("BACKGROUND", (1, 5), (1, 5), rating_color),
        ("TEXTCOLOR", (1, 5), (1, 5), colors.white),
        ("FONTNAME", (1, 5), (1, 5), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.4 * cm))

    # Score breakdown
    story.append(Paragraph("Score Breakdown", h2_style))
    score_data = [
        ["Component", "Score", "Weight"],
        ["Technical Analysis", f"{technicals.score}/10", "25%"],
        ["Fundamental Analysis", f"{fundamentals.score}/10", "35%"],
        ["Valuation", f"{valuation.score if valuation else 5.0}/10", "25%"],
        ["News Sentiment", f"{news.sentiment_score}/10", "15%"],
        ["Composite", f"{composite_score}/10", "100%"],
    ]
    st = Table(score_data, colWidths=[8 * cm, 4 * cm, 4 * cm])
    st.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d6efd")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, colors.HexColor("#f8f9fa")]),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#e9ecef")),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
    ]))
    story.append(st)
    story.append(Spacer(1, 0.4 * cm))

    # Charts
    for chart_path, title in [
        (paths.chart_price, "Price Chart"),
        (paths.chart_indicators, "Technical Indicators"),
        (paths.chart_fundamentals, "Fundamental Overview"),
    ]:
        if chart_path and os.path.exists(chart_path):
            story.append(Paragraph(title, h2_style))
            img = Image(chart_path, width=16 * cm, height=8 * cm)
            story.append(img)
            story.append(Spacer(1, 0.3 * cm))

    def add_section(title: str, content: str) -> None:
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#dee2e6")))
        story.append(Paragraph(title, h2_style))
        if content:
            for line in content.split("\n"):
                if line.strip():
                    story.append(Paragraph(line, body_style))
        story.append(Spacer(1, 0.3 * cm))

    add_section("Technical Analysis", technicals.summary or "N/A")
    add_section("Fundamental Analysis", fundamentals.summary or "N/A")
    add_section("Valuation", valuation.summary if valuation else "N/A")
    add_section("Earnings Analysis", earnings.summary or "N/A")
    add_section("News & Sentiment", news.summary or "N/A")
    add_section("Investment Thesis", thesis.full_thesis or "N/A")

    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#dee2e6")))
    story.append(Paragraph(
        "Disclaimer: This report is for informational purposes only and does not constitute financial advice.",
        ParagraphStyle("Disclaimer", parent=styles["Normal"], fontSize=8, textColor=colors.grey, spaceBefore=6),
    ))

    doc.build(story)
    logger.info("PDF report written to %s", pdf_path)
