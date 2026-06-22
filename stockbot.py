#!/usr/bin/env python3
"""
StockBot — Stock Analysis CLI.

Usage:
    python stockbot.py AAPL
    python stockbot.py AAPL --no-pdf
    python stockbot.py AAPL --output-dir my_reports
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

# Add the stockbot package directory to sys.path so we can import modules directly.
# This avoids a name collision between stockbot.py (this file) and the stockbot/ directory.
sys.path.insert(0, str(Path(__file__).parent / "stockbot"))

from modules import (  # noqa: E402
    technical,
    fundamentals,
    valuation,
    earnings,
    news,
    ai_analysis,
    report,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stockbot")


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_stock_data(symbol: str) -> tuple[yf.Ticker, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Download stock price data from Yahoo Finance.

    Returns
    -------
    tuple of (Ticker, daily_df, weekly_df, monthly_df)
    """
    logger.info("Fetching data for %s …", symbol)
    ticker = yf.Ticker(symbol)

    try:
        daily = ticker.history(period="2y", interval="1d")
        if daily.empty:
            daily = ticker.history(period="1y", interval="1d")
    except Exception as exc:
        logger.error("Failed to fetch daily data: %s", exc)
        daily = pd.DataFrame()

    try:
        weekly = ticker.history(period="5y", interval="1wk")
    except Exception as exc:
        logger.warning("Failed to fetch weekly data: %s", exc)
        weekly = pd.DataFrame()

    try:
        monthly = ticker.history(period="10y", interval="1mo")
    except Exception as exc:
        logger.warning("Failed to fetch monthly data: %s", exc)
        monthly = pd.DataFrame()

    logger.info(
        "Data fetched — daily: %d rows, weekly: %d rows, monthly: %d rows",
        len(daily), len(weekly), len(monthly),
    )
    return ticker, daily, weekly, monthly


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def analyze_stock(
    symbol: str,
    output_dir: str = "reports",
    charts_dir: str = "charts",
    generate_pdf: bool = True,
) -> report.ReportPaths:
    """
    Run the full stock analysis pipeline.

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g., "AAPL").
    output_dir : str
        Directory to save report files.
    charts_dir : str
        Directory to save chart images.
    generate_pdf : bool
        Whether to generate a PDF report.

    Returns
    -------
    ReportPaths
        Paths to the generated report files.
    """
    symbol = symbol.upper().strip()
    logger.info("=" * 60)
    logger.info("Starting analysis: %s", symbol)
    logger.info("=" * 60)

    # 1. Fetch data
    ticker, daily, weekly, monthly = fetch_stock_data(symbol)

    if daily.empty:
        logger.error("No price data found for %s. Aborting.", symbol)
        sys.exit(1)

    # 2. Technical analysis
    logger.info("Running technical analysis …")
    tech = technical.analyze(daily)

    # 3. Fundamental analysis
    logger.info("Running fundamental analysis …")
    fund = fundamentals.analyze(ticker)

    # 4. Valuation
    logger.info("Running valuation model …")
    val = valuation.analyze(fund)

    # 5. Earnings
    logger.info("Analyzing earnings …")
    earn = earnings.analyze(ticker)

    # 6. News
    logger.info("Fetching and analyzing news …")
    news_result = news.analyze(ticker, symbol)

    # 7. AI thesis
    logger.info("Generating investment thesis …")
    thesis = ai_analysis.analyze(fund, tech, val, earn, news_result)

    # 8. Generate report
    logger.info("Generating reports …")
    paths = report.generate(
        symbol=symbol,
        df_daily=daily,
        technicals=tech,
        fundamentals=fund,
        valuation=val,
        earnings=earn,
        news=news_result,
        thesis=thesis,
        output_dir=output_dir,
        charts_dir=charts_dir,
    )

    # 9. Console summary
    rating, composite = report.compute_final_rating(
        tech.score, fund.score,
        val.score if val else 5.0,
        news_result.sentiment_score,
    )
    _print_summary(symbol, fund, val, rating, composite, paths)

    return paths


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def _print_summary(
    symbol: str,
    fund: fundamentals.FundamentalMetrics,
    val: valuation.ValuationResult,
    rating: str,
    composite: float,
    paths: report.ReportPaths,
) -> None:
    """Print a concise summary to the terminal."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    rating_colors = {
        "Strong Buy": "\033[92m",   # bright green
        "Buy": "\033[32m",           # green
        "Hold": "\033[93m",          # yellow
        "Sell": "\033[91m",          # red
        "Strong Sell": "\033[31m",   # dark red
    }
    color = rating_colors.get(rating, "")

    print()
    print(f"{BOLD}{'='*50}{RESET}")
    print(f"{BOLD}  {symbol} Investment Report{RESET}")
    print(f"{'='*50}")
    print(f"  Company:        {fund.company_name}")
    print(f"  Sector:         {fund.sector}")
    print()
    price_str = f"${fund.current_price:.2f}" if fund.current_price else "N/A"
    fv_str = f"${val.fair_value:.2f}" if val and val.fair_value else "N/A"
    upside_str = f"{val.upside_pct:+.1f}%" if val and val.upside_pct is not None else "N/A"

    print(f"  Current Price:  {BOLD}{price_str}{RESET}")
    print(f"  Fair Value:     {fv_str}")
    print(f"  Upside/Down:    {upside_str}")
    print()
    print(f"  Rating:         {color}{BOLD}{rating}{RESET}")
    print(f"  Composite:      {composite}/10")
    print(f"{'='*50}")
    print()
    print(f"  Reports saved to:")
    if paths.markdown:
        print(f"    📄 Markdown : {paths.markdown}")
    if paths.html:
        print(f"    🌐 HTML     : {paths.html}")
    if paths.pdf:
        print(f"    📑 PDF      : {paths.pdf}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="stockbot",
        description="Stock analysis bot — generates a comprehensive investment report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("symbol", help="Ticker symbol to analyze (e.g., AAPL)")
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory to save report files",
    )
    parser.add_argument(
        "--charts-dir",
        default="charts",
        help="Directory to save chart images",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip PDF report generation",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    start = time.time()
    paths = analyze_stock(
        symbol=args.symbol,
        output_dir=args.output_dir,
        charts_dir=args.charts_dir,
        generate_pdf=not args.no_pdf,
    )
    elapsed = time.time() - start
    logger.info("Analysis complete in %.1fs", elapsed)


if __name__ == "__main__":
    main()
