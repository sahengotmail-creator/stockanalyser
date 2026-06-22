"""
Technical Analysis Module.

Calculates indicators: SMA, EMA, RSI, MACD, Bollinger Bands.
Detects signals: Golden Cross, Death Cross, Overbought, Oversold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TechnicalIndicators:
    """Container for all computed technical indicators."""

    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_20: Optional[float] = None
    ema_50: Optional[float] = None
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    golden_cross: bool = False
    death_cross: bool = False
    overbought: bool = False
    oversold: bool = False
    score: float = 0.0
    summary: str = ""
    signals: list[str] = field(default_factory=list)


def _safe_last(series: pd.Series) -> Optional[float]:
    """Return the last non-NaN value of a series, or None."""
    valid = series.dropna()
    if valid.empty:
        return None
    return float(valid.iloc[-1])


def compute_sma(close: pd.Series, window: int) -> pd.Series:
    """Compute Simple Moving Average."""
    return close.rolling(window=window).mean()


def compute_ema(close: pd.Series, window: int) -> pd.Series:
    """Compute Exponential Moving Average."""
    return close.ewm(span=window, adjust=False).mean()


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD line, signal line, and histogram."""
    ema_fast = compute_ema(close, fast)
    ema_slow = compute_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute Bollinger Bands (upper, middle, lower)."""
    middle = compute_sma(close, window)
    std = close.rolling(window=window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def analyze(df: pd.DataFrame) -> TechnicalIndicators:
    """
    Run full technical analysis on OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least a 'Close' column.

    Returns
    -------
    TechnicalIndicators
        Computed indicators, signals, score, and narrative summary.
    """
    indicators = TechnicalIndicators()

    if df.empty or "Close" not in df.columns:
        logger.warning("Empty or invalid DataFrame passed to technical analysis.")
        indicators.summary = "Insufficient data for technical analysis."
        return indicators

    close = df["Close"].dropna()
    if len(close) < 30:
        logger.warning("Fewer than 30 data points; some indicators may be unreliable.")

    # --- Moving Averages ---
    sma_20 = compute_sma(close, 20)
    sma_50 = compute_sma(close, 50)
    sma_200 = compute_sma(close, 200)
    ema_20 = compute_ema(close, 20)
    ema_50 = compute_ema(close, 50)

    indicators.sma_20 = _safe_last(sma_20)
    indicators.sma_50 = _safe_last(sma_50)
    indicators.sma_200 = _safe_last(sma_200)
    indicators.ema_20 = _safe_last(ema_20)
    indicators.ema_50 = _safe_last(ema_50)

    # --- RSI ---
    rsi_series = compute_rsi(close)
    indicators.rsi = _safe_last(rsi_series)

    # --- MACD ---
    macd_line, signal_line, histogram = compute_macd(close)
    indicators.macd = _safe_last(macd_line)
    indicators.macd_signal = _safe_last(signal_line)
    indicators.macd_hist = _safe_last(histogram)

    # --- Bollinger Bands ---
    bb_upper, bb_middle, bb_lower = compute_bollinger_bands(close)
    indicators.bb_upper = _safe_last(bb_upper)
    indicators.bb_middle = _safe_last(bb_middle)
    indicators.bb_lower = _safe_last(bb_lower)

    current_price = float(close.iloc[-1])
    signals: list[str] = []
    score_components: list[float] = []

    # --- Golden / Death Cross ---
    if indicators.sma_50 is not None and indicators.sma_200 is not None:
        if len(sma_50.dropna()) >= 2 and len(sma_200.dropna()) >= 2:
            prev_sma50 = float(sma_50.dropna().iloc[-2])
            prev_sma200 = float(sma_200.dropna().iloc[-2])
            cur_sma50 = indicators.sma_50
            cur_sma200 = indicators.sma_200
            if prev_sma50 <= prev_sma200 and cur_sma50 > cur_sma200:
                indicators.golden_cross = True
                signals.append("Golden Cross detected (SMA50 crossed above SMA200) — Bullish")
                score_components.append(1.0)
            elif prev_sma50 >= prev_sma200 and cur_sma50 < cur_sma200:
                indicators.death_cross = True
                signals.append("Death Cross detected (SMA50 crossed below SMA200) — Bearish")
                score_components.append(-1.0)
            elif cur_sma50 > cur_sma200:
                signals.append("Price above SMA200 (SMA50 > SMA200) — Bullish trend")
                score_components.append(0.5)
            else:
                signals.append("Price below SMA200 (SMA50 < SMA200) — Bearish trend")
                score_components.append(-0.5)

    # --- RSI Signals ---
    if indicators.rsi is not None:
        if indicators.rsi >= 70:
            indicators.overbought = True
            signals.append(f"RSI {indicators.rsi:.1f} — Overbought (≥70)")
            score_components.append(-0.75)
        elif indicators.rsi <= 30:
            indicators.oversold = True
            signals.append(f"RSI {indicators.rsi:.1f} — Oversold (≤30)")
            score_components.append(0.75)
        elif indicators.rsi >= 55:
            signals.append(f"RSI {indicators.rsi:.1f} — Bullish momentum")
            score_components.append(0.25)
        elif indicators.rsi <= 45:
            signals.append(f"RSI {indicators.rsi:.1f} — Bearish momentum")
            score_components.append(-0.25)
        else:
            signals.append(f"RSI {indicators.rsi:.1f} — Neutral")
            score_components.append(0.0)

    # --- MACD Signal ---
    if indicators.macd_hist is not None:
        if indicators.macd_hist > 0:
            signals.append(f"MACD histogram positive ({indicators.macd_hist:.4f}) — Bullish")
            score_components.append(0.5)
        else:
            signals.append(f"MACD histogram negative ({indicators.macd_hist:.4f}) — Bearish")
            score_components.append(-0.5)

    # --- Bollinger Band Position ---
    if indicators.bb_upper is not None and indicators.bb_lower is not None:
        bb_range = indicators.bb_upper - indicators.bb_lower
        if bb_range > 0:
            pct_b = (current_price - indicators.bb_lower) / bb_range
            if pct_b > 1.0:
                signals.append("Price above upper Bollinger Band — potential reversal down")
                score_components.append(-0.25)
            elif pct_b < 0.0:
                signals.append("Price below lower Bollinger Band — potential reversal up")
                score_components.append(0.25)
            elif pct_b > 0.8:
                signals.append("Price near upper Bollinger Band — mildly overbought")
                score_components.append(-0.1)
            elif pct_b < 0.2:
                signals.append("Price near lower Bollinger Band — mildly oversold")
                score_components.append(0.1)
            else:
                signals.append(f"%B = {pct_b:.2f} — Price in mid-Bollinger Band range")
                score_components.append(0.0)

    # --- Price vs Moving Averages ---
    if indicators.sma_20 is not None:
        if current_price > indicators.sma_20:
            score_components.append(0.2)
        else:
            score_components.append(-0.2)
    if indicators.ema_20 is not None:
        if current_price > indicators.ema_20:
            score_components.append(0.2)
        else:
            score_components.append(-0.2)

    # --- Composite Score: normalize to [-10, 10] scale → map to 0-10 ---
    raw_score = sum(score_components)
    max_possible = len(score_components) * 1.0 if score_components else 1.0
    normalized = raw_score / max_possible  # -1 to 1
    indicators.score = round((normalized + 1) / 2 * 10, 2)  # 0 to 10

    indicators.signals = signals
    indicators.summary = _build_summary(indicators, current_price)
    logger.info("Technical analysis complete. Score: %.2f", indicators.score)
    return indicators


def _build_summary(ind: TechnicalIndicators, price: float) -> str:
    """Build a human-readable technical analysis narrative."""
    lines: list[str] = [f"Current Price: ${price:.2f}"]

    if ind.sma_20:
        lines.append(f"SMA20: ${ind.sma_20:.2f} | SMA50: ${ind.sma_50:.2f} | SMA200: ${ind.sma_200:.2f}")
    if ind.ema_20:
        lines.append(f"EMA20: ${ind.ema_20:.2f} | EMA50: ${ind.ema_50:.2f}")
    if ind.rsi:
        lines.append(f"RSI(14): {ind.rsi:.1f}")
    if ind.macd is not None:
        lines.append(f"MACD: {ind.macd:.4f} | Signal: {ind.macd_signal:.4f} | Hist: {ind.macd_hist:.4f}")
    if ind.bb_upper:
        lines.append(f"Bollinger Bands — Upper: ${ind.bb_upper:.2f} | Mid: ${ind.bb_middle:.2f} | Lower: ${ind.bb_lower:.2f}")

    lines.append("")
    lines.append("Signals:")
    for sig in ind.signals:
        lines.append(f"  • {sig}")

    trend = "Bullish" if ind.score >= 6 else ("Bearish" if ind.score <= 4 else "Neutral")
    lines.append(f"\nTechnical Outlook: {trend} (Score: {ind.score}/10)")

    return "\n".join(lines)
