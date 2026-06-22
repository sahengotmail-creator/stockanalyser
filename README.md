# StockBot — AI-Powered Stock Analysis Bot

A complete stock research assistant that analyzes any publicly traded stock and generates comprehensive investment reports.

## Features

- **Technical Analysis**: SMA 20/50/200, EMA 20/50, RSI, MACD, Bollinger Bands; Golden/Death Cross detection
- **Fundamental Analysis**: Market cap, revenue, PE/PEG/P/B ratios, ROE, margins, FCF
- **Valuation Model**: Discounted Cash Flow (DCF), Fair Value, Margin of Safety
- **Earnings Analysis**: Last 4 quarters, revenue/EPS growth, earnings surprises
- **News Analysis**: Sentiment scoring, bullish/bearish developments, risks, catalysts
- **AI Investment Thesis**: Business overview, bull/bear case, risk factors, 5-year outlook (OpenAI API or rule-based fallback)
- **Report Generation**: Markdown, HTML, and PDF reports with charts

## Project Structure

```
stockbot/
├── data/
├── reports/
├── charts/
├── modules/
│   ├── technical.py       # SMA, EMA, RSI, MACD, Bollinger Bands
│   ├── fundamentals.py    # Financial metrics + scoring
│   ├── valuation.py       # DCF model
│   ├── earnings.py        # Quarterly earnings analysis
│   ├── news.py            # News fetching + sentiment
│   ├── ai_analysis.py     # LLM investment thesis
│   └── report.py          # Report generation + charts
│
├── stockbot.py            # CLI entry point
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Analyze Apple Inc.
python stockbot.py AAPL

# Custom output directory
python stockbot.py TSLA --output-dir my_reports

# Skip PDF generation
python stockbot.py MSFT --no-pdf

# Verbose debug output
python stockbot.py NVDA --log-level DEBUG
```

### Sample Output

```
==================================================
  AAPL Investment Report
==================================================
  Company:        Apple Inc.
  Sector:         Technology

  Current Price:  $195.50
  Fair Value:     $210.30
  Upside/Down:    +7.6%

  Rating:         Buy
  Composite:      7.1/10
==================================================

  Reports saved to:
    📄 Markdown : reports/AAPL_20241201_120000.md
    🌐 HTML     : reports/AAPL_20241201_120000.html
    📑 PDF      : reports/AAPL_20241201_120000.pdf
```

## OpenAI Integration (Optional)

Set the `OPENAI_API_KEY` environment variable to enable LLM-powered investment theses:

```bash
export OPENAI_API_KEY=your_api_key_here
python stockbot.py AAPL
```

Without an API key, a comprehensive rule-based thesis is generated automatically.

## Rating System

| Rating | Score Range |
|--------|-------------|
| Strong Buy | 7.5 – 10.0 |
| Buy | 6.5 – 7.4 |
| Hold | 4.5 – 6.4 |
| Sell | 3.0 – 4.4 |
| Strong Sell | 0.0 – 2.9 |

**Score Weights:**
- Technical Analysis: 25%
- Fundamental Analysis: 35%
- Valuation: 25%
- News Sentiment: 15%

## Disclaimer

This tool is for informational purposes only and does not constitute financial advice.