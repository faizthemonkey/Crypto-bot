# Futures Trading Bot - Complete Documentation

## Version 2.0 - Enhanced GUI with Strategy Explanations & Trade Markers

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage Guide](#usage-guide)
5. [Strategies Explained](#strategies-explained)
6. [Settings Configuration](#settings-configuration)
7. [Chart Features](#chart-features)
8. [Saved Backtests](#saved-backtests)
9. [Live Paper Trading](#live-paper-trading)
10. [Technical Architecture](#technical-architecture)
11. [Troubleshooting](#troubleshooting)

---

## Overview

The Futures Trading Bot is a comprehensive cryptocurrency futures trading simulation platform built with Streamlit. It connects to the **Binance Futures Public API** (no authentication required) to fetch real market data and allows users to:

- Backtest trading strategies on historical data
- Paper trade with live market data (no real money)
- Analyze performance with detailed metrics
- Save and compare backtest results

---

## Features

### ✅ Version 2.0 Enhancements

| Feature | Description |
|---------|-------------|
| **Strategy Explanations** | Each strategy has detailed documentation explaining entry/exit logic |
| **Trade Markers on Chart** | Clear visualization of entry points (triangles) and exit points (X markers) |
| **Entry-Exit Lines** | Dotted lines connecting entry to exit showing trade progression |
| **Color-Coded Results** | Green for profitable trades, Red for losses |
| **Improved Contrast** | High-contrast metrics with dark theme styling |
| **Saved Backtests** | Automatic saving and retrieval of backtest results |
| **Mouse Controls** | Scroll zoom, pan, and interactive chart navigation |
| **Applied Settings Display** | Shows exactly which settings were used for each backtest |
| **Clear Results Button** | Clears chart before new backtest |

---

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Steps

```bash
# 1. Navigate to project directory
cd "c:\Users\Faiz\OneDrive\Desktop\Trading bot\CascadeProjects\windsurf-project"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
streamlit run streamlit_app.py
```

### Dependencies
```
streamlit>=1.28.0
plotly>=5.18.0
pandas>=2.1.0
numpy>=1.24.3
websocket-client>=1.6.4
requests>=2.31.0
python-binance>=1.0.19
```

---

## Usage Guide

### Starting the Application

**Option 1: Command Line**
```bash
streamlit run streamlit_app.py
```

**Option 2: Batch File (Windows)**
```bash
run_futures_gui.bat
```

The application will open at `http://localhost:8501`

### Navigation

The sidebar contains three modes:
1. **📊 Backtest** - Historical strategy testing
2. **📡 Live Paper** - Real-time paper trading
3. **💾 Saved** - View saved backtest results

---

## Strategies Explained

### 1. RSI (Relative Strength Index)

**Logic:**
- **Long Entry**: RSI crosses below 30 (oversold)
- **Short Entry**: RSI crosses above 70 (overbought)
- **Exit**: RSI returns to neutral zone (35-65)

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| rsi_period | 14 | Lookback period |
| overbought | 70 | Overbought threshold |
| oversold | 30 | Oversold threshold |
| exit_overbought | 65 | Long exit threshold |
| exit_oversold | 35 | Short exit threshold |

**Best For:** Range-bound, sideways markets

---

### 2. MACD (Moving Average Convergence Divergence)

**Logic:**
- **Long Entry**: MACD histogram crosses from negative to positive
- **Short Entry**: MACD histogram crosses from positive to negative
- **Exit**: Opposite signal

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| fast_period | 12 | Fast EMA period |
| slow_period | 26 | Slow EMA period |
| signal_period | 9 | Signal line period |
| use_histogram | True | Use histogram crossover |

**Best For:** Trending markets

---

### 3. Bollinger Bands

**Logic:**
- **Long Entry**: Price touches lower band
- **Short Entry**: Price touches upper band
- **Exit**: Price reaches middle band

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| period | 20 | SMA period |
| std_dev | 2.0 | Standard deviations |
| exit_at_middle | True | Exit at middle band |

**Best For:** Mean-reverting, range-bound markets

---

### 4. EMA Crossover

**Logic:**
- **Long Entry**: Fast EMA crosses above slow EMA
- **Short Entry**: Fast EMA crosses below slow EMA
- **Exit**: Opposite crossover

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| fast_period | 9 | Fast EMA period |
| slow_period | 21 | Slow EMA period |

**Best For:** Trending markets

---

### 5. Supertrend

**Logic:**
- **Long Entry**: Price closes above Supertrend line
- **Short Entry**: Price closes below Supertrend line
- **Exit**: Direction reversal

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| period | 10 | ATR period |
| multiplier | 3.0 | ATR multiplier |

**Best For:** Trending markets

---

### 6. RSI + MACD Combined

**Logic:**
- **Long Entry**: MACD bullish AND RSI < 40
- **Short Entry**: MACD bearish AND RSI > 60
- **Exit**: Opposite combined signal

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| rsi_period | 14 | RSI period |
| rsi_long_threshold | 40 | Max RSI for long |
| rsi_short_threshold | 60 | Min RSI for short |
| macd_fast | 12 | MACD fast period |
| macd_slow | 26 | MACD slow period |
| macd_signal | 9 | MACD signal period |

**Best For:** Filtered, high-probability entries

---

### 7. Stochastic RSI

**Logic:**
- **Long Entry**: %K crosses above %D in oversold zone (<20)
- **Short Entry**: %K crosses below %D in overbought zone (>80)
- **Exit**: Enter opposite zone

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| rsi_period | 14 | RSI period |
| stoch_period | 14 | Stochastic period |
| k_period | 3 | %K smoothing |
| d_period | 3 | %D smoothing |
| overbought | 80 | Overbought level |
| oversold | 20 | Oversold level |

**Best For:** Range-bound markets with clear cycles

---

### 8. Triple EMA

**Logic:**
- **Long Entry**: Fast > Medium > Slow EMA
- **Short Entry**: Fast < Medium < Slow EMA
- **Exit**: Alignment breaks

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| fast_period | 8 | Fast EMA period |
| medium_period | 21 | Medium EMA period |
| slow_period | 55 | Slow EMA period |

**Best For:** Strong trending markets

---

## Settings Configuration

### Trading Configuration

| Setting | Range | Description |
|---------|-------|-------------|
| Initial Balance | $100+ | Starting capital |
| Leverage | 1x - 125x | Position leverage multiplier |
| Position Size | 10% - 100% | % of balance per trade |

### Risk Management

| Setting | Description |
|---------|-------------|
| Stop Loss | Automatic exit at % loss from entry |
| Take Profit | Automatic exit at % gain from entry |

### Settings Applied Display

After running a backtest, the applied settings are displayed:
```
Settings: BTCUSDT | 1h | RSI | 10x | 100% | SL: 2% | TP: 4%
```

This confirms all configurations were correctly applied.

---

## Chart Features

### Trade Markers

| Marker | Meaning |
|--------|---------|
| 🔺 Green Triangle Up | Long Entry |
| 🔻 Red Triangle Down | Short Entry |
| ✕ Green X | Profitable Exit |
| ✕ Red X | Loss Exit |
| Dotted Line | Entry to Exit connection |

### Mouse Controls

| Action | Function |
|--------|----------|
| Scroll Wheel | Zoom in/out |
| Click + Drag | Pan chart |
| Double Click | Reset zoom |
| Hover | Show price details |

### Toolbar Buttons

- **Zoom** - Box select to zoom
- **Pan** - Click and drag to pan
- **Zoom In/Out** - Incremental zoom
- **Auto Scale** - Fit all data
- **Reset** - Return to original view

### Indicators Available

- SMA 20, SMA 50
- EMA 12, EMA 26
- Bollinger Bands
- Volume
- RSI (with overbought/oversold zones)
- MACD (with histogram)

---

## Saved Backtests

### Auto-Save Feature

When "Auto-save" is checked (default), backtests are automatically saved after completion.

### Saved Data

Each saved backtest stores:
- Symbol, Strategy, Interval
- Initial Balance, Final Equity
- Total Return %, Win Rate %
- Sharpe Ratio, Max Drawdown
- Total Trades, Profit Factor
- Leverage, Position Size
- Stop Loss, Take Profit settings

### Managing Saved Backtests

- View in "💾 Saved" tab
- Expand each entry for details
- Delete individual backtests
- Clear all backtests

### Storage

Backtests are stored in `saved_backtests.json` (max 50 entries).

---

## Live Paper Trading

### Starting Paper Trading

1. Configure settings in sidebar
2. Click "▶️ Start"
3. Click "🔄 Refresh Data" to fetch current market

### Position Management

| Button | Action |
|--------|--------|
| 📈 Long | Open long position |
| 📉 Short | Open short position |
| ❌ Close | Close current position |

### Real-Time Metrics

- Current Equity
- Unrealized PnL
- Position Status (Long/Short/Flat)
- Total Trades
- Win Rate

### Signal Display

The current strategy signal is shown:
- 🟢 LONG - Strategy suggests long
- 🔴 SHORT - Strategy suggests short
- ⚪ FLAT - No position recommended

---

## Technical Architecture

### Project Structure

```
crypto_bot/
├── futures_exchange.py    # Binance Futures API client
├── futures_backtest.py    # Backtesting engine
├── indicators.py          # Technical indicators (30+)
├── strategies/
│   ├── base_strategy.py   # Abstract base class
│   ├── futures_strategies.py  # 8 strategies
│   └── __init__.py
├── config.py
└── __init__.py

streamlit_app.py           # Main GUI application
saved_backtests.json       # Saved backtest storage
requirements.txt           # Dependencies
DOCUMENTATION.md           # This file
```

### API Endpoints Used

**Binance Futures Public API** (no auth required):
- `GET /fapi/v1/klines` - Historical candlesticks
- `GET /fapi/v1/ticker/price` - Current price
- `GET /fapi/v1/exchangeInfo` - Trading pairs

### Performance Metrics Calculated

| Metric | Formula |
|--------|---------|
| Total Return | (Final - Initial) / Initial |
| Sharpe Ratio | (Return - RiskFree) / Volatility |
| Sortino Ratio | (Return - RiskFree) / Downside Deviation |
| Calmar Ratio | Annual Return / Max Drawdown |
| Profit Factor | Gross Profit / Gross Loss |
| Win Rate | Winning Trades / Total Trades |

---

## Troubleshooting

### Common Issues

**1. "Error fetching data"**
- Check internet connection
- Binance API may be temporarily unavailable
- Try refreshing the page

**2. "No trades executed"**
- Strategy parameters may be too strict
- Try increasing the number of candles
- Adjust strategy parameters

**3. Charts not displaying**
- Ensure Plotly is installed
- Check browser console for errors
- Try a different browser

**4. Metrics showing poor contrast**
- Clear browser cache
- The CSS should auto-apply
- Check if custom themes are interfering

### Reset Application

```bash
# Clear saved data
del saved_backtests.json

# Restart server
streamlit run streamlit_app.py
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Dec 2024 | Initial release |
| 2.0 | Dec 2024 | Strategy explanations, trade markers, saved backtests, improved UI |

---

## Disclaimer

⚠️ **This is a paper trading system for educational purposes only.**

- No real money is involved in any trades
- Past performance does not guarantee future results
- Cryptocurrency futures trading involves significant risk
- Always do your own research before trading with real funds

---

## Support

For issues or feature requests, please review this documentation first. The system is designed to be self-contained and requires no external API keys for market data access.
