# Futures Trading Bot

A comprehensive cryptocurrency futures trading bot with Streamlit GUI for live paper trading and historical backtesting.

## Features

### 📊 Historical Backtesting
- Test strategies on historical data from Binance Futures
- Support for all USDT perpetual futures pairs
- Multiple timeframes (1m to 1d)
- Comprehensive performance metrics

### 📡 Live Paper Trading
- Real-time data from Binance Futures public API
- No API keys required for market data
- Simulated trading without real money
- Position management (Long/Short/Flat)

### 📈 Built-in Strategies
1. **RSI Strategy** - Relative Strength Index based
2. **MACD Strategy** - Moving Average Convergence Divergence
3. **Bollinger Bands** - Mean reversion strategy
4. **EMA Crossover** - Fast/Slow EMA crossover
5. **Supertrend** - Trend following indicator
6. **RSI + MACD** - Combined momentum strategy
7. **Stochastic RSI** - Momentum oscillator
8. **Triple EMA** - Three EMA trend confirmation

### 🔧 Technical Indicators
- **Trend**: SMA, EMA, WMA, MACD, ADX, Supertrend, Ichimoku
- **Momentum**: RSI, Stochastic, Stochastic RSI, CCI, Williams %R, ROC
- **Volatility**: ATR, Bollinger Bands, Keltner Channels, Donchian
- **Volume**: OBV, VWAP, MFI, A/D Line, Chaikin Oscillator

### 💰 Trading Configuration
- Adjustable leverage (1x - 125x)
- Position sizing (% of balance)
- Stop Loss / Take Profit
- Fee simulation (maker/taker)
- Slippage modeling

### 📉 Performance Metrics
- Total Return / Annualized Return
- Sharpe Ratio / Sortino Ratio / Calmar Ratio
- Max Drawdown / Drawdown Duration
- Win Rate / Profit Factor
- Risk/Reward Ratio
- Long/Short breakdown

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### Start the GUI
```bash
# Option 1: Run the batch file (Windows)
run_futures_gui.bat

# Option 2: Run directly
streamlit run streamlit_app.py
```

### Access the Application
Open your browser and navigate to: `http://localhost:8501`

## Project Structure

```
crypto_bot/
├── futures_exchange.py    # Binance Futures API client
├── futures_backtest.py    # Backtesting engine with leverage
├── indicators.py          # Technical indicators library
├── strategies/
│   ├── base_strategy.py
│   ├── futures_strategies.py  # All futures strategies
│   └── ...
streamlit_app.py           # Main Streamlit GUI application
run_futures_gui.bat        # Windows startup script
```

## API Information

This bot uses the **Binance Futures Public API** which does NOT require authentication for:
- Historical kline/candlestick data
- Real-time price data
- Order book and trades
- Funding rates and open interest

**Base URL**: `https://fapi.binance.com`
**WebSocket**: `wss://fstream.binance.com/ws`

## Disclaimer

⚠️ **This is a paper trading system for educational purposes only.**

- No real money is involved
- Past performance does not guarantee future results
- Always do your own research before trading
- Cryptocurrency futures trading involves significant risk
