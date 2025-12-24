# Crypto Trading Bot Project Documentation

## Overview
This project is a modular, Python-based cryptocurrency trading bot designed for the Binance exchange. It features a robust architecture that supports live trading, historical data analysis, and strategy backtesting (both remote and local).

## Core Capabilities

### 1. Trading Engine
- **Binance Integration**: Seamless connection to Binance Spot API.
- **Environment Support**: 
  - **Testnet**: Safe environment for testing with paper money.
  - **Live**: Real-money trading capability.
- **Order Management**: Supports Market orders (expandable to Limit orders).
- **Real-time Data**: WebSocket integration for streaming live candlestick (kline) data.

### 2. Strategy Framework
- **Modular Design**: Abstract base class (`BaseStrategy`) allows for easy creation of custom strategies.
- **Standard Interface**: 
  - `calculate_indicators(data)`: For technical analysis.
  - `generate_signals(data)`: For trade logic (Buy/Sell/Hold).
- **Included Strategies**:
  - `MovingAverageCrossover`: A classic trend-following strategy using Fast and Slow Moving Averages.

### 3. Backtesting System
- **Dual Modes**:
  - **Remote**: Fetches historical data directly from Binance API.
  - **Local (Offline)**: Runs backtests on local CSV files, requiring no API keys or internet connection.
- **Performance Metrics**:
  - Total Return & Annualized Return
  - Sharpe Ratio & Volatility
  - Max Drawdown
  - Win Rate & Number of Trades
- **Equity Curve**: Tracks portfolio value over time.

### 4. Graphical User Interface (GUI)
- **Interactive Backtester**: Tkinter-based GUI for visual strategy validation.
- **Visualizations**:
  - Interactive Candlestick charts (using `mplfinance`).
  - Buy/Sell trade markers overlaid on price action.
  - Equity curve plotting.
- **Controls**:
  - File picker for local CSV data.
  - Adjustable strategy parameters (MA periods, Initial Balance).

### 5. Data Management
- **Data Loader**: Specialized utility (`load_ohlcv_csv`) to sanitize and standardize historical CSV data.
- **Configuration**: centralized configuration using `.env` files for secure API key management.

## Workflow

### Setup Phase
1. **Environment**: Install dependencies via `pip install -r requirements.txt`.
2. **Configuration**: Create a `.env` file with `BINANCE_API_KEY` and `BINANCE_API_SECRET` (optional for local backtesting).

### Development Phase
1. **Strategy Creation**: 
   - Create a new class in `crypto_bot/strategies/` inheriting from `BaseStrategy`.
   - Implement logic to return Buy (1), Sell (-1), or Hold (0) signals.
2. **Data Acquisition**: Download historical data (CSV) for testing.

### Testing Phase (Backtesting)
1. **GUI Method**:
   - Run `python -m crypto_bot.gui_backtest`.
   - Load your CSV file.
   - Adjust parameters and visualize entries/exits.
2. **Script Method**:
   - Use `examples_local_backtest.py` for automated or headless testing.
   - Analyze the printed performance report.

### Deployment Phase (Live Trading)
1. **Configuration**: Set `TESTNET=False` in `.env` for real trading.
2. **Execution**: 
   - Initialize `TradingBot` with your strategy.
   - Call `bot.run_live()` to start WebSocket listeners.
   - The bot will process incoming candles and execute trades automatically.

## Project Structure

```text
windsurf-project/
├── crypto_bot/
│   ├── strategies/         # Strategy implementations
│   ├── backtest.py         # Backtesting engine (Local & Remote)
│   ├── bot.py              # Main live trading logic
│   ├── config.py           # Settings management
│   ├── data_loader.py      # CSV data ingestion
│   ├── exchange.py         # Binance API wrapper
│   └── gui_backtest.py     # GUI application
├── examples_local_backtest.py  # Usage example
├── requirements.txt        # Dependencies
└── .env                    # Secrets (Excluded from git)
```

## Future Extensibility
- **Multiple Timeframes**: Adapt strategies to use multi-timeframe analysis.
- **Risk Management**: Implement stop-loss and take-profit logic in the `TradingBot` class.
- **Portfolio Management**: Support for multi-asset trading.
