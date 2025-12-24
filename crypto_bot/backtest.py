import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .exchange import BinanceClient
from .strategies import BaseStrategy
from .config import Config


class Backtester:
    """Backtesting engine for trading strategies.

    Supports both remote data (via BinanceClient) and fully local backtests
    on pre-downloaded historical data (e.g., CSV files loaded into a DataFrame).
    """

    def __init__(self, config: Config):
        """Initialize the backtester with configuration.

        ``exchange`` is created lazily so that purely local backtests do not
        require Binance API keys.
        """
        self.config = config
        self.exchange: Optional[BinanceClient] = None
        self.results: Dict[str, Any] | None = None

    def run_backtest(
        self,
        strategy: BaseStrategy,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: str = None,
        initial_balance: float = 10000.0,
    ) -> Dict[str, Any]:
        """Run a backtest on historical data fetched from the exchange.

        This method keeps the existing behaviour (fetch from Binance), but all
        performance calculations are delegated to ``run_backtest_on_data`` so
        that local/offline backtests behave identically.
        """

        if self.exchange is None:
            self.exchange = BinanceClient(self.config)

        klines = self.exchange.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start_date,
            end_str=end_date,
        )

        if klines.empty:
            raise ValueError("No historical data found for the given parameters")

        return self.run_backtest_on_data(
            strategy=strategy,
            data=klines,
            initial_balance=initial_balance,
        )

    def run_backtest_on_data(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        initial_balance: float = 10000.0,
    ) -> Dict[str, Any]:
        """Run a backtest on an in-memory DataFrame of historical data.

        ``data`` must contain at least the columns::

            open_time, open, high, low, close, volume
        """

        if data is None or data.empty:
            raise ValueError("No historical data provided for backtest")

        price_data = data.copy()

        # Generate signals from the strategy
        signals = strategy.generate_signals(price_data)

        # Combine data and signals
        combined = pd.concat([price_data.reset_index(drop=True), signals.reset_index(drop=True)], axis=1)

        # Calculate simple returns
        combined["returns"] = combined["close"].pct_change()
        combined["strategy_returns"] = combined["signal"].shift(1) * combined["returns"]

        # Calculate cumulative returns (market and strategy)
        combined["cumulative_returns"] = (1 + combined["returns"]).cumprod()
        combined["cumulative_strategy_returns"] = (1 + combined["strategy_returns"]).cumprod()

        # Equity curve in quote currency
        combined["equity_curve"] = initial_balance * combined["cumulative_strategy_returns"]

        # Infer bar frequency to annualize metrics (fallback to hourly assumption)
        bars_per_year = 365 * 24
        if "open_time" in combined.columns and pd.api.types.is_datetime64_any_dtype(combined["open_time"]):
            deltas = combined["open_time"].diff().dt.total_seconds().dropna()
            if not deltas.empty:
                avg_seconds = deltas.median()
                if avg_seconds > 0:
                    bars_per_year = int((365 * 24 * 3600) / avg_seconds)

        total_return = combined["cumulative_strategy_returns"].iloc[-1] - 1
        # Guard against very short series
        if len(combined) > 1:
            annualized_return = (1 + total_return) ** (bars_per_year / len(combined)) - 1
        else:
            annualized_return = 0.0

        annualized_vol = combined["strategy_returns"].std() * np.sqrt(bars_per_year)
        sharpe_ratio = (annualized_return - 0.02) / annualized_vol if annualized_vol not in (0, np.nan) else 0

        # Max drawdown on strategy equity curve
        cum_returns = combined["cumulative_strategy_returns"]
        rolling_max = cum_returns.cummax()
        drawdown = (cum_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        self.results = {
            "data": combined,
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "annualized_volatility": float(annualized_vol) if not np.isnan(annualized_vol) else 0.0,
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "num_trades": int(len(combined[combined["positions"] != 0])) if "positions" in combined.columns else 0,
            "win_rate": self._calculate_win_rate(combined),
        }

        return self.results
    
    def _calculate_win_rate(self, data: pd.DataFrame) -> float:
        """Calculate the win rate of trades."""
        if "positions" not in data.columns or "strategy_returns" not in data.columns:
            return 0.0

        trades = data[data["positions"] != 0].copy()
        if len(trades) == 0:
            return 0.0

        trades["trade_returns"] = trades["strategy_returns"]
        win_trades = len(trades[trades["trade_returns"] > 0])

        return win_trades / len(trades)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the backtest results."""
        if self.results is None:
            raise ValueError("No backtest results available. Run backtest() first.")
        
        return {
            'Total Return (%)': self.results['total_return'] * 100,
            'Annualized Return (%)': self.results['annualized_return'] * 100,
            'Annualized Volatility': self.results['annualized_volatility'],
            'Sharpe Ratio': self.results['sharpe_ratio'],
            'Max Drawdown (%)': self.results['max_drawdown'] * 100,
            'Number of Trades': self.results['num_trades'],
            'Win Rate (%)': self.results['win_rate'] * 100
        }
