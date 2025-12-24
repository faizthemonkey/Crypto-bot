"""
Tests for Futures Backtesting Engine v3.0
Tests stop loss execution, fixed amount mode, and leverage calculations
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_bot.futures_backtest import (
    FuturesBacktester, LivePaperTrader, BacktestConfig, 
    PositionSide, Trade, Position
)


def create_test_data(prices: list, start_time: datetime = None) -> pd.DataFrame:
    """Create test OHLCV data from price list."""
    if start_time is None:
        start_time = datetime(2024, 1, 1)
    
    data = []
    for i, price in enumerate(prices):
        # Create candle with some variation
        high = price * 1.01
        low = price * 0.99
        data.append({
            'open_time': start_time + timedelta(hours=i),
            'open': price,
            'high': high,
            'low': low,
            'close': price,
            'volume': 1000
        })
    return pd.DataFrame(data)


def create_test_signals(signals: list) -> pd.DataFrame:
    """Create test signals DataFrame."""
    return pd.DataFrame({'signal': signals})


class TestStopLossExecution:
    """Tests for stop loss functionality."""
    
    def test_stop_loss_triggers_at_correct_price(self):
        """Stop loss should trigger at the stop price, not close price."""
        config = BacktestConfig(
            initial_balance=10000,
            leverage=10,
            position_size_pct=100,
            use_stop_loss=True,
            stop_loss_pct=20  # 20% on margin = 2% price move at 10x leverage
        )
        
        # Entry at 100, SL at 20% margin loss = 2% price drop = 98
        prices = [100, 100, 95]  # Price drops to 95, hitting stop at 98
        data = create_test_data(prices)
        data.loc[2, 'low'] = 97  # Low goes to 97, triggers stop at 98
        
        # Use signal 0 (flat) after bar 0 to avoid re-entry after SL
        signals = create_test_signals([1, 0, 0])
        
        backtester = FuturesBacktester(config)
        results = backtester.run(data, signals)
        
        # First trade should be closed by signal, let's test with maintained position
        # Actually test using signal=1 to keep position and verify SL exits at correct price
        config2 = BacktestConfig(
            initial_balance=10000, leverage=10, position_size_pct=100,
            use_stop_loss=True, stop_loss_pct=20
        )
        signals2 = create_test_signals([1, 1, 1])  # Maintain position
        results2 = FuturesBacktester(config2).run(data, signals2)
        
        # The FIRST trade should be closed by stop loss at correct price
        first_trade = results2['trades'][0]
        expected_stop_price = 100 * (1 - 20/10/100)  # 98
        assert abs(first_trade.exit_price - expected_stop_price) < 0.1, f"Expected {expected_stop_price}, got {first_trade.exit_price}"
        assert abs(first_trade.pnl_percent - (-20.0)) < 1  # -20% ± 1%
    
    def test_stop_loss_accounts_for_leverage(self):
        """20% SL on 20x leverage = 1% price move."""
        config = BacktestConfig(
            initial_balance=10000,
            leverage=20,
            position_size_pct=100,
            use_stop_loss=True,
            stop_loss_pct=20
        )
        
        # At 20x leverage, 20% margin loss = 1% price move
        # Entry at 100, stop at 99
        prices = [100, 100, 98]
        data = create_test_data(prices)
        data.loc[2, 'low'] = 98.5  # Low goes to 98.5, triggers stop at 99
        
        signals = create_test_signals([1, 0, 0])
        
        backtester = FuturesBacktester(config)
        results = backtester.run(data, signals)
        
        closed_trades = [t for t in results['trades'] if not t.is_open]
        assert len(closed_trades) == 1
        
        trade = closed_trades[0]
        expected_stop = 100 * (1 - 20/20/100)  # 99
        assert abs(trade.exit_price - expected_stop) < 0.1
    
    def test_take_profit_triggers_correctly(self):
        """Take profit should trigger at TP price."""
        config = BacktestConfig(
            initial_balance=10000,
            leverage=10,
            position_size_pct=100,
            use_take_profit=True,
            take_profit_pct=20  # 20% on margin = 2% price gain
        )
        
        # Entry at 100, TP at 102
        prices = [100, 100, 105]
        data = create_test_data(prices)
        data.loc[2, 'high'] = 103  # High reaches 103, triggers TP at 102
        
        signals = create_test_signals([1, 1, 1])  # Stay long
        
        backtester = FuturesBacktester(config)
        results = backtester.run(data, signals)
        
        # First trade should be closed by TP at correct price
        first_trade = results['trades'][0]
        expected_tp = 100 * (1 + 20/10/100)  # 102
        assert abs(first_trade.exit_price - expected_tp) < 0.1, f"Expected {expected_tp}, got {first_trade.exit_price}"
        assert first_trade.pnl > 0
        assert abs(first_trade.pnl_percent - 20.0) < 1  # +20% ± 1%
    
    def test_short_stop_loss(self):
        """Stop loss for short positions should trigger on price rise."""
        config = BacktestConfig(
            initial_balance=10000,
            leverage=10,
            position_size_pct=100,
            use_stop_loss=True,
            stop_loss_pct=20
        )
        
        # Short entry at 100, SL at 102
        prices = [100, 100, 105]
        data = create_test_data(prices)
        data.loc[2, 'high'] = 103  # High reaches 103, triggers stop at 102
        
        signals = create_test_signals([-1, -1, -1])  # Stay short
        
        backtester = FuturesBacktester(config)
        results = backtester.run(data, signals)
        
        # First trade should be closed by SL
        first_trade = results['trades'][0]
        assert first_trade.side == PositionSide.SHORT
        expected_stop = 100 * (1 + 20/10/100)  # 102
        assert abs(first_trade.exit_price - expected_stop) < 0.1, f"Expected {expected_stop}, got {first_trade.exit_price}"
        assert abs(first_trade.pnl_percent - (-20.0)) < 1  # -20% ± 1%


class TestFixedAmountMode:
    """Tests for fixed amount trading mode."""
    
    def test_fixed_amount_position_sizing(self):
        """Fixed amount mode should use fixed dollar amount per trade."""
        config = BacktestConfig(
            initial_balance=10000,
            leverage=10,
            fixed_amount_mode=True,
            fixed_amount=1000  # $1000 per trade
        )
        
        prices = [100, 100, 110]  # 10% gain
        data = create_test_data(prices)
        signals = create_test_signals([1, 0, 0])
        
        backtester = FuturesBacktester(config)
        results = backtester.run(data, signals)
        
        trades = results['trades']
        assert len(trades) == 1
        
        # Position value should be $1000, not 100% of balance
        trade = trades[0]
        position_value = trade.size * trade.entry_price
        assert abs(position_value - 1000) < 10  # Small tolerance
    
    def test_fixed_amount_capped_by_balance(self):
        """Fixed amount should not exceed available balance."""
        config = BacktestConfig(
            initial_balance=500,  # Only $500 available
            leverage=10,
            fixed_amount_mode=True,
            fixed_amount=1000  # $1000 requested
        )
        
        prices = [100, 100, 110]
        data = create_test_data(prices)
        signals = create_test_signals([1, 0, 0])
        
        backtester = FuturesBacktester(config)
        results = backtester.run(data, signals)
        
        trades = results['trades']
        trade = trades[0]
        position_value = trade.size * trade.entry_price
        assert position_value <= 500  # Should be capped at balance
    
    def test_percentage_mode_default(self):
        """Percentage mode should use % of balance."""
        config = BacktestConfig(
            initial_balance=10000,
            leverage=10,
            fixed_amount_mode=False,
            position_size_pct=50  # 50% of balance
        )
        
        prices = [100, 100, 110]
        data = create_test_data(prices)
        signals = create_test_signals([1, 0, 0])
        
        backtester = FuturesBacktester(config)
        results = backtester.run(data, signals)
        
        trades = results['trades']
        trade = trades[0]
        position_value = trade.size * trade.entry_price
        expected = 10000 * 0.5  # 50% of balance
        assert abs(position_value - expected) < 10


class TestLivePaperTrader:
    """Tests for LivePaperTrader."""
    
    def test_sl_tp_check_method(self):
        """Test the check_stop_loss_take_profit method."""
        config = BacktestConfig(
            initial_balance=10000,
            leverage=10,
            use_stop_loss=True,
            stop_loss_pct=20,
            use_take_profit=True,
            take_profit_pct=40
        )
        
        trader = LivePaperTrader(config)
        trader.update_price(100)
        trader.open_long()
        
        # Entry at 100, SL at 98, TP at 104 (for 10x leverage)
        
        # No trigger
        result = trader.check_stop_loss_take_profit(high=101, low=99)
        assert result is None
        
        # SL trigger
        result = trader.check_stop_loss_take_profit(high=101, low=97)
        expected_sl = 100 * (1 - 20/10/100)  # 98
        assert result is not None
        assert abs(result - expected_sl) < 0.1
    
    def test_fixed_amount_in_live_trader(self):
        """LivePaperTrader should support fixed amount mode."""
        config = BacktestConfig(
            initial_balance=10000,
            leverage=10,
            fixed_amount_mode=True,
            fixed_amount=1000
        )
        
        trader = LivePaperTrader(config)
        trader.update_price(100)
        trader.open_long()
        
        # Position should be $1000 worth
        position_value = trader.position.size * trader.position.entry_price
        assert abs(position_value - 1000) < 10


class TestPnLCalculation:
    """Tests for PnL calculations with leverage."""
    
    def test_leveraged_pnl(self):
        """PnL should be multiplied by leverage."""
        config = BacktestConfig(
            initial_balance=10000,
            leverage=10,
            position_size_pct=100
        )
        
        # 10% price gain with 10x leverage = 100% return on margin
        prices = [100, 110]
        data = create_test_data(prices)
        signals = create_test_signals([1, 0])  # Open and hold to close
        
        backtester = FuturesBacktester(config)
        results = backtester.run(data, signals)
        
        # With $10k at 10x, position value is $100k
        # 10% gain on $100k = $10k profit
        # Return should be ~100% (minus fees)
        assert results['total_return_pct'] > 90
        assert results['total_return_pct'] < 110
    
    def test_short_position_pnl(self):
        """Short positions should profit from price drops."""
        config = BacktestConfig(
            initial_balance=10000,
            leverage=10,
            position_size_pct=100
        )
        
        prices = [100, 90]  # 10% drop
        data = create_test_data(prices)
        signals = create_test_signals([-1, 0])
        
        backtester = FuturesBacktester(config)
        results = backtester.run(data, signals)
        
        # Should be profitable
        assert results['total_return_pct'] > 0


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_no_trades(self):
        """No trades should return empty results."""
        config = BacktestConfig(initial_balance=10000)
        
        prices = [100, 101, 102]
        data = create_test_data(prices)
        signals = create_test_signals([0, 0, 0])
        
        backtester = FuturesBacktester(config)
        results = backtester.run(data, signals)
        
        assert results['total_trades'] == 0
        assert results['final_equity'] == 10000
    
    def test_rapid_signal_changes(self):
        """Rapid signal changes should handle position reversals."""
        config = BacktestConfig(
            initial_balance=10000,
            leverage=10,
            position_size_pct=100
        )
        
        prices = [100, 101, 100, 99, 100]
        data = create_test_data(prices)
        signals = create_test_signals([1, -1, 1, -1, 0])  # Rapid switches
        
        backtester = FuturesBacktester(config)
        results = backtester.run(data, signals)
        
        # Should have multiple trades
        assert results['total_trades'] >= 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
