"""
Futures Backtesting Engine with Leverage Support
Supports long/short positions, leverage, and comprehensive metrics
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    side: PositionSide
    size: float
    leverage: float
    pnl: float = 0.0
    pnl_percent: float = 0.0
    fees: float = 0.0
    is_open: bool = True
    
    def close(self, exit_time: datetime, exit_price: float, fee_rate: float = 0.0004):
        """Close the trade and calculate PnL."""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.is_open = False
        
        # Calculate PnL based on position side
        if self.side == PositionSide.LONG:
            price_diff = exit_price - self.entry_price
        else:  # SHORT
            price_diff = self.entry_price - exit_price
        
        # Calculate fees (entry + exit)
        self.fees = (self.entry_price + exit_price) * self.size * fee_rate
        
        # PnL with leverage
        self.pnl = (price_diff * self.size * self.leverage) - self.fees
        self.pnl_percent = (price_diff / self.entry_price) * self.leverage * 100


@dataclass
class Position:
    """Current position state."""
    side: PositionSide = PositionSide.FLAT
    size: float = 0.0
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    leverage: float = 1.0
    unrealized_pnl: float = 0.0
    margin: float = 0.0


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_balance: float = 10000.0
    leverage: float = 1.0
    position_size_pct: float = 100.0  # % of balance to use per trade
    maker_fee: float = 0.0002  # 0.02%
    taker_fee: float = 0.0004  # 0.04%
    slippage_pct: float = 0.0  # Slippage percentage
    use_stop_loss: bool = False
    stop_loss_pct: float = 2.0  # Stop loss percentage
    use_take_profit: bool = False
    take_profit_pct: float = 4.0  # Take profit percentage


class FuturesBacktester:
    """
    Futures backtesting engine with support for:
    - Long/Short positions
    - Leverage trading
    - Fees and slippage simulation
    - Stop loss / Take profit
    - Comprehensive performance metrics
    """
    
    def __init__(self, config: BacktestConfig = None):
        """Initialize the backtester."""
        self.config = config or BacktestConfig()
        self.reset()
    
    def reset(self):
        """Reset all state."""
        self.balance = self.config.initial_balance
        self.equity_curve = []
        self.trades: List[Trade] = []
        self.position = Position()
        self.signals_history = []
        self.current_drawdown = 0.0
        self.max_equity = self.config.initial_balance
    
    def run(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Run backtest on historical data with signals.
        
        Args:
            data: DataFrame with OHLCV data
            signals: DataFrame with 'signal' column (1=long, -1=short, 0=close/flat)
            verbose: Print trade details
        
        Returns:
            Dictionary with backtest results
        """
        self.reset()
        
        # Ensure we have required columns
        required_cols = ['open_time', 'open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        if 'signal' not in signals.columns:
            raise ValueError("Signals DataFrame must have 'signal' column")
        
        # Combine data and signals
        df = data.copy()
        df['signal'] = signals['signal'].values
        
        # Run through each bar
        for i in range(len(df)):
            row = df.iloc[i]
            current_time = row['open_time']
            current_price = row['close']
            signal = row['signal']
            
            # Update unrealized PnL if in position
            if self.position.side != PositionSide.FLAT:
                self._update_unrealized_pnl(current_price)
                
                # Check stop loss / take profit
                if self.config.use_stop_loss or self.config.use_take_profit:
                    if self._check_exit_conditions(row):
                        self._close_position(current_time, current_price)
            
            # Process signal
            if signal == 1 and self.position.side != PositionSide.LONG:
                # Close short if open
                if self.position.side == PositionSide.SHORT:
                    self._close_position(current_time, current_price)
                # Open long
                self._open_position(current_time, current_price, PositionSide.LONG)
                
            elif signal == -1 and self.position.side != PositionSide.SHORT:
                # Close long if open
                if self.position.side == PositionSide.LONG:
                    self._close_position(current_time, current_price)
                # Open short
                self._open_position(current_time, current_price, PositionSide.SHORT)
                
            elif signal == 0 and self.position.side != PositionSide.FLAT:
                # Close any open position
                self._close_position(current_time, current_price)
            
            # Record equity
            equity = self.balance + self.position.unrealized_pnl
            self.equity_curve.append({
                'time': current_time,
                'equity': equity,
                'balance': self.balance,
                'unrealized_pnl': self.position.unrealized_pnl,
                'position_side': self.position.side.value,
                'signal': signal
            })
            
            # Track max equity and drawdown
            if equity > self.max_equity:
                self.max_equity = equity
            self.current_drawdown = (self.max_equity - equity) / self.max_equity
            
            if verbose and len(self.trades) > 0 and not self.trades[-1].is_open:
                trade = self.trades[-1]
                print(f"Trade closed: {trade.side.value} | PnL: {trade.pnl:.2f} ({trade.pnl_percent:.2f}%)")
        
        # Close any remaining position at end
        if self.position.side != PositionSide.FLAT:
            final_row = df.iloc[-1]
            self._close_position(final_row['open_time'], final_row['close'])
        
        return self._calculate_results(df)
    
    def _open_position(self, time: datetime, price: float, side: PositionSide):
        """Open a new position."""
        # Apply slippage
        if side == PositionSide.LONG:
            entry_price = price * (1 + self.config.slippage_pct / 100)
        else:
            entry_price = price * (1 - self.config.slippage_pct / 100)
        
        # Calculate position size
        position_value = self.balance * (self.config.position_size_pct / 100)
        margin = position_value / self.config.leverage
        size = position_value / entry_price
        
        # Deduct fees
        fee = entry_price * size * self.config.taker_fee
        
        self.position = Position(
            side=side,
            size=size,
            entry_price=entry_price,
            entry_time=time,
            leverage=self.config.leverage,
            margin=margin
        )
        
        # Create trade record
        self.trades.append(Trade(
            entry_time=time,
            exit_time=None,
            entry_price=entry_price,
            exit_price=None,
            side=side,
            size=size,
            leverage=self.config.leverage
        ))
    
    def _close_position(self, time: datetime, price: float):
        """Close current position."""
        if self.position.side == PositionSide.FLAT:
            return
        
        # Apply slippage
        if self.position.side == PositionSide.LONG:
            exit_price = price * (1 - self.config.slippage_pct / 100)
        else:
            exit_price = price * (1 + self.config.slippage_pct / 100)
        
        # Close the trade
        if self.trades and self.trades[-1].is_open:
            self.trades[-1].close(time, exit_price, self.config.taker_fee)
            self.balance += self.trades[-1].pnl
        
        # Reset position
        self.position = Position()
    
    def _update_unrealized_pnl(self, current_price: float):
        """Update unrealized PnL for open position."""
        if self.position.side == PositionSide.FLAT:
            self.position.unrealized_pnl = 0.0
            return
        
        if self.position.side == PositionSide.LONG:
            price_diff = current_price - self.position.entry_price
        else:
            price_diff = self.position.entry_price - current_price
        
        self.position.unrealized_pnl = price_diff * self.position.size * self.position.leverage
    
    def _check_exit_conditions(self, row: pd.Series) -> bool:
        """Check if stop loss or take profit is hit."""
        if self.position.side == PositionSide.FLAT:
            return False
        
        entry_price = self.position.entry_price
        
        if self.position.side == PositionSide.LONG:
            # For long position
            if self.config.use_stop_loss:
                stop_price = entry_price * (1 - self.config.stop_loss_pct / 100)
                if row['low'] <= stop_price:
                    return True
            
            if self.config.use_take_profit:
                tp_price = entry_price * (1 + self.config.take_profit_pct / 100)
                if row['high'] >= tp_price:
                    return True
        else:
            # For short position
            if self.config.use_stop_loss:
                stop_price = entry_price * (1 + self.config.stop_loss_pct / 100)
                if row['high'] >= stop_price:
                    return True
            
            if self.config.use_take_profit:
                tp_price = entry_price * (1 - self.config.take_profit_pct / 100)
                if row['low'] <= tp_price:
                    return True
        
        return False
    
    def _calculate_results(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive backtest results."""
        equity_df = pd.DataFrame(self.equity_curve)
        
        if len(self.trades) == 0:
            return self._empty_results(data)
        
        # Basic metrics
        final_equity = equity_df['equity'].iloc[-1]
        total_return = (final_equity - self.config.initial_balance) / self.config.initial_balance
        
        # Trade statistics
        completed_trades = [t for t in self.trades if not t.is_open]
        winning_trades = [t for t in completed_trades if t.pnl > 0]
        losing_trades = [t for t in completed_trades if t.pnl < 0]
        
        total_trades = len(completed_trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Average metrics
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Risk/Reward ratio
        risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Drawdown calculations
        equity_series = equity_df['equity']
        rolling_max = equity_series.cummax()
        drawdown = (rolling_max - equity_series) / rolling_max
        max_drawdown = drawdown.max()
        
        # Calculate drawdown duration
        drawdown_durations = []
        in_drawdown = False
        dd_start = 0
        for i in range(len(drawdown)):
            if drawdown.iloc[i] > 0 and not in_drawdown:
                in_drawdown = True
                dd_start = i
            elif drawdown.iloc[i] == 0 and in_drawdown:
                in_drawdown = False
                drawdown_durations.append(i - dd_start)
        
        avg_drawdown_duration = np.mean(drawdown_durations) if drawdown_durations else 0
        
        # Annualized metrics
        if 'time' in equity_df.columns and len(equity_df) > 1:
            time_delta = (equity_df['time'].iloc[-1] - equity_df['time'].iloc[0]).total_seconds()
            years = time_delta / (365.25 * 24 * 3600)
        else:
            years = len(data) / (365 * 24)  # Assume hourly data
        
        if years > 0:
            annualized_return = (1 + total_return) ** (1 / years) - 1
        else:
            annualized_return = total_return
        
        # Returns for Sharpe/Sortino
        equity_df['returns'] = equity_df['equity'].pct_change()
        daily_returns = equity_df['returns'].dropna()
        
        if len(daily_returns) > 1:
            # Assume hourly bars, 24 per day
            returns_std = daily_returns.std() * np.sqrt(24 * 365)
            sharpe_ratio = (annualized_return - 0.02) / returns_std if returns_std > 0 else 0
            
            # Sortino ratio (using downside deviation)
            downside_returns = daily_returns[daily_returns < 0]
            downside_std = downside_returns.std() * np.sqrt(24 * 365) if len(downside_returns) > 0 else 0
            sortino_ratio = (annualized_return - 0.02) / downside_std if downside_std > 0 else 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else float('inf')
        
        # Trade duration
        trade_durations = []
        for t in completed_trades:
            if t.entry_time and t.exit_time:
                duration = (t.exit_time - t.entry_time).total_seconds() / 3600  # Hours
                trade_durations.append(duration)
        
        avg_trade_duration = np.mean(trade_durations) if trade_durations else 0
        
        # Long/Short breakdown
        long_trades = [t for t in completed_trades if t.side == PositionSide.LONG]
        short_trades = [t for t in completed_trades if t.side == PositionSide.SHORT]
        
        long_pnl = sum(t.pnl for t in long_trades)
        short_pnl = sum(t.pnl for t in short_trades)
        
        long_win_rate = len([t for t in long_trades if t.pnl > 0]) / len(long_trades) if long_trades else 0
        short_win_rate = len([t for t in short_trades if t.pnl > 0]) / len(short_trades) if short_trades else 0
        
        # Total fees
        total_fees = sum(t.fees for t in completed_trades)
        
        # Consecutive wins/losses
        pnl_sequence = [t.pnl > 0 for t in completed_trades]
        max_consecutive_wins = self._max_consecutive(pnl_sequence, True)
        max_consecutive_losses = self._max_consecutive(pnl_sequence, False)
        
        return {
            # Summary
            'initial_balance': self.config.initial_balance,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'annualized_return': annualized_return,
            'annualized_return_pct': annualized_return * 100,
            
            # Risk metrics
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'avg_drawdown_duration': avg_drawdown_duration,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            
            # Trade statistics
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'profit_factor': profit_factor,
            'risk_reward_ratio': risk_reward,
            
            # Averages
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade_pnl': (gross_profit - gross_loss) / total_trades if total_trades > 0 else 0,
            'avg_trade_duration_hours': avg_trade_duration,
            
            # Long/Short breakdown
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'long_pnl': long_pnl,
            'short_pnl': short_pnl,
            'long_win_rate': long_win_rate,
            'short_win_rate': short_win_rate,
            
            # Streaks
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            
            # Costs
            'total_fees': total_fees,
            'fees_pct_of_profit': (total_fees / gross_profit * 100) if gross_profit > 0 else 0,
            
            # Data
            'equity_curve': equity_df,
            'trades': self.trades,
            'data': data
        }
    
    def _empty_results(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Return empty results when no trades."""
        return {
            'initial_balance': self.config.initial_balance,
            'final_equity': self.config.initial_balance,
            'total_return': 0,
            'total_return_pct': 0,
            'annualized_return': 0,
            'annualized_return_pct': 0,
            'max_drawdown': 0,
            'max_drawdown_pct': 0,
            'avg_drawdown_duration': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'win_rate_pct': 0,
            'profit_factor': 0,
            'risk_reward_ratio': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'avg_trade_pnl': 0,
            'avg_trade_duration_hours': 0,
            'long_trades': 0,
            'short_trades': 0,
            'long_pnl': 0,
            'short_pnl': 0,
            'long_win_rate': 0,
            'short_win_rate': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'total_fees': 0,
            'fees_pct_of_profit': 0,
            'equity_curve': pd.DataFrame(),
            'trades': [],
            'data': data
        }
    
    def _max_consecutive(self, sequence: List[bool], value: bool) -> int:
        """Find maximum consecutive occurrences of value in sequence."""
        max_count = 0
        current_count = 0
        for item in sequence:
            if item == value:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        return max_count


class LivePaperTrader:
    """
    Paper trading engine for live market simulation.
    Processes real-time price updates without real money.
    """
    
    def __init__(self, config: BacktestConfig = None):
        """Initialize paper trader."""
        self.config = config or BacktestConfig()
        self.reset()
    
    def reset(self):
        """Reset trading state."""
        self.balance = self.config.initial_balance
        self.position = Position()
        self.trades: List[Trade] = []
        self.equity_history = []
        self.max_equity = self.config.initial_balance
        self.current_price = 0.0
    
    def update_price(self, price: float, timestamp: datetime = None):
        """Update current market price."""
        self.current_price = price
        timestamp = timestamp or datetime.now()
        
        # Update unrealized PnL
        if self.position.side != PositionSide.FLAT:
            if self.position.side == PositionSide.LONG:
                price_diff = price - self.position.entry_price
            else:
                price_diff = self.position.entry_price - price
            self.position.unrealized_pnl = price_diff * self.position.size * self.position.leverage
        
        # Record equity
        equity = self.balance + self.position.unrealized_pnl
        self.equity_history.append({
            'time': timestamp,
            'price': price,
            'equity': equity,
            'balance': self.balance,
            'unrealized_pnl': self.position.unrealized_pnl,
            'position': self.position.side.value
        })
        
        if equity > self.max_equity:
            self.max_equity = equity
        
        return equity
    
    def open_long(self, timestamp: datetime = None):
        """Open a long position."""
        if self.position.side != PositionSide.FLAT:
            return False
        
        return self._open_position(PositionSide.LONG, timestamp)
    
    def open_short(self, timestamp: datetime = None):
        """Open a short position."""
        if self.position.side != PositionSide.FLAT:
            return False
        
        return self._open_position(PositionSide.SHORT, timestamp)
    
    def close_position(self, timestamp: datetime = None):
        """Close current position."""
        if self.position.side == PositionSide.FLAT:
            return None
        
        timestamp = timestamp or datetime.now()
        exit_price = self.current_price
        
        # Apply slippage
        if self.position.side == PositionSide.LONG:
            exit_price *= (1 - self.config.slippage_pct / 100)
        else:
            exit_price *= (1 + self.config.slippage_pct / 100)
        
        # Close trade
        if self.trades and self.trades[-1].is_open:
            self.trades[-1].close(timestamp, exit_price, self.config.taker_fee)
            self.balance += self.trades[-1].pnl
        
        trade_result = self.trades[-1] if self.trades else None
        self.position = Position()
        
        return trade_result
    
    def _open_position(self, side: PositionSide, timestamp: datetime = None):
        """Open a new position."""
        timestamp = timestamp or datetime.now()
        entry_price = self.current_price
        
        # Apply slippage
        if side == PositionSide.LONG:
            entry_price *= (1 + self.config.slippage_pct / 100)
        else:
            entry_price *= (1 - self.config.slippage_pct / 100)
        
        position_value = self.balance * (self.config.position_size_pct / 100)
        size = position_value / entry_price
        
        self.position = Position(
            side=side,
            size=size,
            entry_price=entry_price,
            entry_time=timestamp,
            leverage=self.config.leverage,
            margin=position_value / self.config.leverage
        )
        
        self.trades.append(Trade(
            entry_time=timestamp,
            exit_time=None,
            entry_price=entry_price,
            exit_price=None,
            side=side,
            size=size,
            leverage=self.config.leverage
        ))
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current trading statistics."""
        if not self.equity_history:
            return {
                'current_equity': self.config.initial_balance,
                'total_return_pct': 0,
                'total_trades': 0,
                'win_rate': 0,
                'position': 'FLAT'
            }
        
        equity = self.balance + self.position.unrealized_pnl
        completed_trades = [t for t in self.trades if not t.is_open]
        winning_trades = [t for t in completed_trades if t.pnl > 0]
        
        return {
            'current_equity': equity,
            'balance': self.balance,
            'unrealized_pnl': self.position.unrealized_pnl,
            'total_return_pct': ((equity - self.config.initial_balance) / self.config.initial_balance) * 100,
            'max_drawdown_pct': ((self.max_equity - equity) / self.max_equity) * 100 if self.max_equity > 0 else 0,
            'total_trades': len(completed_trades),
            'winning_trades': len(winning_trades),
            'win_rate': len(winning_trades) / len(completed_trades) * 100 if completed_trades else 0,
            'position': self.position.side.value,
            'position_size': self.position.size,
            'entry_price': self.position.entry_price
        }
