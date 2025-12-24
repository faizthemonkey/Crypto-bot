"""
Futures Trading Strategies
Multiple indicator-based strategies for futures trading
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from .base_strategy import BaseStrategy
from ..indicators import Indicators


class RSIStrategy(BaseStrategy):
    """
    RSI-based trading strategy.
    
    - Long when RSI crosses below oversold level
    - Short when RSI crosses above overbought level
    - Close when RSI returns to neutral zone
    """
    
    DEFAULT_PARAMS = {
        'rsi_period': 14,
        'overbought': 70,
        'oversold': 30,
        'exit_overbought': 65,
        'exit_oversold': 35
    }
    
    def __init__(self, params: dict = None):
        super().__init__(params)
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['rsi'] = Indicators.rsi(df['close'], self.params['rsi_period'])
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = self.calculate_indicators(data)
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        signals['positions'] = 0
        
        rsi = df['rsi']
        prev_rsi = rsi.shift(1)
        
        # Entry signals
        long_entry = (prev_rsi >= self.params['oversold']) & (rsi < self.params['oversold'])
        short_entry = (prev_rsi <= self.params['overbought']) & (rsi > self.params['overbought'])
        
        # Exit signals
        long_exit = rsi > self.params['exit_overbought']
        short_exit = rsi < self.params['exit_oversold']
        
        # Generate signals
        position = 0
        for i in range(len(signals)):
            if long_entry.iloc[i] and position <= 0:
                signals.iloc[i, signals.columns.get_loc('signal')] = 1
                position = 1
            elif short_entry.iloc[i] and position >= 0:
                signals.iloc[i, signals.columns.get_loc('signal')] = -1
                position = -1
            elif position == 1 and long_exit.iloc[i]:
                signals.iloc[i, signals.columns.get_loc('signal')] = 0
                position = 0
            elif position == -1 and short_exit.iloc[i]:
                signals.iloc[i, signals.columns.get_loc('signal')] = 0
                position = 0
            else:
                signals.iloc[i, signals.columns.get_loc('signal')] = position
            
            signals.iloc[i, signals.columns.get_loc('positions')] = position
        
        return signals


class MACDStrategy(BaseStrategy):
    """
    MACD-based trading strategy.
    
    - Long when MACD crosses above signal line
    - Short when MACD crosses below signal line
    """
    
    DEFAULT_PARAMS = {
        'fast_period': 12,
        'slow_period': 26,
        'signal_period': 9,
        'use_histogram': True
    }
    
    def __init__(self, params: dict = None):
        super().__init__(params)
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        macd, signal, hist = Indicators.macd(
            df['close'],
            self.params['fast_period'],
            self.params['slow_period'],
            self.params['signal_period']
        )
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = self.calculate_indicators(data)
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        signals['positions'] = 0
        
        if self.params['use_histogram']:
            # Use histogram for signals
            hist = df['macd_hist']
            prev_hist = hist.shift(1)
            
            long_signal = (prev_hist < 0) & (hist >= 0)
            short_signal = (prev_hist > 0) & (hist <= 0)
        else:
            # Use MACD/Signal crossover
            macd = df['macd']
            signal_line = df['macd_signal']
            prev_macd = macd.shift(1)
            prev_signal = signal_line.shift(1)
            
            long_signal = (prev_macd <= prev_signal) & (macd > signal_line)
            short_signal = (prev_macd >= prev_signal) & (macd < signal_line)
        
        position = 0
        for i in range(len(signals)):
            if long_signal.iloc[i]:
                signals.iloc[i, signals.columns.get_loc('signal')] = 1
                position = 1
            elif short_signal.iloc[i]:
                signals.iloc[i, signals.columns.get_loc('signal')] = -1
                position = -1
            else:
                signals.iloc[i, signals.columns.get_loc('signal')] = position
            
            signals.iloc[i, signals.columns.get_loc('positions')] = position
        
        return signals


class BollingerBandStrategy(BaseStrategy):
    """
    Bollinger Bands mean reversion strategy.
    
    - Long when price touches lower band
    - Short when price touches upper band
    - Exit at middle band
    """
    
    DEFAULT_PARAMS = {
        'period': 20,
        'std_dev': 2.0,
        'exit_at_middle': True
    }
    
    def __init__(self, params: dict = None):
        super().__init__(params)
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        upper, middle, lower = Indicators.bollinger_bands(
            df['close'],
            self.params['period'],
            self.params['std_dev']
        )
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        df['bb_pct'] = (df['close'] - lower) / (upper - lower)
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = self.calculate_indicators(data)
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        signals['positions'] = 0
        
        close = df['close']
        upper = df['bb_upper']
        middle = df['bb_middle']
        lower = df['bb_lower']
        
        position = 0
        for i in range(len(signals)):
            if pd.isna(upper.iloc[i]):
                signals.iloc[i, signals.columns.get_loc('signal')] = 0
                continue
            
            # Entry conditions
            if close.iloc[i] <= lower.iloc[i] and position <= 0:
                signals.iloc[i, signals.columns.get_loc('signal')] = 1
                position = 1
            elif close.iloc[i] >= upper.iloc[i] and position >= 0:
                signals.iloc[i, signals.columns.get_loc('signal')] = -1
                position = -1
            # Exit at middle band
            elif self.params['exit_at_middle']:
                if position == 1 and close.iloc[i] >= middle.iloc[i]:
                    signals.iloc[i, signals.columns.get_loc('signal')] = 0
                    position = 0
                elif position == -1 and close.iloc[i] <= middle.iloc[i]:
                    signals.iloc[i, signals.columns.get_loc('signal')] = 0
                    position = 0
                else:
                    signals.iloc[i, signals.columns.get_loc('signal')] = position
            else:
                signals.iloc[i, signals.columns.get_loc('signal')] = position
            
            signals.iloc[i, signals.columns.get_loc('positions')] = position
        
        return signals


class EMACrossoverStrategy(BaseStrategy):
    """
    EMA Crossover Strategy.
    
    - Long when fast EMA crosses above slow EMA
    - Short when fast EMA crosses below slow EMA
    """
    
    DEFAULT_PARAMS = {
        'fast_period': 9,
        'slow_period': 21
    }
    
    def __init__(self, params: dict = None):
        super().__init__(params)
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['ema_fast'] = Indicators.ema(df['close'], self.params['fast_period'])
        df['ema_slow'] = Indicators.ema(df['close'], self.params['slow_period'])
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = self.calculate_indicators(data)
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        signals['positions'] = 0
        
        fast = df['ema_fast']
        slow = df['ema_slow']
        prev_fast = fast.shift(1)
        prev_slow = slow.shift(1)
        
        long_signal = (prev_fast <= prev_slow) & (fast > slow)
        short_signal = (prev_fast >= prev_slow) & (fast < slow)
        
        position = 0
        for i in range(len(signals)):
            if long_signal.iloc[i]:
                signals.iloc[i, signals.columns.get_loc('signal')] = 1
                position = 1
            elif short_signal.iloc[i]:
                signals.iloc[i, signals.columns.get_loc('signal')] = -1
                position = -1
            else:
                signals.iloc[i, signals.columns.get_loc('signal')] = position
            
            signals.iloc[i, signals.columns.get_loc('positions')] = position
        
        return signals


class SupertrendStrategy(BaseStrategy):
    """
    Supertrend indicator strategy.
    
    - Long when price is above Supertrend
    - Short when price is below Supertrend
    """
    
    DEFAULT_PARAMS = {
        'period': 10,
        'multiplier': 3.0
    }
    
    def __init__(self, params: dict = None):
        super().__init__(params)
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        supertrend, direction = Indicators.supertrend(
            df['high'], df['low'], df['close'],
            self.params['period'],
            self.params['multiplier']
        )
        df['supertrend'] = supertrend
        df['supertrend_direction'] = direction
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = self.calculate_indicators(data)
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = df['supertrend_direction'].fillna(0).astype(int)
        signals['positions'] = signals['signal']
        return signals


class RSIMACDStrategy(BaseStrategy):
    """
    Combined RSI + MACD strategy.
    
    - Long when RSI < 40 AND MACD crosses above signal
    - Short when RSI > 60 AND MACD crosses below signal
    """
    
    DEFAULT_PARAMS = {
        'rsi_period': 14,
        'rsi_long_threshold': 40,
        'rsi_short_threshold': 60,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9
    }
    
    def __init__(self, params: dict = None):
        super().__init__(params)
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['rsi'] = Indicators.rsi(df['close'], self.params['rsi_period'])
        
        macd, signal, hist = Indicators.macd(
            df['close'],
            self.params['macd_fast'],
            self.params['macd_slow'],
            self.params['macd_signal']
        )
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = self.calculate_indicators(data)
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        signals['positions'] = 0
        
        rsi = df['rsi']
        hist = df['macd_hist']
        prev_hist = hist.shift(1)
        
        macd_cross_up = (prev_hist < 0) & (hist >= 0)
        macd_cross_down = (prev_hist > 0) & (hist <= 0)
        
        long_signal = macd_cross_up & (rsi < self.params['rsi_long_threshold'])
        short_signal = macd_cross_down & (rsi > self.params['rsi_short_threshold'])
        
        position = 0
        for i in range(len(signals)):
            if long_signal.iloc[i]:
                signals.iloc[i, signals.columns.get_loc('signal')] = 1
                position = 1
            elif short_signal.iloc[i]:
                signals.iloc[i, signals.columns.get_loc('signal')] = -1
                position = -1
            else:
                signals.iloc[i, signals.columns.get_loc('signal')] = position
            
            signals.iloc[i, signals.columns.get_loc('positions')] = position
        
        return signals


class StochasticRSIStrategy(BaseStrategy):
    """
    Stochastic RSI strategy.
    
    - Long when Stoch RSI K crosses above D in oversold zone
    - Short when Stoch RSI K crosses below D in overbought zone
    """
    
    DEFAULT_PARAMS = {
        'rsi_period': 14,
        'stoch_period': 14,
        'k_period': 3,
        'd_period': 3,
        'overbought': 80,
        'oversold': 20
    }
    
    def __init__(self, params: dict = None):
        super().__init__(params)
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        k, d = Indicators.stochastic_rsi(
            df['close'],
            self.params['rsi_period'],
            self.params['stoch_period'],
            self.params['k_period'],
            self.params['d_period']
        )
        df['stoch_rsi_k'] = k
        df['stoch_rsi_d'] = d
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = self.calculate_indicators(data)
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        signals['positions'] = 0
        
        k = df['stoch_rsi_k']
        d = df['stoch_rsi_d']
        prev_k = k.shift(1)
        prev_d = d.shift(1)
        
        # Crossovers in zones
        long_signal = (prev_k <= prev_d) & (k > d) & (k < self.params['oversold'])
        short_signal = (prev_k >= prev_d) & (k < d) & (k > self.params['overbought'])
        
        # Exit signals
        long_exit = k > self.params['overbought']
        short_exit = k < self.params['oversold']
        
        position = 0
        for i in range(len(signals)):
            if long_signal.iloc[i] and position <= 0:
                signals.iloc[i, signals.columns.get_loc('signal')] = 1
                position = 1
            elif short_signal.iloc[i] and position >= 0:
                signals.iloc[i, signals.columns.get_loc('signal')] = -1
                position = -1
            elif position == 1 and long_exit.iloc[i]:
                signals.iloc[i, signals.columns.get_loc('signal')] = 0
                position = 0
            elif position == -1 and short_exit.iloc[i]:
                signals.iloc[i, signals.columns.get_loc('signal')] = 0
                position = 0
            else:
                signals.iloc[i, signals.columns.get_loc('signal')] = position
            
            signals.iloc[i, signals.columns.get_loc('positions')] = position
        
        return signals


class TripleEMAStrategy(BaseStrategy):
    """
    Triple EMA Strategy.
    
    Uses 3 EMAs to confirm trend direction.
    - Long when fast > medium > slow
    - Short when fast < medium < slow
    """
    
    DEFAULT_PARAMS = {
        'fast_period': 8,
        'medium_period': 21,
        'slow_period': 55
    }
    
    def __init__(self, params: dict = None):
        super().__init__(params)
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['ema_fast'] = Indicators.ema(df['close'], self.params['fast_period'])
        df['ema_medium'] = Indicators.ema(df['close'], self.params['medium_period'])
        df['ema_slow'] = Indicators.ema(df['close'], self.params['slow_period'])
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = self.calculate_indicators(data)
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        signals['positions'] = 0
        
        fast = df['ema_fast']
        medium = df['ema_medium']
        slow = df['ema_slow']
        
        # Trend conditions
        long_trend = (fast > medium) & (medium > slow)
        short_trend = (fast < medium) & (medium < slow)
        
        position = 0
        for i in range(len(signals)):
            if long_trend.iloc[i] and position <= 0:
                signals.iloc[i, signals.columns.get_loc('signal')] = 1
                position = 1
            elif short_trend.iloc[i] and position >= 0:
                signals.iloc[i, signals.columns.get_loc('signal')] = -1
                position = -1
            elif not long_trend.iloc[i] and not short_trend.iloc[i]:
                signals.iloc[i, signals.columns.get_loc('signal')] = 0
                position = 0
            else:
                signals.iloc[i, signals.columns.get_loc('signal')] = position
            
            signals.iloc[i, signals.columns.get_loc('positions')] = position
        
        return signals


# Strategy registry for easy access
STRATEGY_REGISTRY = {
    'RSI': RSIStrategy,
    'MACD': MACDStrategy,
    'Bollinger Bands': BollingerBandStrategy,
    'EMA Crossover': EMACrossoverStrategy,
    'Supertrend': SupertrendStrategy,
    'RSI + MACD': RSIMACDStrategy,
    'Stochastic RSI': StochasticRSIStrategy,
    'Triple EMA': TripleEMAStrategy
}


def get_strategy(name: str, params: dict = None) -> BaseStrategy:
    """Get strategy instance by name."""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[name](params)


def get_strategy_params(name: str) -> dict:
    """Get default parameters for a strategy."""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}")
    return STRATEGY_REGISTRY[name].DEFAULT_PARAMS.copy()
