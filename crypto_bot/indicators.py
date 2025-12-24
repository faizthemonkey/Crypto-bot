"""
Technical Indicators Library for Crypto Trading
Provides a comprehensive set of indicators for strategy development
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional


class Indicators:
    """Collection of technical indicators for trading analysis."""
    
    # ==================== TREND INDICATORS ====================
    
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return series.rolling(window=period).mean()
    
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def wma(series: pd.Series, period: int) -> pd.Series:
        """Weighted Moving Average."""
        weights = np.arange(1, period + 1)
        return series.rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    @staticmethod
    def macd(
        series: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD (Moving Average Convergence Divergence).
        
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        fast_ema = series.ewm(span=fast_period, adjust=False).mean()
        slow_ema = series.ewm(span=slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def adx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Average Directional Index.
        
        Returns:
            Tuple of (adx, plus_di, minus_di)
        """
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx, plus_di, minus_di
    
    @staticmethod
    def supertrend(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 10,
        multiplier: float = 3.0
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Supertrend Indicator.
        
        Returns:
            Tuple of (supertrend_line, direction)
            direction: 1 for uptrend, -1 for downtrend
        """
        atr = Indicators.atr(high, low, close, period)
        hl2 = (high + low) / 2
        
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        supertrend = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(index=close.index, dtype=float)
        
        for i in range(period, len(close)):
            if close.iloc[i] > upper_band.iloc[i-1]:
                direction.iloc[i] = 1
            elif close.iloc[i] < lower_band.iloc[i-1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i-1] if i > period else 1
                
                if direction.iloc[i] == 1 and lower_band.iloc[i] < lower_band.iloc[i-1]:
                    lower_band.iloc[i] = lower_band.iloc[i-1]
                elif direction.iloc[i] == -1 and upper_band.iloc[i] > upper_band.iloc[i-1]:
                    upper_band.iloc[i] = upper_band.iloc[i-1]
            
            supertrend.iloc[i] = lower_band.iloc[i] if direction.iloc[i] == 1 else upper_band.iloc[i]
        
        return supertrend, direction
    
    # ==================== MOMENTUM INDICATORS ====================
    
    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator.
        
        Returns:
            Tuple of (%K, %D)
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        stoch_d = stoch_k.rolling(window=d_period).mean()
        
        return stoch_k, stoch_d
    
    @staticmethod
    def stochastic_rsi(
        series: pd.Series,
        rsi_period: int = 14,
        stoch_period: int = 14,
        k_period: int = 3,
        d_period: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic RSI.
        
        Returns:
            Tuple of (stoch_rsi_k, stoch_rsi_d)
        """
        rsi = Indicators.rsi(series, rsi_period)
        
        lowest_rsi = rsi.rolling(window=stoch_period).min()
        highest_rsi = rsi.rolling(window=stoch_period).max()
        
        stoch_rsi = (rsi - lowest_rsi) / (highest_rsi - lowest_rsi)
        stoch_rsi_k = stoch_rsi.rolling(window=k_period).mean() * 100
        stoch_rsi_d = stoch_rsi_k.rolling(window=d_period).mean()
        
        return stoch_rsi_k, stoch_rsi_d
    
    @staticmethod
    def cci(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20
    ) -> pd.Series:
        """Commodity Channel Index."""
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (tp - sma_tp) / (0.015 * mad)
        return cci
    
    @staticmethod
    def williams_r(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """Williams %R."""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr
    
    @staticmethod
    def momentum(series: pd.Series, period: int = 10) -> pd.Series:
        """Momentum Indicator."""
        return series.diff(period)
    
    @staticmethod
    def roc(series: pd.Series, period: int = 10) -> pd.Series:
        """Rate of Change."""
        return ((series - series.shift(period)) / series.shift(period)) * 100
    
    # ==================== VOLATILITY INDICATORS ====================
    
    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def bollinger_bands(
        series: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands.
        
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        return upper, middle, lower
    
    @staticmethod
    def keltner_channels(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20,
        atr_multiplier: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Keltner Channels.
        
        Returns:
            Tuple of (upper_channel, middle_channel, lower_channel)
        """
        middle = close.ewm(span=period, adjust=False).mean()
        atr = Indicators.atr(high, low, close, period)
        
        upper = middle + (atr_multiplier * atr)
        lower = middle - (atr_multiplier * atr)
        
        return upper, middle, lower
    
    @staticmethod
    def donchian_channels(
        high: pd.Series,
        low: pd.Series,
        period: int = 20
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Donchian Channels.
        
        Returns:
            Tuple of (upper_channel, middle_channel, lower_channel)
        """
        upper = high.rolling(window=period).max()
        lower = low.rolling(window=period).min()
        middle = (upper + lower) / 2
        
        return upper, middle, lower
    
    # ==================== VOLUME INDICATORS ====================
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume."""
        return (np.sign(close.diff()) * volume).fillna(0).cumsum()
    
    @staticmethod
    def vwap(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """Volume Weighted Average Price."""
        tp = (high + low + close) / 3
        return (tp * volume).cumsum() / volume.cumsum()
    
    @staticmethod
    def mfi(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """Money Flow Index."""
        tp = (high + low + close) / 3
        raw_mf = tp * volume
        
        mf_sign = np.sign(tp.diff())
        positive_mf = (raw_mf * (mf_sign > 0)).rolling(window=period).sum()
        negative_mf = (raw_mf * (mf_sign < 0)).rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / abs(negative_mf)))
        return mfi
    
    @staticmethod
    def accumulation_distribution(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """Accumulation/Distribution Line."""
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)
        ad = (clv * volume).cumsum()
        return ad
    
    @staticmethod
    def chaikin_oscillator(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        fast_period: int = 3,
        slow_period: int = 10
    ) -> pd.Series:
        """Chaikin Oscillator."""
        ad = Indicators.accumulation_distribution(high, low, close, volume)
        fast_ema = ad.ewm(span=fast_period, adjust=False).mean()
        slow_ema = ad.ewm(span=slow_period, adjust=False).mean()
        return fast_ema - slow_ema
    
    # ==================== SUPPORT/RESISTANCE ====================
    
    @staticmethod
    def pivot_points(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> dict:
        """
        Calculate Standard Pivot Points.
        
        Returns:
            Dictionary with PP, R1, R2, R3, S1, S2, S3
        """
        pp = (high + low + close) / 3
        
        r1 = 2 * pp - low
        s1 = 2 * pp - high
        r2 = pp + (high - low)
        s2 = pp - (high - low)
        r3 = high + 2 * (pp - low)
        s3 = low - 2 * (high - pp)
        
        return {
            'PP': pp,
            'R1': r1, 'R2': r2, 'R3': r3,
            'S1': s1, 'S2': s2, 'S3': s3
        }
    
    @staticmethod
    def fibonacci_retracement(
        high: float,
        low: float,
        is_uptrend: bool = True
    ) -> dict:
        """
        Calculate Fibonacci Retracement Levels.
        
        Returns:
            Dictionary with Fibonacci levels
        """
        diff = high - low
        levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        
        if is_uptrend:
            return {f'{int(l*100)}%': high - diff * l for l in levels}
        else:
            return {f'{int(l*100)}%': low + diff * l for l in levels}
    
    # ==================== CUSTOM COMBINATION INDICATORS ====================
    
    @staticmethod
    def squeeze_momentum(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        bb_period: int = 20,
        bb_std: float = 2.0,
        kc_period: int = 20,
        kc_mult: float = 1.5,
        mom_period: int = 12
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Squeeze Momentum Indicator (LazyBear).
        
        Returns:
            Tuple of (momentum_value, squeeze_on)
            squeeze_on: True when BB is inside KC (squeeze is on)
        """
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = Indicators.bollinger_bands(close, bb_period, bb_std)
        
        # Keltner Channels
        kc_upper, kc_middle, kc_lower = Indicators.keltner_channels(
            high, low, close, kc_period, kc_mult
        )
        
        # Squeeze detection
        squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
        
        # Momentum using linear regression
        hl2 = (high + low) / 2
        sma_hl2 = hl2.rolling(window=kc_period).mean()
        delta = close - sma_hl2
        
        momentum = delta.rolling(window=mom_period).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] * (len(x) - 1) + np.polyfit(range(len(x)), x, 1)[1],
            raw=True
        )
        
        return momentum, squeeze_on
    
    @staticmethod
    def ichimoku_cloud(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52,
        displacement: int = 26
    ) -> dict:
        """
        Ichimoku Cloud.
        
        Returns:
            Dictionary with tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span
        """
        # Tenkan-sen (Conversion Line)
        tenkan_high = high.rolling(window=tenkan_period).max()
        tenkan_low = low.rolling(window=tenkan_period).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (Base Line)
        kijun_high = high.rolling(window=kijun_period).max()
        kijun_low = low.rolling(window=kijun_period).min()
        kijun_sen = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
        
        # Senkou Span B (Leading Span B)
        senkou_b_high = high.rolling(window=senkou_b_period).max()
        senkou_b_low = low.rolling(window=senkou_b_period).min()
        senkou_span_b = ((senkou_b_high + senkou_b_low) / 2).shift(displacement)
        
        # Chikou Span (Lagging Span)
        chikou_span = close.shift(-displacement)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }


# Convenience function to add all indicators to a DataFrame
def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add common indicators to a DataFrame with OHLCV data.
    
    Expects columns: open, high, low, close, volume
    """
    result = df.copy()
    
    # Trend
    result['sma_20'] = Indicators.sma(df['close'], 20)
    result['sma_50'] = Indicators.sma(df['close'], 50)
    result['ema_12'] = Indicators.ema(df['close'], 12)
    result['ema_26'] = Indicators.ema(df['close'], 26)
    
    macd, signal, hist = Indicators.macd(df['close'])
    result['macd'] = macd
    result['macd_signal'] = signal
    result['macd_hist'] = hist
    
    # Momentum
    result['rsi'] = Indicators.rsi(df['close'])
    stoch_k, stoch_d = Indicators.stochastic(df['high'], df['low'], df['close'])
    result['stoch_k'] = stoch_k
    result['stoch_d'] = stoch_d
    
    # Volatility
    result['atr'] = Indicators.atr(df['high'], df['low'], df['close'])
    bb_upper, bb_middle, bb_lower = Indicators.bollinger_bands(df['close'])
    result['bb_upper'] = bb_upper
    result['bb_middle'] = bb_middle
    result['bb_lower'] = bb_lower
    
    # Volume
    if 'volume' in df.columns:
        result['obv'] = Indicators.obv(df['close'], df['volume'])
        result['vwap'] = Indicators.vwap(df['high'], df['low'], df['close'], df['volume'])
    
    return result
