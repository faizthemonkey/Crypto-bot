import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy

class MovingAverageCrossover(BaseStrategy):
    """Moving Average Crossover strategy.
    
    This strategy generates signals based on the crossover of two moving averages.
    A buy signal is generated when the fast MA crosses above the slow MA,
    and a sell signal is generated when the fast MA crosses below the slow MA.
    """
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        """Initialize the strategy with fast and slow moving average periods."""
        super().__init__(params={
            'fast_period': fast_period,
            'slow_period': slow_period
        })
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate fast and slow moving averages."""
        df = data.copy()
        fast_period = self.params['fast_period']
        slow_period = self.params['slow_period']
        
        # Calculate moving averages
        df['fast_ma'] = df['close'].rolling(window=fast_period).mean()
        df['slow_ma'] = df['close'].rolling(window=slow_period).mean()
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on MA crossovers."""
        df = self.calculate_indicators(data)
        
        # Initialize signals column with zeros (no position)
        df['signal'] = 0
        
        # Generate signals (1 for buy, -1 for sell, 0 for hold)
        df.loc[df['fast_ma'] > df['slow_ma'], 'signal'] = 1
        df.loc[df['fast_ma'] <= df['slow_ma'], 'signal'] = -1
        
        # Only keep the points where the signal changes
        df['positions'] = df['signal'].diff()
        
        return df[['signal', 'positions']]
