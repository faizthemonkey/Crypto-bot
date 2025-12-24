from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, params: dict = None):
        """Initialize the strategy with parameters."""
        self.params = params or {}
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on the strategy logic.
        
        Args:
            data: DataFrame containing price data
            
        Returns:
            DataFrame with signals (1 for buy, -1 for sell, 0 for hold)
        """
        pass
    
    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators required for the strategy.
        
        Args:
            data: DataFrame containing price data
            
        Returns:
            DataFrame with calculated indicators
        """
        pass
    
    def set_parameters(self, params: dict):
        """Update strategy parameters."""
        self.params.update(params)
        
    def __str__(self):
        return self.__class__.__name__
