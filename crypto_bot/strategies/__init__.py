"""
Trading strategies for the crypto trading bot.
"""
from .base_strategy import BaseStrategy
from .moving_average_crossover import MovingAverageCrossover
from .futures_strategies import (
    RSIStrategy,
    MACDStrategy,
    BollingerBandStrategy,
    EMACrossoverStrategy,
    SupertrendStrategy,
    RSIMACDStrategy,
    StochasticRSIStrategy,
    TripleEMAStrategy,
    STRATEGY_REGISTRY,
    get_strategy,
    get_strategy_params
)

__all__ = [
    'BaseStrategy',
    'MovingAverageCrossover',
    'RSIStrategy',
    'MACDStrategy',
    'BollingerBandStrategy',
    'EMACrossoverStrategy',
    'SupertrendStrategy',
    'RSIMACDStrategy',
    'StochasticRSIStrategy',
    'TripleEMAStrategy',
    'STRATEGY_REGISTRY',
    'get_strategy',
    'get_strategy_params'
]
