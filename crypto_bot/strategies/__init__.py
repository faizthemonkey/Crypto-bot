"""
Trading strategies for the crypto trading bot.
"""
from .base_strategy import BaseStrategy
from .moving_average_crossover import MovingAverageCrossover

__all__ = ['BaseStrategy', 'MovingAverageCrossover']
