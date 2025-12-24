import logging
from typing import Optional, Dict, Any
import pandas as pd
from .exchange import BinanceClient
from .strategies import BaseStrategy
from .config import Config

class TradingBot:
    """Main trading bot class that handles strategy execution."""
    
    def __init__(self, config: Config):
        """Initialize the trading bot with configuration."""
        self.config = config
        self.exchange = BinanceClient(config)
        self.strategy: Optional[BaseStrategy] = None
        self.logger = self._setup_logging()
    
    def _setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(self.__class__.__name__)
    
    def set_strategy(self, strategy: BaseStrategy):
        """Set the trading strategy to use."""
        self.strategy = strategy
        self.logger.info(f"Strategy set to: {strategy}")
    
    def run_live(self):
        """Run the bot in live trading mode."""
        if not self.strategy:
            raise ValueError("No strategy set. Use set_strategy() first.")
        
        self.logger.info("Starting live trading...")
        
        def handle_klines(msg):
            """Handle incoming kline/websocket messages."""
            try:
                # Extract kline data
                kline = msg['k']
                if kline['x']:  # If candle is closed
                    df = pd.DataFrame([{
                        'open_time': pd.to_datetime(kline['t'], unit='ms'),
                        'open': float(kline['o']),
                        'high': float(kline['h']),
                        'low': float(kline['l']),
                        'close': float(kline['c']),
                        'volume': float(kline['v']),
                        'close_time': pd.to_datetime(kline['T'], unit='ms'),
                    }])
                    
                    # Generate signals
                    signals = self.strategy.generate_signals(df)
                    
                    # Execute trades based on signals
                    self._execute_trades(signals.iloc[-1])
                    
            except Exception as e:
                self.logger.error(f"Error processing kline: {e}", exc_info=True)
        
        # Start websocket
        self.exchange.start_websocket(
            symbol=self.config.SYMBOL,
            interval=self.config.TIMEFRAME,
            callback=handle_klines
        )
    
    def _execute_trades(self, signal):
        """Execute trades based on the generated signals."""
        try:
            if signal['positions'] > 0:  # Buy signal
                self.logger.info(f"Buy signal detected for {self.config.SYMBOL}")
                # Place a market buy order
                order = self.exchange.place_order(
                    symbol=self.config.SYMBOL,
                    side='BUY',
                    order_type='MARKET',
                    quantity=self.config.QUANTITY
                )
                self.logger.info(f"Buy order executed: {order}")
                
            elif signal['positions'] < 0:  # Sell signal
                self.logger.info(f"Sell signal detected for {self.config.SYMBOL}")
                # Place a market sell order
                order = self.exchange.place_order(
                    symbol=self.config.SYMBOL,
                    side='SELL',
                    order_type='MARKET',
                    quantity=self.config.QUANTITY
                )
                self.logger.info(f"Sell order executed: {order}")
                
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}", exc_info=True)
    
    def stop(self):
        """Stop the trading bot and clean up resources."""
        self.exchange.stop_websocket()
        self.logger.info("Trading bot stopped")
