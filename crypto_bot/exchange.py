from binance.client import Client
from binance import ThreadedWebsocketManager
from typing import Dict, List, Optional, Callable
from .config import Config
import pandas as pd

class BinanceClient:
    def __init__(self, config):
        """Initialize the Binance client."""
        self.config = config
        self.client = Client(
            api_key=config.API_KEY,
            api_secret=config.API_SECRET,
            testnet=config.TESTNET
        )
        self.wsm = ThreadedWebsocketManager(api_key=config.API_KEY, api_secret=config.API_SECRET)
        self.wsm.start()
    
    def get_historical_klines(self, symbol: str, interval: str, start_str: str = None, end_str: str = None, limit: int = 500) -> pd.DataFrame:
        """Get historical klines (candlestick data) from Binance."""
        klines = self.client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start_str,
            end_str=end_str,
            limit=limit
        )
        
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert string columns to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
        # Convert timestamp to datetime
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        return df
    
    def get_account_balance(self, asset: str = None) -> Dict:
        """Get account balance for a specific asset or all assets."""
        account = self.client.get_account()
        if asset:
            return next((item for item in account['balances'] if item['asset'] == asset), None)
        return account['balances']
    
    def place_order(self, symbol: str, side: str, order_type: str, **kwargs):
        """Place a new order."""
        return self.client.create_order(
            symbol=symbol,
            side=side,
            type=order_type,
            **kwargs
        )
    
    def start_websocket(self, symbol: str, interval: str, callback: Callable):
        """Start websocket for kline data."""
        self.wsm.start_kline_socket(
            symbol=symbol,
            callback=callback,
            interval=interval
        )
    
    def stop_websocket(self):
        """Stop the websocket."""
        self.wsm.stop()
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'wsm'):
            self.wsm.stop()
