"""
Binance Futures Exchange Client - Uses public API (no authentication required for market data)
"""
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from datetime import datetime
import time
import threading
import websocket
import json


class BinanceFuturesClient:
    """Client for Binance Futures public API - no API keys required for market data."""
    
    BASE_URL = "https://fapi.binance.com"
    WS_URL = "wss://fstream.binance.com/ws"
    
    INTERVALS = {
        "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "8h": "8h", "12h": "12h",
        "1d": "1d", "3d": "3d", "1w": "1w", "1M": "1M"
    }
    
    def __init__(self):
        """Initialize the Futures client - no auth needed for public data."""
        self.session = requests.Session()
        self.ws = None
        self.ws_thread = None
        self.callbacks = {}
        self._running = False
    
    def get_exchange_info(self) -> Dict:
        """Get exchange trading rules and symbol information."""
        response = self.session.get(f"{self.BASE_URL}/fapi/v1/exchangeInfo")
        return response.json()
    
    def get_futures_symbols(self) -> List[str]:
        """Get list of all available futures trading pairs."""
        info = self.get_exchange_info()
        symbols = [s['symbol'] for s in info['symbols'] if s['status'] == 'TRADING']
        return sorted(symbols)
    
    def get_ticker_price(self, symbol: str = None) -> Dict:
        """Get latest price for a symbol or all symbols."""
        params = {}
        if symbol:
            params['symbol'] = symbol
        response = self.session.get(f"{self.BASE_URL}/fapi/v1/ticker/price", params=params)
        return response.json()
    
    def get_ticker_24h(self, symbol: str = None) -> Dict:
        """Get 24hr ticker price change statistics."""
        params = {}
        if symbol:
            params['symbol'] = symbol
        response = self.session.get(f"{self.BASE_URL}/fapi/v1/ticker/24hr", params=params)
        return response.json()
    
    def get_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """Get order book depth."""
        params = {'symbol': symbol, 'limit': limit}
        response = self.session.get(f"{self.BASE_URL}/fapi/v1/depth", params=params)
        return response.json()
    
    def get_recent_trades(self, symbol: str, limit: int = 500) -> List[Dict]:
        """Get recent trades."""
        params = {'symbol': symbol, 'limit': limit}
        response = self.session.get(f"{self.BASE_URL}/fapi/v1/trades", params=params)
        return response.json()
    
    def get_historical_klines(
        self,
        symbol: str,
        interval: str,
        start_time: int = None,
        end_time: int = None,
        limit: int = 500
    ) -> pd.DataFrame:
        """
        Get historical klines/candlestick data.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Kline interval (e.g., '1h', '4h', '1d')
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            limit: Number of klines to fetch (max 1500)
        
        Returns:
            DataFrame with OHLCV data
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': min(limit, 1500)
        }
        
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        
        response = self.session.get(f"{self.BASE_URL}/fapi/v1/klines", params=params)
        klines = response.json()
        
        if not klines:
            return pd.DataFrame()
        
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert types
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
            df[col] = df[col].astype(float)
        
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        df['trades'] = df['trades'].astype(int)
        
        return df
    
    def get_continuous_klines(
        self,
        pair: str,
        contract_type: str,
        interval: str,
        start_time: int = None,
        end_time: int = None,
        limit: int = 500
    ) -> pd.DataFrame:
        """
        Get continuous contract klines.
        
        Args:
            pair: Trading pair (e.g., 'BTCUSDT')
            contract_type: 'PERPETUAL', 'CURRENT_QUARTER', 'NEXT_QUARTER'
            interval: Kline interval
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            limit: Number of klines (max 1500)
        """
        params = {
            'pair': pair,
            'contractType': contract_type,
            'interval': interval,
            'limit': min(limit, 1500)
        }
        
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        
        response = self.session.get(f"{self.BASE_URL}/fapi/v1/continuousKlines", params=params)
        klines = response.json()
        
        if not klines:
            return pd.DataFrame()
        
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
            df[col] = df[col].astype(float)
        
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        return df
    
    def get_funding_rate(self, symbol: str = None, limit: int = 100) -> pd.DataFrame:
        """Get funding rate history."""
        params = {'limit': limit}
        if symbol:
            params['symbol'] = symbol
        
        response = self.session.get(f"{self.BASE_URL}/fapi/v1/fundingRate", params=params)
        data = response.json()
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
        df['fundingRate'] = df['fundingRate'].astype(float)
        
        return df
    
    def get_mark_price(self, symbol: str = None) -> Dict:
        """Get mark price and funding rate."""
        params = {}
        if symbol:
            params['symbol'] = symbol
        response = self.session.get(f"{self.BASE_URL}/fapi/v1/premiumIndex", params=params)
        return response.json()
    
    def get_open_interest(self, symbol: str) -> Dict:
        """Get current open interest."""
        params = {'symbol': symbol}
        response = self.session.get(f"{self.BASE_URL}/fapi/v1/openInterest", params=params)
        return response.json()
    
    def get_top_trader_positions(self, symbol: str, period: str = "5m") -> Dict:
        """Get top trader long/short positions ratio."""
        params = {'symbol': symbol, 'period': period}
        response = self.session.get(
            f"{self.BASE_URL}/futures/data/topLongShortPositionRatio",
            params=params
        )
        return response.json()
    
    # WebSocket methods for live data
    def _on_ws_message(self, ws, message):
        """Handle incoming WebSocket messages."""
        data = json.loads(message)
        stream = data.get('stream', '')
        
        for key, callback in self.callbacks.items():
            if key in stream:
                callback(data.get('data', data))
    
    def _on_ws_error(self, ws, error):
        """Handle WebSocket errors."""
        print(f"WebSocket error: {error}")
    
    def _on_ws_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close."""
        print("WebSocket connection closed")
        self._running = False
    
    def _on_ws_open(self, ws):
        """Handle WebSocket open."""
        print("WebSocket connection established")
    
    def start_kline_stream(self, symbol: str, interval: str, callback: Callable):
        """Start streaming kline data via WebSocket."""
        stream_name = f"{symbol.lower()}@kline_{interval}"
        self.callbacks[f"kline_{interval}"] = callback
        
        ws_url = f"{self.WS_URL}/{stream_name}"
        
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_message=self._on_ws_message,
            on_error=self._on_ws_error,
            on_close=self._on_ws_close,
            on_open=self._on_ws_open
        )
        
        self._running = True
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()
    
    def start_agg_trade_stream(self, symbol: str, callback: Callable):
        """Start streaming aggregated trade data via WebSocket."""
        stream_name = f"{symbol.lower()}@aggTrade"
        self.callbacks["aggTrade"] = callback
        
        ws_url = f"{self.WS_URL}/{stream_name}"
        
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_message=self._on_ws_message,
            on_error=self._on_ws_error,
            on_close=self._on_ws_close,
            on_open=self._on_ws_open
        )
        
        self._running = True
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()
    
    def start_multi_stream(self, streams: List[str], callback: Callable):
        """Start multiple streams via combined WebSocket."""
        stream_str = "/".join(streams)
        self.callbacks["multi"] = callback
        
        ws_url = f"{self.WS_URL}/stream?streams={stream_str}"
        
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_message=self._on_ws_message,
            on_error=self._on_ws_error,
            on_close=self._on_ws_close,
            on_open=self._on_ws_open
        )
        
        self._running = True
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()
    
    def stop_stream(self):
        """Stop the WebSocket stream."""
        self._running = False
        if self.ws:
            self.ws.close()
        if self.ws_thread and self.ws_thread.is_alive():
            self.ws_thread.join(timeout=2)
    
    def __del__(self):
        """Clean up resources."""
        self.stop_stream()
