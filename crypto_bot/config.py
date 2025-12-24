import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

class Config:
    # API Configuration
    API_KEY = os.getenv('BINANCE_API_KEY')
    API_SECRET = os.getenv('BINANCE_API_SECRET')
    TESTNET = os.getenv('TESTNET', 'False').lower() in ('true', '1', 't')
    
    # Trading Parameters
    SYMBOL = 'BTCUSDT'
    TIMEFRAME = '1h'
    QUANTITY = 0.001  # Default trade size in BTC
    
    # API Endpoints
    if TESTNET:
        BASE_URL = 'https://testnet.binance.vision/api'
        STREAM_URL = 'wss://testnet.binance.vision/ws'
    else:
        BASE_URL = 'https://api.binance.com/api'
        STREAM_URL = 'wss://stream.binance.com:9443/ws'
    
    # Logging
    LOG_LEVEL = 'INFO'
    
    @classmethod
    def validate(cls):
        """Validate that all required configurations are set."""
        required_vars = ['API_KEY', 'API_SECRET']
        missing = [var for var in required_vars if not getattr(cls, var)]
        if missing:
            raise ValueError(f"Missing required configuration: {', '.join(missing)}")
        return True
