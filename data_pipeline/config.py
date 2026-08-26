import os
from decimal import Decimal
from dotenv import load_dotenv

from .binance_contracts import BinanceUniverseRules

load_dotenv()

class Config:
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "crypto_quant")
    DB_DSN = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    CHAIN = "solana"
    TIMEFRAME = "1m" # 也支持 15min
    MIN_LIQUIDITY_USD = 500000.0  
    MIN_FDV = 10000000.0            
    MAX_FDV = float('inf') 
    BIRDEYE_API_KEY = os.getenv("BIRDEYE_API_KEY", "")
    BIRDEYE_BASE_URL = os.getenv("BIRDEYE_BASE_URL", "https://public-api.birdeye.so")
    BASE_URL = BIRDEYE_BASE_URL
    BIRDEYE_IS_PAID = True
    USE_DEXSCREENER = False
    CONCURRENCY = 20
    HISTORY_DAYS = 30
    BINANCE_BASE_URL = os.getenv("BINANCE_BASE_URL", "https://data-api.binance.vision")
    BINANCE_ARCHIVE_BASE_URL = os.getenv(
        "BINANCE_ARCHIVE_BASE_URL",
        "https://data.binance.vision/data/spot/monthly/klines",
    )
    BINANCE_MIN_COVERAGE = float(os.getenv("BINANCE_MIN_COVERAGE", "0.995"))
    BINANCE_RULES = BinanceUniverseRules(
        history_days=int(os.getenv("BINANCE_HISTORY_DAYS", "365")),
        max_symbols=int(os.getenv("BINANCE_MAX_SYMBOLS", "50")),
        minimum_listing_days=int(os.getenv("BINANCE_MIN_LISTING_DAYS", "30")),
        minimum_quote_volume=Decimal(os.getenv("BINANCE_MIN_QUOTE_VOLUME", "10000000")),
    )
