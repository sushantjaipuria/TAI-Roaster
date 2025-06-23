"""
Configuration settings for TAI trading system
"""

import os
import json
import logging
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
import asyncio
import httpx
import yfinance as yf
import pandas as pd

# Configure logging for this module
logger = logging.getLogger(__name__)

class TAIConfig:
    """Dynamic configuration system for TAI parameters"""
    
    def __init__(self):
        # Use project-relative path
        config_dir = Path(__file__).parent / "config"
        config_dir.mkdir(exist_ok=True)
        self.config_file = config_dir / "tai_config.json"
        self.load_config()
    
    def load_config(self):
        """Load configuration from file or create defaults"""
        default_config = {
            "risk_thresholds": {
                "high_concentration": 0.25,
                "max_single_stock": 0.20,
                "min_diversification_holdings": 5,
                "max_sector_allocation": 0.30
            },
            "market_cap_thresholds": {
                "large_cap_min": 50000000000,  # 50B INR
                "mid_cap_min": 10000000000     # 10B INR
            },
            "performance_targets": {
                "risk_free_rate": 0.06,
                "market_return": 0.12,
                "min_sharpe_ratio": 1.0
            },
            "model_weights": {
                "xgboost": 0.3,
                "ngboost": 0.3,
                "quantile": 0.2,
                "classifier": 0.2
            },
            "confidence_thresholds": {
                "high": 0.8,
                "medium": 0.6,
                "low": 0.4
            }
        }
        
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
            else:
                self.config = default_config
                self.save_config()
        except Exception as e:
            logger.warning(f"Failed to load config, using defaults: {e}")
            self.config = default_config
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def get(self, key_path: str, default=None):
        """Get configuration value by dot notation path"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

# Initialize TAI configuration
tai_config = TAIConfig()

@dataclass
class TradingConfig:
    """Main trading configuration"""
    
    # Data settings
    data_dir: str = "data"
    models_dir: str = "intelligence/models"
    cache_dir: str = "cache"
    
    # Market settings
    market_caps: List[str] = None
    risk_tolerance: str = "medium"
    investment_amount: float = 100000.0
    
    # Model settings
    prediction_horizons: List[int] = None
    confidence_threshold: float = 0.7
    
    # API settings
    openai_api_key: Optional[str] = None
    dhan_client_id: Optional[str] = None
    dhan_access_token: Optional[str] = None
    
    # Dhan API specific settings
    DHAN_REQUEST_DELAY: float = 0.5  # Delay between requests in seconds
    DHAN_MAX_DAYS_INTRADAY: int = 30  # Maximum days for intraday data requests
    
    # Logging
    log_level: str = "INFO"
    enable_debug: bool = False
    
    def __post_init__(self):
        if self.market_caps is None:
            self.market_caps = ["largecap", "midcap"]
        if self.prediction_horizons is None:
            self.prediction_horizons = [1, 7, 30, 90]
        
        # Load from environment variables
        self.openai_api_key = os.getenv("OPENAI_API_KEY", self.openai_api_key)
        self.dhan_client_id = os.getenv("DHAN_CLIENT_ID", self.dhan_client_id)
        self.dhan_access_token = os.getenv("DHAN_ACCESS_TOKEN", self.dhan_access_token)
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

@dataclass
class ModelConfig:
    """ML Model configuration"""
    
    # XGBoost settings
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    
    # Random Forest settings
    rf_n_estimators: int = 100
    rf_max_depth: int = 10
    
    # NGBoost settings
    ngb_n_estimators: int = 500
    ngb_learning_rate: float = 0.01
    
    # Feature engineering
    technical_indicators: bool = True
    sentiment_analysis: bool = True
    macro_economic: bool = True

@dataclass
class DataConfig:
    """Data processing configuration"""
    
    # Data sources
    use_dhan_api: bool = True
    use_yfinance_fallback: bool = True
    
    # Data processing
    min_data_points: int = 100
    max_lookback_days: int = 1000
    
    # Feature engineering
    feature_lag_days: List[int] = None
    rolling_window_sizes: List[int] = None
    
    def __post_init__(self):
        if self.feature_lag_days is None:
            self.feature_lag_days = [1, 5, 10, 20]
        if self.rolling_window_sizes is None:
            self.rolling_window_sizes = [5, 10, 20, 50]

@dataclass 
class RealTimeMarketData:
    """Comprehensive real-time market data structure"""
    ticker: str
    current_price: float
    previous_close: float
    day_change: float
    day_change_percent: float
    volume: int
    market_cap: float
    pe_ratio: Optional[float]
    pb_ratio: Optional[float]
    roe: Optional[float]
    debt_equity: Optional[float]
    dividend_yield: Optional[float]
    sector: str
    industry: str
    market_cap_category: str

# Global configuration instances
trading_config = TradingConfig()
model_config = ModelConfig()
data_config = DataConfig()

class RealMarketDataProvider:
    """Complete market data provider with caching and fallback"""
    
    def __init__(self):
        self.cache_duration = 300  # 5 minutes
        self.cache = {}
        self.session = None
        self.nse_api_base = "https://www.nseindia.com/api"
    
    async def __aenter__(self):
        self.session = httpx.AsyncClient()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()
    
    @lru_cache(maxsize=1000)
    def _get_ticker_info(self, ticker: str) -> Dict[str, Any]:
        """Get basic ticker information with caching"""
        try:
            symbol = ticker.replace('.NS', '').replace('.BO', '')
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE'),
                'pb_ratio': info.get('priceToBook'),
                'roe': info.get('returnOnEquity'),
                'debt_equity': info.get('debtToEquity'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                'previous_close': info.get('previousClose'),
                'current_price': info.get('currentPrice') or info.get('regularMarketPrice'),
                'volume': info.get('volume')
            }
        except Exception as e:
            logger.warning(f"Failed to get ticker info for {ticker}: {e}")
            return {}
    
    def _categorize_market_cap(self, market_cap: float) -> str:
        """Categorize market cap using configuration thresholds"""
        large_cap_min = tai_config.get('market_cap_thresholds.large_cap_min', 50000000000)
        mid_cap_min = tai_config.get('market_cap_thresholds.mid_cap_min', 10000000000)
        if market_cap > large_cap_min:
            return "Large Cap"
        elif market_cap > mid_cap_min:
            return "Mid Cap"
        else:
            return "Small Cap"
    
    async def get_stock_data(self, ticker: str) -> RealTimeMarketData:
        """Get comprehensive stock data"""
        cache_key = f"stock_data_{ticker}"
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now().timestamp() - timestamp < self.cache_duration:
                return cached_data
        try:
            info = self._get_ticker_info(ticker)
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2d")
            if len(hist) > 0:
                current_price = float(hist['Close'].iloc[-1])
                previous_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
                volume = int(hist['Volume'].iloc[-1])
            else:
                current_price = info.get('current_price', 100.0)
                previous_close = info.get('previous_close', 100.0)
                volume = info.get('volume', 1000000)
            day_change = current_price - previous_close
            day_change_percent = (day_change / previous_close * 100) if previous_close > 0 else 0
            market_cap = info.get('market_cap', 0)
            market_cap_category = self._categorize_market_cap(market_cap)
            data = RealTimeMarketData(
                ticker=ticker,
                current_price=current_price,
                previous_close=previous_close,
                day_change=day_change,
                day_change_percent=day_change_percent,
                volume=volume,
                market_cap=market_cap,
                pe_ratio=info.get('pe_ratio'),
                pb_ratio=info.get('pb_ratio'),
                roe=info.get('roe'),
                debt_equity=info.get('debt_equity'),
                dividend_yield=info.get('dividend_yield'),
                sector=info.get('sector', 'Unknown'),
                industry=info.get('industry', 'Unknown'),
                market_cap_category=market_cap_category
            )
            self.cache[cache_key] = (data, datetime.now().timestamp())
            return data
        except Exception as e:
            logger.error(f"Failed to get stock data for {ticker}: {e}")
            return RealTimeMarketData(
                ticker=ticker,
                current_price=100.0,
                previous_close=98.0,
                day_change=2.0,
                day_change_percent=2.04,
                volume=1000000,
                market_cap=50000000000,
                pe_ratio=22.5,
                pb_ratio=3.2,
                roe=15.2,
                debt_equity=0.6,
                dividend_yield=1.5,
                sector='Unknown',
                industry='Unknown',
                market_cap_category='Large Cap'
            )
    
    async def get_historical_data(self, ticker: str, period: str = "1Y") -> Optional[pd.DataFrame]:
        """Get historical data for analysis"""
        cache_key = f"hist_data_{ticker}_{period}"
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now().timestamp() - timestamp < 3600:  # 1 hour cache
                return cached_data
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period.lower())
            if len(hist) > 0:
                self.cache[cache_key] = (hist, datetime.now().timestamp())
                return hist
            else:
                return None
        except Exception as e:
            logger.error(f"Failed to get historical data for {ticker}: {e}")
            return None
    
    async def get_benchmark_data(self, benchmark: str = "^NSEI") -> Optional[pd.DataFrame]:
        """Get benchmark data for comparison"""
        try:
            benchmark_map = {
                "NIFTY50": "^NSEI",
                "SENSEX": "^BSESN",
                "NIFTY100": "^CNX100",
                "NIFTY500": "^CNX500"
            }
            ticker = benchmark_map.get(benchmark, benchmark)
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")
            return hist if len(hist) > 0 else None
        except Exception as e:
            logger.error(f"Failed to get benchmark data for {benchmark}: {e}")
            return None

# Stock universe definitions
STOCK_UNIVERSE = {
    "largecap": [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", 
        "INFY.NS", "HINDUNILVR.NS", "ITC.NS", "SBIN.NS",
        "BHARTIARTL.NS", "LT.NS", "KOTAKBANK.NS", "WIPRO.NS"
    ],
    "midcap": [
        "ADANIPORTS.NS", "ASIANPAINT.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS",
        "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS"
    ],
    "smallcap": [
        "BANDHANBNK.NS", "BATAINDIA.NS", "BERGEPAINT.NS", "BIOCON.NS"
    ]
}

# Risk tolerance mappings
RISK_PROFILES = {
    "conservative": {
        "max_allocation_per_stock": 0.15,
        "max_sector_allocation": 0.25,
        "min_diversification": 8,
        "preferred_market_caps": ["largecap"]
    },
    "medium": {
        "max_allocation_per_stock": 0.20,
        "max_sector_allocation": 0.30,
        "min_diversification": 6,
        "preferred_market_caps": ["largecap", "midcap"]
    },
    "aggressive": {
        "max_allocation_per_stock": 0.25,
        "max_sector_allocation": 0.35,
        "min_diversification": 5,
        "preferred_market_caps": ["largecap", "midcap", "smallcap"]
    }
}

# Model file paths
MODEL_PATHS = {
    "xgboost": "intelligence/models/xgboost_model.pkl",
    "ngboost": "intelligence/models/ngboost_model.pkl",
    "quantile": "intelligence/models/quantile_model.pkl",
    "classifier": "intelligence/models/classifier_model.pkl"
}

# API endpoints
API_CONFIG = {
    "dhan_base_url": "https://api.dhan.co",
    "openai_base_url": "https://api.openai.com/v1",
    "timeout": 30,
    "max_retries": 3
}

# Feature engineering settings
FEATURE_CONFIG = {
    "technical_indicators": [
        "sma_5", "sma_10", "sma_20", "sma_50",
        "ema_12", "ema_26", "rsi", "macd",
        "bollinger_upper", "bollinger_lower",
        "stochastic_k", "stochastic_d",
        "atr", "adx", "cci", "williams_r"
    ],
    "price_features": [
        "returns_1d", "returns_5d", "returns_10d", "returns_20d",
        "volatility_5d", "volatility_10d", "volatility_20d",
        "volume_ratio", "price_position"
    ],
    "macro_features": [
        "nifty_returns", "bank_nifty_returns", "vix",
        "usd_inr", "crude_oil", "gold_prices"
    ]
}

def get_stock_universe(market_cap: str = None) -> List[str]:
    """Get stock universe for given market cap"""
    if market_cap is None:
        # Return all stocks
        all_stocks = []
        for stocks in STOCK_UNIVERSE.values():
            all_stocks.extend(stocks)
        return list(set(all_stocks))
    
    return STOCK_UNIVERSE.get(market_cap, [])

def get_risk_profile(risk_tolerance: str) -> Dict:
    """Get risk profile configuration"""
    return RISK_PROFILES.get(risk_tolerance, RISK_PROFILES["medium"])

def get_model_path(model_name: str) -> str:
    """Get model file path"""
    return MODEL_PATHS.get(model_name, "")

def get_config():
    """Get global configuration"""
    return {
        "trading": trading_config,
        "model": model_config,
        "data": data_config
    } 