"""
Configuration settings for TAI trading system
"""

import os
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path

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

# Global configuration instances
trading_config = TradingConfig()
model_config = ModelConfig()
data_config = DataConfig()

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