"""
Application Configuration

Centralized configuration management for the TAI Roaster API:
- Application settings
- API configuration
- File upload settings
- Portfolio validation settings
- Security settings
"""

from pydantic_settings import BaseSettings
from typing import List, Dict, Any, Optional
import os
import sys
from pathlib import Path


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    """
    
    # =============================================================================
    # APPLICATION SETTINGS
    # =============================================================================
    
    APP_NAME: str = "TAI Roaster API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "Portfolio Analysis and Investment Recommendations API"
    DEBUG: bool = True
    ENVIRONMENT: str = "development"  # development, staging, production
    
    # =============================================================================
    # API SETTINGS
    # =============================================================================
    
    API_PREFIX: str = "/api"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1
    
    # =============================================================================
    # CORS SETTINGS
    # =============================================================================
    
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080"
    ]
    
    ALLOWED_METHODS: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    ALLOWED_HEADERS: List[str] = ["*"]
    ALLOW_CREDENTIALS: bool = True
    
    # =============================================================================
    # SECURITY SETTINGS
    # =============================================================================
    
    SECRET_KEY: str = "your-secret-key-change-in-production-make-it-very-long-and-secure"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Rate limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # seconds
    
    # =============================================================================
    # FILE UPLOAD SETTINGS
    # =============================================================================
    
    # File size limits
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    MAX_FILES_PER_REQUEST: int = 5
    
    # Allowed file types
    ALLOWED_FILE_TYPES: List[str] = [".csv", ".xlsx", ".xls", ".tsv"]
    ALLOWED_MIME_TYPES: List[str] = [
        "text/csv",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
        "text/tab-separated-values",
        "application/csv",
        "text/x-csv"
    ]
    
    # Upload directories
    UPLOAD_DIR: str = "uploads"
    TEMP_DIR: str = "temp"
    PROCESSED_DIR: str = "processed"
    
    # File processing
    FILE_RETENTION_DAYS: int = 7
    AUTO_CLEANUP_ENABLED: bool = True
    
    # =============================================================================
    # PORTFOLIO VALIDATION SETTINGS
    # =============================================================================
    
    # Portfolio constraints
    MIN_PORTFOLIO_VALUE: float = 1000.0  # ₹1,000
    MAX_PORTFOLIO_VALUE: float = 100000000.0  # ₹10 crores
    MIN_HOLDINGS_COUNT: int = 1
    MAX_HOLDINGS_COUNT: int = 100
    
    # Diversification settings
    MAX_SINGLE_STOCK_CONCENTRATION: float = 30.0  # 30%
    MAX_SECTOR_CONCENTRATION: float = 50.0  # 50%
    RECOMMENDED_MIN_HOLDINGS: int = 5
    RECOMMENDED_MAX_HOLDINGS: int = 20
    
    # Price validation
    MIN_STOCK_PRICE: float = 0.01  # ₹0.01
    MAX_STOCK_PRICE: float = 100000.0  # ₹1,00,000
    
    # Quantity validation
    MIN_QUANTITY: int = 1
    MAX_QUANTITY: int = 10000000  # 1 crore shares
    
    # Date validation
    MIN_PURCHASE_YEAR: int = 1990
    
    # =============================================================================
    # ANALYSIS SETTINGS
    # =============================================================================
    
    # Processing timeouts
    ANALYSIS_TIMEOUT: int = 300  # 5 minutes
    VALIDATION_TIMEOUT: int = 30  # 30 seconds
    FILE_PARSING_TIMEOUT: int = 60  # 1 minute
    
    # Background processing
    ENABLE_BACKGROUND_PROCESSING: bool = True
    MAX_CONCURRENT_ANALYSES: int = 5
    
    # Caching
    ENABLE_VALIDATION_CACHE: bool = True
    CACHE_TTL_SECONDS: int = 3600  # 1 hour
    MAX_CACHE_SIZE: int = 1000
    
    # =============================================================================
    # MARKET DATA SETTINGS
    # =============================================================================
    
    # NSE/BSE settings
    DEFAULT_EXCHANGE: str = "NSE"
    SUPPORTED_EXCHANGES: List[str] = ["NSE", "BSE"]
    
    # Market data refresh
    MARKET_DATA_REFRESH_INTERVAL: int = 300  # 5 minutes
    ENABLE_REAL_TIME_PRICES: bool = False  # Set to True in production with market data API
    
    # =============================================================================
    # LOGGING SETTINGS
    # =============================================================================
    
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: str = "tai_roaster.log"
    LOG_MAX_SIZE: int = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT: int = 5
    
    # =============================================================================
    # DATABASE SETTINGS (for future use)
    # =============================================================================
    
    DATABASE_URL: str = "sqlite:///./tai_roaster.db"
    DATABASE_ECHO: bool = False
    
    # =============================================================================
    # EXTERNAL API SETTINGS
    # =============================================================================
    
    # Market data APIs (for future integration)
    ALPHA_VANTAGE_API_KEY: str = ""
    YAHOO_FINANCE_ENABLED: bool = False
    POLYGON_API_KEY: str = ""
    
    # Indian market data APIs
    NSE_API_ENABLED: bool = False
    BSE_API_ENABLED: bool = False
    
    # =============================================================================
    # FEATURE FLAGS
    # =============================================================================
    
    # Portfolio features
    ENABLE_PORTFOLIO_IMPORT: bool = True
    ENABLE_PORTFOLIO_EXPORT: bool = True
    ENABLE_PORTFOLIO_SHARING: bool = False
    
    # Analysis features
    ENABLE_ADVANCED_ANALYSIS: bool = True
    ENABLE_ML_RECOMMENDATIONS: bool = False
    ENABLE_RISK_ANALYSIS: bool = True
    
    # =============================================================================
    # INTELLIGENCE MODULE SETTINGS
    # =============================================================================
    
    # Intelligence module configuration
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    ML_MODELS_PATH: str = "intelligence/models/enhanced/"
    INTELLIGENCE_MODULE_PATH: str = "intelligence/"
    MARKET_DATA_CACHE_TTL: int = 300  # 5 minutes
    ENABLE_LLM_ANALYSIS: bool = False
    
    # Model configuration
    MODEL_ENSEMBLE_WEIGHTS: Dict[str, float] = {
        "xgboost": 0.3,
        "lightgbm": 0.25,
        "catboost": 0.25,
        "ngboost": 0.2
    }
    RISK_FREE_RATE: float = 0.06
    MARKET_RETURN: float = 0.12
    CONFIDENCE_THRESHOLD: float = 0.7
    
    # TAI scoring configuration
    TAI_SCORE_WEIGHTS: Dict[str, float] = {
        "performance": 0.25,
        "risk_management": 0.20,
        "diversification": 0.20,
        "ml_confidence": 0.15,
        "liquidity": 0.10,
        "cost_efficiency": 0.10
    }
    
    # Risk thresholds
    HIGH_CONCENTRATION_THRESHOLD: float = 0.25
    MAX_SINGLE_STOCK_ALLOCATION: float = 0.20
    MIN_DIVERSIFICATION_HOLDINGS: int = 5
    MAX_SECTOR_ALLOCATION: float = 0.30
    
    # Market cap thresholds (in INR)
    LARGE_CAP_MIN: float = 50000000000  # ₹50,000 crores
    MID_CAP_MIN: float = 10000000000    # ₹10,000 crores
    
    # Performance targets
    MIN_SHARPE_RATIO: float = 1.0
    TARGET_ANNUAL_RETURN: float = 0.15  # 15%
    MAX_ACCEPTABLE_DRAWDOWN: float = 0.20  # 20%
    
    # Validation features
    ENABLE_STRICT_VALIDATION: bool = True
    ENABLE_TICKER_VALIDATION: bool = True
    ENABLE_PRICE_VALIDATION: bool = True
    
    # =============================================================================
    # NOTIFICATION SETTINGS
    # =============================================================================
    
    # Email settings (for future use)
    SMTP_SERVER: str = ""
    SMTP_PORT: int = 587
    SMTP_USERNAME: str = ""
    SMTP_PASSWORD: str = ""
    EMAIL_FROM: str = "noreply@tairoaster.com"
    
    # Notification preferences
    ENABLE_EMAIL_NOTIFICATIONS: bool = False
    ENABLE_WEBHOOK_NOTIFICATIONS: bool = False
    
    # =============================================================================
    # PERFORMANCE SETTINGS
    # =============================================================================
    
    # Request limits
    MAX_REQUEST_SIZE: int = 50 * 1024 * 1024  # 50MB
    REQUEST_TIMEOUT: int = 30  # seconds
    
    # Memory limits
    MAX_MEMORY_PER_REQUEST: int = 500 * 1024 * 1024  # 500MB
    
    # =============================================================================
    # DEVELOPMENT SETTINGS
    # =============================================================================
    
    # Development features
    ENABLE_AUTO_RELOAD: bool = True
    ENABLE_DEBUG_LOGGING: bool = True
    ENABLE_PROFILING: bool = False
    
    # Testing
    TESTING: bool = False
    TEST_DATABASE_URL: str = "sqlite:///./test_tai_roaster.db"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_directories()
        self._setup_intelligence_module()
    
    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.UPLOAD_DIR,
            self.TEMP_DIR,
            self.PROCESSED_DIR,
            "logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _setup_intelligence_module(self):
        """Setup intelligence module Python path and directories."""
        # Add project root to Python path for intelligence module imports
        project_root = Path(__file__).parent.parent.parent.parent  # Go up to TAI-Roaster root
        if str(project_root) not in sys.path:
            sys.path.append(str(project_root))
        
        # Create only necessary intelligence module subdirectories
        # Note: Other directories exist in their proper module locations:
        # - cache -> backend/cache/
        # - data -> intelligence/data/
        # - models -> intelligence/models/
        # - reports -> backend/reports/
        # - uploads -> backend/uploads/
        intelligence_directories = [
            "intelligence/models/enhanced",
            "intelligence/config"
        ]
        
        for directory in intelligence_directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_cors_settings(self) -> Dict[str, Any]:
        """Get CORS settings for FastAPI."""
        return {
            "allow_origins": self.ALLOWED_ORIGINS,
            "allow_credentials": self.ALLOW_CREDENTIALS,
            "allow_methods": self.ALLOWED_METHODS,
            "allow_headers": self.ALLOWED_HEADERS,
        }
    
    def get_upload_settings(self) -> Dict[str, Any]:
        """Get file upload settings."""
        return {
            "max_file_size": self.MAX_FILE_SIZE,
            "allowed_types": self.ALLOWED_FILE_TYPES,
            "allowed_mime_types": self.ALLOWED_MIME_TYPES,
            "upload_dir": self.UPLOAD_DIR,
            "max_files": self.MAX_FILES_PER_REQUEST
        }
    
    def get_validation_settings(self) -> Dict[str, Any]:
        """Get portfolio validation settings."""
        return {
            "min_portfolio_value": self.MIN_PORTFOLIO_VALUE,
            "max_portfolio_value": self.MAX_PORTFOLIO_VALUE,
            "min_holdings": self.MIN_HOLDINGS_COUNT,
            "max_holdings": self.MAX_HOLDINGS_COUNT,
            "max_single_concentration": self.MAX_SINGLE_STOCK_CONCENTRATION,
            "max_sector_concentration": self.MAX_SECTOR_CONCENTRATION,
            "min_price": self.MIN_STOCK_PRICE,
            "max_price": self.MAX_STOCK_PRICE
        }
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT.lower() == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT.lower() == "development"


# Create global settings instance
settings = Settings()

# Validate critical settings
if settings.is_production():
    assert settings.SECRET_KEY != "your-secret-key-change-in-production-make-it-very-long-and-secure", \
        "Please change the SECRET_KEY in production!"
    assert settings.DEBUG is False, "DEBUG should be False in production!"

# Export commonly used settings
MAX_FILE_SIZE = settings.MAX_FILE_SIZE
ALLOWED_FILE_TYPES = settings.ALLOWED_FILE_TYPES
UPLOAD_DIR = settings.UPLOAD_DIR
API_PREFIX = settings.API_PREFIX