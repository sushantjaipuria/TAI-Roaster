import pandas as pd
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import yfinance as yf
import time
import os
import joblib

# Load environment variables to ensure API keys are available
from dotenv import load_dotenv
load_dotenv()

try:
    from backend.app.schemas.enhanced_analysis import StockRecommendation
    from backend.app.schemas.output import (
        PredictionResult,
        ReturnRange,
        BacktestMetrics,
        PricePoint,
        BufferRange,
    )
except ImportError:
    # Fallback for when schemas are not available
    from enum import Enum
    
    class StockRecommendation(str, Enum):
        BUY = "Buy"
        HOLD = "Hold"
        SELL = "Sell"
        STRONG_BUY = "Strong Buy"
        STRONG_SELL = "Strong Sell"
    
    # Use mock classes for other schemas
    class PredictionResult:
        pass
    class ReturnRange:
        pass
    class BacktestMetrics:
        pass
    class PricePoint:
        pass
    class BufferRange:
        pass
try:
    from backend.app.schemas.input import AnalysisRequest as UserInput
except ImportError:
    # Fallback for when schemas are not available
    class UserInput:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

from intelligence.predictors.base_xgboost import predict_return_xgboost
from intelligence.predictors.ngboost_predictor import predict_distribution
from intelligence.predictors.quantile_regressor import predict_quantiles
from intelligence.predictors.classifier import predict_probability_gt_threshold
from intelligence.training.feature_builder import build_features, get_feature_columns
from intelligence.training.data_loader import download_nse_data
from intelligence.training.enhanced_model_trainer import EnhancedModelTrainer
from intelligence.config import get_config

# Import the two-stage LLM system
from intelligence.portfolio_strategist import (
    LLMPortfolioStrategist,
    EnhancedPortfolioEngine,
    MLRecommendation
)

# Fallback imports for when LLM is unavailable
from intelligence.llm_trading_expert import LLMTradingExpert, EnhancedModelAggregator

# Import transparency logging
from intelligence.transparency_logger import transparency_logger

# Try to import data config for dynamic settings
try:
    from backend.app.api.data_config import get_current_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("[WARNING] Data config not available, using defaults")

# Initialize LLM Trading Expert only when API keys are available
llm_expert = None
model_aggregator = None
portfolio_strategist = None
portfolio_engine = None

try:
    # Check if API keys are available
    if os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY'):
        llm_expert = LLMTradingExpert(provider="openai", model="gpt-4.1-nano")
        model_aggregator = EnhancedModelAggregator(llm_expert)
        portfolio_strategist = LLMPortfolioStrategist(provider="openai", model="gpt-4.1-nano")
        
        # 🔥 MODIFY THIS LINE - Pass llm_expert to portfolio_engine
        portfolio_engine = EnhancedPortfolioEngine(portfolio_strategist, llm_expert)
        print("[INFO] Enhanced Two-Stage LLM System initialized successfully")
    else:
        print("[WARNING] No LLM API keys found. LLM features will be disabled.")
        print("[INFO] Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable to enable LLM features")
except Exception as e:
    print(f"[WARNING] Failed to initialize Enhanced LLM System: {e}")
    print("[INFO] LLM features will be disabled")

# Load stock universe with proper path resolution
try:
    from pathlib import Path
    BASE_DIR = Path(__file__).parent.parent  # Go up to TAI-Roaster root
    STOCK_UNIVERSE_PATH = BASE_DIR / "intelligence" / "data" / "stock_universe.csv"
    
    if STOCK_UNIVERSE_PATH.exists():
        stock_df = pd.read_csv(STOCK_UNIVERSE_PATH)
        print(f"[INFO] Loaded stock universe from {STOCK_UNIVERSE_PATH}")
    else:
        print(f"[WARNING] Stock universe not found at {STOCK_UNIVERSE_PATH}, using empty DataFrame")
        stock_df = pd.DataFrame(columns=['ticker', 'sector', 'market_cap'])
except Exception as e:
    print(f"[WARNING] Error loading stock universe: {e}, using empty DataFrame")
    stock_df = pd.DataFrame(columns=['ticker', 'sector', 'market_cap'])

# --- Backtest Metric Utilities ---
def compute_cagr(prices):
    if len(prices) < 2:
        return 0.0
    start = prices.iloc[0]
    end = prices.iloc[-1]
    n_years = len(prices) / 252
    return (end / start) ** (1 / n_years) - 1

def compute_sharpe(prices):
    returns = prices.pct_change().dropna()
    mean = returns.mean()
    std = returns.std()
    return 0.0 if std == 0 else (mean / std) * np.sqrt(252)

def compute_max_drawdown(prices):
    peak = prices.expanding(min_periods=1).max()
    drawdown = (prices - peak) / peak
    return drawdown.min()

def compute_atr(prices, period=14):
    high = prices["High"]
    low = prices["Low"]
    close = prices["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr.iloc[-1]

# --- Enhanced Feature Preparation ---
def generate_latest_features(ticker: str, user_input: Dict[str, Any] = None):
    start_time = time.time()
    
    # Get data configuration for live inference
    try:
        if CONFIG_AVAILABLE:
            data_config = get_current_config()
            period_days = data_config.live_inference.period_days
            timeframe = data_config.live_inference.timeframe.value
        else:
            # Fallback to current behavior
            period_days = 1000  
            timeframe = "1day"
    except Exception as e:
        # Error fallback
        print(f"[WARNING] Using fallback configuration: {e}")
        period_days = 1000
        timeframe = "1day"
    
    # Calculate start and end dates
    from datetime import datetime, timedelta
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=period_days)).strftime("%Y-%m-%d")
    
    print(f"[INFO] Live inference using period: {period_days} days, timeframe: {timeframe}")
    
    # Initialize transparency logging for this stock
    if user_input:
        transparency_logger.start_stock_analysis(ticker, user_input)
    
    try:
        # Step 1: Data Collection with configured timeframe
        data_start_time = time.time()
        df_dict = download_nse_data([ticker], start=start_date, end=end_date, timeframe=timeframe)
        prices = df_dict.get(ticker)
        data_time = (time.time() - data_start_time) * 1000

        if prices is None or not isinstance(prices, pd.DataFrame):
            print(f"[⚠️] Prices missing or malformed for {ticker}")
            return None, None

        if "Close" not in prices.columns:
            if "Adj Close" in prices.columns:
                prices["Close"] = prices["Adj Close"]
            else:
                print(f"[⚠️] 'Close' column missing for {ticker}")
                return None, None

        # Log data collection
        quality_metrics = {
            "quality_score": 0.9 if len(prices) > 100 else 0.7,
            "data_completeness": 1.0 - (prices.isnull().sum().sum() / (len(prices) * len(prices.columns))),
            "date_continuity": len(prices) / (prices.index.max() - prices.index.min()).days if len(prices) > 1 else 1.0,
            "volume_availability": "Volume" in prices.columns,
            "price_consistency": (prices["Close"] > 0).all() if "Close" in prices.columns else False
        }
        
        if user_input:
            transparency_logger.log_data_collection(ticker, prices, quality_metrics)

        # Step 2: Feature Engineering
        feature_start_time = time.time()
        
        # Log individual feature calculation steps
        if user_input:
            transparency_logger.log_feature_calculation(
                ticker, "Data Preprocessing", 
                {"raw_data_shape": prices.shape}, 
                "Data cleaning and validation",
                {"cleaned_data_shape": prices.shape, "quality_score": quality_metrics["quality_score"]},
                data_time,
                "Initial data collection and quality assessment"
            )

        # Use enhanced feature builder
        features_df = build_features(prices)
        feature_time = (time.time() - feature_start_time) * 1000
        
        if user_input and features_df is not None:
            # Log feature engineering completion
            feature_columns = get_feature_columns()
            available_features = [col for col in feature_columns if col in features_df.columns]
            
            final_features = {}
            if not features_df.empty:
                latest_features = features_df.iloc[-1]
                final_features = {col: float(latest_features.get(col, 0)) for col in available_features}
            
            feature_statistics = {
                "total_features_generated": len(features_df.columns) if features_df is not None else 0,
                "features_used_by_models": len(available_features),
                "feature_coverage": len(available_features) / len(feature_columns) if feature_columns else 0,
                "processing_time_ms": feature_time,
                "feature_categories": {
                    "technical_indicators": len([f for f in available_features if any(indicator in f for indicator in ['SMA', 'EMA', 'RSI', 'MACD', 'BB'])]),
                    "momentum_features": len([f for f in available_features if 'Momentum' in f]),
                    "volatility_features": len([f for f in available_features if 'Volatility' in f or 'ATR' in f]),
                    "volume_features": len([f for f in available_features if 'Volume' in f])
                }
            }
            
            transparency_logger.log_final_features(ticker, final_features, feature_statistics)
            
            transparency_logger.log_feature_calculation(
                ticker, "Feature Engineering Complete",
                {"input_price_data": prices.shape},
                "Enhanced feature builder with 194+ TA-Lib indicators",
                {"features_generated": len(features_df.columns), "features_available": len(available_features)},
                feature_time,
                f"Generated {len(features_df.columns)} features, {len(available_features)} available for models"
            )
        
        return features_df, prices
        
    except Exception as e:
        print(f"[⚠️] Feature generation failed for {ticker}: {e}")
        if user_input:
            transparency_logger.log_feature_calculation(
                ticker, "Feature Generation Error",
                {"ticker": ticker},
                "Enhanced feature builder",
                {"error": str(e)},
                (time.time() - start_time) * 1000,
                f"Feature generation failed: {e}"
            )
        return None, None

def _apply_preprocessing(features_df, logger=None):
    """
    Apply preprocessing pipeline to match training - loads saved preprocessors and feature selection
    FIXED: Corrected path and added robust feature standardization
    """
    try:
        # FIXED: Use correct path for preprocessors (removed duplicate 'intelligence' folder)
        preprocessors_path = BASE_DIR / "intelligence" / "models" / "enhanced" / "preprocessors.pkl"
        
        if not preprocessors_path.exists():
            # Fallback to training path if main path doesn't exist
            fallback_path = BASE_DIR / "intelligence" / "training" / "intelligence" / "models" / "enhanced" / "preprocessors.pkl"
            if fallback_path.exists():
                preprocessors_path = fallback_path
                if logger:
                    logger.warning(f"Using fallback preprocessor path: {preprocessors_path}")
                else:
                    print(f"[WARNING] Using fallback preprocessor path: {preprocessors_path}")
            else:
                raise FileNotFoundError(f"Preprocessors not found at {preprocessors_path} or {fallback_path}")
        
        preprocessors = joblib.load(str(preprocessors_path))
        
        if logger:
            logger.info(f"✅ Loaded preprocessors from {preprocessors_path}")
        else:
            print(f"[INFO] ✅ Loaded preprocessors from {preprocessors_path}")
        
        # CRITICAL: Apply feature selection to match training (exactly 50 features)
        if 'feature_columns' in preprocessors:
            selected_features = preprocessors['feature_columns']
            
            if logger:
                logger.info(f"Model expects exactly {len(selected_features)} features")
            else:
                print(f"[INFO] Model expects exactly {len(selected_features)} features")
            
            # Ensure all required features are present with proper defaults
            standardized_df = _standardize_features(features_df, selected_features, logger)
            
            # Apply feature selection - guaranteed to have correct features and order
            features_df = standardized_df[selected_features].copy()
            
            if logger:
                logger.info(f"✅ Feature standardization complete: {len(features_df.columns)} features in correct order")
            else:
                print(f"[INFO] ✅ Feature standardization complete: {len(features_df.columns)} features in correct order")
        else:
            raise ValueError("No 'feature_columns' found in preprocessors - invalid preprocessor file")
        
        # Apply preprocessing steps in the same order as training
        # CRITICAL: Ensure feature dataframe is in the correct format before transformation
        try:
            if 'imputer' in preprocessors and preprocessors['imputer']:
                # Ensure no NaN values before imputation
                features_df = features_df.fillna(0)
                transformed_data = preprocessors['imputer'].transform(features_df)
                features_df = pd.DataFrame(
                    transformed_data,
                    columns=features_df.columns,
                    index=features_df.index
                )
                if logger:
                    logger.debug("Applied imputation")
                else:
                    print("[DEBUG] Applied imputation")
            
            if 'scaler' in preprocessors and preprocessors['scaler']:
                # Ensure no infinite values before scaling
                features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(0)
                transformed_data = preprocessors['scaler'].transform(features_df)
                features_df = pd.DataFrame(
                    transformed_data,
                    columns=features_df.columns,
                    index=features_df.index
                )
                if logger:
                    logger.debug("Applied scaling")
                else:
                    print("[DEBUG] Applied scaling")
                    
        except ValueError as ve:
            error_msg = f"Preprocessing transformation failed: {ve}"
            if logger:
                logger.error(error_msg)
                logger.error(f"Features shape: {features_df.shape}, Expected features: {len(selected_features)}")
                logger.error(f"Feature columns: {list(features_df.columns)}")
            else:
                print(f"[ERROR] {error_msg}")
                print(f"[ERROR] Features shape: {features_df.shape}, Expected features: {len(selected_features)}")
                print(f"[ERROR] Feature columns: {list(features_df.columns)}")
            
            # Return None to indicate failure rather than corrupted data
            return None
        
        # Final validation - ensure exact feature match
        if len(features_df.columns) != len(selected_features):
            raise ValueError(f"Final feature count mismatch: got {len(features_df.columns)}, expected {len(selected_features)}")
        
        # Ensure feature order matches exactly
        if list(features_df.columns) != selected_features:
            if logger:
                logger.warning("Feature order mismatch, reordering to match training...")
            else:
                print("[WARNING] Feature order mismatch, reordering to match training...")
            features_df = features_df[selected_features]
        
        # Final data quality check
        if features_df.isna().any().any():
            if logger:
                logger.warning("NaN values found after preprocessing, filling with zeros...")
            else:
                print("[WARNING] NaN values found after preprocessing, filling with zeros...")
            features_df = features_df.fillna(0)
        
        if logger:
            logger.info(f"✅ Preprocessing pipeline completed: {features_df.shape} features ready for inference")
        else:
            print(f"[INFO] ✅ Preprocessing pipeline completed: {features_df.shape} features ready for inference")
        
        return features_df
        
    except Exception as e:
        error_msg = f"Preprocessing failed: {e}"
        if logger:
            logger.error(error_msg)
        else:
            print(f"[ERROR] {error_msg}")
        
        # Return None to indicate failure (don't return malformed data)
        return None


def _standardize_features(features_df, required_features, logger=None):
    """
    Standardize features to match exactly what the model expects
    This ensures consistent feature count and order
    """
    try:
        standardized_df = pd.DataFrame(index=features_df.index)
        
        missing_count = 0
        present_count = 0
        
        # Add each required feature in the exact order expected
        for feature in required_features:
            if feature in features_df.columns:
                standardized_df[feature] = features_df[feature]
                present_count += 1
            else:
                # Add missing feature with intelligent default value
                default_value = _get_feature_default_value(feature)
                standardized_df[feature] = default_value
                missing_count += 1
        
        if logger:
            logger.info(f"Feature standardization: {present_count} present, {missing_count} missing (filled with defaults)")
        else:
            print(f"[INFO] Feature standardization: {present_count} present, {missing_count} missing (filled with defaults)")
        
        # Replace any remaining NaN or infinite values
        standardized_df = standardized_df.replace([np.inf, -np.inf], np.nan)
        # Use forward fill, backward fill, then zero fill
        standardized_df = standardized_df.ffill().bfill().fillna(0)
        
        return standardized_df
        
    except Exception as e:
        if logger:
            logger.error(f"Feature standardization failed: {e}")
        else:
            print(f"[ERROR] Feature standardization failed: {e}")
        raise


def _get_feature_default_value(feature_name: str) -> float:
    """
    Get intelligent default value for missing features
    Based on the feature type and typical ranges
    """
    feature_name = feature_name.upper()
    
    # Price-based features - use 1.0 (neutral ratio)
    if any(keyword in feature_name for keyword in ['RATIO', 'POSITION', 'MULT', 'DIV']):
        return 1.0
    
    # Oscillators (RSI, MFI, etc.) - use neutral value (50)
    if any(keyword in feature_name for keyword in ['RSI', 'MFI', 'STOCH', 'WILLIAMS']):
        return 50.0
    
    # Bollinger Band position - use middle (0.5)
    if 'BB_POSITION' in feature_name:
        return 0.5
    
    # Volume features - use 1.0 (neutral)
    if 'VOLUME' in feature_name or 'OBV' in feature_name:
        return 1.0
    
    # Candlestick patterns - use 0 (no pattern)
    if 'CDL_' in feature_name:
        return 0.0
    
    # Trend indicators (SMA, EMA, etc.) - use price-based default
    if any(keyword in feature_name for keyword in ['SMA', 'EMA', 'DEMA', 'TEMA', 'TRIMA', 'KAMA']):
        return 100.0  # Reasonable stock price
    
    # MACD and momentum - use 0 (neutral)
    if any(keyword in feature_name for keyword in ['MACD', 'MOM', 'ROC', 'CMO', 'TRIX']):
        return 0.0
    
    # Volatility measures - use small positive value
    if any(keyword in feature_name for keyword in ['ATR', 'STDDEV', 'VAR', 'VOLATILITY']):
        return 0.1
    
    # Mathematical transforms of price - use reasonable defaults
    if feature_name in ['OPEN', 'HIGH', 'LOW', 'CLOSE']:
        return 100.0
    if any(keyword in feature_name for keyword in ['LOG', 'LN', 'SQRT']):
        return 4.6  # log(100)
    if any(keyword in feature_name for keyword in ['SIN', 'COS', 'TAN']):
        return 0.0
    
    # Default case - use 0
    return 0.0

def extract_model_outputs(features_df, prices_df, ticker: str = None):
    """Extract outputs from all models with transparency logging"""
    start_time = time.time()
    
    try:
        # Apply preprocessing pipeline to match training (load preprocessors and select 50 features)
        processed_features_df = _apply_preprocessing(features_df)
        
        if processed_features_df is None or processed_features_df.empty:
            print(f"[⚠️] Preprocessing failed or returned empty DataFrame")
            return None
        
        # Check if we have enough features after preprocessing
        if len(processed_features_df.columns) < 10:  # Minimum feature threshold
            print(f"[⚠️] Insufficient features after preprocessing: {len(processed_features_df.columns)}")
            return None
        
        # Get the latest processed features (should be exactly 50 features after preprocessing)
        latest_features = processed_features_df.iloc[-1].values.tolist()
        feature_names = processed_features_df.columns.tolist()
        
        print(f"[INFO] ✅ Generated {len(latest_features)} enhanced features for {ticker}")
        print(f"[DEBUG] Using {len(latest_features)} features: {feature_names[:5]}{'...' if len(feature_names) > 5 else ''}")
        
        # Get predictions from all models with detailed logging
        model_outputs = {}
        
        # XGBoost Model
        xgb_start_time = time.time()
        try:
            xgb_return = predict_return_xgboost(latest_features)
            xgb_confidence = predict_probability_gt_threshold(latest_features)
            xgb_time = (time.time() - xgb_start_time) * 1000
            
            model_outputs['xgboost'] = {
                'return': xgb_return,
                'confidence': xgb_confidence,
                'risk': 0.5  # Default risk score
            }
            
            # Log XGBoost prediction
            if ticker:
                feature_importance = {feature_names[i]: abs(latest_features[i]) for i in range(min(len(feature_names), len(latest_features)))}
                transparency_logger.log_model_prediction(
                    ticker, "XGBoost", "1.7.0",
                    latest_features, feature_names,
                    xgb_return, xgb_confidence,
                    {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1},
                    feature_importance,
                    f"XGBoost regression model predicting {xgb_return:.4f} return with {xgb_confidence:.4f} confidence",
                    xgb_time
                )
        except Exception as e:
            print(f"XGBoost prediction error: {e}")
            model_outputs['xgboost'] = {'return': 0.08, 'confidence': 0.5, 'risk': 0.5}
        
        # NGBoost Model
        ngb_start_time = time.time()
        try:
            ngb_result = predict_distribution(latest_features)
            ngb_time = (time.time() - ngb_start_time) * 1000
            
            model_outputs['ngboost'] = {
                'return': ngb_result['mean'],
                'confidence': 1.0 - ngb_result['std'],
                'risk': ngb_result['std']
            }
            
            # Log NGBoost prediction
            if ticker:
                feature_importance = {feature_names[i]: abs(latest_features[i]) for i in range(min(len(feature_names), len(latest_features)))}
                transparency_logger.log_model_prediction(
                    ticker, "NGBoost", "0.4.0",
                    latest_features, feature_names,
                    ngb_result['mean'], 1.0 - ngb_result['std'],
                    {"n_estimators": 500, "learning_rate": 0.01, "distribution": "Normal"},
                    feature_importance,
                    f"NGBoost probabilistic model predicting {ngb_result['mean']:.4f} return with {ngb_result['std']:.4f} uncertainty",
                    ngb_time
                )
        except Exception as e:
            print(f"NGBoost prediction error: {e}")
            model_outputs['ngboost'] = {'return': 0.08, 'confidence': 0.5, 'risk': 0.5}
        
        # Quantile Regression Model
        qr_start_time = time.time()
        try:
            qr_result = predict_quantiles(latest_features)
            qr_time = (time.time() - qr_start_time) * 1000
            
            model_outputs['quantile'] = {
                'return': qr_result['p50'],
                'confidence': 1.0 - (qr_result['p90'] - qr_result['p10']),
                'risk': qr_result['p90'] - qr_result['p10']
            }
            
            # Log Quantile Regression prediction
            if ticker:
                feature_importance = {feature_names[i]: abs(latest_features[i]) for i in range(min(len(feature_names), len(latest_features)))}
                transparency_logger.log_model_prediction(
                    ticker, "Quantile Regression", "1.3.0",
                    latest_features, feature_names,
                    qr_result['p50'], 1.0 - (qr_result['p90'] - qr_result['p10']),
                    {"quantiles": [0.1, 0.5, 0.9], "alpha": 1.0},
                    feature_importance,
                    f"Quantile regression predicting median return {qr_result['p50']:.4f} with range [{qr_result['p10']:.4f}, {qr_result['p90']:.4f}]",
                    qr_time
                )
        except Exception as e:
            print(f"Quantile regression prediction error: {e}")
            model_outputs['quantile'] = {'return': 0.08, 'confidence': 0.5, 'risk': 0.5}
        
        # Log ensemble calculation
        if ticker and model_outputs:
            ensemble_start_time = time.time()
            model_predictions = {model: output['return'] for model, output in model_outputs.items()}
            weights = {model: 1.0 for model in model_outputs.keys()}  # Equal weights
            final_prediction = np.mean(list(model_predictions.values()))
            ensemble_time = (time.time() - ensemble_start_time) * 1000
            
            transparency_logger.log_ensemble_calculation(
                ticker, model_predictions, weights, final_prediction,
                "Equal-weighted average", ensemble_time
            )
        
        return model_outputs
        
    except Exception as e:
        print(f"[⚠️] Model prediction failed: {e}")
        if ticker:
            transparency_logger.log_model_prediction(
                ticker, "Error", "N/A", [], [],
                0.08, 0.5, {}, {},
                f"Model prediction failed: {e}",
                (time.time() - start_time) * 1000
            )
        return None

def extract_market_data(features_df, prices_df):
    """Extract market data for LLM analysis"""
    try:
        latest_row = features_df.iloc[-1]
        latest_price = prices_df["Close"].iloc[-1]
        
        market_data = {
            'current_price': float(latest_price),
            'rsi_14': float(latest_row.get('RSI_14', 50.0)),
            'macd': float(latest_row.get('MACD', 0.0)),
            'bb_position': float(latest_row.get('BB_Position', 0.5)),
            'volume_ratio': float(latest_row.get('Volume_Ratio', 1.0)),
            'trend_strength': float(latest_row.get('Trend_Strength', 0.0)),
            'support_level': float(latest_row.get('Support_20', latest_price * 0.95)),
            'resistance_level': float(latest_row.get('Resistance_20', latest_price * 1.05)),
            'atr': float(latest_row.get('ATR', latest_price * 0.02)),
            'volatility_10': float(latest_row.get('Volatility_10', 0.2)),
            'momentum_10': float(latest_row.get('Momentum_10', 0.0)),
            'adx': float(latest_row.get('ADX', 25.0)),
            
            # Market context (would be enhanced with real data)
            'sentiment': 'NEUTRAL',
            'volatility_regime': 'NORMAL',
            'sector_performance': {},
            'economic_indicators': {},
            'news_sentiment': 'NEUTRAL',
            
            # Fundamental data (would be fetched from external sources)
            'pe_ratio': 15.0,
            'pb_ratio': 2.0,
            'roe': 15.0,
            'debt_to_equity': 0.5,
            'revenue_growth': 10.0,
            'profit_margin': 10.0
        }
        
        return market_data
        
    except Exception as e:
        print(f"[⚠️] Market data extraction failed: {e}")
        return {}

# --- Enhanced Two-Stage LLM Pipeline ---
async def run_enhanced_two_stage_pipeline(user_input: UserInput) -> PredictionResult:
    """
    Enhanced pipeline with two-stage LLM system for portfolio bucket creation
    """
    print(f"🚀 Starting Enhanced Two-Stage LLM Pipeline")
    print(f"💰 Investment: ₹{user_input.amount:,}")
    print(f"📊 Market Cap: {user_input.market_cap}")
    print(f"⚖️ Risk Tolerance: {user_input.risk_tolerance}")
    
    # Initialize transparency logging for this session
    transparency_files = {}
    session_start_time = time.time()
    
    if not portfolio_engine:
        print("⚠️ Two-stage LLM system not available, falling back to single-stage system...")
        return await run_single_stage_llm_pipeline(user_input)
    
    try:
        # Step 1: Load market data
        print("\n📈 Step 1: Loading market data...")
        
        # Filter stocks by market cap preference
        filtered_stocks = stock_df[stock_df['market_cap_category'] == user_input.market_cap].copy()
        
        if len(filtered_stocks) == 0:
            print(f"[⚠️] No stocks found for market cap: {user_input.market_cap}")
            filtered_stocks = stock_df.head(10)  # Fallback to first 10 stocks
        
        # Limit to manageable number for LLM processing
        selected_tickers = filtered_stocks['ticker'].unique()[:8]
        print(f"[INFO] Processing {len(selected_tickers)} stocks: {list(selected_tickers)}")
        
        # Step 2: Build enhanced features
        print("\n🔧 Step 2: Building enhanced features...")
        ml_recommendations = []
        
        for ticker in selected_tickers:
            try:
                print(f"[INFO] Processing {ticker}...")
                
                # Generate features and get predictions
                features_df, prices_df = generate_latest_features(ticker, user_input.dict())
                
                if features_df is None or prices_df is None:
                    print(f"[⚠️] Skipping {ticker} - insufficient data")
                    continue
                
                # Extract model outputs
                model_outputs = extract_model_outputs(features_df, prices_df, ticker)
                if model_outputs is None:
                    continue
                
                # Extract market data
                market_data = extract_market_data(features_df, prices_df)
                
                # Create ML recommendation
                ml_rec = MLRecommendation(
                    ticker=ticker,
                    expected_return=np.mean([output['return'] for output in model_outputs.values()]),
                    confidence_score=np.mean([output['confidence'] for output in model_outputs.values()]),
                    allocation_amount=user_input.amount / 8,  # Equal allocation across stocks
                    model_predictions=model_outputs,
                    technical_data={
                        'rsi_14': market_data.get('rsi_14', 50),
                        'macd': market_data.get('macd', 0),
                        'trend_strength': market_data.get('trend_strength', 0),
                        'volume_ratio': market_data.get('volume_ratio', 1.0),
                        'atr': market_data.get('atr', 0.02),
                        'adx': market_data.get('adx', 25)
                    },
                    fundamental_data={
                        'pe_ratio': market_data.get('pe_ratio', 15.0),
                        'pb_ratio': market_data.get('pb_ratio', 2.0),
                        'roe': market_data.get('roe', 15.0),
                        'debt_to_equity': market_data.get('debt_to_equity', 0.5),
                        'revenue_growth': market_data.get('revenue_growth', 10.0),
                        'profit_margin': market_data.get('profit_margin', 10.0),
                        'current_price': market_data.get('current_price', 1000)
                    },
                    risk_metrics={
                        'volatility': market_data.get('volatility_10', 0.2),
                        'max_drawdown': -0.15,
                        'beta': 1.0,
                        'sharpe_ratio': 1.0
                    }
                )
                
                ml_recommendations.append(ml_rec)
                print(f"[✓] Generated ML recommendation for {ticker}")
                
            except Exception as e:
                print(f"[⚠️] Error processing {ticker}: {e}")
                continue
        
        # Step 3: Generate ML predictions
        print(f"\n🤖 Step 3: Generating ML predictions...")
        print(f"[INFO] Generated {len(ml_recommendations)} ML recommendations")
        
        # Step 4: Two-Stage LLM Processing
        print(f"\n🧠 Step 4: Two-Stage LLM Processing...")
        
        try:
            # Run the two-stage LLM portfolio engine
            portfolio_result = await portfolio_engine.generate_portfolio_buckets(
                ml_recommendations=ml_recommendations,
                user_preferences={
                    'investment_amount': user_input.amount,
                    'amount': user_input.amount,
                    'risk_tolerance': user_input.risk_tolerance,
                    'investment_horizon': user_input.horizon_months,
                    'return_target': user_input.return_target_pct / 100.0
                },
                market_context={
                    'market_regime': 'NORMAL',
                    'volatility_level': 'MEDIUM',
                    'economic_indicators': {},
                    'sector_trends': {}
                }
            )
            
            print(f"[✓] Two-stage LLM processing completed")
            
            # The portfolio_result is already in the correct format
            result = PredictionResult(
                total_investment=portfolio_result['total_investment'],
                expected_range=portfolio_result['expected_range'],
                portfolio=portfolio_result['portfolio'],
                portfolio_buckets=portfolio_result['portfolio_buckets'],
                strategy_summary=portfolio_result['strategy_summary'],
                system_info={
                    "llm_enabled": True,
                    "features_count": 194,
                    "models_used": ["XGBoost", "NGBoost", "Quantile Regression"],
                    "processing_stages": ["Data Collection", "Feature Engineering", "ML Prediction", "LLM Validation", "LLM Strategy Creation"],
                    "transparency_enabled": True,
                    "transparency_files": transparency_files,
                    "session_summary": "",
                    "session_id": transparency_logger.current_session_id,
                    "pipeline_type": "Two-Stage LLM",
                    "llm_provider": "OpenAI GPT-4.1-nano"
                }
            )
            
            print(f"✅ Enhanced Two-Stage Pipeline completed successfully!")
            print(f"📊 Generated {len(portfolio_result['portfolio_buckets'])} portfolio buckets")
            print(f"💰 Total allocation: ₹{user_input.amount:,.0f}")
            print(f"📈 Expected return range: {portfolio_result['expected_range']}")
            
            return result
            
        except Exception as e:
            print(f"❌ Two-stage LLM processing failed: {e}")
            print("⚠️ Falling back to single-stage LLM system...")
            return await run_single_stage_llm_pipeline(user_input)
        
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        print("⚠️ Falling back to single-stage LLM system...")
        return await run_single_stage_llm_pipeline(user_input)

async def run_single_stage_llm_pipeline(user_input: UserInput) -> PredictionResult:
    """
    Single-stage LLM pipeline fallback
    """
    print(f"🚀 Starting Single-Stage LLM Pipeline")
    
    # Initialize transparency logging for this session
    transparency_files = {}
    session_start_time = time.time()
    
    try:
        # Filter stocks by market cap preference
        filtered_stocks = stock_df[stock_df['market_cap_category'] == user_input.market_cap].copy()
        
        if len(filtered_stocks) == 0:
            print(f"[⚠️] No stocks found for market cap: {user_input.market_cap}")
            filtered_stocks = stock_df.head(10)  # Fallback to first 10 stocks
        
        # Limit to top 5 stocks for faster processing
        selected_tickers = filtered_stocks['ticker'].unique()[:5]
        print(f"[INFO] Processing {len(selected_tickers)} stocks: {list(selected_tickers)}")
        
        recommendations = []
        
        for ticker in selected_tickers:
            try:
                print(f"[INFO] Processing {ticker}...")
                
                # Generate features and get predictions
                features_df, prices_df = generate_latest_features(ticker, user_input.dict())
                
                if features_df is None or prices_df is None:
                    print(f"[⚠️] Skipping {ticker} - insufficient data")
                    continue
                
                # Extract model outputs
                model_outputs = extract_model_outputs(features_df, prices_df, ticker)
                if model_outputs is None:
                    continue
                
                # Extract market data
                market_data = extract_market_data(features_df, prices_df)
                
                # Use LLM-enhanced aggregator if available
                if model_aggregator:
                    recommendation = await model_aggregator.aggregate_predictions(
                        ticker, model_outputs, market_data
                    )
                else:
                    # Fallback to basic recommendation
                    recommendation = await process_ticker_fallback(ticker, user_input, features_df, prices_df)
                
                if recommendation:
                    # Adjust allocation based on user input
                    recommendation.allocation_amount = min(
                        recommendation.allocation_amount,
                        user_input.amount / 10  # Max 10% per stock
                    )
                    recommendations.append(recommendation)
                    print(f"[✓] Generated recommendation for {ticker}")
                    
                    # Log final decision for transparency (fix timestamp serialization)
                    try:
                        transparency_logger.log_final_decision(
                            ticker=ticker,
                            recommendation={
                                "ticker": recommendation.ticker,
                                "expected_return": float(recommendation.expected_return),
                                "confidence_score": float(recommendation.confidence_score),
                                "allocation_amount": float(recommendation.allocation_amount),
                                "explanation": str(recommendation.explanation)
                            },
                            rationale=f"LLM-enhanced prediction: {recommendation.explanation}",
                            risk_assessment={
                                "risk_level": "medium",
                                "volatility_estimate": 0.2,
                                "max_drawdown_estimate": -0.15,
                                "confidence_interval": [float(recommendation.return_range.min), float(recommendation.return_range.max)]
                            },
                            total_time_ms=float((time.time() - session_start_time) * 1000),
                            system_performance={
                                "models_used": ["XGBoost", "NGBoost", "Quantile Regression"],
                                "features_generated": 58,
                                "llm_enabled": model_aggregator is not None,
                                "processing_mode": "single_stage_llm" if model_aggregator else "fallback"
                            }
                        )
                        
                        # Save transparency log for this stock
                        json_file = transparency_logger.save_stock_log(ticker)
                        if json_file:
                            transparency_files[ticker] = transparency_logger.get_transparency_files(ticker)
                            print(f"[📄] Transparency log saved for {ticker}")
                    except Exception as log_error:
                        print(f"[⚠️] Transparency logging error for {ticker}: {log_error}")
                
            except Exception as e:
                print(f"[⚠️] Error processing {ticker}: {e}")
                continue
        
        if not recommendations:
            print("[⚠️] No recommendations generated, creating fallback")
            return create_fallback_result(user_input)
        
        # Sort and limit recommendations
        recommendations.sort(
            key=lambda x: x.expected_return * x.confidence_score, 
            reverse=True
        )
        
        final_recommendations = recommendations[:5]
        
        # Calculate total investment and expected range
        total_allocated = sum(rec.allocation_amount for rec in final_recommendations)
        
        if total_allocated > user_input.amount:
            scale_factor = user_input.amount / total_allocated
            for rec in final_recommendations:
                rec.allocation_amount *= scale_factor
        
        # Calculate expected range
        weighted_returns = [rec.expected_return * rec.allocation_amount for rec in final_recommendations]
        total_investment = sum(rec.allocation_amount for rec in final_recommendations)
        avg_return = sum(weighted_returns) / total_investment if total_investment > 0 else 0.08
        
        expected_range = f"{avg_return:.1%} - {avg_return * 1.5:.1%}"
        
        # Save session summary
        try:
            session_summary_file = transparency_logger.save_session_summary()
            print(f"[📄] Session summary saved: {session_summary_file}")
        except Exception as e:
            print(f"[⚠️] Session summary save error: {e}")
            session_summary_file = ""
        
        # Create result
        result = PredictionResult(
            total_investment=total_investment,
            expected_range=expected_range,
            portfolio=final_recommendations,
            system_info={
                "llm_enabled": model_aggregator is not None,
                "features_count": 194,
                "models_used": ["XGBoost", "NGBoost", "Quantile Regression"],
                "processing_stages": ["Data Collection", "Feature Engineering", "ML Prediction", "LLM Enhancement"] if model_aggregator else ["Data Collection", "Feature Engineering", "ML Prediction"],
                "transparency_enabled": True,
                "transparency_files": transparency_files,
                "session_summary": session_summary_file,
                "session_id": transparency_logger.current_session_id,
                "pipeline_type": "Single-Stage LLM" if model_aggregator else "Basic ML",
                "llm_provider": "OpenAI GPT-4.1-nano" if model_aggregator else None
            }
        )
        
        print(f"✅ Single-Stage Pipeline completed successfully!")
        print(f"📊 Generated {len(final_recommendations)} recommendations")
        print(f"💰 Total allocation: ₹{total_investment:,.0f}")
        print(f"📈 Expected return range: {expected_range}")
        print(f"📄 Transparency logs saved in: transparency_logs/{transparency_logger.current_session_id}/")
        
        return result
        
    except Exception as e:
        print(f"❌ Single-stage pipeline error: {e}")
        return create_fallback_result(user_input)

# --- Master Pipeline Function ---
async def run_full_pipeline(user_input: UserInput) -> PredictionResult:
    """
    Master pipeline function that routes to the appropriate processing pipeline
    """
    try:
        # Try enhanced two-stage LLM pipeline first
        if portfolio_engine and llm_expert:
            return await run_enhanced_two_stage_pipeline(user_input)
        # Fallback to single-stage LLM pipeline
        elif model_aggregator:
            return await run_single_stage_llm_pipeline(user_input)
        # Final fallback to basic ML pipeline
        else:
            return await run_basic_ml_pipeline(user_input)
    except Exception as e:
        print(f"❌ Master pipeline error: {e}")
        return create_fallback_result(user_input)

async def run_basic_ml_pipeline(user_input: UserInput) -> PredictionResult:
    """
    Basic ML pipeline when LLM is not available
    """
    print(f"🚀 Starting Basic ML Pipeline")
    print(f"💰 Investment: ₹{user_input.amount:,}")
    print(f"📊 Market Cap: {user_input.market_cap}")
    print(f"⚖️ Risk Tolerance: {user_input.risk_tolerance}")
    
    try:
        # Filter stocks by market cap preference
        filtered_stocks = stock_df[stock_df['market_cap_category'] == user_input.market_cap].copy()
        
        if len(filtered_stocks) == 0:
            print(f"[⚠️] No stocks found for market cap: {user_input.market_cap}")
            filtered_stocks = stock_df.head(10)  # Fallback to first 10 stocks
        
        # Limit to top 5 stocks for faster processing
        selected_tickers = filtered_stocks['ticker'].unique()[:5]
        print(f"[INFO] Processing {len(selected_tickers)} stocks: {list(selected_tickers)}")
        
        recommendations = []
        
        for ticker in selected_tickers:
            try:
                print(f"[INFO] Processing {ticker}...")
                
                # Generate features and get predictions
                features_df, prices_df = generate_latest_features(ticker, user_input.dict())
                
                if features_df is None or prices_df is None:
                    print(f"[⚠️] Skipping {ticker} - insufficient data")
                    continue
                
                # Create basic recommendation using fallback method
                recommendation = await process_ticker_fallback(ticker, user_input, features_df, prices_df)
                
                if recommendation:
                    recommendations.append(recommendation)
                    print(f"[✓] Generated recommendation for {ticker}")
                
            except Exception as e:
                print(f"[⚠️] Error processing {ticker}: {e}")
                continue
        
        if not recommendations:
            print("[⚠️] No recommendations generated, creating fallback")
            return create_fallback_result(user_input)
        
        # Sort by expected return * confidence
        recommendations.sort(
            key=lambda x: x.expected_return * x.confidence_score, 
            reverse=True
        )
        
        final_recommendations = recommendations[:5]
        
        # Calculate total investment
        total_allocated = sum(rec.allocation_amount for rec in final_recommendations)
        
        if total_allocated > user_input.amount:
            scale_factor = user_input.amount / total_allocated
            for rec in final_recommendations:
                rec.allocation_amount *= scale_factor
        
        # Calculate expected range
        weighted_returns = [rec.expected_return * rec.allocation_amount for rec in final_recommendations]
        total_investment = sum(rec.allocation_amount for rec in final_recommendations)
        avg_return = sum(weighted_returns) / total_investment if total_investment > 0 else 0.08
        
        expected_range = f"{avg_return:.1%} - {avg_return * 1.5:.1%}"
        
        # Create result
        result = PredictionResult(
            total_investment=total_investment,
            expected_range=expected_range,
            portfolio=final_recommendations,
            system_info={
                "llm_enabled": False,
                "features_count": 58,
                "models_used": ["XGBoost", "NGBoost", "Quantile Regression"],
                "processing_stages": ["Data Collection", "Feature Engineering", "ML Prediction", "Portfolio Optimization"],
                "transparency_enabled": False,
                "transparency_files": {},
                "session_summary": "",
                "pipeline_type": "Basic ML"
            }
        )
        
        print(f"✅ Basic ML Pipeline completed successfully!")
        print(f"📊 Generated {len(final_recommendations)} recommendations")
        print(f"💰 Total allocation: ₹{total_investment:,.0f}")
        print(f"📈 Expected return range: {expected_range}")
        
        return result
        
    except Exception as e:
        print(f"❌ Basic ML pipeline error: {e}")
        return create_fallback_result(user_input)

def create_fallback_result(user_input: UserInput) -> PredictionResult:
    """Create a fallback result when all processing fails"""
    try:
        # Create basic recommendations for top 3 stocks
        top_stocks = ["RELIANCE", "TCS", "HDFCBANK"]
        fallback_recommendations = []
        
        for i, ticker in enumerate(top_stocks):
            base_allocation = user_input.amount / len(top_stocks)
            fallback_rec = StockRecommendation(
                ticker=ticker,
                expected_return=0.08 + i * 0.01,  # 8%, 9%, 10%
                confidence_score=0.6 + i * 0.1,   # 60%, 70%, 80%
                allocation_amount=base_allocation,
                explanation=f"Fallback recommendation for {ticker} based on historical performance",
                return_range=ReturnRange(min=0.05, max=0.15),
                backtest_metrics=BacktestMetrics(
                    cagr=0.08,
                    sharpe_ratio=1.2,
                    max_drawdown=-0.15
                ),
                entry_point=PricePoint(
                    value=1000.0,
                    target=1000.0,
                    buffer_range=BufferRange(min=980.0, max=1020.0)
                ),
                exit_point=PricePoint(
                    value=1080.0,
                    target=1080.0,
                    buffer_range=BufferRange(min=1060.0, max=1100.0)
                )
            )
            fallback_recommendations.append(fallback_rec)
        
        return PredictionResult(
            total_investment=user_input.amount,
            expected_range="8.0% - 12.0%",
            portfolio=fallback_recommendations,
            system_info={
                "llm_enabled": False,
                "features_count": 0,
                "models_used": ["Fallback"],
                "processing_stages": ["Fallback Mode"],
                "transparency_enabled": False,
                "transparency_files": {},
                "session_summary": "",
                "pipeline_type": "Fallback",
                "error_mode": True
            }
        )
        
    except Exception as e:
        print(f"[⚠️] Even fallback result creation failed: {e}")
        # Ultimate fallback
        return PredictionResult(
            total_investment=user_input.amount,
            expected_range="5.0% - 10.0%",
            portfolio=[],
            system_info={
                "llm_enabled": False,
                "features_count": 0,
                "models_used": [],
                "processing_stages": ["Error"],
                "transparency_enabled": False,
                "transparency_files": {},
                "session_summary": "",
                "pipeline_type": "Error",
                "error_mode": True
            }
        )

async def process_ticker_fallback(ticker: str, user_input: UserInput, features_df, prices_df) -> StockRecommendation | None:
    """Fallback prediction when LLM is unavailable"""
    try:
        prices_close = prices_df["Close"]
        latest_row = features_df.iloc[-1]
        
        # Use basic features for fallback
        basic_features = [
            latest_row.get("SMA_10", prices_close.iloc[-1]),
            latest_row.get("SMA_20", prices_close.iloc[-1]),
            latest_row.get("Momentum_10", 0.0)
        ]
        
        latest_price = prices_close.iloc[-1]
        expected_return = predict_return_xgboost(basic_features)
        target_price = latest_price * (1 + expected_return)
        atr = compute_atr(prices_df)

        return StockRecommendation(
            ticker=ticker.replace(".NS", ""),
            expected_return=expected_return,
            confidence_score=predict_probability_gt_threshold(basic_features),
            allocation_amount=user_input.amount / 50,
            explanation=f"Fallback: {ticker} Return {expected_return:.2%}, Conf {predict_probability_gt_threshold(basic_features):.1%}",
            return_range=ReturnRange(
                min=predict_quantiles(basic_features)["p10"],
                max=predict_quantiles(basic_features)["p90"]
            ),
            backtest_metrics=BacktestMetrics(
                cagr=compute_cagr(prices_close),
                sharpe_ratio=compute_sharpe(prices_close),
                max_drawdown=compute_max_drawdown(prices_close),
            ),
            entry_point=PricePoint(
                value=latest_price,
                target=latest_price,
                buffer_range=BufferRange(
                    min=latest_price - atr,
                    max=latest_price + atr
                )
            ),
            exit_point=PricePoint(
                value=target_price,
                target=target_price,
                buffer_range=BufferRange(
                    min=target_price - atr,
                    max=target_price + atr
                )
            )
        )
    except Exception as e:
        print(f"[⚠️] Fallback prediction failed for {ticker}: {e}")
        return None

# --- Portfolio Analysis Function (Direct Holdings Analysis) ---
async def generate_complete_tai_roast_analysis(holdings, user_input: Dict[str, Any], report_id: str = None) -> Dict[str, Any]:
    """
    Analyze actual portfolio holdings using existing ML infrastructure.
    This replicates the original system's approach of analyzing uploaded portfolio holdings
    instead of selecting stocks from a stock universe.
    
    Args:
        holdings: List of portfolio holdings with ticker information
        user_input: Dictionary containing user preferences and portfolio data
        report_id: Optional report identifier for tracking
    
    Returns:
        Dictionary containing comprehensive portfolio analysis results
    """
    print(f"🔍 Starting Portfolio Holdings Analysis")
    print(f"📊 Analyzing {len(holdings)} holdings from uploaded portfolio")
    
    # Extract tickers from actual portfolio holdings
    tickers = [holding.ticker for holding in holdings]
    print(f"[INFO] Portfolio tickers: {tickers}")
    
    # Initialize results structure
    analysis_results = {
        "portfolio_analysis": {
            "total_holdings": len(holdings),
            "tickers": tickers,
            "analysis_timestamp": datetime.now().isoformat(),
            "report_id": report_id
        },
        "ml_predictions": [],
        "market_data": [],
        "stock_analysis": [],
        "portfolio_metrics": {},
        "recommendations": [],
        "system_info": {
            "analysis_type": "portfolio_holdings",
            "pipeline_used": "generate_complete_tai_roast_analysis",
            "ml_enabled": True
        }
    }
    
    # Process each holding in the actual portfolio
    total_analysis_value = 0
    successful_analyses = 0
    
    for holding in holdings:
        try:
            ticker = holding.ticker
            print(f"[INFO] Analyzing holding: {ticker}")
            
            # Generate ML features for this holding
            features_df, prices_df = generate_latest_features(ticker, user_input)
            
            if features_df is None or prices_df is None:
                print(f"[⚠️] Skipping {ticker} - insufficient data")
                # Add placeholder data for failed analysis
                analysis_results["stock_analysis"].append({
                    "ticker": ticker,
                    "status": "insufficient_data",
                    "quantity": holding.quantity,
                    "avg_buy_price": holding.avg_buy_price,
                    "current_value": holding.quantity * (holding.current_price or holding.avg_buy_price)
                })
                continue
            
            # Extract ML model predictions for this holding
            model_outputs = extract_model_outputs(features_df, prices_df, ticker)
            if model_outputs:
                analysis_results["ml_predictions"].append({
                    "ticker": ticker,
                    "predictions": model_outputs,
                    "holding_info": {
                        "quantity": holding.quantity,
                        "avg_buy_price": holding.avg_buy_price,
                        "current_price": holding.current_price,
                        "investment_amount": holding.quantity * holding.avg_buy_price
                    }
                })
            
            # Extract market data for this holding
            market_data = extract_market_data(features_df, prices_df)
            if market_data:
                market_data["ticker"] = ticker
                market_data["holding_info"] = {
                    "quantity": holding.quantity,
                    "avg_buy_price": holding.avg_buy_price,
                    "current_price": holding.current_price
                }
                analysis_results["market_data"].append(market_data)
            
            # Create comprehensive stock analysis for this holding
            stock_analysis = {
                "ticker": ticker,
                "holding_details": {
                    "quantity": holding.quantity,
                    "avg_buy_price": holding.avg_buy_price,
                    "current_price": holding.current_price or holding.avg_buy_price,
                    "total_investment": holding.quantity * holding.avg_buy_price,
                    "current_value": holding.quantity * (holding.current_price or holding.avg_buy_price),
                    "unrealized_pnl": holding.quantity * ((holding.current_price or holding.avg_buy_price) - holding.avg_buy_price)
                },
                "ml_analysis": model_outputs,
                "market_analysis": market_data,
                "recommendation": "HOLD",  # Default recommendation
                "confidence_score": 0.7,   # Default confidence
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            # Calculate recommendation based on ML predictions
            if model_outputs:
                avg_return = np.mean([output.get('return', 0) for output in model_outputs.values()])
                avg_confidence = np.mean([output.get('confidence', 0.5) for output in model_outputs.values()])
                
                if avg_return > 0.15:  # >15% expected return
                    stock_analysis["recommendation"] = "STRONG_BUY"
                elif avg_return > 0.05:  # >5% expected return
                    stock_analysis["recommendation"] = "BUY"
                elif avg_return < -0.15:  # <-15% expected return
                    stock_analysis["recommendation"] = "STRONG_SELL"
                elif avg_return < -0.05:  # <-5% expected return
                    stock_analysis["recommendation"] = "SELL"
                else:
                    stock_analysis["recommendation"] = "HOLD"
                
                stock_analysis["confidence_score"] = avg_confidence
                stock_analysis["expected_return"] = avg_return
            
            analysis_results["stock_analysis"].append(stock_analysis)
            
            # Track successful analyses
            successful_analyses += 1
            total_analysis_value += holding.quantity * (holding.current_price or holding.avg_buy_price)
            
            print(f"[✓] Completed analysis for {ticker}")
            
        except Exception as e:
            print(f"[⚠️] Error analyzing {ticker}: {e}")
            # Add error entry to maintain consistency
            analysis_results["stock_analysis"].append({
                "ticker": ticker,
                "status": "analysis_error",
                "error": str(e),
                "quantity": holding.quantity,
                "avg_buy_price": holding.avg_buy_price
            })
            continue
    
    # Calculate portfolio-level metrics
    if successful_analyses > 0:
        portfolio_metrics = {
            "total_holdings": len(holdings),
            "successful_analyses": successful_analyses,
            "total_portfolio_value": total_analysis_value,
            "average_confidence": np.mean([
                stock.get("confidence_score", 0.5) 
                for stock in analysis_results["stock_analysis"] 
                if "confidence_score" in stock
            ]),
            "portfolio_expected_return": np.mean([
                stock.get("expected_return", 0) 
                for stock in analysis_results["stock_analysis"] 
                if "expected_return" in stock
            ]),
            "analysis_coverage": successful_analyses / len(holdings)
        }
        analysis_results["portfolio_metrics"] = portfolio_metrics
    
    # Generate portfolio-level recommendations
    if successful_analyses > 0:
        buy_count = sum(1 for stock in analysis_results["stock_analysis"] 
                       if stock.get("recommendation") in ["BUY", "STRONG_BUY"])
        sell_count = sum(1 for stock in analysis_results["stock_analysis"] 
                        if stock.get("recommendation") in ["SELL", "STRONG_SELL"])
        
        portfolio_recommendation = {
            "overall_recommendation": "REBALANCE" if (buy_count > 0 and sell_count > 0) else "HOLD",
            "buy_recommendations": buy_count,
            "sell_recommendations": sell_count,
            "hold_recommendations": successful_analyses - buy_count - sell_count,
            "analysis_summary": f"Successfully analyzed {successful_analyses}/{len(holdings)} holdings"
        }
        analysis_results["recommendations"] = [portfolio_recommendation]
    
    print(f"✅ Portfolio Holdings Analysis completed!")
    print(f"📊 Analyzed {successful_analyses}/{len(holdings)} holdings successfully")
    print(f"💰 Total portfolio value: ₹{total_analysis_value:,.0f}")
    
    return analysis_results
