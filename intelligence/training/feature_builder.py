# intelligence/training/feature_builder.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# TA-Lib is now REQUIRED for technical indicators
try:
    import talib
    TALIB_AVAILABLE = True
    print("[DEBUG] Using 'TA-Lib' (talib) for technical indicators")
except ImportError:
    TALIB_AVAILABLE = False
    print("[ERROR] 'TA-Lib' library is required for feature generation")
    print("[INFO] Install with: conda install -c conda-forge ta-lib")
    raise ImportError("TA-Lib is required but not available")

def validate_data_requirements(df: pd.DataFrame) -> bool:
    """
    Comprehensive data validation as per implementation plan
    """
    required_columns = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_columns):
        print(f"[ERROR] Missing required columns. Need: {required_columns}, Have: {df.columns.tolist()}")
        return False
        
    if len(df) < 50:  # Minimum for most indicators
        print(f"[ERROR] Insufficient data: {len(df)} rows. Need at least 50 for reliable indicators")
        return False
        
    # Check for valid price data
    if (df[required_columns] <= 0).any().any():
        print("[ERROR] Invalid price data: found zero or negative values")
        return False
        
    # Check for excessive NaN values
    nan_percentage = df[required_columns].isnull().sum().sum() / (len(df) * len(required_columns))
    if nan_percentage > 0.1:  # More than 10% NaN
        print(f"[ERROR] Too many missing values: {nan_percentage:.1%}")
        return False
        
    return True

def build_features(df: pd.DataFrame, standardize_for_model: bool = True) -> pd.DataFrame:
    """
    Build comprehensive feature set from price data using TA-Lib exclusively
    Implementation of 158+ technical indicators as per plan
    
    Args:
        df: Input DataFrame with OHLCV data
        standardize_for_model: If True, standardize to exactly 50 features for model compatibility
    """
    if df.empty or len(df) < 20:
        print("[WARNING] Insufficient data for feature engineering")
        return pd.DataFrame()
    
    # Validate data requirements
    if not validate_data_requirements(df):
        print("[ERROR] Data validation failed")
        return pd.DataFrame()
    
    try:
        # Create a copy to avoid modifying original data
        features_df = df.copy()
        
        # Create target variable (5-day forward return) only if not standardizing for model
        if not standardize_for_model:
            features_df['Target_5D_Return'] = features_df['Close'].pct_change(5).shift(-5)
        
        # Basic price features
        features_df = add_basic_features(features_df)
        
        # Technical indicators (TA-Lib only)
        features_df = add_talib_indicators(features_df)
        
        # Advanced features
        features_df = add_advanced_features(features_df)
        
        # Enhanced cleaning and validation
        features_df = clean_and_validate_features(features_df)
        
        # CRITICAL FIX: Standardize features for model compatibility
        if standardize_for_model:
            features_df = standardize_features_for_model(features_df)
            print(f"[INFO] ✅ Standardized to {len(features_df.columns)} features for model compatibility")
        else:
            # Remove rows where target is NaN (last 5 rows typically) only if target exists
            if 'Target_5D_Return' in features_df.columns:
                features_df = features_df.dropna(subset=['Target_5D_Return'])
        
        print(f"[DEBUG] Final feature set: {len(features_df.columns)} features for {len(features_df)} samples")
        return features_df
        
    except Exception as e:
        print(f"[ERROR] Feature building failed: {e}")
        return pd.DataFrame()

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic price-based features (non-TA-Lib)"""
    
    # Price ratios
    df['HL_Ratio'] = (df['High'] - df['Low']) / df['Close']
    df['OC_Ratio'] = (df['Open'] - df['Close']) / df['Close']
    
    # Returns (use pandas for basic calculations)
    df['Return_1'] = df['Close'].pct_change(1)
    df['Return_5'] = df['Close'].pct_change(5)
    df['Return_10'] = df['Close'].pct_change(10)
    
    # Rolling volatility (using pandas)
    for period in [5, 10, 20]:
        df[f'Volatility_{period}'] = df['Return_1'].rolling(window=period).std()
    
    # Volume features (if available) - basic calculation
    if 'Volume' in df.columns:
        df['Volume_SMA_10'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_10']
    else:
        df['Volume_Ratio'] = 1.0
    
    return df

def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add comprehensive trend/overlap study indicators"""
    try:
        # Simple Moving Averages - Multiple periods
        for period in [5, 10, 20, 50, 200]:
            df[f'SMA_{period}'] = talib.SMA(df['Close'], timeperiod=period)
            df[f'Price_SMA_{period}_Ratio'] = df['Close'] / df[f'SMA_{period}']
        
        # Exponential Moving Averages - Multiple periods
        for period in [5, 10, 12, 20, 26]:
            df[f'EMA_{period}'] = talib.EMA(df['Close'], timeperiod=period)
        
        # Advanced Moving Averages
        df['WMA_20'] = talib.WMA(df['Close'], timeperiod=20)
        df['DEMA_20'] = talib.DEMA(df['Close'], timeperiod=20)
        df['TEMA_20'] = talib.TEMA(df['Close'], timeperiod=20)
        df['TRIMA_20'] = talib.TRIMA(df['Close'], timeperiod=20)
        df['KAMA_20'] = talib.KAMA(df['Close'], timeperiod=20)
        df['T3_20'] = talib.T3(df['Close'], timeperiod=20)
        
        # MAMA (MESA Adaptive Moving Average)
        df['MAMA'], df['FAMA'] = talib.MAMA(df['Close'])
        
        # Bollinger Bands (preserving existing)
        bb_high, bb_low, bb_mid = talib.BBANDS(df['Close'])
        df['BB_High'] = bb_high
        df['BB_Low'] = bb_low
        df['BB_Mid'] = bb_mid
        df['BB_Width'] = (bb_high - bb_low) / bb_mid
        df['BB_Position'] = (df['Close'] - bb_low) / (bb_high - bb_low)
        
        # Additional Bollinger Bands
        bb_high_50, bb_low_50, bb_mid_50 = talib.BBANDS(df['Close'], timeperiod=50)
        df['BB_High_50'] = bb_high_50
        df['BB_Low_50'] = bb_low_50
        df['BB_Mid_50'] = bb_mid_50
        
        # Parabolic SAR
        df['SAR'] = talib.SAR(df['High'], df['Low'])
        
        # Price Points
        df['MIDPOINT'] = talib.MIDPOINT(df['Close'], timeperiod=14)
        df['MIDPRICE'] = talib.MIDPRICE(df['High'], df['Low'], timeperiod=14)
        
        # Hilbert Transform Trend
        df['HT_TRENDLINE'] = talib.HT_TRENDLINE(df['Close'])
        
    except Exception as e:
        print(f"[WARNING] Some trend indicators failed: {e}")
    
    return df

def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum indicators"""
    try:
        # MACD (preserving existing)
        macd, macd_signal, macd_hist = talib.MACD(df['Close'])
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        df['MACD_Hist'] = macd_hist
        
        # MACD variants
        df['MACD_EXT'], df['MACD_EXT_Signal'], df['MACD_EXT_Hist'] = talib.MACDEXT(df['Close'])
        df['MACD_FIX'], df['MACD_FIX_Signal'], df['MACD_FIX_Hist'] = talib.MACDFIX(df['Close'])
        
        # RSI family (preserving existing)
        df['RSI_14'] = talib.RSI(df['Close'], timeperiod=14)
        df['RSI_7'] = talib.RSI(df['Close'], timeperiod=7)
        df['RSI_21'] = talib.RSI(df['Close'], timeperiod=21)
        
        # Stochastic family (preserving existing)
        df['Stoch_K'], df['Stoch_D'] = talib.STOCH(df['High'], df['Low'], df['Close'])
        df['STOCHF_K'], df['STOCHF_D'] = talib.STOCHF(df['High'], df['Low'], df['Close'])
        df['STOCHRSI_K'], df['STOCHRSI_D'] = talib.STOCHRSI(df['Close'])
        
        # Williams %R (preserving existing)
        df['Williams_R'] = talib.WILLR(df['High'], df['Low'], df['Close'])
        
        # Rate of Change family
        df['MOM_10'] = talib.MOM(df['Close'], timeperiod=10)
        df['MOM_20'] = talib.MOM(df['Close'], timeperiod=20)
        df['ROC_5'] = talib.ROC(df['Close'], timeperiod=5)
        df['ROC_10'] = talib.ROC(df['Close'], timeperiod=10)
        df['ROC_20'] = talib.ROC(df['Close'], timeperiod=20)
        df['ROCP_10'] = talib.ROCP(df['Close'], timeperiod=10)
        df['ROCR_10'] = talib.ROCR(df['Close'], timeperiod=10)
        df['ROCR100_10'] = talib.ROCR100(df['Close'], timeperiod=10)
        
        # ADX family (preserving existing ADX)
        df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'])
        df['ADXR'] = talib.ADXR(df['High'], df['Low'], df['Close'])
        df['DX'] = talib.DX(df['High'], df['Low'], df['Close'])
        df['MINUS_DI'] = talib.MINUS_DI(df['High'], df['Low'], df['Close'])
        df['PLUS_DI'] = talib.PLUS_DI(df['High'], df['Low'], df['Close'])
        df['MINUS_DM'] = talib.MINUS_DM(df['High'], df['Low'])
        df['PLUS_DM'] = talib.PLUS_DM(df['High'], df['Low'])
        
        # Oscillators
        df['APO'] = talib.APO(df['Close'])
        df['PPO'] = talib.PPO(df['Close'])
        df['AROONDOWN'], df['AROONUP'] = talib.AROON(df['High'], df['Low'])
        df['AROONOSC'] = talib.AROONOSC(df['High'], df['Low'])
        df['BOP'] = talib.BOP(df['Open'], df['High'], df['Low'], df['Close'])
        df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'])
        df['CMO'] = talib.CMO(df['Close'])
        df['TRIX'] = talib.TRIX(df['Close'])
        df['ULTOSC'] = talib.ULTOSC(df['High'], df['Low'], df['Close'])
        
    except Exception as e:
        print(f"[WARNING] Some momentum indicators failed: {e}")
    
    return df

def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add volatility indicators"""
    try:
        # ATR family (preserving existing ATR)
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'])
        df['NATR'] = talib.NATR(df['High'], df['Low'], df['Close'])
        df['TRANGE'] = talib.TRANGE(df['High'], df['Low'], df['Close'])
        
    except Exception as e:
        print(f"[WARNING] Volatility indicators failed: {e}")
    
    return df

def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume indicators (when volume available)"""
    try:
        if 'Volume' in df.columns:
            # Existing volume indicators (preserving)
            df['OBV'] = talib.OBV(df['Close'], df['Volume'])
            df['MFI'] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'])
            # VWAP calculation (TA-Lib doesn't have direct VWAP)
            df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
            
            # Additional volume indicators
            df['AD'] = talib.AD(df['High'], df['Low'], df['Close'], df['Volume'])
            df['ADOSC'] = talib.ADOSC(df['High'], df['Low'], df['Close'], df['Volume'])
        else:
            # Create placeholder volume features when volume is not available
            df['OBV'] = 0.0
            df['MFI'] = 50.0  # Neutral MFI
            df['VWAP'] = df['Close']  # Use close price as fallback
            df['AD'] = 0.0
            df['ADOSC'] = 0.0
    
    except Exception as e:
        print(f"[WARNING] Volume indicators failed: {e}")
    
    return df

def add_cycle_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add cycle indicators (Hilbert Transform)"""
    try:
        df['HT_DCPERIOD'] = talib.HT_DCPERIOD(df['Close'])
        df['HT_DCPHASE'] = talib.HT_DCPHASE(df['Close'])
        df['HT_PHASOR_INPHASE'], df['HT_PHASOR_QUADRATURE'] = talib.HT_PHASOR(df['Close'])
        df['HT_SINE'], df['HT_LEADSINE'] = talib.HT_SINE(df['Close'])
        df['HT_TRENDMODE'] = talib.HT_TRENDMODE(df['Close'])
        
    except Exception as e:
        print(f"[WARNING] Cycle indicators failed: {e}")
    
    return df

def add_statistical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add statistical functions"""
    try:
        df['BETA'] = talib.BETA(df['High'], df['Low'], timeperiod=5)
        df['CORREL'] = talib.CORREL(df['High'], df['Low'], timeperiod=30)
        df['LINEARREG'] = talib.LINEARREG(df['Close'], timeperiod=14)
        df['LINEARREG_ANGLE'] = talib.LINEARREG_ANGLE(df['Close'], timeperiod=14)
        df['LINEARREG_INTERCEPT'] = talib.LINEARREG_INTERCEPT(df['Close'], timeperiod=14)
        df['LINEARREG_SLOPE'] = talib.LINEARREG_SLOPE(df['Close'], timeperiod=14)
        df['STDDEV'] = talib.STDDEV(df['Close'], timeperiod=5)
        df['TSF'] = talib.TSF(df['Close'], timeperiod=14)
        df['VAR'] = talib.VAR(df['Close'], timeperiod=5)
        
        # Math Operators
        df['MAX_HIGH_20'] = talib.MAX(df['High'], timeperiod=20)
        df['MIN_LOW_20'] = talib.MIN(df['Low'], timeperiod=20)
        df['MAXINDEX_20'] = talib.MAXINDEX(df['High'], timeperiod=20)
        df['MININDEX_20'] = talib.MININDEX(df['Low'], timeperiod=20)
        df['SUM_VOLUME_10'] = talib.SUM(df['Volume'] if 'Volume' in df.columns else df['Close'], timeperiod=10)
        
    except Exception as e:
        print(f"[WARNING] Statistical indicators failed: {e}")
    
    return df

def add_price_transform_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add price transform and math function indicators"""
    try:
        # Price transform functions
        df['AVGPRICE'] = talib.AVGPRICE(df['Open'], df['High'], df['Low'], df['Close'])
        df['MEDPRICE'] = talib.MEDPRICE(df['High'], df['Low'])
        df['TYPPRICE'] = talib.TYPPRICE(df['High'], df['Low'], df['Close'])
        df['WCLPRICE'] = talib.WCLPRICE(df['High'], df['Low'], df['Close'])
        
        # Math transforms on close price
        df['SQRT_CLOSE'] = talib.SQRT(df['Close'])
        df['LN_CLOSE'] = talib.LN(df['Close'])
        df['LOG10_CLOSE'] = talib.LOG10(df['Close'])
        df['FLOOR_CLOSE'] = talib.FLOOR(df['Close'])
        df['CEIL_CLOSE'] = talib.CEIL(df['Close'])
        
        # Additional math operators
        df['ADD_HIGH_LOW'] = talib.ADD(df['High'], df['Low'])
        df['SUB_HIGH_LOW'] = talib.SUB(df['High'], df['Low'])
        df['MULT_HIGH_LOW'] = talib.MULT(df['High'], df['Low'])
        df['DIV_HIGH_LOW'] = talib.DIV(df['High'], df['Low'])
        
        # Trigonometric functions (normalized to prevent overflow)
        close_norm = df['Close'] / df['Close'].mean()  # Normalize relative to mean
        df['SIN_CLOSE'] = talib.SIN(close_norm)
        df['COS_CLOSE'] = talib.COS(close_norm)
        df['TAN_CLOSE'] = talib.TAN(close_norm)
        
        # Normalized for inverse trig functions (ensure values are between -1 and 1)
        close_max = df['Close'].max()
        close_min = df['Close'].min()
        if close_max > close_min:  # Avoid division by zero
            close_norm_unit = (df['Close'] - close_min) / (close_max - close_min)
            # Scale to [-1, 1] range for asin/acos
            close_norm_unit = 2 * close_norm_unit - 1
            df['ASIN_NORM'] = talib.ASIN(close_norm_unit)
            df['ACOS_NORM'] = talib.ACOS(close_norm_unit)
        else:
            df['ASIN_NORM'] = 0.0
            df['ACOS_NORM'] = 0.0
        
        df['ATAN_CLOSE'] = talib.ATAN(close_norm)
        
    except Exception as e:
        print(f"[WARNING] Price transform indicators failed: {e}")
    
    return df

def add_pattern_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add comprehensive candlestick pattern recognition (40+ patterns)"""
    try:
        # Single candlestick patterns
        df['CDL_DOJI'] = talib.CDLDOJI(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_DRAGONFLY_DOJI'] = talib.CDLDRAGONFLYDOJI(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_GRAVESTONE_DOJI'] = talib.CDLGRAVESTONEDOJI(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_LONG_LEGGED_DOJI'] = talib.CDLLONGLEGGEDDOJI(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_HAMMER'] = talib.CDLHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_INVERTED_HAMMER'] = talib.CDLINVERTEDHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_HANGING_MAN'] = talib.CDLHANGINGMAN(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_SHOOTING_STAR'] = talib.CDLSHOOTINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_MARUBOZU'] = talib.CDLMARUBOZU(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_SPINNING_TOP'] = talib.CDLSPINNINGTOP(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_RICKSHAW_MAN'] = talib.CDLRICKSHAWMAN(df['Open'], df['High'], df['Low'], df['Close'])
        
        # Two candlestick patterns
        df['CDL_ENGULFING'] = talib.CDLENGULFING(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_HARAMI'] = talib.CDLHARAMI(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_HARAMI_CROSS'] = talib.CDLHARAMICROSS(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_PIERCING'] = talib.CDLPIERCING(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_DARK_CLOUD'] = talib.CDLDARKCLOUDCOVER(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_HIKKAKE'] = talib.CDLHIKKAKE(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_IN_NECK'] = talib.CDLINNECK(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_ON_NECK'] = talib.CDLONNECK(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_THRUSTING'] = talib.CDLTHRUSTING(df['Open'], df['High'], df['Low'], df['Close'])
        
        # Three candlestick patterns
        df['CDL_3_BLACK_CROWS'] = talib.CDL3BLACKCROWS(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_3_WHITE_SOLDIERS'] = talib.CDL3WHITESOLDIERS(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_MORNING_STAR'] = talib.CDLMORNINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_EVENING_STAR'] = talib.CDLEVENINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_3_INSIDE'] = talib.CDL3INSIDE(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_3_OUTSIDE'] = talib.CDL3OUTSIDE(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_3_STARS_IN_SOUTH'] = talib.CDL3STARSINSOUTH(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_3_LINE_STRIKE'] = talib.CDL3LINESTRIKE(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_ABANDONED_BABY'] = talib.CDLABANDONEDBABY(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_ADVANCE_BLOCK'] = talib.CDLADVANCEBLOCK(df['Open'], df['High'], df['Low'], df['Close'])
        
        # Additional complex patterns
        df['CDL_BELT_HOLD'] = talib.CDLBELTHOLD(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_BREAKAWAY'] = talib.CDLBREAKAWAY(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_CLOSING_MARUBOZU'] = talib.CDLCLOSINGMARUBOZU(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_CONCEALING_BABY_SWALLOW'] = talib.CDLCONCEALBABYSWALL(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_COUNTERATTACK'] = talib.CDLCOUNTERATTACK(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_DOJI_STAR'] = talib.CDLDOJISTAR(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_GAP_SIDE_BY_SIDE_WHITE'] = talib.CDLGAPSIDESIDEWHITE(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_HOMING_PIGEON'] = talib.CDLHOMINGPIGEON(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_IDENTICAL_3_CROWS'] = talib.CDLIDENTICAL3CROWS(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_KICKING'] = talib.CDLKICKING(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_LADDER_BOTTOM'] = talib.CDLLADDERBOTTOM(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_LONG_LINE'] = talib.CDLLONGLINE(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_MATCHING_LOW'] = talib.CDLMATCHINGLOW(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_RISING_FALLING_3_METHODS'] = talib.CDLRISEFALL3METHODS(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_SEPARATING_LINES'] = talib.CDLSEPARATINGLINES(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_SHORT_LINE'] = talib.CDLSHORTLINE(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_STALLED_PATTERN'] = talib.CDLSTALLEDPATTERN(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_STICK_SANDWICH'] = talib.CDLSTICKSANDWICH(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_TAKURI'] = talib.CDLTAKURI(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_TASUKI_GAP'] = talib.CDLTASUKIGAP(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_TRISTAR'] = talib.CDLTRISTAR(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_UNIQUE_3_RIVER'] = talib.CDLUNIQUE3RIVER(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_UPSIDE_GAP_2_CROWS'] = talib.CDLUPSIDEGAP2CROWS(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_XSIDE_GAP_3_METHODS'] = talib.CDLXSIDEGAP3METHODS(df['Open'], df['High'], df['Low'], df['Close'])
        
    except Exception as e:
        print(f"[WARNING] Pattern indicators failed: {e}")
    
    return df

def add_talib_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add comprehensive technical indicators using TA-Lib"""
    
    try:
        # Call individual category functions
        df = add_trend_indicators(df)
        df = add_momentum_indicators(df) 
        df = add_volatility_indicators(df)
        df = add_volume_indicators(df)
        df = add_cycle_indicators(df)
        df = add_statistical_indicators(df)
        df = add_price_transform_indicators(df)
        df = add_pattern_indicators(df)
        
    except Exception as e:
        print(f"[ERROR] TA-Lib indicators calculation failed: {e}")
        print(f"[INFO] Attempting to continue with partial indicators...")
        
        # Try to calculate minimal required indicators
        try:
            if 'RSI_14' not in df.columns:
                df['RSI_14'] = talib.RSI(df['Close'], timeperiod=14)
            if 'MACD' not in df.columns:
                macd, macd_signal, _ = talib.MACD(df['Close'])
                df['MACD'] = macd
                df['MACD_Signal'] = macd_signal
            if 'ATR' not in df.columns:
                df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'])
        except Exception as e2:
            print(f"[ERROR] Critical indicators also failed: {e2}")
            raise RuntimeError("TA-Lib indicator calculation completely failed") from e
    
    return df

def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add advanced features"""
    
    # Momentum features
    for period in [5, 10, 20]:
        df[f'Momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
    
    # Price position in range
    for period in [10, 20, 50]:
        high_period = df['High'].rolling(window=period).max()
        low_period = df['Low'].rolling(window=period).min()
        df[f'Price_Position_{period}'] = (df['Close'] - low_period) / (high_period - low_period)
    
    # Trend strength
    df['Trend_Strength'] = (df['Close'] - df['Close'].shift(20)) / df['Close'].shift(20)
    
    # Support and resistance levels
    df['Support_20'] = df['Low'].rolling(window=20).min()
    df['Resistance_20'] = df['High'].rolling(window=20).max()
    
    # Gap analysis
    df['Gap'] = (df['Open'] - df['Close'].shift()) / df['Close'].shift()
    
    # Intraday range
    df['Intraday_Range'] = (df['High'] - df['Low']) / df['Open']
    
    # Price acceleration
    df['Price_Acceleration'] = df['Close'].diff().diff()
    
    return df

def clean_and_validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced cleaning and validation as per implementation plan"""
    
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Count NaN values before cleaning
    initial_nan_count = df.isnull().sum().sum()
    
    # Forward fill NaN values (preserves trends)
    df = df.fillna(method='ffill')
    
    # Backward fill remaining NaN at start
    df = df.fillna(method='bfill')
    
    # For remaining NaN values, use intelligent defaults
    for col in df.columns:
        if df[col].isnull().any():
            if 'RSI' in col or 'MFI' in col:
                df[col] = df[col].fillna(50.0)  # Neutral for oscillators
            elif 'BB_Position' in col:
                df[col] = df[col].fillna(0.5)  # Middle of BB
            elif 'Volume' in col:
                df[col] = df[col].fillna(1.0)  # Neutral volume ratio
            elif 'CDL_' in col:
                df[col] = df[col].fillna(0)  # No pattern
            else:
                df[col] = df[col].fillna(0.0)  # Default to zero
    
    # Final check and log cleaning results
    final_nan_count = df.isnull().sum().sum()
    if initial_nan_count > 0:
        print(f"[INFO] Cleaned {initial_nan_count} NaN values, {final_nan_count} remaining")
    
    return df

def get_feature_columns() -> List[str]:
    """
    Return comprehensive list of feature columns available after TA-Lib feature engineering
    """
    # Basic price features (13 features - removed duplicates)
    basic_features = [
        'HL_Ratio', 'OC_Ratio', 'Volume_Ratio',
        'Return_1', 'Return_5', 'Return_10',
        'Volatility_5', 'Volatility_10', 'Volatility_20'
    ]
    
    # Trend indicators (33 features - including SMAs and EMA ratios)
    trend_features = [
        'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200',
        'Price_SMA_5_Ratio', 'Price_SMA_10_Ratio', 'Price_SMA_20_Ratio', 'Price_SMA_50_Ratio', 'Price_SMA_200_Ratio',
        'EMA_5', 'EMA_10', 'EMA_12', 'EMA_20', 'EMA_26',
        'WMA_20', 'DEMA_20', 'TEMA_20', 'TRIMA_20', 'KAMA_20', 'T3_20',
        'MAMA', 'FAMA', 'BB_High', 'BB_Low', 'BB_Mid', 'BB_Width', 'BB_Position',
        'BB_High_50', 'BB_Low_50', 'BB_Mid_50', 'SAR', 'MIDPOINT', 'MIDPRICE', 'HT_TRENDLINE'
    ]
    
    # Momentum indicators (44 features)
    momentum_features = [
        'MACD', 'MACD_Signal', 'MACD_Hist', 'MACD_EXT', 'MACD_EXT_Signal', 'MACD_EXT_Hist',
        'MACD_FIX', 'MACD_FIX_Signal', 'MACD_FIX_Hist', 'RSI_14', 'RSI_7', 'RSI_21',
        'Stoch_K', 'Stoch_D', 'STOCHF_K', 'STOCHF_D', 'STOCHRSI_K', 'STOCHRSI_D',
        'Williams_R', 'MOM_10', 'MOM_20', 'ROC_5', 'ROC_10', 'ROC_20', 'ROCP_10',
        'ROCR_10', 'ROCR100_10', 'ADX', 'ADXR', 'DX', 'MINUS_DI', 'PLUS_DI',
        'MINUS_DM', 'PLUS_DM', 'APO', 'PPO', 'AROONDOWN', 'AROONUP', 'AROONOSC',
        'BOP', 'CCI', 'CMO', 'TRIX', 'ULTOSC'
    ]
    
    # Volatility indicators (3 features)
    volatility_features = [
        'ATR', 'NATR', 'TRANGE'
    ]
    
    # Volume indicators (5 features)
    volume_features = [
        'OBV', 'MFI', 'VWAP', 'AD', 'ADOSC'
    ]
    
    # Cycle indicators (7 features)
    cycle_features = [
        'HT_DCPERIOD', 'HT_DCPHASE', 'HT_PHASOR_INPHASE', 'HT_PHASOR_QUADRATURE',
        'HT_SINE', 'HT_LEADSINE', 'HT_TRENDMODE'
    ]
    
    # Statistical indicators (14 features)
    statistical_features = [
        'BETA', 'CORREL', 'LINEARREG', 'LINEARREG_ANGLE', 'LINEARREG_INTERCEPT',
        'LINEARREG_SLOPE', 'STDDEV', 'TSF', 'VAR', 'MAX_HIGH_20', 'MIN_LOW_20',
        'MAXINDEX_20', 'MININDEX_20', 'SUM_VOLUME_10'
    ]
    
    # Price transform indicators (19 features)
    price_transform_features = [
        'AVGPRICE', 'MEDPRICE', 'TYPPRICE', 'WCLPRICE', 'SQRT_CLOSE', 'LN_CLOSE',
        'LOG10_CLOSE', 'FLOOR_CLOSE', 'CEIL_CLOSE', 'ADD_HIGH_LOW', 'SUB_HIGH_LOW',
        'MULT_HIGH_LOW', 'DIV_HIGH_LOW', 'SIN_CLOSE', 'COS_CLOSE', 'TAN_CLOSE',
        'ASIN_NORM', 'ACOS_NORM', 'ATAN_CLOSE'
    ]
    
    # Pattern recognition indicators (47 features)
    pattern_features = [
        'CDL_DOJI', 'CDL_DRAGONFLY_DOJI', 'CDL_GRAVESTONE_DOJI', 'CDL_LONG_LEGGED_DOJI',
        'CDL_HAMMER', 'CDL_INVERTED_HAMMER', 'CDL_HANGING_MAN', 'CDL_SHOOTING_STAR',
        'CDL_MARUBOZU', 'CDL_SPINNING_TOP', 'CDL_RICKSHAW_MAN', 'CDL_ENGULFING',
        'CDL_HARAMI', 'CDL_HARAMI_CROSS', 'CDL_PIERCING', 'CDL_DARK_CLOUD',
        'CDL_HIKKAKE', 'CDL_IN_NECK', 'CDL_ON_NECK', 'CDL_THRUSTING',
        'CDL_3_BLACK_CROWS', 'CDL_3_WHITE_SOLDIERS', 'CDL_MORNING_STAR', 'CDL_EVENING_STAR',
        'CDL_3_INSIDE', 'CDL_3_OUTSIDE', 'CDL_3_STARS_IN_SOUTH', 'CDL_3_LINE_STRIKE',
        'CDL_ABANDONED_BABY', 'CDL_ADVANCE_BLOCK', 'CDL_BELT_HOLD', 'CDL_BREAKAWAY',
        'CDL_CLOSING_MARUBOZU', 'CDL_CONCEALING_BABY_SWALLOW', 'CDL_COUNTERATTACK',
        'CDL_DOJI_STAR', 'CDL_GAP_SIDE_BY_SIDE_WHITE', 'CDL_HOMING_PIGEON',
        'CDL_IDENTICAL_3_CROWS', 'CDL_KICKING', 'CDL_LADDER_BOTTOM', 'CDL_LONG_LINE',
        'CDL_MATCHING_LOW', 'CDL_RISING_FALLING_3_METHODS', 'CDL_SEPARATING_LINES',
        'CDL_SHORT_LINE', 'CDL_STALLED_PATTERN', 'CDL_STICK_SANDWICH', 'CDL_TAKURI',
        'CDL_TASUKI_GAP', 'CDL_TRISTAR', 'CDL_UNIQUE_3_RIVER', 'CDL_UPSIDE_GAP_2_CROWS',
        'CDL_XSIDE_GAP_3_METHODS'
    ]
    
    # Advanced features (12 features)
    advanced_features = [
        'Momentum_5', 'Momentum_10', 'Momentum_20',
        'Price_Position_10', 'Price_Position_20', 'Price_Position_50',
        'Trend_Strength', 'Support_20', 'Resistance_20',
        'Gap', 'Intraday_Range', 'Price_Acceleration'
    ]
    
    # Combine all features
    all_features = (
        basic_features + trend_features + momentum_features + volatility_features +
        volume_features + cycle_features + statistical_features + 
        price_transform_features + pattern_features + advanced_features
    )
    
    return all_features

def standardize_features_for_model(df: pd.DataFrame, required_features: List[str] = None) -> pd.DataFrame:
    """
    Standardize generated features to match exactly what the models expect
    This ensures consistent feature count and order for all stocks
    
    Args:
        df: DataFrame with generated features
        required_features: List of specific features expected by the model
        
    Returns:
        DataFrame with exactly the required features in the correct order
    """
    if required_features is None:
        # Get the exact 50 features that the models expect (from actual preprocessors.pkl)
        required_features = [
            'Open', 'High', 'Low', 'Close', 'SMA_5', 'SMA_10', 'SMA_20', 'EMA_5', 'EMA_10', 'EMA_12',
            'EMA_20', 'EMA_26', 'WMA_20', 'DEMA_20', 'TEMA_20', 'TRIMA_20', 'KAMA_20', 'MAMA', 'FAMA',
            'BB_High', 'BB_Mid', 'BB_High_50', 'SAR', 'MIDPOINT', 'MIDPRICE', 'HT_TRENDLINE', 'MOM_10',
            'MINUS_DI', 'PLUS_DM', 'LINEARREG', 'LINEARREG_INTERCEPT', 'TSF', 'MAX_HIGH_20', 'MIN_LOW_20',
            'AVGPRICE', 'MEDPRICE', 'TYPPRICE', 'WCLPRICE', 'SQRT_CLOSE', 'LN_CLOSE', 'LOG10_CLOSE',
            'FLOOR_CLOSE', 'CEIL_CLOSE', 'ADD_HIGH_LOW', 'SIN_CLOSE', 'COS_CLOSE', 'TAN_CLOSE',
            'ASIN_NORM', 'ACOS_NORM', 'ATAN_CLOSE'
        ]
    
    print(f"[INFO] Standardizing features to match model requirements ({len(required_features)} features)")
    
    # Create new DataFrame with exact features in correct order
    standardized_df = pd.DataFrame(index=df.index)
    
    missing_features = []
    present_features = []
    
    for feature in required_features:
        if feature in df.columns:
            standardized_df[feature] = df[feature]
            present_features.append(feature)
        else:
            # Add missing feature with intelligent default
            default_value = _get_intelligent_default(feature, df)
            standardized_df[feature] = default_value
            missing_features.append(feature)
    
    print(f"[INFO] Features standardized: {len(present_features)} present, {len(missing_features)} missing (filled)")
    
    if missing_features:
        print(f"[DEBUG] Missing features filled: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")
    
    # Clean final data
    standardized_df = clean_and_validate_features(standardized_df)
    
    # Ensure exactly the right number of features
    assert len(standardized_df.columns) == len(required_features), \
        f"Feature count mismatch: got {len(standardized_df.columns)}, expected {len(required_features)}"
    
    return standardized_df

def _get_intelligent_default(feature_name: str, existing_df: pd.DataFrame) -> pd.Series:
    """
    Generate intelligent default values for missing features based on existing data
    """
    # Create series with same index as input
    default_series = pd.Series(index=existing_df.index, dtype=float)
    
    feature_upper = feature_name.upper()
    
    # For price-based features, use existing price data if available
    if feature_name in ['Open', 'High', 'Low', 'Close']:
        if 'Close' in existing_df.columns:
            # Use Close price as default for all OHLC
            default_series = existing_df['Close'].copy()
        else:
            default_series = 100.0  # Reasonable default price
    
    # For moving averages, approximate using Close price
    elif any(ma in feature_upper for ma in ['SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 'TRIMA', 'KAMA']):
        if 'Close' in existing_df.columns:
            default_series = existing_df['Close'].copy()
        else:
            default_series = 100.0
    
    # For Bollinger Bands, use price-based defaults
    elif 'BB_' in feature_upper:
        if 'Close' in existing_df.columns:
            if 'POSITION' in feature_upper:
                default_series = 0.5  # Middle of BB
            else:
                default_series = existing_df['Close'].copy()  # Price level
        else:
            default_series = 0.5 if 'POSITION' in feature_upper else 100.0
    
    # For oscillators, use neutral values
    elif any(osc in feature_upper for osc in ['RSI', 'STOCH', 'WILLIAMS']):
        default_series = 50.0  # Neutral oscillator value
    
    # For MACD and momentum indicators
    elif any(mom in feature_upper for mom in ['MACD', 'ROC', 'MOM']):
        default_series = 0.0  # Neutral momentum
    
    # For ADX and directional indicators
    elif feature_upper in ['ADX', 'MINUS_DI', 'PLUS_DI']:
        if 'ADX' in feature_upper:
            default_series = 25.0  # Neutral trend strength
        else:
            default_series = 14.0  # Balanced directional movement
    
    # For volatility measures
    elif feature_upper in ['ATR']:
        if 'High' in existing_df.columns and 'Low' in existing_df.columns:
            # Approximate ATR as 2% of average price range
            avg_range = (existing_df['High'] - existing_df['Low']).mean()
            default_series = avg_range if not pd.isna(avg_range) else 2.0
        else:
            default_series = 2.0  # Default volatility
    
    # For volume indicators
    elif feature_upper in ['OBV']:
        default_series = 1000000  # Default volume level
    
    # For price transforms
    elif feature_upper in ['TYPPRICE', 'AVGPRICE', 'MIDPOINT', 'MIDPRICE']:
        if 'Close' in existing_df.columns:
            default_series = existing_df['Close'].copy()
        else:
            default_series = 100.0
    
    # For mathematical transforms
    elif any(math_func in feature_upper for math_func in ['LOG', 'FLOOR', 'CEIL', 'SIN', 'COS', 'TAN']):
        if 'LOG' in feature_upper:
            default_series = 4.6  # log10(100)
        elif any(trig in feature_upper for trig in ['SIN', 'COS', 'TAN', 'ASIN', 'ACOS', 'ATAN']):
            default_series = 0.0  # Neutral trigonometric value
        elif 'FLOOR' in feature_upper or 'CEIL' in feature_upper:
            default_series = 100.0  # Price-based
        else:
            default_series = 0.0
    
    # For arithmetic operations on High/Low
    elif 'ADD_HIGH_LOW' in feature_upper:
        if 'High' in existing_df.columns and 'Low' in existing_df.columns:
            default_series = existing_df['High'] + existing_df['Low']
        else:
            default_series = 200.0  # Approximate sum of high and low
    
    # For trend lines and other complex indicators
    elif any(trend in feature_upper for trend in ['HT_TRENDLINE', 'SAR', 'MAMA', 'FAMA']):
        if 'Close' in existing_df.columns:
            default_series = existing_df['Close'].copy()
        else:
            default_series = 100.0
    
    # Default fallback
    else:
        default_series = 0.0
    
    # Ensure no NaN values in the default series
    if isinstance(default_series, pd.Series):
        default_series = default_series.fillna(0.0)
    else:
        # If it's a scalar, broadcast to series
        default_series = pd.Series(default_series, index=existing_df.index)
    
    return default_series
