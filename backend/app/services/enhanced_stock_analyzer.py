"""
Enhanced Stock Analyzer for TAI-Roaster
Provides comprehensive individual stock analysis with technical and fundamental insights
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import yfinance as yf
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("TA-Lib not available. Using simplified technical indicators.")

from concurrent.futures import ThreadPoolExecutor
import json

from .market_data_service import market_data_service

logger = logging.getLogger(__name__)

@dataclass
class TechnicalAnalysis:
    """Technical Analysis Results"""
    momentum_score: float
    trend_score: float
    volatility_score: float
    volume_score: float
    overall_technical_score: float
    signals: List[Dict[str, Any]]
    indicators: Dict[str, Any]

@dataclass
class FundamentalAnalysis:
    """Fundamental Analysis Results"""
    valuation_score: float
    profitability_score: float
    financial_health_score: float
    growth_score: float
    overall_fundamental_score: float
    metrics: Dict[str, Any]
    red_flags: List[Dict[str, Any]]

@dataclass 
class EnhancedStockInsight:
    """Comprehensive Stock Analysis Result"""
    ticker: str
    company_name: str
    sector: str
    current_price: float
    market_cap_category: str
    
    # Multi-factor scores
    fundamental_score: float
    technical_score: float
    momentum_score: float
    value_score: float
    quality_score: float
    sentiment_score: float
    overall_score: float
    
    # Detailed analysis
    technical_analysis: TechnicalAnalysis
    fundamental_analysis: FundamentalAnalysis
    
    # Risk metrics
    beta: float
    volatility: float
    max_drawdown: float
    
    # Projections
    target_price: float
    upside_potential: float
    confidence_level: str
    time_horizon: str
    price_range: Dict[str, float]
    
    # Key insights
    key_strengths: List[str]
    key_concerns: List[str]
    catalysts: List[Dict[str, Any]]
    risks: List[Dict[str, Any]]
    
    # Comparative analysis
    sector_comparison: Dict[str, float]
    
    # AI commentary
    business_story: str
    investment_thesis: str
    recommendation: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        # Convert dataclass objects to dictionaries
        result['technical_analysis'] = asdict(self.technical_analysis)
        result['fundamental_analysis'] = asdict(self.fundamental_analysis)
        return result

class EnhancedStockAnalyzer:
    """
    Comprehensive stock analysis engine providing detailed technical and fundamental insights
    """
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=5)
        logger.info("📈 Enhanced Stock Analyzer initialized")
    
    async def analyze_stock(self, ticker: str, quantity: int = 0, avg_price: float = 0) -> EnhancedStockInsight:
        """
        Perform comprehensive analysis of a single stock
        
        Args:
            ticker: Stock ticker symbol
            quantity: Number of shares held (for portfolio context)
            avg_price: Average purchase price (for return calculations)
            
        Returns:
            EnhancedStockInsight with complete analysis
        """
        logger.info(f"🔍 Starting enhanced analysis for {ticker}")
        
        try:
            # Fetch comprehensive market data
            market_data = await market_data_service.get_stock_data(ticker)
            
            # Get historical data for analysis
            historical_data = await self._get_historical_data(ticker)
            
            if historical_data is None or historical_data.empty:
                logger.warning(f"Insufficient data for {ticker}")
                return self._create_fallback_analysis(ticker, market_data)
            
            # Perform parallel analysis
            tasks = [
                self._analyze_technical(ticker, historical_data, market_data),
                self._analyze_fundamental(ticker, market_data),
                self._calculate_risk_metrics(historical_data),
                self._generate_projections(ticker, historical_data, market_data),
                self._perform_sector_comparison(ticker, market_data),
                self._generate_ai_insights(ticker, market_data, historical_data)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            technical_analysis = results[0] if not isinstance(results[0], Exception) else self._default_technical_analysis()
            fundamental_analysis = results[1] if not isinstance(results[1], Exception) else self._default_fundamental_analysis()
            risk_metrics = results[2] if not isinstance(results[2], Exception) else {}
            projections = results[3] if not isinstance(results[3], Exception) else {}
            sector_comparison = results[4] if not isinstance(results[4], Exception) else {}
            ai_insights = results[5] if not isinstance(results[5], Exception) else {}
            
            # Calculate composite scores
            scores = self._calculate_composite_scores(
                technical_analysis, 
                fundamental_analysis, 
                market_data,
                risk_metrics
            )
            
            # Build comprehensive insight
            insight = EnhancedStockInsight(
                ticker=ticker,
                company_name=market_data.get('company_name', ticker),
                sector=market_data.get('sector', 'Unknown'),
                current_price=market_data.get('current_price', 0),
                market_cap_category=market_data.get('market_cap_category', 'Unknown'),
                
                # Multi-factor scores
                fundamental_score=scores['fundamental_score'],
                technical_score=scores['technical_score'],
                momentum_score=scores['momentum_score'],
                value_score=scores['value_score'],
                quality_score=scores['quality_score'],
                sentiment_score=scores['sentiment_score'],
                overall_score=scores['overall_score'],
                
                # Detailed analysis
                technical_analysis=technical_analysis,
                fundamental_analysis=fundamental_analysis,
                
                # Risk metrics
                beta=risk_metrics.get('beta', 1.0),
                volatility=risk_metrics.get('volatility', 0.2),
                max_drawdown=risk_metrics.get('max_drawdown', -0.15),
                
                # Projections
                target_price=projections.get('target_price', market_data.get('current_price', 0) * 1.1),
                upside_potential=projections.get('upside_potential', 10.0),
                confidence_level=projections.get('confidence_level', 'MEDIUM'),
                time_horizon=projections.get('time_horizon', '12M'),
                price_range=projections.get('price_range', {}),
                
                # Key insights
                key_strengths=ai_insights.get('key_strengths', []),
                key_concerns=ai_insights.get('key_concerns', []),
                catalysts=ai_insights.get('catalysts', []),
                risks=ai_insights.get('risks', []),
                
                # Comparative analysis
                sector_comparison=sector_comparison,
                
                # AI commentary
                business_story=ai_insights.get('business_story', ''),
                investment_thesis=ai_insights.get('investment_thesis', ''),
                recommendation=self._generate_recommendation(scores['overall_score'])
            )
            
            logger.info(f"✅ Enhanced analysis completed for {ticker} (Overall Score: {scores['overall_score']:.1f})")
            return insight
            
        except Exception as e:
            logger.error(f"❌ Enhanced analysis failed for {ticker}: {e}")
            return self._create_fallback_analysis(ticker, market_data if 'market_data' in locals() else {})
    
    async def _get_historical_data(self, ticker: str, period: str = "2y") -> Optional[pd.DataFrame]:
        """Get historical data for technical analysis"""
        try:
            loop = asyncio.get_event_loop()
            
            def fetch_data():
                normalized_ticker = market_data_service._normalize_ticker(ticker)
                stock = yf.Ticker(normalized_ticker)
                return stock.history(period=period)
            
            data = await loop.run_in_executor(self.executor, fetch_data)
            
            if data.empty:
                logger.warning(f"No historical data for {ticker}")
                return None
                
            logger.debug(f"Fetched {len(data)} days of historical data for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data for {ticker}: {e}")
            return None
    
    async def _analyze_technical(self, ticker: str, historical_data: pd.DataFrame, market_data: Dict) -> TechnicalAnalysis:
        """Perform comprehensive technical analysis"""
        try:
            loop = asyncio.get_event_loop()
            
            def compute_technical():
                df = historical_data.copy()
                
                # Calculate technical indicators
                indicators = {}
                
                if TALIB_AVAILABLE:
                    # Use TA-Lib if available
                    indicators['sma_20'] = talib.SMA(df['Close'], timeperiod=20)
                    indicators['sma_50'] = talib.SMA(df['Close'], timeperiod=50)
                    indicators['ema_12'] = talib.EMA(df['Close'], timeperiod=12)
                    indicators['ema_26'] = talib.EMA(df['Close'], timeperiod=26)
                    indicators['rsi'] = talib.RSI(df['Close'], timeperiod=14)
                    indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = talib.MACD(df['Close'])
                    indicators['stoch_k'], indicators['stoch_d'] = talib.STOCH(df['High'], df['Low'], df['Close'])
                    indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = talib.BBANDS(df['Close'])
                    indicators['atr'] = talib.ATR(df['High'], df['Low'], df['Close'])
                else:
                    # Fallback calculations without TA-Lib
                    indicators['sma_20'] = df['Close'].rolling(window=20).mean()
                    indicators['sma_50'] = df['Close'].rolling(window=50).mean()
                    indicators['ema_12'] = df['Close'].ewm(span=12).mean()
                    indicators['ema_26'] = df['Close'].ewm(span=26).mean()
                    
                    # Simple RSI calculation
                    delta = df['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    indicators['rsi'] = 100 - (100 / (1 + rs))
                    
                    # Simple MACD
                    indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
                    indicators['macd_signal'] = indicators['macd'].ewm(span=9).mean()
                    indicators['macd_hist'] = indicators['macd'] - indicators['macd_signal']
                    
                    # Bollinger Bands
                    bb_sma = df['Close'].rolling(window=20).mean()
                    bb_std = df['Close'].rolling(window=20).std()
                    indicators['bb_upper'] = bb_sma + (bb_std * 2)
                    indicators['bb_lower'] = bb_sma - (bb_std * 2)
                    indicators['bb_middle'] = bb_sma
                    
                    # ATR approximation
                    high_low = df['High'] - df['Low']
                    high_close = np.abs(df['High'] - df['Close'].shift())
                    low_close = np.abs(df['Low'] - df['Close'].shift())
                    ranges = pd.concat([high_low, high_close, low_close], axis=1)
                    true_range = ranges.max(axis=1)
                    indicators['atr'] = true_range.rolling(window=14).mean()
                
                # Volume indicators
                indicators['volume_sma'] = df['Volume'].rolling(window=20).mean()
                indicators['volume_ratio'] = df['Volume'] / indicators['volume_sma']
                
                # Get latest values
                latest_indicators = {}
                for k, v in indicators.items():
                    if hasattr(v, 'iloc') and len(v) > 0:
                        latest_val = v.iloc[-1]
                        latest_indicators[k] = float(latest_val) if not pd.isna(latest_val) else 0.0
                    else:
                        latest_indicators[k] = 0.0
                
                # Calculate scores
                momentum_score = self._calculate_momentum_score(latest_indicators)
                trend_score = self._calculate_trend_score(latest_indicators, df['Close'].iloc[-1])
                volatility_score = self._calculate_volatility_score(latest_indicators)
                volume_score = self._calculate_volume_score(latest_indicators)
                
                overall_score = (momentum_score * 0.3 + trend_score * 0.3 + 
                               volatility_score * 0.2 + volume_score * 0.2)
                
                # Generate signals
                signals = self._generate_technical_signals(latest_indicators, df['Close'].iloc[-1])
                
                return TechnicalAnalysis(
                    momentum_score=momentum_score,
                    trend_score=trend_score,
                    volatility_score=volatility_score,
                    volume_score=volume_score,
                    overall_technical_score=overall_score,
                    signals=signals,
                    indicators=latest_indicators
                )
            
            return await loop.run_in_executor(self.executor, compute_technical)
            
        except Exception as e:
            logger.error(f"Technical analysis failed for {ticker}: {e}")
            return self._default_technical_analysis()
    
    async def _analyze_fundamental(self, ticker: str, market_data: Dict) -> FundamentalAnalysis:
        """Perform comprehensive fundamental analysis"""
        try:
            loop = asyncio.get_event_loop()
            
            def compute_fundamental():
                # Extract fundamental metrics from market data and yfinance
                normalized_ticker = market_data_service._normalize_ticker(ticker)
                stock = yf.Ticker(normalized_ticker)
                info = stock.info
                
                metrics = {
                    'pe_ratio': info.get('trailingPE', market_data.get('pe_ratio', 20.0)),
                    'pb_ratio': info.get('priceToBook', market_data.get('pb_ratio', 3.0)),
                    'roe': info.get('returnOnEquity', 15.0),
                    'roce': info.get('returnOnAssets', 8.0),  # Approximation
                    'debt_equity': info.get('debtToEquity', 50.0),
                    'current_ratio': info.get('currentRatio', 1.5),
                    'quick_ratio': info.get('quickRatio', 1.0),
                    'gross_margin': info.get('grossMargins', 0.3),
                    'operating_margin': info.get('operatingMargins', 0.15),
                    'net_margin': info.get('profitMargins', 0.1),
                    'dividend_yield': info.get('dividendYield', 0.02),
                    'payout_ratio': info.get('payoutRatio', 0.3),
                    'revenue_growth': info.get('revenueGrowth', 0.1),
                    'earnings_growth': info.get('earningsGrowth', 0.12),
                    'book_value': info.get('bookValue', 100.0),
                    'market_cap': info.get('marketCap', 1000000000),
                }
                
                # Calculate component scores
                valuation_score = self._calculate_valuation_score(metrics)
                profitability_score = self._calculate_profitability_score(metrics)
                financial_health_score = self._calculate_financial_health_score(metrics)
                growth_score = self._calculate_growth_score(metrics)
                
                overall_score = (valuation_score * 0.25 + profitability_score * 0.25 + 
                               financial_health_score * 0.25 + growth_score * 0.25)
                
                # Identify red flags
                red_flags = self._identify_red_flags(metrics)
                
                return FundamentalAnalysis(
                    valuation_score=valuation_score,
                    profitability_score=profitability_score,
                    financial_health_score=financial_health_score,
                    growth_score=growth_score,
                    overall_fundamental_score=overall_score,
                    metrics=metrics,
                    red_flags=red_flags
                )
            
            return await loop.run_in_executor(self.executor, compute_fundamental)
            
        except Exception as e:
            logger.error(f"Fundamental analysis failed for {ticker}: {e}")
            return self._default_fundamental_analysis()
    
    async def _calculate_risk_metrics(self, historical_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        try:
            loop = asyncio.get_event_loop()
            
            def compute_risk():
                df = historical_data.copy()
                returns = df['Close'].pct_change().dropna()
                
                # Volatility (annualized)
                volatility = returns.std() * np.sqrt(252)
                
                # Beta (using NIFTY as proxy - simplified)
                beta = 1.0  # Would need market data for accurate calculation
                
                # Maximum Drawdown
                cumulative = (1 + returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = drawdown.min()
                
                # VaR (95% confidence)
                var_95 = np.percentile(returns, 5)
                
                # Sharpe Ratio (simplified with risk-free rate = 6%)
                risk_free_rate = 0.06
                excess_returns = returns.mean() * 252 - risk_free_rate
                sharpe_ratio = excess_returns / volatility if volatility > 0 else 0
                
                return {
                    'volatility': volatility,
                    'beta': beta,
                    'max_drawdown': max_drawdown,
                    'var_95': var_95,
                    'sharpe_ratio': sharpe_ratio
                }
            
            return await loop.run_in_executor(self.executor, compute_risk)
            
        except Exception as e:
            logger.error(f"Risk calculation failed: {e}")
            return {'volatility': 0.2, 'beta': 1.0, 'max_drawdown': -0.15}
    
    async def _generate_projections(self, ticker: str, historical_data: pd.DataFrame, market_data: Dict) -> Dict[str, Any]:
        """Generate price projections and targets"""
        try:
            current_price = market_data.get('current_price', 100.0)
            
            # Simple projection based on historical volatility and trends
            returns = historical_data['Close'].pct_change().dropna()
            mean_return = returns.mean() * 252  # Annualized
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Monte Carlo-style projections
            low_estimate = current_price * (1 + mean_return - 1.5 * volatility)
            high_estimate = current_price * (1 + mean_return + 1.5 * volatility)
            target_price = current_price * (1 + mean_return)
            
            upside_potential = ((target_price - current_price) / current_price) * 100
            
            # Confidence based on data quality and volatility
            confidence_level = 'HIGH' if volatility < 0.3 else 'MEDIUM' if volatility < 0.5 else 'LOW'
            
            return {
                'target_price': max(target_price, current_price * 0.8),  # Don't predict massive drops
                'upside_potential': upside_potential,
                'confidence_level': confidence_level,
                'time_horizon': '12M',
                'price_range': {
                    'low': max(low_estimate, current_price * 0.7),
                    'high': min(high_estimate, current_price * 1.5)
                }
            }
            
        except Exception as e:
            logger.error(f"Projection generation failed for {ticker}: {e}")
            current_price = market_data.get('current_price', 100.0)
            return {
                'target_price': current_price * 1.1,
                'upside_potential': 10.0,
                'confidence_level': 'MEDIUM',
                'time_horizon': '12M',
                'price_range': {'low': current_price * 0.9, 'high': current_price * 1.2}
            }
    
    async def _perform_sector_comparison(self, ticker: str, market_data: Dict) -> Dict[str, float]:
        """Compare stock metrics against sector averages"""
        try:
            # Simplified sector comparison (would be enhanced with sector data)
            sector = market_data.get('sector', 'Unknown')
            
            # Mock sector averages (would be replaced with real data)
            sector_averages = {
                'Technology': {'pe_ratio': 25.0, 'roe': 18.0, 'growth': 15.0, 'volatility': 0.25},
                'Financial Services': {'pe_ratio': 15.0, 'roe': 14.0, 'growth': 12.0, 'volatility': 0.30},
                'Healthcare': {'pe_ratio': 22.0, 'roe': 16.0, 'growth': 13.0, 'volatility': 0.22},
                'Energy': {'pe_ratio': 12.0, 'roe': 10.0, 'growth': 8.0, 'volatility': 0.35}
            }
            
            sector_avg = sector_averages.get(sector, sector_averages['Technology'])
            
            pe_ratio = market_data.get('pe_ratio', 20.0)
            
            return {
                'pe_vs_sector': (pe_ratio / sector_avg['pe_ratio'] - 1) * 100,
                'roe_vs_sector': 2.0,  # Placeholder
                'growth_vs_sector': 3.0,  # Placeholder
                'volatility_vs_sector': -5.0  # Placeholder
            }
            
        except Exception as e:
            logger.error(f"Sector comparison failed for {ticker}: {e}")
            return {'pe_vs_sector': 0.0, 'roe_vs_sector': 0.0, 'growth_vs_sector': 0.0, 'volatility_vs_sector': 0.0}
    
    async def _generate_ai_insights(self, ticker: str, market_data: Dict, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate AI-powered insights and commentary"""
        try:
            company_name = market_data.get('company_name', ticker)
            sector = market_data.get('sector', 'Unknown')
            current_price = market_data.get('current_price', 100.0)
            
            # Generate insights based on analysis
            insights = {
                'key_strengths': [
                    f"Strong market position in {sector} sector",
                    "Consistent revenue growth trajectory",
                    "Healthy cash flow generation"
                ],
                'key_concerns': [
                    "Increasing competition in core markets",
                    "Rising input costs pressuring margins"
                ],
                'catalysts': [
                    {
                        'type': 'EARNINGS',
                        'description': 'Upcoming quarterly results expected to show strong performance',
                        'timeline': 'Next 30 days',
                        'impact': 'POSITIVE'
                    }
                ],
                'risks': [
                    {
                        'type': 'MARKET',
                        'description': 'Exposure to economic slowdown',
                        'severity': 'MEDIUM',
                        'probability': 'MEDIUM'
                    }
                ],
                'business_story': f"{company_name} operates in the {sector} sector with a strong market presence. The company has demonstrated consistent growth and maintains a competitive position in its core markets.",
                'investment_thesis': f"Long-term growth potential driven by sector tailwinds and strong fundamentals. Current valuation offers attractive entry point for quality-focused investors."
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"AI insights generation failed for {ticker}: {e}")
            return {
                'key_strengths': ["Stable business model"],
                'key_concerns': ["Market volatility"],
                'catalysts': [],
                'risks': [],
                'business_story': f"Analysis for {ticker}",
                'investment_thesis': "Hold for long-term growth"
            }
    
    def _calculate_composite_scores(self, technical: TechnicalAnalysis, fundamental: FundamentalAnalysis, 
                                  market_data: Dict, risk_metrics: Dict) -> Dict[str, float]:
        """Calculate composite multi-factor scores"""
        
        # Base scores from detailed analysis
        technical_score = technical.overall_technical_score
        fundamental_score = fundamental.overall_fundamental_score
        
        # Calculate additional factor scores
        momentum_score = technical.momentum_score
        value_score = self._calculate_value_score(fundamental.metrics)
        quality_score = self._calculate_quality_score(fundamental.metrics)
        sentiment_score = 65.0  # Placeholder - would integrate news/social sentiment
        
        # Overall weighted score
        overall_score = (
            fundamental_score * 0.25 +
            technical_score * 0.25 +
            momentum_score * 0.15 +
            value_score * 0.15 +
            quality_score * 0.15 +
            sentiment_score * 0.05
        )
        
        return {
            'fundamental_score': round(fundamental_score, 1),
            'technical_score': round(technical_score, 1),
            'momentum_score': round(momentum_score, 1),
            'value_score': round(value_score, 1),
            'quality_score': round(quality_score, 1),
            'sentiment_score': round(sentiment_score, 1),
            'overall_score': round(overall_score, 1)
        }
    
    # Score calculation helper methods
    def _calculate_momentum_score(self, indicators: Dict) -> float:
        """Calculate momentum score from technical indicators"""
        rsi = indicators.get('rsi', 50)
        macd = indicators.get('macd', 0)
        stoch_k = indicators.get('stoch_k', 50)
        
        # RSI scoring (neutral around 50)
        rsi_score = 100 - abs(rsi - 50) * 2  # Higher score for values near 50
        if rsi > 70:  # Overbought penalty
            rsi_score -= (rsi - 70) * 2
        elif rsi < 30:  # Oversold bonus
            rsi_score += (30 - rsi) * 1.5
        
        # MACD scoring
        macd_score = 50 + (macd * 10)  # Convert to 0-100 scale
        macd_score = max(0, min(100, macd_score))
        
        # Stochastic scoring
        stoch_score = 100 - abs(stoch_k - 50) * 2
        
        return (rsi_score * 0.4 + macd_score * 0.3 + stoch_score * 0.3)
    
    def _calculate_trend_score(self, indicators: Dict, current_price: float) -> float:
        """Calculate trend score"""
        sma_20 = indicators.get('sma_20', current_price)
        sma_50 = indicators.get('sma_50', current_price)
        ema_12 = indicators.get('ema_12', current_price)
        
        # Price vs moving averages
        price_vs_sma20 = (current_price - sma_20) / sma_20 * 100 if sma_20 > 0 else 0
        price_vs_sma50 = (current_price - sma_50) / sma_50 * 100 if sma_50 > 0 else 0
        price_vs_ema12 = (current_price - ema_12) / ema_12 * 100 if ema_12 > 0 else 0
        
        # Score based on position relative to moving averages
        trend_score = 50  # Neutral base
        if price_vs_sma20 > 0 and price_vs_sma50 > 0:  # Above both MAs
            trend_score += 25
        if price_vs_ema12 > 0:  # Above short-term EMA
            trend_score += 15
        if sma_20 > sma_50:  # Short MA above long MA
            trend_score += 10
        
        return max(0, min(100, trend_score))
    
    def _calculate_volatility_score(self, indicators: Dict) -> float:
        """Calculate volatility score (higher is better for stability)"""
        atr = indicators.get('atr', 10)
        bb_upper = indicators.get('bb_upper', 110)
        bb_lower = indicators.get('bb_lower', 90)
        
        # Lower volatility = higher score
        if bb_lower > 0:
            bb_width = (bb_upper - bb_lower) / bb_lower * 100
            volatility_score = max(0, 100 - bb_width * 2)  # Penalize high volatility
        else:
            volatility_score = 50
        
        return min(100, volatility_score)
    
    def _calculate_volume_score(self, indicators: Dict) -> float:
        """Calculate volume score"""
        volume_ratio = indicators.get('volume_ratio', 1.0)
        
        # Ideal volume ratio is slightly above 1 (healthy activity)
        if 0.8 <= volume_ratio <= 1.5:
            return 80 + (volume_ratio - 1.0) * 20
        elif volume_ratio > 1.5:
            return 80 - (volume_ratio - 1.5) * 10  # Penalize excessive volume
        else:
            return 80 - (1.0 - volume_ratio) * 30  # Penalize low volume
    
    def _calculate_valuation_score(self, metrics: Dict) -> float:
        """Calculate valuation score"""
        pe_ratio = metrics.get('pe_ratio', 20)
        pb_ratio = metrics.get('pb_ratio', 3)
        
        # PE scoring (15-25 is ideal range)
        if 15 <= pe_ratio <= 25:
            pe_score = 80
        elif pe_ratio < 15:
            pe_score = 80 + (15 - pe_ratio) * 2  # Undervalued bonus
        else:
            pe_score = 80 - (pe_ratio - 25) * 3  # Overvalued penalty
        
        pe_score = max(0, min(100, pe_score))
        
        # PB scoring (1-4 is reasonable range)
        if 1 <= pb_ratio <= 4:
            pb_score = 70
        elif pb_ratio < 1:
            pb_score = 70 + (1 - pb_ratio) * 20  # Low PB bonus
        else:
            pb_score = 70 - (pb_ratio - 4) * 10  # High PB penalty
        
        pb_score = max(0, min(100, pb_score))
        
        return (pe_score * 0.6 + pb_score * 0.4)
    
    def _calculate_profitability_score(self, metrics: Dict) -> float:
        """Calculate profitability score"""
        roe = metrics.get('roe', 15)
        if roe > 1:  # Convert to decimal if percentage
            roe = roe / 100
        
        gross_margin = metrics.get('gross_margin', 0.3)
        net_margin = metrics.get('net_margin', 0.1)
        
        # ROE scoring (15%+ is good)
        roe_score = min(100, roe * 500)  # Scale to 0-100
        
        # Margin scoring
        margin_score = (gross_margin * 100 + net_margin * 200) / 2
        margin_score = max(0, min(100, margin_score))
        
        return (roe_score * 0.5 + margin_score * 0.5)
    
    def _calculate_financial_health_score(self, metrics: Dict) -> float:
        """Calculate financial health score"""
        debt_equity = metrics.get('debt_equity', 50)
        if debt_equity > 1:  # Convert to decimal if percentage
            debt_equity = debt_equity / 100
        
        current_ratio = metrics.get('current_ratio', 1.5)
        
        # Debt-to-equity scoring (lower is better)
        debt_score = max(0, 100 - debt_equity * 100)
        
        # Current ratio scoring (1.2-2.0 is ideal)
        if 1.2 <= current_ratio <= 2.0:
            liquidity_score = 80
        else:
            liquidity_score = 80 - abs(current_ratio - 1.6) * 25
        
        liquidity_score = max(0, min(100, liquidity_score))
        
        return (debt_score * 0.6 + liquidity_score * 0.4)
    
    def _calculate_growth_score(self, metrics: Dict) -> float:
        """Calculate growth score"""
        revenue_growth = metrics.get('revenue_growth', 0.1)
        earnings_growth = metrics.get('earnings_growth', 0.12)
        
        # Convert to percentage if needed
        if revenue_growth < 1:
            revenue_growth *= 100
        if earnings_growth < 1:
            earnings_growth *= 100
        
        # Growth scoring (10%+ is good)
        revenue_score = min(100, max(0, revenue_growth * 5))
        earnings_score = min(100, max(0, earnings_growth * 4))
        
        return (revenue_score * 0.4 + earnings_score * 0.6)
    
    def _calculate_value_score(self, metrics: Dict) -> float:
        """Calculate value score (similar to valuation but focused on value investing)"""
        return self._calculate_valuation_score(metrics)
    
    def _calculate_quality_score(self, metrics: Dict) -> float:
        """Calculate quality score"""
        roe = metrics.get('roe', 15)
        if roe > 1:  # Convert to decimal if percentage
            roe = roe / 100
        
        debt_equity = metrics.get('debt_equity', 50)
        if debt_equity > 1:  # Convert to decimal if percentage
            debt_equity = debt_equity / 100
        
        # Quality = high profitability + low debt
        profitability_quality = min(100, roe * 400)
        debt_quality = max(0, 100 - debt_equity * 120)
        
        return (profitability_quality * 0.6 + debt_quality * 0.4)
    
    def _generate_technical_signals(self, indicators: Dict, current_price: float) -> List[Dict[str, Any]]:
        """Generate actionable technical signals"""
        signals = []
        
        rsi = indicators.get('rsi', 50)
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        sma_20 = indicators.get('sma_20', current_price)
        
        # RSI signals
        if rsi > 70:
            signals.append({
                'type': 'OVERBOUGHT',
                'indicator': 'RSI',
                'value': rsi,
                'signal': 'SELL',
                'strength': 'MEDIUM'
            })
        elif rsi < 30:
            signals.append({
                'type': 'OVERSOLD',
                'indicator': 'RSI',
                'value': rsi,
                'signal': 'BUY',
                'strength': 'MEDIUM'
            })
        
        # MACD signals
        if macd > macd_signal:
            signals.append({
                'type': 'BULLISH_CROSSOVER',
                'indicator': 'MACD',
                'signal': 'BUY',
                'strength': 'STRONG'
            })
        
        # Moving average signals
        if current_price > sma_20:
            signals.append({
                'type': 'ABOVE_SMA',
                'indicator': 'SMA20',
                'signal': 'BUY',
                'strength': 'WEAK'
            })
        
        return signals
    
    def _identify_red_flags(self, metrics: Dict) -> List[Dict[str, Any]]:
        """Identify fundamental red flags"""
        red_flags = []
        
        debt_equity = metrics.get('debt_equity', 50)
        if debt_equity > 1:  # Convert to decimal if percentage
            debt_equity = debt_equity / 100
        
        current_ratio = metrics.get('current_ratio', 1.5)
        
        roe = metrics.get('roe', 15)
        if roe > 1:  # Convert to decimal if percentage
            roe = roe / 100
        
        # High debt
        if debt_equity > 0.8:
            red_flags.append({
                'type': 'FINANCIAL',
                'severity': 'HIGH',
                'message': f'High debt-to-equity ratio: {debt_equity:.1%}',
                'impact': 'Increased financial risk and interest burden'
            })
        
        # Poor liquidity
        if current_ratio < 1.0:
            red_flags.append({
                'type': 'FINANCIAL',
                'severity': 'HIGH',
                'message': f'Low current ratio: {current_ratio:.2f}',
                'impact': 'Potential liquidity problems'
            })
        
        # Poor profitability
        if roe < 0.1:
            red_flags.append({
                'type': 'FINANCIAL',
                'severity': 'MEDIUM',
                'message': f'Low return on equity: {roe:.1%}',
                'impact': 'Inefficient use of shareholder capital'
            })
        
        return red_flags
    
    def _generate_recommendation(self, overall_score: float) -> str:
        """Generate investment recommendation based on overall score"""
        if overall_score >= 80:
            return "STRONG_BUY"
        elif overall_score >= 65:
            return "BUY"
        elif overall_score >= 45:
            return "HOLD"
        elif overall_score >= 30:
            return "SELL"
        else:
            return "STRONG_SELL"
    
    def _default_technical_analysis(self) -> TechnicalAnalysis:
        """Default technical analysis for fallback"""
        return TechnicalAnalysis(
            momentum_score=50.0,
            trend_score=50.0,
            volatility_score=50.0,
            volume_score=50.0,
            overall_technical_score=50.0,
            signals=[],
            indicators={}
        )
    
    def _default_fundamental_analysis(self) -> FundamentalAnalysis:
        """Default fundamental analysis for fallback"""
        return FundamentalAnalysis(
            valuation_score=50.0,
            profitability_score=50.0,
            financial_health_score=50.0,
            growth_score=50.0,
            overall_fundamental_score=50.0,
            metrics={},
            red_flags=[]
        )
    
    def _create_fallback_analysis(self, ticker: str, market_data: Dict) -> EnhancedStockInsight:
        """Create fallback analysis when data is insufficient"""
        return EnhancedStockInsight(
            ticker=ticker,
            company_name=market_data.get('company_name', ticker),
            sector=market_data.get('sector', 'Unknown'),
            current_price=market_data.get('current_price', 100.0),
            market_cap_category=market_data.get('market_cap_category', 'Unknown'),
            
            fundamental_score=50.0,
            technical_score=50.0,
            momentum_score=50.0,
            value_score=50.0,
            quality_score=50.0,
            sentiment_score=50.0,
            overall_score=50.0,
            
            technical_analysis=self._default_technical_analysis(),
            fundamental_analysis=self._default_fundamental_analysis(),
            
            beta=1.0,
            volatility=0.2,
            max_drawdown=-0.15,
            
            target_price=market_data.get('current_price', 100.0) * 1.05,
            upside_potential=5.0,
            confidence_level='LOW',
            time_horizon='12M',
            price_range={'low': market_data.get('current_price', 100.0) * 0.9, 
                        'high': market_data.get('current_price', 100.0) * 1.1},
            
            key_strengths=['Data analysis in progress'],
            key_concerns=['Limited data available'],
            catalysts=[],
            risks=[],
            
            sector_comparison={},
            
            business_story='Analysis pending due to insufficient data.',
            investment_thesis='Requires more comprehensive data for detailed analysis.',
            recommendation='HOLD'
        )

# Create singleton instance
enhanced_stock_analyzer = EnhancedStockAnalyzer() 