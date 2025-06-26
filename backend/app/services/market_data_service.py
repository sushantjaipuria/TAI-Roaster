"""
Enhanced Market Data Service for TAI-Roaster
Provides real-time market data integration using yfinance for Indian stocks
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import yfinance as yf
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)

class RealTimeMarketDataService:
    """Service for real-time market data using yfinance"""
    
    def __init__(self):
        """Initialize market data service"""
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes cache
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.initialized = True
        logger.info("📊 Real-time Market Data Service initialized with yfinance")
    
    def _normalize_ticker(self, ticker: str) -> str:
        """
        Normalize ticker symbol for yfinance
        Convert Indian stock symbols to Yahoo Finance format
        """
        ticker = ticker.upper().strip()
        
        # If already has exchange suffix, return as is
        if any(suffix in ticker for suffix in ['.NS', '.BO', '.BSE']):
            return ticker
        
        # Map some common index tickers
        index_mapping = {
            '^NSEI': '^NSEI',
            '^BSESN': '^BSESN', 
            '^NSEBANK': '^NSEBANK',
            'NIFTY': '^NSEI',
            'SENSEX': '^BSESN',
            'BANKNIFTY': '^NSEBANK'
        }
        
        if ticker in index_mapping:
            return index_mapping[ticker]
        
        # For regular stocks, default to NSE (.NS)
        return f"{ticker}.NS"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache:
            return False
        
        cached_time = self.cache[cache_key].get('cache_timestamp')
        if not cached_time:
            return False
        
        return (datetime.now() - cached_time).total_seconds() < self.cache_ttl
    
    async def get_stock_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get comprehensive real-time stock data for a single ticker
        
        Args:
            ticker: Stock ticker symbol (e.g., 'RELIANCE', 'TCS', 'INFY')
            
        Returns:
            Dictionary containing real-time stock data
        """
        normalized_ticker = self._normalize_ticker(ticker)
        cache_key = f"{normalized_ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            logger.debug(f"Cache hit for {ticker}")
            return self.cache[cache_key]
        
        try:
            # Run yfinance call in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            stock_data = await loop.run_in_executor(
                self.executor, 
                self._fetch_stock_data_sync, 
                normalized_ticker, 
                ticker
            )
            
            # Cache the result with timestamp
            stock_data['cache_timestamp'] = datetime.now()
            self.cache[cache_key] = stock_data
            
            logger.info(f"✅ Fetched real-time data for {ticker}: ₹{stock_data.get('current_price', 'N/A')}")
            return stock_data
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch data for {ticker}: {e}")
            return self._get_fallback_data(ticker)
    
    def _fetch_stock_data_sync(self, normalized_ticker: str, original_ticker: str) -> Dict[str, Any]:
        """
        Synchronous function to fetch stock data using yfinance
        This runs in a thread pool to avoid blocking the async event loop
        """
        try:
            # Create yfinance ticker object
            stock = yf.Ticker(normalized_ticker)
            
            # Get basic info and current price
            info = stock.info
            hist = stock.history(period="5d")
            
            if hist.empty:
                raise ValueError(f"No historical data available for {normalized_ticker}")
            
            # Get latest price data
            latest_data = hist.iloc[-1]
            previous_close = hist.iloc[-2]['Close'] if len(hist) > 1 else latest_data['Close']
            
            current_price = float(latest_data['Close'])
            change = current_price - previous_close
            change_percent = (change / previous_close) * 100 if previous_close != 0 else 0
            
            # Extract company information
            company_name = info.get('longName', info.get('shortName', original_ticker))
            sector = info.get('sector', 'Unknown')
            market_cap = info.get('marketCap', 0)
            
            # Determine market cap category
            market_cap_category = self._get_market_cap_category(market_cap)
            
            # Build comprehensive stock data
            stock_data = {
                'ticker': original_ticker,
                'normalized_ticker': normalized_ticker,
                'company_name': company_name,
                'sector': sector,
                'current_price': round(current_price, 2),
                'previous_close': round(previous_close, 2),
                'change': round(change, 2),
                'change_percent': round(change_percent, 2),
                'day_high': round(float(latest_data['High']), 2),
                'day_low': round(float(latest_data['Low']), 2),
                'volume': int(latest_data['Volume']),
                'market_cap': market_cap,
                'market_cap_category': market_cap_category,
                
                # Additional financial metrics from yfinance info
                'pe_ratio': info.get('trailingPE', 0),
                'pb_ratio': info.get('priceToBook', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
                
                # Data source and timestamp
                'data_source': 'yfinance_real_time',
                'timestamp': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'exchange': 'NSE' if '.NS' in normalized_ticker else 'BSE' if '.BO' in normalized_ticker else 'INDEX'
            }
            
            return stock_data
            
        except Exception as e:
            logger.error(f"Error in _fetch_stock_data_sync for {normalized_ticker}: {e}")
            raise
    
    def _get_market_cap_category(self, market_cap: int) -> str:
        """
        Categorize market cap for Indian stocks
        Based on Indian market standards (in INR crores)
        """
        if market_cap == 0:
            return "Unknown"
        
        # Convert to crores (market_cap is usually in the stock's currency)
        market_cap_crores = market_cap / 10000000  # Assuming market_cap is in INR
        
        if market_cap_crores >= 20000:  # ₹20,000 crores+
            return "Large Cap"
        elif market_cap_crores >= 5000:  # ₹5,000 - ₹20,000 crores
            return "Mid Cap"
        elif market_cap_crores >= 500:   # ₹500 - ₹5,000 crores
            return "Small Cap"
        else:
            return "Micro Cap"
    
    async def get_portfolio_data(self, tickers: List[str]) -> List[Dict[str, Any]]:
        """
        Get real-time data for multiple stocks concurrently
        
        Args:
            tickers: List of stock ticker symbols
            
        Returns:
            List of stock data dictionaries with real-time prices
        """
        if not tickers:
            return []
        
        logger.info(f"🔄 Fetching real-time data for {len(tickers)} stocks: {', '.join(tickers)}")
        
        # Create concurrent tasks for all tickers
        tasks = [self.get_stock_data(ticker) for ticker in tickers]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            portfolio_data = []
            successful_fetches = 0
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to fetch data for {tickers[i]}: {result}")
                    portfolio_data.append(self._get_fallback_data(tickers[i]))
                else:
                    portfolio_data.append(result)
                    if result.get('data_source') == 'yfinance_real_time':
                        successful_fetches += 1
            
            logger.info(f"✅ Successfully fetched real-time data for {successful_fetches}/{len(tickers)} stocks")
            return portfolio_data
            
        except Exception as e:
            logger.error(f"❌ Error in batch data fetch: {e}")
            return [self._get_fallback_data(ticker) for ticker in tickers]
    
    async def get_market_indices(self) -> Dict[str, Any]:
        """Get major Indian market indices data"""
        major_indices = ['^NSEI', '^BSESN', '^NSEBANK']  # NIFTY 50, SENSEX, BANK NIFTY
        
        logger.info("🔄 Fetching market indices data...")
        
        try:
            index_data = await self.get_portfolio_data(major_indices)
            
            indices_dict = {
                'nifty_50': index_data[0] if len(index_data) > 0 else {},
                'sensex': index_data[1] if len(index_data) > 1 else {},
                'bank_nifty': index_data[2] if len(index_data) > 2 else {}
            }
            
            logger.info("✅ Market indices data fetched successfully")
            return indices_dict
            
        except Exception as e:
            logger.error(f"❌ Error fetching market indices: {e}")
            return {
                'nifty_50': self._get_fallback_data('^NSEI'),
                'sensex': self._get_fallback_data('^BSESN'),
                'bank_nifty': self._get_fallback_data('^NSEBANK')
            }
    
    async def get_multiple_prices(self, tickers: List[str]) -> Dict[str, float]:
        """
        Get current prices for multiple stocks quickly
        Optimized for just price fetching (used by portfolio valuation)
        
        Args:
            tickers: List of stock ticker symbols
            
        Returns:
            Dictionary mapping ticker to current price
        """
        portfolio_data = await self.get_portfolio_data(tickers)
        
        prices = {}
        for data in portfolio_data:
            ticker = data.get('ticker')
            price = data.get('current_price', 0.0)
            if ticker:
                prices[ticker] = price
        
        return prices
    
    def _get_fallback_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get fallback data when real data is unavailable
        Now provides more realistic fallback data
        """
        # Some realistic price ranges for common Indian stocks
        fallback_prices = {
            'RELIANCE': 2400,
            'TCS': 3500,
            'INFY': 1800,
            'HDFCBANK': 1600,
            'ICICIBANK': 1100,
            'SBIN': 600,
            'ITC': 400,
            'HINDUNILVR': 2500,
            'BHARTIARTL': 900,
            'KOTAKBANK': 1800
        }
        
        base_price = fallback_prices.get(ticker.upper(), 1000.0)
        
        return {
            'ticker': ticker,
            'normalized_ticker': self._normalize_ticker(ticker),
            'company_name': f"{ticker} Limited",
            'sector': 'Unknown',
            'current_price': base_price,
            'previous_close': base_price,
            'change': 0.0,
            'change_percent': 0.0,
            'day_high': base_price * 1.02,
            'day_low': base_price * 0.98,
            'volume': 1000000,
            'market_cap': 0,
            'market_cap_category': 'Unknown',
            'pe_ratio': 20.0,
            'pb_ratio': 3.0,
            'dividend_yield': 1.5,
            'fifty_two_week_high': base_price * 1.5,
            'fifty_two_week_low': base_price * 0.7,
            'data_source': 'fallback',
            'timestamp': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'exchange': 'NSE',
            'error': 'Real market data unavailable - using fallback data'
        }
    
    def clear_cache(self):
        """Clear the data cache"""
        cache_size = len(self.cache)
        self.cache.clear()
        logger.info(f"📊 Market data cache cleared ({cache_size} entries removed)")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics"""
        now = datetime.now()
        valid_entries = 0
        expired_entries = 0
        
        for key, data in self.cache.items():
            cache_time = data.get('cache_timestamp')
            if cache_time and (now - cache_time).total_seconds() < self.cache_ttl:
                valid_entries += 1
            else:
                expired_entries += 1
        
        return {
            'total_cache_entries': len(self.cache),
            'valid_entries': valid_entries,
            'expired_entries': expired_entries,
            'cache_ttl_seconds': self.cache_ttl,
            'cache_hit_rate': f"{(valid_entries / len(self.cache) * 100):.1f}%" if self.cache else "0%"
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check by testing data fetch for a common stock
        """
        test_ticker = "RELIANCE"
        start_time = time.time()
        
        try:
            data = await self.get_stock_data(test_ticker)
            response_time = (time.time() - start_time) * 1000  # ms
            
            is_healthy = (
                data.get('data_source') == 'yfinance_real_time' and
                data.get('current_price', 0) > 0 and
                response_time < 5000  # Less than 5 seconds
            )
            
            return {
                'status': 'healthy' if is_healthy else 'degraded',
                'response_time_ms': round(response_time, 2),
                'test_ticker': test_ticker,
                'data_source': data.get('data_source'),
                'price_fetched': data.get('current_price', 0),
                'cache_stats': self.get_cache_stats(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'response_time_ms': (time.time() - start_time) * 1000,
                'timestamp': datetime.now().isoformat()
            }

# Create singleton instance with new implementation
market_data_service = RealTimeMarketDataService() 