"""
Market Data Service for TAI-Roaster
Provides real-time market data integration using the intelligence module
"""

import sys
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime, timedelta

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from intelligence.config import RealMarketDataProvider
except ImportError as e:
    print(f"Warning: Market data provider import failed: {e}")
    RealMarketDataProvider = None

logger = logging.getLogger(__name__)

class MarketDataService:
    """Service for real-time market data"""
    
    def __init__(self):
        """Initialize market data service"""
        try:
            if RealMarketDataProvider:
                self.provider = RealMarketDataProvider()
                self.initialized = True
            else:
                self.provider = None
                self.initialized = False
            
            self.cache = {}
            self.cache_ttl = 300  # 5 minutes
            logger.info("📊 Market Data Service initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Market Data Service: {e}")
            self.initialized = False
            self.provider = None
    
    async def get_stock_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get comprehensive stock data for a single ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing stock data
        """
        if not self.initialized:
            return self._get_fallback_data(ticker)
        
        # Check cache first
        cache_key = f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Get data using intelligence module provider
            stock_data = await asyncio.to_thread(
                self.provider.get_stock_data, ticker
            )
            
            # Cache the result
            self.cache[cache_key] = stock_data
            
            return stock_data
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return self._get_fallback_data(ticker)
    
    async def get_portfolio_data(self, tickers: List[str]) -> List[Dict[str, Any]]:
        """
        Get data for multiple stocks concurrently
        
        Args:
            tickers: List of stock ticker symbols
            
        Returns:
            List of stock data dictionaries
        """
        if not tickers:
            return []
        
        # Create concurrent tasks for all tickers
        tasks = [self.get_stock_data(ticker) for ticker in tickers]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            portfolio_data = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error fetching data for {tickers[i]}: {result}")
                    portfolio_data.append(self._get_fallback_data(tickers[i]))
                else:
                    portfolio_data.append(result)
            
            return portfolio_data
            
        except Exception as e:
            logger.error(f"Error in batch data fetch: {e}")
            return [self._get_fallback_data(ticker) for ticker in tickers]
    
    async def get_market_indices(self) -> Dict[str, Any]:
        """Get major market indices data"""
        major_indices = ['^NSEI', '^BSESN', '^NSEBANK']  # NIFTY 50, SENSEX, BANK NIFTY
        
        try:
            index_data = await self.get_portfolio_data(major_indices)
            
            return {
                'nifty_50': index_data[0] if len(index_data) > 0 else {},
                'sensex': index_data[1] if len(index_data) > 1 else {},
                'bank_nifty': index_data[2] if len(index_data) > 2 else {}
            }
            
        except Exception as e:
            logger.error(f"Error fetching market indices: {e}")
            return {}
    
    def _get_fallback_data(self, ticker: str) -> Dict[str, Any]:
        """Get fallback data when real data is unavailable"""
        return {
            'ticker': ticker,
            'current_price': 100.0,  # Placeholder price
            'change': 0.0,
            'change_percent': 0.0,
            'volume': 0,
            'market_cap': 0,
            'pe_ratio': 0,
            'data_source': 'fallback',
            'timestamp': datetime.now().isoformat(),
            'error': 'Real market data unavailable'
        }
    
    def clear_cache(self):
        """Clear the data cache"""
        self.cache.clear()
        logger.info("📊 Market data cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cache_size': len(self.cache),
            'cache_keys': list(self.cache.keys())
        }

# Create singleton instance
market_data_service = MarketDataService() 