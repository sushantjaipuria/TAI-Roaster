from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any, Optional
from datetime import datetime, date
import logging
from pydantic import BaseModel

from ...services.portfolio_performance_calculator import (
    PortfolioPerformanceCalculator, 
    PortfolioHolding
)
from ...core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Pydantic models for request/response
class PortfolioHoldingRequest(BaseModel):
    ticker: str
    quantity: float
    purchase_date: date
    purchase_price: float

class PortfolioPerformanceRequest(BaseModel):
    holdings: List[PortfolioHoldingRequest]
    time_periods: List[str] = ["6M", "1Y", "3Y", "5Y"]

class PerformanceMetrics(BaseModel):
    timeframe: str
    returns: float
    annualizedReturn: float
    benchmarkReturns: float
    outperformance: float
    metrics: Dict[str, float]

class PortfolioPerformanceResponse(BaseModel):
    performance_metrics: List[PerformanceMetrics]
    time_series_data: Dict[str, List[Dict[str, Any]]]
    calculation_timestamp: str
    data_sources: Dict[str, str]

@router.get("/portfolio-performance/{portfolio_id}")
async def get_portfolio_performance(
    portfolio_id: str,
    periods: str = Query(default="6M,1Y,3Y,5Y", description="Comma-separated time periods")
) -> PortfolioPerformanceResponse:
    """
    Calculate portfolio performance vs NIFTY50 benchmark using XIRR methodology.
    
    This endpoint implements the comprehensive XIRR calculation requirements:
    - Real-time stock price fetching with caching
    - XIRR calculation using Newton-Raphson method
    - NIFTY50 benchmark comparison with dividend adjustment
    - Time series data generation for interactive charts
    - Comprehensive risk metrics (Alpha, Beta, Sharpe Ratio, etc.)
    
    Args:
        portfolio_id: Unique identifier for the portfolio
        periods: Time periods for analysis (6M,1Y,3Y,5Y)
        
    Returns:
        Comprehensive performance analysis with XIRR calculations
    """
    try:
        # Parse time periods
        time_periods = [p.strip() for p in periods.split(",")]
        
        # Validate time periods
        valid_periods = {"6M", "1Y", "3Y", "5Y"}
        invalid_periods = set(time_periods) - valid_periods
        if invalid_periods:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid time periods: {invalid_periods}. Valid options: {valid_periods}"
            )
        
        # Get portfolio holdings from database
        holdings = await get_portfolio_holdings_with_dates(portfolio_id)
        
        if not holdings:
            raise HTTPException(
                status_code=404, 
                detail=f"No holdings found for portfolio {portfolio_id}"
            )
        
        # Initialize performance calculator
        calculator = PortfolioPerformanceCalculator()
        
        # Calculate performance vs benchmark
        performance_data = await calculator.calculate_portfolio_vs_benchmark_performance(
            holdings, time_periods
        )
        
        if "error" in performance_data:
            raise HTTPException(
                status_code=500,
                detail=f"Performance calculation failed: {performance_data['error']}"
            )
        
        return PortfolioPerformanceResponse(**performance_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Portfolio performance endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.post("/calculate-performance")
async def calculate_portfolio_performance(
    request: PortfolioPerformanceRequest
) -> PortfolioPerformanceResponse:
    """
    Calculate portfolio performance for provided holdings data.
    
    This endpoint allows direct calculation without requiring database storage.
    Useful for testing and ad-hoc analysis.
    
    Args:
        request: Portfolio holdings and time periods for analysis
        
    Returns:
        Comprehensive performance analysis with XIRR calculations
    """
    try:
        # Convert request holdings to internal format
        holdings = [
            PortfolioHolding(
                ticker=h.ticker,
                quantity=h.quantity,
                purchase_date=h.purchase_date,
                purchase_price=h.purchase_price
            )
            for h in request.holdings
        ]
        
        # Initialize performance calculator
        calculator = PortfolioPerformanceCalculator()
        
        # Calculate performance vs benchmark
        performance_data = await calculator.calculate_portfolio_vs_benchmark_performance(
            holdings, request.time_periods
        )
        
        if "error" in performance_data:
            raise HTTPException(
                status_code=500,
                detail=f"Performance calculation failed: {performance_data['error']}"
            )
        
        return PortfolioPerformanceResponse(**performance_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Performance calculation error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/benchmark-data")
async def get_benchmark_data(
    periods: str = Query(default="6M,1Y,3Y,5Y", description="Comma-separated time periods")
) -> Dict[str, float]:
    """
    Get NIFTY50 benchmark returns for specified time periods.
    
    Useful for standalone benchmark analysis and validation.
    
    Args:
        periods: Time periods for benchmark data
        
    Returns:
        Dictionary of time period to annualized return percentages
    """
    try:
        time_periods = [p.strip() for p in periods.split(",")]
        
        calculator = PortfolioPerformanceCalculator()
        
        # Get current date range for NIFTY data
        end_date = datetime.now().date()
        start_date = date(end_date.year - 6, end_date.month, end_date.day)  # 6 years back
        
        # Fetch NIFTY data
        nifty_data = await calculator.get_nifty50_historical_data(start_date, end_date)
        
        # Calculate returns
        nifty_returns = calculator.calculate_nifty50_returns(nifty_data, time_periods)
        
        return {
            "benchmark_returns": nifty_returns,
            "calculation_timestamp": datetime.now().isoformat(),
            "data_source": "yahoo_finance_nifty50"
        }
        
    except Exception as e:
        logger.error(f"Benchmark data error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch benchmark data: {str(e)}"
        )

# Helper function to get portfolio holdings (to be implemented based on your database structure)
async def get_portfolio_holdings_with_dates(portfolio_id: str) -> List[PortfolioHolding]:
    """
    Retrieve portfolio holdings with purchase dates from database.
    
    This function should be implemented based on your actual database schema.
    The function should return holdings with:
    - ticker: Stock symbol
    - quantity: Number of shares
    - purchase_date: Date of purchase
    - purchase_price: Price per share at purchase
    
    Args:
        portfolio_id: Portfolio identifier
        
    Returns:
        List of portfolio holdings with purchase information
    """
    # TODO: Implement based on your database schema
    # This is a placeholder implementation
    
    # Example implementation structure:
    # async with get_db_session() as session:
    #     query = select(PortfolioHolding).where(PortfolioHolding.portfolio_id == portfolio_id)
    #     result = await session.execute(query)
    #     holdings = result.scalars().all()
    #     
    #     return [
    #         PortfolioHolding(
    #             ticker=h.ticker,
    #             quantity=h.quantity,
    #             purchase_date=h.purchase_date,
    #             purchase_price=h.purchase_price
    #         )
    #         for h in holdings
    #     ]
    
    # For now, return empty list - replace with actual database query
    logger.warning(f"get_portfolio_holdings_with_dates not implemented for portfolio {portfolio_id}")
    return []

# Health check endpoint for the performance calculator
@router.get("/health")
async def performance_calculator_health() -> Dict[str, Any]:
    """
    Health check for the portfolio performance calculator service.
    
    Returns:
        Service health status and capability information
    """
    try:
        calculator = PortfolioPerformanceCalculator()
        
        # Test XIRR calculation
        from ...services.portfolio_performance_calculator import test_xirr_calculation
        xirr_test_passed = test_xirr_calculation()
        
        # Test market data access (simplified check)
        try:
            import yfinance as yf
            test_data = yf.download("^NSEI", period="1d", progress=False)
            market_data_available = not test_data.empty
        except:
            market_data_available = False
        
        return {
            "service": "portfolio_performance_calculator",
            "status": "healthy",
            "xirr_calculation": "working" if xirr_test_passed else "error",
            "market_data_access": "working" if market_data_available else "limited",
            "supported_periods": ["6M", "1Y", "3Y", "5Y"],
            "calculation_methods": ["xirr", "newton_raphson", "bisection_fallback"],
            "benchmark": "NIFTY50",
            "cache_duration": {
                "stock_prices": "5_minutes",
                "nifty_data": "30_minutes"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "service": "portfolio_performance_calculator", 
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        } 