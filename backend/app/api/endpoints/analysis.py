from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict
import uuid
import asyncio
import json
import os
from datetime import datetime
import re
import time
import sys
from pathlib import Path
import logging

from app.models.analysis import (
    AnalysisRequest,
    AnalysisRequestResponse,
    AnalysisStatusResponse,
    AnalysisResultResponse,
    AnalysisStatus
)
from app.api.endpoints.onboarding import sessions_storage
from app.api.endpoints.portfolio import portfolios_storage

# Import file saver for loading actual analysis results
from app.utils.file_saver import analysis_file_saver

# Import market data service for real-time prices
from app.services.market_data_service import market_data_service

logger = logging.getLogger(__name__)

# Add enhanced analysis imports
try:
    from app.schemas.enhanced_analysis import (
        EnhancedAnalysisRequest,
        EnhancedAnalysisResponse,
        AnalysisStatus as EnhancedAnalysisStatus,
        AnalysisError
    )
    from app.services.intelligence_service import intelligence_service
    from app.services.market_data_service import market_data_service
    from app.core.ml_models import model_manager
    ENHANCED_ANALYSIS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Enhanced analysis not available: {e}")
    ENHANCED_ANALYSIS_AVAILABLE = False

router = APIRouter()

# In-memory storage for analysis requests (kept for compatibility)
analysis_requests: Dict[str, AnalysisRequest] = {}
analysis_results: Dict[str, any] = {}


def camel_to_snake(data):
    """
    Convert camelCase keys to snake_case recursively for JSON data.
    Also handles specific format conversions for compatibility.
    """
    if isinstance(data, dict):
        new_data = {}
        for key, value in data.items():
            # Convert camelCase to snake_case
            snake_key = re.sub(r'([A-Z])', r'_\1', key).lower()
            
            # Special handling for insights field
            if snake_key == 'insights' and isinstance(value, list):
                # Convert list of insights to a dictionary format
                # Use generic keys for the list items
                insights_dict = {}
                for i, insight in enumerate(value):
                    insights_dict[f"insight_{i}"] = insight
                new_data[snake_key] = insights_dict
            else:
                new_data[snake_key] = camel_to_snake(value)
        return new_data
    elif isinstance(data, list):
        return [camel_to_snake(item) for item in data]
    else:
        return data


def load_demo_analysis():
    """
    Load the demo portfolio analysis data.
    """
    processed_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "processed")
    json_file_path = os.path.join(processed_dir, "analysis_demo-portfolio-analysis.json")
    
    if os.path.exists(json_file_path):
        try:
            with open(json_file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Invalid demo analysis JSON file format")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load demo analysis file: {str(e)}")
    else:
        raise HTTPException(status_code=500, detail="Demo analysis file not found")


def load_analysis_by_id(analysis_id: str):
    """
    Load analysis data by ID, handling both full UUIDs and truncated UUIDs.
    Tries multiple strategies to find the analysis file.
    """
    try:
        # Strategy 1: Try to load using the full analysis_id as-is
        success, analysis_data, error = analysis_file_saver.get_analysis_by_id(analysis_id)
        if success:
            return analysis_data
        
        # Strategy 2: Try with truncated UUID (first 8 characters) for portfolio-analysis files
        truncated_id = analysis_id[:8]
        portfolio_analysis_id = f"portfolio-analysis-{truncated_id}"
        success, analysis_data, error = analysis_file_saver.get_analysis_by_id(portfolio_analysis_id)
        if success:
            return analysis_data
        
        # Strategy 3: Try with just the truncated ID
        success, analysis_data, error = analysis_file_saver.get_analysis_by_id(truncated_id)
        if success:
            return analysis_data
        
        # Strategy 4: Fallback to demo analysis if nothing found
        print(f"Warning: No analysis found for ID {analysis_id}, falling back to demo data")
        return load_demo_analysis()
        
    except Exception as e:
        print(f"Error loading analysis {analysis_id}: {e}, falling back to demo data")
        return load_demo_analysis()


@router.post("/analysis/request", response_model=AnalysisRequestResponse)
async def request_analysis(
    background_tasks: BackgroundTasks,
    session_id: str,
    analysis_type: str = "comprehensive"
):
    """
    Request a new portfolio analysis.
    Now returns demo analysis immediately instead of processing in background.
    """
    try:
        # Check if session exists
        if session_id not in sessions_storage:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if session_id not in portfolios_storage:
            raise HTTPException(status_code=404, detail="Portfolio data not found for session")
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Create analysis request (for compatibility)
        request = AnalysisRequest(
            request_id=request_id,
            session_id=session_id,
            status=AnalysisStatus.COMPLETED,  # Immediately mark as completed
            progress=100,
            completed_at=datetime.now()
        )
        
        # Store request
        analysis_requests[request_id] = request
        
        # Load and store demo analysis results immediately
        demo_analysis = load_demo_analysis()
        analysis_results[request_id] = demo_analysis
        
        return AnalysisRequestResponse(
            success=True,
            request_id=request_id,
            message="Analysis completed using demo data",
            estimated_processing_time="0 seconds"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to request analysis: {str(e)}")


@router.get("/analysis/{request_id}/status", response_model=AnalysisStatusResponse)
async def get_analysis_status(request_id: str):
    """
    Get the status of an analysis request.
    """
    try:
        if request_id not in analysis_requests:
            raise HTTPException(status_code=404, detail="Analysis request not found")
        
        request = analysis_requests[request_id]
        
        # Since we now return demo data immediately, all analyses are completed
        return AnalysisStatusResponse(
            success=True,
            request_id=request_id,
            status=AnalysisStatus.COMPLETED,
            progress=100,
            estimated_remaining=None,
            error_message=request.error_message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analysis status: {str(e)}")


@router.get("/analysis/{request_id}", response_model=AnalysisResultResponse)
async def get_analysis_results(request_id: str):
    """
    Get the results of a completed analysis.
    Loads actual analysis data by ID, with fallback to demo data.
    """
    try:
        # Load analysis data by ID (tries multiple strategies to find the file)
        analysis_data = load_analysis_by_id(request_id)
        
        # Return analysis data for the request
        from fastapi.responses import JSONResponse
        return JSONResponse(content={
            "success": True,
            "request_id": request_id,
            "analysis": analysis_data,  # Keep original camelCase for frontend
            "completed_at": datetime.now().isoformat()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analysis results: {str(e)}")


@router.delete("/analysis/{request_id}")
async def delete_analysis(request_id: str):
    """
    Delete an analysis request and its results.
    """
    try:
        if request_id not in analysis_requests:
            raise HTTPException(status_code=404, detail="Analysis request not found")
        
        # Delete request and results
        del analysis_requests[request_id]
        if request_id in analysis_results:
            del analysis_results[request_id]
        
        return {
            "success": True,
            "message": "Analysis deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete analysis: {str(e)}")


# ================================
# ENHANCED ANALYSIS ENDPOINTS
# ================================

@router.post("/analysis/enhanced", response_model=EnhancedAnalysisResponse)
async def enhanced_analysis(request: EnhancedAnalysisRequest):
    """
    New enhanced analysis endpoint using intelligence module
    Provides ML-powered portfolio analysis with TAI scoring
    """
    if not ENHANCED_ANALYSIS_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Enhanced analysis service not available. Please ensure intelligence module is properly installed."
        )
    
    start_time = time.time()
    
    try:
        # Check if intelligence service is initialized
        if not intelligence_service.initialized:
            # Try fallback analysis
            return await _fallback_enhanced_analysis(request)
        
        # Perform enhanced analysis using intelligence service
        enhanced_results = await intelligence_service.analyze_portfolio(
            request.portfolio, 
            request.user_profile
        )
        
        # Add metadata
        processing_time = time.time() - start_time
        enhanced_results['processing_time'] = processing_time
        enhanced_results['analysis_type'] = request.analysis_type
        enhanced_results['benchmark_used'] = request.benchmark
        enhanced_results['model_version'] = "1.0.0"
        
        # Convert to response model
        return EnhancedAnalysisResponse(**enhanced_results)
        
    except Exception as e:
        # Log error and try fallback
        print(f"Enhanced analysis error: {e}")
        return await _fallback_enhanced_analysis(request)


@router.get("/analysis/models/status")
async def get_models_status():
    """
    Get status of ML models and intelligence services
    """
    try:
        status = {
            "enhanced_analysis_available": ENHANCED_ANALYSIS_AVAILABLE,
            "intelligence_service_initialized": intelligence_service.initialized if ENHANCED_ANALYSIS_AVAILABLE else False,
            "market_data_service_initialized": market_data_service.initialized if ENHANCED_ANALYSIS_AVAILABLE else False,
        }
        
        if ENHANCED_ANALYSIS_AVAILABLE:
            status.update({
                "model_manager_info": model_manager.get_model_info(),
                "market_data_cache_stats": market_data_service.get_cache_stats()
            })
        
        return status
        
    except Exception as e:
        return {
            "enhanced_analysis_available": False,
            "error": str(e)
        }


@router.post("/analysis/models/reload")
async def reload_models():
    """
    Reload ML models (admin endpoint)
    """
    if not ENHANCED_ANALYSIS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Enhanced analysis not available")
    
    try:
        model_manager.reload_models()
        market_data_service.clear_cache()
        
        return {
            "success": True,
            "message": "Models reloaded successfully",
            "model_info": model_manager.get_model_info()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload models: {str(e)}")


async def _fallback_enhanced_analysis(request: EnhancedAnalysisRequest) -> EnhancedAnalysisResponse:
    """
    Fallback enhanced analysis using demo data when intelligence module is not available
    """
    try:
        # Load demo analysis and enhance it with basic ML-like structure
        demo_analysis = load_demo_analysis()
        
        # Calculate basic metrics
        total_invested = sum(holding.investment_amount for holding in request.portfolio.holdings)
        current_value = sum(
            holding.quantity * holding.current_price 
            for holding in request.portfolio.holdings
        )
        absolute_return = current_value - total_invested
        absolute_return_pct = (absolute_return / total_invested * 100) if total_invested > 0 else 0
        
        # Create mock ML predictions
        ml_predictions = []
        for holding in request.portfolio.holdings:
            ml_predictions.append({
                "ticker": holding.ticker,
                "xgboost_prediction": 0.05,  # Mock prediction
                "ngboost_mean": 0.03,
                "ngboost_std": 0.02,
                "ensemble_prediction": 0.04,
                "ensemble_confidence": 0.7
            })
        
        # Create mock TAI scores
        tai_scores = {
            "overall_score": 75.0,
            "performance_score": 78.0,
            "risk_management_score": 72.0,
            "diversification_score": 68.0,
            "ml_confidence_score": 70.0,
            "liquidity_score": 80.0,
            "cost_efficiency_score": 85.0,
            "grade": "B",
            "description": "Good portfolio with room for improvement"
        }
        
        # Create enhanced response structure
        enhanced_response = {
            "overall_score": 75.0,
            "risk_level": "Medium",
            "analysis_date": datetime.now().isoformat(),
            "portfolio_name": f"Portfolio Analysis {len(request.portfolio.holdings)} Holdings",
            "total_invested": total_invested,
            "current_value": current_value,
            "absolute_return": absolute_return,
            "absolute_return_pct": absolute_return_pct,
            "tai_scores": tai_scores,
            "ml_predictions": ml_predictions,
            "performance_metrics": {
                "total_return": absolute_return_pct,
                "annualized_return": absolute_return_pct * 0.8,
                "volatility": 18.5,
                "sharpe_ratio": 1.2,
                "sortino_ratio": 1.4,
                "max_drawdown": -8.3,
                "beta": 1.1,
                "alpha": 2.1
            },
            "allocation": {
                "sector_allocation": {"Technology": 30.0, "Financial": 25.0, "Healthcare": 20.0, "Others": 25.0},
                "market_cap_allocation": {"Large Cap": 70.0, "Mid Cap": 20.0, "Small Cap": 10.0},
                "concentration_risk": 35.0,
                "diversification_ratio": 0.75
            },
            "stocks": [
                {
                    "ticker": holding.ticker,
                    "quantity": holding.quantity,
                    "current_price": holding.current_price,
                    "investment_amount": holding.investment_amount,
                    "current_value": holding.quantity * holding.current_price,
                    "weight": (holding.investment_amount / total_invested * 100) if total_invested > 0 else 0,
                    "ml_prediction": 0.04,
                    "confidence_score": 0.7,
                    "recommendation": "Hold"
                }
                for holding in request.portfolio.holdings
            ],
            "action_plan": {
                "immediate_actions": ["Review sector concentration", "Consider rebalancing"],
                "short_term_goals": ["Improve diversification", "Monitor risk metrics"],
                "long_term_strategy": ["Build systematic approach", "Regular portfolio review"],
                "rebalancing_suggestions": ["Reduce overweight positions", "Add defensive assets"]
            },
            "recommendations": [
                "Consider improving sector diversification",
                "Monitor concentration risk in top holdings",
                "Review risk-return profile quarterly"
            ],
            "risk_warnings": [
                {
                    "severity": "Medium",
                    "category": "Concentration Risk",
                    "message": "Portfolio has moderate concentration in top holdings",
                    "recommendation": "Consider reducing position sizes in largest holdings"
                }
            ],
            "opportunities": [
                {
                    "category": "Diversification",
                    "description": "Add exposure to underrepresented sectors",
                    "potential_impact": "Improved risk-adjusted returns",
                    "action_required": "Research and add positions in missing sectors"
                }
            ],
            "hygiene": {
                "sector_balance": "Moderate",
                "concentration_check": "Needs attention",
                "risk_alignment": "Good"
            },
            "rating": {
                "grade": "B",
                "score": 75.0,
                "description": "Good portfolio with room for improvement"
            },
            "analysis_type": request.analysis_type,
            "benchmark_used": request.benchmark,
            "model_version": "1.0.0-fallback",
            "processing_time": 0.5
        }
        
        return EnhancedAnalysisResponse(**enhanced_response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fallback enhanced analysis failed: {str(e)}")


@router.get("/market/prices")
async def get_real_time_prices(tickers: str):
    """
    Get real-time prices for multiple stocks
    
    Args:
        tickers: Comma-separated list of stock tickers (e.g., "RELIANCE,TCS,INFY")
    
    Returns:
        Real-time price data for requested stocks
    """
    try:
        ticker_list = [ticker.strip().upper() for ticker in tickers.split(',') if ticker.strip()]
        
        if not ticker_list:
            raise HTTPException(status_code=400, detail="No valid tickers provided")
        
        if len(ticker_list) > 50:  # Limit to 50 stocks per request
            raise HTTPException(status_code=400, detail="Too many tickers requested (max 50)")
        
        logger.info(f"🔄 Real-time price request for: {', '.join(ticker_list)}")
        
        # Get real-time data
        portfolio_data = await market_data_service.get_portfolio_data(ticker_list)
        
        # Extract just the essential price info
        price_data = []
        for data in portfolio_data:
            price_info = {
                'ticker': data.get('ticker'),
                'current_price': data.get('current_price', 0),
                'change': data.get('change', 0),
                'change_percent': data.get('change_percent', 0),
                'volume': data.get('volume', 0),
                'data_source': data.get('data_source'),
                'last_updated': data.get('timestamp')
            }
            price_data.append(price_info)
        
        return {
            "success": True,
            "data": price_data,
            "total_stocks": len(price_data),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to fetch real-time prices: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch real-time prices: {str(e)}")


@router.get("/market/indices")
async def get_market_indices():
    """
    Get current market indices (NIFTY 50, SENSEX, BANK NIFTY)
    """
    try:
        logger.info("🔄 Fetching market indices...")
        
        indices_data = await market_data_service.get_market_indices()
        
        return {
            "success": True,
            "data": indices_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch market indices: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch market indices: {str(e)}")


@router.get("/market/health")
async def market_data_health_check():
    """
    Check health of market data service
    """
    try:
        health_data = await market_data_service.health_check()
        
        status_code = 200 if health_data.get('status') == 'healthy' else 503
        
        return JSONResponse(
            status_code=status_code,
            content={
                "success": health_data.get('status') == 'healthy',
                "health": health_data
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Market data health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        ) 