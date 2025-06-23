"""
Output Pydantic Models

This module defines Pydantic models for validating and structuring API responses:
- Prediction results with confidence intervals
- Stock recommendations with scores
- Portfolio analysis results
- Error responses and status updates

Key models:
- PredictionResult: Individual stock prediction with confidence
- StockRecommendation: Recommendation with score and reasoning
- PortfolioAnalysis: Complete portfolio analysis results
- APIResponse: Standard API response wrapper
- ErrorResponse: Standardized error response format
- InsightData: LLM-generated insights and explanations

Response features:
- Consistent response structure across all endpoints
- Type-safe response data
- Standardized error handling
- Metadata inclusion (timestamps, versions, etc.)
- Confidence scores and uncertainty quantification
- Actionable insights and recommendations

Integration:
- Used by routes/predict.py for response formatting
- Integrated with services/formatter.py
- Consumed by frontend for type-safe API calls
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class PredictionResult(BaseModel):
    """Individual stock prediction with confidence intervals"""
    ticker: str = Field(..., description="Stock ticker symbol")
    predicted_return: float = Field(..., description="Predicted return percentage")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence (0-1)")
    prediction_date: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "ticker": "RELIANCE.NS",
                "predicted_return": 12.5,
                "confidence": 0.85,
                "prediction_date": "2024-01-15T10:30:00"
            }
        }

class ReturnRange(BaseModel):
    """Return range with confidence intervals"""
    lower_bound: float = Field(..., description="Lower bound of return range")
    upper_bound: float = Field(..., description="Upper bound of return range")
    median: float = Field(..., description="Median expected return")
    confidence_level: float = Field(default=0.95, description="Confidence level for range")

class BacktestMetrics(BaseModel):
    """Backtesting performance metrics"""
    total_return: float = Field(..., description="Total return percentage")
    annualized_return: float = Field(..., description="Annualized return percentage")
    volatility: float = Field(..., description="Return volatility")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown percentage")
    win_rate: float = Field(..., ge=0, le=1, description="Win rate (0-1)")

class PricePoint(BaseModel):
    """Price point with timestamp"""
    date: datetime = Field(..., description="Price date")
    price: float = Field(..., description="Price value")
    volume: Optional[float] = Field(None, description="Trading volume")

class BufferRange(BaseModel):
    """Buffer range for price movements"""
    lower_buffer: float = Field(..., description="Lower buffer percentage")
    upper_buffer: float = Field(..., description="Upper buffer percentage")
    target_price: float = Field(..., description="Target price")

class PortfolioAnalysis(BaseModel):
    """Complete portfolio analysis results"""
    total_value: float = Field(..., description="Total portfolio value")
    total_return: float = Field(..., description="Total return percentage")
    risk_score: float = Field(..., ge=0, le=100, description="Risk score (0-100)")
    diversification_score: float = Field(..., ge=0, le=100, description="Diversification score")
    recommendations: List[str] = Field(..., description="Portfolio recommendations")

class APIResponse(BaseModel):
    """Standard API response wrapper"""
    success: bool = Field(..., description="Request success status")
    message: str = Field(..., description="Response message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    version: str = Field(default="1.0.0", description="API version")

class ErrorResponse(BaseModel):
    """Standardized error response format"""
    success: bool = Field(default=False, description="Request success status")
    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Human-readable error message")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")

class InsightData(BaseModel):
    """LLM-generated insights and explanations"""
    insight_type: str = Field(..., description="Type of insight")
    title: str = Field(..., description="Insight title")
    description: str = Field(..., description="Detailed insight description")
    confidence: float = Field(..., ge=0, le=1, description="Insight confidence")
    actionable: bool = Field(default=False, description="Whether insight is actionable")
    recommended_action: Optional[str] = Field(None, description="Recommended action if actionable")
