"""
Enhanced Analysis Schemas for TAI-Roaster Intelligence Integration
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum

from app.models.portfolio import PortfolioInput
from app.models.onboarding import UserProfile

class AnalysisType(str, Enum):
    """Analysis type options"""
    COMPREHENSIVE = "comprehensive"
    QUICK = "quick"
    DEEP_DIVE = "deep_dive"

class BenchmarkType(str, Enum):
    """Benchmark options for comparison"""
    NIFTY50 = "NIFTY50"
    SENSEX = "SENSEX"
    NIFTY_BANK = "NIFTY_BANK"
    CUSTOM = "CUSTOM"

class RiskLevel(str, Enum):
    """Risk level classifications"""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    VERY_HIGH = "Very High"

class StockRecommendation(str, Enum):
    """Stock-level recommendations"""
    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"
    STRONG_BUY = "Strong Buy"
    STRONG_SELL = "Strong Sell"

# Request Schemas
class EnhancedAnalysisRequest(BaseModel):
    """Request schema for enhanced analysis"""
    portfolio: PortfolioInput
    user_profile: UserProfile
    analysis_type: AnalysisType = AnalysisType.COMPREHENSIVE
    benchmark: BenchmarkType = BenchmarkType.NIFTY50
    include_llm_analysis: bool = False
    
    class Config:
        use_enum_values = True

# Response Component Schemas
class TAIScore(BaseModel):
    """TAI Score breakdown"""
    overall_score: float = Field(..., ge=0, le=100, description="Overall TAI Score (0-100)")
    performance_score: float = Field(..., ge=0, le=100, description="Performance component score")
    risk_management_score: float = Field(..., ge=0, le=100, description="Risk management score")
    diversification_score: float = Field(..., ge=0, le=100, description="Diversification score")
    ml_confidence_score: float = Field(..., ge=0, le=100, description="ML confidence score")
    liquidity_score: float = Field(..., ge=0, le=100, description="Liquidity score")
    cost_efficiency_score: float = Field(..., ge=0, le=100, description="Cost efficiency score")
    
    grade: str = Field(..., description="Letter grade (A-D)")
    description: str = Field(..., description="Score description")

class MLPrediction(BaseModel):
    """ML model prediction for a single stock"""
    ticker: str = Field(..., description="Stock ticker symbol")
    xgboost_prediction: float = Field(..., description="XGBoost model prediction")
    lightgbm_prediction: Optional[float] = Field(None, description="LightGBM model prediction")
    catboost_prediction: Optional[float] = Field(None, description="CatBoost model prediction")
    ngboost_mean: float = Field(..., description="NGBoost mean prediction")
    ngboost_std: float = Field(..., description="NGBoost standard deviation")
    ensemble_prediction: float = Field(..., description="Ensemble model prediction")
    ensemble_confidence: float = Field(..., ge=0, le=1, description="Ensemble confidence (0-1)")
    
    class Config:
        schema_extra = {
            "example": {
                "ticker": "RELIANCE.NS",
                "xgboost_prediction": 0.15,
                "ngboost_mean": 0.12,
                "ngboost_std": 0.05,
                "ensemble_prediction": 0.135,
                "ensemble_confidence": 0.85
            }
        }

class PerformanceMetrics(BaseModel):
    """Performance metrics"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    beta: float
    alpha: float
    var_95: Optional[float] = None
    expected_shortfall: Optional[float] = None
    
    class Config:
        schema_extra = {
            "example": {
                "total_return": 15.5,
                "annualized_return": 12.3,
                "volatility": 18.2,
                "sharpe_ratio": 1.2,
                "sortino_ratio": 1.5,
                "max_drawdown": -8.5,
                "beta": 1.1,
                "alpha": 2.3
            }
        }

class AllocationBreakdown(BaseModel):
    """Portfolio allocation breakdown"""
    sector_allocation: Dict[str, float] = Field(..., description="Sector-wise allocation percentages")
    market_cap_allocation: Dict[str, float] = Field(..., description="Market cap allocation")
    concentration_risk: float = Field(..., ge=0, le=100, description="Concentration risk percentage")
    diversification_ratio: float = Field(..., description="Diversification ratio")
    
    class Config:
        schema_extra = {
            "example": {
                "sector_allocation": {
                    "Technology": 25.5,
                    "Financial Services": 20.3,
                    "Healthcare": 15.2
                },
                "market_cap_allocation": {
                    "Large Cap": 65.0,
                    "Mid Cap": 25.0,
                    "Small Cap": 10.0
                },
                "concentration_risk": 35.5,
                "diversification_ratio": 0.75
            }
        }

class EnhancedStock(BaseModel):
    """Enhanced stock analysis with ML insights"""
    ticker: str
    company_name: Optional[str] = None
    quantity: float
    current_price: float
    investment_amount: float
    current_value: float
    weight: float
    
    # ML insights
    ml_prediction: float = Field(..., description="ML prediction value")
    confidence_score: float = Field(..., ge=0, le=1, description="Prediction confidence")
    recommendation: StockRecommendation
    
    # Risk metrics
    volatility: Optional[float] = None
    beta: Optional[float] = None
    
    # Performance
    returns_1d: Optional[float] = None
    returns_1w: Optional[float] = None
    returns_1m: Optional[float] = None
    returns_3m: Optional[float] = None
    
    class Config:
        use_enum_values = True

class ActionPlan(BaseModel):
    """Action plan recommendations"""
    immediate_actions: List[str] = Field(..., description="Actions to take immediately")
    short_term_goals: List[str] = Field(..., description="1-3 month goals")
    long_term_strategy: List[str] = Field(..., description="6+ month strategy")
    rebalancing_suggestions: List[str] = Field(..., description="Portfolio rebalancing suggestions")
    
    class Config:
        schema_extra = {
            "example": {
                "immediate_actions": [
                    "Reduce concentration in technology sector",
                    "Consider adding defensive stocks"
                ],
                "short_term_goals": [
                    "Achieve better sector diversification",
                    "Improve risk-adjusted returns"
                ],
                "long_term_strategy": [
                    "Build a core-satellite portfolio structure",
                    "Implement systematic rebalancing"
                ]
            }
        }

class RiskWarning(BaseModel):
    """Risk warning"""
    severity: str = Field(..., description="Warning severity (High/Medium/Low)")
    category: str = Field(..., description="Risk category")
    message: str = Field(..., description="Warning message")
    recommendation: str = Field(..., description="Recommended action")

class Opportunity(BaseModel):
    """Investment opportunity"""
    category: str = Field(..., description="Opportunity category")
    description: str = Field(..., description="Opportunity description")
    potential_impact: str = Field(..., description="Potential positive impact")
    action_required: str = Field(..., description="Action required to capitalize")

# Main Response Schema
class EnhancedAnalysisResponse(BaseModel):
    """Enhanced analysis response with ML insights and TAI scoring"""
    
    # Core Analysis Results
    overall_score: float = Field(..., ge=0, le=100, description="Overall TAI score")
    risk_level: RiskLevel
    analysis_date: str = Field(..., description="Analysis timestamp")
    portfolio_name: str = Field(..., description="Portfolio identifier")
    
    # Financial Summary
    total_invested: float = Field(..., description="Total amount invested")
    current_value: float = Field(..., description="Current portfolio value")
    absolute_return: float = Field(..., description="Absolute return amount")
    absolute_return_pct: float = Field(..., description="Absolute return percentage")
    
    # Enhanced Components
    tai_scores: TAIScore
    ml_predictions: List[MLPrediction]
    performance_metrics: PerformanceMetrics
    allocation: AllocationBreakdown
    stocks: List[EnhancedStock]
    
    # Recommendations and Insights
    action_plan: ActionPlan
    recommendations: List[str] = Field(..., description="Strategic recommendations")
    risk_warnings: List[RiskWarning] = Field(..., description="Risk warnings")
    opportunities: List[Opportunity] = Field(..., description="Investment opportunities")
    
    # Hygiene and Rating
    hygiene: Dict[str, Any] = Field(..., description="Portfolio hygiene checks")
    rating: Dict[str, Any] = Field(..., description="Overall rating information")
    
    # Analysis Metadata
    analysis_type: AnalysisType
    benchmark_used: BenchmarkType
    model_version: str = Field(default="1.0.0", description="Analysis model version")
    processing_time: Optional[float] = Field(None, description="Analysis processing time in seconds")
    
    class Config:
        use_enum_values = True
        schema_extra = {
            "example": {
                "overall_score": 85.3,
                "risk_level": "Medium",
                "analysis_date": "2024-01-15T10:30:00Z",
                "portfolio_name": "Growth Portfolio",
                "total_invested": 500000.0,
                "current_value": 575000.0,
                "absolute_return": 75000.0,
                "absolute_return_pct": 15.0
            }
        }

# Status and Progress Schemas
class AnalysisStatus(BaseModel):
    """Analysis status for tracking progress"""
    analysis_id: str
    status: str = Field(..., description="Current status (processing/completed/failed)")
    progress: float = Field(..., ge=0, le=100, description="Progress percentage")
    message: str = Field(..., description="Status message")
    estimated_completion: Optional[datetime] = None
    
class AnalysisError(BaseModel):
    """Analysis error details"""
    error_code: str
    error_message: str
    error_details: Optional[Dict[str, Any]] = None
    suggested_action: Optional[str] = None 