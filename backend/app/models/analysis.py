from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Recommendation(str, Enum):
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"


class AnalysisStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class StockAnalysis(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    score: float = Field(..., ge=0, le=100, description="Analysis score (0-100)")
    recommendation: Recommendation = Field(..., description="Buy/Hold/Sell recommendation")
    reasoning: str = Field(..., description="Reasoning for the recommendation")
    risk_level: RiskLevel = Field(..., description="Risk level assessment")
    target_price: Optional[float] = Field(None, description="Target price")
    price_change: Optional[float] = Field(None, description="Expected price change %")


class AllocationBreakdown(BaseModel):
    sectors: Dict[str, float] = Field(default_factory=dict, description="Sector allocation percentages")
    asset_types: Dict[str, float] = Field(default_factory=dict, description="Asset type allocation percentages")
    risk_levels: Dict[str, float] = Field(default_factory=dict, description="Risk level allocation percentages")


class PortfolioAnalysis(BaseModel):
    overall_score: float = Field(..., ge=0, le=100, description="Overall portfolio score")
    risk_level: RiskLevel = Field(..., description="Overall portfolio risk level")
    diversification_score: float = Field(..., ge=0, le=100, description="Diversification score")
    summary: str = Field(..., description="Analysis summary")
    recommendations: List[str] = Field(default_factory=list, description="Portfolio recommendations")
    red_flags: List[str] = Field(default_factory=list, description="Risk warnings and red flags")
    allocation: Dict[str, AllocationBreakdown] = Field(
        default_factory=dict, 
        description="Current and recommended allocations"
    )
    stocks: List[StockAnalysis] = Field(default_factory=list, description="Individual stock analyses")
    insights: Dict[str, str] = Field(default_factory=dict, description="Stock-specific insights")


class AnalysisRequest(BaseModel):
    request_id: str = Field(..., description="Unique request identifier")
    session_id: str = Field(..., description="User session ID")
    status: AnalysisStatus = Field(default=AnalysisStatus.PENDING, description="Analysis status")
    created_at: datetime = Field(default_factory=datetime.now, description="Request creation time")
    completed_at: Optional[datetime] = Field(None, description="Analysis completion time")
    progress: float = Field(default=0, ge=0, le=100, description="Analysis progress percentage")
    error_message: Optional[str] = Field(None, description="Error message if analysis failed")


class AnalysisRequestResponse(BaseModel):
    success: bool
    message: str
    request_id: str
    estimated_time: Optional[int] = None  # seconds


class AnalysisStatusResponse(BaseModel):
    success: bool
    request_id: str
    status: AnalysisStatus
    progress: float
    estimated_remaining: Optional[int] = None  # seconds
    error_message: Optional[str] = None


class AnalysisResultResponse(BaseModel):
    success: bool
    request_id: str
    analysis: Optional[PortfolioAnalysis] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None 