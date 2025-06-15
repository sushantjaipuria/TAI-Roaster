from pydantic import BaseModel, Field, validator
from typing import List, Optional
from enum import Enum


class RiskTolerance(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class InvestmentStyle(str, Enum):
    GROWTH = "growth"
    VALUE = "value"
    INCOME = "income"
    BALANCED = "balanced"


class TimeHorizon(str, Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class UserProfileRequest(BaseModel):
    risk_tolerance: RiskTolerance = Field(..., description="User's risk tolerance level")
    investment_amount: float = Field(..., gt=0, description="Investment amount in USD")
    investment_style: InvestmentStyle = Field(..., description="Preferred investment style")
    time_horizon: TimeHorizon = Field(..., description="Investment time horizon")
    goals: List[str] = Field(default_factory=list, description="Investment goals")
    
    @validator('investment_amount')
    def validate_investment_amount(cls, v):
        if v <= 0:
            raise ValueError('Investment amount must be positive')
        if v > 100_000_000:  # 100M limit
            raise ValueError('Investment amount too large')
        return v
    
    @validator('goals')
    def validate_goals(cls, v):
        if len(v) > 10:
            raise ValueError('Too many goals specified')
        return v


class UserProfileResponse(BaseModel):
    success: bool
    session_id: str
    message: str
    data: Optional[UserProfileRequest] = None


class OnboardingSession(BaseModel):
    session_id: str
    user_profile: UserProfileRequest
    created_at: str
    updated_at: str 