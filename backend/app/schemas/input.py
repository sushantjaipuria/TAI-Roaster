"""
Input Pydantic Models

This module defines Pydantic models for validating incoming API requests:
- Portfolio data structure validation
- File upload validation
- Prediction request parameters

Key models:
- PortfolioHolding: Individual stock holding data
- Portfolio: Complete portfolio with list of holdings
- PredictionRequest: Parameters for prediction API calls
- FileUploadRequest: File upload metadata and validation
- UserPreferences: User settings for predictions

Validation features:
- Type checking for all fields
- Required vs optional field validation
- Range validation for numeric fields
- Format validation for dates and strings
- Custom validators for stock symbols
- File format validation

Integration:
- Used by routes/predict.py for request validation
- Integrated with services/parser.py for data processing
- Provides type safety for API endpoints
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Union
from datetime import date, datetime
from enum import Enum


class RiskTolerance(str, Enum):
    """User risk tolerance levels"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class InvestmentStyle(str, Enum):
    """Investment style preferences"""
    GROWTH = "growth"
    VALUE = "value"
    INCOME = "income"
    BALANCED = "balanced"


class TimeHorizon(str, Enum):
    """Investment time horizon"""
    SHORT = "short"    # < 3 years
    MEDIUM = "medium"  # 3-10 years
    LONG = "long"      # > 10 years


class PortfolioHolding(BaseModel):
    """Individual stock holding in a portfolio"""
    ticker: str = Field(..., description="Stock ticker symbol (NSE format)")
    quantity: int = Field(..., gt=0, description="Number of shares (must be positive)")
    avg_buy_price: float = Field(..., gt=0, description="Average buy price per share")
    buy_date: Optional[date] = Field(None, description="Purchase date")
    current_price: Optional[float] = Field(None, gt=0, description="Current market price")
    
    @field_validator('ticker')
    @classmethod
    def validate_ticker(cls, v):
        """Validate NSE ticker format"""
        if not v:
            raise ValueError('Ticker symbol is required')
        
        # Clean and normalize ticker
        ticker = v.upper().strip()
        
        # Basic NSE ticker validation (1-10 alphanumeric characters)
        if not ticker.replace('.', '').replace('-', '').isalnum():
            raise ValueError('Ticker must contain only alphanumeric characters, dots, and hyphens')
        
        if len(ticker) < 1 or len(ticker) > 15:
            raise ValueError('Ticker must be between 1 and 15 characters')
        
        return ticker
    
    @field_validator('quantity')
    @classmethod
    def validate_quantity(cls, v):
        """Validate quantity is positive integer"""
        if v <= 0:
            raise ValueError('Quantity must be a positive integer')
        return v
    
    @field_validator('avg_buy_price')
    @classmethod
    def validate_avg_buy_price(cls, v):
        """Validate average buy price is positive"""
        if v <= 0:
            raise ValueError('Average buy price must be positive')
        
        # Check for reasonable price range (0.01 to 1,00,000 INR)
        if v < 0.01 or v > 100000:
            raise ValueError('Average buy price must be between ₹0.01 and ₹1,00,000')
        
        return round(v, 2)
    
    @field_validator('current_price')
    @classmethod
    def validate_current_price(cls, v):
        """Validate current price if provided"""
        if v is not None:
            if v <= 0:
                raise ValueError('Current price must be positive')
            
            # Check for reasonable price range
            if v < 0.01 or v > 100000:
                raise ValueError('Current price must be between ₹0.01 and ₹1,00,000')
            
            return round(v, 2)
        return v
    
    @field_validator('buy_date')
    @classmethod
    def validate_buy_date(cls, v):
        """Validate buy date is not in future"""
        if v is not None:
            if v > date.today():
                raise ValueError('Buy date cannot be in the future')
            
            # Check for reasonable date range (not before 1990)
            if v < date(1990, 1, 1):
                raise ValueError('Buy date seems too old (before 1990)')
        
        return v


class PortfolioInput(BaseModel):
    """Complete portfolio input data"""
    holdings: List[PortfolioHolding] = Field(..., min_items=1, max_items=100, 
                                           description="List of portfolio holdings")
    name: Optional[str] = Field(None, max_length=100, description="Portfolio name")
    
    @field_validator('holdings')
    @classmethod
    def validate_holdings(cls, v):
        """Validate portfolio holdings list"""
        if not v:
            raise ValueError('Portfolio must contain at least one holding')
        
        if len(v) > 100:
            raise ValueError('Portfolio cannot contain more than 100 holdings')
        
        # Check for duplicate tickers
        tickers = [holding.ticker for holding in v]
        if len(tickers) != len(set(tickers)):
            raise ValueError('Portfolio contains duplicate ticker symbols')
        
        # Check minimum portfolio value
        total_value = sum(holding.quantity * holding.avg_buy_price for holding in v)
        if total_value < 1000:
            raise ValueError('Portfolio total value must be at least ₹1,000')
        
        return v
    
    @model_validator(mode='after')
    def validate_portfolio_diversity(self):
        """Validate portfolio diversity requirements"""
        holdings = self.holdings
        
        if len(holdings) < 3:
            # For portfolios with less than 3 stocks, check concentration
            total_value = sum(h.quantity * h.avg_buy_price for h in holdings)
            for holding in holdings:
                holding_value = holding.quantity * holding.avg_buy_price
                concentration = (holding_value / total_value) * 100
                if concentration > 80:
                    raise ValueError(f'Single stock ({holding.ticker}) represents {concentration:.1f}% of portfolio. Consider diversifying.')
        
        return self


class UserProfile(BaseModel):
    """User profile and investment preferences"""
    risk_tolerance: RiskTolerance = Field(..., description="Risk tolerance level")
    investment_amount: float = Field(..., gt=0, description="Total investment amount")
    investment_style: InvestmentStyle = Field(..., description="Investment style preference")
    time_horizon: TimeHorizon = Field(..., description="Investment time horizon")
    goals: List[str] = Field(default_factory=list, max_items=10, description="Investment goals")
    age: Optional[int] = Field(None, ge=18, le=100, description="User age")
    annual_income: Optional[float] = Field(None, gt=0, description="Annual income")
    
    @field_validator('investment_amount')
    @classmethod
    def validate_investment_amount(cls, v):
        """Validate investment amount"""
        if v < 1000:
            raise ValueError('Investment amount must be at least ₹1,000')
        if v > 10000000:  # 1 crore
            raise ValueError('Investment amount cannot exceed ₹1,00,00,000')
        return v
    
    @field_validator('goals')
    @classmethod
    def validate_goals(cls, v):
        """Validate investment goals"""
        if v:
            # Clean and validate goals
            cleaned_goals = []
            for goal in v:
                if isinstance(goal, str) and goal.strip():
                    cleaned_goals.append(goal.strip()[:100])  # Limit length
            return cleaned_goals
        return v


class FileUploadRequest(BaseModel):
    """File upload metadata and validation"""
    filename: str = Field(..., description="Name of uploaded file")
    file_size: int = Field(..., gt=0, description="File size in bytes")
    content_type: Optional[str] = Field(None, description="MIME content type")
    
    @field_validator('filename')
    @classmethod
    def validate_filename(cls, v):
        """Validate file name and extension"""
        if not v:
            raise ValueError('Filename is required')
        
        # Check for valid extensions
        valid_extensions = ['.csv', '.xlsx', '.xls']
        if not any(v.lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(f'File must have one of these extensions: {", ".join(valid_extensions)}')
        
        # Basic filename validation
        if len(v) > 255:
            raise ValueError('Filename too long (max 255 characters)')
        
        return v
    
    @field_validator('file_size')
    @classmethod
    def validate_file_size(cls, v):
        """Validate file size"""
        max_size = 10 * 1024 * 1024  # 10MB
        if v > max_size:
            raise ValueError(f'File size too large (max {max_size // (1024*1024)}MB)')
        
        if v <= 0:
            raise ValueError('File size must be positive')
        
        return v


class AnalysisRequest(BaseModel):
    """Complete analysis request with portfolio and user profile"""
    portfolio: PortfolioInput = Field(..., description="Portfolio data")
    user_profile: UserProfile = Field(..., description="User profile and preferences")
    analysis_type: str = Field("comprehensive", description="Type of analysis to perform")
    include_recommendations: bool = Field(True, description="Include buy/sell recommendations")
    
    @field_validator('analysis_type')
    @classmethod
    def validate_analysis_type(cls, v):
        """Validate analysis type"""
        valid_types = ['comprehensive', 'quick', 'risk_only', 'diversification_only']
        if v not in valid_types:
            raise ValueError(f'Analysis type must be one of: {", ".join(valid_types)}')
        return v


class BulkPortfolioUpload(BaseModel):
    """Bulk portfolio upload from file"""
    file_request: FileUploadRequest = Field(..., description="File upload metadata")
    user_profile: UserProfile = Field(..., description="User profile")
    override_validation: bool = Field(False, description="Override basic validation rules")
    
    @field_validator('override_validation')
    @classmethod
    def validate_override(cls, v):
        """Limit override usage"""
        # In production, you might want to restrict this to admin users
        return v


class PortfolioUpdateRequest(BaseModel):
    """Request to update existing portfolio"""
    portfolio_id: Optional[str] = Field(None, description="Existing portfolio ID")
    holdings_to_add: List[PortfolioHolding] = Field(default_factory=list, description="New holdings to add")
    holdings_to_remove: List[str] = Field(default_factory=list, description="Ticker symbols to remove")
    holdings_to_update: List[PortfolioHolding] = Field(default_factory=list, description="Holdings to update")
    
    @model_validator(mode='after')
    def validate_update_request(self):
        """Validate that at least one update operation is specified"""
        add_list = self.holdings_to_add
        remove_list = self.holdings_to_remove
        update_list = self.holdings_to_update
        
        if not any([add_list, remove_list, update_list]):
            raise ValueError('At least one update operation must be specified')
        
        return self


# Response models for API endpoints
class ValidationError(BaseModel):
    """Validation error details"""
    field: str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Error message")
    value: Optional[Union[str, int, float]] = Field(None, description="Invalid value")


class PortfolioValidationResponse(BaseModel):
    """Portfolio validation response"""
    is_valid: bool = Field(..., description="Whether portfolio is valid")
    errors: List[ValidationError] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    total_value: Optional[float] = Field(None, description="Total portfolio value")
    holdings_count: Optional[int] = Field(None, description="Number of holdings")


class FileParseResponse(BaseModel):
    """File parsing response"""
    success: bool = Field(..., description="Whether file was parsed successfully")
    portfolio: Optional[PortfolioInput] = Field(None, description="Parsed portfolio data")
    errors: List[str] = Field(default_factory=list, description="Parsing errors")
    warnings: List[str] = Field(default_factory=list, description="Parsing warnings")
    rows_processed: Optional[int] = Field(None, description="Number of rows processed")
    rows_skipped: Optional[int] = Field(None, description="Number of rows skipped")
