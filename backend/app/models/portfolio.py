from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from datetime import datetime


class PortfolioItem(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    quantity: float = Field(..., gt=0, description="Number of shares")
    avg_price: float = Field(..., gt=0, description="Average price per share")
    current_price: Optional[float] = Field(None, description="Current market price")
    total_value: Optional[float] = Field(None, description="Total position value")
    allocation: Optional[float] = Field(None, description="Allocation percentage")
    
    @field_validator('ticker')
    @classmethod
    def validate_ticker(cls, v):
        if not v or len(v) > 10:
            raise ValueError('Invalid ticker symbol')
        return v.upper().strip()
    
    @field_validator('quantity')
    @classmethod
    def validate_quantity(cls, v):
        if v <= 0:
            raise ValueError('Quantity must be positive')
        return v
    
    @field_validator('avg_price')
    @classmethod
    def validate_avg_price(cls, v):
        if v <= 0:
            raise ValueError('Average price must be positive')
        return v


class Portfolio(BaseModel):
    items: List[PortfolioItem] = Field(..., description="Portfolio holdings")
    total_value: float = Field(default=0, description="Total portfolio value")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    @field_validator('items')
    @classmethod
    def validate_items(cls, v):
        if not v:
            raise ValueError('Portfolio cannot be empty')
        if len(v) > 1000:
            raise ValueError('Too many portfolio items')
        
        # Check for duplicate tickers
        tickers = [item.ticker for item in v]
        if len(tickers) != len(set(tickers)):
            raise ValueError('Duplicate tickers found')
        
        return v
    
    def calculate_total_value(self):
        """Calculate total portfolio value"""
        self.total_value = sum(
            item.quantity * (item.current_price or item.avg_price) 
            for item in self.items
        )
        return self.total_value
    
    def calculate_allocations(self):
        """Calculate allocation percentages"""
        if self.total_value > 0:
            for item in self.items:
                item_value = item.quantity * (item.current_price or item.avg_price)
                item.allocation = (item_value / self.total_value) * 100


class PortfolioUploadRequest(BaseModel):
    session_id: str = Field(..., description="User session ID")


class PortfolioUploadResponse(BaseModel):
    success: bool
    message: str
    portfolio: Optional[Portfolio] = None
    errors: List[str] = Field(default_factory=list)


class PortfolioUpdateRequest(BaseModel):
    portfolio: Portfolio


class PortfolioUpdateResponse(BaseModel):
    success: bool
    message: str
    portfolio: Optional[Portfolio] = None 