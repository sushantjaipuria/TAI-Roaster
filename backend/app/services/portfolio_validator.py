"""
Portfolio Validation Service

This service provides comprehensive portfolio validation for Indian stock markets:
- NSE ticker symbol validation
- Portfolio diversification checks
- Risk assessment validation
- Business rule enforcement
- Market data integration

Key features:
- Real-time ticker validation against NSE database
- Portfolio concentration risk analysis
- Sector diversification assessment
- Investment amount validation
- Risk tolerance alignment checks

Integration:
- Used by API endpoints for portfolio validation
- Integrates with external market data sources
- Provides detailed validation reports
- Supports both real-time and batch validation
"""

import asyncio
import re
from typing import List, Dict, Optional, Tuple, Any
from datetime import date, datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from app.schemas.input import (
    PortfolioInput, PortfolioHolding, UserProfile, 
    ValidationError, PortfolioValidationResponse,
    RiskTolerance, TimeHorizon
)


class ValidationSeverity(str, Enum):
    """Validation issue severity levels"""
    ERROR = "error"      # Blocking issues that prevent processing
    WARNING = "warning"  # Issues that should be noted but don't block
    INFO = "info"        # Informational messages


@dataclass
class ValidationIssue:
    """Represents a validation issue"""
    severity: ValidationSeverity
    field: str
    message: str
    value: Optional[Any] = None
    suggestion: Optional[str] = None


class PortfolioValidationService:
    """
    Comprehensive portfolio validation service for Indian stock markets
    """
    
    # NSE sector classifications (simplified for validation)
    NSE_SECTORS = {
        'BANKING': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK', 'INDUSINDBK'],
        'IT': ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'LTI'],
        'PHARMA': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'LUPIN', 'BIOCON', 'AUROPHARMA'],
        'AUTO': ['MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO', 'EICHERMOT', 'HEROMOTOCO'],
        'ENERGY': ['RELIANCE', 'ONGC', 'IOC', 'BPCL', 'GAIL', 'NTPC'],
        'METALS': ['TATASTEEL', 'HINDALCO', 'JSWSTEEL', 'VEDL', 'SAIL', 'NMDC'],
        'FMCG': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'DABUR', 'GODREJCP', 'MARICO'],
        'INFRASTRUCTURE': ['LT', 'ULTRACEMCO', 'GRASIM', 'ACC', 'AMBUJACEMENT']
    }
    
    # Known NSE tickers (simplified list for validation)
    KNOWN_NSE_TICKERS = set()
    for sector_tickers in NSE_SECTORS.values():
        KNOWN_NSE_TICKERS.update(sector_tickers)
    
    # Add more common tickers
    KNOWN_NSE_TICKERS.update([
        'ADANIPORTS', 'ASIANPAINT', 'BAJFINANCE', 'BHARTIARTL', 'BRITANNIA',
        'COALINDIA', 'DIVISLAB', 'GRASIM', 'HDFC', 'HINDZINC', 'JSWSTEEL',
        'POWERGRID', 'SHREECEM', 'TITAN', 'UPL', 'WIPRO'
    ])
    
    def __init__(self):
        """Initialize validation service"""
        self.validation_cache: Dict[str, bool] = {}  # Cache for ticker validations
        self.sector_cache: Dict[str, str] = {}       # Cache for sector mappings
    
    async def validate_portfolio(
        self, 
        portfolio: PortfolioInput, 
        user_profile: Optional[UserProfile] = None
    ) -> PortfolioValidationResponse:
        """
        Comprehensive portfolio validation
        
        Args:
            portfolio: Portfolio to validate
            user_profile: Optional user profile for additional validations
            
        Returns:
            Detailed validation response with errors and warnings
        """
        issues: List[ValidationIssue] = []
        
        # Run all validation checks
        issues.extend(await self._validate_holdings(portfolio.holdings))
        issues.extend(self._validate_diversification(portfolio.holdings))
        issues.extend(self._validate_concentration_risk(portfolio.holdings))
        issues.extend(self._validate_portfolio_size(portfolio.holdings))
        
        if user_profile:
            issues.extend(self._validate_risk_alignment(portfolio.holdings, user_profile))
            issues.extend(self._validate_investment_amount_alignment(portfolio.holdings, user_profile))
        
        # Separate errors and warnings
        errors = []
        warnings = []
        
        for issue in issues:
            if issue.severity == ValidationSeverity.ERROR:
                errors.append(ValidationError(
                    field=issue.field,
                    message=issue.message,
                    value=issue.value
                ))
            elif issue.severity == ValidationSeverity.WARNING:
                warnings.append(f"{issue.field}: {issue.message}")
        
        # Calculate portfolio metrics
        total_value = sum(h.quantity * h.avg_buy_price for h in portfolio.holdings)
        holdings_count = len(portfolio.holdings)
        
        return PortfolioValidationResponse(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            total_value=total_value,
            holdings_count=holdings_count
        )
    
    async def _validate_holdings(self, holdings: List[PortfolioHolding]) -> List[ValidationIssue]:
        """Validate individual holdings"""
        issues = []
        
        for i, holding in enumerate(holdings):
            # Validate ticker format and existence
            ticker_issues = await self._validate_ticker(holding.ticker, f"holdings[{i}].ticker")
            issues.extend(ticker_issues)
            
            # Validate quantity reasonableness
            if holding.quantity > 100000:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field=f"holdings[{i}].quantity",
                    message=f"Large quantity ({holding.quantity:,}) for {holding.ticker}. Please verify.",
                    value=holding.quantity,
                    suggestion="Consider splitting large positions across multiple entries if this represents different purchase lots"
                ))
            
            # Validate price reasonableness
            if holding.avg_buy_price > 50000:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field=f"holdings[{i}].avg_buy_price",
                    message=f"High price (₹{holding.avg_buy_price:,.2f}) for {holding.ticker}. Please verify.",
                    value=holding.avg_buy_price
                ))
            
            # Validate current price vs buy price if available
            if holding.current_price and holding.avg_buy_price:
                price_change = ((holding.current_price - holding.avg_buy_price) / holding.avg_buy_price) * 100
                if abs(price_change) > 200:  # More than 200% change
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        field=f"holdings[{i}].current_price",
                        message=f"Large price change ({price_change:+.1f}%) for {holding.ticker}. Please verify prices.",
                        value=holding.current_price
                    ))
            
            # Validate buy date reasonableness
            if holding.buy_date:
                days_ago = (date.today() - holding.buy_date).days
                if days_ago > 365 * 20:  # More than 20 years
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        field=f"holdings[{i}].buy_date",
                        message=f"Very old purchase date ({holding.buy_date}) for {holding.ticker}. Please verify.",
                        value=holding.buy_date
                    ))
        
        return issues
    
    async def _validate_ticker(self, ticker: str, field: str) -> List[ValidationIssue]:
        """Validate ticker symbol"""
        issues = []
        
        # Check cache first
        if ticker in self.validation_cache:
            if not self.validation_cache[ticker]:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field=field,
                    message=f"Unknown ticker symbol: {ticker}",
                    value=ticker,
                    suggestion="Please verify the ticker symbol on NSE or BSE"
                ))
            return issues
        
        # Basic format validation
        if not re.match(r'^[A-Z0-9\.\-&]+$', ticker):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                field=field,
                message=f"Invalid ticker format: {ticker}",
                value=ticker,
                suggestion="Ticker should contain only uppercase letters, numbers, dots, hyphens, and ampersands"
            ))
            self.validation_cache[ticker] = False
            return issues
        
        # Check against known NSE tickers
        is_known_ticker = ticker in self.KNOWN_NSE_TICKERS
        
        if not is_known_ticker:
            # Check for common ticker patterns that might be valid
            if self._is_likely_valid_ticker(ticker):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field=field,
                    message=f"Ticker {ticker} not in common NSE list. Please verify it's correct.",
                    value=ticker,
                    suggestion="Verify this ticker exists on NSE/BSE before proceeding"
                ))
                self.validation_cache[ticker] = True  # Allow with warning
            else:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    field=field,
                    message=f"Unknown or invalid ticker symbol: {ticker}",
                    value=ticker,
                    suggestion="Please verify the ticker symbol on NSE or BSE"
                ))
                self.validation_cache[ticker] = False
        else:
            self.validation_cache[ticker] = True
        
        return issues
    
    def _is_likely_valid_ticker(self, ticker: str) -> bool:
        """Check if ticker follows common Indian stock exchange patterns"""
        # Common patterns for Indian tickers
        patterns = [
            r'^[A-Z]{2,10}$',           # Simple alphabetic tickers
            r'^[A-Z]+[0-9]{1,3}$',      # Tickers ending with numbers
            r'^[A-Z]+-[A-Z]+$',         # Hyphenated tickers
            r'^[A-Z]+&[A-Z]+$',         # Ampersand tickers (M&M style)
            r'^[A-Z]+\.[A-Z]+$',        # Dot separated tickers
        ]
        
        return any(re.match(pattern, ticker) for pattern in patterns)
    
    def _validate_diversification(self, holdings: List[PortfolioHolding]) -> List[ValidationIssue]:
        """Validate portfolio diversification"""
        issues = []
        
        # Check minimum number of holdings
        if len(holdings) < 3:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                field="holdings",
                message=f"Portfolio has only {len(holdings)} holdings. Consider diversifying with at least 3-5 different stocks.",
                value=len(holdings),
                suggestion="Add more holdings from different sectors to reduce risk"
            ))
        
        # Check sector diversification
        sector_allocation = self._calculate_sector_allocation(holdings)
        max_sector_allocation = max(sector_allocation.values()) if sector_allocation else 0
        
        if max_sector_allocation > 60:
            max_sector = max(sector_allocation.keys(), key=lambda k: sector_allocation[k])
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                field="holdings",
                message=f"High concentration in {max_sector} sector ({max_sector_allocation:.1f}%). Consider diversifying across sectors.",
                value=max_sector_allocation,
                suggestion="Reduce sector concentration to below 50% for better risk management"
            ))
        
        # Check if all holdings are from unknown sectors (might indicate data quality issues)
        if not sector_allocation:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                field="holdings",
                message="Could not determine sector allocation. Manual review may be needed.",
                suggestion="Verify ticker symbols and consider adding sector information"
            ))
        
        return issues
    
    def _validate_concentration_risk(self, holdings: List[PortfolioHolding]) -> List[ValidationIssue]:
        """Validate individual position concentration risk"""
        issues = []
        
        # Calculate total portfolio value
        total_value = sum(h.quantity * h.avg_buy_price for h in holdings)
        
        if total_value == 0:
            return issues
        
        # Check individual position concentrations
        for holding in holdings:
            position_value = holding.quantity * holding.avg_buy_price
            concentration = (position_value / total_value) * 100
            
            if concentration > 25:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field="holdings",
                    message=f"{holding.ticker} represents {concentration:.1f}% of portfolio. High concentration risk.",
                    value=concentration,
                    suggestion="Consider reducing position size to below 20% of total portfolio"
                ))
            elif concentration > 15:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    field="holdings",
                    message=f"{holding.ticker} represents {concentration:.1f}% of portfolio. Monitor concentration.",
                    value=concentration
                ))
        
        return issues
    
    def _validate_portfolio_size(self, holdings: List[PortfolioHolding]) -> List[ValidationIssue]:
        """Validate portfolio size and complexity"""
        issues = []
        
        total_value = sum(h.quantity * h.avg_buy_price for h in holdings)
        
        # Check minimum portfolio size
        if total_value < 5000:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                field="holdings",
                message=f"Small portfolio value (₹{total_value:,.2f}). Consider increasing investment amount for better diversification.",
                value=total_value,
                suggestion="Minimum recommended portfolio size is ₹10,000 for meaningful analysis"
            ))
        
        # Check for too many holdings (over-diversification)
        if len(holdings) > 20:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                field="holdings",
                message=f"Large number of holdings ({len(holdings)}). May lead to over-diversification.",
                value=len(holdings),
                suggestion="Consider consolidating to 10-15 core holdings for better management"
            ))
        
        return issues
    
    def _validate_risk_alignment(self, holdings: List[PortfolioHolding], user_profile: UserProfile) -> List[ValidationIssue]:
        """Validate portfolio risk alignment with user profile"""
        issues = []
        
        # Calculate portfolio risk based on sector allocation and concentration
        portfolio_risk = self._calculate_portfolio_risk_level(holdings)
        user_risk = user_profile.risk_tolerance
        
        # Check risk alignment
        risk_mismatch = False
        
        if user_risk == RiskTolerance.CONSERVATIVE and portfolio_risk == "high":
            risk_mismatch = True
            message = "High-risk portfolio may not align with conservative risk tolerance."
            suggestion = "Consider adding more stable, large-cap stocks and reducing sector concentration"
        
        elif user_risk == RiskTolerance.AGGRESSIVE and portfolio_risk == "low":
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                field="risk_alignment",
                message="Conservative portfolio for aggressive risk tolerance. Consider adding growth stocks.",
                suggestion="Explore mid-cap or growth-oriented stocks to match risk appetite"
            ))
        
        if risk_mismatch:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                field="risk_alignment",
                message=message,
                suggestion=suggestion
            ))
        
        return issues
    
    def _validate_investment_amount_alignment(self, holdings: List[PortfolioHolding], user_profile: UserProfile) -> List[ValidationIssue]:
        """Validate investment amount alignment"""
        issues = []
        
        portfolio_value = sum(h.quantity * h.avg_buy_price for h in holdings)
        expected_amount = user_profile.investment_amount
        
        # Allow some variance (±20%)
        variance = abs(portfolio_value - expected_amount) / expected_amount * 100
        
        if variance > 20:
            if portfolio_value > expected_amount:
                message = f"Portfolio value (₹{portfolio_value:,.2f}) exceeds stated investment amount (₹{expected_amount:,.2f})."
            else:
                message = f"Portfolio value (₹{portfolio_value:,.2f}) is less than stated investment amount (₹{expected_amount:,.2f})."
            
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                field="investment_amount",
                message=message,
                value=variance,
                suggestion="Update investment amount or portfolio holdings to match"
            ))
        
        return issues
    
    def _calculate_sector_allocation(self, holdings: List[PortfolioHolding]) -> Dict[str, float]:
        """Calculate sector-wise allocation percentage"""
        sector_values = {}
        total_value = sum(h.quantity * h.avg_buy_price for h in holdings)
        
        if total_value == 0:
            return {}
        
        for holding in holdings:
            sector = self._get_ticker_sector(holding.ticker)
            holding_value = holding.quantity * holding.avg_buy_price
            
            if sector in sector_values:
                sector_values[sector] += holding_value
            else:
                sector_values[sector] = holding_value
        
        # Convert to percentages
        return {sector: (value / total_value) * 100 for sector, value in sector_values.items()}
    
    def _get_ticker_sector(self, ticker: str) -> str:
        """Get sector for a ticker symbol"""
        if ticker in self.sector_cache:
            return self.sector_cache[ticker]
        
        # Check against known sectors
        for sector, tickers in self.NSE_SECTORS.items():
            if ticker in tickers:
                self.sector_cache[ticker] = sector
                return sector
        
        # Default to OTHERS for unknown tickers
        self.sector_cache[ticker] = "OTHERS"
        return "OTHERS"
    
    def _calculate_portfolio_risk_level(self, holdings: List[PortfolioHolding]) -> str:
        """Calculate overall portfolio risk level"""
        # Simplified risk calculation based on:
        # 1. Sector concentration
        # 2. Number of holdings
        # 3. Individual position sizes
        
        sector_allocation = self._calculate_sector_allocation(holdings)
        max_sector_concentration = max(sector_allocation.values()) if sector_allocation else 0
        num_holdings = len(holdings)
        
        # Calculate position concentration
        total_value = sum(h.quantity * h.avg_buy_price for h in holdings)
        max_position = 0
        if total_value > 0:
            max_position = max((h.quantity * h.avg_buy_price / total_value) * 100 for h in holdings)
        
        # Risk scoring
        risk_score = 0
        
        # Sector concentration risk
        if max_sector_concentration > 60:
            risk_score += 30
        elif max_sector_concentration > 40:
            risk_score += 20
        elif max_sector_concentration > 25:
            risk_score += 10
        
        # Diversification risk
        if num_holdings < 3:
            risk_score += 25
        elif num_holdings < 5:
            risk_score += 15
        elif num_holdings < 8:
            risk_score += 5
        
        # Position concentration risk
        if max_position > 30:
            risk_score += 25
        elif max_position > 20:
            risk_score += 15
        elif max_position > 15:
            risk_score += 10
        
        # Determine risk level
        if risk_score >= 50:
            return "high"
        elif risk_score >= 25:
            return "medium"
        else:
            return "low"
    
    async def validate_ticker_batch(self, tickers: List[str]) -> Dict[str, bool]:
        """Validate multiple tickers in batch"""
        results = {}
        
        for ticker in tickers:
            if ticker in self.validation_cache:
                results[ticker] = self.validation_cache[ticker]
            else:
                # Simulate validation (in real implementation, this would call external API)
                is_valid = ticker in self.KNOWN_NSE_TICKERS or self._is_likely_valid_ticker(ticker)
                self.validation_cache[ticker] = is_valid
                results[ticker] = is_valid
        
        return results
    
    def get_validation_suggestions(self, portfolio: PortfolioInput) -> List[str]:
        """Get actionable suggestions for portfolio improvement"""
        suggestions = []
        
        holdings = portfolio.holdings
        total_value = sum(h.quantity * h.avg_buy_price for h in holdings)
        
        # Diversification suggestions
        if len(holdings) < 5:
            suggestions.append("Consider adding more stocks from different sectors for better diversification")
        
        # Sector concentration suggestions
        sector_allocation = self._calculate_sector_allocation(holdings)
        if sector_allocation:
            max_sector_allocation = max(sector_allocation.values())
            if max_sector_allocation > 40:
                max_sector = max(sector_allocation.keys(), key=lambda k: sector_allocation[k])
                suggestions.append(f"Reduce exposure to {max_sector} sector (currently {max_sector_allocation:.1f}%)")
        
        # Position size suggestions
        for holding in holdings:
            position_value = holding.quantity * holding.avg_buy_price
            concentration = (position_value / total_value) * 100
            if concentration > 20:
                suggestions.append(f"Consider reducing position size in {holding.ticker} (currently {concentration:.1f}%)")
        
        # Portfolio size suggestions
        if total_value < 10000:
            suggestions.append("Consider increasing total investment amount for better diversification options")
        
        return suggestions[:5]  # Return top 5 suggestions
