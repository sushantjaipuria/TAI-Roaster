import random
import asyncio
from typing import Dict, List
from datetime import datetime

from app.models.portfolio import Portfolio
from app.models.onboarding import UserProfileRequest
from app.models.analysis import (
    PortfolioAnalysis,
    StockAnalysis,
    AllocationBreakdown,
    RiskLevel,
    Recommendation
)


class MockAnalysisService:
    """
    Mock analysis service that generates realistic portfolio analysis data.
    In production, this would be replaced with actual AI/ML analysis.
    """
    
    # Mock data for stock sectors and companies
    STOCK_SECTORS = {
        'AAPL': 'Technology',
        'GOOGL': 'Technology',
        'MSFT': 'Technology',
        'AMZN': 'Consumer Discretionary',
        'TSLA': 'Consumer Discretionary',
        'JPM': 'Financial Services',
        'BAC': 'Financial Services',
        'JNJ': 'Healthcare',
        'PFE': 'Healthcare',
        'XOM': 'Energy',
        'CVX': 'Energy',
        'KO': 'Consumer Staples',
        'PG': 'Consumer Staples',
        'DIS': 'Communication Services',
        'NFLX': 'Communication Services',
    }
    
    RISK_PROFILES = {
        'AAPL': 'medium',
        'GOOGL': 'medium',
        'MSFT': 'low',
        'AMZN': 'high',
        'TSLA': 'high',
        'JPM': 'medium',
        'BAC': 'medium',
        'JNJ': 'low',
        'PFE': 'low',
        'XOM': 'medium',
        'CVX': 'medium',
        'KO': 'low',
        'PG': 'low',
        'DIS': 'medium',
        'NFLX': 'high',
    }
    
    async def analyze_portfolio(
        self, 
        portfolio: Portfolio, 
        user_profile: UserProfileRequest
    ) -> PortfolioAnalysis:
        """
        Generate mock portfolio analysis based on user profile and portfolio.
        """
        
        # Simulate processing delay
        await asyncio.sleep(1)
        
        # Generate stock analyses
        stock_analyses = []
        insights = {}
        
        for item in portfolio.items:
            stock_analysis = self._generate_stock_analysis(item, user_profile)
            stock_analyses.append(stock_analysis)
            insights[item.ticker] = self._generate_stock_insight(item, user_profile)
        
        # Calculate overall metrics
        overall_score = self._calculate_overall_score(stock_analyses, user_profile)
        risk_level = self._determine_portfolio_risk(stock_analyses)
        diversification_score = self._calculate_diversification_score(portfolio)
        
        # Generate allocation breakdowns
        current_allocation = self._calculate_current_allocation(portfolio)
        recommended_allocation = self._generate_recommended_allocation(user_profile)
        
        # Generate recommendations and red flags
        recommendations = self._generate_recommendations(portfolio, user_profile, stock_analyses)
        red_flags = self._generate_red_flags(portfolio, stock_analyses)
        
        # Generate summary
        summary = self._generate_summary(overall_score, risk_level, diversification_score)
        
        return PortfolioAnalysis(
            overall_score=overall_score,
            risk_level=risk_level,
            diversification_score=diversification_score,
            summary=summary,
            recommendations=recommendations,
            red_flags=red_flags,
            allocation={
                "current": current_allocation,
                "recommended": recommended_allocation
            },
            stocks=stock_analyses,
            insights=insights
        )
    
    def _generate_stock_analysis(self, item, user_profile: UserProfileRequest) -> StockAnalysis:
        """Generate analysis for individual stock."""
        ticker = item.ticker
        
        # Generate score based on various factors
        base_score = random.uniform(40, 90)
        
        # Adjust score based on user risk tolerance
        if user_profile.risk_tolerance == 'conservative':
            if ticker in ['JNJ', 'PG', 'KO', 'MSFT']:
                base_score += random.uniform(5, 15)
            elif ticker in ['TSLA', 'AMZN', 'NFLX']:
                base_score -= random.uniform(10, 20)
        elif user_profile.risk_tolerance == 'aggressive':
            if ticker in ['TSLA', 'AMZN', 'NFLX']:
                base_score += random.uniform(5, 15)
            elif ticker in ['JNJ', 'PG', 'KO']:
                base_score -= random.uniform(5, 10)
        
        score = max(0, min(100, base_score))
        
        # Determine recommendation
        if score >= 75:
            recommendation = Recommendation.BUY
        elif score >= 50:
            recommendation = Recommendation.HOLD
        else:
            recommendation = Recommendation.SELL
        
        # Get risk level
        risk_level = RiskLevel(self.RISK_PROFILES.get(ticker, 'medium'))
        
        # Generate reasoning
        reasoning = self._generate_stock_reasoning(ticker, score, recommendation, risk_level)
        
        # Generate target price
        current_price = item.current_price or item.avg_price
        price_change = random.uniform(-20, 30)  # -20% to +30%
        target_price = current_price * (1 + price_change / 100)
        
        return StockAnalysis(
            ticker=ticker,
            score=round(score, 1),
            recommendation=recommendation,
            reasoning=reasoning,
            risk_level=risk_level,
            target_price=round(target_price, 2),
            price_change=round(price_change, 1)
        )
    
    def _generate_stock_reasoning(self, ticker: str, score: float, recommendation: Recommendation, risk_level: RiskLevel) -> str:
        """Generate reasoning for stock recommendation."""
        reasons = []
        
        if score >= 80:
            reasons.append("Strong fundamentals and growth prospects")
        elif score >= 60:
            reasons.append("Solid performance with moderate upside potential")
        else:
            reasons.append("Concerns about current valuation and market position")
        
        if risk_level == RiskLevel.LOW:
            reasons.append("Low volatility and stable dividend history")
        elif risk_level == RiskLevel.HIGH:
            reasons.append("High growth potential but increased volatility")
        
        if recommendation == Recommendation.BUY:
            reasons.append("Favorable risk-adjusted returns expected")
        elif recommendation == Recommendation.SELL:
            reasons.append("Better opportunities available elsewhere")
        
        return ". ".join(reasons) + "."
    
    def _generate_stock_insight(self, item, user_profile: UserProfileRequest) -> str:
        """Generate AI-like insight for stock."""
        ticker = item.ticker
        sector = self.STOCK_SECTORS.get(ticker, 'Unknown')
        
        insights = [
            f"{ticker} operates in the {sector} sector, which shows strong fundamentals.",
            f"Given your {user_profile.risk_tolerance} risk tolerance, {ticker} aligns well with your investment profile.",
            f"The stock's current allocation of {item.allocation:.1f}% in your portfolio is within recommended ranges.",
            f"Market trends suggest {ticker} has potential for long-term growth based on industry analysis.",
        ]
        
        return random.choice(insights)
    
    def _calculate_overall_score(self, stock_analyses: List[StockAnalysis], user_profile: UserProfileRequest) -> float:
        """Calculate overall portfolio score."""
        if not stock_analyses:
            return 0
        
        # Weighted average of stock scores
        total_score = sum(analysis.score for analysis in stock_analyses)
        base_score = total_score / len(stock_analyses)
        
        # Adjust based on user profile alignment
        adjustment = 0
        risk_counts = {'low': 0, 'medium': 0, 'high': 0}
        for analysis in stock_analyses:
            risk_counts[analysis.risk_level] += 1
        
        total_stocks = len(stock_analyses)
        if user_profile.risk_tolerance == 'conservative':
            adjustment = (risk_counts['low'] / total_stocks) * 10 - (risk_counts['high'] / total_stocks) * 10
        elif user_profile.risk_tolerance == 'aggressive':
            adjustment = (risk_counts['high'] / total_stocks) * 5 - (risk_counts['low'] / total_stocks) * 5
        
        return max(0, min(100, base_score + adjustment))
    
    def _determine_portfolio_risk(self, stock_analyses: List[StockAnalysis]) -> RiskLevel:
        """Determine overall portfolio risk level."""
        if not stock_analyses:
            return RiskLevel.MEDIUM
        
        risk_counts = {'low': 0, 'medium': 0, 'high': 0}
        for analysis in stock_analyses:
            risk_counts[analysis.risk_level] += 1
        
        total = len(stock_analyses)
        high_ratio = risk_counts['high'] / total
        low_ratio = risk_counts['low'] / total
        
        if high_ratio > 0.5:
            return RiskLevel.HIGH
        elif low_ratio > 0.5:
            return RiskLevel.LOW
        else:
            return RiskLevel.MEDIUM
    
    def _calculate_diversification_score(self, portfolio: Portfolio) -> float:
        """Calculate portfolio diversification score."""
        if not portfolio.items:
            return 0
        
        # Count sectors
        sectors = {}
        for item in portfolio.items:
            sector = self.STOCK_SECTORS.get(item.ticker, 'Unknown')
            sectors[sector] = sectors.get(sector, 0) + 1
        
        # Calculate diversification based on sector distribution
        num_sectors = len(sectors)
        max_concentration = max(sectors.values()) / len(portfolio.items)
        
        # Score based on number of sectors and concentration
        sector_score = min(100, num_sectors * 20)  # Up to 5 sectors for full points
        concentration_penalty = max_concentration * 50  # Penalty for concentration
        
        return max(0, min(100, sector_score - concentration_penalty))
    
    def _calculate_current_allocation(self, portfolio: Portfolio) -> AllocationBreakdown:
        """Calculate current portfolio allocation breakdown."""
        sectors = {}
        risk_levels = {'low': 0, 'medium': 0, 'high': 0}
        asset_types = {'stocks': 100}  # Assuming all stocks for now
        
        for item in portfolio.items:
            sector = self.STOCK_SECTORS.get(item.ticker, 'Unknown')
            risk_level = self.RISK_PROFILES.get(item.ticker, 'medium')
            
            sectors[sector] = sectors.get(sector, 0) + (item.allocation or 0)
            risk_levels[risk_level] += (item.allocation or 0)
        
        return AllocationBreakdown(
            sectors=sectors,
            asset_types=asset_types,
            risk_levels=risk_levels
        )
    
    def _generate_recommended_allocation(self, user_profile: UserProfileRequest) -> AllocationBreakdown:
        """Generate recommended allocation based on user profile."""
        if user_profile.risk_tolerance == 'conservative':
            risk_levels = {'low': 60, 'medium': 30, 'high': 10}
        elif user_profile.risk_tolerance == 'aggressive':
            risk_levels = {'low': 20, 'medium': 30, 'high': 50}
        else:  # moderate
            risk_levels = {'low': 30, 'medium': 50, 'high': 20}
        
        sectors = {
            'Technology': 25,
            'Healthcare': 20,
            'Financial Services': 15,
            'Consumer Staples': 15,
            'Consumer Discretionary': 10,
            'Energy': 8,
            'Communication Services': 7
        }
        
        asset_types = {'stocks': 100}  # Assuming all stocks for now
        
        return AllocationBreakdown(
            sectors=sectors,
            asset_types=asset_types,
            risk_levels=risk_levels
        )
    
    def _generate_recommendations(self, portfolio: Portfolio, user_profile: UserProfileRequest, stock_analyses: List[StockAnalysis]) -> List[str]:
        """Generate portfolio recommendations."""
        recommendations = []
        
        # Diversification recommendations
        sectors = {}
        for item in portfolio.items:
            sector = self.STOCK_SECTORS.get(item.ticker, 'Unknown')
            sectors[sector] = sectors.get(sector, 0) + 1
        
        if len(sectors) < 3:
            recommendations.append("Consider diversifying across more sectors to reduce concentration risk")
        
        # Risk alignment recommendations
        high_risk_count = sum(1 for analysis in stock_analyses if analysis.risk_level == RiskLevel.HIGH)
        if user_profile.risk_tolerance == 'conservative' and high_risk_count > len(stock_analyses) * 0.3:
            recommendations.append("Consider reducing exposure to high-risk stocks to match your conservative risk tolerance")
        
        # Performance recommendations
        sell_count = sum(1 for analysis in stock_analyses if analysis.recommendation == Recommendation.SELL)
        if sell_count > 0:
            recommendations.append(f"Consider reviewing {sell_count} underperforming position(s) for potential rebalancing")
        
        # Generic recommendations
        recommendations.extend([
            "Regularly review and rebalance your portfolio quarterly",
            "Consider dollar-cost averaging for new investments",
            "Monitor expense ratios and fees to maximize returns"
        ])
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def _generate_red_flags(self, portfolio: Portfolio, stock_analyses: List[StockAnalysis]) -> List[str]:
        """Generate red flags and warnings."""
        red_flags = []
        
        # High concentration risk
        for item in portfolio.items:
            if item.allocation and item.allocation > 25:
                red_flags.append(f"High concentration in {item.ticker} ({item.allocation:.1f}% of portfolio)")
        
        # Poor performing stocks
        poor_performers = [analysis for analysis in stock_analyses if analysis.score < 40]
        if poor_performers:
            tickers = [analysis.ticker for analysis in poor_performers]
            red_flags.append(f"Poor performance indicators for: {', '.join(tickers)}")
        
        # Risk misalignment
        high_risk_stocks = [analysis for analysis in stock_analyses if analysis.risk_level == RiskLevel.HIGH]
        if len(high_risk_stocks) > len(stock_analyses) * 0.6:
            red_flags.append("Portfolio has high overall risk - consider adding more stable investments")
        
        return red_flags
    
    def _generate_summary(self, overall_score: float, risk_level: RiskLevel, diversification_score: float) -> str:
        """Generate portfolio analysis summary."""
        score_desc = "excellent" if overall_score >= 80 else "good" if overall_score >= 60 else "needs improvement"
        risk_desc = f"{risk_level.value}-risk"
        div_desc = "well-diversified" if diversification_score >= 70 else "moderately diversified" if diversification_score >= 40 else "poorly diversified"
        
        return f"Your portfolio shows {score_desc} overall performance with a {risk_desc} profile and is {div_desc}. " \
               f"The portfolio score of {overall_score:.1f} reflects the current market positioning and alignment with your investment goals." 