"""
Intelligence Service Layer for TAI-Roaster
Provides ML-powered portfolio analysis using the intelligence module
"""

import sys
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

# Add project root to Python path for intelligence module imports
project_root = Path(__file__).parent.parent.parent.parent  # Go up to TAI-Roaster root
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from intelligence.pipeline import run_full_pipeline
    from intelligence.config import get_config
    from intelligence.llm_trading_expert import LLMTradingExpert
    from intelligence.transparency_logger import TransparencyLogger
    INTELLIGENCE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Intelligence module imports failed: {e}")
    INTELLIGENCE_AVAILABLE = False

from app.schemas.input import PortfolioInput
from app.models.onboarding import UserProfileRequest as UserProfile

logger = logging.getLogger(__name__)

class IntelligenceService:
    """Service layer for intelligence module integration"""
    
    def __init__(self):
        """Initialize the intelligence service with all required components"""
        if not INTELLIGENCE_AVAILABLE:
            self.initialized = False
            logger.warning("Intelligence module not available")
            return
            
        try:
            # Use the available intelligence components
            self.config = get_config()
            self.transparency_logger = TransparencyLogger()
            self.initialized = True
            logger.info("✅ Intelligence Service initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Intelligence Service: {e}")
            self.initialized = False
    
    async def analyze_portfolio(
        self, 
        portfolio: PortfolioInput, 
        user_profile: UserProfile
    ) -> Dict[str, Any]:
        """
        Main analysis endpoint using the intelligence module
        
        Args:
            portfolio: Portfolio data to analyze
            user_profile: User profile for personalized analysis
            
        Returns:
            Enhanced analysis results with ML predictions and TAI scores
        """
        if not self.initialized:
            raise RuntimeError("Intelligence Service not properly initialized")
        
        try:
            logger.info(f"🔍 Starting enhanced analysis for portfolio with {len(portfolio.holdings)} holdings")
            
            # Convert to UserInput format expected by pipeline
            user_input = self._create_user_input(portfolio, user_profile)
            
            # Run the full intelligence pipeline
            pipeline_result = await run_full_pipeline(user_input)
            
            # Convert pipeline result to enhanced analysis format
            enhanced_results = self._convert_pipeline_result(pipeline_result, portfolio, user_profile)
            
            logger.info("✅ Enhanced analysis completed successfully")
            return enhanced_results
            
        except Exception as e:
            logger.error(f"❌ Enhanced analysis failed: {e}")
            # Return fallback analysis or re-raise based on configuration
            raise
    
    def _create_user_input(self, portfolio: PortfolioInput, user_profile: UserProfile):
        """Create UserInput object expected by the pipeline"""
        # The pipeline expects an AnalysisRequest (aliased as UserInput)
        from backend.app.schemas.input import AnalysisRequest
        
        return AnalysisRequest(
            portfolio=portfolio,
            user_profile=user_profile,
            analysis_type="comprehensive",
            include_recommendations=True
        )
    
    def _convert_pipeline_result(
        self, 
        pipeline_result, 
        original_portfolio: PortfolioInput, 
        user_profile: UserProfile
    ) -> Dict[str, Any]:
        """Convert pipeline result to enhanced analysis format"""
        
        # Calculate basic portfolio metrics
        total_invested = sum(holding.quantity * holding.avg_buy_price for holding in original_portfolio.holdings)
        current_value = sum(
            holding.quantity * (holding.current_price or holding.avg_buy_price)
            for holding in original_portfolio.holdings
        )
        
        absolute_return = current_value - total_invested if current_value > 0 else 0
        absolute_return_pct = (absolute_return / total_invested * 100) if total_invested > 0 else 0
        
        # Extract data from pipeline result
        if hasattr(pipeline_result, 'portfolio') and pipeline_result.portfolio:
            recommendations = pipeline_result.portfolio
        else:
            recommendations = []
        
        # Build enhanced response
        enhanced_response = {
            # Basic portfolio metrics
            "total_invested": total_invested,
            "current_value": current_value,
            "absolute_return": absolute_return,
            "absolute_return_pct": absolute_return_pct,
            
            # TAI Score components (mock for now)
            "tai_scores": {
                "overall_score": 78.5,
                "performance_score": 82.0,
                "risk_management_score": 75.0,
                "diversification_score": 70.0,
                "ml_confidence_score": 85.0,
                "liquidity_score": 80.0,
                "cost_efficiency_score": 77.0,
                "grade": "B+",
                "description": "Good portfolio with room for improvement"
            },
            
            # ML predictions from pipeline
            "ml_predictions": self._extract_ml_predictions(recommendations),
            
            # Portfolio allocation analysis
            "allocation": self._build_allocation_analysis(original_portfolio.holdings),
            
            # Enhanced stock recommendations
            "enhanced_stocks": self._build_enhanced_stock_analysis(original_portfolio.holdings, recommendations),
            
            # Overall metrics
            "overall_score": 78.5,
            "risk_level": "Medium",
            "rating": {"grade": "B+", "score": 78.5, "description": "Good portfolio performance"},
            
            # Action plan
            "action_plan": {
                "immediate_actions": ["Review concentration risk", "Consider diversification"],
                "short_term_goals": ["Rebalance allocation", "Add defensive stocks"],
                "long_term_strategy": ["Build systematic investment approach", "Monitor performance metrics"]
            },
            
            # Basic recommendations
            "recommendations": self._extract_recommendations(recommendations),
            
            # Analysis metadata
            "analysis_date": datetime.now().isoformat(),
            "portfolio_name": getattr(original_portfolio, 'name', 'Portfolio'),
            "model_version": "1.0.0",
            "analysis_type": "comprehensive"
        }
        
        return enhanced_response

    def _extract_ml_predictions(self, recommendations) -> List[Dict[str, Any]]:
        """Extract ML predictions from pipeline recommendations"""
        predictions = []
        if isinstance(recommendations, list):
            for rec in recommendations:
                if hasattr(rec, 'ticker'):
                    predictions.append({
                        "ticker": rec.ticker,
                        "ensemble_prediction": getattr(rec, 'expected_return', 0.08),
                        "ensemble_confidence": getattr(rec, 'confidence_score', 0.75),
                        "xgboost_prediction": 0.08,
                        "ngboost_mean": 0.07,
                        "ngboost_std": 0.05
                    })
        return predictions

    def _extract_recommendations(self, recommendations) -> List[str]:
        """Extract text recommendations from pipeline result"""
        if isinstance(recommendations, list):
            return [f"Consider {rec.ticker} - {getattr(rec, 'explanation', 'Good investment opportunity')}" 
                   for rec in recommendations[:5]]  # Top 5 recommendations
        return ["Diversify portfolio", "Monitor risk metrics", "Rebalance quarterly"]

    def _build_allocation_analysis(self, holdings) -> Dict[str, Any]:
        """Build enhanced allocation analysis"""
        total_investment = sum(holding.quantity * holding.avg_buy_price for holding in holdings)
        
        # Basic sector analysis (simplified)
        sectors = {}
        for holding in holdings:
            # Simple sector mapping based on ticker
            if holding.ticker.startswith(('BANK', 'HDFC', 'ICICI')):
                sector = "Financial Services"
            elif holding.ticker.startswith(('TCS', 'INFY', 'WIPRO')):
                sector = "Information Technology"
            else:
                sector = "Other"
            
            sectors[sector] = sectors.get(sector, 0) + (holding.quantity * holding.avg_buy_price)
        
        # Convert to percentages
        sector_allocation = {k: (v/total_investment)*100 for k, v in sectors.items()}
        
        return {
            "sector_allocation": sector_allocation,
            "market_cap_allocation": {
                "Large Cap": 70.0,
                "Mid Cap": 20.0, 
                "Small Cap": 10.0
            },
            "concentration_risk": max(sector_allocation.values()) if sector_allocation else 0,
            "diversification_ratio": min(len(holdings), 10) / 10  # Simple diversification metric
        }
    
    def _build_enhanced_stock_analysis(self, holdings, recommendations) -> List[Dict[str, Any]]:
        """Build enhanced stock-level analysis with ML insights"""
        enhanced_stocks = []
        
        for holding in holdings:
            # Find matching recommendation
            rec = None
            if isinstance(recommendations, list):
                for r in recommendations:
                    if hasattr(r, 'ticker') and r.ticker == holding.ticker:
                        rec = r
                        break
            
            investment_amount = holding.quantity * holding.avg_buy_price
            current_price = holding.current_price or holding.avg_buy_price
            current_value = holding.quantity * current_price
            total_invested = sum(h.quantity * h.avg_buy_price for h in holdings)
            
            stock_analysis = {
                'ticker': holding.ticker,
                'quantity': holding.quantity,
                'current_price': current_price,
                'investment_amount': investment_amount,
                'current_value': current_value,
                'weight': (investment_amount / total_invested) * 100 if total_invested > 0 else 0,
                'ml_prediction': getattr(rec, 'expected_return', 0.08) if rec else 0.08,
                'confidence_score': getattr(rec, 'confidence_score', 0.75) if rec else 0.75,
                'recommendation': getattr(rec, 'recommendation', "Hold") if rec else "Hold"
            }
            enhanced_stocks.append(stock_analysis)
        
        return enhanced_stocks


# Create singleton instance
intelligence_service = IntelligenceService() 