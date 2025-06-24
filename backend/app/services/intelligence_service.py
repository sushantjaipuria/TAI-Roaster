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
    from intelligence.pipeline import run_full_pipeline, generate_complete_tai_roast_analysis
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
            
            # Convert to UserInput format expected by portfolio analysis function
            user_input = self._create_user_input(portfolio, user_profile)
            
            # Use the new portfolio holdings analysis function instead of stock selection pipeline
            portfolio_analysis_result = await generate_complete_tai_roast_analysis(
                holdings=portfolio.holdings,
                user_input=user_input.dict(),
                report_id=f"portfolio_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Convert portfolio analysis result to enhanced analysis format
            enhanced_results = self._convert_portfolio_analysis_result(portfolio_analysis_result, portfolio, user_profile)
            
            logger.info("✅ Enhanced analysis completed successfully")
            return enhanced_results
            
        except Exception as e:
            logger.error(f"❌ Enhanced analysis failed: {e}")
            # Return fallback analysis or re-raise based on configuration
            raise
    
    def _create_user_input(self, portfolio: PortfolioInput, user_profile: UserProfile):
        """Create UserInput object expected by the pipeline"""
        
        # Create our own UserInput class that accepts dynamic attributes
        # The pipeline imports AnalysisRequest as UserInput, but we need the fallback behavior
        class UserInput:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
            
            def dict(self):
                """Provide dict() method expected by pipeline"""
                return {k: v for k, v in self.__dict__.items()}
        
        # Extract values from nested structure and create mappings
        
        # Direct mappings from user_profile
        amount = user_profile.investment_amount
        risk_tolerance = user_profile.risk_tolerance.value if hasattr(user_profile.risk_tolerance, 'value') else user_profile.risk_tolerance
        
        # Map time_horizon to horizon_months
        time_horizon_mapping = {
            "short": 12,    # < 3 years -> 12 months
            "medium": 60,   # 3-10 years -> 60 months  
            "long": 120     # > 10 years -> 120 months
        }
        time_horizon_value = user_profile.time_horizon.value if hasattr(user_profile.time_horizon, 'value') else user_profile.time_horizon
        horizon_months = time_horizon_mapping.get(time_horizon_value, 60)  # Default to medium term
        
        # Set return_target_pct based on risk tolerance
        return_target_mapping = {
            "conservative": 8.0,   # Conservative: 8% annual return target
            "moderate": 12.0,      # Moderate: 12% annual return target
            "aggressive": 18.0     # Aggressive: 18% annual return target
        }
        return_target_pct = return_target_mapping.get(risk_tolerance, 12.0)  # Default to moderate
        
        # Default market_cap (pipeline expects this but it's not in user_profile)
        market_cap = "large_cap"  # Conservative default
        
        # Create UserInput object with all attributes the pipeline expects
        user_input = UserInput(
            # Core attributes that pipeline accesses directly
            amount=amount,
            risk_tolerance=risk_tolerance,
            market_cap=market_cap,
            horizon_months=horizon_months,
            return_target_pct=return_target_pct,
            
            # Preserve original nested structure for compatibility
            portfolio=portfolio,
            user_profile=user_profile,
            
            # Additional metadata
            analysis_type="comprehensive",
            include_recommendations=True
        )
        
        # Log the transformation for debugging
        logger.info(f"💡 Created UserInput with: amount=₹{amount:,}, risk_tolerance={risk_tolerance}, "
                   f"horizon_months={horizon_months}, return_target_pct={return_target_pct}%, market_cap={market_cap}")
        
        return user_input
    
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

    def _convert_portfolio_analysis_result(
        self, 
        portfolio_analysis_result: Dict[str, Any], 
        original_portfolio: PortfolioInput, 
        user_profile: UserProfile
    ) -> Dict[str, Any]:
        """Convert portfolio analysis result to enhanced analysis format"""
        
        # Extract data from the new portfolio analysis result
        portfolio_metrics = portfolio_analysis_result.get("portfolio_metrics", {})
        stock_analyses = portfolio_analysis_result.get("stock_analysis", [])
        ml_predictions = portfolio_analysis_result.get("ml_predictions", [])
        
        # Calculate basic portfolio metrics from analysis result
        total_portfolio_value = portfolio_metrics.get("total_portfolio_value", 0)
        total_invested = sum(holding.quantity * holding.avg_buy_price for holding in original_portfolio.holdings)
        
        absolute_return = total_portfolio_value - total_invested if total_portfolio_value > 0 else 0
        absolute_return_pct = (absolute_return / total_invested * 100) if total_invested > 0 else 0
        
        # Calculate TAI scores based on actual analysis
        portfolio_expected_return = portfolio_metrics.get("portfolio_expected_return", 0.08)
        average_confidence = portfolio_metrics.get("average_confidence", 0.75)
        analysis_coverage = portfolio_metrics.get("analysis_coverage", 1.0)
        
        # Convert expected return and confidence to scores (0-100)
        performance_score = min(100, max(0, (portfolio_expected_return + 0.2) * 400))  # Scale to 0-100
        ml_confidence_score = average_confidence * 100
        diversification_score = min(100, len(stock_analyses) * 10)  # 10 points per stock, max 100
        risk_score = 100 - (abs(portfolio_expected_return - 0.12) * 500)  # Penalty for deviation from 12% target
        
        overall_score = (performance_score + ml_confidence_score + diversification_score + risk_score) / 4
        
        # Determine grade based on score
        if overall_score >= 90:
            grade = "A+"
        elif overall_score >= 80:
            grade = "A"
        elif overall_score >= 70:
            grade = "B+"
        elif overall_score >= 60:
            grade = "B"
        else:
            grade = "C"
        
        # Build enhanced response using actual analysis data
        enhanced_response = {
            # Basic portfolio metrics
            "total_invested": total_invested,
            "current_value": total_portfolio_value,
            "absolute_return": absolute_return,
            "absolute_return_pct": absolute_return_pct,
            
            # TAI Score components based on actual analysis
            "tai_scores": {
                "overall_score": round(overall_score, 1),
                "performance_score": round(performance_score, 1),
                "risk_management_score": round(risk_score, 1),
                "diversification_score": round(diversification_score, 1),
                "ml_confidence_score": round(ml_confidence_score, 1),
                "liquidity_score": 80.0,  # Default for now
                "cost_efficiency_score": 75.0,  # Default for now
                "grade": grade,
                "description": f"Portfolio analysis with {analysis_coverage:.1%} coverage"
            },
            
            # ML predictions from actual analysis
            "ml_predictions": self._extract_ml_predictions_from_analysis(ml_predictions),
            
            # Portfolio allocation analysis
            "allocation": self._build_allocation_analysis(original_portfolio.holdings),
            
            # Enhanced stock analysis from actual holdings analysis
            "enhanced_stocks": self._build_enhanced_stock_analysis_from_results(stock_analyses),
            
            # Overall metrics
            "overall_score": round(overall_score, 1),
            "risk_level": "Medium" if abs(portfolio_expected_return - 0.12) < 0.05 else "High" if portfolio_expected_return > 0.17 else "Low",
            "rating": {"grade": grade, "score": round(overall_score, 1), "description": f"Analysis based on actual portfolio holdings"},
            
            # Action plan based on recommendations
            "action_plan": self._build_action_plan_from_analysis(portfolio_analysis_result),
            
            # Recommendations from analysis
            "recommendations": self._extract_recommendations_from_analysis(portfolio_analysis_result),
            
            # Analysis metadata
            "analysis_date": datetime.now().isoformat(),
            "portfolio_name": getattr(original_portfolio, 'name', 'Portfolio'),
            "model_version": "2.0.0",
            "analysis_type": "portfolio_holdings_analysis",
            "total_holdings_analyzed": len(stock_analyses),
            "successful_analyses": portfolio_metrics.get("successful_analyses", 0)
        }
        
        return enhanced_response

    def _extract_ml_predictions_from_analysis(self, ml_predictions) -> List[Dict[str, Any]]:
        """Extract ML predictions from portfolio analysis result"""
        predictions = []
        for pred in ml_predictions:
            ticker = pred.get("ticker")
            predictions_data = pred.get("predictions", {})
            
            # Extract predictions from the ML model outputs
            xgboost_pred = 0.08  # Default
            ngboost_mean = 0.07  # Default
            ngboost_std = 0.05   # Default
            ensemble_pred = 0.08 # Default
            ensemble_conf = 0.75 # Default
            
            if predictions_data:
                # Extract from various models if available
                if "xgboost" in predictions_data:
                    xgboost_pred = predictions_data["xgboost"].get("return", 0.08)
                if "ngboost" in predictions_data:
                    ngboost_mean = predictions_data["ngboost"].get("return", 0.07)
                    ngboost_std = predictions_data["ngboost"].get("std", 0.05)
                
                # Calculate ensemble prediction
                all_returns = [model_data.get("return", 0.08) for model_data in predictions_data.values()]
                ensemble_pred = sum(all_returns) / len(all_returns) if all_returns else 0.08
                
                # Calculate ensemble confidence
                all_confidences = [model_data.get("confidence", 0.75) for model_data in predictions_data.values()]
                ensemble_conf = sum(all_confidences) / len(all_confidences) if all_confidences else 0.75
            
            predictions.append({
                "ticker": ticker,
                "ensemble_prediction": ensemble_pred,
                "ensemble_confidence": ensemble_conf,
                "xgboost_prediction": xgboost_pred,
                "ngboost_mean": ngboost_mean,
                "ngboost_std": ngboost_std
            })
        
        return predictions

    def _build_enhanced_stock_analysis_from_results(self, stock_analyses) -> List[Dict[str, Any]]:
        """Build enhanced stock analysis from portfolio analysis results"""
        enhanced_stocks = []
        
        total_value = sum(stock.get("holding_details", {}).get("current_value", 0) for stock in stock_analyses)
        
        for stock in stock_analyses:
            holding_details = stock.get("holding_details", {})
            
            enhanced_stock = {
                'ticker': stock.get("ticker"),
                'quantity': holding_details.get("quantity", 0),
                'current_price': holding_details.get("current_price", 0),
                'investment_amount': holding_details.get("total_investment", 0),
                'current_value': holding_details.get("current_value", 0),
                'weight': (holding_details.get("current_value", 0) / total_value) * 100 if total_value > 0 else 0,
                'ml_prediction': stock.get("expected_return", 0.08),
                'confidence_score': stock.get("confidence_score", 0.75),
                'recommendation': stock.get("recommendation", "HOLD"),
                'unrealized_pnl': holding_details.get("unrealized_pnl", 0)
            }
            enhanced_stocks.append(enhanced_stock)
        
        return enhanced_stocks

    def _build_action_plan_from_analysis(self, portfolio_analysis_result) -> Dict[str, List[str]]:
        """Build action plan from portfolio analysis recommendations"""
        recommendations = portfolio_analysis_result.get("recommendations", [])
        
        immediate_actions = []
        short_term_goals = []
        long_term_strategy = []
        
        # Extract recommendations and categorize
        for rec in recommendations:
            if isinstance(rec, dict):
                if rec.get("buy_recommendations", 0) > 0:
                    immediate_actions.append("Consider buying recommended stocks")
                if rec.get("sell_recommendations", 0) > 0:
                    immediate_actions.append("Review sell recommendations")
                if rec.get("overall_recommendation") == "REBALANCE":
                    short_term_goals.append("Rebalance portfolio allocation")
        
        # Default actions if none found
        if not immediate_actions:
            immediate_actions = ["Review portfolio performance", "Monitor market conditions"]
        if not short_term_goals:
            short_term_goals = ["Diversify holdings", "Monitor risk metrics"]
        if not long_term_strategy:
            long_term_strategy = ["Build systematic investment approach", "Review and rebalance quarterly"]
        
        return {
            "immediate_actions": immediate_actions,
            "short_term_goals": short_term_goals,
            "long_term_strategy": long_term_strategy
        }

    def _extract_recommendations_from_analysis(self, portfolio_analysis_result) -> List[str]:
        """Extract text recommendations from portfolio analysis result"""
        recommendations = []
        stock_analyses = portfolio_analysis_result.get("stock_analysis", [])
        
        # Generate recommendations based on analysis
        buy_stocks = [stock for stock in stock_analyses if stock.get("recommendation") in ["BUY", "STRONG_BUY"]]
        sell_stocks = [stock for stock in stock_analyses if stock.get("recommendation") in ["SELL", "STRONG_SELL"]]
        
        for stock in buy_stocks[:3]:  # Top 3 buy recommendations
            recommendations.append(f"Consider accumulating {stock.get('ticker')} - Strong buy signal")
        
        for stock in sell_stocks[:2]:  # Top 2 sell recommendations  
            recommendations.append(f"Consider reducing position in {stock.get('ticker')} - Sell signal")
        
        # Add general recommendations
        portfolio_metrics = portfolio_analysis_result.get("portfolio_metrics", {})
        if portfolio_metrics.get("analysis_coverage", 0) < 1.0:
            recommendations.append("Some holdings could not be analyzed - consider data availability")
        
        return recommendations[:5]  # Limit to 5 recommendations

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