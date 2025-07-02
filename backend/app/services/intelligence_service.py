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
import pandas as pd
import yfinance as yf
import numpy as np

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
from .enhanced_stock_analyzer import enhanced_stock_analyzer

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
                'unrealized_pnl': holding_details.get("unrealized_pnl", 0),
                
                # Add enhanced analysis placeholder - will be populated in enhanced mode
                'enhanced_analysis_available': True,
                'fundamental_score': 75.0,  # Placeholder
                'technical_score': 70.0,    # Placeholder
                'overall_score': 72.5,      # Placeholder
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

    async def get_enhanced_portfolio_analysis(
        self, 
        portfolio_input: PortfolioInput, 
        user_profile: UserProfile
    ) -> Dict[str, Any]:
        """
        Get enhanced portfolio analysis with detailed individual stock analysis
        """
        try:
            logger.info("🚀 Starting enhanced portfolio analysis with detailed stock insights")
            
            # First get the standard portfolio analysis
            standard_analysis = await self.analyze_portfolio(portfolio_input, user_profile)
            
            # Extract holdings for enhanced analysis
            holdings = portfolio_input.holdings
            
            # Perform enhanced analysis for each stock
            logger.info(f"📈 Performing enhanced analysis for {len(holdings)} stocks")
            
            enhanced_stock_tasks = []
            for holding in holdings:
                task = enhanced_stock_analyzer.analyze_stock(
                    ticker=holding.ticker,
                    quantity=holding.quantity,
                    avg_price=holding.avg_buy_price
                )
                enhanced_stock_tasks.append(task)
            
            # Execute enhanced analyses concurrently
            enhanced_results = await asyncio.gather(*enhanced_stock_tasks, return_exceptions=True)
            
            # Process enhanced results
            enhanced_stocks = []
            for i, result in enumerate(enhanced_results):
                holding = holdings[i]
                
                if isinstance(result, Exception):
                    logger.warning(f"Enhanced analysis failed for {holding.ticker}: {result}")
                    # Create fallback enhanced stock data
                    enhanced_stock = self._create_fallback_enhanced_stock(holding, standard_analysis)
                else:
                    # Convert enhanced analysis to portfolio context
                    enhanced_stock = self._convert_enhanced_analysis_to_portfolio_context(
                        result, holding, standard_analysis
                    )
                
                enhanced_stocks.append(enhanced_stock)
            
            # Calculate REAL performance metrics instead of synthetic ones
            real_performance_metrics = await self.calculate_real_performance_metrics(holdings)
            
            # Enhance the standard analysis with detailed stock insights
            enhanced_analysis = standard_analysis.copy()
            enhanced_analysis['enhanced_stocks'] = enhanced_stocks
            enhanced_analysis['enhanced_analysis_metadata'] = {
                'total_stocks_analyzed': len(holdings),
                'successful_enhanced_analyses': len([r for r in enhanced_results if not isinstance(r, Exception)]),
                'analysis_type': 'enhanced_portfolio_with_detailed_stocks',
                'timestamp': datetime.now().isoformat()
            }
            
            # Calculate portfolio-level enhanced metrics
            enhanced_analysis['portfolio_enhanced_metrics'] = self._calculate_portfolio_enhanced_metrics(enhanced_stocks)
            
            # Replace synthetic performance metrics with real calculations
            if real_performance_metrics:
                enhanced_analysis['performanceMetrics'] = real_performance_metrics
                enhanced_analysis['benchmarkName'] = 'NIFTY 50'
                logger.info(f"✅ Added real performance metrics for {len(real_performance_metrics)} timeframes")
            else:
                logger.warning("⚠️ Could not calculate real performance metrics, keeping existing data")
            
            logger.info("✅ Enhanced portfolio analysis completed successfully")
            return enhanced_analysis
            
        except Exception as e:
            logger.error(f"❌ Enhanced portfolio analysis failed: {e}")
            # Fallback to standard analysis
            return await self.analyze_portfolio(portfolio_input, user_profile)
    
    def _convert_enhanced_analysis_to_portfolio_context(
        self, 
        enhanced_insight, 
        holding, 
        standard_analysis
    ) -> Dict[str, Any]:
        """Convert enhanced stock insight to portfolio context"""
        try:
            current_value = holding.quantity * (holding.current_price or holding.avg_buy_price)
            investment_amount = holding.quantity * holding.avg_buy_price
            unrealized_pnl = current_value - investment_amount
            
            # Calculate portfolio weight safely
            portfolio_total_value = sum(h.quantity * (h.current_price or h.avg_buy_price) for h in standard_analysis.get('stocks', []))
            if portfolio_total_value == 0:
                portfolio_weight = 0.0
            else:
                portfolio_weight = (current_value / portfolio_total_value) * 100
            
            return {
                # Basic holding information
                'ticker': enhanced_insight.ticker,
                'company_name': enhanced_insight.company_name,
                'sector': enhanced_insight.sector,
                'quantity': holding.quantity,
                'avg_buy_price': holding.avg_buy_price,
                'current_price': enhanced_insight.current_price,
                'investment_amount': investment_amount,
                'current_value': current_value,
                'unrealized_pnl': unrealized_pnl,
                'weight': portfolio_weight,
                
                # Enhanced multi-factor scores
                'fundamental_score': enhanced_insight.fundamental_score,
                'technical_score': enhanced_insight.technical_score,
                'momentum_score': enhanced_insight.momentum_score,
                'value_score': enhanced_insight.value_score,
                'quality_score': enhanced_insight.quality_score,
                'sentiment_score': enhanced_insight.sentiment_score,
                'overall_score': enhanced_insight.overall_score,
                
                # Risk and projections
                'beta': enhanced_insight.beta,
                'volatility': enhanced_insight.volatility,
                'max_drawdown': enhanced_insight.max_drawdown,
                'target_price': enhanced_insight.target_price,
                'upside_potential': enhanced_insight.upside_potential,
                'confidence_level': enhanced_insight.confidence_level,
                
                # Insights and recommendations
                'recommendation': enhanced_insight.recommendation,
                'key_strengths': enhanced_insight.key_strengths,
                'key_concerns': enhanced_insight.key_concerns,
                'catalysts': enhanced_insight.catalysts,
                'risks': enhanced_insight.risks,
                
                # Detailed analysis
                'technical_analysis': enhanced_insight.technical_analysis.to_dict() if hasattr(enhanced_insight.technical_analysis, 'to_dict') else enhanced_insight.technical_analysis.__dict__,
                'fundamental_analysis': enhanced_insight.fundamental_analysis.to_dict() if hasattr(enhanced_insight.fundamental_analysis, 'to_dict') else enhanced_insight.fundamental_analysis.__dict__,
                
                # AI commentary
                'business_story': enhanced_insight.business_story,
                'investment_thesis': enhanced_insight.investment_thesis,
                
                # Comparative analysis
                'sector_comparison': enhanced_insight.sector_comparison,
                
                # Enhanced analysis metadata
                'enhanced_analysis_available': True,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to convert enhanced analysis for {holding.ticker}: {e}")
            return self._create_fallback_enhanced_stock(holding, standard_analysis)
    
    def _create_fallback_enhanced_stock(self, holding, standard_analysis) -> Dict[str, Any]:
        """Create fallback enhanced stock data when detailed analysis fails"""
        current_value = holding.quantity * (holding.current_price or holding.avg_buy_price)
        investment_amount = holding.quantity * holding.avg_buy_price
        
        return {
            'ticker': holding.ticker,
            'company_name': holding.ticker,
            'sector': 'Unknown',
            'quantity': holding.quantity,
            'avg_buy_price': holding.avg_buy_price,
            'current_price': holding.current_price or holding.avg_buy_price,
            'investment_amount': investment_amount,
            'current_value': current_value,
            'unrealized_pnl': current_value - investment_amount,
            'weight': 10.0,  # Placeholder
            
            # Default scores
            'fundamental_score': 50.0,
            'technical_score': 50.0,
            'momentum_score': 50.0,
            'value_score': 50.0,
            'quality_score': 50.0,
            'sentiment_score': 50.0,
            'overall_score': 50.0,
            
            'recommendation': 'HOLD',
            'key_strengths': ['Analysis pending'],
            'key_concerns': ['Limited data available'],
            'enhanced_analysis_available': False,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _calculate_portfolio_enhanced_metrics(self, enhanced_stocks) -> Dict[str, Any]:
        """Calculate portfolio-level metrics from enhanced stock analyses"""
        try:
            # Filter stocks with successful analysis
            valid_stocks = [s for s in enhanced_stocks if s.get('enhanced_analysis_available', False)]
            
            if not valid_stocks:
                return {}
            
            # Calculate weighted averages
            total_value = sum(s['current_value'] for s in valid_stocks)
            
            # Handle division by zero
            if total_value == 0:
                logger.warning("Total portfolio value is zero, using equal weights for scoring")
                weighted_fundamental_score = sum(s['fundamental_score'] for s in valid_stocks) / len(valid_stocks)
                weighted_technical_score = sum(s['technical_score'] for s in valid_stocks) / len(valid_stocks)
                weighted_overall_score = sum(s['overall_score'] for s in valid_stocks) / len(valid_stocks)
            else:
                weighted_fundamental_score = sum(s['fundamental_score'] * s['current_value'] for s in valid_stocks) / total_value
                weighted_technical_score = sum(s['technical_score'] * s['current_value'] for s in valid_stocks) / total_value
                weighted_overall_score = sum(s['overall_score'] * s['current_value'] for s in valid_stocks) / total_value
            
            return {
                'enhanced_coverage': len(valid_stocks) / len(enhanced_stocks) if enhanced_stocks else 0,
                'portfolio_fundamental_score': weighted_fundamental_score,
                'portfolio_technical_score': weighted_technical_score,
                'portfolio_overall_score': weighted_overall_score,
                'total_analyzed_value': total_value,
                'analysis_depth': 'detailed' if len(valid_stocks) > 0 else 'basic'
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate portfolio enhanced metrics: {e}")
            return {}

    def _get_price_column(self, dataframe: pd.DataFrame, column_name: str, ticker: str) -> tuple[bool, Any]:
        """
        Helper function to get the appropriate price column from yfinance data,
        handling both simple and MultiIndex column structures.
        
        Args:
            dataframe: The yfinance DataFrame
            column_name: 'Close' (price column name)
            ticker: The ticker symbol for MultiIndex columns
        
        Returns:
            tuple: (found, column_reference)
            - found: Boolean indicating if column was found
            - column_reference: The actual column key to use for data access
        """
        if dataframe.empty:
            return False, None
            
        # Check if columns are MultiIndex (tuples) by examining first column
        if len(dataframe.columns) > 0 and isinstance(dataframe.columns[0], tuple):
            # MultiIndex columns: look for (column_name, ticker)
            target_column = (column_name, ticker)
            if target_column in dataframe.columns:
                return True, target_column
            else:
                return False, None
        else:
            # Simple columns: look for column_name directly
            if column_name in dataframe.columns:
                return True, column_name
            else:
                return False, None

    async def calculate_real_performance_metrics(
        self, 
        holdings: List[Any], 
        timeframes: List[str] = ['1M', '3M', '1Y']
    ) -> List[Dict[str, Any]]:
        """
        Calculate REAL performance metrics using historical data
        No random generation - only actual calculations
        """
        try:
            from datetime import datetime, timedelta
            import yfinance as yf
            import numpy as np
            import pandas as pd
            
            logger.info(f"🔢 Calculating real performance metrics for {len(holdings)} holdings")
            
            performance_data = []
            
            for timeframe in timeframes:
                # Calculate date range for timeframe
                end_date = datetime.now()
                if timeframe == '1M':
                    start_date = end_date - timedelta(days=30)
                    period_days = 30
                elif timeframe == '3M':
                    start_date = end_date - timedelta(days=90)
                    period_days = 90
                elif timeframe == '1Y':
                    start_date = end_date - timedelta(days=365)
                    period_days = 365
                else:
                    continue
                
                try:
                    # Get historical data for all holdings
                    portfolio_returns = []
                    portfolio_weights = []
                    benchmark_data = None
                    
                    # Calculate total portfolio value for weights
                    total_portfolio_value = sum(
                        holding.quantity * (holding.current_price or holding.avg_buy_price) 
                        for holding in holdings
                    )
                    
                    # Get historical data for each holding
                    for holding in holdings:
                        try:
                            # Fetch historical data - ensure .NS suffix for yFinance
                            yf_ticker = holding.ticker if holding.ticker.endswith('.NS') else f"{holding.ticker}.NS"
                            ticker_data = yf.download(
                                yf_ticker, 
                                start=start_date, 
                                end=end_date,
                                progress=False
                            )
                            
                            if not ticker_data.empty and len(ticker_data) > 1:
                                # Log columns for debugging
                                logger.info(f"Columns for {holding.ticker}: {ticker_data.columns.tolist()}")
                                
                                # Use helper function to find Close
                                close_found, close_col = self._get_price_column(ticker_data, 'Close', yf_ticker)
                                
                                if close_found:
                                    stock_returns = ticker_data[close_col].pct_change().dropna()
                                else:
                                    logger.error(f"'Close' column not found for {holding.ticker}.")
                                    continue
                                
                                # Calculate weight in portfolio
                                holding_value = holding.quantity * (holding.current_price or holding.avg_buy_price)
                                weight = holding_value / total_portfolio_value if total_portfolio_value > 0 else 0
                                
                                portfolio_returns.append(stock_returns)
                                portfolio_weights.append(weight)
                                
                                logger.debug(f"✅ Got {len(stock_returns)} returns for {holding.ticker}")
                            else:
                                logger.warning(f"⚠️ No data for {holding.ticker}")
                                
                        except Exception as e:
                            logger.warning(f"Failed to get data for {holding.ticker}: {e}")
                            continue
                    
                    # Get benchmark data (NIFTY 50)
                    try:
                        benchmark_data = yf.download("^NSEI", start=start_date, end=end_date, progress=False)
                        if not benchmark_data.empty:
                            logger.info(f"Benchmark columns: {benchmark_data.columns.tolist()}")
                            
                            # Use helper function to find Close for benchmark
                            close_found, close_col = self._get_price_column(benchmark_data, 'Close', "^NSEI")
                            
                            if close_found:
                                benchmark_returns = benchmark_data[close_col].pct_change().dropna()
                            else:
                                logger.error("'Close' column not found for benchmark.")
                                benchmark_returns = None
                    except Exception as e:
                        logger.warning(f"Failed to get benchmark data: {e}")
                        benchmark_returns = None
                    
                    # Calculate portfolio metrics if we have data
                    if portfolio_returns and len(portfolio_returns) > 0:
                        # Align all return series to common dates
                        aligned_returns = []
                        common_dates = None
                        
                        for i, returns in enumerate(portfolio_returns):
                            if common_dates is None:
                                common_dates = returns.index
                            else:
                                common_dates = common_dates.intersection(returns.index)
                        
                        if len(common_dates) > 10:  # Need at least 10 data points
                            # Calculate weighted portfolio returns
                            portfolio_return_series = pd.Series(0.0, index=common_dates)
                            
                            for i, returns in enumerate(portfolio_returns):
                                aligned_returns = returns.reindex(common_dates, fill_value=0)
                                portfolio_return_series += aligned_returns * portfolio_weights[i]
                            
                            # Calculate performance metrics
                            metrics = self._calculate_performance_metrics(
                                portfolio_return_series, 
                                benchmark_returns.reindex(common_dates, fill_value=0) if benchmark_returns is not None else None,
                                period_days
                            )
                            
                            # Calculate total returns
                            total_return = (1 + portfolio_return_series).prod() - 1
                            annualized_return = ((1 + total_return) ** (365 / period_days)) - 1
                            
                            benchmark_total_return = 0
                            if benchmark_returns is not None:
                                benchmark_aligned = benchmark_returns.reindex(common_dates, fill_value=0)
                                benchmark_total_return = (1 + benchmark_aligned).prod() - 1
                                benchmark_annualized = ((1 + benchmark_total_return) ** (365 / period_days)) - 1
                            else:
                                benchmark_annualized = 0
                            
                            performance_data.append({
                                'timeframe': timeframe,
                                'returns': total_return * 100,  # Convert to percentage
                                'annualizedReturn': annualized_return * 100,
                                'benchmarkReturns': benchmark_total_return * 100,
                                'outperformance': (annualized_return - benchmark_annualized) * 100,
                                'metrics': metrics
                            })
                            
                            logger.info(f"✅ Calculated real metrics for {timeframe}: {annualized_return*100:.1f}% return")
                        else:
                            logger.warning(f"Insufficient data for {timeframe}: only {len(common_dates)} common dates")
                    else:
                        logger.warning(f"No portfolio data available for {timeframe}")
                        
                except Exception as e:
                    logger.error(f"Failed to calculate metrics for {timeframe}: {e}")
                    continue
            
            # If we couldn't calculate any real metrics, return empty list
            if not performance_data:
                logger.warning("⚠️ Could not calculate any real performance metrics")
                return []
            
            logger.info(f"✅ Successfully calculated real performance metrics for {len(performance_data)} timeframes")
            return performance_data
            
        except Exception as e:
            logger.error(f"❌ Real performance calculation failed: {e}")
            return []

    def _calculate_performance_metrics(
        self, 
        portfolio_returns: pd.Series, 
        benchmark_returns: pd.Series = None,
        period_days: int = 365
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics from return series"""
        try:
            import numpy as np
            
            # Risk-free rate (10-year Indian Government Bond yield - approximately 7%)
            risk_free_rate = 0.07
            
            # Basic return metrics
            mean_return = portfolio_returns.mean()
            volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized volatility
            
            # CAGR calculation
            total_return = (1 + portfolio_returns).prod() - 1
            years = period_days / 365
            cagr = ((1 + total_return) ** (1 / years)) - 1 if years > 0 else 0
            
            # Sharpe Ratio
            excess_return = mean_return - (risk_free_rate / 252)  # Daily risk-free rate
            sharpe_ratio = (excess_return / portfolio_returns.std()) * np.sqrt(252) if portfolio_returns.std() > 0 else 0
            
            # Sortino Ratio (downside deviation)
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
            sortino_ratio = ((cagr - risk_free_rate) / downside_deviation) if downside_deviation > 0 else 0
            
            # Maximum Drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            # Beta and Alpha (if benchmark available)
            beta = 1.0
            alpha = 0.0
            r_squared = 0.0
            tracking_error = volatility
            information_ratio = 0.0
            
            if benchmark_returns is not None and len(benchmark_returns) > 0:
                # Align returns
                aligned_portfolio = portfolio_returns.reindex(benchmark_returns.index, fill_value=0)
                aligned_benchmark = benchmark_returns.reindex(portfolio_returns.index, fill_value=0)
                
                # Calculate beta
                covariance = np.cov(aligned_portfolio, aligned_benchmark)[0, 1]
                benchmark_variance = np.var(aligned_benchmark)
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
                
                # Calculate alpha
                benchmark_mean = aligned_benchmark.mean()
                alpha = (mean_return - (risk_free_rate / 252) - beta * (benchmark_mean - (risk_free_rate / 252))) * 252
                
                # R-squared
                correlation = np.corrcoef(aligned_portfolio, aligned_benchmark)[0, 1]
                r_squared = correlation ** 2 if not np.isnan(correlation) else 0.0
                
                # Tracking error
                excess_returns = aligned_portfolio - aligned_benchmark
                tracking_error = excess_returns.std() * np.sqrt(252)
                
                # Information ratio
                information_ratio = (alpha / tracking_error) if tracking_error > 0 else 0.0
            
            # Calmar Ratio
            calmar_ratio = abs(cagr / max_drawdown) if max_drawdown < 0 else 0.0
            
            return {
                'cagr': cagr,
                'alpha': alpha,
                'beta': beta,
                'rSquared': r_squared,
                'sharpeRatio': sharpe_ratio,
                'sortinoRatio': sortino_ratio,
                'volatility': volatility,
                'downsideDeviation': downside_deviation,
                'maxDrawdown': max_drawdown,
                'trackingError': tracking_error,
                'informationRatio': information_ratio,
                'calmarRatio': calmar_ratio
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate performance metrics: {e}")
            # Return safe defaults
            return {
                'cagr': 0.0,
                'alpha': 0.0,
                'beta': 1.0,
                'rSquared': 0.0,
                'sharpeRatio': 0.0,
                'sortinoRatio': 0.0,
                'volatility': 0.2,  # 20% default volatility
                'downsideDeviation': 0.15,
                'maxDrawdown': -0.1,  # -10% default
                'trackingError': 0.1,
                'informationRatio': 0.0,
                'calmarRatio': 0.0
            }


# Create singleton instance
intelligence_service = IntelligenceService() 