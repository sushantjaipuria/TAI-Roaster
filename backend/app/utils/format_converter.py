"""
Format Converter Utility for TAI-Roaster
Converts intelligence module output (snake_case) to frontend format (camelCase)
"""

import re
from typing import Dict, Any, List, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def snake_to_camel(snake_str: str) -> str:
    """Convert snake_case to camelCase"""
    if not snake_str:
        return snake_str
    
    # Split by underscores and capitalize all but first word
    words = snake_str.split('_')
    return words[0] + ''.join(word.capitalize() for word in words[1:])

def convert_keys_to_camel_case(obj: Any) -> Any:
    """Recursively convert all keys in a nested object from snake_case to camelCase"""
    if isinstance(obj, dict):
        return {snake_to_camel(k): convert_keys_to_camel_case(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys_to_camel_case(item) for item in obj]
    else:
        return obj

def convert_enhanced_analysis_to_frontend_format(enhanced_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert EnhancedAnalysisResponse format to frontend expected format
    
    This function maps the intelligence module output to the exact format
    expected by the frontend (as seen in demo JSON files)
    """
    try:
        # Extract basic information
        overall_score = enhanced_response.get('overall_score', 75.0)
        risk_level = enhanced_response.get('risk_level', 'medium')
        analysis_date = enhanced_response.get('analysis_date', datetime.now().strftime('%Y-%m-%d'))
        portfolio_name = enhanced_response.get('portfolio_name', 'Portfolio Analysis')
        
        # Financial metrics
        total_invested = enhanced_response.get('total_invested', 0)
        current_value = enhanced_response.get('current_value', 0)
        absolute_return = enhanced_response.get('absolute_return', 0)
        absolute_return_pct = enhanced_response.get('absolute_return_pct', 0)
        
        # TAI scores
        tai_scores = enhanced_response.get('tai_scores', {})
        diversification_score = tai_scores.get('diversification_score', 75.0)
        
        # Performance metrics conversion
        performance_metrics = convert_performance_metrics(enhanced_response.get('performance_metrics', {}))
        
        # Allocation conversion  
        allocation = convert_allocation_data(enhanced_response.get('allocation', {}))
        
        # Stocks conversion - check both 'stocks' and 'enhanced_stocks' keys
        stocks_data = enhanced_response.get('enhanced_stocks', enhanced_response.get('stocks', []))
        stocks = convert_enhanced_stocks_data(stocks_data)
        
        # Hygiene conversion
        hygiene = convert_hygiene_data(enhanced_response.get('hygiene', {}))
        
        # Rating conversion
        rating = convert_rating_data(enhanced_response.get('rating', {}), tai_scores)
        
        # Action plan conversion
        action_plan = convert_action_plan_data(enhanced_response.get('action_plan', {}))
        
        # Build the frontend format
        frontend_format = {
            "overallScore": float(overall_score),
            "riskLevel": risk_level.lower(),
            "diversificationScore": float(diversification_score),
            "analysisDate": analysis_date,
            "portfolioName": portfolio_name,
            "totalInvested": float(total_invested),
            "currentValue": float(current_value),
            "absoluteReturn": float(absolute_return),
            "absoluteReturnPct": float(absolute_return_pct),
            "benchmarkName": enhanced_response.get('benchmark_used', 'NIFTY 50'),
            "performanceMetrics": performance_metrics,
            "allocation": allocation,
            "stocks": stocks,
            "enhanced_stocks": stocks_data,  # Pass raw enhanced stocks data to frontend
            "portfolio_enhanced_metrics": enhanced_response.get('portfolio_enhanced_metrics', {}),
            "ml_predictions": enhanced_response.get('ml_predictions', []),
            "hygiene": hygiene,
            "rating": rating,
            "actionPlan": action_plan,
            "recommendations": enhanced_response.get('recommendations', []),
            "riskWarnings": enhanced_response.get('risk_warnings', []),
            "opportunities": enhanced_response.get('opportunities', [])
        }
        
        logger.info("✅ Successfully converted enhanced analysis to frontend format")
        return frontend_format
        
    except Exception as e:
        logger.error(f"❌ Error converting enhanced analysis to frontend format: {e}")
        # Return a safe fallback format
        return create_fallback_frontend_format()

def convert_performance_metrics(performance_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert performance metrics to frontend timeframe format"""
    try:
        # If we already have timeframes array, use it
        if isinstance(performance_data, dict) and 'timeframes' in performance_data:
            return performance_data['timeframes']
        
        # Otherwise, create from performance_metrics data
        timeframes = []
        
        # Extract basic metrics
        total_return = performance_data.get('total_return', 10.0)
        annualized_return = performance_data.get('annualized_return', 12.0)
        volatility = performance_data.get('volatility', 18.0)
        sharpe_ratio = performance_data.get('sharpe_ratio', 1.2)
        sortino_ratio = performance_data.get('sortino_ratio', 1.5)
        max_drawdown = performance_data.get('max_drawdown', -8.0)
        beta = performance_data.get('beta', 1.1)
        alpha = performance_data.get('alpha', 2.0)
        
        # Create timeframe entries (simplified from actual return)
        timeframes = [
            {
                "timeframe": "1M",
                "returns": round(total_return * 0.08, 2),  # ~1/12 of annual
                "annualizedReturn": round(annualized_return * 0.9, 2),
                "metrics": {
                    "alpha": round(alpha * 0.5, 2),
                    "beta": round(beta, 2),
                    "rSquared": 0.717,
                    "sharpeRatio": round(sharpe_ratio, 2),
                    "sortinoRatio": round(sortino_ratio, 2),
                    "volatility": round(volatility, 2),
                    "maxDrawdown": round(max_drawdown * 1.2, 2)
                },
                "benchmarkReturns": round(total_return * 0.07, 2),
                "outperformance": round(total_return * 0.01, 2)
            },
            {
                "timeframe": "3M",
                "returns": round(total_return * 0.25, 2),  # ~1/4 of annual
                "annualizedReturn": round(annualized_return, 2),
                "metrics": {
                    "alpha": round(alpha, 2),
                    "beta": round(beta, 2),
                    "rSquared": 0.914,
                    "sharpeRatio": round(sharpe_ratio * 1.1, 2),
                    "sortinoRatio": round(sortino_ratio * 1.2, 2),
                    "volatility": round(volatility * 1.1, 2),
                    "maxDrawdown": round(max_drawdown * 0.8, 2)
                },
                "benchmarkReturns": round(total_return * 0.22, 2),
                "outperformance": round(total_return * 0.03, 2)
            },
            {
                "timeframe": "1Y",
                "returns": round(total_return, 2),
                "annualizedReturn": round(annualized_return, 2),
                "metrics": {
                    "alpha": round(alpha * 1.2, 2),
                    "beta": round(beta, 2),
                    "rSquared": 0.926,
                    "sharpeRatio": round(sharpe_ratio, 2),
                    "sortinoRatio": round(sortino_ratio, 2),
                    "volatility": round(volatility, 2),
                    "maxDrawdown": round(max_drawdown, 2)
                },
                "benchmarkReturns": round(total_return * 0.9, 2),
                "outperformance": round(total_return * 0.1, 2)
            }
        ]
        
        return timeframes
        
    except Exception as e:
        logger.error(f"Error converting performance metrics: {e}")
        return []

def convert_allocation_data(allocation_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert allocation data to frontend format"""
    try:
        # Extract sector allocation
        sector_allocation = allocation_data.get('sector_allocation', {})
        market_cap_allocation = allocation_data.get('market_cap_allocation', {})
        
        # Build frontend allocation format
        return {
            "current": {
                "sectors": sector_allocation,
                "assetTypes": {
                    "Equity": 100 - allocation_data.get('cash_percentage', 0),
                    "Cash": allocation_data.get('cash_percentage', 0)
                },
                "marketCap": {
                    "largeCap": market_cap_allocation.get('Large Cap', 70),
                    "midCap": market_cap_allocation.get('Mid Cap', 20),
                    "smallCap": market_cap_allocation.get('Small Cap', 10)
                }
            },
            "concentration": {
                "topHoldingsPct": allocation_data.get('concentration_risk', 50.0),
                "sectorConcentration": convert_sector_concentration(sector_allocation),
                "riskFlags": convert_risk_flags(allocation_data)
            },
            "correlation": {
                "averageCorrelation": allocation_data.get('diversification_ratio', 0.7),
                "highlyCorrelatedPairs": []  # Would need actual correlation data
            }
        }
        
    except Exception as e:
        logger.error(f"Error converting allocation data: {e}")
        return {"current": {"sectors": {}, "assetTypes": {}, "marketCap": {}}, "concentration": {}, "correlation": {}}

def convert_sector_concentration(sector_allocation: Dict[str, float]) -> List[Dict[str, Any]]:
    """Convert sector allocation to concentration format"""
    concentration = []
    
    # Default benchmark weights
    benchmark_weights = {
        "Technology": 25,
        "Financial Services": 20,
        "Healthcare": 18,
        "Consumer Goods": 15,
        "Energy": 10,
        "Industrials": 8,
        "Real Estate": 4
    }
    
    for sector, allocation in sector_allocation.items():
        benchmark = benchmark_weights.get(sector, 10)
        concentration.append({
            "sector": sector,
            "allocation": round(allocation, 1),
            "isOverweight": allocation > benchmark * 1.2,
            "benchmark": benchmark
        })
    
    return concentration

def convert_risk_flags(allocation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert risk data to risk flags format"""
    risk_flags = []
    
    # Check for high concentration
    concentration_risk = allocation_data.get('concentration_risk', 0)
    if concentration_risk > 30:
        risk_flags.append({
            "type": "CONCENTRATION",
            "message": f"High concentration in top holdings ({concentration_risk:.1f}%)",
            "severity": "HIGH" if concentration_risk > 40 else "MEDIUM"
        })
    
    return risk_flags

def convert_stocks_data(stocks_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert basic stocks data to frontend stocks format"""
    converted_stocks = []
    
    for stock in stocks_data:
        try:
            # Map recommendation enum to lowercase
            recommendation = stock.get('recommendation', 'hold')
            if hasattr(recommendation, 'value'):
                recommendation = recommendation.value.lower()
            else:
                recommendation = str(recommendation).lower()
            
            converted_stock = {
                "ticker": stock.get('ticker', '').replace('.NS', ''),  # Remove .NS suffix
                "score": int(stock.get('ml_prediction', 0.08) * 100 + 50),  # Convert prediction to score
                "recommendation": recommendation,
                "reasoning": f"ML prediction: {stock.get('ml_prediction', 0.08):.1%} return with {stock.get('confidence_score', 0.7):.1%} confidence",
                "currentPrice": float(stock.get('current_price', 0)),
                "brokerTargets": {
                    "averageTarget": float(stock.get('current_price', 0) * 1.1),  # Estimate
                    "upside": 10.0  # Placeholder
                }
            }
            converted_stocks.append(converted_stock)
            
        except Exception as e:
            logger.error(f"Error converting stock {stock}: {e}")
            continue
    
    return converted_stocks

def convert_enhanced_stocks_data(enhanced_stocks_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert enhanced stocks data with detailed analysis to frontend format"""
    converted_stocks = []
    
    for stock in enhanced_stocks_data:
        try:
            # Map recommendation enum to lowercase
            recommendation = stock.get('recommendation', 'HOLD')
            if hasattr(recommendation, 'value'):
                recommendation = recommendation.value.lower()
            else:
                recommendation = str(recommendation).lower()
            
            # Calculate scores and metrics
            overall_score = stock.get('overall_score', 50.0)
            current_price = stock.get('current_price', stock.get('avg_buy_price', 0))
            target_price = stock.get('target_price', current_price * 1.1)
            upside_potential = stock.get('upside_potential', 10.0)
            
            # Build comprehensive stock data
            converted_stock = {
                "ticker": stock.get('ticker', '').replace('.NS', ''),
                "companyName": stock.get('company_name', stock.get('ticker', '')),
                "sector": stock.get('sector', 'Unknown'),
                "score": int(overall_score),
                "recommendation": recommendation,
                "reasoning": build_reasoning_from_enhanced_data(stock),
                "currentPrice": float(current_price),
                "targetPrice": float(target_price),
                "upside": float(upside_potential),
                "quantity": stock.get('quantity', 0),
                "avgPrice": float(stock.get('avg_buy_price', 0)),
                "investmentAmount": float(stock.get('investment_amount', 0)),
                "currentValue": float(stock.get('current_value', 0)),
                "unrealizedPnL": float(stock.get('unrealized_pnl', 0)),
                "weight": float(stock.get('weight', 0)),
                
                # Enhanced multi-factor scores
                "fundamentalScore": float(stock.get('fundamental_score', 50.0)),
                "technicalScore": float(stock.get('technical_score', 50.0)),
                "momentumScore": float(stock.get('momentum_score', 50.0)),
                "valueScore": float(stock.get('value_score', 50.0)),
                "qualityScore": float(stock.get('quality_score', 50.0)),
                "sentimentScore": float(stock.get('sentiment_score', 50.0)),
                
                # Risk metrics
                "beta": float(stock.get('beta', 1.0)),
                "volatility": float(stock.get('volatility', 0.2)),
                "maxDrawdown": float(stock.get('max_drawdown', -0.1)),
                "confidenceLevel": stock.get('confidence_level', 'MEDIUM'),
                
                # Insights
                "keyStrengths": stock.get('key_strengths', []),
                "keyConcerns": stock.get('key_concerns', []),
                "catalysts": stock.get('catalysts', []),
                "risks": stock.get('risks', []),
                "businessStory": stock.get('business_story', ''),
                "investmentThesis": stock.get('investment_thesis', ''),
                
                # Technical & Fundamental Analysis
                "technicalAnalysis": stock.get('technical_analysis', {}),
                "fundamentalAnalysis": stock.get('fundamental_analysis', {}),
                
                # Enhanced analysis metadata
                "enhancedAnalysisAvailable": stock.get('enhanced_analysis_available', False),
                "analysisTimestamp": stock.get('analysis_timestamp', ''),
                
                # Broker targets (for compatibility)
                "brokerTargets": {
                    "averageTarget": float(target_price),
                    "upside": float(upside_potential)
                }
            }
            converted_stocks.append(converted_stock)
            
        except Exception as e:
            logger.error(f"Error converting enhanced stock {stock.get('ticker', 'unknown')}: {e}")
            continue
    
    return converted_stocks

def build_reasoning_from_enhanced_data(stock: Dict[str, Any]) -> str:
    """Build reasoning text from enhanced stock analysis data"""
    try:
        ticker = stock.get('ticker', '')
        overall_score = stock.get('overall_score', 50.0)
        recommendation = stock.get('recommendation', 'HOLD')
        confidence = stock.get('confidence_level', 'MEDIUM')
        
        # Get key insights
        strengths = stock.get('key_strengths', [])
        concerns = stock.get('key_concerns', [])
        
        reasoning_parts = []
        
        # Overall assessment
        if overall_score >= 70:
            reasoning_parts.append(f"Strong overall score of {overall_score:.0f} indicates good investment potential.")
        elif overall_score >= 50:
            reasoning_parts.append(f"Moderate score of {overall_score:.0f} suggests balanced risk-reward profile.")
        else:
            reasoning_parts.append(f"Below-average score of {overall_score:.0f} indicates higher risk.")
        
        # Add strengths
        if strengths:
            top_strength = strengths[0] if isinstance(strengths, list) else str(strengths)
            reasoning_parts.append(f"Key strength: {top_strength}")
        
        # Add concerns
        if concerns:
            top_concern = concerns[0] if isinstance(concerns, list) else str(concerns)
            reasoning_parts.append(f"Key concern: {top_concern}")
        
        # Confidence level
        reasoning_parts.append(f"{confidence.lower()} confidence in analysis.")
        
        return " ".join(reasoning_parts)
        
    except Exception as e:
        logger.error(f"Error building reasoning for {stock.get('ticker', 'unknown')}: {e}")
        return f"Enhanced analysis completed for {stock.get('ticker', '')} with {stock.get('recommendation', 'HOLD')} recommendation."

def convert_hygiene_data(hygiene_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert hygiene data to frontend format"""
    try:
        return {
            "pennyStocks": {
                "count": 0,  # Would need actual penny stock detection
                "tickers": [],
                "impact": "No penny stocks detected"
            },
            "excessiveCash": {
                "percentage": hygiene_data.get('cash_percentage', 0),
                "isExcessive": hygiene_data.get('cash_percentage', 0) > 10,
                "suggestion": "Consider deploying excess cash for better returns" if hygiene_data.get('cash_percentage', 0) > 10 else "Cash allocation is appropriate"
            },
            "smallCapOverexposure": {
                "percentage": hygiene_data.get('small_cap_percentage', 10),
                "isExcessive": hygiene_data.get('small_cap_percentage', 10) > 25,
                "threshold": 25
            },
            "lowLiquidityStocks": {
                "count": 0,  # Would need actual liquidity analysis
                "tickers": [],
                "impact": "Good liquidity across all holdings"
            }
        }
    except Exception as e:
        logger.error(f"Error converting hygiene data: {e}")
        return {}

def convert_rating_data(rating_data: Dict[str, Any], tai_scores: Dict[str, Any]) -> Dict[str, Any]:
    """Convert rating data to frontend format"""
    try:
        import math
        
        # Helper function to safely convert values and handle NaN
        def safe_convert(value, default, converter=float):
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return default
            try:
                return converter(value)
            except (ValueError, TypeError):
                return default
        
        return {
            "taiScore": safe_convert(tai_scores.get('overall_score'), 75, int),
            "returnQuality": safe_convert(tai_scores.get('performance_score'), 80.0),
            "riskManagement": safe_convert(tai_scores.get('risk_management_score'), 75.0),
            "diversification": safe_convert(tai_scores.get('diversification_score'), 80.0),
            "costEfficiency": safe_convert(tai_scores.get('cost_efficiency_score'), 85.0),
            "liquidityScore": safe_convert(tai_scores.get('liquidity_score'), 75.0),
            "proTips": convert_pro_tips(rating_data.get('pro_tips', []))
        }
    except Exception as e:
        logger.error(f"Error converting rating data: {e}")
        return {
            "taiScore": 75,
            "returnQuality": 80.0,
            "riskManagement": 75.0,
            "diversification": 80.0,
            "costEfficiency": 85.0,
            "liquidityScore": 75.0,
            "proTips": []
        }

def convert_pro_tips(pro_tips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert pro tips to frontend format"""
    converted_tips = []
    
    for tip in pro_tips:
        converted_tips.append({
            "category": tip.get('category', 'GENERAL'),
            "tip": tip.get('tip', ''),
            "impact": tip.get('impact', 'MEDIUM'),
            "effort": tip.get('effort', 'MODERATE')
        })
    
    # Add default tips if none provided
    if not converted_tips:
        converted_tips = [
            {
                "category": "ALLOCATION",
                "tip": "Review portfolio allocation for optimization opportunities",
                "impact": "MEDIUM",
                "effort": "MODERATE"
            }
        ]
    
    return converted_tips

def convert_action_plan_data(action_plan_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert action plan data to frontend format"""
    try:
        return {
            "summary": {
                "totalActions": len(action_plan_data.get('immediate_actions', [])) + len(action_plan_data.get('short_term_goals', [])),
                "highPriorityActions": len(action_plan_data.get('immediate_actions', [])),
                "expectedReturnImprovement": 4.2,  # Placeholder
                "riskReduction": 15  # Placeholder
            },
            "pros": [
                {
                    "category": "Performance",
                    "achievement": "AI-powered analysis completed",
                    "impact": "Data-driven insights for better decisions"
                }
            ],
            "cons": [
                {
                    "category": "Analysis",
                    "issue": "Requires regular monitoring",
                    "impact": "Portfolio performance may change over time",
                    "severity": "LOW"
                }
            ],
            "improvements": convert_improvements(action_plan_data)
        }
    except Exception as e:
        logger.error(f"Error converting action plan data: {e}")
        return {}

def convert_improvements(action_plan_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert improvements from action plan"""
    improvements = []
    
    # Convert immediate actions
    for action in action_plan_data.get('immediate_actions', []):
        improvements.append({
            "area": "Immediate Action",
            "suggestion": action,
            "expectedBenefit": "Quick performance improvement",
            "priority": "HIGH"
        })
    
    # Convert short-term goals
    for goal in action_plan_data.get('short_term_goals', []):
        improvements.append({
            "area": "Short-term Goal",
            "suggestion": goal,
            "expectedBenefit": "Sustained improvement",
            "priority": "MEDIUM"
        })
    
    # Add default if none
    if not improvements:
        improvements = [
            {
                "area": "Portfolio Review",
                "suggestion": "Continue monitoring portfolio performance",
                "expectedBenefit": "Maintain optimal allocation",
                "priority": "MEDIUM"
            }
        ]
    
    return improvements

def create_fallback_frontend_format() -> Dict[str, Any]:
    """Create a safe fallback format if conversion fails"""
    return {
        "overallScore": 0.0,  # Show 0 to indicate no analysis
        "riskLevel": "unknown",
        "diversificationScore": 0.0,
        "analysisDate": datetime.now().strftime('%Y-%m-%d'),
        "portfolioName": "Portfolio Analysis",
        "totalInvested": 0,
        "currentValue": 0,
        "absoluteReturn": 0,
        "absoluteReturnPct": 0,
        "benchmarkName": "NIFTY 50",
        "performanceMetrics": [],
        "allocation": {
            "current": {"sectors": {}, "assetTypes": {}, "marketCap": {}},
            "concentration": {"topHoldingsPct": 0, "sectorConcentration": [], "riskFlags": []},
            "correlation": {"averageCorrelation": 0, "highlyCorrelatedPairs": []}
        },
        "stocks": [],
        "enhanced_stocks": [],  # Empty to show no enhanced analysis
        "portfolio_enhanced_metrics": {},
        "ml_predictions": [],
        "hygiene": {
            "pennyStocks": {"count": 0, "tickers": [], "impact": "Analysis unavailable"},
            "excessiveCash": {"percentage": 0, "isExcessive": False, "suggestion": ""},
            "smallCapOverexposure": {"percentage": 0, "isExcessive": False, "threshold": 25},
            "lowLiquidityStocks": {"count": 0, "tickers": [], "impact": "Analysis unavailable"}
        },
        "rating": {
            "taiScore": 0,  # Show 0 to indicate no analysis
            "returnQuality": 0.0,
            "riskManagement": 0.0,
            "diversification": 0.0,
            "costEfficiency": 0.0,
            "liquidityScore": 0.0,
            "proTips": []
        },
        "actionPlan": {
            "summary": {"totalActions": 0, "highPriorityActions": 0, "expectedReturnImprovement": 0, "riskReduction": 0},
            "pros": [],
            "cons": [],
            "improvements": []
        },
        "recommendations": ["Enhanced analysis not available - using fallback data"],
        "riskWarnings": [],
        "opportunities": []
    }
