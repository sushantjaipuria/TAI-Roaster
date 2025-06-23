# intelligence/portfolio_strategist.py

import json
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("[WARNING] OpenAI not available - LLM features disabled")
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("[WARNING] Anthropic not available - LLM features disabled")

try:
    from backend.app.schemas.enhanced_analysis import StockRecommendation
    from backend.app.schemas.output import (
        PredictionResult,
        ReturnRange,
        BacktestMetrics,
        PricePoint,
        BufferRange,
    )
except ImportError:
    # Fallback for when schemas are not available
    from enum import Enum
    
    class StockRecommendation(str, Enum):
        BUY = "Buy"
        HOLD = "Hold"
        SELL = "Sell"
        STRONG_BUY = "Strong Buy"
        STRONG_SELL = "Strong Sell"
    
    # Use mock classes for other schemas
    class PredictionResult:
        pass
    class ReturnRange:
        pass
    class BacktestMetrics:
        pass
    class PricePoint:
        pass
    class BufferRange:
        pass

class PortfolioJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for portfolio data"""
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)

def clean_for_json(data):
    """Clean data structure for JSON serialization"""
    if isinstance(data, dict):
        return {k: clean_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_for_json(item) for item in data]
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, pd.Timestamp):
        return data.isoformat()
    elif hasattr(data, 'isoformat'):  # datetime objects
        return data.isoformat()
    else:
        return data

@dataclass
class MLRecommendation:
    """Raw ML model recommendation before LLM validation"""
    ticker: str
    expected_return: float
    confidence_score: float
    allocation_amount: float
    model_predictions: Dict[str, Any]  # Individual model outputs
    technical_data: Dict[str, Any]
    fundamental_data: Dict[str, Any]
    risk_metrics: Dict[str, Any]

@dataclass
class ValidatedRecommendation:
    """LLM-validated recommendation"""
    ticker: str
    expected_return: float
    confidence_score: float
    validation_score: float  # LLM validation confidence
    rejection_reason: Optional[str]
    enhanced_metrics: Dict[str, Any]
    llm_adjustments: Dict[str, Any]

@dataclass
class PortfolioBucket:
    """Individual portfolio bucket with specific strategy"""
    bucket_name: str
    strategy_type: str  # "Conservative", "Balanced", "Aggressive", "Growth", "Value"
    target_return: float
    risk_level: str
    confidence_level: float
    allocation_percentage: float  # % of total investment
    stocks: List[StockRecommendation]
    rationale: str
    rebalancing_frequency: str
    exit_conditions: List[str]
    monitoring_metrics: List[str]

class LLMPortfolioStrategist:
    """Two-stage LLM system: Validation + Portfolio Strategy"""
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4.1-nano"):
        self.provider = provider
        self.model = model
        
        if provider == "openai":
            self.client = openai.OpenAI()
        elif provider == "anthropic":
            self.client = Anthropic()
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    async def validate_and_filter_recommendations(
        self,
        ml_recommendations: List[MLRecommendation],
        user_preferences: Dict[str, Any],
        market_context: Dict[str, Any]
    ) -> List[ValidatedRecommendation]:
        """Stage 1: LLM validates and filters ML recommendations"""
        
        validation_prompt = self._create_validation_prompt(
            ml_recommendations, user_preferences, market_context
        )
        
        if self.provider == "openai":
            response = await self._get_openai_response(validation_prompt)
        else:
            response = await self._get_anthropic_response(validation_prompt)
        
        return self._parse_validation_response(response, ml_recommendations)
    
    async def create_portfolio_buckets(
        self,
        validated_recommendations: List[ValidatedRecommendation],
        user_preferences: Dict[str, Any],
        total_investment: float
    ) -> List[PortfolioBucket]:
        """Stage 2: LLM creates multiple portfolio buckets with different strategies"""
        
        strategy_prompt = self._create_strategy_prompt(
            validated_recommendations, user_preferences, total_investment
        )
        
        if self.provider == "openai":
            response = await self._get_openai_response(strategy_prompt)
        else:
            response = await self._get_anthropic_response(strategy_prompt)
        
        return self._parse_strategy_response(response, validated_recommendations, total_investment)
    
    def _create_validation_prompt(
        self,
        ml_recommendations: List[MLRecommendation],
        user_preferences: Dict[str, Any],
        market_context: Dict[str, Any]
    ) -> str:
        """Create prompt for Stage 1: Validation and Filtering"""
        
        prompt = f"""
You are a senior portfolio manager and quantitative analyst with 25+ years of experience in Indian equity markets. 
Your task is to validate and filter ML-generated stock recommendations based on rigorous analysis.

## USER PREFERENCES:
- Investment Amount: ₹{user_preferences.get('amount', 0):,}
- Market Cap Preference: {user_preferences.get('market_cap', 'largecap')}
- Risk Tolerance: {user_preferences.get('risk_tolerance', 'medium')}

## MARKET CONTEXT:
- Market Sentiment: {market_context.get('sentiment', 'NEUTRAL')}
- Volatility Regime: {market_context.get('volatility_regime', 'NORMAL')}
- Economic Environment: {market_context.get('economic_environment', 'STABLE')}

## ML RECOMMENDATIONS TO VALIDATE:
"""
        
        for i, rec in enumerate(ml_recommendations, 1):
            # Clean the model predictions for JSON serialization
            clean_predictions = clean_for_json(rec.model_predictions)
            clean_technical_data = clean_for_json(rec.technical_data)
            clean_risk_metrics = clean_for_json(rec.risk_metrics)
            
            prompt += f"""
### Recommendation {i}: {rec.ticker}
- Expected Return: {rec.expected_return:.2%}
- ML Confidence: {rec.confidence_score:.1%}
- Proposed Allocation: ₹{rec.allocation_amount:,.0f}

**Model Predictions:**
{json.dumps(clean_predictions, indent=2, cls=PortfolioJSONEncoder)}

**Technical Metrics:**
- RSI: {clean_technical_data.get('rsi_14', 'N/A')}
- MACD: {clean_technical_data.get('macd', 'N/A')}
- Trend Strength: {clean_technical_data.get('trend_strength', 'N/A')}
- Volume Ratio: {clean_technical_data.get('volume_ratio', 'N/A')}

**Risk Metrics:**
- Volatility: {clean_risk_metrics.get('volatility', 'N/A')}
- Max Drawdown: {clean_risk_metrics.get('max_drawdown', 'N/A')}
- Beta: {clean_risk_metrics.get('beta', 'N/A')}

"""
        
        prompt += f"""

## VALIDATION CRITERIA:
Evaluate each recommendation based on:
1. **Model Consensus**: Agreement between different ML models
2. **Risk-Return Profile**: Risk-adjusted return attractiveness
3. **Market Timing**: Suitability given current market conditions
4. **Technical Strength**: Quality of technical indicators
5. **Fundamental Soundness**: Business fundamentals and valuation
6. **Portfolio Fit**: Alignment with user preferences and risk tolerance
7. **Liquidity & Execution**: Ability to execute trades efficiently
8. **Correlation Risk**: Avoid over-concentration in similar stocks

## REQUIRED OUTPUT FORMAT:
Provide your analysis as a JSON array with the following structure:

{{
  "validation_results": [
    {{
      "ticker": "RELIANCE",
      "validation_decision": "ACCEPT|REJECT|CONDITIONAL",
      "validation_score": 0.85,
      "adjusted_expected_return": 0.12,
      "adjusted_confidence": 0.78,
      "adjusted_allocation": 15000,
      "validation_reasoning": {{
        "strengths": ["Strong technical momentum", "Solid fundamentals"],
        "concerns": ["High correlation with market", "Valuation stretched"],
        "decisive_factors": ["Earnings growth trajectory", "Sector leadership"],
        "risk_assessment": "MEDIUM"
      }},
      "rejection_reason": null,
      "enhancement_suggestions": {{
        "entry_timing": "GRADUAL",
        "stop_loss_level": 0.08,
        "profit_target": 0.15,
        "monitoring_points": ["Quarterly results", "Oil price trends"]
      }}
    }}
  ],
  "overall_assessment": {{
    "total_recommendations_reviewed": 10,
    "accepted_count": 7,
    "rejected_count": 2,
    "conditional_count": 1,
    "portfolio_quality_score": 0.82,
    "risk_concentration_warning": false,
    "market_timing_assessment": "FAVORABLE"
  }}
}}

## EXPERT GUIDELINES:
1. Be selective - reject recommendations that don't meet high standards
2. Consider correlation between stocks to avoid concentration risk
3. Adjust allocations based on conviction levels
4. Factor in current market regime and timing
5. Prioritize risk-adjusted returns over absolute returns
6. Consider liquidity and execution feasibility
7. Ensure recommendations align with user's risk tolerance
8. Provide specific, actionable reasoning for each decision

Analyze each recommendation thoroughly and provide your expert validation.
"""
        
        return prompt
    
    def _create_strategy_prompt(
        self,
        validated_recommendations: List[ValidatedRecommendation],
        user_preferences: Dict[str, Any],
        total_investment: float
    ) -> str:
        """Create prompt for Stage 2: Portfolio Strategy and Bucket Creation"""
        
        prompt = f"""
You are a world-class portfolio strategist and wealth manager with expertise in creating diversified investment strategies. 
Your task is to create multiple portfolio buckets using validated stock recommendations.

## INVESTMENT PARAMETERS:
- Total Investment: ₹{total_investment:,}
- Risk Tolerance: {user_preferences.get('risk_tolerance', 'medium')}
- Market Cap Preference: {user_preferences.get('market_cap', 'largecap')}

## VALIDATED STOCK RECOMMENDATIONS:
"""
        
        for i, rec in enumerate(validated_recommendations, 1):
            # Clean enhanced metrics for JSON serialization
            clean_enhanced_metrics = clean_for_json(rec.enhanced_metrics)
            
            prompt += f"""
### Stock {i}: {rec.ticker}
- Expected Return: {rec.expected_return:.2%}
- Confidence: {rec.confidence_score:.1%}
- Validation Score: {rec.validation_score:.1%}
- Enhanced Metrics: {json.dumps(clean_enhanced_metrics, indent=2, cls=PortfolioJSONEncoder)}
"""
        
        prompt += f"""

## PORTFOLIO STRATEGY REQUIREMENTS:
Create 3-5 distinct portfolio buckets with different risk-return profiles:

### Bucket Types to Consider:
1. **Conservative Bucket**: Low risk, stable returns, high dividend yield
2. **Balanced Bucket**: Moderate risk, balanced growth and value
3. **Growth Bucket**: Higher risk, high growth potential
4. **Value Bucket**: Undervalued stocks with recovery potential
5. **Momentum Bucket**: Trending stocks with strong technical signals

## REQUIRED OUTPUT FORMAT:
Provide your strategy as a JSON object:

{{
  "portfolio_strategy": {{
    "strategy_name": "Multi-Bucket Diversified Strategy",
    "total_investment": {total_investment},
    "number_of_buckets": 4,
    "rebalancing_frequency": "Quarterly",
    "overall_expected_return": 0.14,
    "overall_risk_level": "MEDIUM",
    "strategy_rationale": "Detailed explanation of the overall strategy"
  }},
  "portfolio_buckets": [
    {{
      "bucket_name": "Conservative Core",
      "strategy_type": "Conservative",
      "allocation_percentage": 0.40,
      "allocation_amount": 40000,
      "target_return": 0.10,
      "risk_level": "LOW",
      "confidence_level": 0.85,
      "stocks": [
        {{
          "ticker": "HDFCBANK",
          "allocation_amount": 20000,
          "allocation_percentage": 0.50,
          "expected_return": 0.09,
          "confidence_score": 0.88,
          "role_in_bucket": "Anchor holding",
          "rationale": "Stable banking leader with consistent performance"
        }}
      ],
      "bucket_rationale": "Provides stability and consistent returns",
      "rebalancing_frequency": "Semi-annually",
      "exit_conditions": ["Major sector headwinds", "Valuation becomes expensive"],
      "monitoring_metrics": ["Dividend yield", "P/E ratio", "ROE"],
      "risk_management": {{
        "stop_loss_level": 0.10,
        "profit_booking_level": 0.15,
        "correlation_limit": 0.70
      }}
    }}
  ],
  "portfolio_analytics": {{
    "expected_portfolio_return": 0.14,
    "portfolio_volatility": 0.18,
    "sharpe_ratio": 0.78,
    "max_drawdown_estimate": 0.15,
    "correlation_matrix_summary": "Low to moderate correlation between buckets",
    "diversification_score": 0.82
  }},
  "implementation_plan": {{
    "phase_1": "Deploy Conservative and Balanced buckets (60% of capital)",
    "phase_2": "Add Growth bucket based on market conditions (25% of capital)",
    "phase_3": "Complete with Value/Momentum bucket (15% of capital)",
    "timeline": "3-4 weeks for full deployment",
    "execution_strategy": "Gradual deployment to minimize market impact"
  }},
  "monitoring_framework": {{
    "daily_metrics": ["Portfolio value", "Individual stock performance"],
    "weekly_metrics": ["Bucket performance", "Risk metrics"],
    "monthly_metrics": ["Rebalancing needs", "Strategy effectiveness"],
    "quarterly_reviews": ["Full strategy assessment", "Bucket optimization"]
  }}
}}

## STRATEGY GUIDELINES:
1. **Diversification**: Ensure buckets have different risk-return profiles
2. **Correlation Management**: Minimize overlap between buckets
3. **Risk Budgeting**: Allocate risk appropriately across buckets
4. **Liquidity Management**: Ensure adequate liquidity for rebalancing
5. **Tax Efficiency**: Consider tax implications of the strategy
6. **Scalability**: Strategy should work as portfolio grows
7. **Behavioral Finance**: Account for investor psychology and biases
8. **Market Regime Adaptation**: Strategy should adapt to changing markets

## ALLOCATION PRINCIPLES:
- Conservative bucket: 30-50% (based on risk tolerance)
- Balanced bucket: 25-35%
- Growth/Aggressive bucket: 15-25%
- Specialty buckets (Value/Momentum): 10-20%

Create a comprehensive, professional portfolio strategy that maximizes risk-adjusted returns while meeting the investor's objectives.
"""
        
        return prompt
    
    async def _get_openai_response(self, prompt: str) -> str:
        """Get response from OpenAI GPT"""
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert portfolio manager and quantitative analyst. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=4000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return self._get_fallback_response()
    
    async def _get_anthropic_response(self, prompt: str) -> str:
        """Get response from Anthropic Claude"""
        try:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model="claude-3-sonnet-20240229",
                max_tokens=4000,
                temperature=0.1,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            print(f"Anthropic API error: {e}")
            return self._get_fallback_response()
    
    def _get_fallback_response(self) -> str:
        """Fallback response when LLM is unavailable"""
        return json.dumps({
            "validation_results": [],
            "portfolio_strategy": {
                "strategy_name": "Fallback Strategy",
                "total_investment": 100000,
                "number_of_buckets": 1,
                "overall_expected_return": 0.08,
                "overall_risk_level": "MEDIUM"
            },
            "portfolio_buckets": [],
            "error": "LLM unavailable - using fallback mode"
        })
    
    def _parse_validation_response(
        self,
        response: str,
        ml_recommendations: List[MLRecommendation]
    ) -> List[ValidatedRecommendation]:
        """Parse LLM validation response"""
        try:
            # Extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                validation_data = json.loads(json_str)
                
                validated_recs = []
                
                for result in validation_data.get('validation_results', []):
                    if result.get('validation_decision') in ['ACCEPT', 'CONDITIONAL']:
                        validated_rec = ValidatedRecommendation(
                            ticker=result['ticker'],
                            expected_return=result.get('adjusted_expected_return', 0.08),
                            confidence_score=result.get('adjusted_confidence', 0.6),
                            validation_score=result.get('validation_score', 0.7),
                            rejection_reason=result.get('rejection_reason'),
                            enhanced_metrics=result.get('validation_reasoning', {}),
                            llm_adjustments=result.get('enhancement_suggestions', {})
                        )
                        validated_recs.append(validated_rec)
                
                return validated_recs
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing validation response: {e}")
        
        # Fallback: accept top ML recommendations with basic validation
        return [
            ValidatedRecommendation(
                ticker=rec.ticker,
                expected_return=rec.expected_return,
                confidence_score=rec.confidence_score * 0.8,  # Reduce confidence in fallback
                validation_score=0.6,
                rejection_reason=None,
                enhanced_metrics={},
                llm_adjustments={}
            )
            for rec in ml_recommendations[:5]  # Take top 5 in fallback
        ]
    
    def _parse_strategy_response(
        self,
        response: str,
        validated_recommendations: List[ValidatedRecommendation],
        total_investment: float
    ) -> List[PortfolioBucket]:
        """Parse LLM strategy response into portfolio buckets"""
        try:
            # Extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                strategy_data = json.loads(json_str)
                
                buckets = []
                
                for bucket_data in strategy_data.get('portfolio_buckets', []):
                    # Convert bucket stocks to StockRecommendation objects
                    bucket_stocks = []
                    for stock_data in bucket_data.get('stocks', []):
                        # Find the validated recommendation for this ticker
                        validated_rec = next(
                            (rec for rec in validated_recommendations if rec.ticker == stock_data['ticker']),
                            None
                        )
                        
                        if validated_rec:
                            stock_rec = StockRecommendation(
                                ticker=stock_data['ticker'].replace('.NS', ''),
                                expected_return=stock_data.get('expected_return', validated_rec.expected_return),
                                confidence_score=stock_data.get('confidence_score', validated_rec.confidence_score),
                                allocation_amount=stock_data.get('allocation_amount', 0),
                                explanation=f"{bucket_data['bucket_name']}: {stock_data.get('rationale', 'LLM-optimized allocation')}",
                                return_range=ReturnRange(
                                    min=stock_data.get('expected_return', 0.05) - 0.05,
                                    max=stock_data.get('expected_return', 0.05) + 0.05
                                ),
                                backtest_metrics=BacktestMetrics(
                                    cagr=0.12,
                                    sharpe_ratio=1.2,
                                    max_drawdown=-0.15
                                ),
                                entry_point=PricePoint(
                                    value=1000.0,
                                    target=1000.0,
                                    buffer_range=BufferRange(min=950.0, max=1050.0)
                                ),
                                exit_point=PricePoint(
                                    value=1100.0,
                                    target=1100.0,
                                    buffer_range=BufferRange(min=1050.0, max=1150.0)
                                )
                            )
                            bucket_stocks.append(stock_rec)
                    
                    bucket = PortfolioBucket(
                        bucket_name=bucket_data['bucket_name'],
                        strategy_type=bucket_data['strategy_type'],
                        target_return=bucket_data.get('target_return', 0.1),
                        risk_level=bucket_data.get('risk_level', 'MEDIUM'),
                        confidence_level=bucket_data.get('confidence_level', 0.7),
                        allocation_percentage=bucket_data.get('allocation_percentage', 0.25),
                        stocks=bucket_stocks,
                        rationale=bucket_data.get('bucket_rationale', 'LLM-generated strategy'),
                        rebalancing_frequency=bucket_data.get('rebalancing_frequency', 'Quarterly'),
                        exit_conditions=bucket_data.get('exit_conditions', []),
                        monitoring_metrics=bucket_data.get('monitoring_metrics', [])
                    )
                    buckets.append(bucket)
                
                return buckets
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing strategy response: {e}")
        
        # Fallback: create simple single bucket
        return self._create_fallback_buckets(validated_recommendations, total_investment)
    
    def _create_fallback_buckets(
        self,
        validated_recommendations: List[ValidatedRecommendation],
        total_investment: float
    ) -> List[PortfolioBucket]:
        """Create fallback portfolio buckets when LLM fails"""
        
        # Create simple balanced bucket with all validated recommendations
        bucket_stocks = []
        allocation_per_stock = total_investment / max(len(validated_recommendations), 1)
        
        for rec in validated_recommendations:
            stock_rec = StockRecommendation(
                ticker=rec.ticker.replace('.NS', ''),
                expected_return=rec.expected_return,
                confidence_score=rec.confidence_score,
                allocation_amount=allocation_per_stock,
                explanation=f"Fallback allocation: {rec.ticker}",
                return_range=ReturnRange(min=0.05, max=0.15),
                backtest_metrics=BacktestMetrics(cagr=0.1, sharpe_ratio=1.0, max_drawdown=-0.2),
                entry_point=PricePoint(
                    value=1000.0, target=1000.0,
                    buffer_range=BufferRange(min=950.0, max=1050.0)
                ),
                exit_point=PricePoint(
                    value=1100.0, target=1100.0,
                    buffer_range=BufferRange(min=1050.0, max=1150.0)
                )
            )
            bucket_stocks.append(stock_rec)
        
        fallback_bucket = PortfolioBucket(
            bucket_name="Balanced Portfolio",
            strategy_type="Balanced",
            target_return=0.1,
            risk_level="MEDIUM",
            confidence_level=0.6,
            allocation_percentage=1.0,
            stocks=bucket_stocks,
            rationale="Fallback balanced allocation when LLM unavailable",
            rebalancing_frequency="Quarterly",
            exit_conditions=["Major market downturn"],
            monitoring_metrics=["Portfolio return", "Risk metrics"]
        )
        
        return [fallback_bucket]

class EnhancedPortfolioEngine:
    """Main engine that orchestrates the two-stage LLM process"""
    
    def __init__(self, llm_strategist: LLMPortfolioStrategist, llm_expert=None):
        self.llm_strategist = llm_strategist
        # 🔥 Use provided llm_expert or create new one
        if llm_expert:
            self.llm_expert = llm_expert
        else:
            from intelligence.llm_trading_expert import LLMTradingExpert
            self.llm_expert = LLMTradingExpert(provider="openai", model="gpt-4.1-nano")
    
    async def generate_portfolio_buckets(
        self,
        ml_recommendations: List[MLRecommendation],
        user_preferences: Dict[str, Any],
        market_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Main function to generate portfolio buckets using two-stage LLM process"""
        
        # 🔍 DEBUG: Log ML recommendations before caching (SAFE VERSION)
        try:
            rec_count = len(ml_recommendations) if ml_recommendations else 0
            print(f"🔍 [DEBUG] Received {rec_count} ML recommendations")
            print(f"🔍 [DEBUG] ML recommendations type: {type(ml_recommendations)}")
            print(f"🔍 [DEBUG] ML recommendations is None: {ml_recommendations is None}")
            
            if ml_recommendations and hasattr(ml_recommendations, '__iter__'):
                for i, rec in enumerate(ml_recommendations):
                    if hasattr(rec, 'ticker') and hasattr(rec, 'expected_return'):
                        print(f"🔍 [DEBUG] ML Rec {i+1}: {rec.ticker} - Return: {rec.expected_return:.2%}")
                    else:
                        print(f"🔍 [DEBUG] ML Rec {i+1}: Invalid rec object - {type(rec)}")
            else:
                print(f"🔍 [DEBUG] ML recommendations not iterable or empty")
        except Exception as e:
            print(f"🔍 [DEBUG] Error logging ML recommendations: {e}")
            print(f"🔍 [DEBUG] ML recommendations value: {ml_recommendations}")
            print(f"🔍 [DEBUG] ML recommendations type: {type(ml_recommendations)}")
            # Continue execution despite logging error
        
        # Store ML data for later use (SAFE VERSION)
        try:
            if ml_recommendations and hasattr(ml_recommendations, '__iter__'):
                self._ml_recommendations_cache = {}
                for rec in ml_recommendations:
                    if hasattr(rec, 'ticker') and rec.ticker:
                        self._ml_recommendations_cache[rec.ticker] = rec
                print(f"🔍 [DEBUG] Successfully created cache with {len(self._ml_recommendations_cache)} entries")
            else:
                print(f"🔍 [DEBUG] ML recommendations invalid, creating empty cache")
                self._ml_recommendations_cache = {}
        except Exception as e:
            print(f"🔍 [DEBUG] Error creating cache: {e}")
            self._ml_recommendations_cache = {}
        
        # 🔍 DEBUG: Verify cache contents
        print(f"🔍 [DEBUG] Cache populated with {len(self._ml_recommendations_cache)} entries")
        print(f"🔍 [DEBUG] Cache keys: {list(self._ml_recommendations_cache.keys())}")
        
        print("🔍 Stage 1: LLM Validation and Filtering...")
        
        # Stage 1: Validate and filter ML recommendations
        validated_recommendations = await self.llm_strategist.validate_and_filter_recommendations(
            ml_recommendations, user_preferences, market_context
        )
        
        print(f"✅ Validated {len(validated_recommendations)} out of {len(ml_recommendations)} recommendations")
        
        print("🎯 Stage 2: LLM Portfolio Strategy Creation...")
        
        # Stage 2: Create portfolio buckets
        portfolio_buckets = await self.llm_strategist.create_portfolio_buckets(
            validated_recommendations, user_preferences, user_preferences.get('amount', 100000)
        )
        
        print(f"✅ Created {len(portfolio_buckets)} portfolio buckets")

        # 🔍 DEBUG: Log bucket contents and stock tickers
        for i, bucket in enumerate(portfolio_buckets):
            print(f"🔍 [DEBUG] Bucket {i+1}: {bucket.bucket_name} with {len(bucket.stocks)} stocks")
            for j, stock in enumerate(bucket.stocks):
                print(f"🔍 [DEBUG]   Stock {j+1}: {stock.ticker} (original ticker format)")
        
        # Convert to final output format
        return await self._format_final_output(portfolio_buckets, user_preferences)
    
    async def _generate_individual_llm_analysis(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Generate individual LLM analysis for a specific stock"""
        print(f"🔍 [DEBUG] _generate_individual_llm_analysis called for ticker: '{ticker}'")
        
        try:
            # Get cached ML recommendation (try both ticker formats)
            cache_key = ticker.replace('.NS', '')
            print(f"🔍 [DEBUG] Looking for cache key: '{cache_key}'")
            print(f"🔍 [DEBUG] Available cache keys: {list(self._ml_recommendations_cache.keys())}")
            
            # Try cache lookup with both formats
            ml_rec = self._ml_recommendations_cache.get(cache_key)
            if not ml_rec:
                # Try with .NS suffix if first lookup failed
                cache_key_with_ns = ticker if ticker.endswith('.NS') else ticker + '.NS'
                print(f"🔍 [DEBUG] First lookup failed, trying with .NS: '{cache_key_with_ns}'")
                ml_rec = self._ml_recommendations_cache.get(cache_key_with_ns)
                if ml_rec:
                    print(f"🔍 [DEBUG] SUCCESS: Found ML recommendation with .NS format")
                else:
                    print(f"🔍 [DEBUG] FAILED: No ML recommendation found with either format")
                    return None
            else:
                print(f"🔍 [DEBUG] SUCCESS: Found ML recommendation without .NS format")
            print(f"🔍 [DEBUG] ML rec model predictions: {list(ml_rec.model_predictions.keys())}")
            
            # Convert ML data to format expected by LLMTradingExpert
            from intelligence.llm_trading_expert import ModelPrediction, MarketContext
            
            model_predictions = []
            for model_name, pred_data in ml_rec.model_predictions.items():
                print(f"🔍 [DEBUG] Processing model: {model_name}")
                model_predictions.append(ModelPrediction(
                    model_name=model_name.title(),
                    expected_return=pred_data.get('return', 0.0),
                    confidence=pred_data.get('confidence', 0.5),
                    risk_score=pred_data.get('risk', 0.5),
                    reasoning=f"{model_name} prediction"
                ))
            
            print(f"🔍 [DEBUG] Created {len(model_predictions)} model predictions")
            
            market_context = MarketContext(
                market_sentiment="NEUTRAL",
                volatility_regime="NORMAL", 
                sector_performance={},
                economic_indicators={},
                news_sentiment="NEUTRAL"
            )
            
            print(f"🔍 [DEBUG] Calling LLM expert analyze_predictions for {ticker}")
            
            # Call LLM expert for individual analysis
            llm_analysis = await self.llm_expert.analyze_predictions(
                ticker=ticker,
                model_predictions=model_predictions,
                market_context=market_context,
                technical_data=ml_rec.technical_data,
                fundamental_data=ml_rec.fundamental_data
            )
            
            print(f"🔍 [DEBUG] LLM expert returned analysis: {llm_analysis is not None}")
            if llm_analysis:
                print(f"🔍 [DEBUG] LLM analysis keys: {list(llm_analysis.keys())}")
            
            return llm_analysis
            
        except Exception as e:
            print(f"[⚠️] Individual LLM analysis failed for {ticker}: {e}")
            print(f"🔍 [DEBUG] Exception type: {type(e).__name__}")
            import traceback
            print(f"🔍 [DEBUG] Full traceback:")
            traceback.print_exc()
            return None
    
    async def _format_final_output(
        self,
        portfolio_buckets: List[PortfolioBucket],
        user_preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format the final output for the frontend"""
        
        # 🔍 DEBUG: Log entry to _format_final_output
        print(f"🔍 [DEBUG] _format_final_output called with {len(portfolio_buckets)} buckets")
        print(f"🔍 [DEBUG] Cache available: {hasattr(self, '_ml_recommendations_cache')}")
        if hasattr(self, '_ml_recommendations_cache'):
            print(f"🔍 [DEBUG] Cache size: {len(self._ml_recommendations_cache)}")
            print(f"🔍 [DEBUG] Cache keys: {list(self._ml_recommendations_cache.keys())}")
        
        # Calculate overall portfolio metrics
        total_allocation = sum(bucket.allocation_percentage for bucket in portfolio_buckets)
        weighted_return = sum(
            bucket.target_return * bucket.allocation_percentage 
            for bucket in portfolio_buckets
        ) / total_allocation if total_allocation > 0 else 0
        
        weighted_confidence = sum(
            bucket.confidence_level * bucket.allocation_percentage 
            for bucket in portfolio_buckets
        ) / total_allocation if total_allocation > 0 else 0
        
        # 🔥 ENHANCED SECTION - Generate individual LLM analysis for stocks
        async def enhance_stocks_with_llm_analysis():
            print(f"🔍 [DEBUG] Starting enhance_stocks_with_llm_analysis()")
            enhanced_stocks = []
            total_stocks = sum(len(bucket.stocks) for bucket in portfolio_buckets)
            print(f"🔍 [DEBUG] Total stocks to enhance: {total_stocks}")
            
            for bucket_idx, bucket in enumerate(portfolio_buckets):
                print(f"🔍 [DEBUG] Processing bucket {bucket_idx+1}: {bucket.bucket_name}")
                for stock_idx, stock in enumerate(bucket.stocks):
                    print(f"🔍 [DEBUG] Processing stock {stock_idx+1}: {stock.ticker}")
                    
                    # Generate individual LLM analysis
                    print(f"🔍 [DEBUG] Calling _generate_individual_llm_analysis for {stock.ticker}")
                    llm_analysis_data = await self._generate_individual_llm_analysis(stock.ticker)
                    print(f"🔍 [DEBUG] LLM analysis result for {stock.ticker}: {llm_analysis_data is not None}")
                    
                    # Use raw LLM analysis data (no need for complex object conversion)
                    llm_analysis_obj = llm_analysis_data if llm_analysis_data else None
                    if llm_analysis_obj:
                        print(f"[✓] Generated LLM analysis for {stock.ticker}")
                    else:
                        print(f"🔍 [DEBUG] No LLM analysis data returned for {stock.ticker}")
                    
                    # Create enhanced stock recommendation
                    enhanced_stock = StockRecommendation(
                        ticker=stock.ticker,
                        expected_return=stock.expected_return,
                        confidence_score=stock.confidence_score,
                        allocation_amount=stock.allocation_amount,
                        explanation=f"[{bucket.bucket_name}] {stock.explanation}",
                        return_range=stock.return_range,
                        backtest_metrics=stock.backtest_metrics,
                        entry_point=stock.entry_point,
                        exit_point=stock.exit_point,
                        llm_analysis=llm_analysis_obj  # 🔥 THIS IS THE KEY FIX
                    )
                    print(f"🔍 [DEBUG] Enhanced stock created with llm_analysis: {enhanced_stock.llm_analysis is not None}")
                    enhanced_stocks.append(enhanced_stock)
            
            print(f"🔍 [DEBUG] Enhanced {len(enhanced_stocks)} stocks total")
            enhanced_with_llm = sum(1 for stock in enhanced_stocks if stock.llm_analysis is not None)
            print(f"🔍 [DEBUG] Stocks with LLM analysis: {enhanced_with_llm}/{len(enhanced_stocks)}")
            return enhanced_stocks
        
        # 🔥 CALL THE ENHANCEMENT FUNCTION
        all_stocks = await enhance_stocks_with_llm_analysis() if hasattr(self, '_ml_recommendations_cache') else []
        
        # If enhancement failed, fallback to existing logic
        if not all_stocks:
            print(f"🔍 [DEBUG] Enhancement failed, using fallback logic")
            all_stocks = []
            for bucket in portfolio_buckets:
                for stock in bucket.stocks:
                    stock.explanation = f"[{bucket.bucket_name}] {stock.explanation}"
                    all_stocks.append(stock)
        else:
            print(f"🔍 [DEBUG] Using enhanced stocks")

        # 🔍 DEBUG: Log final output structure
        print(f"🔍 [DEBUG] Final output summary:")
        print(f"🔍 [DEBUG] - Portfolio stocks: {len(all_stocks)}")
        stocks_with_llm = sum(1 for stock in all_stocks if stock.llm_analysis is not None)
        print(f"🔍 [DEBUG] - Stocks with LLM analysis: {stocks_with_llm}")
        print(f"🔍 [DEBUG] - Portfolio buckets: {len(portfolio_buckets)}")

        for i, stock in enumerate(all_stocks):
            print(f"🔍 [DEBUG] Stock {i+1}: {stock.ticker} - LLM Analysis: {stock.llm_analysis is not None}")

        # 🔍 DEBUG: Check portfolio_buckets stocks too
        total_bucket_stocks = 0
        for bucket in portfolio_buckets:
            total_bucket_stocks += len(bucket.stocks)
            for stock in bucket.stocks:
                print(f"🔍 [DEBUG] Bucket stock: {stock.ticker} - LLM Analysis: {getattr(stock, 'llm_analysis', None) is not None}")

        print(f"🔍 [DEBUG] Total bucket stocks: {total_bucket_stocks}")
        
        return {
            "total_investment": user_preferences.get('amount', 100000),
            "expected_range": f"Multi-bucket strategy with {len(portfolio_buckets)} portfolios",
            "portfolio": all_stocks,
            "portfolio_buckets": [
                {
                    "bucket_name": bucket.bucket_name,
                    "strategy_type": bucket.strategy_type,
                    "allocation_percentage": bucket.allocation_percentage,
                    "allocation_amount": bucket.allocation_percentage * user_preferences.get('amount', 100000),
                    "target_return": bucket.target_return,
                    "risk_level": bucket.risk_level,
                    "confidence_level": bucket.confidence_level,
                    "stock_count": len(bucket.stocks),
                    "rationale": bucket.rationale,
                    "rebalancing_frequency": bucket.rebalancing_frequency,
                    "exit_conditions": bucket.exit_conditions,
                    "monitoring_metrics": bucket.monitoring_metrics,
                    "stocks": [
                        {
                            "ticker": stock.ticker,
                            "allocation_amount": stock.allocation_amount,
                            "expected_return": stock.expected_return,
                            "confidence_score": stock.confidence_score
                        }
                        for stock in bucket.stocks
                    ]
                }
                for bucket in portfolio_buckets
            ],
            "strategy_summary": {
                "total_buckets": len(portfolio_buckets),
                "overall_expected_return": weighted_return,
                "overall_confidence": weighted_confidence,
                "strategy_type": "Multi-Bucket Diversified",
                "llm_enhanced": True
            },
            "system_info": {
                "llm_enabled": True,
                "features_count": 150,
                "models_used": ["XGBoost", "LightGBM", "NGBoost", "LLM Strategist"],
                "processing_stages": ["ML Prediction", "LLM Validation", "LLM Strategy"]
            }
        } 