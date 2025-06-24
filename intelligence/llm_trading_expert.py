# intelligence/llm_trading_expert.py

import json
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

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

@dataclass
class ModelPrediction:
    """Structure for individual model predictions"""
    model_name: str
    expected_return: float
    confidence: float
    risk_score: float
    reasoning: str

@dataclass
class MarketContext:
    """Market context information for LLM analysis"""
    market_sentiment: str
    volatility_regime: str
    sector_performance: Dict[str, float]
    economic_indicators: Dict[str, float]
    news_sentiment: str

class LLMTradingExpert:
    """LLM-powered Trading Expert for final decision making"""
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4.1-nano"):
        self.provider = provider
        self.model = model
        
        if provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ValueError("OpenAI package not available")
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self.client = openai.OpenAI(api_key=api_key)
        elif provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ValueError("Anthropic package not available")
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            self.client = Anthropic(api_key=api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    async def analyze_predictions(
        self,
        ticker: str,
        model_predictions: List[ModelPrediction],
        market_context: MarketContext,
        technical_data: Dict[str, Any],
        fundamental_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze multiple model predictions and provide expert trading decision
        """
        
        # Prepare the analysis prompt
        prompt = self._create_analysis_prompt(
            ticker, model_predictions, market_context, 
            technical_data, fundamental_data
        )
        
        # Get LLM analysis
        if self.provider == "openai":
            response = await self._get_openai_analysis(prompt)
        else:
            response = await self._get_anthropic_analysis(prompt)
        
        return self._parse_llm_response(response)
    
    def _create_analysis_prompt(
        self,
        ticker: str,
        model_predictions: List[ModelPrediction],
        market_context: MarketContext,
        technical_data: Dict[str, Any],
        fundamental_data: Dict[str, Any]
    ) -> str:
        """Create comprehensive analysis prompt for LLM"""
        
        prompt = f"""
You are an expert quantitative trader and portfolio manager with 20+ years of experience in Indian equity markets. 
Analyze the following data for {ticker} and provide a comprehensive trading recommendation.

## MODEL PREDICTIONS:
"""
        
        for pred in model_predictions:
            prompt += f"""
- {pred.model_name}: Return={pred.expected_return:.2%}, Confidence={pred.confidence:.1%}, Risk={pred.risk_score:.2f}
  Reasoning: {pred.reasoning}
"""
        
        prompt += f"""

## MARKET CONTEXT:
- Market Sentiment: {market_context.market_sentiment}
- Volatility Regime: {market_context.volatility_regime}
- News Sentiment: {market_context.news_sentiment}
- Sector Performance: {json.dumps(market_context.sector_performance, indent=2)}
- Economic Indicators: {json.dumps(market_context.economic_indicators, indent=2)}

## TECHNICAL ANALYSIS:
- Current Price: ₹{technical_data.get('current_price', 'N/A')}
- RSI (14): {technical_data.get('rsi_14', 'N/A')}
- MACD: {technical_data.get('macd', 'N/A')}
- Bollinger Band Position: {technical_data.get('bb_position', 'N/A')}
- Volume Ratio: {technical_data.get('volume_ratio', 'N/A')}
- Trend Strength: {technical_data.get('trend_strength', 'N/A')}
- Support Level: ₹{technical_data.get('support_level', 'N/A')}
- Resistance Level: ₹{technical_data.get('resistance_level', 'N/A')}

## FUNDAMENTAL DATA:
- P/E Ratio: {fundamental_data.get('pe_ratio', 'N/A')}
- P/B Ratio: {fundamental_data.get('pb_ratio', 'N/A')}
- ROE: {fundamental_data.get('roe', 'N/A')}%
- Debt-to-Equity: {fundamental_data.get('debt_to_equity', 'N/A')}
- Revenue Growth: {fundamental_data.get('revenue_growth', 'N/A')}%
- Profit Margin: {fundamental_data.get('profit_margin', 'N/A')}%

## ANALYSIS REQUIREMENTS:
Please provide a comprehensive analysis in the following JSON format:

{{
    "final_recommendation": "BUY|HOLD|SELL",
    "confidence_score": 0.85,
    "expected_return": 0.12,
    "risk_assessment": {{
        "overall_risk": "LOW|MEDIUM|HIGH",
        "risk_factors": ["factor1", "factor2"],
        "risk_mitigation": "strategy"
    }},
    "position_sizing": {{
        "recommended_allocation": 0.05,
        "max_allocation": 0.08,
        "reasoning": "explanation"
    }},
    "entry_strategy": {{
        "entry_price": 1250.0,
        "stop_loss": 1180.0,
        "take_profit": 1400.0,
        "time_horizon": "3-6 months"
    }},
    "model_consensus": {{
        "agreement_level": "HIGH|MEDIUM|LOW",
        "conflicting_signals": ["signal1", "signal2"],
        "weight_adjustments": {{"xgboost": 0.3, "ngboost": 0.25}}
    }},
    "market_timing": {{
        "entry_timing": "IMMEDIATE|WAIT|GRADUAL",
        "market_conditions": "FAVORABLE|NEUTRAL|UNFAVORABLE",
        "catalysts": ["catalyst1", "catalyst2"]
    }},
    "reasoning": {{
        "key_strengths": ["strength1", "strength2"],
        "key_concerns": ["concern1", "concern2"],
        "decisive_factors": ["factor1", "factor2"],
        "alternative_scenarios": ["scenario1", "scenario2"]
    }},
    "monitoring_points": ["point1", "point2", "point3"]
}}

## EXPERT GUIDELINES:
1. Consider model disagreement as a risk factor
2. Weight recent performance and market regime changes
3. Factor in sector rotation and macro trends
4. Apply position sizing based on conviction and risk
5. Consider liquidity and market impact
6. Account for correlation with existing positions
7. Evaluate risk-adjusted returns, not just absolute returns
8. Consider behavioral biases and market psychology
9. Factor in transaction costs and taxes
10. Provide actionable, specific recommendations

Provide your analysis as a valid JSON response only.
"""
        
        return prompt
    
    async def _get_openai_analysis(self, prompt: str) -> str:
        """Get analysis from OpenAI GPT"""
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert quantitative trader and portfolio manager. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return self._get_fallback_analysis()
    
    async def _get_anthropic_analysis(self, prompt: str) -> str:
        """Get analysis from Anthropic Claude"""
        try:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                temperature=0.1,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            print(f"Anthropic API error: {e}")
            return self._get_fallback_analysis()
    
    def _get_fallback_analysis(self) -> str:
        """Fallback analysis when LLM is unavailable"""
        return json.dumps({
            "final_recommendation": "HOLD",
            "confidence_score": 0.5,
            "expected_return": 0.05,
            "risk_assessment": {
                "overall_risk": "MEDIUM",
                "risk_factors": ["LLM unavailable", "Limited analysis"],
                "risk_mitigation": "Conservative approach"
            },
            "position_sizing": {
                "recommended_allocation": 0.02,
                "max_allocation": 0.03,
                "reasoning": "Conservative due to limited analysis"
            },
            "entry_strategy": {
                "entry_price": 0.0,
                "stop_loss": 0.0,
                "take_profit": 0.0,
                "time_horizon": "1-3 months"
            },
            "model_consensus": {
                "agreement_level": "MEDIUM",
                "conflicting_signals": [],
                "weight_adjustments": {}
            },
            "market_timing": {
                "entry_timing": "WAIT",
                "market_conditions": "NEUTRAL",
                "catalysts": []
            },
            "reasoning": {
                "key_strengths": ["Fallback mode"],
                "key_concerns": ["LLM analysis unavailable"],
                "decisive_factors": ["Conservative approach"],
                "alternative_scenarios": ["Wait for better analysis"]
            },
            "monitoring_points": ["LLM availability", "Model performance", "Market conditions"]
        })
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate LLM response"""
        try:
            # Extract JSON from response if it contains other text
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                analysis = json.loads(json_str)
                
                # Validate required fields
                required_fields = [
                    'final_recommendation', 'confidence_score', 'expected_return',
                    'risk_assessment', 'position_sizing', 'entry_strategy',
                    'reasoning'
                ]
                
                for field in required_fields:
                    if field not in analysis:
                        raise ValueError(f"Missing required field: {field}")
                
                return analysis
            else:
                raise ValueError("No valid JSON found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing LLM response: {e}")
            return json.loads(self._get_fallback_analysis())

class EnhancedModelAggregator:
    """Enhanced model aggregator with LLM integration"""
    
    def __init__(self, llm_expert: LLMTradingExpert):
        self.llm_expert = llm_expert
    
    async def aggregate_predictions(
        self,
        ticker: str,
        model_outputs: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> StockRecommendation:
        """
        Aggregate model predictions using LLM expert analysis
        """
        
        # Extract model predictions
        model_predictions = self._extract_model_predictions(model_outputs)
        
        # Get market context
        market_context = self._build_market_context(market_data)
        
        # Extract technical and fundamental data
        technical_data = self._extract_technical_data(market_data)
        fundamental_data = self._extract_fundamental_data(market_data)
        
        # Get LLM analysis
        llm_analysis = await self.llm_expert.analyze_predictions(
            ticker, model_predictions, market_context,
            technical_data, fundamental_data
        )
        
        # Convert to StockRecommendation
        return self._create_stock_recommendation(
            ticker, llm_analysis, model_outputs, market_data
        )
    
    def _extract_model_predictions(self, model_outputs: Dict[str, Any]) -> List[ModelPrediction]:
        """Extract predictions from individual models"""
        predictions = []
        
        if 'xgboost' in model_outputs:
            predictions.append(ModelPrediction(
                model_name="XGBoost",
                expected_return=model_outputs['xgboost'].get('return', 0.0),
                confidence=model_outputs['xgboost'].get('confidence', 0.5),
                risk_score=model_outputs['xgboost'].get('risk', 0.5),
                reasoning="Gradient boosting ensemble prediction"
            ))
        
        if 'ngboost' in model_outputs:
            predictions.append(ModelPrediction(
                model_name="NGBoost",
                expected_return=model_outputs['ngboost'].get('return', 0.0),
                confidence=model_outputs['ngboost'].get('confidence', 0.5),
                risk_score=model_outputs['ngboost'].get('risk', 0.5),
                reasoning="Probabilistic gradient boosting with uncertainty"
            ))
        
        if 'quantile' in model_outputs:
            predictions.append(ModelPrediction(
                model_name="Quantile Regression",
                expected_return=model_outputs['quantile'].get('return', 0.0),
                confidence=model_outputs['quantile'].get('confidence', 0.5),
                risk_score=model_outputs['quantile'].get('risk', 0.5),
                reasoning="Risk-aware quantile-based prediction"
            ))
        
        return predictions
    
    def _build_market_context(self, market_data: Dict[str, Any]) -> MarketContext:
        """Build market context from available data"""
        return MarketContext(
            market_sentiment=market_data.get('sentiment', 'NEUTRAL'),
            volatility_regime=market_data.get('volatility_regime', 'NORMAL'),
            sector_performance=market_data.get('sector_performance', {}),
            economic_indicators=market_data.get('economic_indicators', {}),
            news_sentiment=market_data.get('news_sentiment', 'NEUTRAL')
        )
    
    def _extract_technical_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract technical analysis data"""
        return {
            'current_price': market_data.get('current_price', 0.0),
            'rsi_14': market_data.get('rsi_14', 50.0),
            'macd': market_data.get('macd', 0.0),
            'bb_position': market_data.get('bb_position', 0.5),
            'volume_ratio': market_data.get('volume_ratio', 1.0),
            'trend_strength': market_data.get('trend_strength', 0.0),
            'support_level': market_data.get('support_level', 0.0),
            'resistance_level': market_data.get('resistance_level', 0.0)
        }
    
    def _extract_fundamental_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract fundamental analysis data"""
        return {
            'pe_ratio': market_data.get('pe_ratio', 15.0),
            'pb_ratio': market_data.get('pb_ratio', 2.0),
            'roe': market_data.get('roe', 15.0),
            'debt_to_equity': market_data.get('debt_to_equity', 0.5),
            'revenue_growth': market_data.get('revenue_growth', 10.0),
            'profit_margin': market_data.get('profit_margin', 10.0)
        }
    
    def _create_stock_recommendation(
        self,
        ticker: str,
        llm_analysis: Dict[str, Any],
        model_outputs: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> StockRecommendation:
        """Create final stock recommendation from LLM analysis"""
        
        entry_strategy = llm_analysis.get('entry_strategy', {})
        current_price = market_data.get('current_price', 1000.0)
        atr = market_data.get('atr', current_price * 0.02)
        
        return StockRecommendation(
            ticker=ticker.replace(".NS", ""),
            expected_return=llm_analysis.get('expected_return', 0.05),
            confidence_score=llm_analysis.get('confidence_score', 0.5),
            allocation_amount=llm_analysis.get('position_sizing', {}).get('recommended_allocation', 0.02) * 1000000,  # Assuming 1M portfolio
            explanation=self._create_explanation(llm_analysis),
            return_range=ReturnRange(
                min=llm_analysis.get('expected_return', 0.05) - 0.1,
                max=llm_analysis.get('expected_return', 0.05) + 0.1
            ),
            backtest_metrics=BacktestMetrics(
                cagr=0.12,  # Would be calculated from historical data
                sharpe_ratio=1.2,
                max_drawdown=-0.15
            ),
            entry_point=PricePoint(
                value=current_price,
                target=entry_strategy.get('entry_price', current_price),
                buffer_range=BufferRange(
                    min=current_price - atr,
                    max=current_price + atr
                )
            ),
            exit_point=PricePoint(
                value=entry_strategy.get('take_profit', current_price * 1.1),
                target=entry_strategy.get('take_profit', current_price * 1.1),
                buffer_range=BufferRange(
                    min=entry_strategy.get('stop_loss', current_price * 0.95),
                    max=entry_strategy.get('take_profit', current_price * 1.1)
                )
            )
        )
    
    def _create_explanation(self, llm_analysis: Dict[str, Any]) -> str:
        """Create human-readable explanation from LLM analysis"""
        reasoning = llm_analysis.get('reasoning', {})
        recommendation = llm_analysis.get('final_recommendation', 'HOLD')
        confidence = llm_analysis.get('confidence_score', 0.5)
        expected_return = llm_analysis.get('expected_return', 0.05)
        
        strengths = reasoning.get('key_strengths', [])
        concerns = reasoning.get('key_concerns', [])
        
        explanation = f"LLM Expert Recommendation: {recommendation} (Confidence: {confidence:.1%})\n"
        explanation += f"Expected Return: {expected_return:.2%}\n"
        
        if strengths:
            explanation += f"Key Strengths: {', '.join(strengths[:2])}\n"
        
        if concerns:
            explanation += f"Key Concerns: {', '.join(concerns[:2])}\n"
        
        return explanation 