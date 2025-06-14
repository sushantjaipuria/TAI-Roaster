"""
Output Pydantic Models

This module defines Pydantic models for validating and structuring API responses:
- Prediction results with confidence intervals
- Stock recommendations with scores
- Portfolio analysis results
- Error responses and status updates

Key models:
- PredictionResult: Individual stock prediction with confidence
- StockRecommendation: Recommendation with score and reasoning
- PortfolioAnalysis: Complete portfolio analysis results
- APIResponse: Standard API response wrapper
- ErrorResponse: Standardized error response format
- InsightData: LLM-generated insights and explanations

Response features:
- Consistent response structure across all endpoints
- Type-safe response data
- Standardized error handling
- Metadata inclusion (timestamps, versions, etc.)
- Confidence scores and uncertainty quantification
- Actionable insights and recommendations

Integration:
- Used by routes/predict.py for response formatting
- Integrated with services/formatter.py
- Consumed by frontend for type-safe API calls
"""

# TODO: Define PredictionResult model
# TODO: Define StockRecommendation model
# TODO: Define PortfolioAnalysis model
# TODO: Define APIResponse wrapper model
# TODO: Define ErrorResponse model
# TODO: Define InsightData model for LLM outputs
# TODO: Add response metadata fields
# TODO: Add confidence and uncertainty fields
