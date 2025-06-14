"""
Input Pydantic Models

This module defines Pydantic models for validating incoming API requests:
- Portfolio data structure validation
- File upload validation
- Prediction request parameters

Key models:
- PortfolioHolding: Individual stock holding data
- Portfolio: Complete portfolio with list of holdings
- PredictionRequest: Parameters for prediction API calls
- FileUploadRequest: File upload metadata and validation
- UserPreferences: User settings for predictions

Validation features:
- Type checking for all fields
- Required vs optional field validation
- Range validation for numeric fields
- Format validation for dates and strings
- Custom validators for stock symbols
- File format validation

Integration:
- Used by routes/predict.py for request validation
- Integrated with services/parser.py for data processing
- Provides type safety for API endpoints
"""

# TODO: Define PortfolioHolding model
# TODO: Define Portfolio model with holdings list
# TODO: Define PredictionRequest model
# TODO: Define FileUploadRequest model
# TODO: Add field validation rules
# TODO: Add custom validators for stock symbols
# TODO: Add date and numeric range validation
# TODO: Define UserPreferences model
