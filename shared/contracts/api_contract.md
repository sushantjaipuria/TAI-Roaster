# API Contract Documentation

This document defines the API contract between the Frontend (Next.js) and Backend (FastAPI) for consistent integration.

## Base Configuration

**Backend URL**: `http://localhost:8000` (development)
**API Version**: `/api/v1`
**Content-Type**: `application/json`

## Authentication

Currently no authentication required. Future versions may include:
- API key authentication
- JWT token authentication
- OAuth integration

## Endpoints

### 1. Predict Portfolio

**POST** `/predict`

**Description**: Analyze portfolio and generate predictions/recommendations

**Request Body**:
```json
{
  "portfolio": {
    "holdings": [
      {
        "symbol": "RELIANCE.NS",
        "quantity": 100,
        "purchase_price": 2500.00,
        "purchase_date": "2024-01-15"
      }
    ],
    "name": "My Portfolio",
    "currency": "INR"
  },
  "preferences": {
    "risk_tolerance": "moderate",
    "time_horizon": "long_term",
    "include_insights": true
  }
}
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "portfolio_summary": {
      "total_value": 250000.00,
      "total_return": 12.5,
      "risk_score": 6.2
    },
    "recommendations": [
      {
        "symbol": "RELIANCE.NS",
        "recommendation": "HOLD",
        "confidence": 0.78,
        "target_price": 2650.00,
        "reasoning": "Strong fundamentals with moderate upside"
      }
    ],
    "insights": "Portfolio shows good diversification..."
  },
  "timestamp": "2024-01-20T10:30:00Z"
}
```

### 2. Upload Portfolio File

**POST** `/upload`

**Description**: Upload CSV/Excel file containing portfolio data

**Request**: `multipart/form-data`
- `file`: Portfolio file (CSV/Excel)
- `format`: File format specification (optional)

**Response**:
```json
{
  "status": "success", 
  "data": {
    "file_id": "abc123",
    "parsed_holdings": 15,
    "portfolio": { /* parsed portfolio data */ }
  }
}
```

### 3. Health Check

**GET** `/health`

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-20T10:30:00Z"
}
```

## Error Responses

All error responses follow this format:

```json
{
  "status": "error",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid stock symbol",
    "details": {
      "field": "symbol",
      "value": "INVALID"
    }
  },
  "timestamp": "2024-01-20T10:30:00Z"
}
```

## Status Codes

- `200`: Success
- `400`: Bad Request (validation errors)
- `401`: Unauthorized
- `404`: Not Found
- `422`: Unprocessable Entity
- `500`: Internal Server Error

## Data Types

### Portfolio Holding
```typescript
interface PortfolioHolding {
  symbol: string;          // Stock symbol (e.g., "RELIANCE.NS")
  quantity: number;        // Number of shares
  purchase_price: number;  // Price per share at purchase
  purchase_date: string;   // ISO date string
}
```

### Prediction Result
```typescript
interface PredictionResult {
  symbol: string;
  recommendation: "BUY" | "HOLD" | "SELL";
  confidence: number;      // 0-1 confidence score
  target_price: number;
  current_price: number;
  reasoning: string;
}
```

## Frontend Implementation

The frontend should use TypeScript interfaces matching these contracts and handle all error cases gracefully.

## Validation Rules

- Stock symbols must be valid and tradeable
- Quantities must be positive integers
- Prices must be positive numbers
- Dates must be valid ISO format
- File uploads limited to 10MB
- Maximum 100 holdings per portfolio
