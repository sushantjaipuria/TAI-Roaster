# Backend & Intelligence Team Requirements
## WHOOP-Inspired Portfolio Dashboard Enhancement

### 📋 Overview

This document outlines the detailed requirements for enhancing the TAI Roaster backend and intelligence systems to support the new WHOOP-inspired portfolio dashboard. The frontend has been completely redesigned with modern UX patterns and now requires additional data endpoints and intelligence features.

---

## 🎯 Backend Team Requirements

### 🚀 **Priority 1: Critical Data Enhancements**

#### 1.1 Real-Time Stock Prices
**Current Issue**: Frontend shows placeholder values for current stock prices
**Required**: Add current market price fetching for portfolio holdings

```python
# Add to stock analysis response
{
    "ticker": "RELIANCE",
    "companyName": "Reliance Industries Limited",
    "sector": "Oil Gas & Consumable Fuels",
    "currentPrice": 2456.75,          # ← ADD THIS
    "entryPrice": 2200.00,            # ← ADD THIS (rename from avg_price)
    "avgPrice": 2200.00,              # Keep for compatibility
    "quantity": 100,
    "marketCap": "Large",             # ← ADD THIS
    "investmentAmount": 220000,
    "currentValue": 245675,           # ← ADD THIS (currentPrice * quantity)
    "absoluteReturn": 25675,          # ← ADD THIS (currentValue - investmentAmount)
    "absoluteReturnPct": 11.67        # ← ADD THIS ((currentValue - investmentAmount) / investmentAmount * 100)
}
```

**Implementation Steps**:
1. Enhance `services/market_data_service.py` to fetch real-time prices
2. Add price caching mechanism (5-minute intervals during market hours)
3. Update portfolio analysis to include calculated returns
4. Add error handling for price fetch failures

#### 1.2 Portfolio Timeline Data
**Current Issue**: No historical portfolio value tracking
**Required**: New endpoint for portfolio value history

```python
# New endpoint: GET /api/analysis/{analysis_id}/timeline
{
    "historicalValues": [
        {
            "date": "2024-01-01",
            "portfolioValue": 1500000,
            "totalInvested": 1400000,
            "dailyReturn": 0.5,
            "cumulativeReturn": 7.14
        }
        # ... more historical data
    ],
    "benchmarkValues": [
        {
            "date": "2024-01-01", 
            "benchmarkValue": 1485000,
            "benchmarkReturn": 6.07
        }
        # ... corresponding benchmark data
    ]
}
```

**Implementation Steps**:
1. Create new database table: `portfolio_timeline`
2. Add background job to calculate daily portfolio values
3. Implement endpoint in `api/endpoints/analysis.py`
4. Add data backfilling for existing portfolios

#### 1.3 Enhanced Risk Metrics
**Current Issue**: Limited risk analysis data
**Required**: Comprehensive risk metrics for each timeframe

```python
# Enhance performanceMetrics in analysis response
{
    "timeframe": "1Y",
    "returns": 12.5,
    "annualizedReturn": 12.5,
    "benchmarkReturns": 10.2,
    "outperformance": 2.3,
    "metrics": {
        "volatility": 18.5,           # ← ADD THIS
        "maxDrawdown": -12.3,         # ← ADD THIS  
        "sharpeRatio": 1.2,           # ← ADD THIS
        "sortinoRatio": 1.45,         # ← ADD THIS
        "beta": 0.85,                 # ← ADD THIS
        "alpha": 2.8,                 # ← ADD THIS
        "rSquared": 0.76,             # ← ADD THIS
        "trackingError": 4.2,         # ← ADD THIS
        "informationRatio": 0.55      # ← ADD THIS
    }
}
```

**Implementation Steps**:
1. Add risk calculation functions to `services/intelligence_service.py`
2. Integrate with existing performance analysis
3. Add benchmark comparison logic
4. Ensure calculations work across all timeframes

### 📈 **Priority 2: Enhanced Features**

#### 2.1 Market Commentary & Outlook
**Current Issue**: No contextual market insights
**Required**: AI-generated market commentary and outlook

```python
# Add to main analysis response
{
    "marketOutlook": "Current market conditions favor large-cap technology stocks due to strong Q3 earnings and AI sector momentum. However, rising interest rates pose risks to growth stocks...",
    "riskWarnings": [
        "High inflation may impact interest-sensitive sectors",
        "Geopolitical tensions could affect global supply chains",
        "Banking sector facing NPA concerns"
    ],
    "opportunities": [
        "Defensive sectors showing strength amid market volatility",
        "Export-oriented companies benefiting from rupee depreciation",
        "ESG-focused companies gaining institutional interest"
    ],
    "marketSentiment": "CAUTIOUSLY_OPTIMISTIC", # BULLISH | NEUTRAL | BEARISH | CAUTIOUSLY_OPTIMISTIC | CAUTIOUSLY_PESSIMISTIC
    "lastUpdated": "2024-06-24T19:11:21Z"
}
```

**Implementation Steps**:
1. Create `services/market_commentary_service.py`
2. Integrate with LLM for context-aware commentary generation
3. Add market sentiment analysis
4. Implement caching (update every 4 hours)

#### 2.2 Sector Performance Analysis
**Current Issue**: Limited sector-level insights
**Required**: Comprehensive sector performance data

```python
# New endpoint: GET /api/market/sector-performance
{
    "sectors": [
        {
            "name": "Technology",
            "performance1M": 5.2,
            "performance3M": 12.1,
            "performance6M": 18.7,
            "performance1Y": 24.3,
            "benchmark": 8.5,
            "outperformance": 15.8,
            "volatility": 22.1,
            "marketCap": 2850000000000,  # Total market cap
            "topStocks": ["TCS", "INFY", "WIPRO"],
            "trend": "BULLISH",          # BULLISH | BEARISH | NEUTRAL
            "outlook": "Positive momentum expected to continue with AI adoption"
        }
        # ... other sectors
    ],
    "lastUpdated": "2024-06-24T19:11:21Z"
}
```

**Implementation Steps**:
1. Create sector mapping database table
2. Implement sector performance calculations  
3. Add sector trend analysis
4. Create automated sector report generation

#### 2.3 Correlation Analysis
**Current Issue**: No portfolio correlation insights
**Required**: Stock correlation analysis for risk assessment

```python
# Add to allocation.concentration in analysis response
{
    "correlation": {
        "averageCorrelation": 0.65,
        "portfolioCorrelation": 0.72,
        "marketCorrelation": 0.68,
        "highlyCorrelatedPairs": [
            {
                "stock1": "TCS",
                "stock2": "INFY", 
                "correlation": 0.85,
                "riskLevel": "HIGH"    # HIGH | MEDIUM | LOW
            }
        ],
        "correlationMatrix": [
            ["TCS", "INFY", "WIPRO"],
            [1.0, 0.85, 0.72],
            [0.85, 1.0, 0.68],
            [0.72, 0.68, 1.0]
        ],
        "diversificationScore": 78,    # 0-100 (higher = better diversified)
        "riskFlags": [
            {
                "type": "HIGH_CORRELATION",
                "message": "TCS and INFY show high correlation (0.85)",
                "severity": "MEDIUM",
                "impact": "Reduces portfolio diversification benefits"
            }
        ]
    }
}
```

**Implementation Steps**:
1. Add correlation calculation functions
2. Implement rolling correlation analysis
3. Add correlation-based risk flagging
4. Include in portfolio optimization recommendations

### 🎯 **Priority 3: Advanced Analytics**

#### 3.1 Enhanced Stock Analysis
**Current Issue**: Limited individual stock insights
**Required**: Comprehensive per-stock analysis

```python
# Enhance individual stock data in analysis response
{
    "ticker": "RELIANCE",
    "analysis": {
        "fundamentalScore": 78,        # ← ADD THIS (0-100)
        "technicalScore": 65,          # ← ADD THIS (0-100)
        "momentumScore": 82,           # ← ADD THIS (0-100)
        "valueScore": 71,              # ← ADD THIS (0-100)
        "qualityScore": 89,            # ← ADD THIS (0-100)
        "sentimentScore": 76,          # ← ADD THIS (0-100)
        "overallScore": 77,            # ← ADD THIS (weighted average)
        
        "keyMetrics": {
            "peRatio": 15.2,
            "pegRatio": 1.1,
            "pbRatio": 2.3,
            "debtEquityRatio": 0.45,
            "roe": 18.5,
            "roce": 16.2,
            "marketCap": 1850000000000
        },
        
        "projections": {
            "targetPrice": 2750,
            "upside": 11.9,
            "confidence": "MEDIUM",      # HIGH | MEDIUM | LOW
            "timeHorizon": "12M",
            "priceRange": {
                "low": 2200,
                "high": 3000
            }
        },
        
        "risks": [
            {
                "type": "REGULATORY",
                "description": "Potential changes in petroleum policy",
                "severity": "MEDIUM",
                "probability": "LOW"
            }
        ],
        
        "catalysts": [
            {
                "type": "EARNINGS",
                "description": "Q4 results expected to show strong petrochemicals growth",
                "timeline": "Next 30 days",
                "impact": "POSITIVE"
            }
        ]
    }
}
```

**Implementation Steps**:
1. Enhance `services/intelligence_service.py` with multi-factor scoring
2. Add fundamental analysis calculations
3. Implement technical analysis indicators
4. Add sentiment analysis from news/social media

#### 3.2 Portfolio Optimization Suggestions
**Current Issue**: Generic action plan recommendations
**Required**: Specific, actionable optimization suggestions

```python
# Enhance actionPlan in analysis response
{
    "optimizations": [
        {
            "id": "opt_001",
            "type": "REBALANCE",
            "priority": "HIGH",
            "category": "RISK_REDUCTION",
            "title": "Reduce Technology Sector Overweight",
            "description": "Technology allocation at 45% vs recommended 30%",
            "currentAllocation": 45.0,
            "targetAllocation": 30.0,
            "actions": [
                {
                    "action": "SELL",
                    "ticker": "TCS",
                    "amount": 150000,
                    "reason": "Partial profit booking in overperforming tech stock"
                },
                {
                    "action": "BUY", 
                    "sector": "Healthcare",
                    "amount": 150000,
                    "reason": "Add defensive sector exposure"
                }
            ],
            "expectedImpact": {
                "riskReduction": 15,     # Percentage reduction in portfolio risk
                "returnImpact": -2,      # Expected return impact (can be negative)
                "diversificationGain": 12 # Improvement in diversification score
            },
            "timeframe": "1-2 weeks",
            "effort": "MODERATE",
            "costBenefit": {
                "implementationCost": 5000,  # Transaction costs
                "expectedBenefit": 45000     # Annual benefit in rupees
            }
        }
    ],
    
    "taxOptimization": {
        "ltcgLiability": 15000,
        "stcgLiability": 8000,
        "harvestingOpportunities": [
            {
                "ticker": "WIPRO",
                "action": "SELL_AND_REBUY",
                "currentLoss": -25000,
                "taxSaving": 7500,
                "recommendation": "Book loss for tax harvesting, rebuy after 1 day"
            }
        ]
    }
}
```

**Implementation Steps**:
1. Add portfolio optimization engine
2. Implement tax optimization calculations
3. Add transaction cost analysis
4. Create Monte Carlo simulation for optimization validation

---

## 🧠 Intelligence Team Requirements

### 🚀 **Priority 1: Model Enhancements**

#### 1.1 Fix Preprocessing Pipeline
**Current Issue**: "The number of columns in this dataset is different..." errors
**Required**: Fix feature consistency in ML pipeline

**Error Analysis**:
```
[ERROR] Preprocessing failed: The number of columns in this dataset is different from the one used to fit this transformer
```

**Root Cause**: Feature engineering generates different numbers of features for different stocks

**Solution Steps**:
1. **Audit Feature Engineering Pipeline**:
   ```python
   # In intelligence/training/feature_builder.py
   def ensure_consistent_features(df, expected_features=50):
       """Ensure all stocks generate exactly the same number of features"""
       # Add missing features with default values
       # Remove extra features
       # Maintain feature order consistency
       pass
   ```

2. **Standardize Feature Names**:
   ```python
   # Create feature registry
   REQUIRED_FEATURES = [
       'Open', 'High', 'Low', 'Close', 'Volume',
       'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200',
       'EMA_12', 'EMA_26', 'RSI_14', 'MACD', 'MACD_Signal',
       # ... define all 50 required features
   ]
   ```

3. **Add Feature Validation**:
   ```python
   def validate_features(df, required_features):
       """Validate feature consistency before model inference"""
       assert df.shape[1] == len(required_features)
       assert list(df.columns) == required_features
   ```

#### 1.2 Enhanced Prediction Models
**Current Issue**: Limited prediction confidence and ranges
**Required**: Multi-model ensemble with confidence intervals

```python
# Enhance prediction output
{
    "ticker": "RELIANCE",
    "predictions": {
        "models": {
            "xgboost": {
                "prediction": 2450.75,
                "confidence": 0.82,
                "weight": 0.25
            },
            "lightgbm": {
                "prediction": 2478.50,
                "confidence": 0.79,
                "weight": 0.25
            },
            "ngboost": {
                "prediction": 2465.20,
                "confidence": 0.85,
                "weight": 0.30
            },
            "neural_network": {
                "prediction": 2441.30,
                "confidence": 0.76,
                "weight": 0.20
            }
        },
        "ensemble": {
            "prediction": 2458.94,
            "confidence": "HIGH",           # HIGH | MEDIUM | LOW
            "confidenceScore": 0.81,
            "predictionRange": {
                "low": 2200.50,            # 10th percentile
                "median": 2458.94,         # 50th percentile  
                "high": 2750.80            # 90th percentile
            },
            "riskMetrics": {
                "volatility": 18.5,
                "maxDrawdown": -12.3,
                "valueAtRisk95": -156.78   # 95% VaR
            }
        }
    }
}
```

**Implementation Steps**:
1. Fix preprocessing consistency across all models
2. Implement ensemble prediction with weights
3. Add confidence interval calculations
4. Add model performance tracking and adaptive weighting

#### 1.3 Sentiment Analysis Integration
**Current Issue**: No sentiment analysis in stock evaluation
**Required**: News and social media sentiment analysis

```python
# Add to stock analysis
{
    "ticker": "RELIANCE", 
    "sentiment": {
        "overall": "POSITIVE",           # POSITIVE | NEGATIVE | NEUTRAL
        "score": 0.73,                   # -1 to +1 scale
        "sources": {
            "news": {
                "score": 0.68,
                "articlesAnalyzed": 45,
                "lastUpdated": "2024-06-24T18:00:00Z"
            },
            "socialMedia": {
                "score": 0.78,
                "postsAnalyzed": 1250,
                "lastUpdated": "2024-06-24T19:00:00Z"
            },
            "analystReports": {
                "score": 0.75,
                "reportsAnalyzed": 12,
                "lastUpdated": "2024-06-24T10:00:00Z"
            }
        },
        "keyTopics": [
            {
                "topic": "Quarterly Results",
                "sentiment": "POSITIVE",
                "weight": 0.35
            },
            {
                "topic": "Oil Prices",
                "sentiment": "NEUTRAL", 
                "weight": 0.25
            }
        ]
    }
}
```

**Implementation Steps**:
1. Integrate news API (Economic Times, Business Standard, etc.)
2. Add social media sentiment analysis (Twitter, Reddit)
3. Implement topic modeling for key themes
4. Add sentiment weighting to prediction models

### 📈 **Priority 2: Advanced Analytics**

#### 2.1 Portfolio Risk Analytics
**Current Issue**: Limited portfolio-level risk analysis
**Required**: Comprehensive risk modeling system

```python
# New module: intelligence/portfolio_risk_analyzer.py
class PortfolioRiskAnalyzer:
    def analyze_portfolio_risk(self, portfolio):
        return {
            "riskMetrics": {
                "portfolioVaR95": -125000,      # 95% Value at Risk
                "portfolioVaR99": -185000,      # 99% Value at Risk
                "expectedShortfall": -205000,   # Expected loss beyond VaR
                "portfolioBeta": 0.87,
                "portfolioVolatility": 16.8,
                "sharpeRatio": 1.24,
                "sortinoRatio": 1.67,
                "calmarRatio": 0.89
            },
            "stressTests": [
                {
                    "scenario": "Market Crash 2008",
                    "expectedLoss": -385000,
                    "lossProbability": 0.05
                },
                {
                    "scenario": "COVID-19 2020",
                    "expectedLoss": -275000, 
                    "lossProbability": 0.15
                }
            ],
            "riskContribution": [
                {
                    "ticker": "RELIANCE",
                    "riskContribution": 0.28,
                    "marginalVaR": -35000
                }
                # ... for each holding
            ]
        }
```

**Implementation Steps**:
1. Implement Monte Carlo simulation for risk analysis
2. Add historical stress testing scenarios
3. Create risk contribution analysis
4. Implement dynamic hedging suggestions

#### 2.2 Market Regime Detection
**Current Issue**: No market regime awareness in predictions
**Required**: Market regime detection and regime-specific models

```python
# Add market regime detection
{
    "marketRegime": {
        "current": "BULLISH_TRENDING",     # BULLISH_TRENDING | BEARISH_TRENDING | SIDEWAYS | HIGH_VOLATILITY
        "confidence": 0.82,
        "duration": "45 days",
        "characteristics": {
            "volatility": "MEDIUM",
            "momentum": "POSITIVE", 
            "correlation": "INCREASING"
        },
        "regimeHistory": [
            {
                "regime": "SIDEWAYS",
                "startDate": "2024-04-01",
                "endDate": "2024-05-15",
                "duration": "44 days"
            }
        ],
        "modelAdjustments": {
            "riskAdjustment": 1.15,        # Increase risk by 15%
            "returnAdjustment": 1.08,      # Increase expected return by 8%
            "confidenceAdjustment": 0.95   # Reduce confidence by 5%
        }
    }
}
```

**Implementation Steps**:
1. Implement regime detection algorithms (HMM, LSTM)
2. Create regime-specific prediction models
3. Add regime transition probability analysis
4. Implement dynamic model weighting based on regime

#### 2.3 ESG Scoring Integration
**Current Issue**: No ESG considerations in analysis
**Required**: ESG scoring and sustainable investing insights

```python
# Add ESG analysis to stock data
{
    "ticker": "RELIANCE",
    "esg": {
        "overallScore": 72,              # 0-100 scale
        "environmental": 68,
        "social": 75, 
        "governance": 74,
        "rating": "B+",                  # A+ to D- scale
        "riskLevel": "MEDIUM",           # LOW | MEDIUM | HIGH
        "controversies": [
            {
                "type": "ENVIRONMENTAL",
                "description": "Carbon emissions disclosure concerns",
                "severity": "MEDIUM",
                "date": "2024-03-15"
            }
        ],
        "improvements": [
            {
                "area": "RENEWABLE_ENERGY",
                "progress": "Installing 20GW solar capacity by 2025",
                "impact": "POSITIVE"
            }
        ],
        "benchmarks": {
            "sectorAverage": 65,
            "nifty50Average": 69,
            "globalAverage": 71
        }
    }
}
```

**Implementation Steps**:
1. Integrate ESG data providers (Bloomberg ESG, MSCI)
2. Create ESG scoring methodology
3. Add ESG risk factor to portfolio analysis
4. Implement ESG-based portfolio suggestions

---

## 🔧 Technical Implementation Guidelines

### Database Schema Updates

#### Portfolio Timeline Table
```sql
CREATE TABLE portfolio_timeline (
    id SERIAL PRIMARY KEY,
    analysis_id VARCHAR(255) NOT NULL,
    date DATE NOT NULL,
    portfolio_value DECIMAL(15,2) NOT NULL,
    total_invested DECIMAL(15,2) NOT NULL,
    daily_return DECIMAL(8,4),
    cumulative_return DECIMAL(8,4),
    benchmark_value DECIMAL(15,2),
    benchmark_return DECIMAL(8,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (analysis_id) REFERENCES analysis_results(id)
);

CREATE INDEX idx_portfolio_timeline_analysis_date ON portfolio_timeline(analysis_id, date);
```

#### Market Data Cache Table
```sql
CREATE TABLE market_data_cache (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    volume BIGINT,
    market_cap DECIMAL(20,2),
    sector VARCHAR(100),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    data_source VARCHAR(50) NOT NULL,
    UNIQUE KEY unique_ticker_source (ticker, data_source)
);

CREATE INDEX idx_market_data_ticker_updated ON market_data_cache(ticker, last_updated);
```

### API Endpoint Structure

#### New Endpoints to Implement
```python
# 1. Portfolio Timeline
GET /api/analysis/{analysis_id}/timeline
Response: PortfolioTimelineResponse

# 2. Market Data
GET /api/market/sector-performance
Response: SectorPerformanceResponse

# 3. Real-time Prices
GET /api/market/prices?tickers=RELIANCE,TCS,INFY
Response: RealTimePricesResponse

# 4. Risk Analysis
GET /api/analysis/{analysis_id}/risk-details
Response: RiskAnalysisResponse

# 5. Market Commentary
GET /api/market/commentary
Response: MarketCommentaryResponse
```

### Error Handling & Logging

#### Required Error Handling
```python
# Add to all new endpoints
try:
    # API logic
    pass
except ExternalAPIError as e:
    logger.warning(f"External API failed: {e}")
    # Return cached data or fallback
except ValidationError as e:
    logger.error(f"Data validation failed: {e}")
    return JSONResponse(status_code=422, content={"error": "Invalid data"})
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    return JSONResponse(status_code=500, content={"error": "Internal server error"})
```

#### Logging Requirements
```python
# Add structured logging
import structlog
logger = structlog.get_logger()

# Log all data fetching operations
logger.info("Fetching market data", ticker=ticker, source=data_source)

# Log performance metrics
logger.info("Analysis completed", 
           analysis_id=analysis_id, 
           duration_ms=duration,
           holdings_count=len(portfolio))
```

---

## 📊 Testing Requirements

### Unit Tests Required

#### Backend Tests
```python
# tests/test_market_data_service.py
def test_fetch_real_time_prices():
    """Test real-time price fetching with various scenarios"""
    pass

def test_price_cache_expiry():
    """Test price cache expiration and refresh logic"""
    pass

# tests/test_portfolio_timeline.py
def test_portfolio_value_calculation():
    """Test portfolio value calculation accuracy"""
    pass

def test_timeline_data_consistency():
    """Test timeline data consistency across date ranges"""
    pass
```

#### Intelligence Tests
```python
# tests/test_feature_consistency.py
def test_feature_count_consistency():
    """Ensure all stocks generate exactly 50 features"""
    pass

def test_preprocessing_pipeline():
    """Test preprocessing pipeline doesn't fail"""
    pass

# tests/test_ensemble_predictions.py
def test_ensemble_confidence():
    """Test ensemble prediction confidence calculations"""
    pass
```

### Integration Tests
```python
# tests/test_end_to_end.py
def test_complete_analysis_pipeline():
    """Test complete analysis from portfolio input to dashboard data"""
    # Submit portfolio -> Get analysis -> Verify all dashboard data present
    pass

def test_real_time_data_flow():
    """Test real-time data flows correctly through system"""
    pass
```

### Performance Tests
```python
# tests/test_performance.py
def test_analysis_response_time():
    """Analysis should complete within 30 seconds for 20-stock portfolio"""
    pass

def test_concurrent_analysis():
    """System should handle 10 concurrent analyses"""
    pass
```

---

## 🚀 Deployment & Monitoring

### Environment Variables to Add
```bash
# Market Data APIs
DHAN_API_KEY=your_dhan_api_key
YAHOO_FINANCE_API_KEY=your_yahoo_finance_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key

# External Services
NEWS_API_KEY=your_news_api_key
TWITTER_API_KEY=your_twitter_api_key
ESG_DATA_API_KEY=your_esg_data_key

# Cache Configuration
REDIS_URL=redis://localhost:6379/0
CACHE_TTL_SECONDS=300

# Database
POSTGRES_POOL_SIZE=20
POSTGRES_MAX_OVERFLOW=10
```

### Monitoring & Alerts
```python
# Add monitoring for:
# 1. API response times
# 2. External API failures
# 3. Model prediction errors
# 4. Data quality issues
# 5. Cache hit rates

# Health check endpoints
GET /api/health/database
GET /api/health/external-apis
GET /api/health/ml-models
GET /api/health/cache
```

---

## 📅 Implementation Timeline

### Phase 1 (Week 1-2): Critical Fixes
- [ ] Fix ML preprocessing pipeline consistency
- [ ] Add real-time price fetching
- [ ] Implement basic risk metrics
- [ ] Add portfolio timeline endpoint

### Phase 2 (Week 3-4): Enhanced Features  
- [ ] Market commentary generation
- [ ] Sector performance analysis
- [ ] Correlation analysis
- [ ] Enhanced stock scoring

### Phase 3 (Week 5-6): Advanced Analytics
- [ ] Portfolio optimization engine
- [ ] Sentiment analysis integration
- [ ] Market regime detection
- [ ] ESG scoring

### Phase 4 (Week 7-8): Performance & Polish
- [ ] Performance optimization
- [ ] Comprehensive testing
- [ ] Monitoring setup
- [ ] Documentation completion

---

## 🎯 Success Metrics

### Technical Metrics
- [ ] ML preprocessing success rate: 100%
- [ ] API response time: <2 seconds average
- [ ] System uptime: >99.5%
- [ ] Data accuracy: >98%

### Business Metrics
- [ ] User engagement with new dashboard: +40%
- [ ] Time to insights: <30 seconds
- [ ] Action plan completion rate: +25%
- [ ] User satisfaction score: >4.5/5

---

## 📞 Support & Communication

### Escalation Path
1. **Technical Issues**: Backend Team Lead → CTO
2. **Data Quality**: Intelligence Team Lead → Head of AI
3. **API Dependencies**: DevOps Lead → Infrastructure Team
4. **Timeline Issues**: Project Manager → Product Owner

### Communication Channels
- **Daily Standups**: 9:00 AM IST
- **Weekly Reviews**: Fridays 4:00 PM IST  
- **Emergency Contact**: Slack #tai-roaster-urgent
- **Documentation**: Confluence space "TAI Roaster Dashboard"

---

This document serves as the single source of truth for the dashboard enhancement project. All teams should refer to this document and update it as requirements evolve. 