/**
 * Extended type definitions for Portfolio Analysis Results
 * These types extend the basic backend analysis with additional detailed metrics
 */

import { PortfolioAnalysis, StockAnalysis, AllocationBreakdown } from './types'

// =============================================================================
// PERFORMANCE METRICS TYPES
// =============================================================================

export interface PerformanceMetrics {
  cagr: number // Compound Annual Growth Rate
  alpha: number // Alpha vs benchmark
  beta: number // Beta vs benchmark
  rSquared: number // R-squared correlation
  sharpeRatio: number // Sharpe ratio
  sortinoRatio: number // Sortino ratio
  volatility: number // Annualized volatility
  downsideDeviation: number // Downside deviation
  maxDrawdown: number // Maximum drawdown %
  trackingError: number // Tracking error vs benchmark
  informationRatio: number // Information ratio
  calmarRatio: number // Calmar ratio
}

export interface TimeframePerformance {
  timeframe: '1M' | '3M' | '6M' | '1Y' | 'INCEPTION'
  returns: number // Total return %
  annualizedReturn: number // Annualized return %
  metrics: PerformanceMetrics
  benchmarkReturns: number // Benchmark return for same period
  outperformance: number // Outperformance vs benchmark
}

export interface HistoricalData {
  date: string // ISO date string
  portfolioValue: number
  benchmarkValue: number
  dailyReturn: number
  cumulativeReturn: number
}

// =============================================================================
// FUNDAMENTAL ANALYSIS TYPES
// =============================================================================

export interface FundamentalMetrics {
  peRatio: number
  pegRatio: number
  debtEquityRatio: number
  roe: number // Return on Equity
  roce: number // Return on Capital Employed
  currentRatio: number
  quickRatio: number
  grossMargin: number
  operatingMargin: number
  netMargin: number
  bookValue: number
  priceToBook: number
  dividendYield: number
  payoutRatio: number
}

export interface QuarterlyGrowth {
  quarter: string // Q1 2024
  revenue: number
  revenueGrowthYoY: number
  pat: number // Profit After Tax
  patGrowthYoY: number
  ebitda: number
  ebitdaMargin: number
  eps: number
  epsGrowthYoY: number
}

export interface PromoterData {
  currentHolding: number // Current promoter holding %
  pledgedShares: number // % of shares pledged
  holdingTrend: Array<{
    date: string
    holding: number
  }>
}

export interface CashFlowAnalysis {
  operatingCashFlow: number
  freeCashFlow: number
  fcfYield: number // Free cash flow yield
  cashFlowTrend: Array<{
    quarter: string
    operatingCF: number
    freeCF: number
  }>
}

// =============================================================================
// TECHNICAL ANALYSIS TYPES
// =============================================================================

export interface TechnicalIndicators {
  rsi: number // Relative Strength Index
  sma50: number // 50-day Simple Moving Average
  sma200: number // 200-day Simple Moving Average
  ema20: number // 20-day Exponential Moving Average
  macd: {
    macd: number
    signal: number
    histogram: number
  }
  bollingerBands: {
    upper: number
    middle: number
    lower: number
  }
  volumeAverage: number // 30-day average volume
  pricePattern: 'BULLISH' | 'BEARISH' | 'NEUTRAL' | 'BREAKOUT' | 'BREAKDOWN'
  support: number
  resistance: number
}

export interface BrokerTargets {
  consensusRating: 'BUY' | 'HOLD' | 'SELL'
  averageTarget: number
  highTarget: number
  lowTarget: number
  upside: number // % upside to average target
  totalBrokers: number
  buyRatings: number
  holdRatings: number
  sellRatings: number
  lastUpdated: string
}

// =============================================================================
// ENHANCED STOCK ANALYSIS TYPES
// =============================================================================

export interface StockAnalysisDetailed extends Omit<StockAnalysis, 'targetPrice' | 'priceChange'> {
  // Basic info
  sector: string
  marketCap: number
  currentPrice: number
  entryPrice: number
  quantity: number
  investedAmount: number
  currentValue: number
  absoluteReturn: number
  absoluteReturnPct: number
  
  // Performance metrics
  performance: {
    '1M': number
    '3M': number
    '6M': number
    '1Y': number
    '3Y': number
    rolling1Y: number
    rolling3Y: number
  }
  
  // Risk metrics
  volatility: number
  maxDrawdown: number
  sharpeRatio: number
  sortinoRatio: number
  beta: number
  
  // Projections
  projections: {
    quantile10: number // 10th percentile projection
    quantile50: number // 50th percentile projection
    quantile90: number // 90th percentile projection
    confidenceScore: 'HIGH' | 'MEDIUM' | 'LOW'
  }
  
  // Fundamental analysis
  fundamentals: FundamentalMetrics
  quarterlyGrowth: QuarterlyGrowth[]
  promoterData: PromoterData
  cashFlow: CashFlowAnalysis
  
  // Technical analysis
  technical: TechnicalIndicators
  brokerTargets: BrokerTargets
  
  // Comparative analysis
  sectorComparison: {
    peVsSector: number // PE ratio vs sector average
    roeVsSector: number // ROE vs sector average
    growthVsSector: number // Growth vs sector average
    volatilityVsSector: number // Volatility vs sector
  }
  
  // Red flags
  redFlags: Array<{
    type: 'FINANCIAL' | 'GOVERNANCE' | 'MARKET' | 'REGULATORY'
    severity: 'HIGH' | 'MEDIUM' | 'LOW'
    message: string
    impact: string
  }>
  
  // AI commentary
  aiInsight: string
  businessStory: string
}

// =============================================================================
// ENHANCED ALLOCATION TYPES
// =============================================================================

export interface AllocationBreakdownDetailed extends AllocationBreakdown {
  marketCap: {
    largeCap: number
    midCap: number
    smallCap: number
  }
  geography: {
    domestic: number
    international: number
  }
  theme: {
    growth: number
    value: number
    dividend: number
    momentum: number
  }
}

export interface ConcentrationAnalysis {
  topHoldingsPct: number // % in top 5 holdings
  sectorConcentration: Array<{
    sector: string
    allocation: number
    isOverweight: boolean
    benchmark: number
  }>
  riskFlags: Array<{
    type: 'CONCENTRATION' | 'CORRELATION' | 'EXPOSURE'
    message: string
    severity: 'HIGH' | 'MEDIUM' | 'LOW'
  }>
}

export interface CorrelationMatrix {
  stocks: string[] // Ticker symbols
  correlations: number[][] // Correlation matrix
  averageCorrelation: number
  highlyCorrelatedPairs: Array<{
    stock1: string
    stock2: string
    correlation: number
  }>
}

// =============================================================================
// HYGIENE & RATING TYPES
// =============================================================================

export interface HygieneFlags {
  pennyStocks: {
    count: number
    tickers: string[]
    impact: string
  }
  excessiveCash: {
    percentage: number
    isExcessive: boolean
    suggestion: string
  }
  smallCapOverexposure: {
    percentage: number
    isExcessive: boolean
    threshold: number
  }
  lowLiquidityStocks: {
    count: number
    tickers: string[]
    impact: string
  }
  highVolatilityStocks: {
    count: number
    tickers: string[]
    avgVolatility: number
  }
}

export interface PortfolioRatingDetailed {
  taiScore: number // 0-100 overall score
  returnQuality: number // 0-100
  riskManagement: number // 0-100
  diversification: number // 0-100
  costEfficiency: number // 0-100
  liquidityScore: number // 0-100
  concentrationRisk: number // 0-100 (higher is riskier)
  
  breakdown: {
    strengths: string[]
    weaknesses: string[]
    opportunities: string[]
    threats: string[]
  }
  
  proTips: Array<{
    category: 'ALLOCATION' | 'RISK' | 'COST' | 'TAX'
    tip: string
    impact: 'HIGH' | 'MEDIUM' | 'LOW'
    effort: 'EASY' | 'MODERATE' | 'DIFFICULT'
  }>
}

// =============================================================================
// ACTION PLAN TYPES
// =============================================================================

export interface ActionItem {
  id: string
  type: 'BUY' | 'SELL' | 'TRIM' | 'ADD' | 'REBALANCE' | 'HEDGE'
  priority: 'HIGH' | 'MEDIUM' | 'LOW'
  category: 'RETURN_OPTIMIZATION' | 'RISK_REDUCTION' | 'DIVERSIFICATION' | 'COST_REDUCTION'
  
  description: string
  reasoning: string
  
  // Impact assessment
  expectedReturn: number // Expected return improvement
  riskImpact: number // Risk reduction/increase
  effortRequired: 'EASY' | 'MODERATE' | 'DIFFICULT'
  timeframe: 'IMMEDIATE' | 'SHORT_TERM' | 'MEDIUM_TERM' | 'LONG_TERM'
  
  // Specific action details
  ticker?: string
  targetAllocation?: number
  currentAllocation?: number
  suggestedAmount?: number
  
  // Dependencies
  dependencies: string[]
  prerequisites: string[]
}

export interface ActionPlanDetailed {
  summary: {
    totalActions: number
    highPriorityActions: number
    expectedReturnImprovement: number
    riskReduction: number
  }
  
  pros: Array<{
    category: string
    achievement: string
    impact: string
  }>
  
  cons: Array<{
    category: string
    issue: string
    impact: string
    severity: 'HIGH' | 'MEDIUM' | 'LOW'
  }>
  
  improvements: Array<{
    area: string
    suggestion: string
    expectedBenefit: string
    priority: 'HIGH' | 'MEDIUM' | 'LOW'
  }>
  
  actions: ActionItem[]
}

// =============================================================================
// MAIN ENHANCED ANALYSIS TYPE
// =============================================================================

export interface PortfolioAnalysisDetailed extends Omit<PortfolioAnalysis, 'stocks' | 'allocation'> {
  // Enhanced basic info
  analysisDate: string
  portfolioName: string
  totalInvested: number
  currentValue: number
  absoluteReturn: number
  absoluteReturnPct: number
  
  // Performance analysis
  performanceMetrics: TimeframePerformance[]
  historicalData: HistoricalData[]
  benchmarkName: string
  
  // Enhanced allocation
  allocation: {
    current: AllocationBreakdownDetailed
    recommended: AllocationBreakdownDetailed
    concentration: ConcentrationAnalysis
    correlation: CorrelationMatrix
  }
  
  // Enhanced stock analysis
  stocks: StockAnalysisDetailed[]
  
  // Hygiene and rating
  hygiene: HygieneFlags
  rating: PortfolioRatingDetailed
  
  // Action plan
  actionPlan: ActionPlanDetailed
  
  // Additional insights
  marketOutlook: string
  riskWarnings: string[]
  opportunities: string[]
  
  // Export data
  generatedAt: string
  validUntil: string
}

// =============================================================================
// COMPONENT PROP TYPES
// =============================================================================

export interface ResultsComponentProps {
  data: PortfolioAnalysisDetailed
}

export interface ChartDataPoint {
  date: string
  value: number
  benchmark?: number
}

export interface MetricCardProps {
  title: string
  value: string | number
  change?: number
  trend?: 'up' | 'down' | 'neutral'
  subtitle?: string
  icon?: React.ReactNode
  color?: 'green' | 'red' | 'blue' | 'gray'
}

// =============================================================================
// PDF EXPORT TYPES
// =============================================================================

export interface PDFExportOptions {
  includeCharts: boolean
  includeDetailedStockAnalysis: boolean
  includeActionPlan: boolean
  format: 'SUMMARY' | 'DETAILED' | 'EXECUTIVE'
  branding: boolean
}

export interface PDFExportData {
  analysis: PortfolioAnalysisDetailed
  chartImages: { [key: string]: string } // Base64 encoded images
  options: PDFExportOptions
  metadata: {
    generatedAt: string
    generatedBy: string
    version: string
  }
} 