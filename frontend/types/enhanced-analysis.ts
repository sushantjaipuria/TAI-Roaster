/**
 * Enhanced Analysis Types for TAI-Roaster Intelligence Integration
 * Matches backend enhanced analysis schemas
 */

// Enums
export enum AnalysisType {
  COMPREHENSIVE = "comprehensive",
  QUICK = "quick",
  DEEP_DIVE = "deep_dive"
}

export enum BenchmarkType {
  NIFTY50 = "NIFTY50",
  SENSEX = "SENSEX", 
  NIFTY_BANK = "NIFTY_BANK",
  CUSTOM = "CUSTOM"
}

export enum RiskLevel {
  LOW = "Low",
  MEDIUM = "Medium", 
  HIGH = "High",
  VERY_HIGH = "Very High"
}

export enum StockRecommendation {
  BUY = "Buy",
  HOLD = "Hold",
  SELL = "Sell",
  STRONG_BUY = "Strong Buy",
  STRONG_SELL = "Strong Sell"
}

// Request Types
export interface EnhancedAnalysisRequest {
  portfolio: any; // PortfolioInput from existing types
  user_profile: any; // UserProfile from existing types
  analysis_type?: AnalysisType;
  benchmark?: BenchmarkType;
  include_llm_analysis?: boolean;
}

// Response Component Types
export interface TAIScore {
  overall_score: number;
  performance_score: number;
  risk_management_score: number;
  diversification_score: number;
  ml_confidence_score: number;
  liquidity_score: number;
  cost_efficiency_score: number;
  grade: string;
  description: string;
}

export interface MLPrediction {
  ticker: string;
  xgboost_prediction: number;
  lightgbm_prediction?: number;
  catboost_prediction?: number;
  ngboost_mean: number;
  ngboost_std: number;
  ensemble_prediction: number;
  ensemble_confidence: number;
}

export interface PerformanceMetrics {
  total_return: number;
  annualized_return: number;
  volatility: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown: number;
  beta: number;
  alpha: number;
  var_95?: number;
  expected_shortfall?: number;
}

export interface AllocationBreakdown {
  sector_allocation: Record<string, number>;
  market_cap_allocation: Record<string, number>;
  concentration_risk: number;
  diversification_ratio: number;
}

export interface EnhancedStock {
  ticker: string;
  company_name?: string;
  quantity: number;
  current_price: number;
  investment_amount: number;
  current_value: number;
  weight: number;
  
  // ML insights
  ml_prediction: number;
  confidence_score: number;
  recommendation: StockRecommendation;
  
  // Risk metrics
  volatility?: number;
  beta?: number;
  
  // Performance
  returns_1d?: number;
  returns_1w?: number;
  returns_1m?: number;
  returns_3m?: number;
}

export interface ActionPlan {
  immediate_actions: string[];
  short_term_goals: string[];
  long_term_strategy: string[];
  rebalancing_suggestions: string[];
}

export interface RiskWarning {
  severity: string;
  category: string;
  message: string;
  recommendation: string;
}

export interface Opportunity {
  category: string;
  description: string;
  potential_impact: string;
  action_required: string;
}

// Main Response Type
export interface EnhancedAnalysisResponse {
  // Core Analysis Results
  overall_score: number;
  risk_level: RiskLevel;
  analysis_date: string;
  portfolio_name: string;
  
  // Financial Summary
  total_invested: number;
  current_value: number;
  absolute_return: number;
  absolute_return_pct: number;
  
  // Enhanced Components
  tai_scores: TAIScore;
  ml_predictions: MLPrediction[];
  performance_metrics: PerformanceMetrics;
  allocation: AllocationBreakdown;
  stocks: EnhancedStock[];
  
  // Recommendations and Insights
  action_plan: ActionPlan;
  recommendations: string[];
  risk_warnings: RiskWarning[];
  opportunities: Opportunity[];
  
  // Hygiene and Rating
  hygiene: Record<string, any>;
  rating: Record<string, any>;
  
  // Analysis Metadata
  analysis_type: AnalysisType;
  benchmark_used: BenchmarkType;
  model_version: string;
  processing_time?: number;
}

// Status and Progress Types
export interface AnalysisStatus {
  analysis_id: string;
  status: string;
  progress: number;
  message: string;
  estimated_completion?: string;
}

export interface AnalysisError {
  error_code: string;
  error_message: string;
  error_details?: Record<string, any>;
  suggested_action?: string;
} 