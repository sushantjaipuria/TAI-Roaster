import React, { useState } from 'react'
import { 
  TrendingUp, 
  TrendingDown, 
  AlertTriangle, 
  CheckCircle, 
  Star,
  Target,
  BarChart3,
  Activity,
  Brain,
  Zap,
  Eye,
  ArrowUp,
  ArrowDown,
  Minus,
  Info,
  Shield,
  DollarSign
} from 'lucide-react'

interface EnhancedStock {
  ticker: string
  company_name?: string
  sector?: string
  overall_score: number
  fundamental_score: number
  technical_score: number
  momentum_score: number
  value_score: number
  quality_score: number
  sentiment_score: number
  current_price: number
  target_price?: number
  upside_potential?: number
  recommendation: string
  confidence_level: string
  key_strengths?: string[]
  key_concerns?: string[]
  catalysts?: string[]
  risks?: string[]
  business_story?: string
  investment_thesis?: string
  beta?: number
  volatility?: number
  max_drawdown?: number
  weight: number
  enhanced_analysis_available: boolean
}

interface Props {
  enhanced_stocks: EnhancedStock[]
  portfolio_enhanced_metrics?: any
}

const ScoreCard = ({ title, score, icon: Icon, color }: { 
  title: string
  score: number
  icon: any
  color: string
}) => {
  const getScoreColor = (score: number) => {
    if (score >= 70) return 'text-green-600 bg-green-50 border-green-200'
    if (score >= 50) return 'text-blue-600 bg-blue-50 border-blue-200'
    return 'text-red-600 bg-red-50 border-red-200'
  }

  return (
    <div className={`flex items-center gap-3 p-3 rounded-lg border ${getScoreColor(score)}`}>
      <Icon className={`w-5 h-5 ${color}`} />
      <div>
        <div className="text-sm font-medium">{title}</div>
        <div className="text-lg font-bold">{score.toFixed(0)}/100</div>
      </div>
    </div>
  )
}

const RecommendationBadge = ({ recommendation, confidence }: { 
  recommendation: string
  confidence: string 
}) => {
  const getRecommendationStyle = (rec: string) => {
    switch (rec.toUpperCase()) {
      case 'STRONG_BUY':
        return 'bg-green-100 text-green-800 border-green-300'
      case 'BUY':
        return 'bg-green-50 text-green-700 border-green-200'
      case 'HOLD':
        return 'bg-blue-50 text-blue-700 border-blue-200'
      case 'SELL':
        return 'bg-red-50 text-red-700 border-red-200'
      case 'STRONG_SELL':
        return 'bg-red-100 text-red-800 border-red-300'
      default:
        return 'bg-gray-50 text-gray-700 border-gray-200'
    }
  }

  const getConfidenceStyle = (conf: string) => {
    switch (conf.toUpperCase()) {
      case 'HIGH':
        return 'bg-green-100 text-green-700'
      case 'MEDIUM':
        return 'bg-yellow-100 text-yellow-700'
      case 'LOW':
        return 'bg-red-100 text-red-700'
      default:
        return 'bg-gray-100 text-gray-700'
    }
  }

  return (
    <div className="flex items-center gap-2">
      <span className={`px-3 py-1 rounded-full text-sm font-medium border ${getRecommendationStyle(recommendation)}`}>
        {recommendation.replace('_', ' ')}
      </span>
      <span className={`px-2 py-1 rounded text-xs font-medium ${getConfidenceStyle(confidence)}`}>
        {confidence} confidence
      </span>
    </div>
  )
}

const EnhancedStockCard = ({ stock }: { stock: EnhancedStock }) => {
  const [expanded, setExpanded] = useState(false)

  const upside = stock.upside_potential || 0
  const getUpsideIcon = () => {
    if (upside > 10) return <ArrowUp className="w-4 h-4 text-green-600" />
    if (upside > 0) return <ArrowUp className="w-4 h-4 text-blue-600" />
    if (upside < -5) return <ArrowDown className="w-4 h-4 text-red-600" />
    return <Minus className="w-4 h-4 text-gray-600" />
  }

  const getOverallScoreColor = (score: number) => {
    if (score >= 75) return 'text-green-600'
    if (score >= 60) return 'text-blue-600'
    if (score >= 45) return 'text-yellow-600'
    return 'text-red-600'
  }

  return (
    <div className="bg-white rounded-xl border border-gray-200 hover:shadow-lg transition-all duration-200">
      {/* Header */}
      <div className="p-6 border-b border-gray-100">
        <div className="flex items-start justify-between mb-4">
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-2">
              <h3 className="text-xl font-bold text-gray-900">{stock.ticker}</h3>
              {stock.enhanced_analysis_available && (
                <div className="flex items-center gap-1">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span className="text-xs text-green-600 font-medium">Enhanced</span>
                </div>
              )}
            </div>
            <p className="text-sm text-gray-600 mb-1">{stock.company_name || stock.ticker}</p>
            <p className="text-xs text-gray-500">{stock.sector || 'Unknown Sector'}</p>
          </div>
          <div className="text-right">
            <div className={`text-2xl font-bold ${getOverallScoreColor(stock.overall_score)}`}>
              {stock.overall_score.toFixed(0)}
            </div>
            <div className="text-xs text-gray-500">Overall Score</div>
          </div>
        </div>

        {/* Recommendation & Target */}
        <div className="flex items-center justify-between mb-4">
          <RecommendationBadge 
            recommendation={stock.recommendation} 
            confidence={stock.confidence_level}
          />
          {stock.target_price && (
            <div className="text-right">
              <div className="flex items-center gap-1 text-sm text-gray-600">
                <Target className="w-4 h-4" />
                Target: ₹{stock.target_price.toFixed(0)}
              </div>
              <div className="flex items-center gap-1 text-sm font-medium">
                {getUpsideIcon()}
                {upside >= 0 ? '+' : ''}{upside.toFixed(1)}% upside
              </div>
            </div>
          )}
        </div>

        {/* Price & Weight Info */}
        <div className="grid grid-cols-3 gap-4 p-4 bg-gray-50 rounded-lg">
          <div className="text-center">
            <div className="text-sm text-gray-600 mb-1">Current Price</div>
            <div className="text-lg font-bold text-gray-900">₹{stock.current_price.toFixed(2)}</div>
          </div>
          <div className="text-center">
            <div className="text-sm text-gray-600 mb-1">Portfolio Weight</div>
            <div className="text-lg font-bold text-blue-600">{stock.weight.toFixed(1)}%</div>
          </div>
          <div className="text-center">
            <div className="text-sm text-gray-600 mb-1">Beta</div>
            <div className="text-lg font-bold text-gray-900">{stock.beta?.toFixed(2) || 'N/A'}</div>
          </div>
        </div>
      </div>

      {/* Multi-Factor Scores */}
      <div className="p-6 border-b border-gray-100">
        <h4 className="text-sm font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <Activity className="w-4 h-4 text-purple-600" />
          Multi-Factor Analysis
        </h4>
        <div className="grid grid-cols-2 gap-3">
          <ScoreCard 
            title="Fundamental" 
            score={stock.fundamental_score} 
            icon={BarChart3}
            color="text-blue-600"
          />
          <ScoreCard 
            title="Technical" 
            score={stock.technical_score} 
            icon={TrendingUp}
            color="text-green-600"
          />
          <ScoreCard 
            title="Momentum" 
            score={stock.momentum_score} 
            icon={Zap}
            color="text-yellow-600"
          />
          <ScoreCard 
            title="Value" 
            score={stock.value_score} 
            icon={DollarSign}
            color="text-purple-600"
          />
          <ScoreCard 
            title="Quality" 
            score={stock.quality_score} 
            icon={Shield}
            color="text-indigo-600"
          />
          <ScoreCard 
            title="Sentiment" 
            score={stock.sentiment_score} 
            icon={Brain}
            color="text-pink-600"
          />
        </div>
      </div>

      {/* Key Insights Preview */}
      <div className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h4 className="text-sm font-semibold text-gray-900 flex items-center gap-2">
            <Eye className="w-4 h-4 text-blue-600" />
            Key Insights
          </h4>
          <button
            onClick={() => setExpanded(!expanded)}
            className="text-xs text-blue-600 hover:text-blue-800 font-medium transition-colors"
          >
            {expanded ? 'Show Less' : 'Show More'}
          </button>
        </div>

        {/* Strengths & Concerns Preview */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          {stock.key_strengths && stock.key_strengths.length > 0 && (
            <div className="p-3 bg-green-50 rounded-lg border-l-4 border-green-500">
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle className="w-4 h-4 text-green-600" />
                <span className="text-sm font-medium text-green-800">Key Strengths</span>
              </div>
              <ul className="text-sm text-green-700 space-y-1">
                {stock.key_strengths.slice(0, expanded ? undefined : 2).map((strength, index) => (
                  <li key={index} className="flex items-start gap-2">
                    <span className="text-green-500 mt-1">•</span>
                    <span>{strength}</span>
                  </li>
                ))}
              </ul>
              {!expanded && stock.key_strengths.length > 2 && (
                <div className="text-xs text-green-600 mt-1">+{stock.key_strengths.length - 2} more</div>
              )}
            </div>
          )}

          {stock.key_concerns && stock.key_concerns.length > 0 && (
            <div className="p-3 bg-yellow-50 rounded-lg border-l-4 border-yellow-500">
              <div className="flex items-center gap-2 mb-2">
                <AlertTriangle className="w-4 h-4 text-yellow-600" />
                <span className="text-sm font-medium text-yellow-800">Key Concerns</span>
              </div>
              <ul className="text-sm text-yellow-700 space-y-1">
                {stock.key_concerns.slice(0, expanded ? undefined : 2).map((concern, index) => (
                  <li key={index} className="flex items-start gap-2">
                    <span className="text-yellow-500 mt-1">•</span>
                    <span>{concern}</span>
                  </li>
                ))}
              </ul>
              {!expanded && stock.key_concerns.length > 2 && (
                <div className="text-xs text-yellow-600 mt-1">+{stock.key_concerns.length - 2} more</div>
              )}
            </div>
          )}
        </div>

        {/* Expanded Content */}
        {expanded && (
          <div className="space-y-4 border-t border-gray-100 pt-4">
            {/* Investment Thesis */}
            {stock.investment_thesis && (
              <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
                <h5 className="text-sm font-semibold text-blue-900 mb-2 flex items-center gap-2">
                  <Brain className="w-4 h-4" />
                  Investment Thesis
                </h5>
                <p className="text-sm text-blue-800">{stock.investment_thesis}</p>
              </div>
            )}

            {/* Business Story */}
            {stock.business_story && (
              <div className="p-4 bg-purple-50 rounded-lg border border-purple-200">
                <h5 className="text-sm font-semibold text-purple-900 mb-2">Business Story</h5>
                <p className="text-sm text-purple-800">{stock.business_story}</p>
              </div>
            )}

            {/* Catalysts & Risks */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {stock.catalysts && stock.catalysts.length > 0 && (
                <div className="p-3 bg-green-50 rounded-lg">
                  <h5 className="text-sm font-semibold text-green-900 mb-2">Growth Catalysts</h5>
                  <ul className="text-sm text-green-800 space-y-1">
                    {stock.catalysts.map((catalyst, index) => (
                      <li key={index} className="flex items-start gap-2">
                        <Star className="w-3 h-3 text-green-500 mt-1 flex-shrink-0" />
                        <span>{catalyst}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {stock.risks && stock.risks.length > 0 && (
                <div className="p-3 bg-red-50 rounded-lg">
                  <h5 className="text-sm font-semibold text-red-900 mb-2">Risk Factors</h5>
                  <ul className="text-sm text-red-800 space-y-1">
                    {stock.risks.map((risk, index) => (
                      <li key={index} className="flex items-start gap-2">
                        <AlertTriangle className="w-3 h-3 text-red-500 mt-1 flex-shrink-0" />
                        <span>{risk}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>

            {/* Risk Metrics */}
            <div className="grid grid-cols-3 gap-4 p-4 bg-gray-50 rounded-lg">
              <div className="text-center">
                <div className="text-xs text-gray-600 mb-1">Volatility</div>
                <div className="text-sm font-bold text-gray-900">{((stock.volatility || 0) * 100).toFixed(1)}%</div>
              </div>
              <div className="text-center">
                <div className="text-xs text-gray-600 mb-1">Max Drawdown</div>
                <div className="text-sm font-bold text-red-600">{((stock.max_drawdown || 0) * 100).toFixed(1)}%</div>
              </div>
              <div className="text-center">
                <div className="text-xs text-gray-600 mb-1">Confidence</div>
                <div className="text-sm font-bold text-blue-600">{stock.confidence_level}</div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default function EnhancedStockInsightsPanel({ enhanced_stocks, portfolio_enhanced_metrics }: Props) {
  const [sortBy, setSortBy] = useState<'score' | 'upside' | 'weight'>('score')
  const [filterBy, setFilterBy] = useState<'all' | 'strong_buy' | 'buy' | 'hold'>('all')

  // Filter and sort stocks
  const filteredStocks = enhanced_stocks
    .filter(stock => {
      if (filterBy === 'all') return true
      return stock.recommendation.toLowerCase().includes(filterBy.replace('_', ''))
    })
    .sort((a, b) => {
      switch (sortBy) {
        case 'score':
          return b.overall_score - a.overall_score
        case 'upside':
          return (b.upside_potential || 0) - (a.upside_potential || 0)
        case 'weight':
          return b.weight - a.weight
        default:
          return 0
      }
    })

  if (!enhanced_stocks || enhanced_stocks.length === 0) {
    return (
      <div className="bg-white rounded-xl border border-gray-200 p-8 text-center">
        <Activity className="w-12 h-12 mx-auto mb-4 text-gray-400" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">No Enhanced Analysis Available</h3>
        <p className="text-gray-600">Enhanced individual stock analysis is not available for this portfolio.</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header with Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">Enhanced Stock Analysis</h3>
          <p className="text-sm text-gray-600">{enhanced_stocks.length} stocks with detailed analysis</p>
        </div>
        
        <div className="flex gap-3">
          {/* Sort Controls */}
          <select 
            value={sortBy} 
            onChange={(e) => setSortBy(e.target.value as any)}
            className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="score">Sort by Score</option>
            <option value="upside">Sort by Upside</option>
            <option value="weight">Sort by Weight</option>
          </select>

          {/* Filter Controls */}
          <select 
            value={filterBy} 
            onChange={(e) => setFilterBy(e.target.value as any)}
            className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="all">All Stocks</option>
            <option value="strong_buy">Strong Buy</option>
            <option value="buy">Buy</option>
            <option value="hold">Hold</option>
          </select>
        </div>
      </div>

      {/* Enhanced Stocks Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {filteredStocks.map((stock) => (
          <EnhancedStockCard key={stock.ticker} stock={stock} />
        ))}
      </div>

      {/* Portfolio Summary */}
      {portfolio_enhanced_metrics && (
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h4 className="text-lg font-semibold text-gray-900 mb-4">Portfolio Enhanced Metrics</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-3 bg-gray-50 rounded-lg">
              <div className="text-sm text-gray-600 mb-1">Weighted Score</div>
              <div className="text-xl font-bold text-blue-600">
                {portfolio_enhanced_metrics.weighted_overall_score?.toFixed(0) || '0'}
              </div>
            </div>
            <div className="text-center p-3 bg-gray-50 rounded-lg">
              <div className="text-sm text-gray-600 mb-1">Portfolio Beta</div>
              <div className="text-xl font-bold text-gray-900">
                {portfolio_enhanced_metrics.portfolio_beta?.toFixed(2) || '0.00'}
              </div>
            </div>
            <div className="text-center p-3 bg-gray-50 rounded-lg">
              <div className="text-sm text-gray-600 mb-1">High Conviction</div>
              <div className="text-xl font-bold text-green-600">
                {portfolio_enhanced_metrics.high_conviction_stocks || 0}
              </div>
            </div>
            <div className="text-center p-3 bg-gray-50 rounded-lg">
              <div className="text-sm text-gray-600 mb-1">Total Upside</div>
              <div className="text-xl font-bold text-purple-600">
                {portfolio_enhanced_metrics.total_upside_potential?.toFixed(1) || '0.0'}%
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
