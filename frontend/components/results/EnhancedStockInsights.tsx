import React, { useState } from 'react'
import { 
  TrendingUp, 
  TrendingDown, 
  AlertTriangle, 
  CheckCircle, 
  Star,
  Target,
  BarChart3,
  Activity
} from 'lucide-react'
import { ResultsComponentProps } from '../../lib/types-results'

// Individual Enhanced Stock Card Component
const StockInsightCard = ({ stock }: { stock: any }) => {
  const [showDetails, setShowDetails] = useState(false)
  
  const getPerformanceColor = (value: number) => {
    if (value >= 5) return 'text-green-600'
    if (value >= 0) return 'text-blue-600'
    return 'text-red-600'
  }

  const getPerformanceIcon = (value: number) => {
    if (value >= 0) return <TrendingUp className="w-4 h-4" />
    return <TrendingDown className="w-4 h-4" />
  }

  const getHealthStatus = (score: number) => {
    if (score >= 80) return { status: 'excellent', color: 'green', label: 'Excellent' }
    if (score >= 65) return { status: 'good', color: 'blue', label: 'Good' }
    if (score >= 50) return { status: 'fair', color: 'yellow', label: 'Fair' }
    return { status: 'poor', color: 'red', label: 'Poor' }
  }

  const getScoreColor = (score: number) => {
    if (score >= 70) return 'bg-green-100 text-green-800 border-green-300'
    if (score >= 50) return 'bg-blue-100 text-blue-800 border-blue-300'
    return 'bg-red-100 text-red-800 border-red-300'
  }

  // Use enhanced scores if available, fallback to basic data
  const overallScore = stock.score || stock.fundamentalScore || 70
  const healthStatus = getHealthStatus(overallScore)
  const returnPct = stock.unrealizedPnL && stock.investmentAmount ? 
    (stock.unrealizedPnL / stock.investmentAmount) * 100 : 
    ((stock.currentPrice - stock.avgPrice) / stock.avgPrice) * 100 || 0

  // Check if enhanced analysis is available
  const hasEnhancedData = stock.enhancedAnalysisAvailable || 
    (stock.fundamentalScore !== undefined && stock.technicalScore !== undefined)

  return (
    <div className="bg-white rounded-xl border-2 border-gray-200 p-6 hover:shadow-lg transition-all duration-200">
      {/* Header with Enhanced Indicator */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <h3 className="text-xl font-bold text-gray-900">{stock.ticker}</h3>
            {hasEnhancedData && (
              <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-purple-100 text-purple-800">
                <Activity className="w-3 h-3 mr-1" />
                Enhanced
              </span>
            )}
          </div>
          <p className="text-sm text-gray-600">{stock.companyName || stock.ticker}</p>
          <p className="text-xs text-gray-500 mt-1">{stock.sector || 'Technology'}</p>
          {stock.targetPrice && (
            <div className="flex items-center gap-1 mt-1">
              <Target className="w-3 h-3 text-blue-500" />
              <span className="text-xs text-blue-600">Target: ₹{stock.targetPrice.toFixed(0)} ({stock.upside?.toFixed(1)}% upside)</span>
            </div>
          )}
        </div>
        <div className="text-right">
          <div className={`px-3 py-1 rounded-full text-xs font-medium bg-${healthStatus.color}-100 text-${healthStatus.color}-800 mb-2`}>
            {healthStatus.label}
          </div>
          <div className="text-xs text-gray-500">Score: {overallScore.toFixed(0)}/100</div>
        </div>
      </div>

      {/* Enhanced Multi-Factor Scores (if available) */}
      {hasEnhancedData && (
        <div className="bg-gradient-to-r from-purple-50 to-blue-50 rounded-lg p-4 mb-4">
          <h4 className="text-sm font-semibold text-gray-900 mb-3 flex items-center gap-2">
            <Activity className="w-4 h-4 text-purple-600" />
            Multi-Factor Analysis
          </h4>
          <div className="grid grid-cols-3 gap-2 text-xs">
            {stock.fundamentalScore !== undefined && (
              <div className="text-center">
                <div className={`px-2 py-1 rounded border ${getScoreColor(stock.fundamentalScore)}`}>
                  {stock.fundamentalScore.toFixed(0)}
                </div>
                <div className="text-gray-600 mt-1">Fundamental</div>
              </div>
            )}
            {stock.technicalScore !== undefined && (
              <div className="text-center">
                <div className={`px-2 py-1 rounded border ${getScoreColor(stock.technicalScore)}`}>
                  {stock.technicalScore.toFixed(0)}
                </div>
                <div className="text-gray-600 mt-1">Technical</div>
              </div>
            )}
            {stock.momentumScore !== undefined && (
              <div className="text-center">
                <div className={`px-2 py-1 rounded border ${getScoreColor(stock.momentumScore)}`}>
                  {stock.momentumScore.toFixed(0)}
                </div>
                <div className="text-gray-600 mt-1">Momentum</div>
              </div>
            )}
            {stock.valueScore !== undefined && (
              <div className="text-center">
                <div className={`px-2 py-1 rounded border ${getScoreColor(stock.valueScore)}`}>
                  {stock.valueScore.toFixed(0)}
                </div>
                <div className="text-gray-600 mt-1">Value</div>
              </div>
            )}
            {stock.qualityScore !== undefined && (
              <div className="text-center">
                <div className={`px-2 py-1 rounded border ${getScoreColor(stock.qualityScore)}`}>
                  {stock.qualityScore.toFixed(0)}
                </div>
                <div className="text-gray-600 mt-1">Quality</div>
              </div>
            )}
            {stock.sentimentScore !== undefined && (
              <div className="text-center">
                <div className={`px-2 py-1 rounded border ${getScoreColor(stock.sentimentScore)}`}>
                  {stock.sentimentScore.toFixed(0)}
                </div>
                <div className="text-gray-600 mt-1">Sentiment</div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Performance Metrics with Real-time Data */}
      <div className="grid grid-cols-3 gap-4 mb-4">
        <div className="text-center">
          <div className="text-sm text-gray-600 mb-1 flex items-center justify-center gap-1">
            {hasEnhancedData && <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>}
            Current Price
          </div>
          <div className="text-lg font-bold text-gray-900">
            ₹{stock.currentPrice?.toFixed(2) || stock.avgPrice?.toFixed(2) || '0.00'}
          </div>
        </div>
        <div className="text-center">
          <div className="text-sm text-gray-600 mb-1">Return</div>
          <div className={`text-lg font-bold flex items-center justify-center gap-1 ${getPerformanceColor(returnPct)}`}>
            {getPerformanceIcon(returnPct)}
            {returnPct >= 0 ? '+' : ''}{returnPct.toFixed(1)}%
          </div>
        </div>
        <div className="text-center">
          <div className="text-sm text-gray-600 mb-1">Weight</div>
          <div className="text-lg font-bold text-blue-600">
            {stock.weight?.toFixed(1) || '0.0'}%
          </div>
        </div>
      </div>

      {/* Investment Details */}
      <div className="bg-gray-50 rounded-lg p-4 mb-4">
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-600">Quantity:</span>
            <span className="font-medium ml-2">{stock.quantity}</span>
          </div>
          <div>
            <span className="text-gray-600">Avg Price:</span>
            <span className="font-medium ml-2">₹{stock.avgPrice?.toFixed(2) || '0.00'}</span>
          </div>
          <div>
            <span className="text-gray-600">Investment:</span>
            <span className="font-medium ml-2">₹{stock.investmentAmount?.toLocaleString('en-IN') || '0'}</span>
          </div>
          <div>
            <span className="text-gray-600">Market Cap:</span>
            <span className="font-medium ml-2">{stock.marketCap || 'Large'}</span>
          </div>
        </div>
      </div>

      {/* Enhanced AI Insights */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h4 className="text-sm font-semibold text-gray-900">Key Insights</h4>
          {hasEnhancedData && (
            <button
              onClick={() => setShowDetails(!showDetails)}
              className="text-xs text-blue-600 hover:text-blue-800 font-medium"
            >
              {showDetails ? 'Hide Details' : 'Show Details'}
            </button>
          )}
        </div>
        
        {/* AI-Generated Reasoning */}
        {stock.reasoning && (
          <div className="p-3 bg-blue-50 rounded-lg border-l-4 border-blue-500">
            <p className="text-sm">{stock.reasoning}</p>
          </div>
        )}

        {/* Key Strengths */}
        {stock.keyStrengths && stock.keyStrengths.length > 0 && (
          <div className="p-3 bg-green-50 rounded-lg border-l-4 border-green-500">
            <p className="text-sm font-medium text-green-800 mb-1">✅ Key Strengths:</p>
            <ul className="text-sm text-green-700 list-disc list-inside">
              {stock.keyStrengths.slice(0, 2).map((strength: string, index: number) => (
                <li key={index}>{strength}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Key Concerns */}
        {stock.keyConcerns && stock.keyConcerns.length > 0 && (
          <div className="p-3 bg-yellow-50 rounded-lg border-l-4 border-yellow-500">
            <p className="text-sm font-medium text-yellow-800 mb-1">⚠️ Key Concerns:</p>
            <ul className="text-sm text-yellow-700 list-disc list-inside">
              {stock.keyConcerns.slice(0, 2).map((concern: string, index: number) => (
                <li key={index}>{concern}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Enhanced Details (expandable) */}
        {showDetails && hasEnhancedData && (
          <div className="space-y-3 mt-4 border-t pt-3">
            {/* Investment Thesis */}
            {stock.investmentThesis && (
              <div className="p-3 bg-purple-50 rounded-lg">
                <p className="text-sm font-medium text-purple-800 mb-1">💡 Investment Thesis:</p>
                <p className="text-sm text-purple-700">{stock.investmentThesis}</p>
              </div>
            )}

            {/* Business Story */}
            {stock.businessStory && (
              <div className="p-3 bg-indigo-50 rounded-lg">
                <p className="text-sm font-medium text-indigo-800 mb-1">📊 Business Overview:</p>
                <p className="text-sm text-indigo-700">{stock.businessStory}</p>
              </div>
            )}

            {/* Catalysts */}
            {stock.catalysts && stock.catalysts.length > 0 && (
              <div className="p-3 bg-emerald-50 rounded-lg">
                <p className="text-sm font-medium text-emerald-800 mb-1">🚀 Growth Catalysts:</p>
                <ul className="text-sm text-emerald-700 list-disc list-inside">
                  {stock.catalysts.map((catalyst: string, index: number) => (
                    <li key={index}>{catalyst}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Risk Factors */}
            {stock.risks && stock.risks.length > 0 && (
              <div className="p-3 bg-red-50 rounded-lg">
                <p className="text-sm font-medium text-red-800 mb-1">⚠️ Risk Factors:</p>
                <ul className="text-sm text-red-700 list-disc list-inside">
                  {stock.risks.map((risk: string, index: number) => (
                    <li key={index}>{risk}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Risk Metrics */}
            {(stock.beta || stock.volatility) && (
              <div className="p-3 bg-gray-50 rounded-lg">
                <p className="text-sm font-medium text-gray-800 mb-2">📈 Risk Metrics:</p>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  {stock.beta && (
                    <div>
                      <span className="text-gray-600">Beta:</span>
                      <span className="font-medium ml-2">{stock.beta.toFixed(2)}</span>
                    </div>
                  )}
                  {stock.volatility && (
                    <div>
                      <span className="text-gray-600">Volatility:</span>
                      <span className="font-medium ml-2">{(stock.volatility * 100).toFixed(1)}%</span>
                    </div>
                  )}
                  {stock.maxDrawdown && (
                    <div>
                      <span className="text-gray-600">Max Drawdown:</span>
                      <span className="font-medium ml-2">{(stock.maxDrawdown * 100).toFixed(1)}%</span>
                    </div>
                  )}
                  {stock.confidenceLevel && (
                    <div>
                      <span className="text-gray-600">Confidence:</span>
                      <span className="font-medium ml-2">{stock.confidenceLevel}</span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Fallback insights for non-enhanced data */}
        {!stock.reasoning && !hasEnhancedData && (
          <>
            <div className={`p-3 rounded-lg border-l-4 ${
              returnPct >= 10 ? 'bg-green-50 border-green-500' : 
              returnPct >= 0 ? 'bg-blue-50 border-blue-500' : 
              'bg-red-50 border-red-500'
            }`}>
              <p className="text-sm">
                {returnPct >= 15 ? '🚀 Strong outperformer in your portfolio' :
                 returnPct >= 5 ? '📈 Performing well, contributing positively' :
                 returnPct >= 0 ? '📊 Holding steady, monitoring required' :
                 returnPct >= -10 ? '⚠️ Underperforming, review recommended' :
                 '🔴 Significant underperformance, action needed'}
              </p>
            </div>

            <div className="p-3 bg-blue-50 rounded-lg border-l-4 border-blue-500">
              <p className="text-sm">
                {returnPct >= 20 ? '💡 Consider partial profit booking' :
                 returnPct >= 10 ? '✅ Hold and monitor quarterly results' :
                 returnPct >= -5 ? '📊 Maintain position, good fundamentals' :
                 returnPct >= -15 ? '⚡ Consider averaging down if fundamentals strong' :
                 '🔄 Review position sizing and exit strategy'}
              </p>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

// Portfolio Holdings Summary
const HoldingsSummaryCard = ({ data }: { data: any }) => {
  const holdings = data.stocks || []
  const totalValue = holdings.reduce((sum: number, stock: any) => 
    sum + (stock.currentPrice || stock.avgPrice || 0) * stock.quantity, 0)
  const totalInvested = holdings.reduce((sum: number, stock: any) => 
    sum + stock.investmentAmount, 0)
  const totalReturn = totalValue - totalInvested
  const totalReturnPct = totalInvested > 0 ? (totalReturn / totalInvested) * 100 : 0

  const topPerformers = holdings
    .map((stock: any) => ({
      ...stock,
      returnPct: ((stock.currentPrice - stock.avgPrice) / stock.avgPrice) * 100 || 0
    }))
    .sort((a: any, b: any) => b.returnPct - a.returnPct)
    .slice(0, 3)

  const bottomPerformers = holdings
    .map((stock: any) => ({
      ...stock,
      returnPct: ((stock.currentPrice - stock.avgPrice) / stock.avgPrice) * 100 || 0
    }))
    .sort((a: any, b: any) => a.returnPct - b.returnPct)
    .slice(0, 3)

  return (
    <div className="bg-white rounded-xl border-2 border-gray-200 p-6 mb-8">
      <div className="flex items-center gap-3 mb-6">
        <BarChart3 className="w-6 h-6 text-blue-600" />
        <h3 className="text-xl font-semibold text-gray-900">Holdings Summary</h3>
      </div>

      {/* Overall Performance */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        <div className="text-center p-4 bg-gray-50 rounded-lg">
          <div className="text-sm text-gray-600 mb-1">Total Holdings</div>
          <div className="text-2xl font-bold text-gray-900">{holdings.length}</div>
        </div>
        <div className="text-center p-4 bg-gray-50 rounded-lg">
          <div className="text-sm text-gray-600 mb-1">Portfolio Value</div>
          <div className="text-2xl font-bold text-blue-600">₹{totalValue.toLocaleString('en-IN')}</div>
        </div>
        <div className="text-center p-4 bg-gray-50 rounded-lg">
          <div className="text-sm text-gray-600 mb-1">Total Return</div>
          <div className={`text-2xl font-bold flex items-center justify-center gap-2 ${
            totalReturnPct >= 0 ? 'text-green-600' : 'text-red-600'
          }`}>
            {totalReturnPct >= 0 ? <TrendingUp className="w-6 h-6" /> : <TrendingDown className="w-6 h-6" />}
            {totalReturnPct >= 0 ? '+' : ''}{totalReturnPct.toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Top & Bottom Performers */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Top Performers */}
        <div>
          <h4 className="text-lg font-semibold text-gray-900 mb-3 flex items-center gap-2">
            <Star className="w-5 h-5 text-green-600" />
            Top Performers
          </h4>
          <div className="space-y-2">
            {topPerformers.map((stock: any, index: number) => (
              <div key={stock.ticker} className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
                <div>
                  <div className="font-medium text-gray-900">{stock.ticker}</div>
                  <div className="text-sm text-gray-600">₹{stock.investmentAmount.toLocaleString('en-IN')}</div>
                </div>
                <div className="text-right">
                  <div className="text-green-600 font-bold">+{stock.returnPct.toFixed(1)}%</div>
                  <div className="text-xs text-gray-500">#{index + 1}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Bottom Performers */}
        <div>
          <h4 className="text-lg font-semibold text-gray-900 mb-3 flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-red-600" />
            Needs Attention
          </h4>
          <div className="space-y-2">
            {bottomPerformers.map((stock: any, index: number) => (
              <div key={stock.ticker} className="flex items-center justify-between p-3 bg-red-50 rounded-lg">
                <div>
                  <div className="font-medium text-gray-900">{stock.ticker}</div>
                  <div className="text-sm text-gray-600">₹{stock.investmentAmount.toLocaleString('en-IN')}</div>
                </div>
                <div className="text-right">
                  <div className="text-red-600 font-bold">{stock.returnPct.toFixed(1)}%</div>
                  <div className="text-xs text-gray-500">Review needed</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

export default function EnhancedStockInsights({ data }: ResultsComponentProps) {
  const [selectedView, setSelectedView] = useState<'overview' | 'detailed'>('overview')
  const holdings = data.stocks || []
  
  // Check for enhanced analysis availability
  const enhancedHoldings = holdings.filter((stock: any) => 
    stock.enhancedAnalysisAvailable || 
    (stock.fundamentalScore !== undefined && stock.technicalScore !== undefined)
  )
  const hasRealTimeData = holdings.some((stock: any) => stock.currentPrice && stock.currentPrice !== stock.avgPrice)

  if (!holdings || holdings.length === 0) {
    return (
      <div className="bg-white py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <Activity className="w-12 h-12 mx-auto mb-4 text-gray-400" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No Individual Stock Data</h3>
          <p className="text-gray-600">Stock-level insights will appear here when individual holdings data is available.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Enhanced Section Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Activity className="w-8 h-8 text-blue-600" />
            <h2 className="text-2xl font-bold text-gray-900">Enhanced Stock Analysis</h2>
            {enhancedHoldings.length > 0 && (
              <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-purple-100 text-purple-800">
                ✨ AI-Powered
              </span>
            )}
          </div>
          <p className="text-gray-600 mb-4">Comprehensive individual stock analysis with real-time data and AI insights</p>
          
          {/* Enhanced Features Status */}
          <div className="flex justify-center gap-6 text-sm">
            <div className={`flex items-center gap-2 ${enhancedHoldings.length > 0 ? 'text-green-600' : 'text-gray-400'}`}>
              <div className={`w-2 h-2 rounded-full ${enhancedHoldings.length > 0 ? 'bg-green-500' : 'bg-gray-300'}`}></div>
              Multi-Factor Analysis ({enhancedHoldings.length}/{holdings.length} stocks)
            </div>
            <div className={`flex items-center gap-2 ${hasRealTimeData ? 'text-green-600' : 'text-gray-400'}`}>
              <div className={`w-2 h-2 rounded-full ${hasRealTimeData ? 'bg-green-500 animate-pulse' : 'bg-gray-300'}`}></div>
              Real-time Market Data
            </div>
            <div className={`flex items-center gap-2 ${enhancedHoldings.length > 0 ? 'text-green-600' : 'text-gray-400'}`}>
              <div className={`w-2 h-2 rounded-full ${enhancedHoldings.length > 0 ? 'bg-green-500' : 'bg-gray-300'}`}></div>
              AI Investment Insights
            </div>
          </div>
        </div>

        {/* View Selector */}
        <div className="flex justify-center gap-2 mb-8">
          <button
            onClick={() => setSelectedView('overview')}
            className={`px-6 py-3 rounded-lg font-medium transition-all duration-200 ${
              selectedView === 'overview'
                ? 'bg-blue-600 text-white shadow-lg'
                : 'bg-white text-gray-700 hover:bg-gray-50'
            }`}
          >
            Portfolio Overview
          </button>
          <button
            onClick={() => setSelectedView('detailed')}
            className={`px-6 py-3 rounded-lg font-medium transition-all duration-200 ${
              selectedView === 'detailed'
                ? 'bg-blue-600 text-white shadow-lg'
                : 'bg-white text-gray-700 hover:bg-gray-50'
            }`}
          >
            Detailed Analysis
          </button>
        </div>

        {/* Content */}
        {selectedView === 'overview' ? (
          <HoldingsSummaryCard data={data} />
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            {holdings.map((stock: any) => (
              <StockInsightCard key={stock.ticker} stock={stock} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
} 