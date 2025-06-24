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

// Individual Stock Card Component
const StockInsightCard = ({ stock }: { stock: any }) => {
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

  const healthStatus = getHealthStatus(stock.fundamentalScore || 70)
  const returnPct = ((stock.currentPrice - stock.avgPrice) / stock.avgPrice) * 100 || 0

  return (
    <div className="bg-white rounded-xl border-2 border-gray-200 p-6 hover:shadow-lg transition-all duration-200">
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div>
          <h3 className="text-xl font-bold text-gray-900">{stock.ticker}</h3>
          <p className="text-sm text-gray-600">{stock.companyName || stock.ticker}</p>
          <p className="text-xs text-gray-500 mt-1">{stock.sector || 'Technology'}</p>
        </div>
        <div className={`px-3 py-1 rounded-full text-xs font-medium bg-${healthStatus.color}-100 text-${healthStatus.color}-800`}>
          {healthStatus.label}
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="text-center">
          <div className="text-sm text-gray-600 mb-1">Current Price</div>
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

      {/* Key Insights */}
      <div className="space-y-2">
        <h4 className="text-sm font-semibold text-gray-900">Key Insights</h4>
        
        {/* Performance Insight */}
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

        {/* Recommendation */}
        <div className="p-3 bg-blue-50 rounded-lg border-l-4 border-blue-500">
          <p className="text-sm">
            {returnPct >= 20 ? '💡 Consider partial profit booking' :
             returnPct >= 10 ? '✅ Hold and monitor quarterly results' :
             returnPct >= -5 ? '📊 Maintain position, good fundamentals' :
             returnPct >= -15 ? '⚡ Consider averaging down if fundamentals strong' :
             '🔄 Review position sizing and exit strategy'}
          </p>
        </div>
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
        {/* Section Header */}
        <div className="text-center mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Stock-Level Insights</h2>
          <p className="text-gray-600">Detailed analysis of your individual holdings</p>
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