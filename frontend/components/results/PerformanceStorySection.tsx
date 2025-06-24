import React, { useState } from 'react'
import { TrendingUp, TrendingDown, Target, Info, Calendar, DollarSign } from 'lucide-react'
import { ResultsComponentProps } from '../../lib/types-results'

// Performance Insight Card
const PerformanceInsightCard = ({ 
  title, 
  value, 
  change, 
  trend, 
  benchmark,
  timeframe,
  icon,
  color = 'blue'
}: {
  title: string
  value: string
  change?: number
  trend?: 'up' | 'down' | 'neutral'
  benchmark?: string
  timeframe?: string
  icon: React.ReactNode
  color?: 'green' | 'red' | 'blue' | 'purple'
}) => {
  const colorClasses = {
    green: 'bg-green-50 border-green-200 text-green-800',
    red: 'bg-red-50 border-red-200 text-red-800',
    blue: 'bg-blue-50 border-blue-200 text-blue-800',
    purple: 'bg-purple-50 border-purple-200 text-purple-800'
  }

  const trendIcon = trend === 'up' ? <TrendingUp className="w-4 h-4" /> : 
                   trend === 'down' ? <TrendingDown className="w-4 h-4" /> : null

  return (
    <div className={`relative p-6 rounded-xl border-2 ${colorClasses[color]} transition-all duration-200 hover:shadow-lg`}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          {icon}
          <span className="text-sm font-medium">{title}</span>
        </div>
        <div className="flex items-center gap-2">
          {timeframe && (
            <span className="text-xs px-2 py-1 bg-white rounded-full">{timeframe}</span>
          )}
          {trendIcon}
        </div>
      </div>
      
      <div className="text-3xl font-bold mb-2">{value}</div>
      
      {change !== undefined && (
        <div className="text-sm mb-2">
          <span className={`font-medium ${change >= 0 ? 'text-green-700' : 'text-red-700'}`}>
            {change >= 0 ? '+' : ''}{change.toFixed(1)}%
          </span>
          {benchmark && <span className="text-gray-600"> vs {benchmark}</span>}
        </div>
      )}
    </div>
  )
}

// Portfolio Value Timeline Component
const PortfolioValueCard = ({ data }: { data: any }) => {
  const currentValue = data.currentValue
  const totalInvested = data.totalInvested
  const absoluteReturn = currentValue - totalInvested
  const absoluteReturnPct = ((currentValue - totalInvested) / totalInvested) * 100

  return (
    <div className="bg-white rounded-xl border-2 border-gray-200 p-6">
      <div className="flex items-center gap-3 mb-6">
        <DollarSign className="w-6 h-6 text-blue-600" />
        <h3 className="text-xl font-semibold text-gray-900">Portfolio Value</h3>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Total Invested */}
        <div className="text-center">
          <div className="text-sm text-gray-600 mb-1">Total Invested</div>
          <div className="text-2xl font-bold text-gray-900">
            ₹{totalInvested.toLocaleString('en-IN')}
          </div>
        </div>

        {/* Current Value */}
        <div className="text-center">
          <div className="text-sm text-gray-600 mb-1">Current Value</div>
          <div className="text-2xl font-bold text-blue-600">
            ₹{currentValue.toLocaleString('en-IN')}
          </div>
        </div>

        {/* Absolute Return */}
        <div className="text-center">
          <div className="text-sm text-gray-600 mb-1">Total Return</div>
          <div className={`text-2xl font-bold flex items-center justify-center gap-2 ${
            absoluteReturnPct >= 0 ? 'text-green-600' : 'text-red-600'
          }`}>
            {absoluteReturnPct >= 0 ? <TrendingUp className="w-5 h-5" /> : <TrendingDown className="w-5 h-5" />}
            {absoluteReturnPct >= 0 ? '+' : ''}{absoluteReturnPct.toFixed(1)}%
          </div>
          <div className={`text-sm ${absoluteReturnPct >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            {absoluteReturnPct >= 0 ? '+' : ''}₹{absoluteReturn.toLocaleString('en-IN')}
          </div>
        </div>
      </div>

      {/* Performance Story */}
      <div className="mt-6 p-4 bg-gray-50 rounded-lg">
        <h4 className="font-medium text-gray-900 mb-2">Your Story</h4>
        <p className="text-sm text-gray-700">
          {absoluteReturnPct >= 15 
            ? "Exceptional performance! Your portfolio has significantly outperformed expectations."
            : absoluteReturnPct >= 5
            ? "Strong performance! Your portfolio is generating positive returns above inflation."
            : absoluteReturnPct >= 0
            ? "Steady growth! Your portfolio is maintaining positive momentum."
            : "Your portfolio is facing temporary headwinds. Consider rebalancing opportunities."
          }
        </p>
      </div>
    </div>
  )
}

// Performance Comparison Chart
const PerformanceComparisonCard = ({ data }: { data: any }) => {
  const [selectedTimeframe, setSelectedTimeframe] = useState('1Y')
  
  const selectedPerformance = data.performanceMetrics.find(
    (p: any) => p.timeframe === selectedTimeframe
  ) || data.performanceMetrics[0]

  const timeframes = data.performanceMetrics.map((p: any) => p.timeframe)

  return (
    <div className="bg-white rounded-xl border-2 border-gray-200 p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <Target className="w-6 h-6 text-purple-600" />
          <h3 className="text-xl font-semibold text-gray-900">Performance vs {data.benchmarkName}</h3>
        </div>
      </div>

      {/* Timeframe Selector */}
      <div className="flex flex-wrap gap-2 mb-6">
        {timeframes.map((timeframe: string) => (
          <button
            key={timeframe}
            onClick={() => setSelectedTimeframe(timeframe)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              selectedTimeframe === timeframe
                ? 'bg-purple-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            {timeframe}
          </button>
        ))}
      </div>

      {/* Comparison Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="text-center p-4 bg-blue-50 rounded-lg">
          <div className="text-sm text-gray-600 mb-1">Your Portfolio</div>
          <div className={`text-3xl font-bold ${
            selectedPerformance.returns >= 0 ? 'text-green-600' : 'text-red-600'
          }`}>
            {selectedPerformance.returns >= 0 ? '+' : ''}{selectedPerformance.returns.toFixed(1)}%
          </div>
        </div>

        <div className="text-center p-4 bg-gray-50 rounded-lg">
          <div className="text-sm text-gray-600 mb-1">{data.benchmarkName}</div>
          <div className={`text-3xl font-bold ${
            selectedPerformance.benchmarkReturns >= 0 ? 'text-green-600' : 'text-red-600'
          }`}>
            {selectedPerformance.benchmarkReturns >= 0 ? '+' : ''}{selectedPerformance.benchmarkReturns.toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Outperformance */}
      <div className="mt-6 text-center">
        <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg font-medium ${
          selectedPerformance.outperformance >= 0 
            ? 'bg-green-100 text-green-800' 
            : 'bg-red-100 text-red-800'
        }`}>
          {selectedPerformance.outperformance >= 0 ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
          {selectedPerformance.outperformance >= 0 ? '+' : ''}{selectedPerformance.outperformance.toFixed(1)}% vs benchmark
        </div>
      </div>

      {/* Performance Insight */}
      <div className="mt-4 p-4 bg-gray-50 rounded-lg">
        <p className="text-sm text-gray-700">
          {selectedPerformance.outperformance >= 2
            ? "Outstanding outperformance! Your portfolio is significantly beating the market."
            : selectedPerformance.outperformance >= 0
            ? "Good performance! Your portfolio is outperforming the benchmark."
            : selectedPerformance.outperformance >= -2
            ? "Close performance! Your portfolio is tracking the benchmark closely."
            : "Your portfolio is underperforming. Consider reviewing your strategy."
          }
        </p>
      </div>
    </div>
  )
}

export default function PerformanceStorySection({ data }: ResultsComponentProps) {
  const latestPerformance = data.performanceMetrics[data.performanceMetrics.length - 1]

  return (
    <div className="bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="text-center mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Your Performance Story</h2>
          <p className="text-gray-600">Understanding your portfolio's journey and achievements</p>
        </div>

        {/* Key Performance Insights */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <PerformanceInsightCard
            title="Annual Return"
            value={`${latestPerformance.annualizedReturn.toFixed(1)}%`}
            change={latestPerformance.outperformance}
            trend={latestPerformance.returns >= 0 ? 'up' : 'down'}
            benchmark={data.benchmarkName}
            timeframe="CAGR"
            icon={<TrendingUp className="w-5 h-5" />}
            color={latestPerformance.returns >= 10 ? 'green' : latestPerformance.returns >= 0 ? 'blue' : 'red'}
          />

          <PerformanceInsightCard
            title="Risk Score"
            value={`${latestPerformance.metrics.sharpeRatio.toFixed(2)}`}
            benchmark="Sharpe Ratio"
            icon={<Target className="w-5 h-5" />}
            color={latestPerformance.metrics.sharpeRatio >= 1 ? 'green' : 'blue'}
          />

          <PerformanceInsightCard
            title="Volatility"
            value={`${latestPerformance.metrics.volatility.toFixed(1)}%`}
            benchmark="Annual"
            icon={<Calendar className="w-5 h-5" />}
            color={latestPerformance.metrics.volatility <= 15 ? 'green' : latestPerformance.metrics.volatility <= 25 ? 'blue' : 'red'}
          />

          <PerformanceInsightCard
            title="Max Drawdown"
            value={`${latestPerformance.metrics.maxDrawdown.toFixed(1)}%`}
            benchmark="Peak-to-trough"
            icon={<TrendingDown className="w-5 h-5" />}
            color={Math.abs(latestPerformance.metrics.maxDrawdown) <= 10 ? 'green' : Math.abs(latestPerformance.metrics.maxDrawdown) <= 20 ? 'blue' : 'red'}
          />
        </div>

        {/* Detailed Performance Cards */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <PortfolioValueCard data={data} />
          <PerformanceComparisonCard data={data} />
        </div>
      </div>
    </div>
  )
} 