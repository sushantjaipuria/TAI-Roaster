import React from 'react'
import { TrendingUp, TrendingDown, Shield, Target, DollarSign, BarChart3 } from 'lucide-react'
import { ResultsComponentProps, MetricCardProps } from '../../lib/types-results'

// Simple Metric Card Component
function MetricCard({ title, value, change, trend, subtitle, icon, color = 'blue' }: MetricCardProps) {
  const colorClasses = {
    green: 'bg-green-50 border-green-200 text-green-800',
    red: 'bg-red-50 border-red-200 text-red-800',
    blue: 'bg-blue-50 border-blue-200 text-blue-800',
    gray: 'bg-gray-50 border-gray-200 text-gray-800'
  }

  const trendIcon = trend === 'up' ? <TrendingUp className="w-4 h-4" /> : 
                   trend === 'down' ? <TrendingDown className="w-4 h-4" /> : null

  return (
    <div className={`p-4 rounded-lg border-2 ${colorClasses[color]}`}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          {icon}
          <span className="text-sm font-medium">{title}</span>
        </div>
        {trendIcon}
      </div>
      
      <div className="text-2xl font-bold mb-1">
        {typeof value === 'number' ? value.toLocaleString() : value}
      </div>
      
      {subtitle && (
        <div className="text-xs opacity-75">
          {subtitle}
        </div>
      )}
      
      {change !== undefined && (
        <div className={`text-xs mt-1 ${change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
          {change >= 0 ? '+' : ''}{change.toFixed(1)}% vs benchmark
        </div>
      )}
    </div>
  )
}



export default function PortfolioSummaryCard({ data }: ResultsComponentProps) {
  // Calculate derived metrics
  const totalInvested = data.totalInvested
  const currentValue = data.currentValue
  const absoluteReturn = currentValue - totalInvested
  const absoluteReturnPct = ((currentValue - totalInvested) / totalInvested) * 100

  // Get latest performance metrics
  const latestPerformance = data.performanceMetrics[data.performanceMetrics.length - 1]

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold text-gray-900">📊 Executive Summary</h2>
          <p className="text-sm text-gray-600 mt-1">
            15-second snapshot of portfolio health
          </p>
        </div>
        
        {/* Overall Score Badge */}
        <div className="flex items-center gap-2 px-4 py-2 bg-blue-100 rounded-lg">
          <span className="text-sm font-medium text-blue-800">Overall Score</span>
          <span className="text-xl font-bold text-blue-600">{Math.round(data.overallScore)}</span>
          <span className="text-xs text-blue-600">/100</span>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-8">
        {/* KPI Cards */}
        <div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Total Invested */}
            <MetricCard
              title="Total Invested"
              value={`₹${totalInvested.toLocaleString('en-IN')}`}
              icon={<DollarSign className="w-4 h-4" />}
              color="blue"
            />

            {/* Current Value */}
            <MetricCard
              title="Current Value"
              value={`₹${currentValue.toLocaleString('en-IN')}`}
              subtitle={`${absoluteReturnPct >= 0 ? '+' : ''}₹${absoluteReturn.toLocaleString('en-IN')}`}
              icon={<TrendingUp className="w-4 h-4" />}
              color={absoluteReturnPct >= 0 ? 'green' : 'red'}
              trend={absoluteReturnPct >= 0 ? 'up' : 'down'}
            />

            {/* Net P&L */}
            <MetricCard
              title="Net P&L"
              value={`${absoluteReturnPct >= 0 ? '+' : ''}${absoluteReturnPct.toFixed(1)}%`}
              subtitle={`${absoluteReturnPct >= 0 ? '+' : ''}₹${absoluteReturn.toLocaleString('en-IN')}`}
              icon={absoluteReturnPct >= 0 ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
              color={absoluteReturnPct >= 0 ? 'green' : 'red'}
              change={latestPerformance?.outperformance}
            />

            {/* Risk Profile */}
            <MetricCard
              title="Risk Profile"
              value={data.riskLevel.toUpperCase()}
              subtitle={`Volatility: ${latestPerformance?.metrics.volatility.toFixed(1)}%`}
              icon={<Shield className="w-4 h-4" />}
              color={data.riskLevel === 'low' ? 'green' : data.riskLevel === 'medium' ? 'blue' : 'red'}
            />

            {/* Diversification Score */}
            <MetricCard
              title="Diversification"
              value={`${Math.round(data.diversificationScore)}/100`}
              subtitle="Across sectors & assets"
              icon={<BarChart3 className="w-4 h-4" />}
              color={data.diversificationScore >= 75 ? 'green' : data.diversificationScore >= 50 ? 'blue' : 'red'}
            />

            {/* Sharpe Ratio */}
            <MetricCard
              title="Sharpe Ratio"
              value={latestPerformance?.metrics.sharpeRatio.toFixed(2) || '0.00'}
              subtitle="Risk-adjusted returns"
              icon={<Target className="w-4 h-4" />}
              color={(latestPerformance?.metrics.sharpeRatio || 0) >= 1 ? 'green' : 'blue'}
            />
          </div>
        </div>


      </div>

      {/* Traffic Light Summary */}
      <div className="mt-6 pt-6 border-t border-gray-200">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Returns */}
          <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
            <div className={`w-3 h-3 rounded-full ${absoluteReturnPct >= 10 ? 'bg-green-500' : absoluteReturnPct >= 0 ? 'bg-yellow-500' : 'bg-red-500'}`}></div>
            <div>
              <div className="text-sm font-medium text-gray-900">Returns</div>
              <div className="text-xs text-gray-600">
                {absoluteReturnPct >= 10 ? 'Excellent performance' : absoluteReturnPct >= 0 ? 'Positive returns' : 'Needs attention'}
              </div>
            </div>
          </div>

          {/* Risk */}
          <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
            <div className={`w-3 h-3 rounded-full ${data.riskLevel === 'low' ? 'bg-green-500' : data.riskLevel === 'medium' ? 'bg-yellow-500' : 'bg-red-500'}`}></div>
            <div>
              <div className="text-sm font-medium text-gray-900">Risk</div>
              <div className="text-xs text-gray-600">
                {data.riskLevel === 'low' ? 'Conservative approach' : data.riskLevel === 'medium' ? 'Balanced risk' : 'High risk exposure'}
              </div>
            </div>
          </div>

          {/* Diversification */}
          <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
            <div className={`w-3 h-3 rounded-full ${data.diversificationScore >= 75 ? 'bg-green-500' : data.diversificationScore >= 50 ? 'bg-yellow-500' : 'bg-red-500'}`}></div>
            <div>
              <div className="text-sm font-medium text-gray-900">Diversification</div>
              <div className="text-xs text-gray-600">
                {data.diversificationScore >= 75 ? 'Well diversified' : data.diversificationScore >= 50 ? 'Moderately diversified' : 'Needs diversification'}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
} 