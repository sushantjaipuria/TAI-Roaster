import React, { useState } from 'react'
import { TrendingUp, Info, BarChart3, Target, AlertCircle } from 'lucide-react'
import { ResultsComponentProps, TimeframePerformance } from '../../lib/types-results'

export default function PerformanceMetrics({ data }: ResultsComponentProps) {
  const [selectedTimeframe, setSelectedTimeframe] = useState<string>('1Y')

  // Check if we have real performance data from backend
  const hasRealData = data.performanceMetrics && data.performanceMetrics.length > 0
  
  // Get performance data for selected timeframe
  const selectedPerformance = hasRealData 
    ? data.performanceMetrics.find(p => p.timeframe === selectedTimeframe) || data.performanceMetrics[0]
    : null

  // Metric definitions for tooltips
  const metricDefinitions: { [key: string]: string } = {
    'CAGR': 'Compound Annual Growth Rate - annualized return over the period',
    'Alpha': 'Excess return over benchmark after adjusting for risk',
    'Beta': 'Sensitivity to market movements (1.0 = moves with market)',
    'Sharpe Ratio': 'Risk-adjusted return measure (higher is better)',
    'Sortino Ratio': 'Downside risk-adjusted return (higher is better)',
    'Max Drawdown': 'Largest peak-to-trough decline during the period',
    'Volatility': 'Standard deviation of returns (lower is less risky)',
    'R-Squared': 'Correlation with benchmark (1.0 = perfect correlation)'
  }

  // Component for metric tooltips
  const MetricTooltip = ({ metric, children }: { metric: string; children: React.ReactNode }) => {
    return (
      <div className="group relative">
        {children}
        <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 hidden group-hover:block z-10">
          <div className="bg-gray-900 text-white text-xs rounded py-2 px-3 whitespace-nowrap">
            {metricDefinitions[metric] || 'Performance metric'}
            <div className="absolute top-full left-1/2 transform -translate-x-1/2 border-4 border-transparent border-t-gray-900"></div>
          </div>
        </div>
      </div>
    )
  }

  // If no real data available, show message instead of generating fake data
  if (!hasRealData) {
    return (
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-xl font-bold text-gray-900">📈 Performance Metrics</h2>
            <p className="text-sm text-gray-600 mt-1">
              Portfolio returns vs {data.benchmarkName || 'NIFTY 50'} comparison
            </p>
          </div>
        </div>

        <div className="text-center py-12">
          <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <BarChart3 className="w-8 h-8 text-blue-600" />
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Performance Data Processing</h3>
          <p className="text-gray-600 mb-4">
            We're calculating real performance metrics using historical market data.
          </p>
          <div className="bg-blue-50 rounded-lg p-4 max-w-md mx-auto">
            <p className="text-sm text-blue-800">
              <strong>Real Calculations Include:</strong><br/>
              • Historical returns analysis<br/>
              • Risk-adjusted metrics (Sharpe, Sortino)<br/>
              • Benchmark comparison vs NIFTY 50<br/>
              • Maximum drawdown analysis<br/>
              • Volatility measurements
            </p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold text-gray-900">📈 Performance Metrics</h2>
          <p className="text-sm text-gray-600 mt-1">
            Portfolio returns vs {data.benchmarkName || 'NIFTY 50'} comparison
          </p>
        </div>
        
        {/* Real Data Badge */}
        <div className="flex items-center gap-2 px-3 py-1 bg-green-100 rounded-lg">
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
          <span className="text-sm font-medium text-green-800">Real Data</span>
        </div>
      </div>

      {/* Timeframe Selector */}
      <div className="flex flex-wrap gap-2 mb-6">
        {['1M', '3M', '1Y'].map((timeframe) => {
          // Only show button if we have data for this timeframe
          const hasData = data.performanceMetrics.some(p => p.timeframe === timeframe)
          if (!hasData) return null
          
          return (
            <button
              key={timeframe}
              onClick={() => setSelectedTimeframe(timeframe)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                selectedTimeframe === timeframe
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              {timeframe}
            </button>
          )
        })}
      </div>

      {/* Return Comparison Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        {data.performanceMetrics.map((performance) => {
          const portfolioReturn = performance.returns
          const benchmarkReturn = performance.benchmarkReturns
          const outperformance = performance.outperformance
          const isOutperforming = outperformance >= 0
          
          return (
            <div key={performance.timeframe} className="bg-white border border-gray-200 rounded-lg p-4">
              <div className="text-center">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">{performance.timeframe} Returns</h3>
                
                {/* Portfolio Return */}
                <div className="mb-3">
                  <div className="text-sm text-gray-600">Portfolio</div>
                  <div className={`text-2xl font-bold ${portfolioReturn >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {portfolioReturn >= 0 ? '+' : ''}{portfolioReturn.toFixed(1)}%
                  </div>
                </div>
                
                {/* Benchmark Return */}
                <div className="mb-3">
                  <div className="text-sm text-gray-600">{data.benchmarkName || 'NIFTY 50'}</div>
                  <div className={`text-xl font-medium ${benchmarkReturn >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {benchmarkReturn >= 0 ? '+' : ''}{benchmarkReturn.toFixed(1)}%
                  </div>
                </div>
                
                {/* Outperformance */}
                <div className={`text-sm font-medium px-2 py-1 rounded ${
                  isOutperforming ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                }`}>
                  {isOutperforming ? '+' : ''}{outperformance.toFixed(1)}% vs benchmark
                </div>
              </div>
            </div>
          )
        })}
      </div>

      <div className="grid grid-cols-1 gap-8">
        {/* Metrics Table */}
        {selectedPerformance && (
          <div className="max-w-md mx-auto w-full">
            <div className="bg-gray-50 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                {selectedTimeframe} Metrics
              </h3>
              
              <div className="space-y-4">
                {/* Returns */}
                <div className="pb-3 border-b border-gray-200">
                  <MetricTooltip metric="CAGR">
                    <div className="text-sm font-medium text-gray-700">CAGR</div>
                  </MetricTooltip>
                  <div className="text-xl font-bold text-gray-900">
                    {selectedPerformance.annualizedReturn.toFixed(1)}%
                  </div>
                  <div className={`text-xs ${
                    selectedPerformance.outperformance >= 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {selectedPerformance.outperformance >= 0 ? '+' : ''}{selectedPerformance.outperformance.toFixed(1)}% vs benchmark
                  </div>
                </div>

                {/* Alpha */}
                <div className="pb-3 border-b border-gray-200">
                  <MetricTooltip metric="Alpha">
                    <div className="text-sm font-medium text-gray-700">Alpha</div>
                  </MetricTooltip>
                  <div className={`text-xl font-bold ${
                    selectedPerformance.metrics.alpha >= 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {selectedPerformance.metrics.alpha >= 0 ? '+' : ''}{selectedPerformance.metrics.alpha.toFixed(2)}
                  </div>
                </div>

                {/* Beta */}
                <div className="pb-3 border-b border-gray-200">
                  <MetricTooltip metric="Beta">
                    <div className="text-sm font-medium text-gray-700">Beta</div>
                  </MetricTooltip>
                  <div className="text-xl font-bold text-gray-900">
                    {selectedPerformance.metrics.beta.toFixed(2)}
                  </div>
                </div>

                {/* Sharpe Ratio */}
                <div className="pb-3 border-b border-gray-200">
                  <MetricTooltip metric="Sharpe Ratio">
                    <div className="text-sm font-medium text-gray-700">Sharpe Ratio</div>
                  </MetricTooltip>
                  <div className={`text-xl font-bold ${
                    selectedPerformance.metrics.sharpeRatio >= 1 ? 'text-green-600' : 'text-gray-900'
                  }`}>
                    {selectedPerformance.metrics.sharpeRatio.toFixed(2)}
                  </div>
                </div>

                {/* Sortino Ratio */}
                <div className="pb-3 border-b border-gray-200">
                  <MetricTooltip metric="Sortino Ratio">
                    <div className="text-sm font-medium text-gray-700">Sortino Ratio</div>
                  </MetricTooltip>
                  <div className={`text-xl font-bold ${
                    selectedPerformance.metrics.sortinoRatio >= 1 ? 'text-green-600' : 'text-gray-900'
                  }`}>
                    {selectedPerformance.metrics.sortinoRatio.toFixed(2)}
                  </div>
                </div>

                {/* Volatility */}
                <div className="pb-3 border-b border-gray-200">
                  <MetricTooltip metric="Volatility">
                    <div className="text-sm font-medium text-gray-700">Volatility</div>
                  </MetricTooltip>
                  <div className={`text-xl font-bold ${
                    selectedPerformance.metrics.volatility <= 0.15 ? 'text-green-600' : 
                    selectedPerformance.metrics.volatility <= 0.25 ? 'text-yellow-600' : 'text-red-600'
                  }`}>
                    {(selectedPerformance.metrics.volatility * 100).toFixed(1)}%
                  </div>
                </div>

                {/* Max Drawdown */}
                <div className="pb-3 border-b border-gray-200">
                  <MetricTooltip metric="Max Drawdown">
                    <div className="text-sm font-medium text-gray-700">Max Drawdown</div>
                  </MetricTooltip>
                  <div className={`text-xl font-bold ${
                    selectedPerformance.metrics.maxDrawdown >= -0.1 ? 'text-green-600' : 
                    selectedPerformance.metrics.maxDrawdown >= -0.2 ? 'text-yellow-600' : 'text-red-600'
                  }`}>
                    {(selectedPerformance.metrics.maxDrawdown * 100).toFixed(1)}%
                  </div>
                </div>

                {/* R-Squared */}
                <div>
                  <MetricTooltip metric="R-Squared">
                    <div className="text-sm font-medium text-gray-700">R-Squared</div>
                  </MetricTooltip>
                  <div className="text-xl font-bold text-gray-900">
                    {selectedPerformance.metrics.rSquared.toFixed(2)}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
} 