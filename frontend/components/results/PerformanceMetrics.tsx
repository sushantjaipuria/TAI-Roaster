import React, { useState } from 'react'
import { TrendingUp, Info, BarChart3, Target } from 'lucide-react'
import { ResultsComponentProps, TimeframePerformance } from '../../lib/types-results'
import LineChart from '../charts/LineChart'

export default function PerformanceMetrics({ data }: ResultsComponentProps) {
  const [selectedTimeframe, setSelectedTimeframe] = useState<string>('1Y')
  const [showBenchmark, setShowBenchmark] = useState(true)

  // Get performance data for selected timeframe
  const selectedPerformance = data.performanceMetrics.find(
    p => p.timeframe === selectedTimeframe
  ) || data.performanceMetrics[0]

  // Prepare chart data from historical data
  const chartData = data.historicalData.map(point => ({
    date: point.date,
    portfolio: point.cumulativeReturn,
    benchmark: ((point.benchmarkValue - 100000) / 100000) * 100 // Convert to percentage
  }))

  // Line configurations for the chart
  const lineConfigs = [
    {
      dataKey: 'portfolio',
      name: 'Portfolio',
      color: '#3B82F6',
      strokeWidth: 3
    },
    ...(showBenchmark ? [{
      dataKey: 'benchmark',
      name: data.benchmarkName,
      color: '#10B981',
      strokeWidth: 2,
      strokeDasharray: '5 5'
    }] : [])
  ]

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

  const MetricTooltip = ({ metric, children }: { metric: string; children: React.ReactNode }) => {
    const [showTooltip, setShowTooltip] = useState(false)
    
    return (
      <div className="relative">
        <div 
          className="flex items-center gap-1 cursor-help"
          onMouseEnter={() => setShowTooltip(true)}
          onMouseLeave={() => setShowTooltip(false)}
        >
          {children}
          <Info className="w-3 h-3 text-gray-400" />
        </div>
        
        {showTooltip && (
          <div className="absolute bottom-full left-0 mb-2 p-2 bg-gray-900 text-white text-xs rounded-lg max-w-xs z-10">
            {metricDefinitions[metric] || 'Metric definition not available'}
            <div className="absolute top-full left-2 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-gray-900"></div>
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold text-gray-900">📈 Performance Metrics</h2>
          <p className="text-sm text-gray-600 mt-1">
            Multi-dimensional risk-adjusted return assessment
          </p>
        </div>
        
        {/* Benchmark Toggle */}
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={showBenchmark}
              onChange={(e) => setShowBenchmark(e.target.checked)}
              className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            Show {data.benchmarkName}
          </label>
        </div>
      </div>

      {/* Timeframe Selector */}
      <div className="flex flex-wrap gap-2 mb-6">
        {data.performanceMetrics.map((performance) => (
          <button
            key={performance.timeframe}
            onClick={() => setSelectedTimeframe(performance.timeframe)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              selectedTimeframe === performance.timeframe
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            {performance.timeframe}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Performance Chart */}
        <div className="lg:col-span-2">
          <div className="bg-gray-50 rounded-lg p-6">
            <LineChart
              data={chartData}
              lines={lineConfigs}
              title={`Performance vs ${data.benchmarkName}`}
              height={400}
              formatYAxis={(value) => `${value.toFixed(1)}%`}
              formatTooltip={(value, name) => `${value.toFixed(2)}%`}
              yAxisLabel="Cumulative Return (%)"
            />
          </div>
        </div>

        {/* Metrics Table */}
        <div className="lg:col-span-1">
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
                <div className="text-xl font-bold text-gray-900">
                  {selectedPerformance.metrics.volatility.toFixed(1)}%
                </div>
              </div>

              {/* Max Drawdown */}
              <div className="pb-3 border-b border-gray-200">
                <MetricTooltip metric="Max Drawdown">
                  <div className="text-sm font-medium text-gray-700">Max Drawdown</div>
                </MetricTooltip>
                <div className="text-xl font-bold text-red-600">
                  {selectedPerformance.metrics.maxDrawdown.toFixed(1)}%
                </div>
              </div>

              {/* R-Squared */}
              <div>
                <MetricTooltip metric="R-Squared">
                  <div className="text-sm font-medium text-gray-700">R-Squared</div>
                </MetricTooltip>
                <div className="text-xl font-bold text-gray-900">
                  {selectedPerformance.metrics.rSquared.toFixed(3)}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Performance Summary Cards */}
      <div className="mt-8 pt-6 border-t border-gray-200">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {/* Best Period */}
          <div className="bg-green-50 rounded-lg p-4 text-center">
            <TrendingUp className="w-6 h-6 text-green-600 mx-auto mb-2" />
            <div className="text-sm font-medium text-green-800">Best Period</div>
            <div className="text-lg font-bold text-green-600">
              {Math.max(...data.performanceMetrics.map(p => p.returns)).toFixed(1)}%
            </div>
            <div className="text-xs text-green-600">
              {data.performanceMetrics.find(p => p.returns === Math.max(...data.performanceMetrics.map(p => p.returns)))?.timeframe}
            </div>
          </div>

          {/* Risk-Adjusted Return */}
          <div className="bg-blue-50 rounded-lg p-4 text-center">
            <Target className="w-6 h-6 text-blue-600 mx-auto mb-2" />
            <div className="text-sm font-medium text-blue-800">Best Sharpe</div>
            <div className="text-lg font-bold text-blue-600">
              {Math.max(...data.performanceMetrics.map(p => p.metrics.sharpeRatio)).toFixed(2)}
            </div>
            <div className="text-xs text-blue-600">Risk-adjusted</div>
          </div>

          {/* Consistency */}
          <div className="bg-purple-50 rounded-lg p-4 text-center">
            <BarChart3 className="w-6 h-6 text-purple-600 mx-auto mb-2" />
            <div className="text-sm font-medium text-purple-800">Avg Outperformance</div>
            <div className="text-lg font-bold text-purple-600">
              {(data.performanceMetrics.reduce((sum, p) => sum + p.outperformance, 0) / data.performanceMetrics.length).toFixed(1)}%
            </div>
            <div className="text-xs text-purple-600">vs {data.benchmarkName}</div>
          </div>

          {/* Risk Level */}
          <div className="bg-orange-50 rounded-lg p-4 text-center">
            <div className="w-6 h-6 mx-auto mb-2 flex items-center justify-center">
              <div className={`w-4 h-4 rounded-full ${
                selectedPerformance.metrics.volatility < 15 ? 'bg-green-500' :
                selectedPerformance.metrics.volatility < 25 ? 'bg-yellow-500' : 'bg-red-500'
              }`}></div>
            </div>
            <div className="text-sm font-medium text-orange-800">Risk Level</div>
            <div className="text-lg font-bold text-orange-600">
              {selectedPerformance.metrics.volatility < 15 ? 'Low' :
               selectedPerformance.metrics.volatility < 25 ? 'Medium' : 'High'}
            </div>
            <div className="text-xs text-orange-600">
              {selectedPerformance.metrics.volatility.toFixed(1)}% volatility
            </div>
          </div>
        </div>
      </div>
    </div>
  )
} 