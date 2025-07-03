'use client'

import React, { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { TrendingUp, RefreshCw, AlertTriangle, TrendingDown } from 'lucide-react'

interface PerformanceData {
  period: string
  portfolioValue: number
  benchmarkValue: number
  portfolioReturns: number
  benchmarkReturns: number
  outperformance: number
}

interface PerformanceMetrics {
  portfolioXIRR: number
  benchmarkXIRR: number
  outperformance: number
  alpha: number
  beta: number
  sharpeRatio: number
  volatility: number
  maxDrawdown: number
}

interface PortfolioPerformanceChartProps {
  portfolioData: {
    analysisId: string
    stocks: Array<{
      ticker: string
      quantity: number
      entryPrice: number
      currentPrice?: number
      purchaseDate: string
    }>
  }
}

export default function PortfolioPerformanceChart({ portfolioData }: PortfolioPerformanceChartProps) {
  const [timeframe, setTimeframe] = useState<'6M' | '1Y' | '3Y' | '5Y'>('1Y')
  const [performanceData, setPerformanceData] = useState<PerformanceData[]>([])
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date())

  // Auto-refresh every 5 minutes
  useEffect(() => {
    const interval = setInterval(() => {
      fetchPerformanceData()
    }, 5 * 60 * 1000) // 5 minutes

    return () => clearInterval(interval)
  }, [timeframe, portfolioData])

  // Initial load and when timeframe changes
  useEffect(() => {
    fetchPerformanceData()
  }, [timeframe, portfolioData])

  const fetchPerformanceData = async () => {
    if (!portfolioData?.analysisId || !portfolioData?.stocks?.length) {
      setError('Invalid portfolio data')
      return
    }

    setLoading(true)
    setError(null)

    try {
      // DEBUG: Log original portfolio data
      console.log('[DEBUG-XIRR] Original portfolio data:', {
        analysisId: portfolioData.analysisId,
        stocksCount: portfolioData.stocks.length,
        stocks: portfolioData.stocks.map(stock => ({
          ticker: stock.ticker,
          quantity: stock.quantity,
          entryPrice: stock.entryPrice,
          currentPrice: stock.currentPrice,
          purchaseDate: stock.purchaseDate,
          totalInvested: stock.quantity * stock.entryPrice
        }))
      })

      // Calculate single investment date based on timeframe (mutual fund approach)
      const now = new Date()
      let investmentDate: string
      
      switch (timeframe) {
        case '6M':
          investmentDate = new Date(now.getTime() - 180 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]
          break
        case '1Y':
          investmentDate = new Date(now.getTime() - 365 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]
          break
        case '3Y':
          investmentDate = new Date(now.getTime() - 3 * 365 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]
          break
        case '5Y':
          investmentDate = new Date(now.getTime() - 5 * 365 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]
          break
        default:
          investmentDate = new Date(now.getTime() - 365 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]
      }

      // DEBUG: Log timeframe calculation
      console.log('[DEBUG-XIRR] Timeframe calculation:', {
        selectedTimeframe: timeframe,
        calculatedInvestmentDate: investmentDate,
        currentDate: now.toISOString().split('T')[0],
        daysDifference: Math.round((now.getTime() - new Date(investmentDate).getTime()) / (1000 * 60 * 60 * 24))
      })

      const requestBody = {
        holdings: portfolioData.stocks.map(stock => ({
          ticker: stock.ticker.endsWith('.NS') ? stock.ticker : `${stock.ticker}.NS`,
          quantity: stock.quantity,
          purchase_price: stock.entryPrice,
          purchase_date: investmentDate // Same date for all holdings (mutual fund approach)
        })),
        time_periods: [timeframe]
      }

      // DEBUG: Log complete request body with totals
      const totalInvestment = requestBody.holdings.reduce((sum, h) => sum + (h.quantity * h.purchase_price), 0)
      console.log('[DEBUG-XIRR] API Request Details:', {
        endpoint: '/api/portfolio-performance/calculate-performance-v4',
        holdingsCount: requestBody.holdings.length,
        timeframe: timeframe,
        totalInvestment: totalInvestment,
        holdings: requestBody.holdings,
        requestBody: requestBody
      })

      const response = await fetch('/api/portfolio-performance/calculate-performance-v4', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      })

      // DEBUG: Log response status
      console.log('[DEBUG-XIRR] API Response Status:', {
        status: response.status,
        statusText: response.statusText,
        ok: response.ok,
        headers: Object.fromEntries(response.headers.entries())
      })

      if (!response.ok) {
        throw new Error(`API Error: ${response.status} - ${response.statusText}`)
      }

      const data = await response.json()
      
      // DEBUG: Log complete response data structure
      console.log('[DEBUG-XIRR] Complete API Response:', {
        hasPerformanceMetrics: !!data.performance_metrics,
        performanceMetricsCount: data.performance_metrics?.length || 0,
        performanceMetrics: data.performance_metrics,
        hasTimeSeriesData: !!data.time_series_data,
        timeSeriesKeys: data.time_series_data ? Object.keys(data.time_series_data) : [],
        timeSeriesData: data.time_series_data,
        calculationTimestamp: data.calculation_timestamp,
        dataSources: data.data_sources,
        error: data.error,
        fullResponse: data
      })

      if (data.performance_metrics && data.performance_metrics.length > 0) {
        // Find the performance data for the current timeframe
        const currentMetrics = data.performance_metrics.find((metric: any) => 
          metric.timeframe === timeframe
        ) || data.performance_metrics[0]
        
        // DEBUG: Log metrics selection and values
        console.log('[DEBUG-XIRR] Metrics Processing:', {
          requestedTimeframe: timeframe,
          availableTimeframes: data.performance_metrics.map((m: any) => m.timeframe),
          foundExactMatch: !!data.performance_metrics.find((m: any) => m.timeframe === timeframe),
          selectedMetrics: currentMetrics,
          returns: currentMetrics?.returns,
          benchmarkReturns: currentMetrics?.benchmarkReturns,
          outperformance: currentMetrics?.outperformance
        })
        
        // Transform time series data for the current timeframe
        const timeSeriesForPeriod = data.time_series_data[timeframe] || []
        
        // DEBUG: Log time series processing
        console.log('[DEBUG-XIRR] Time Series Processing:', {
          timeframe: timeframe,
          timeSeriesLength: timeSeriesForPeriod.length,
          timeSeriesData: timeSeriesForPeriod,
          firstPoint: timeSeriesForPeriod[0],
          lastPoint: timeSeriesForPeriod[timeSeriesForPeriod.length - 1]
        })
        
        const chartData: PerformanceData[] = timeSeriesForPeriod.map((point: any, index: number) => {
          // Fix chart values to be more realistic
          const monthsElapsed = index + 1
          const portfolioReturn = point.portfolio_return || currentMetrics.returns
          const benchmarkReturn = point.benchmark_return || currentMetrics.benchmarkReturns
          
          // DEBUG: Log chart data point processing
          if (index === 0) {
            console.log('[DEBUG-XIRR] Chart Data Point Processing (V4 - Optimized Real XIRR):', {
              originalPortfolioReturn: point.portfolio_return,
              fallbackReturn: currentMetrics.returns,
              finalPortfolioReturn: portfolioReturn,
              originalBenchmarkReturn: point.benchmark_return,
              fallbackBenchmarkReturn: currentMetrics.benchmarkReturns,
              finalBenchmarkReturn: benchmarkReturn,
              calculationMethod: 'Real historical XIRR for each time point (V4)'
            })
          }
          
          return {
            period: point.period,
            portfolioValue: 1000000 * Math.pow(1 + portfolioReturn/100/12, monthsElapsed),
            benchmarkValue: 1000000 * Math.pow(1 + benchmarkReturn/100/12, monthsElapsed),
            portfolioReturns: portfolioReturn,
            benchmarkReturns: benchmarkReturn,
            outperformance: portfolioReturn - benchmarkReturn
          }
        })

        const performanceMetrics: PerformanceMetrics = {
          portfolioXIRR: currentMetrics.returns,
          benchmarkXIRR: currentMetrics.benchmarkReturns,
          outperformance: currentMetrics.outperformance,
          alpha: currentMetrics.metrics.alpha,
          beta: currentMetrics.metrics.beta,
          sharpeRatio: currentMetrics.metrics.sharpeRatio,
          volatility: currentMetrics.metrics.volatility,
          maxDrawdown: currentMetrics.metrics.maxDrawdown
        }

        // DEBUG: Log final display values
        console.log('[DEBUG-XIRR] Final Display Values:', {
          metricsBoxXIRR: performanceMetrics.portfolioXIRR,
          metricsBoxBenchmark: performanceMetrics.benchmarkXIRR,
          metricsBoxOutperformance: performanceMetrics.outperformance,
          chartDataLength: chartData.length,
          firstChartPoint: chartData[0],
          lastChartPoint: chartData[chartData.length - 1],
          performanceMetrics: performanceMetrics
        })

        setPerformanceData(chartData)
        setMetrics(performanceMetrics)
        setLastRefresh(new Date())
      } else {
        console.error('[DEBUG-XIRR] No performance metrics in response:', data)
        throw new Error(data.error || 'Failed to calculate XIRR')
      }
    } catch (err) {
      console.error('[DEBUG-XIRR] XIRR calculation error details:', {
        error: err,
        message: err instanceof Error ? err.message : 'Unknown error',
        stack: err instanceof Error ? err.stack : undefined,
        portfolioData: portfolioData,
        timeframe: timeframe
      })
      setError(err instanceof Error ? err.message : 'Failed to fetch performance data')
      setPerformanceData([])
      setMetrics(null)
    } finally {
      setLoading(false)
    }
  }

  const manualRefresh = () => {
    fetchPerformanceData()
  }

  if (!portfolioData?.stocks?.length) {
    return (
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center">
            <TrendingUp className="w-6 h-6 text-purple-600" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-gray-900">Portfolio Performance (XIRR)</h2>
            <p className="text-gray-600">Real-time XIRR vs NIFTY 50 comparison</p>
          </div>
        </div>
        <div className="text-center py-8 text-gray-500">
          <AlertTriangle className="w-12 h-12 mx-auto mb-4 text-gray-400" />
          No portfolio data available for XIRR calculation
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center">
            <TrendingUp className="w-6 h-6 text-purple-600" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-gray-900">Portfolio Performance (XIRR)</h2>
            <p className="text-gray-600">Real-time XIRR vs NIFTY 50 comparison</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Timeframe Selection */}
          <div className="flex bg-gray-100 rounded-lg p-1">
            {(['6M', '1Y', '3Y', '5Y'] as const).map((period) => (
              <button
                key={period}
                onClick={() => setTimeframe(period)}
                className={`px-3 py-1 text-sm rounded ${
                  timeframe === period
                    ? 'bg-white text-purple-600 shadow-sm' 
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                {period}
              </button>
            ))}
          </div>

          {/* Manual Refresh */}
          <button
            onClick={manualRefresh}
            disabled={loading}
            className="p-2 text-gray-600 hover:text-gray-900 rounded-lg hover:bg-gray-100 disabled:opacity-50"
            title="Refresh data"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-red-600" />
            <p className="text-red-800 font-medium">Performance Data Unavailable</p>
          </div>
          <p className="text-red-700 text-sm mt-1">{error}</p>
        </div>
      )}

      {/* Performance Summary Cards - Reference UI Style */}
      {metrics && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-blue-50 rounded-lg p-4 border border-blue-100">
            <div className="text-blue-600 text-sm font-medium mb-1">Portfolio XIRR</div>
            <div className="text-blue-900 text-3xl font-bold">
              {Math.abs(metrics.portfolioXIRR) < 0.01 ? 
                metrics.portfolioXIRR.toFixed(3) : 
                metrics.portfolioXIRR.toFixed(1)}%
            </div>
            <div className="text-blue-700 text-xs mt-1">{timeframe} annualized</div>
          </div>
          
          <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
            <div className="text-gray-600 text-sm font-medium mb-1">NIFTY 50 CAGR</div>
            <div className="text-gray-900 text-3xl font-bold">
              {metrics.benchmarkXIRR.toFixed(1)}%
            </div>
            <div className="text-gray-700 text-xs mt-1">Benchmark returns</div>
          </div>
          
          <div className={`rounded-lg p-4 border ${
            metrics.outperformance >= 0 
              ? 'bg-green-50 border-green-200' 
              : 'bg-red-50 border-red-200'
          }`}>
            <div className={`text-sm font-medium mb-1 ${
              metrics.outperformance >= 0 ? 'text-green-600' : 'text-red-600'
            }`}>
              Outperformance
            </div>
            <div className={`text-3xl font-bold ${
              metrics.outperformance >= 0 ? 'text-green-900' : 'text-red-900'
            }`}>
              {metrics.outperformance >= 0 ? '+' : ''}{metrics.outperformance.toFixed(1)}%
            </div>
            <div className={`text-xs mt-1 ${
              metrics.outperformance >= 0 ? 'text-green-700' : 'text-red-700'
            }`}>
              vs benchmark
            </div>
          </div>
          
          <div className="bg-orange-50 rounded-lg p-4 border border-orange-100">
            <div className="text-orange-600 text-sm font-medium mb-1">Sharpe Ratio</div>
            <div className="text-orange-900 text-3xl font-bold">
              {metrics.sharpeRatio.toFixed(2)}
            </div>
            <div className="text-orange-700 text-xs mt-1">Risk-adj returns</div>
          </div>
        </div>
      )}

      {/* Chart */}
      {performanceData.length > 0 ? (
        <div className="h-[400px] mb-6">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={performanceData}>
              <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
              <XAxis 
                dataKey="period" 
                tick={{ fontSize: 12 }}
                angle={timeframe === '6M' ? 0 : -45}
                textAnchor={timeframe === '6M' ? 'middle' : 'end'}
                height={timeframe === '6M' ? 50 : 80}
              />
              <YAxis 
                label={{ value: 'Returns (%)', angle: -90, position: 'insideLeft' }}
                tick={{ fontSize: 12 }}
                tickFormatter={(value) => `${value.toFixed(1)}%`}
              />
              <Tooltip 
                formatter={(value: number, name: string) => {
                  if (name === 'Annualised Returns or XIRR') {
                    return [`${value.toFixed(2)}%`, 'Portfolio XIRR']
                  } else if (name === 'NIFTY50') {
                    return [`${value.toFixed(2)}%`, 'NIFTY 50 CAGR']
                  }
                  return [`${value.toFixed(2)}%`, name]
                }}
                labelFormatter={(label: string) => `Period: ${label}`}
                contentStyle={{
                  backgroundColor: 'rgba(255, 255, 255, 0.95)',
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px',
                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                }}
              />
              <Line 
                type="monotone" 
                dataKey="portfolioReturns" 
                stroke="#3B82F6" 
                strokeWidth={3}
                dot={{ fill: '#3B82F6', strokeWidth: 2, r: 3 }}
                activeDot={{ r: 5, stroke: '#3B82F6', strokeWidth: 2 }}
                name="Annualised Returns or XIRR"
              />
              <Line 
                type="monotone" 
                dataKey="benchmarkReturns" 
                stroke="#1F2937" 
                strokeWidth={2}
                strokeDasharray="8 4"
                dot={{ fill: '#1F2937', strokeWidth: 1, r: 2 }}
                activeDot={{ r: 4, stroke: '#1F2937', strokeWidth: 2 }}
                name="NIFTY50"
              />
              <Legend />
            </LineChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div className="h-[400px] mb-6 flex items-center justify-center bg-gray-50 rounded-lg">
          <div className="text-center">
            {loading ? (
              <div className="flex items-center gap-3">
                <RefreshCw className="w-6 h-6 animate-spin text-blue-600" />
                <span className="text-gray-600">Calculating XIRR performance...</span>
              </div>
            ) : (
              <div>
                <AlertTriangle className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                <p className="text-gray-600">No performance data available</p>
                <p className="text-gray-500 text-sm mt-2">Please check backend connection</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Risk Metrics & Insights */}
      {metrics && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-3">Risk Metrics</h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Beta (Market Risk)</span>
                <span className="font-medium">{metrics.beta.toFixed(2)}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Alpha</span>
                <span className="font-medium">{metrics.alpha.toFixed(2)}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Volatility</span>
                <span className="font-medium">{metrics.volatility.toFixed(1)}%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Max Drawdown</span>
                <span className="font-medium text-red-600">
                  {metrics.maxDrawdown.toFixed(1)}%
                </span>
              </div>
            </div>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-3">Performance Insights</h3>
            <div className="space-y-2 text-sm">
              {metrics.outperformance > 2 && (
                <div className="p-3 bg-green-50 border-l-4 border-green-500 rounded">
                  <p className="text-green-800 font-medium">Excellent Performance</p>
                  <p className="text-green-700">XIRR is beating NIFTY 50 by {metrics.outperformance.toFixed(1)}%</p>
                </div>
              )}
              {metrics.outperformance < -2 && (
                <div className="p-3 bg-red-50 border-l-4 border-red-500 rounded">
                  <p className="text-red-800 font-medium">Underperformance</p>
                  <p className="text-red-700">XIRR is lagging NIFTY 50 by {Math.abs(metrics.outperformance).toFixed(1)}%</p>
                </div>
              )}
              {Math.abs(metrics.outperformance) <= 2 && (
                <div className="p-3 bg-blue-50 border-l-4 border-blue-500 rounded">
                  <p className="text-blue-800 font-medium">Market Performance</p>
                  <p className="text-blue-700">XIRR is tracking close to NIFTY 50</p>
                </div>
              )}
              {metrics.sharpeRatio > 1.5 && (
                <div className="p-3 bg-purple-50 border-l-4 border-purple-500 rounded">
                  <p className="text-purple-800 font-medium">Strong Risk-Adjusted Returns</p>
                  <p className="text-purple-700">Sharpe ratio of {metrics.sharpeRatio.toFixed(2)} indicates excellent risk management</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Last Updated */}
      <div className="mt-6 pt-4 border-t border-gray-200">
        <div className="flex items-center justify-between text-sm text-gray-500">
          <span>Last updated: {lastRefresh.toLocaleTimeString()}</span>
          <span>Market data: Live • Auto-refresh: 5min</span>
        </div>
      </div>
    </div>
  )
} 