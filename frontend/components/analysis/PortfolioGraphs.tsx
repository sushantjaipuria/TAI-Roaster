'use client'

import React, { useState } from 'react'
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, LineChart, Line, AreaChart, Area } from 'recharts'
import { TrendingUp, Users, Target, AlertTriangle } from 'lucide-react'
import { PortfolioAnalysisDetailed } from '@/lib/types-results'

interface PortfolioGraphsProps {
  data: PortfolioAnalysisDetailed
}

// Color schemes for consistent visualization
const COLORS = {
  concentration: ['#8B5CF6', '#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#6B7280'],
  capCategory: {
    Large: '#10B981',
    Mid: '#3B82F6', 
    Small: '#F59E0B'
  },
  style: {
    Value: '#8B5CF6',
    Growth: '#10B981',
    Income: '#3B82F6'
  }
}

const EquityConcentrationChart = ({ data }: { data: PortfolioAnalysisDetailed }) => {
  const [chartType, setChartType] = useState<'pie' | 'bar'>('pie')
  const [showCumulative, setShowCumulative] = useState(false)

  if (!data?.stocks || data.stocks.length === 0) {
    return (
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-4">Equity Concentration</h2>
        <div className="text-center py-8 text-gray-500">No stock data available</div>
      </div>
    )
  }

  // Process concentration data
  const concentrationData = data.stocks
    .map(stock => ({
      name: stock.ticker,
      value: parseFloat(stock.currentValue?.toString() || '0'),
      weight: stock.weight || 0,
      sector: stock.sector || 'Unknown'
    }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 10) // Top 10 holdings

  // Calculate cumulative weights for concentration analysis
  let cumulativeWeight = 0
  const concentrationWithCumulative = concentrationData.map(item => {
    cumulativeWeight += item.weight
    return {
      ...item,
      cumulativeWeight
    }
  })

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center">
            <Users className="w-6 h-6 text-blue-600" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-gray-900">Equity Concentration</h2>
            <p className="text-gray-600">Top holdings distribution</p>
          </div>
        </div>
        
        {/* Interactive Controls */}
        <div className="flex items-center gap-3">
          <div className="flex bg-gray-100 rounded-lg p-1">
            <button
              onClick={() => setChartType('pie')}
              className={`px-3 py-1 text-sm rounded ${
                chartType === 'pie' 
                  ? 'bg-white text-blue-600 shadow-sm' 
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              Pie Chart
            </button>
            <button
              onClick={() => setChartType('bar')}
              className={`px-3 py-1 text-sm rounded ${
                chartType === 'bar' 
                  ? 'bg-white text-blue-600 shadow-sm' 
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              Bar Chart
            </button>
          </div>
          <div className="flex bg-gray-100 rounded-lg p-1">
            <button
              onClick={() => setShowCumulative(!showCumulative)}
              className={`px-3 py-1 text-sm rounded ${
                showCumulative 
                  ? 'bg-white text-purple-600 shadow-sm' 
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              Cumulative %
            </button>
          </div>
        </div>
      </div>

      <div className="h-[400px] mb-6">
        <ResponsiveContainer width="100%" height="100%">
          {chartType === 'pie' ? (
            <PieChart>
              <Pie
                data={concentrationData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, weight }) => `${name}: ${weight.toFixed(1)}%`}
                outerRadius={120}
                fill="#8884d8"
                dataKey="weight"
              >
                {concentrationData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS.concentration[index % COLORS.concentration.length]} />
                ))}
              </Pie>
              <Tooltip formatter={(value: number) => [`${value.toFixed(2)}%`, 'Weight']} />
              <Legend />
            </PieChart>
          ) : (
            <BarChart data={concentrationWithCumulative}>
              <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
              <XAxis 
                dataKey="name" 
                angle={-45} 
                textAnchor="end" 
                height={80}
                tick={{ fontSize: 11 }}
              />
              <YAxis 
                label={{ value: 'Weight (%)', angle: -90, position: 'insideLeft' }}
                tick={{ fontSize: 12 }}
              />
              <Tooltip 
                formatter={(value: number, name: string) => {
                  if (name === 'weight') return [`${value.toFixed(2)}%`, 'Portfolio Weight']
                  if (name === 'cumulativeWeight') return [`${value.toFixed(2)}%`, 'Cumulative Weight']
                  return [value, name]
                }}
                contentStyle={{
                  backgroundColor: 'rgba(255, 255, 255, 0.95)',
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px',
                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                }}
              />
              <Bar dataKey="weight" fill="#3B82F6" name="Individual Weight" />
              {showCumulative && (
                <Bar dataKey="cumulativeWeight" fill="#10B981" name="Cumulative Weight" opacity={0.7} />
              )}
              <Legend />
            </BarChart>
          )}
        </ResponsiveContainer>
      </div>

      {/* Concentration Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-4 bg-gray-50 rounded-lg">
        <div>
          <div className="text-sm text-gray-600 mb-1">Top 5 Holdings</div>
          <div className="text-2xl font-bold text-gray-900">
            {concentrationWithCumulative.slice(0, 5).reduce((sum, stock) => sum + stock.weight, 0).toFixed(1)}%
          </div>
        </div>
        <div>
          <div className="text-sm text-gray-600 mb-1">Top 10 Holdings</div>
          <div className="text-2xl font-bold text-gray-900">
            {concentrationWithCumulative.slice(0, 10).reduce((sum, stock) => sum + stock.weight, 0).toFixed(1)}%
          </div>
        </div>
        <div>
          <div className="text-sm text-gray-600 mb-1">Concentration Risk</div>
          <div className={`text-2xl font-bold ${
            concentrationWithCumulative.slice(0, 5).reduce((sum, stock) => sum + stock.weight, 0) > 60 
              ? 'text-red-600' 
              : concentrationWithCumulative.slice(0, 5).reduce((sum, stock) => sum + stock.weight, 0) > 40
                ? 'text-yellow-600'
                : 'text-green-600'
          }`}>
            {concentrationWithCumulative.slice(0, 5).reduce((sum, stock) => sum + stock.weight, 0) > 60 
              ? 'High' 
              : concentrationWithCumulative.slice(0, 5).reduce((sum, stock) => sum + stock.weight, 0) > 40
                ? 'Medium'
                : 'Low'}
          </div>
        </div>
      </div>
    </div>
  )
}

const EquityAllocationTable = ({ data }: { data: PortfolioAnalysisDetailed }) => {
  const [viewMode, setViewMode] = useState<'table' | 'heatmap'>('table')

  if (!data?.stocks || data.stocks.length === 0) {
    return (
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-4">Equity Allocation</h2>
        <div className="text-center py-8 text-gray-500">No stock data available</div>
      </div>
    )
  }

  // Enhanced categorization function for Indian market
  const categorizeStock = (stock: any) => {
    const marketCap = stock.fundamentalAnalysis?.metrics?.market_cap
    const peRatio = stock.fundamentalAnalysis?.metrics?.pe_ratio
    const revenueGrowth = stock.fundamentalAnalysis?.metrics?.revenue_growth
    const dividendYield = stock.fundamentalAnalysis?.metrics?.dividend_yield
    const weight = stock.weight || 0

    // Indian market cap classification (in crores)
    let capCategory = 'Small'
    if (marketCap) {
      if (marketCap > 200000) capCategory = 'Large' // >2L crores
      else if (marketCap > 50000) capCategory = 'Mid' // 50K-2L crores
    }

    // Known Indian large-cap stocks
    const largeCaps = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'SBIN', 'WIPRO', 'MARUTI', 'ASIANPAINT', 'NESTLEIND', 'BHARTIARTL', 'ITC']
    if (largeCaps.includes(stock.ticker)) {
      capCategory = 'Large'
    }

    // Investment style classification
    let styleCategory = 'Value'
    if (revenueGrowth && revenueGrowth > 15) styleCategory = 'Growth'
    if (peRatio && peRatio > 30) styleCategory = 'Growth'
    if (dividendYield && dividendYield > 2) styleCategory = 'Income'

    return { capCategory, styleCategory, weight }
  }

  // Build allocation matrix
  const allocationMatrix = {
    Large: { Value: 0, Growth: 0, Income: 0 },
    Mid: { Value: 0, Growth: 0, Income: 0 },
    Small: { Value: 0, Growth: 0, Income: 0 }
  }

  data.stocks.forEach(stock => {
    const { capCategory, styleCategory, weight } = categorizeStock(stock)
    if (allocationMatrix[capCategory as keyof typeof allocationMatrix]) {
      allocationMatrix[capCategory as keyof typeof allocationMatrix][styleCategory as keyof typeof allocationMatrix.Large] += weight
    }
  })

  // Generate recommendations based on allocation
  const generateRecommendations = () => {
    const recommendations = []
    const totalLargeCap = allocationMatrix.Large.Value + allocationMatrix.Large.Growth + allocationMatrix.Large.Income
    const totalMidCap = allocationMatrix.Mid.Value + allocationMatrix.Mid.Growth + allocationMatrix.Mid.Income
    const totalSmallCap = allocationMatrix.Small.Value + allocationMatrix.Small.Growth + allocationMatrix.Small.Income

    // Indian market recommendations
    if (totalLargeCap < 65) {
      recommendations.push({
        type: 'Increase Large Cap Allocation',
        suggestion: `Consider increasing large-cap exposure to 65-75%. Current: ${totalLargeCap.toFixed(1)}%`,
        priority: 'High' as const
      })
    }

    if (totalMidCap > 25) {
      recommendations.push({
        type: 'Reduce Mid Cap Exposure',
        suggestion: `Mid-cap allocation seems high. Consider reducing to 15-25%. Current: ${totalMidCap.toFixed(1)}%`,
        priority: 'Medium' as const
      })
    }

    if (totalSmallCap > 15) {
      recommendations.push({
        type: 'High Small Cap Risk',
        suggestion: `Small-cap allocation is high. Consider reducing to 5-15%. Current: ${totalSmallCap.toFixed(1)}%`,
        priority: 'High' as const
      })
    }

    return recommendations
  }

  const recommendations = generateRecommendations()

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center">
            <Target className="w-6 h-6 text-green-600" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-gray-900">Equity Allocation Table</h2>
            <p className="text-gray-600">Market cap vs Investment style breakdown</p>
          </div>
        </div>
        
        <div className="flex bg-gray-100 rounded-lg p-1">
          <button
            onClick={() => setViewMode('table')}
            className={`px-3 py-1 text-sm rounded ${
              viewMode === 'table' 
                ? 'bg-white text-green-600 shadow-sm' 
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            Table View
          </button>
          <button
            onClick={() => setViewMode('heatmap')}
            className={`px-3 py-1 text-sm rounded ${
              viewMode === 'heatmap' 
                ? 'bg-white text-green-600 shadow-sm' 
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            Heatmap
          </button>
        </div>
      </div>

      {viewMode === 'table' ? (
        <div className="overflow-x-auto mb-6">
          <table className="w-full border-collapse">
            <thead>
              <tr className="bg-gray-50">
                <th className="text-left p-3 font-semibold text-gray-900 border">Market Cap</th>
                <th className="text-center p-3 font-semibold text-gray-900 border">Value %</th>
                <th className="text-center p-3 font-semibold text-gray-900 border">Growth %</th>
                <th className="text-center p-3 font-semibold text-gray-900 border">Income %</th>
                <th className="text-center p-3 font-semibold text-gray-900 border">Total %</th>
              </tr>
            </thead>
            <tbody>
              {(['Large', 'Mid', 'Small'] as const).map(capCategory => {
                const total = allocationMatrix[capCategory].Value + allocationMatrix[capCategory].Growth + allocationMatrix[capCategory].Income
                return (
                  <tr key={capCategory} className="hover:bg-gray-50">
                    <td className="p-3 font-medium text-gray-900 border">{capCategory} Cap</td>
                    <td className="text-center p-3 border">
                      <span className="px-2 py-1 bg-purple-100 text-purple-800 rounded text-sm">
                        {allocationMatrix[capCategory].Value.toFixed(1)}%
                      </span>
                    </td>
                    <td className="text-center p-3 border">
                      <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-sm">
                        {allocationMatrix[capCategory].Growth.toFixed(1)}%
                      </span>
                    </td>
                    <td className="text-center p-3 border">
                      <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-sm">
                        {allocationMatrix[capCategory].Income.toFixed(1)}%
                      </span>
                    </td>
                    <td className="text-center p-3 border font-semibold">
                      {total.toFixed(1)}%
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="h-[300px] mb-6">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart
              data={[
                { 
                  name: 'Large Cap', 
                  Value: allocationMatrix.Large.Value, 
                  Growth: allocationMatrix.Large.Growth, 
                  Income: allocationMatrix.Large.Income 
                },
                { 
                  name: 'Mid Cap', 
                  Value: allocationMatrix.Mid.Value, 
                  Growth: allocationMatrix.Mid.Growth, 
                  Income: allocationMatrix.Mid.Income 
                },
                { 
                  name: 'Small Cap', 
                  Value: allocationMatrix.Small.Value, 
                  Growth: allocationMatrix.Small.Growth, 
                  Income: allocationMatrix.Small.Income 
                }
              ]}
              margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
            >
              <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
              <XAxis dataKey="name" tick={{ fontSize: 12 }} />
              <YAxis label={{ value: 'Allocation %', angle: -90, position: 'insideLeft' }} tick={{ fontSize: 12 }} />
              <Tooltip 
                formatter={(value: number, name: string) => [`${value.toFixed(1)}%`, name]}
                contentStyle={{
                  backgroundColor: 'rgba(255, 255, 255, 0.95)',
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px',
                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                }}
              />
              <Area type="monotone" dataKey="Value" stackId="1" stroke={COLORS.style.Value} fill={COLORS.style.Value} />
              <Area type="monotone" dataKey="Growth" stackId="1" stroke={COLORS.style.Growth} fill={COLORS.style.Growth} />
              <Area type="monotone" dataKey="Income" stackId="1" stroke={COLORS.style.Income} fill={COLORS.style.Income} />
              <Legend />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Portfolio Summary */}
      <div className="mb-6 p-4 bg-gray-50 rounded-lg">
        <h3 className="text-lg font-semibold text-gray-900 mb-3">Portfolio Summary</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <div className="text-gray-600">Total Stocks</div>
            <div className="text-xl font-bold text-gray-900">{data.stocks.length}</div>
          </div>
          <div>
            <div className="text-gray-600">Portfolio Value</div>
            <div className="text-xl font-bold text-gray-900">
              ₹{(data.stocks.reduce((sum, stock) => sum + (parseFloat(stock.currentValue?.toString() || '0')), 0) / 100000).toFixed(2)}L
            </div>
          </div>
          <div>
            <div className="text-gray-600">Top Holding</div>
            <div className="text-xl font-bold text-gray-900">
              {data.stocks.sort((a, b) => (b.weight || 0) - (a.weight || 0))[0]?.ticker || 'N/A'}
            </div>
          </div>
          <div>
            <div className="text-gray-600">Sectors</div>
            <div className="text-xl font-bold text-gray-900">
              {Array.from(new Set(data.stocks.map(s => s.sector).filter(Boolean))).length}
            </div>
          </div>
        </div>
      </div>

      {/* Stock Categorization Breakdown */}
      <div className="mb-6 p-4 bg-blue-50 rounded-lg">
        <h3 className="text-lg font-semibold text-gray-900 mb-3">Stock Categorization</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          {(['Large', 'Mid', 'Small'] as const).map(capCategory => {
            const categoryStocks = data.stocks.filter(stock => {
              const { capCategory: stockCap } = categorizeStock(stock)
              return stockCap === capCategory
            })
            
            if (categoryStocks.length === 0) return null
            
            return (
              <div key={capCategory}>
                <h4 className="font-medium text-gray-800 mb-2">{capCategory} Cap Stocks:</h4>
                <div className="space-y-1">
                  {categoryStocks.map(stock => {
                    const { styleCategory, weight } = categorizeStock(stock)
                    return (
                      <div key={stock.ticker} className="flex justify-between text-xs">
                        <span className="font-medium">{stock.ticker}</span>
                        <span className="text-gray-600">{styleCategory} ({weight.toFixed(1)}%)</span>
                      </div>
                    )
                  })}
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Diversification Recommendations */}
      <div>
        <div className="flex items-center gap-2 mb-4">
          <TrendingUp className="w-5 h-5 text-green-600" />
          <h3 className="text-lg font-semibold text-gray-900">Diversification Recommendations</h3>
        </div>
        
        <div className="space-y-3">
          {recommendations.map((rec, index) => (
            <div key={index} className={`p-4 rounded-lg border-l-4 ${
              rec.priority === 'High' ? 'bg-red-50 border-red-500' :
              rec.priority === 'Medium' ? 'bg-yellow-50 border-yellow-500' :
              'bg-blue-50 border-blue-500'
            }`}>
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="font-medium text-gray-900">{rec.type}</h4>
                  <p className="text-sm text-gray-600 mt-1">{rec.suggestion}</p>
                </div>
                <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                  rec.priority === 'High' ? 'bg-red-100 text-red-800' :
                  rec.priority === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                  'bg-blue-100 text-blue-800'
                }`}>
                  {rec.priority}
                </span>
              </div>
            </div>
          ))}
          
          {recommendations.length === 0 && (
            <div className="p-4 bg-green-50 border-l-4 border-green-500 rounded-lg">
              <div className="flex items-center gap-2">
                <div className="w-5 h-5 bg-green-500 rounded-full flex items-center justify-center">
                  <span className="text-white text-xs">✓</span>
                </div>
                <p className="text-green-800 font-medium">Well Diversified Portfolio</p>
              </div>
              <p className="text-green-700 text-sm mt-1">
                Your portfolio shows good diversification across market caps and investment styles.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default function PortfolioGraphs({ data }: PortfolioGraphsProps) {
  if (!data) {
    return (
      <div className="space-y-8">
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <div className="text-center py-8 text-gray-500">No portfolio data available</div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-8">
      <EquityConcentrationChart data={data} />
      <EquityAllocationTable data={data} />
    </div>
  )
} 