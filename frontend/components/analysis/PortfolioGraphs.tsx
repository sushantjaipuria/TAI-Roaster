import React from 'react'
import { BarChart3, PieChart as PieChartIcon, LineChart } from 'lucide-react'
import { LineChart as RechartsLineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Bar, BarChart, ComposedChart } from 'recharts'
import { PortfolioAnalysisDetailed } from '../../lib/types-results'

interface PortfolioGraphsProps {
  data: PortfolioAnalysisDetailed
}

const EquityConcentrationChart = ({ data }: { data: PortfolioAnalysisDetailed }) => {
  // Get concentration metrics from the data
  const concentrationMetrics = data.allocation.concentration.metrics
  const sortedStocks = [...data.stocks].sort((a, b) => b.weight - a.weight)
  
  // Calculate cumulative weights
  let cumulativeWeight = 0
  const chartData = sortedStocks.map(stock => {
    cumulativeWeight += stock.weight
    return {
      ticker: stock.ticker,
      weight: stock.weight,
      cumulative_weight: cumulativeWeight
    }
  })

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center">
          <BarChart3 className="w-6 h-6 text-blue-600" />
        </div>
        <div>
          <h2 className="text-xl font-bold text-gray-900">Equity Concentration</h2>
          <p className="text-gray-600">Portfolio weight distribution and concentration analysis</p>
        </div>
      </div>

      <div className="h-[400px]">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="ticker" angle={-45} textAnchor="end" height={70} />
            <YAxis yAxisId="left" label={{ value: 'Weight (%)', angle: -90, position: 'insideLeft' }} />
            <YAxis yAxisId="right" orientation="right" label={{ value: 'Cumulative Weight (%)', angle: 90, position: 'insideRight' }} />
            <Tooltip 
              formatter={(value: number) => `${value.toFixed(1)}%`}
              labelFormatter={(label: string) => `Stock: ${label}`}
            />
            <Bar yAxisId="left" dataKey="weight" fill="#4F46E5" name="Weight %" />
            <Line yAxisId="right" type="monotone" dataKey="cumulative_weight" stroke="#EA580C" name="Cumulative %" dot={true} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Concentration Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
        <div className="p-4 bg-blue-50 rounded-lg">
          <div className="text-sm text-blue-600 mb-1">Top 3 Holdings</div>
          <div className="text-2xl font-bold text-blue-900">
            {concentrationMetrics.top_3_concentration.toFixed(1)}%
          </div>
          <div className="text-xs text-blue-700 mt-1">of portfolio value</div>
        </div>
        <div className="p-4 bg-orange-50 rounded-lg">
          <div className="text-sm text-orange-600 mb-1">Concentration Level</div>
          <div className="text-2xl font-bold text-orange-900">
            {concentrationMetrics.concentration_level}
          </div>
          <div className="text-xs text-orange-700 mt-1">based on top holdings</div>
        </div>
        <div className="p-4 bg-green-50 rounded-lg">
          <div className="text-sm text-green-600 mb-1">Diversification Score</div>
          <div className="text-2xl font-bold text-green-900">
            {concentrationMetrics.diversification_score.toFixed(0)}/100
          </div>
          <div className="text-xs text-green-700 mt-1">higher is better</div>
        </div>
      </div>
    </div>
  )
}

export default function PortfolioGraphs({ data }: PortfolioGraphsProps) {
  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 bg-purple-100 rounded-full flex items-center justify-center">
            <LineChart className="w-5 h-5 text-purple-600" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-gray-900">Portfolio Visualization</h2>
            <p className="text-gray-600">Interactive charts and visual analytics</p>
          </div>
        </div>
      </div>

      {/* Equity Concentration Chart */}
      <EquityConcentrationChart data={data} />
    </div>
  )
}