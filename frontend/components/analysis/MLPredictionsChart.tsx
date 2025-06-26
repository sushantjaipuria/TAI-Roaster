/**
 * ML Predictions Chart Component
 * Visualizes machine learning model predictions for individual stocks
 */

import React, { useState } from 'react'
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  LineChart,
  Line,
  Area,
  AreaChart,
  Legend,
  Cell
} from 'recharts'
import { 
  Brain, 
  TrendingUp, 
  AlertCircle, 
  Target, 
  Zap,
  BarChart3,
  Info,
  CheckCircle,
  Activity
} from 'lucide-react'

import { MLPrediction } from '../../types/enhanced-analysis';

interface Props {
  predictions: MLPrediction[]
}

// Enhanced ML Insights Component
const MLInsightsPanel = ({ predictions }: { predictions: MLPrediction[] }) => {
  const avgPrediction = predictions.reduce((sum, p) => sum + p.ensemble_prediction, 0) / predictions.length
  const avgConfidence = predictions.reduce((sum, p) => sum + p.ensemble_confidence, 0) / predictions.length
  const highConfidencePredictions = predictions.filter(p => p.ensemble_confidence > 0.8).length
  const strongBuySignals = predictions.filter(p => p.ensemble_prediction > 0.15).length

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6 mb-6">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
          <Brain className="w-5 h-5 text-blue-600" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-gray-900">ML Predictions Summary</h3>
          <p className="text-sm text-gray-600">Ensemble model insights across your portfolio</p>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="text-center p-4 bg-blue-50 rounded-lg">
          <div className="text-2xl font-bold text-blue-600 mb-1">{(avgPrediction * 100).toFixed(1)}%</div>
          <div className="text-sm text-gray-600">Avg Prediction</div>
          <div className="text-xs text-gray-500 mt-1">Expected return</div>
        </div>
        
        <div className="text-center p-4 bg-green-50 rounded-lg">
          <div className="text-2xl font-bold text-green-600 mb-1">{(avgConfidence * 100).toFixed(0)}%</div>
          <div className="text-sm text-gray-600">Avg Confidence</div>
          <div className="text-xs text-gray-500 mt-1">Model certainty</div>
        </div>
        
        <div className="text-center p-4 bg-purple-50 rounded-lg">
          <div className="text-2xl font-bold text-purple-600 mb-1">{highConfidencePredictions}</div>
          <div className="text-sm text-gray-600">High Confidence</div>
                     <div className="text-xs text-gray-500 mt-1">Predictions {'>'}80%</div>
        </div>
        
        <div className="text-center p-4 bg-orange-50 rounded-lg">
          <div className="text-2xl font-bold text-orange-600 mb-1">{strongBuySignals}</div>
          <div className="text-sm text-gray-600">Strong Signals</div>
                     <div className="text-xs text-gray-500 mt-1">Returns {'>'}15%</div>
        </div>
      </div>
    </div>
  )
}

// Custom Tooltip Component
const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload
    return (
      <div className="bg-white p-4 border border-gray-200 rounded-lg shadow-lg">
        <h4 className="font-semibold text-gray-900 mb-2">{label}</h4>
        <div className="space-y-1">
          <div className="flex items-center justify-between gap-4">
            <span className="text-sm text-gray-600">Ensemble Prediction:</span>
            <span className="font-medium text-blue-600">{(data.ensemble_prediction * 100).toFixed(1)}%</span>
          </div>
          <div className="flex items-center justify-between gap-4">
            <span className="text-sm text-gray-600">Confidence:</span>
            <span className="font-medium text-green-600">{(data.ensemble_confidence * 100).toFixed(0)}%</span>
          </div>
          {data.xgboost_prediction && (
            <div className="flex items-center justify-between gap-4">
              <span className="text-sm text-gray-600">XGBoost:</span>
              <span className="font-medium text-purple-600">{(data.xgboost_prediction * 100).toFixed(1)}%</span>
            </div>
          )}
          {data.ngboost_mean && (
            <div className="flex items-center justify-between gap-4">
              <span className="text-sm text-gray-600">NGBoost Mean:</span>
              <span className="font-medium text-indigo-600">{(data.ngboost_mean * 100).toFixed(1)}%</span>
            </div>
          )}
        </div>
      </div>
    )
  }
  return null
}

// Confidence Level Badge
const ConfidenceBadge = ({ confidence }: { confidence: number }) => {
  const getConfidenceStyle = (conf: number) => {
    if (conf >= 0.8) return { color: 'text-green-700', bg: 'bg-green-100', label: 'High' }
    if (conf >= 0.6) return { color: 'text-blue-700', bg: 'bg-blue-100', label: 'Medium' }
    return { color: 'text-yellow-700', bg: 'bg-yellow-100', label: 'Low' }
  }

  const style = getConfidenceStyle(confidence)
  
  return (
    <span className={`px-2 py-1 rounded-full text-xs font-medium ${style.bg} ${style.color}`}>
      {style.label} ({(confidence * 100).toFixed(0)}%)
    </span>
  )
}

export function MLPredictionsChart({ predictions }: Props) {
  const [chartType, setChartType] = useState<'bar' | 'line' | 'confidence'>('bar')
  const [sortBy, setSortBy] = useState<'prediction' | 'confidence' | 'ticker'>('prediction')

  if (!predictions || predictions.length === 0) {
    return (
      <div className="bg-white rounded-xl border border-gray-200 p-8 text-center">
        <Activity className="w-12 h-12 mx-auto mb-4 text-gray-400" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">No ML Predictions Available</h3>
        <p className="text-gray-600">Machine learning predictions will appear here when models are active.</p>
      </div>
    )
  }

  // Sort predictions
  const sortedPredictions = [...predictions].sort((a, b) => {
    switch (sortBy) {
      case 'prediction':
        return b.ensemble_prediction - a.ensemble_prediction
      case 'confidence':
        return b.ensemble_confidence - a.ensemble_confidence
      case 'ticker':
        return a.ticker.localeCompare(b.ticker)
      default:
        return 0
    }
  })

  // Prepare chart data
  const chartData = sortedPredictions.map(pred => ({
    ticker: pred.ticker,
    prediction: pred.ensemble_prediction * 100,
    confidence: pred.ensemble_confidence * 100,
    ensemble_prediction: pred.ensemble_prediction,
    ensemble_confidence: pred.ensemble_confidence,
    xgboost_prediction: pred.xgboost_prediction,
    ngboost_mean: pred.ngboost_mean,
    ngboost_std: pred.ngboost_std
  }))

  // Get colors for predictions
  const getBarColor = (value: number) => {
    if (value > 15) return '#059669' // green-600
    if (value > 8) return '#2563eb'  // blue-600
    if (value > 0) return '#7c3aed'  // violet-600
    return '#dc2626' // red-600
  }

  const renderChart = () => {
    switch (chartType) {
      case 'bar':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
              <XAxis 
                dataKey="ticker" 
                tick={{ fontSize: 12 }}
                angle={-45}
                textAnchor="end"
                height={80}
              />
              <YAxis 
                tick={{ fontSize: 12 }}
                label={{ value: 'Expected Return (%)', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Bar 
                dataKey="prediction" 
                radius={[4, 4, 0, 0]}
                fill="#2563eb"
              >
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={getBarColor(entry.prediction)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )

      case 'line':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
              <XAxis 
                dataKey="ticker" 
                tick={{ fontSize: 12 }}
                angle={-45}
                textAnchor="end"
                height={80}
              />
              <YAxis 
                tick={{ fontSize: 12 }}
                label={{ value: 'Expected Return (%)', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Line 
                type="monotone" 
                dataKey="prediction" 
                stroke="#2563eb" 
                strokeWidth={3}
                dot={{ fill: '#2563eb', strokeWidth: 2, r: 6 }}
                activeDot={{ r: 8, stroke: '#2563eb', strokeWidth: 2 }}
              />
            </LineChart>
          </ResponsiveContainer>
        )

      case 'confidence':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <AreaChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
              <XAxis 
                dataKey="ticker" 
                tick={{ fontSize: 12 }}
                angle={-45}
                textAnchor="end"
                height={80}
              />
              <YAxis 
                tick={{ fontSize: 12 }}
                label={{ value: 'Confidence (%)', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip 
                content={({ active, payload, label }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload
                    return (
                      <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
                        <h4 className="font-semibold text-gray-900 mb-1">{label}</h4>
                        <div className="text-sm text-gray-600">
                          Prediction Confidence: {data.confidence.toFixed(0)}%
                        </div>
                      </div>
                    )
                  }
                  return null
                }}
              />
              <Area 
                type="monotone" 
                dataKey="confidence" 
                stroke="#059669" 
                fill="#dcfce7"
                strokeWidth={2}
              />
            </AreaChart>
          </ResponsiveContainer>
        )

      default:
        return null
    }
  }

  return (
    <div className="space-y-6">
      {/* ML Insights Panel */}
      <MLInsightsPanel predictions={predictions} />

      {/* Chart Controls */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6">
          <div>
            <h3 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-blue-600" />
              ML Predictions Visualization
            </h3>
            <p className="text-sm text-gray-600">Interactive charts showing model predictions and confidence levels</p>
          </div>
          
          <div className="flex gap-3">
            {/* Chart Type Selector */}
            <select 
              value={chartType} 
              onChange={(e) => setChartType(e.target.value as any)}
              className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="bar">Bar Chart</option>
              <option value="line">Line Chart</option>
              <option value="confidence">Confidence Chart</option>
            </select>

            {/* Sort Selector */}
            <select 
              value={sortBy} 
              onChange={(e) => setSortBy(e.target.value as any)}
              className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="prediction">Sort by Prediction</option>
              <option value="confidence">Sort by Confidence</option>
              <option value="ticker">Sort by Ticker</option>
            </select>
          </div>
        </div>

        {/* Chart */}
        {renderChart()}

        {/* Chart Legend */}
        <div className="mt-4 p-4 bg-gray-50 rounded-lg">
          <h4 className="text-sm font-semibold text-gray-900 mb-3">Model Information</h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-green-600 rounded"></div>
                             <span className="text-gray-600">Strong Buy: {'>'}15% predicted return</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-blue-600 rounded"></div>
              <span className="text-gray-600">Buy: 8-15% predicted return</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-violet-600 rounded"></div>
              <span className="text-gray-600">Hold: 0-8% predicted return</span>
            </div>
          </div>
        </div>
      </div>

      {/* Detailed Predictions Table */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <Target className="w-5 h-5 text-green-600" />
          Detailed Predictions
        </h3>
        
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-3 px-4 font-medium text-gray-900">Stock</th>
                <th className="text-left py-3 px-4 font-medium text-gray-900">Ensemble Prediction</th>
                <th className="text-left py-3 px-4 font-medium text-gray-900">Confidence</th>
                <th className="text-left py-3 px-4 font-medium text-gray-900">XGBoost</th>
                <th className="text-left py-3 px-4 font-medium text-gray-900">NGBoost</th>
                <th className="text-left py-3 px-4 font-medium text-gray-900">Signal</th>
              </tr>
            </thead>
            <tbody>
              {sortedPredictions.map((pred, index) => (
                <tr key={pred.ticker} className={index % 2 === 0 ? 'bg-gray-50' : 'bg-white'}>
                  <td className="py-3 px-4">
                    <div className="font-medium text-gray-900">{pred.ticker}</div>
                  </td>
                  <td className="py-3 px-4">
                    <div className={`font-medium ${
                      pred.ensemble_prediction > 0.15 ? 'text-green-600' : 
                      pred.ensemble_prediction > 0.08 ? 'text-blue-600' : 
                      pred.ensemble_prediction > 0 ? 'text-violet-600' : 'text-red-600'
                    }`}>
                      {(pred.ensemble_prediction * 100).toFixed(1)}%
                    </div>
                  </td>
                  <td className="py-3 px-4">
                    <ConfidenceBadge confidence={pred.ensemble_confidence} />
                  </td>
                  <td className="py-3 px-4">
                    <span className="text-gray-600">
                      {pred.xgboost_prediction ? `${(pred.xgboost_prediction * 100).toFixed(1)}%` : 'N/A'}
                    </span>
                  </td>
                  <td className="py-3 px-4">
                    <span className="text-gray-600">
                      {pred.ngboost_mean ? `${(pred.ngboost_mean * 100).toFixed(1)}%` : 'N/A'}
                    </span>
                  </td>
                  <td className="py-3 px-4">
                    {pred.ensemble_prediction > 0.15 ? (
                      <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                        <TrendingUp className="w-3 h-3" />
                        Strong Buy
                      </span>
                    ) : pred.ensemble_prediction > 0.08 ? (
                      <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                        <TrendingUp className="w-3 h-3" />
                        Buy
                      </span>
                    ) : pred.ensemble_prediction > 0 ? (
                      <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium bg-violet-100 text-violet-800">
                        <CheckCircle className="w-3 h-3" />
                        Hold
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800">
                        <AlertCircle className="w-3 h-3" />
                        Sell
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
} 