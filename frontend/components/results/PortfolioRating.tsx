import React from 'react'
import { AlertTriangle, CheckCircle, Info } from 'lucide-react'
import { ResultsComponentProps } from '../../lib/types-results'

export default function PortfolioRating({ data }: ResultsComponentProps) {
  // Simple circular progress component
  const CircularProgress = ({ value, size = 120, strokeWidth = 8 }: { value: number; size?: number; strokeWidth?: number }) => {
    const radius = (size - strokeWidth) / 2
    const circumference = 2 * Math.PI * radius
    const strokeDasharray = circumference
    const strokeDashoffset = circumference - (value / 100) * circumference

    const getColor = (score: number) => {
      if (score >= 80) return '#10B981' // green
      if (score >= 60) return '#F59E0B' // yellow
      return '#EF4444' // red
    }

    return (
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="transform -rotate-90">
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke="#E5E7EB"
            strokeWidth={strokeWidth}
            fill="none"
          />
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke={getColor(value)}
            strokeWidth={strokeWidth}
            fill="none"
            strokeDasharray={strokeDasharray}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            className="transition-all duration-1000 ease-out"
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">{Math.round(value)}</div>
            <div className="text-xs text-gray-500">/ 100</div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold text-gray-900">🌟 Portfolio Rating & Hygiene Report</h2>
          <p className="text-sm text-gray-600 mt-1">
            Comprehensive quality assessment and improvement areas
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* TAI Score */}
        <div className="text-center">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">TAI Score</h3>
          <CircularProgress value={data.rating.taiScore} />
          <div className="mt-4">
            <p className="text-sm text-gray-600">
              {data.rating.taiScore >= 80 ? 'Excellent Portfolio' :
               data.rating.taiScore >= 60 ? 'Good Portfolio' :
               data.rating.taiScore >= 40 ? 'Average Portfolio' : 'Needs Improvement'}
            </p>
          </div>
        </div>

        {/* Rating Breakdown */}
        <div>
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Rating Breakdown</h3>
          <div className="space-y-3">
            {[
              { label: 'Return Quality', value: data.rating.returnQuality },
              { label: 'Risk Management', value: data.rating.riskManagement },
              { label: 'Diversification', value: data.rating.diversification },
              { label: 'Cost Efficiency', value: data.rating.costEfficiency },
              { label: 'Liquidity', value: data.rating.liquidityScore }
            ].map((metric, index) => (
              <div key={index}>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-700">{metric.label}</span>
                  <span className="font-medium">{Math.round(metric.value)}/100</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className={`h-2 rounded-full transition-all duration-500 ${
                      metric.value >= 80 ? 'bg-green-500' :
                      metric.value >= 60 ? 'bg-yellow-500' : 'bg-red-500'
                    }`}
                    style={{ width: `${metric.value}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Hygiene Flags */}
        <div>
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Hygiene Flags</h3>
          <div className="space-y-3">
            {/* Penny Stocks */}
            <div className="flex items-start gap-2">
              {data.hygiene.pennyStocks.count > 0 ? (
                <AlertTriangle className="w-4 h-4 text-yellow-500 mt-0.5 flex-shrink-0" />
              ) : (
                <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
              )}
              <div className="text-sm">
                <div className="font-medium text-gray-900">Penny Stocks</div>
                <div className="text-gray-600">
                  {data.hygiene.pennyStocks.count > 0 
                    ? `${data.hygiene.pennyStocks.count} holdings identified`
                    : 'No penny stocks found'
                  }
                </div>
              </div>
            </div>

            {/* Excessive Cash */}
            <div className="flex items-start gap-2">
              {data.hygiene.excessiveCash.isExcessive ? (
                <AlertTriangle className="w-4 h-4 text-yellow-500 mt-0.5 flex-shrink-0" />
              ) : (
                <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
              )}
              <div className="text-sm">
                <div className="font-medium text-gray-900">Cash Allocation</div>
                <div className="text-gray-600">
                  {data.hygiene.excessiveCash.percentage.toFixed(1)}% cash
                  {data.hygiene.excessiveCash.isExcessive && ' (excessive)'}
                </div>
              </div>
            </div>

            {/* Small Cap Overexposure */}
            <div className="flex items-start gap-2">
              {data.hygiene.smallCapOverexposure.isExcessive ? (
                <AlertTriangle className="w-4 h-4 text-yellow-500 mt-0.5 flex-shrink-0" />
              ) : (
                <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
              )}
              <div className="text-sm">
                <div className="font-medium text-gray-900">Small Cap Exposure</div>
                <div className="text-gray-600">
                  {data.hygiene.smallCapOverexposure.percentage.toFixed(1)}%
                  {data.hygiene.smallCapOverexposure.isExcessive && ' (high risk)'}
                </div>
              </div>
            </div>

            {/* Low Liquidity */}
            <div className="flex items-start gap-2">
              {data.hygiene.lowLiquidityStocks.count > 0 ? (
                <AlertTriangle className="w-4 h-4 text-yellow-500 mt-0.5 flex-shrink-0" />
              ) : (
                <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
              )}
              <div className="text-sm">
                <div className="font-medium text-gray-900">Liquidity Risk</div>
                <div className="text-gray-600">
                  {data.hygiene.lowLiquidityStocks.count > 0 
                    ? `${data.hygiene.lowLiquidityStocks.count} low liquidity stocks`
                    : 'Good liquidity across holdings'
                  }
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Pro Tips */}
      <div className="mt-8 pt-6 border-t border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Pro Tips</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {data.rating.proTips.map((tip, index) => (
            <div key={index} className={`p-4 rounded-lg border-l-4 ${
              tip.impact === 'HIGH' ? 'border-red-500 bg-red-50' :
              tip.impact === 'MEDIUM' ? 'border-yellow-500 bg-yellow-50' :
              'border-blue-500 bg-blue-50'
            }`}>
              <div className="flex items-start gap-2">
                <Info className={`w-4 h-4 mt-0.5 flex-shrink-0 ${
                  tip.impact === 'HIGH' ? 'text-red-600' :
                  tip.impact === 'MEDIUM' ? 'text-yellow-600' :
                  'text-blue-600'
                }`} />
                <div>
                  <div className={`text-sm font-medium ${
                    tip.impact === 'HIGH' ? 'text-red-800' :
                    tip.impact === 'MEDIUM' ? 'text-yellow-800' :
                    'text-blue-800'
                  }`}>
                    {tip.category}
                  </div>
                  <div className="text-sm text-gray-700 mt-1">{tip.tip}</div>
                  <div className="flex items-center gap-2 mt-2">
                    <span className={`text-xs px-2 py-1 rounded ${
                      tip.impact === 'HIGH' ? 'bg-red-100 text-red-700' :
                      tip.impact === 'MEDIUM' ? 'bg-yellow-100 text-yellow-700' :
                      'bg-blue-100 text-blue-700'
                    }`}>
                      {tip.impact} Impact
                    </span>
                    <span className="text-xs text-gray-500">
                      {tip.effort} effort
                    </span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
} 