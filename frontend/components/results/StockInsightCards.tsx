import React from 'react'
import { ResultsComponentProps } from '../../lib/types-results'

export default function StockInsightCards({ data }: ResultsComponentProps) {
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold text-gray-900">📋 Stock-Level Insights</h2>
          <p className="text-sm text-gray-600 mt-1">
            Detailed analysis of individual holdings
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {data.stocks.slice(0, 6).map((stock, index) => (
          <div key={stock.ticker} className="border border-gray-200 rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-lg font-semibold text-gray-900">{stock.ticker}</h3>
              <span className={`px-2 py-1 rounded text-xs font-medium ${
                stock.recommendation === 'buy' ? 'bg-green-100 text-green-800' :
                stock.recommendation === 'hold' ? 'bg-yellow-100 text-yellow-800' :
                'bg-red-100 text-red-800'
              }`}>
                {stock.recommendation.toUpperCase()}
              </span>
            </div>
            
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600">Score:</span>
                <span className="font-medium">{stock.score.toFixed(1)}/100</span>
              </div>
              
              <div className="flex justify-between">
                <span className="text-gray-600">Current Price:</span>
                <span className="font-medium">₹{stock.currentPrice?.toFixed(2) || 'N/A'}</span>
              </div>
              
              <div className="flex justify-between">
                <span className="text-gray-600">Target Price:</span>
                <span className="font-medium">₹{stock.brokerTargets?.averageTarget?.toFixed(2) || 'N/A'}</span>
              </div>
              
              {stock.brokerTargets?.upside !== undefined && (
                <div className="flex justify-between">
                  <span className="text-gray-600">Expected Upside:</span>
                  <span className={`font-medium ${stock.brokerTargets.upside >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {stock.brokerTargets.upside >= 0 ? '+' : ''}{stock.brokerTargets.upside.toFixed(1)}%
                  </span>
                </div>
              )}
            </div>
            
            <div className="mt-3 pt-3 border-t border-gray-200">
              <p className="text-xs text-gray-600 line-clamp-3">
                {stock.reasoning}
              </p>
            </div>
          </div>
        ))}
      </div>

      {data.stocks.length > 6 && (
        <div className="mt-6 text-center">
          <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
            View All {data.stocks.length} Holdings
          </button>
        </div>
      )}
    </div>
  )
} 