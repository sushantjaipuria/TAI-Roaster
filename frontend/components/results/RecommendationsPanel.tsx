import React from 'react'
import { ResultsComponentProps } from '../../lib/types-results'

export default function RecommendationsPanel({ data }: ResultsComponentProps) {
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold text-gray-900">📌 Recommendations & Insights</h2>
          <p className="text-sm text-gray-600 mt-1">
            AI-powered suggestions for portfolio optimization
          </p>
        </div>
      </div>

      <div className="space-y-4">
        {data.recommendations.map((recommendation, index) => (
          <div key={index} className="border border-gray-200 rounded-lg p-4">
            <p className="text-gray-700">{recommendation}</p>
          </div>
        ))}
      </div>

      {data.recommendations.length === 0 && (
        <div className="text-center py-12">
          <p className="text-gray-500">No specific recommendations available</p>
        </div>
      )}
    </div>
  )
} 