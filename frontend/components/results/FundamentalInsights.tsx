import React from 'react'
import { ResultsComponentProps } from '../../lib/types-results'

export default function FundamentalInsights({ data }: ResultsComponentProps) {
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold text-gray-900">🧳 Fundamental Insights</h2>
          <p className="text-sm text-gray-600 mt-1">
            Financial health and business fundamentals analysis
          </p>
        </div>
      </div>

      <div className="text-center py-12">
        <p className="text-gray-500">Fundamental analysis will be implemented here</p>
        <p className="text-sm text-gray-400 mt-2">
          Revenue, PAT growth, PE ratios, ROE, cash flow analysis, and red flags
        </p>
      </div>
    </div>
  )
} 