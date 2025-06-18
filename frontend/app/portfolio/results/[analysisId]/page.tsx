'use client'

import React, { useEffect, useState } from 'react'
import { useParams } from 'next/navigation'
import { PortfolioAnalysisDetailed } from '../../../../lib/types-results'
import { getAnalysisResults } from '../../../../lib/api-results'

// Import dashboard components (will be created in Phase 3)
import PortfolioSummaryCard from '../../../../components/results/PortfolioSummaryCard'
import AllocationDashboard from '../../../../components/results/AllocationDashboard'
import PerformanceMetrics from '../../../../components/results/PerformanceMetrics'
import StockInsightCards from '../../../../components/results/StockInsightCards'
import FundamentalInsights from '../../../../components/results/FundamentalInsights'
import RecommendationsPanel from '../../../../components/results/RecommendationsPanel'
import PortfolioRating from '../../../../components/results/PortfolioRating'
import ActionPlan from '../../../../components/results/ActionPlan'
import ResultsHeader from '../../../../components/results/ResultsHeader'
import ResultsFooter from '../../../../components/results/ResultsFooter'

export default function PortfolioResultsPage() {
  const params = useParams()
  const analysisId = params.analysisId as string
  
  const [analysisData, setAnalysisData] = useState<PortfolioAnalysisDetailed | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchResults = async () => {
      try {
        setLoading(true)
        
        const basicResults = await getAnalysisResults(analysisId)
        
        if (!basicResults) {
          throw new Error('No analysis data received from API')
        }
        
        // Cast to detailed type and use directly
        const results = basicResults as PortfolioAnalysisDetailed
        setAnalysisData(results)
        
      } catch (err) {
        console.error('Error fetching results:', err)
        setError(err instanceof Error ? err.message : 'Unknown error occurred')
      } finally {
        setLoading(false)
      }
    }

    if (analysisId) {
      fetchResults()
    }
  }, [analysisId])

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading your portfolio analysis...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-red-600 mb-4">Error Loading Analysis</h1>
          <p className="text-gray-600 mb-4">{error}</p>
          <button 
            onClick={() => window.location.reload()} 
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Try Again
          </button>
        </div>
      </div>
    )
  }

  if (!analysisData) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-gray-800 mb-4">Analysis Not Found</h1>
          <p className="text-gray-600">Could not load the requested portfolio analysis.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Results Header */}
      <ResultsHeader analysisId={analysisId} data={analysisData} />
      
      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Portfolio Summary Cards */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
          <div className="lg:col-span-2">
            <PortfolioSummaryCard data={analysisData} />
          </div>
          <div className="lg:col-span-1">
            <PortfolioRating data={analysisData} />
          </div>
        </div>

        {/* Performance & Allocation */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          <PerformanceMetrics data={analysisData} />
          <AllocationDashboard data={analysisData} />
        </div>

        {/* Stock Analysis */}
        <div className="mb-8">
          <StockInsightCards data={analysisData} />
        </div>

        {/* Fundamental Insights */}
        <div className="mb-8">
          <FundamentalInsights data={analysisData} />
        </div>

        {/* Recommendations & Action Plan */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          <RecommendationsPanel data={analysisData} />
          <ActionPlan data={analysisData} />
        </div>
      </div>

      {/* Results Footer */}
      <ResultsFooter data={analysisData} />
    </div>
  )
} 