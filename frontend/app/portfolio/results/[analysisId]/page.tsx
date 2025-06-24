'use client'

import React, { useEffect, useState } from 'react'
import { useParams } from 'next/navigation'
import { PortfolioAnalysisDetailed } from '../../../../lib/types-results'
import { getAnalysisResults } from '../../../../lib/api-results'

// WHOOP-inspired dashboard components
import PortfolioHealthDashboard from '../../../../components/results/PortfolioHealthDashboard'
import PerformanceStorySection from '../../../../components/results/PerformanceStorySection'
import InsightsAndRisksSection from '../../../../components/results/InsightsAndRisksSection'
import EnhancedStockInsights from '../../../../components/results/EnhancedStockInsights'

// Enhanced dashboard components
import AllocationDashboard from '../../../../components/results/AllocationDashboard'
import PerformanceMetrics from '../../../../components/results/PerformanceMetrics'
import ActionPlan from '../../../../components/results/ActionPlan'
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
      {/* WHOOP-Style Portfolio Health Dashboard */}
      <PortfolioHealthDashboard data={analysisData} />
      
      {/* Performance Story Section */}
      <PerformanceStorySection data={analysisData} />
      
      {/* Insights and Risks Analysis */}
      <InsightsAndRisksSection data={analysisData} />

      {/* Enhanced Stock Insights */}
      <EnhancedStockInsights data={analysisData} />

      {/* Enhanced Portfolio Analysis */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 bg-white">
        <div className="text-center mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Detailed Portfolio Analysis</h2>
          <p className="text-gray-600">Deep dive into allocation, performance metrics, and action plans</p>
        </div>

        {/* Performance & Allocation */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          <PerformanceMetrics data={analysisData} />
          <AllocationDashboard data={analysisData} />
        </div>

        {/* Action Plan */}
        <div className="max-w-4xl mx-auto">
          <ActionPlan data={analysisData} />
        </div>
      </div>

      {/* Results Footer */}
      <ResultsFooter />
    </div>
  )
} 