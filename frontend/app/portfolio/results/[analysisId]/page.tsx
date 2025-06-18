'use client'

import React, { useEffect, useState } from 'react'
import { useParams } from 'next/navigation'
import { PortfolioAnalysisDetailed } from '../../../../lib/types-results'
import { getAnalysisResults } from '../../../../lib/api-results'
import { enhanceAnalysisData } from '../../../../lib/mock-data-enhancer'

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
    const fetchAnalysisResults = async () => {
      try {
        setLoading(true)
        setError(null)
        
        console.log('🔍 Fetching analysis results for:', analysisId)
        
        // Fetch basic analysis results from backend
        const basicResults = await getAnalysisResults(analysisId)
        
        console.log('✅ Received basicResults:', basicResults)
        console.log('📊 Basic results type:', typeof basicResults)
        console.log('🗂️ Basic results keys:', basicResults ? Object.keys(basicResults) : 'null/undefined')
        
        if (!basicResults) {
          throw new Error('No analysis data received from API')
        }
        
        // Enhance with mock data for missing features
        console.log('🔄 Calling enhanceAnalysisData...')
        const enhancedResults = enhanceAnalysisData(basicResults)
        
        console.log('✨ Enhanced results:', enhancedResults)
        console.log('📈 Enhanced results type:', typeof enhancedResults)
        console.log('🗂️ Enhanced results keys:', enhancedResults ? Object.keys(enhancedResults) : 'null/undefined')
        
        if (!enhancedResults) {
          throw new Error('enhanceAnalysisData returned null/undefined')
        }
        
        setAnalysisData(enhancedResults)
      } catch (err) {
        console.error('❌ Error in fetchAnalysisResults:', err)
        setError(err instanceof Error ? err.message : 'Failed to load analysis results')
      } finally {
        setLoading(false)
      }
    }

    if (analysisId) {
      fetchAnalysisResults()
    }
  }, [analysisId])

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading portfolio analysis...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <p className="text-red-600 mb-4">Error: {error}</p>
          <button 
            onClick={() => window.location.reload()}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  if (!analysisData) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <p className="text-gray-600">No analysis data found</p>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <ResultsHeader analysisId={analysisId} data={analysisData} />
      
      <div className="max-w-7xl mx-auto px-4 py-8 space-y-8">
        {/* Executive Summary */}
        <PortfolioSummaryCard data={analysisData} />
        
        {/* Asset Allocation & Diversification */}
        <AllocationDashboard data={analysisData} />
        
        {/* Performance Metrics */}
        <PerformanceMetrics data={analysisData} />
        
        {/* Stock-Level Insights */}
        <StockInsightCards data={analysisData} />
        
        {/* Fundamental Insights */}
        <FundamentalInsights data={analysisData} />
        
        {/* Recommendations & Insights */}
        <RecommendationsPanel data={analysisData} />
        
        {/* Portfolio Rating & Hygiene Report */}
        <PortfolioRating data={analysisData} />
        
        {/* Pros, Cons, and Action Plan */}
        <ActionPlan data={analysisData} />
      </div>
      
      <ResultsFooter />
    </div>
  )
} 