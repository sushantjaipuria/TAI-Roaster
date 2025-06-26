'use client'

import React, { useEffect, useState } from 'react'
import { useParams } from 'next/navigation'
import { PortfolioAnalysisDetailed } from '../../../../lib/types-results'
import { getAnalysisResults } from '../../../../lib/api-results'

// Enhanced dashboard component
import EnhancedDashboard from '../../../../components/analysis/EnhancedDashboard'

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
        console.log('📊 Analysis Results:', results) // Debug log
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
          <p className="text-gray-600">Loading your enhanced portfolio analysis...</p>
          <p className="text-sm text-gray-500 mt-2">Processing AI-powered insights and real-time data...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center max-w-md mx-auto">
          <div className="bg-red-100 rounded-full w-16 h-16 mx-auto mb-4 flex items-center justify-center">
            <svg className="w-8 h-8 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.5 0L4.268 16.5c-.77.833.192 2.5 1.732 2.5z" />
            </svg>
          </div>
          <h1 className="text-2xl font-bold text-red-600 mb-4">Analysis Loading Error</h1>
          <p className="text-gray-600 mb-4">{error}</p>
          <button 
            onClick={() => window.location.reload()} 
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium"
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
        <div className="text-center max-w-md mx-auto">
          <div className="bg-gray-100 rounded-full w-16 h-16 mx-auto mb-4 flex items-center justify-center">
            <svg className="w-8 h-8 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <h1 className="text-2xl font-bold text-gray-800 mb-4">Analysis Not Found</h1>
          <p className="text-gray-600">Could not load the requested portfolio analysis.</p>
          <p className="text-sm text-gray-500 mt-2">The analysis may have expired or the ID is invalid.</p>
        </div>
      </div>
    )
  }

  return <EnhancedDashboard data={analysisData} />
} 