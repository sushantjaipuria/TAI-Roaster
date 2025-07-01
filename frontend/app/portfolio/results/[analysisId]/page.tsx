'use client'

import React, { useEffect, useState } from 'react'
import { useParams } from 'next/navigation'
import { PortfolioAnalysisDetailed } from '../../../../lib/types-results'
import { getAnalysisResultsWithMetadata, AnalysisResultWithMetadata } from '../../../../lib/api-results'
import { AlertTriangle, RefreshCw, Info } from 'lucide-react'

// Enhanced dashboard component
import EnhancedDashboard from '../../../../components/analysis/EnhancedDashboard'

export default function PortfolioResultsPage() {
  const params = useParams()
  const analysisId = params.analysisId as string
  
  const [analysisResult, setAnalysisResult] = useState<AnalysisResultWithMetadata | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [retrying, setRetrying] = useState(false)

  const fetchResults = async (isRetry: boolean = false) => {
    try {
      if (isRetry) {
        setRetrying(true)
      } else {
        setLoading(true)
      }
      setError(null)
      
      const result = await getAnalysisResultsWithMetadata(analysisId)
      
      if (!result.data) {
        throw new Error('No analysis data received from API')
      }
      
      // Cast to detailed type and use directly
      const results = result.data as PortfolioAnalysisDetailed
      console.log('📊 Analysis Results:', results) // Debug log
      console.log('🔍 Demo Data Detection:', result.isDemoData) // Debug log
      
      setAnalysisResult(result)
      
    } catch (err) {
      console.error('Error fetching results:', err)
      setError(err instanceof Error ? err.message : 'Unknown error occurred')
    } finally {
      setLoading(false)
      setRetrying(false)
    }
  }

  useEffect(() => {
    if (analysisId) {
      fetchResults()
    }
  }, [analysisId])

  const handleRetry = () => {
    fetchResults(true)
  }

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
          <div className="space-y-3">
            <button 
              onClick={handleRetry}
              disabled={retrying}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center mx-auto"
            >
              {retrying ? (
                <>
                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                  Retrying...
                </>
              ) : (
                <>
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Retry Analysis
                </>
              )}
            </button>
            <button 
              onClick={() => window.location.reload()} 
              className="px-6 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors font-medium"
            >
              Reload Page
            </button>
          </div>
        </div>
      </div>
    )
  }

  if (!analysisResult) {
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

  return (
    <div>
      {/* Demo Data Warning Banner */}
      {analysisResult.isDemoData && (
        <div className="bg-yellow-50 border-b border-yellow-200">
          <div className="max-w-7xl mx-auto px-4 py-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <AlertTriangle className="w-5 h-5 text-yellow-600 mr-2" />
                <div>
                  <h3 className="text-sm font-medium text-yellow-800">
                    Demo Data Warning
                  </h3>
                  <p className="text-sm text-yellow-700">
                    This analysis is using demo/fallback data. Real market data may not be available.
                  </p>
                </div>
              </div>
              <div className="flex items-center space-x-3">
                <button
                  onClick={handleRetry}
                  disabled={retrying}
                  className="inline-flex items-center px-3 py-1.5 border border-yellow-300 text-xs font-medium rounded text-yellow-700 bg-yellow-100 hover:bg-yellow-200 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {retrying ? (
                    <>
                      <RefreshCw className="w-3 h-3 mr-1 animate-spin" />
                      Retrying...
                    </>
                  ) : (
                    <>
                      <RefreshCw className="w-3 h-3 mr-1" />
                      Retry with Real Data
                    </>
                  )}
                </button>
                <button
                  onClick={() => {
                    // Hide the warning for this session
                    const warning = document.querySelector('.bg-yellow-50') as HTMLElement
                    if (warning) {
                      warning.style.display = 'none'
                    }
                  }}
                  className="text-yellow-600 hover:text-yellow-800"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Main Dashboard */}
      <EnhancedDashboard data={analysisResult.data} />
    </div>
  )
} 