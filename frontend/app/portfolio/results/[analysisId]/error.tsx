'use client'

import { useEffect } from 'react'
import { AlertTriangle, RefreshCcw, Home } from 'lucide-react'

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  useEffect(() => {
    // Log the error to an error reporting service
    console.error('Portfolio results error:', error)
  }, [error])

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center px-4">
      <div className="max-w-md w-full bg-white rounded-lg shadow-sm border border-gray-200 p-8 text-center">
        <div className="mb-6">
          <div className="mx-auto w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mb-4">
            <AlertTriangle className="w-8 h-8 text-red-600" />
          </div>
          <h1 className="text-xl font-semibold text-gray-900 mb-2">
            Failed to Load Analysis
          </h1>
          <p className="text-gray-600">
            We encountered an error while loading your portfolio analysis. This could be due to:
          </p>
        </div>

        <div className="mb-6 text-left">
          <ul className="text-sm text-gray-600 space-y-1">
            <li>• Analysis still in progress</li>
            <li>• Network connectivity issues</li>
            <li>• Server temporarily unavailable</li>
            <li>• Invalid analysis ID</li>
          </ul>
        </div>

        <div className="space-y-3">
          <button
            onClick={reset}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
          >
            <RefreshCcw className="w-4 h-4" />
            Try Again
          </button>
          
          <button
            onClick={() => window.location.href = '/'}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50 transition-colors"
          >
            <Home className="w-4 h-4" />
            Back to Home
          </button>
        </div>

        {error.digest && (
          <div className="mt-6 p-3 bg-gray-50 rounded-md">
            <p className="text-xs text-gray-500">
              Error ID: {error.digest}
            </p>
          </div>
        )}
      </div>
    </div>
  )
} 