import React, { useState } from 'react'
import { format } from 'date-fns'
import { Download, Calendar, TrendingUp, AlertCircle } from 'lucide-react'
import { ResultsComponentProps } from '../../lib/types-results'
import { exportAnalysisPDF, downloadPDF } from '../../lib/api-results'

interface ResultsHeaderProps extends ResultsComponentProps {
  analysisId: string
}

export default function ResultsHeader({ analysisId, data }: ResultsHeaderProps) {
  const [isExporting, setIsExporting] = useState(false)

  const handleExportPDF = async () => {
    try {
      setIsExporting(true)
      const pdfBlob = await exportAnalysisPDF(analysisId)
      const filename = `portfolio-analysis-${format(new Date(), 'yyyy-MM-dd')}.pdf`
      downloadPDF(pdfBlob, filename)
    } catch (error) {
      console.error('Failed to export PDF:', error)
      // Show error notification
    } finally {
      setIsExporting(false)
    }
  }

  const getReturnColor = (returnPct: number) => {
    if (returnPct > 0) return 'text-green-600'
    if (returnPct < 0) return 'text-red-600'
    return 'text-gray-600'
  }

  const getReturnIcon = (returnPct: number) => {
    if (returnPct > 0) return <TrendingUp className="w-4 h-4 text-green-600" />
    if (returnPct < 0) return <TrendingUp className="w-4 h-4 text-red-600 rotate-180" />
    return <TrendingUp className="w-4 h-4 text-gray-600" />
  }

  return (
    <div className="bg-white border-b border-gray-200 sticky top-0 z-10">
      <div className="max-w-7xl mx-auto px-4 py-4">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          {/* Left Section - Portfolio Info */}
          <div className="flex-1">
            <div className="flex items-start gap-4">
              <div className="flex-1">
                <h1 className="text-2xl font-bold text-gray-900 mb-1">
                  {data.portfolioName}
                </h1>
                
                <div className="flex flex-wrap items-center gap-4 text-sm text-gray-600">
                  <div className="flex items-center gap-1">
                    <Calendar className="w-4 h-4" />
                    <span>Analysis Date: {format(new Date(data.analysisDate), 'MMM d, yyyy')}</span>
                  </div>
                  
                  <div className="flex items-center gap-1">
                    <span>Analysis ID:</span>
                    <code className="px-2 py-1 bg-gray-100 rounded text-xs font-mono">
                      {analysisId.slice(-8)}
                    </code>
                  </div>
                </div>
              </div>
              
              {/* Quick Performance Summary */}
              <div className="flex items-center gap-4 text-right">
                <div>
                  <div className="text-xs text-gray-500 uppercase tracking-wide">Total Value</div>
                  <div className="text-lg font-semibold text-gray-900">
                    ₹{data.currentValue.toLocaleString('en-IN')}
                  </div>
                </div>
                
                <div>
                  <div className="text-xs text-gray-500 uppercase tracking-wide">Return</div>
                  <div className={`text-lg font-semibold flex items-center gap-1 ${getReturnColor(data.absoluteReturnPct)}`}>
                    {getReturnIcon(data.absoluteReturnPct)}
                    {data.absoluteReturnPct > 0 ? '+' : ''}{data.absoluteReturnPct.toFixed(1)}%
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Right Section - Actions */}
          <div className="flex items-center gap-3">
            {/* Portfolio Score Badge */}
            <div className="flex items-center gap-2 px-3 py-2 bg-blue-50 rounded-lg">
              <div className="text-xs text-blue-600 uppercase tracking-wide font-medium">
                TAI Score
              </div>
              <div className="text-lg font-bold text-blue-600">
                {Math.round(data.rating.taiScore)}
              </div>
            </div>

            {/* Export PDF Button */}
            <button
              onClick={handleExportPDF}
              disabled={isExporting}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Download className="w-4 h-4" />
              {isExporting ? 'Exporting...' : 'Export PDF'}
            </button>
          </div>
        </div>

        {/* Status Indicators */}
        {(data.riskWarnings.length > 0 || data.opportunities.length > 0) && (
          <div className="mt-4 flex flex-wrap gap-2">
            {data.riskWarnings.length > 0 && (
              <div className="flex items-center gap-1 px-2 py-1 bg-yellow-50 text-yellow-700 rounded-md text-xs">
                <AlertCircle className="w-3 h-3" />
                <span>{data.riskWarnings.length} Risk Warning{data.riskWarnings.length > 1 ? 's' : ''}</span>
              </div>
            )}
            
            {data.opportunities.length > 0 && (
              <div className="flex items-center gap-1 px-2 py-1 bg-green-50 text-green-700 rounded-md text-xs">
                <TrendingUp className="w-3 h-3" />
                <span>{data.opportunities.length} Opportunit{data.opportunities.length > 1 ? 'ies' : 'y'}</span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
} 