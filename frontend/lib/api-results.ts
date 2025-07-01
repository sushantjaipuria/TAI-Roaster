/**
 * API Client for Portfolio Analysis Results
 * Handles fetching analysis results and related operations
 */

import { PortfolioAnalysis } from './types'
import { PortfolioAnalysisDetailed, PDFExportOptions } from './types-results'
import { PortfolioApiClient } from './api'

// =============================================================================
// DEMO DATA DETECTION
// =============================================================================

export interface AnalysisResultWithMetadata {
  data: PortfolioAnalysis
  isDemoData: boolean
  error?: string
  retryCount?: number
}

/**
 * Detect if analysis data is demo/fallback data
 */
export const detectDemoData = (data: PortfolioAnalysis): boolean => {
  if (!data) return true
  
  // First check: Look for explicit real data markers
  const hasRealDataMetadata = (data as any).is_real_data === true || 
                             (data as any).data_source === "live_market_data" ||
                             (data as any).file_generated_by?.includes("intelligence_module")
  
  if (hasRealDataMetadata) {
    return false // Definitely real data
  }
  
  // Second check: Look for demo data indicators
  const indicators = [
    // Check if this is explicitly marked as demo
    (data as any).is_demo_data === true || 
    (data as any).data_source === "demo" ||
    (data as any).file_generated_by === "demo_generator",
    
    // Check if stocks are demo stocks (but only if we don't have real data markers)
    data.stocks?.some(stock => 
      ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK'].includes(stock.ticker)
    ) && !hasRealDataMetadata,
    
    // Check if recommendations are generic or empty
    !data.recommendations || data.recommendations.length === 0 || 
    data.recommendations.every(rec => 
      rec.includes('diversify') || rec.includes('consider') || rec.includes('review')
    ),
    
    // Check if overall score is default (0 or 50)
    data.overallScore === 0 || data.overallScore === 50,
    
    // Check if summary is generic
    !data.summary || data.summary.length < 50
  ]
  
  // If any indicator is true, likely demo data
  return indicators.some(indicator => indicator === true)
}

// =============================================================================
// ANALYSIS RESULTS API
// =============================================================================

export class ResultsApiClient {
  
  /**
   * Get analysis results by analysis ID with retry logic and demo detection
   */
  static async getAnalysisResults(
    analysisId: string, 
    maxRetries: number = 3
  ): Promise<AnalysisResultWithMetadata> {
    let lastError: string | undefined
    let retryCount = 0
    
    while (retryCount < maxRetries) {
      try {
        // Use existing API client method
        const response = await PortfolioApiClient.getAnalysisResults(analysisId)
        
        // Detect if this is demo data
        const isDemoData = detectDemoData(response)
        
        return {
          data: response,
          isDemoData,
          retryCount
        }
        
      } catch (error: any) {
        lastError = error.message
        retryCount++
        
        // Wait before retrying (exponential backoff)
        if (retryCount < maxRetries) {
          const delay = Math.min(1000 * Math.pow(2, retryCount - 1), 5000)
          await new Promise(resolve => setTimeout(resolve, delay))
        }
      }
    }
    
    // If all retries failed, throw error
    throw new Error(`Failed to fetch analysis results after ${maxRetries} attempts: ${lastError}`)
  }
  
  /**
   * Check analysis status
   */
  static async getAnalysisStatus(analysisId: string): Promise<{ 
    status: string; 
    progress?: number;
    error?: string 
  }> {
    try {
      return await PortfolioApiClient.getAnalysisStatus(analysisId)
    } catch (error: any) {
      throw new Error(`Failed to check analysis status: ${error.message}`)
    }
  }
  
  /**
   * Export analysis results as PDF
   */
  static async exportAnalysisPDF(
    analysisId: string, 
    options: PDFExportOptions = {
      includeCharts: true,
      includeDetailedStockAnalysis: true,
      includeActionPlan: true,
      format: 'DETAILED',
      branding: true
    }
  ): Promise<Blob> {
    try {
      // This would typically call a backend endpoint
      // For now, we'll prepare for future implementation
      const response = await fetch(`/api/analysis/${analysisId}/export`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(options)
      })
      
      if (!response.ok) {
        throw new Error('Failed to export PDF')
      }
      
      return await response.blob()
    } catch (error: any) {
      throw new Error(`Failed to export PDF: ${error.message}`)
    }
  }
  
  /**
   * Get historical performance data for charts
   */
  static async getHistoricalData(
    analysisId: string, 
    timeframe: '1M' | '3M' | '6M' | '1Y' | 'ALL' = 'ALL'
  ): Promise<any[]> {
    try {
      // This would fetch detailed historical data
      // For now, return empty array - will be filled by mock data enhancer
      return []
    } catch (error: any) {
      throw new Error(`Failed to fetch historical data: ${error.message}`)
    }
  }
}

// =============================================================================
// CONVENIENCE FUNCTIONS
// =============================================================================

/**
 * Get analysis results with enhanced data
 */
export const getAnalysisResults = async (analysisId: string): Promise<PortfolioAnalysis> => {
  const result = await ResultsApiClient.getAnalysisResults(analysisId)
  return result.data
}

/**
 * Get analysis results with metadata (including demo detection)
 */
export const getAnalysisResultsWithMetadata = async (analysisId: string): Promise<AnalysisResultWithMetadata> => {
  return ResultsApiClient.getAnalysisResults(analysisId)
}

/**
 * Check if analysis is complete
 */
export const isAnalysisComplete = async (analysisId: string): Promise<boolean> => {
  try {
    const status = await ResultsApiClient.getAnalysisStatus(analysisId)
    return status.status === 'COMPLETED'
  } catch {
    return false
  }
}

/**
 * Export analysis as PDF
 */
export const exportAnalysisPDF = async (
  analysisId: string, 
  options?: PDFExportOptions
): Promise<Blob> => {
  return ResultsApiClient.exportAnalysisPDF(analysisId, options)
}

/**
 * Download PDF file
 */
export const downloadPDF = (blob: Blob, filename: string = 'portfolio-analysis.pdf') => {
  const url = window.URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  window.URL.revokeObjectURL(url)
} 