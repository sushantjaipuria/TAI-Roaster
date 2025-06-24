/**
 * API Client for Portfolio Analysis Results
 * Handles fetching analysis results and related operations
 */

import { PortfolioAnalysis } from './types'
import { PortfolioAnalysisDetailed, PDFExportOptions } from './types-results'
import { PortfolioApiClient } from './api'

// =============================================================================
// ANALYSIS RESULTS API
// =============================================================================

export class ResultsApiClient {
  
  /**
   * Get analysis results by analysis ID
   */
  static async getAnalysisResults(analysisId: string): Promise<PortfolioAnalysis> {
    try {
      // Use existing API client method
      const response = await PortfolioApiClient.getAnalysisResults(analysisId)
      return response
    } catch (error: any) {
      throw new Error(`Failed to fetch analysis results: ${error.message}`)
    }
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