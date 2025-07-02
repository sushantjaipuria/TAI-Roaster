/**
 * Enhanced API Client for Backend Communication
 * 
 * This module handles all HTTP communication with the FastAPI backend:
 * - Portfolio validation and analysis requests
 * - File upload and parsing
 * - User profile management
 * - Error handling and retry logic
 * - Request/response type safety
 * 
 * API endpoints:
 * - POST /api/portfolio/upload: Upload portfolio files
 * - POST /api/portfolio/validate: Validate portfolio data
 * - POST /api/portfolio/analyze: Start portfolio analysis
 * - GET /api/portfolio/sample-format: Get sample file format
 * - GET /api/portfolio/suggestions: Get portfolio suggestions
 * 
 * Features:
 * - TypeScript interfaces for request/response types
 * - Axios interceptors for error handling
 * - Request retry logic with exponential backoff
 * - Loading state management
 * - File upload progress tracking
 * - Request cancellation support
 */

import axios, { AxiosInstance, AxiosResponse, AxiosProgressEvent, CancelTokenSource } from 'axios'
import { 
  ApiResponse, 
  UserProfile, 
  PortfolioInput,
  PortfolioHolding,
  AnalysisRequest,
  FileParseResponse,
  PortfolioValidationResponse,
  FileUploadRequest,
  OnboardingApiResponse,
  UserProfileApiRequest,
  PortfolioAnalysis
} from './types'

// =============================================================================
// API CONFIGURATION
// =============================================================================

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
const API_TIMEOUT = 30000 // 30 seconds
const UPLOAD_TIMEOUT = 120000 // 2 minutes for file uploads
const ANALYSIS_TIMEOUT = 60000 // 60 seconds for portfolio analysis

// Create axios instance
const api: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: API_TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
  },
})

// =============================================================================
// REQUEST/RESPONSE INTERCEPTORS
// =============================================================================

// Request interceptor for debugging and auth
api.interceptors.request.use(
  (config) => {
    // Log requests in development
    if (process.env.NODE_ENV === 'development') {
      console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`)
    }
    
    // Add authentication headers if available
    const token = localStorage.getItem('auth_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    
    return config
  },
  (error) => {
    console.error('❌ Request Error:', error)
    return Promise.reject(error)
  }
)

// Response interceptor for error handling
api.interceptors.response.use(
  (response: AxiosResponse) => {
    // Log responses in development
    if (process.env.NODE_ENV === 'development') {
      console.log(`✅ API Response: ${response.status} ${response.config.url}`)
    }
    return response
  },
  (error) => {
    // Enhanced error logging for display
    const errorInfo = {
      status: error.response?.status,
      message: error.response?.data?.message || error.message,
      url: error.config?.url,
      method: error.config?.method,
      timestamp: new Date().toISOString(),
      errorType: error?.constructor?.name || 'Unknown',
      hasResponse: !!error?.response,
      hasConfig: !!error?.config,
      hasRequest: !!error?.request
    }
    
    console.error('❌ API Error:', errorInfo)
    
    // Handle specific error cases
    if (error.response?.status === 401) {
      // Handle unauthorized - redirect to login or refresh token
      localStorage.removeItem('auth_token')
      // You could emit an event or redirect here
    } else if (error.response?.status >= 500) {
      // Handle server errors
      console.error('Server error occurred')
    }
    
    return Promise.reject(error)
  }
)

// =============================================================================
// API CLIENT CLASS
// =============================================================================

export class PortfolioApiClient {
  private static cancelTokens: Map<string, CancelTokenSource> = new Map()
  
  // =============================================================================
  // FILE UPLOAD METHODS
  // =============================================================================
  
  /**
   * Upload and parse portfolio file
   */
  static async uploadPortfolioFile(
    file: File,
    onProgress?: (progress: number) => void
  ): Promise<FileParseResponse> {
    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('filename', file.name)
      formData.append('fileSize', file.size.toString())
      formData.append('contentType', file.type)
      
      // Create cancel token
      const cancelToken = axios.CancelToken.source()
      this.cancelTokens.set('upload', cancelToken)
      
      const response = await api.post<FileParseResponse>('/api/portfolio/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: UPLOAD_TIMEOUT,
        cancelToken: cancelToken.token,
        onUploadProgress: (progressEvent: AxiosProgressEvent) => {
          if (progressEvent.total && onProgress) {
            const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total)
            onProgress(progress)
          }
        }
      })
      
      this.cancelTokens.delete('upload')
      return response.data
      
    } catch (error: any) {
      this.cancelTokens.delete('upload')
      throw this.handleError(error, 'Failed to upload portfolio file')
    }
  }
  
  /**
   * Cancel file upload
   */
  static cancelUpload(): void {
    const cancelToken = this.cancelTokens.get('upload')
    if (cancelToken) {
      cancelToken.cancel('Upload cancelled by user')
      this.cancelTokens.delete('upload')
    }
  }
  
  /**
   * Get sample file format information
   */
  static async getSampleFormat(): Promise<any> {
    try {
      const response = await api.get('/api/portfolio/sample-format')
      return response.data.data || response.data
    } catch (error: any) {
      throw this.handleError(error, 'Failed to get sample format')
    }
  }
  
  // =============================================================================
  // ONBOARDING METHODS
  // =============================================================================
  
  /**
   * Transform frontend UserProfile to backend API request format
   */
  private static transformUserProfileToApiRequest(userProfile: UserProfile): UserProfileApiRequest {
    return {
      risk_tolerance: userProfile.riskTolerance,
      investment_amount: userProfile.investmentAmount,
      investment_style: userProfile.investmentStyle,
      time_horizon: userProfile.timeHorizon,
      goals: userProfile.goals
      // Note: age and annualIncome are optional frontend fields not sent to backend
    }
  }

  /**
   * Transform frontend AnalysisRequest to backend API request format
   */
  private static transformAnalysisRequestToApiRequest(
    portfolio: PortfolioInput,
    userProfile: UserProfile,
    analysisType: string = 'comprehensive',
    includeRecommendations: boolean = true
  ) {
    return {
      portfolio,
      user_profile: this.transformUserProfileToApiRequest(userProfile),
      analysis_type: analysisType,
      include_recommendations: includeRecommendations
    }
  }
  
  /**
   * Submit user onboarding information
   */
  static async submitOnboarding(userProfile: UserProfile): Promise<OnboardingApiResponse> {
    try {
      // Transform frontend camelCase to backend snake_case
      const apiRequest = this.transformUserProfileToApiRequest(userProfile)
      
      const response = await api.post<OnboardingApiResponse>('/api/onboarding/', apiRequest)
      
      return response.data
      
    } catch (error: any) {
      throw this.handleError(error, 'Failed to submit onboarding information')
    }
  }
  
  // =============================================================================
  // PORTFOLIO VALIDATION METHODS
  // =============================================================================
  
  /**
   * Validate portfolio data
   */
  static async validatePortfolio(
    portfolio: PortfolioInput,
    userProfile: UserProfile
  ): Promise<PortfolioValidationResponse> {
    try {
      // Transform frontend camelCase to backend snake_case
      const apiRequest = this.transformAnalysisRequestToApiRequest(
        portfolio,
        userProfile,
        'comprehensive',
        false
      )
      
      const response = await api.post<PortfolioValidationResponse>('/api/portfolio/validate', apiRequest)
      return response.data
      
    } catch (error: any) {
      throw this.handleError(error, 'Failed to validate portfolio')
    }
  }
  
  /**
   * Get portfolio improvement suggestions
   */
  static async getPortfolioSuggestions(
    portfolio?: PortfolioInput
  ): Promise<string[]> {
    try {
      const params = portfolio ? { portfolio_data: JSON.stringify(portfolio) } : {}
      const response = await api.get('/api/portfolio/suggestions', { params })
      return response.data.suggestions || []
      
    } catch (error: any) {
      throw this.handleError(error, 'Failed to get portfolio suggestions')
    }
  }
  
  // =============================================================================
  // PORTFOLIO ANALYSIS METHODS
  // =============================================================================
  
  /**
   * Start comprehensive portfolio analysis
   */
  static async analyzePortfolio(
    portfolio: PortfolioInput,
    userProfile: UserProfile,
    analysisType: 'comprehensive' | 'quick' | 'risk_only' | 'diversification_only' = 'comprehensive'
  ): Promise<{ analysisId: string; estimatedProcessingTime: string }> {
    try {
      // Transform frontend camelCase to backend snake_case
      const apiRequest = this.transformAnalysisRequestToApiRequest(
        portfolio,
        userProfile,
        analysisType,
        true
      )
      
      const response = await api.post('/api/portfolio/analyze', apiRequest, {
        timeout: ANALYSIS_TIMEOUT
      })
      
      if (response.data.success) {
        const result = {
          analysisId: response.data.analysisId,
          estimatedProcessingTime: response.data.data?.estimatedProcessingTime || '2-3 minutes'
        }
        return result
      } else {
        throw new Error(response.data.message || 'Analysis request failed')
      }
      
    } catch (error: any) {
      throw this.handleError(error, 'Failed to start portfolio analysis')
    }
  }
  
  /**
   * Get analysis results
   */
  static async getAnalysisResults(analysisId: string): Promise<PortfolioAnalysis> {
    try {
      const response = await api.get(`/api/analysis/${analysisId}`)
      return response.data.analysis
    } catch (error: any) {
      throw this.handleError(error, 'Failed to get analysis results')
    }
  }

  /**
   * Get analysis status
   */
  static async getAnalysisStatus(analysisId: string): Promise<{ status: string; progress?: number }> {
    try {
      const response = await api.get(`/api/portfolio/process-status/${analysisId}`)
      return {
        status: response.data.status || 'unknown',
        progress: response.data.progress
      }
    } catch (error: any) {
      throw this.handleError(error, 'Failed to get analysis status')
    }
  }
  
  // =============================================================================
  // UTILITY METHODS
  // =============================================================================
  
  /**
   * Health check endpoint
   */
  static async healthCheck(): Promise<{ status: string; timestamp: string }> {
    try {
      const response = await api.get('/api/portfolio/health')
      return response.data
    } catch (error: any) {
      throw this.handleError(error, 'Health check failed')
    }
  }
  
  /**
   * Get system statistics
   */
  static async getStats(): Promise<any> {
    try {
      const response = await api.get('/api/portfolio/stats')
      return response.data.data
    } catch (error: any) {
      throw this.handleError(error, 'Failed to get system stats')
    }
  }
  
  /**
   * Clear system cache (development only)
   */
  static async clearCache(): Promise<void> {
    try {
      await api.delete('/api/portfolio/clear-cache')
    } catch (error: any) {
      throw this.handleError(error, 'Failed to clear cache')
    }
  }
  
  // =============================================================================
  // ERROR HANDLING
  // =============================================================================
  
  private static handleError(error: any, defaultMessage: string): Error {
    // Handle axios cancellation
    if (axios.isCancel(error)) {
      return new Error('Request was cancelled')
    }
    
    // Handle network errors
    if (error.code === 'NETWORK_ERROR' || !error.response) {
      return new Error('Network error. Please check your connection and try again.')
    }
    
    // Handle timeout errors
    if (error.code === 'ECONNABORTED') {
      return new Error('Request timed out. Please try again.')
    }
    
    // Handle HTTP errors
    if (error.response) {
      const status = error.response.status
      const data = error.response.data
      
      // Extract error message
      let message = defaultMessage
      
      if (data?.error) {
        message = data.error
      } else if (data?.message) {
        message = data.message
      } else if (data?.detail) {
        message = data.detail
      }
      
      // Handle specific status codes
      switch (status) {
        case 400:
          return new Error(`Invalid request: ${message}`)
        case 401:
          return new Error('Authentication required. Please log in.')
        case 403:
          return new Error('Access denied. You do not have permission.')
        case 404:
          return new Error('Resource not found.')
        case 413:
          return new Error('File too large. Please try a smaller file.')
        case 415:
          return new Error('Unsupported file type.')
        case 422:
          // Validation errors - Handle FastAPI's structured validation error format
          if (data?.detail && Array.isArray(data.detail)) {
            const errorMessages = data.detail.map((err: any) => {
              // Extract the actual error message from FastAPI's format
              if (err.msg) {
                const field = err.loc ? err.loc.join('.') : 'field'
                return `${field}: ${err.msg}`
              }
              if (err.message) return err.message
              if (typeof err === 'string') return err
              return JSON.stringify(err)
            }).join('\n• ')
            return new Error(`Validation errors:\n• ${errorMessages}`)
          } else if (data?.detail && typeof data.detail === 'string') {
            return new Error(`Validation failed: ${data.detail}`)
          } else if (data?.errors && Array.isArray(data.errors)) {
            const errorMessages = data.errors.map((err: any) => err.message || err).join(', ')
            return new Error(`Validation failed: ${errorMessages}`)
          }
          return new Error(`Validation failed: ${message}`)
        case 429:
          return new Error('Too many requests. Please try again later.')
        case 500:
          return new Error('Server error occurred. Please try again later.')
        case 503:
          return new Error('Service temporarily unavailable. Please try again later.')
        default:
          return new Error(message)
      }
    }
    
    // Fallback error
    return new Error(error.message || defaultMessage)
  }
}

// =============================================================================
// CONVENIENCE METHODS
// =============================================================================

/**
 * Upload portfolio file with progress tracking
 */
export const uploadPortfolioFile = (
  file: File,
  onProgress?: (progress: number) => void
): Promise<FileParseResponse> => {
  return PortfolioApiClient.uploadPortfolioFile(file, onProgress)
}

/**
 * Validate portfolio data
 */
export const validatePortfolio = (
  portfolio: PortfolioInput,
  userProfile: UserProfile
): Promise<PortfolioValidationResponse> => {
  return PortfolioApiClient.validatePortfolio(portfolio, userProfile)
}

/**
 * Start portfolio analysis
 */
export const analyzePortfolio = (
  portfolio: PortfolioInput,
  userProfile: UserProfile
): Promise<{ analysisId: string; estimatedProcessingTime: string }> => {
  return PortfolioApiClient.analyzePortfolio(portfolio, userProfile)
}

/**
 * Get sample file format
 */
export const getSampleFormat = (): Promise<any> => {
  return PortfolioApiClient.getSampleFormat()
}

/**
 * Cancel ongoing upload
 */
export const cancelUpload = (): void => {
  PortfolioApiClient.cancelUpload()
}

/**
 * Submit user onboarding information
 */
export const submitOnboarding = (userProfile: UserProfile): Promise<OnboardingApiResponse> => {
  return PortfolioApiClient.submitOnboarding(userProfile)
}

// Export axios instance for direct use if needed
export { api }

// Export API client class as both named and default export
export { PortfolioApiClient as ApiClient }
export default PortfolioApiClient