/**
 * Enhanced Analysis Hook for TAI-Roaster
 * Provides functionality to request and manage enhanced portfolio analysis
 */

import { useState, useCallback } from 'react';
import axios from 'axios';
import { 
  EnhancedAnalysisRequest, 
  EnhancedAnalysisResponse, 
  AnalysisStatus,
  AnalysisError 
} from '../types/enhanced-analysis';

interface UseEnhancedAnalysisReturn {
  // State
  isLoading: boolean;
  analysisData: EnhancedAnalysisResponse | null;
  error: string | null;
  progress: number;
  
  // Actions
  runAnalysis: (request: EnhancedAnalysisRequest) => Promise<void>;
  clearResults: () => void;
  
  // Status
  isAnalysisAvailable: boolean;
}

export const useEnhancedAnalysis = (): UseEnhancedAnalysisReturn => {
  const [isLoading, setIsLoading] = useState(false);
  const [analysisData, setAnalysisData] = useState<EnhancedAnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [isAnalysisAvailable, setIsAnalysisAvailable] = useState(true);

  const runAnalysis = useCallback(async (request: EnhancedAnalysisRequest) => {
    setIsLoading(true);
    setError(null);
    setProgress(0);
    
    try {
      // Check if enhanced analysis is available
      const statusResponse = await axios.get('/api/analysis/models/status');
      
      if (!statusResponse.data.enhanced_analysis_available) {
        setIsAnalysisAvailable(false);
        throw new Error('Enhanced analysis service is not available. Please ensure the intelligence module is properly installed.');
      }
      
      setIsAnalysisAvailable(true);
      setProgress(20);
      
      // Request enhanced analysis
      const response = await axios.post<EnhancedAnalysisResponse>(
        '/api/analysis/enhanced',
        request,
        {
          headers: {
            'Content-Type': 'application/json',
          },
          timeout: 60000, // 60 seconds timeout
          onUploadProgress: (progressEvent) => {
            const progress = progressEvent.total 
              ? Math.round((progressEvent.loaded * 100) / progressEvent.total)
              : 50;
            setProgress(Math.min(progress, 90));
          }
        }
      );
      
      setProgress(100);
      setAnalysisData(response.data);
      
    } catch (err: any) {
      console.error('Enhanced analysis error:', err);
      
      let errorMessage = 'Failed to perform enhanced analysis';
      
      if (axios.isAxiosError(err)) {
        if (err.response?.status === 503) {
          errorMessage = 'Enhanced analysis service is currently unavailable';
        } else if (err.response?.status === 422) {
          errorMessage = 'Invalid portfolio data provided';
        } else if (err.response?.data?.detail) {
          errorMessage = err.response.data.detail;
        } else if (err.message) {
          errorMessage = err.message;
        }
      } else if (err.message) {
        errorMessage = err.message;
      }
      
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const clearResults = useCallback(() => {
    setAnalysisData(null);
    setError(null);
    setProgress(0);
  }, []);

  return {
    isLoading,
    analysisData,
    error,
    progress,
    runAnalysis,
    clearResults,
    isAnalysisAvailable
  };
};

// Helper hook for checking service status
export const useEnhancedAnalysisStatus = () => {
  const [status, setStatus] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  
  const checkStatus = useCallback(async () => {
    setLoading(true);
    try {
      const response = await axios.get('/api/analysis/models/status');
      setStatus(response.data);
    } catch (error) {
      console.error('Failed to check analysis status:', error);
      setStatus({ enhanced_analysis_available: false, error: 'Failed to check status' });
    } finally {
      setLoading(false);
    }
  }, []);
  
  return { status, loading, checkStatus };
}; 