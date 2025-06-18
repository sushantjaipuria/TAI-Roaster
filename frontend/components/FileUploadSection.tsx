/**
 * File Upload Section Component
 * 
 * Handles file upload functionality for portfolio data:
 * - Drag and drop interface
 * - File validation and preview
 * - Upload progress tracking
 * - Error handling and user feedback
 * - Integration with backend file parser
 * 
 * Supported formats:
 * - CSV files (.csv)
 * - Excel files (.xlsx, .xls)
 * - TSV files (.tsv)
 * 
 * Features:
 * - Drag and drop file selection
 * - File format and size validation
 * - Upload progress indication
 * - Parsing result preview
 * - Error and warning display
 * - Sample format guidance
 */

'use client'

import React, { useState, useCallback, useRef } from 'react'
import { 
  FileUploadState, 
  PortfolioInput, 
  FileParseResponse,
  ApiResponse,
  FILE_UPLOAD_CONFIG 
} from '../lib/types'
import PortfolioApiClient from '../lib/api'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import { 
  Upload, 
  FileText, 
  CheckCircle, 
  AlertTriangle, 
  X, 
  Download,
  Info,
  Eye
} from 'lucide-react'

interface FileUploadSectionProps {
  onSuccess: (portfolio: PortfolioInput) => void
  state: FileUploadState
  onStateChange: (state: FileUploadState) => void
}

const FileUploadSection: React.FC<FileUploadSectionProps> = ({
  onSuccess,
  state,
  onStateChange
}) => {
  // =============================================================================
  // LOCAL STATE
  // =============================================================================
  
  const [dragActive, setDragActive] = useState(false)
  const [showSampleFormat, setShowSampleFormat] = useState(false)
  const [sampleData, setSampleData] = useState<any>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  
  // =============================================================================
  // UTILITY FUNCTIONS
  // =============================================================================
  
  const validateFile = useCallback((file: File): { isValid: boolean; errors: string[] } => {
    const errors: string[] = []
    
    // Check file size
    if (file.size > FILE_UPLOAD_CONFIG.maxFileSize) {
      errors.push(`File size (${(file.size / 1024 / 1024).toFixed(1)}MB) exceeds limit of ${FILE_UPLOAD_CONFIG.maxFileSize / 1024 / 1024}MB`)
    }
    
    // Check file type
    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase()
    if (!FILE_UPLOAD_CONFIG.acceptedFormats.includes(fileExtension)) {
      errors.push(`File type ${fileExtension} not supported. Accepted formats: ${FILE_UPLOAD_CONFIG.acceptedFormats.join(', ')}`)
    }
    
    // Check MIME type if available
    if (file.type && !FILE_UPLOAD_CONFIG.acceptedMimeTypes.includes(file.type)) {
      errors.push(`MIME type ${file.type} not supported`)
    }
    
    return {
      isValid: errors.length === 0,
      errors
    }
  }, [])
  
  const updateState = useCallback((updates: Partial<FileUploadState>) => {
    onStateChange({ ...state, ...updates })
  }, [state, onStateChange])
  
  // =============================================================================
  // EVENT HANDLERS
  // =============================================================================
  
  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(true)
  }, [])
  
  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
  }, [])
  
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
  }, [])
  
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    const files = Array.from(e.dataTransfer.files)
    if (files.length > 0) {
      handleFileSelect(files[0])
    }
  }, [])
  
  const handleFileSelect = useCallback((file: File) => {
    const validation = validateFile(file)
    
    if (!validation.isValid) {
      updateState({
        file: null,
        errors: validation.errors,
        warnings: []
      })
      return
    }
    
    updateState({
      file,
      errors: [],
      warnings: [],
      portfolio: null
    })
  }, [validateFile, updateState])
  
  const handleFileInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      handleFileSelect(files[0])
    }
  }, [handleFileSelect])
  
  const handleRemoveFile = useCallback(() => {
    updateState({
      file: null,
      errors: [],
      warnings: [],
      portfolio: null,
      uploadProgress: 0
    })
    
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }, [updateState])
  
  const handleUpload = useCallback(async () => {
    if (!state.file) return
    
    updateState({
      isUploading: true,
      isProcessing: false,
      uploadProgress: 0,
      errors: [],
      warnings: []
    })
    
    try {
      const formData = new FormData()
      formData.append('file', state.file)
      formData.append('filename', state.file.name)
      formData.append('fileSize', state.file.size.toString())
      formData.append('contentType', state.file.type)
      
      // Simulate upload progress
      let currentProgress = 0
      const progressInterval = setInterval(() => {
        currentProgress = Math.min(currentProgress + 10, 90)
        updateState({ uploadProgress: currentProgress })
      }, 200)
      
      const response = await fetch('/api/portfolio/upload', {
        method: 'POST',
        body: formData
      })
      
      clearInterval(progressInterval)
      
      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`)
      }
      
      const result: ApiResponse<FileParseResponse> = await response.json()
      
      updateState({
        isUploading: false,
        isProcessing: false,
        uploadProgress: 100
      })
      
      if (result.success && result.data) {
        const parseResponse = result.data
        
        if (parseResponse.success && parseResponse.portfolio) {
          updateState({
            portfolio: parseResponse.portfolio,
            errors: parseResponse.errors || [],
            warnings: parseResponse.warnings || []
          })
          
          // Auto-accept if no errors
          if (parseResponse.errors.length === 0) {
            onSuccess(parseResponse.portfolio)
          }
        } else {
          updateState({
            errors: parseResponse.errors || ['Failed to parse file'],
            warnings: parseResponse.warnings || []
          })
        }
      } else {
        updateState({
          errors: [result.error || 'Upload failed']
        })
      }
    } catch (error: any) {
      updateState({
        isUploading: false,
        isProcessing: false,
        errors: [error.message || 'Upload failed']
      })
    }
  }, [state.file, updateState, onSuccess])
  
  const handleAcceptParsedData = useCallback(() => {
    if (state.portfolio) {
      onSuccess(state.portfolio)
    }
  }, [state.portfolio, onSuccess])
  
  const fetchSampleFormat = useCallback(async () => {
    try {
      const response = await fetch('/api/portfolio/sample-format')
      const data = await response.json()
      setSampleData(data)
      setShowSampleFormat(true)
    } catch (error) {
      console.error('Failed to fetch sample format:', error)
    }
  }, [])
  
  // =============================================================================
  // RENDER HELPERS
  // =============================================================================
  
  const renderDropZone = () => (
    <div
      className={`
        relative border-2 border-dashed rounded-lg p-8 text-center transition-colors
        ${dragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}
        ${state.file ? 'bg-green-50 border-green-300' : ''}
      `}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      {state.file ? (
        <div className="space-y-4">
          <div className="flex items-center justify-center">
            <FileText className="h-12 w-12 text-green-600" />
          </div>
          <div>
            <p className="text-lg font-medium text-green-700">{state.file.name}</p>
            <p className="text-sm text-gray-600">
              {(state.file.size / 1024 / 1024).toFixed(2)} MB • {state.file.type || 'Unknown type'}
            </p>
          </div>
          <div className="flex items-center justify-center gap-4">
            <Button onClick={handleUpload} disabled={state.isUploading}>
              <Upload className="h-4 w-4 mr-2" />
              {state.isUploading ? 'Uploading...' : 'Upload & Parse'}
            </Button>
            <Button variant="outline" onClick={handleRemoveFile}>
              <X className="h-4 w-4 mr-2" />
              Remove
            </Button>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="flex items-center justify-center">
            <Upload className="h-12 w-12 text-gray-400" />
          </div>
          <div>
            <p className="text-lg font-medium text-gray-700">
              {dragActive ? 'Drop your file here' : 'Drag and drop your portfolio file'}
            </p>
            <p className="text-sm text-gray-500">
              or click to browse (CSV, Excel files supported)
            </p>
          </div>
          <Button variant="outline" onClick={() => fileInputRef.current?.click()}>
            Browse Files
          </Button>
        </div>
      )}
      
      <input
        ref={fileInputRef}
        type="file"
        accept={FILE_UPLOAD_CONFIG.acceptedFormats.join(',')}
        onChange={handleFileInputChange}
        className="hidden"
      />
    </div>
  )
  
  const renderUploadProgress = () => {
    if (!state.isUploading && !state.isProcessing) return null
    
    return (
      <Card>
        <CardContent className="pt-6">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">
                {state.isUploading ? 'Uploading file...' : 'Processing data...'}
              </span>
              <span className="text-sm text-gray-500">{state.uploadProgress}%</span>
            </div>
            <Progress value={state.uploadProgress} className="w-full" />
          </div>
        </CardContent>
      </Card>
    )
  }
  
  const renderParseResults = () => {
    if (!state.portfolio && state.errors.length === 0) return null
    
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            {state.portfolio ? (
              <CheckCircle className="h-5 w-5 text-green-600" />
            ) : (
              <AlertTriangle className="h-5 w-5 text-red-600" />
            )}
            Parse Results
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Errors */}
          {state.errors.length > 0 && (
            <Alert variant="destructive">
              <AlertTriangle className="h-4 w-4" />
              <AlertTitle>Parsing Errors</AlertTitle>
              <AlertDescription>
                <ul className="list-disc list-inside space-y-1">
                  {state.errors.map((error, index) => (
                    <li key={index}>{error}</li>
                  ))}
                </ul>
              </AlertDescription>
            </Alert>
          )}
          
          {/* Warnings */}
          {state.warnings.length > 0 && (
            <Alert>
              <Info className="h-4 w-4" />
              <AlertTitle>Parsing Warnings</AlertTitle>
              <AlertDescription>
                <ul className="list-disc list-inside space-y-1">
                  {state.warnings.map((warning, index) => (
                    <li key={index}>{warning}</li>
                  ))}
                </ul>
              </AlertDescription>
            </Alert>
          )}
          
          {/* Success and Portfolio Preview */}
          {state.portfolio && (
            <div className="space-y-4">
              <Alert>
                <CheckCircle className="h-4 w-4" />
                <AlertTitle>File parsed successfully!</AlertTitle>
                <AlertDescription>
                  Found {state.portfolio.holdings.length} holdings in your portfolio.
                  Total value: ₹{state.portfolio.holdings.reduce((sum, h) => sum + (h.quantity * h.avg_buy_price), 0).toLocaleString('en-IN')}
                </AlertDescription>
              </Alert>
              
              {/* Portfolio Preview */}
              <div className="border rounded-lg p-4 bg-gray-50">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="font-medium">Portfolio Preview</h4>
                  <Badge variant="secondary">{state.portfolio.holdings.length} holdings</Badge>
                </div>
                
                <div className="space-y-2 max-h-60 overflow-y-auto">
                  {state.portfolio.holdings.slice(0, 10).map((holding, index) => (
                    <div key={index} className="flex items-center justify-between py-2 px-3 bg-white rounded text-sm">
                      <span className="font-medium">{holding.ticker}</span>
                      <span>{holding.quantity.toLocaleString()} shares</span>
                      <span>₹{holding.avg_buy_price.toFixed(2)}</span>
                      <span className="font-medium">
                        ₹{(holding.quantity * holding.avg_buy_price).toLocaleString('en-IN')}
                      </span>
                    </div>
                  ))}
                  
                  {state.portfolio.holdings.length > 10 && (
                    <div className="text-center py-2 text-gray-500 text-sm">
                      ... and {state.portfolio.holdings.length - 10} more holdings
                    </div>
                  )}
                </div>
              </div>
              
              <div className="flex items-center justify-end gap-4">
                <Button variant="outline" onClick={handleRemoveFile}>
                  Upload Different File
                </Button>
                <Button onClick={handleAcceptParsedData}>
                  Use This Portfolio
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    )
  }
  
  const renderSampleFormat = () => {
    if (!showSampleFormat) return null
    
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span className="flex items-center gap-2">
              <Eye className="h-5 w-5" />
              Sample File Format
            </span>
            <Button variant="ghost" size="sm" onClick={() => setShowSampleFormat(false)}>
              <X className="h-4 w-4" />
            </Button>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {sampleData && (
            <>
              {/* CSV Example */}
              <div>
                <h4 className="font-medium mb-2">CSV Format Example:</h4>
                <div className="bg-gray-100 p-3 rounded-lg font-mono text-sm overflow-x-auto">
                  <div>{sampleData.csvExample?.headers.join(',')}</div>
                  {sampleData.csvExample?.sampleRows.map((row: string[], index: number) => (
                    <div key={index}>{row.join(',')}</div>
                  ))}
                </div>
              </div>
              
              {/* Column Variations */}
              <div>
                <h4 className="font-medium mb-2">Accepted Column Names:</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {Object.entries(sampleData.columnVariations || {}).map(([key, value]) => (
                    <div key={key} className="text-sm">
                      <span className="font-medium capitalize">{key.replace('_', ' ')}:</span>
                      <br />
                      <span className="text-gray-600">{value as string}</span>
                    </div>
                  ))}
                </div>
              </div>
              
              {/* Notes */}
              <div>
                <h4 className="font-medium mb-2">Important Notes:</h4>
                <ul className="list-disc list-inside space-y-1 text-sm text-gray-600">
                  {(sampleData.notes || []).map((note: string, index: number) => (
                    <li key={index}>{note}</li>
                  ))}
                </ul>
              </div>
            </>
          )}
          
          <div className="flex items-center gap-4">
            <Button variant="outline" onClick={() => {
              // Create and download sample CSV
              const csvContent = "ticker,quantity,avg_buy_price,current_price,buy_date\\nRELIANCE,50,2450.00,2520.00,2023-01-15\\nTCS,25,3200.00,3450.00,2023-02-10\\nHDFCBANK,75,1650.00,1720.00,2023-03-05"
              const blob = new Blob([csvContent], { type: 'text/csv' })
              const url = URL.createObjectURL(blob)
              const a = document.createElement('a')
              a.href = url
              a.download = 'portfolio-sample.csv'
              a.click()
              URL.revokeObjectURL(url)
            }}>
              <Download className="h-4 w-4 mr-2" />
              Download Sample CSV
            </Button>
          </div>
        </CardContent>
      </Card>
    )
  }
  
  // =============================================================================
  // MAIN RENDER
  // =============================================================================
  
  return (
    <div className="space-y-6">
      {/* Help Section */}
      <div className="flex items-center justify-between">
        <p className="text-gray-600">
          Upload your portfolio file (CSV, Excel) for quick data entry
        </p>
        <Button variant="outline" size="sm" onClick={fetchSampleFormat}>
          <Info className="h-4 w-4 mr-2" />
          View Sample Format
        </Button>
      </div>
      
      {/* Drop Zone */}
      {renderDropZone()}
      
      {/* Upload Progress */}
      {renderUploadProgress()}
      
      {/* Parse Results */}
      {renderParseResults()}
      
      {/* Sample Format */}
      {renderSampleFormat()}
    </div>
  )
}

export default FileUploadSection