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
  }, [])\n  \n  const handleDrop = useCallback((e: React.DragEvent) => {\n    e.preventDefault()\n    e.stopPropagation()\n    setDragActive(false)\n    \n    const files = Array.from(e.dataTransfer.files)\n    if (files.length > 0) {\n      handleFileSelect(files[0])\n    }\n  }, [])\n  \n  const handleFileSelect = useCallback((file: File) => {\n    const validation = validateFile(file)\n    \n    if (!validation.isValid) {\n      updateState({\n        file: null,\n        errors: validation.errors,\n        warnings: []\n      })\n      return\n    }\n    \n    updateState({\n      file,\n      errors: [],\n      warnings: [],\n      portfolio: null\n    })\n  }, [validateFile, updateState])\n  \n  const handleFileInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {\n    const files = e.target.files\n    if (files && files.length > 0) {\n      handleFileSelect(files[0])\n    }\n  }, [handleFileSelect])\n  \n  const handleRemoveFile = useCallback(() => {\n    updateState({\n      file: null,\n      errors: [],\n      warnings: [],\n      portfolio: null,\n      uploadProgress: 0\n    })\n    \n    if (fileInputRef.current) {\n      fileInputRef.current.value = ''\n    }\n  }, [updateState])\n  \n  const handleUpload = useCallback(async () => {\n    if (!state.file) return\n    \n    updateState({\n      isUploading: true,\n      isProcessing: false,\n      uploadProgress: 0,\n      errors: [],\n      warnings: []\n    })\n    \n    try {\n      const formData = new FormData()\n      formData.append('file', state.file)\n      formData.append('filename', state.file.name)\n      formData.append('fileSize', state.file.size.toString())\n      formData.append('contentType', state.file.type)\n      \n      // Simulate upload progress\n      const progressInterval = setInterval(() => {\n        updateState(prevState => {\n          const newProgress = Math.min(prevState.uploadProgress + 10, 90)\n          return { ...prevState, uploadProgress: newProgress }\n        })\n      }, 200)\n      \n      const response = await fetch('/api/portfolio/upload', {\n        method: 'POST',\n        body: formData\n      })\n      \n      clearInterval(progressInterval)\n      \n      if (!response.ok) {\n        throw new Error(`Upload failed: ${response.statusText}`)\n      }\n      \n      const result: ApiResponse<FileParseResponse> = await response.json()\n      \n      updateState({\n        isUploading: false,\n        isProcessing: false,\n        uploadProgress: 100\n      })\n      \n      if (result.success && result.data) {\n        const parseResponse = result.data\n        \n        if (parseResponse.success && parseResponse.portfolio) {\n          updateState({\n            portfolio: parseResponse.portfolio,\n            errors: parseResponse.errors || [],\n            warnings: parseResponse.warnings || []\n          })\n          \n          // Auto-accept if no errors\n          if (parseResponse.errors.length === 0) {\n            onSuccess(parseResponse.portfolio)\n          }\n        } else {\n          updateState({\n            errors: parseResponse.errors || ['Failed to parse file'],\n            warnings: parseResponse.warnings || []\n          })\n        }\n      } else {\n        updateState({\n          errors: [result.error || 'Upload failed']\n        })\n      }\n    } catch (error: any) {\n      updateState({\n        isUploading: false,\n        isProcessing: false,\n        errors: [error.message || 'Upload failed']\n      })\n    }\n  }, [state.file, updateState, onSuccess])\n  \n  const handleAcceptParsedData = useCallback(() => {\n    if (state.portfolio) {\n      onSuccess(state.portfolio)\n    }\n  }, [state.portfolio, onSuccess])\n  \n  const fetchSampleFormat = useCallback(async () => {\n    try {\n      const response = await fetch('/api/portfolio/sample-format')\n      const data = await response.json()\n      setSampleData(data)\n      setShowSampleFormat(true)\n    } catch (error) {\n      console.error('Failed to fetch sample format:', error)\n    }\n  }, [])\n  \n  // =============================================================================\n  // RENDER HELPERS\n  // =============================================================================\n  \n  const renderDropZone = () => (\n    <div\n      className={`\n        relative border-2 border-dashed rounded-lg p-8 text-center transition-colors\n        ${dragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}\n        ${state.file ? 'bg-green-50 border-green-300' : ''}\n      `}\n      onDragEnter={handleDragEnter}\n      onDragLeave={handleDragLeave}\n      onDragOver={handleDragOver}\n      onDrop={handleDrop}\n    >\n      {state.file ? (\n        <div className=\"space-y-4\">\n          <div className=\"flex items-center justify-center\">\n            <FileText className=\"h-12 w-12 text-green-600\" />\n          </div>\n          <div>\n            <p className=\"text-lg font-medium text-green-700\">{state.file.name}</p>\n            <p className=\"text-sm text-gray-600\">\n              {(state.file.size / 1024 / 1024).toFixed(2)} MB • {state.file.type || 'Unknown type'}\n            </p>\n          </div>\n          <div className=\"flex items-center justify-center gap-4\">\n            <Button onClick={handleUpload} disabled={state.isUploading}>\n              <Upload className=\"h-4 w-4 mr-2\" />\n              {state.isUploading ? 'Uploading...' : 'Upload & Parse'}\n            </Button>\n            <Button variant=\"outline\" onClick={handleRemoveFile}>\n              <X className=\"h-4 w-4 mr-2\" />\n              Remove\n            </Button>\n          </div>\n        </div>\n      ) : (\n        <div className=\"space-y-4\">\n          <div className=\"flex items-center justify-center\">\n            <Upload className=\"h-12 w-12 text-gray-400\" />\n          </div>\n          <div>\n            <p className=\"text-lg font-medium text-gray-700\">\n              {dragActive ? 'Drop your file here' : 'Drag and drop your portfolio file'}\n            </p>\n            <p className=\"text-sm text-gray-500\">\n              or click to browse (CSV, Excel files supported)\n            </p>\n          </div>\n          <Button variant=\"outline\" onClick={() => fileInputRef.current?.click()}>\n            Browse Files\n          </Button>\n        </div>\n      )}\n      \n      <input\n        ref={fileInputRef}\n        type=\"file\"\n        accept={FILE_UPLOAD_CONFIG.acceptedFormats.join(',')}\n        onChange={handleFileInputChange}\n        className=\"hidden\"\n      />\n    </div>\n  )\n  \n  const renderUploadProgress = () => {\n    if (!state.isUploading && !state.isProcessing) return null\n    \n    return (\n      <Card>\n        <CardContent className=\"pt-6\">\n          <div className=\"space-y-4\">\n            <div className=\"flex items-center justify-between\">\n              <span className=\"text-sm font-medium\">\n                {state.isUploading ? 'Uploading file...' : 'Processing data...'}\n              </span>\n              <span className=\"text-sm text-gray-500\">{state.uploadProgress}%</span>\n            </div>\n            <Progress value={state.uploadProgress} className=\"w-full\" />\n          </div>\n        </CardContent>\n      </Card>\n    )\n  }\n  \n  const renderParseResults = () => {\n    if (!state.portfolio && state.errors.length === 0) return null\n    \n    return (\n      <Card>\n        <CardHeader>\n          <CardTitle className=\"flex items-center gap-2\">\n            {state.portfolio ? (\n              <CheckCircle className=\"h-5 w-5 text-green-600\" />\n            ) : (\n              <AlertTriangle className=\"h-5 w-5 text-red-600\" />\n            )}\n            Parse Results\n          </CardTitle>\n        </CardHeader>\n        <CardContent className=\"space-y-4\">\n          {/* Errors */}\n          {state.errors.length > 0 && (\n            <Alert variant=\"destructive\">\n              <AlertTriangle className=\"h-4 w-4\" />\n              <AlertTitle>Parsing Errors</AlertTitle>\n              <AlertDescription>\n                <ul className=\"list-disc list-inside space-y-1\">\n                  {state.errors.map((error, index) => (\n                    <li key={index}>{error}</li>\n                  ))}\n                </ul>\n              </AlertDescription>\n            </Alert>\n          )}\n          \n          {/* Warnings */}\n          {state.warnings.length > 0 && (\n            <Alert>\n              <Info className=\"h-4 w-4\" />\n              <AlertTitle>Parsing Warnings</AlertTitle>\n              <AlertDescription>\n                <ul className=\"list-disc list-inside space-y-1\">\n                  {state.warnings.map((warning, index) => (\n                    <li key={index}>{warning}</li>\n                  ))}\n                </ul>\n              </AlertDescription>\n            </Alert>\n          )}\n          \n          {/* Success and Portfolio Preview */}\n          {state.portfolio && (\n            <div className=\"space-y-4\">\n              <Alert>\n                <CheckCircle className=\"h-4 w-4\" />\n                <AlertTitle>File parsed successfully!</AlertTitle>\n                <AlertDescription>\n                  Found {state.portfolio.holdings.length} holdings in your portfolio.\n                  Total value: ₹{state.portfolio.holdings.reduce((sum, h) => sum + (h.quantity * h.avgBuyPrice), 0).toLocaleString('en-IN')}\n                </AlertDescription>\n              </Alert>\n              \n              {/* Portfolio Preview */}\n              <div className=\"border rounded-lg p-4 bg-gray-50\">\n                <div className=\"flex items-center justify-between mb-3\">\n                  <h4 className=\"font-medium\">Portfolio Preview</h4>\n                  <Badge variant=\"secondary\">{state.portfolio.holdings.length} holdings</Badge>\n                </div>\n                \n                <div className=\"space-y-2 max-h-60 overflow-y-auto\">\n                  {state.portfolio.holdings.slice(0, 10).map((holding, index) => (\n                    <div key={index} className=\"flex items-center justify-between py-2 px-3 bg-white rounded text-sm\">\n                      <span className=\"font-medium\">{holding.ticker}</span>\n                      <span>{holding.quantity.toLocaleString()} shares</span>\n                      <span>₹{holding.avgBuyPrice.toFixed(2)}</span>\n                      <span className=\"font-medium\">\n                        ₹{(holding.quantity * holding.avgBuyPrice).toLocaleString('en-IN')}\n                      </span>\n                    </div>\n                  ))}\n                  \n                  {state.portfolio.holdings.length > 10 && (\n                    <div className=\"text-center py-2 text-gray-500 text-sm\">\n                      ... and {state.portfolio.holdings.length - 10} more holdings\n                    </div>\n                  )}\n                </div>\n              </div>\n              \n              <div className=\"flex items-center justify-end gap-4\">\n                <Button variant=\"outline\" onClick={handleRemoveFile}>\n                  Upload Different File\n                </Button>\n                <Button onClick={handleAcceptParsedData}>\n                  Use This Portfolio\n                </Button>\n              </div>\n            </div>\n          )}\n        </CardContent>\n      </Card>\n    )\n  }\n  \n  const renderSampleFormat = () => {\n    if (!showSampleFormat) return null\n    \n    return (\n      <Card>\n        <CardHeader>\n          <CardTitle className=\"flex items-center justify-between\">\n            <span className=\"flex items-center gap-2\">\n              <Eye className=\"h-5 w-5\" />\n              Sample File Format\n            </span>\n            <Button variant=\"ghost\" size=\"sm\" onClick={() => setShowSampleFormat(false)}>\n              <X className=\"h-4 w-4\" />\n            </Button>\n          </CardTitle>\n        </CardHeader>\n        <CardContent className=\"space-y-4\">\n          {sampleData && (\n            <>\n              {/* CSV Example */}\n              <div>\n                <h4 className=\"font-medium mb-2\">CSV Format Example:</h4>\n                <div className=\"bg-gray-100 p-3 rounded-lg font-mono text-sm overflow-x-auto\">\n                  <div>{sampleData.csvExample?.headers.join(',')}</div>\n                  {sampleData.csvExample?.sampleRows.map((row: string[], index: number) => (\n                    <div key={index}>{row.join(',')}</div>\n                  ))}\n                </div>\n              </div>\n              \n              {/* Column Variations */}\n              <div>\n                <h4 className=\"font-medium mb-2\">Accepted Column Names:</h4>\n                <div className=\"grid grid-cols-1 md:grid-cols-2 gap-3\">\n                  {Object.entries(sampleData.columnVariations || {}).map(([key, value]) => (\n                    <div key={key} className=\"text-sm\">\n                      <span className=\"font-medium capitalize\">{key.replace('_', ' ')}:</span>\n                      <br />\n                      <span className=\"text-gray-600\">{value as string}</span>\n                    </div>\n                  ))}\n                </div>\n              </div>\n              \n              {/* Notes */}\n              <div>\n                <h4 className=\"font-medium mb-2\">Important Notes:</h4>\n                <ul className=\"list-disc list-inside space-y-1 text-sm text-gray-600\">\n                  {(sampleData.notes || []).map((note: string, index: number) => (\n                    <li key={index}>{note}</li>\n                  ))}\n                </ul>\n              </div>\n            </>\n          )}\n          \n          <div className=\"flex items-center gap-4\">\n            <Button variant=\"outline\" onClick={() => {\n              // Create and download sample CSV\n              const csvContent = \"ticker,quantity,avg_buy_price,current_price,buy_date\\nRELIANCE,50,2450.00,2520.00,2023-01-15\\nTCS,25,3200.00,3450.00,2023-02-10\\nHDFCBANK,75,1650.00,1720.00,2023-03-05\"\n              const blob = new Blob([csvContent], { type: 'text/csv' })\n              const url = URL.createObjectURL(blob)\n              const a = document.createElement('a')\n              a.href = url\n              a.download = 'portfolio-sample.csv'\n              a.click()\n              URL.revokeObjectURL(url)\n            }}>\n              <Download className=\"h-4 w-4 mr-2\" />\n              Download Sample CSV\n            </Button>\n          </div>\n        </CardContent>\n      </Card>\n    )\n  }\n  \n  // =============================================================================\n  // MAIN RENDER\n  // =============================================================================\n  \n  return (\n    <div className=\"space-y-6\">\n      {/* Help Section */}\n      <div className=\"flex items-center justify-between\">\n        <p className=\"text-gray-600\">\n          Upload your portfolio file (CSV, Excel) for quick data entry\n        </p>\n        <Button variant=\"outline\" size=\"sm\" onClick={fetchSampleFormat}>\n          <Info className=\"h-4 w-4 mr-2\" />\n          View Sample Format\n        </Button>\n      </div>\n      \n      {/* Drop Zone */}\n      {renderDropZone()}\n      \n      {/* Upload Progress */}\n      {renderUploadProgress()}\n      \n      {/* Parse Results */}\n      {renderParseResults()}\n      \n      {/* Sample Format */}\n      {renderSampleFormat()}\n    </div>\n  )\n}\n\nexport default FileUploadSection"