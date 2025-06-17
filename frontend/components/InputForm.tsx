/**
 * Portfolio Input Form Component
 * 
 * This component handles comprehensive user input for portfolio data:
 * - Manual stock entry with validation
 * - File upload for CSV/Excel portfolio files
 * - User profile settings
 * - Real-time validation and error display
 * - Portfolio preview and submission
 * 
 * Key features:
 * - Dynamic stock entry with add/remove functionality
 * - File drag-and-drop upload interface
 * - Real-time validation of stock symbols and data
 * - Progress indicators during processing
 * - Comprehensive error handling and user feedback
 * - Responsive design for mobile/desktop
 * - Integration with backend validation service
 * 
 * Form Structure:
 * 1. User Profile Section (risk tolerance, investment goals)
 * 2. Manual Entry Section (add stocks one by one)
 * 3. File Upload Section (bulk upload via CSV/Excel)
 * 4. Portfolio Preview Section (review before submission)
 * 5. Submit Section (final submission to analysis)
 */

'use client'

import React, { useState, useCallback, useEffect } from 'react'
import { 
  UserProfile, 
  PortfolioHolding, 
  PortfolioInput,
  InputFormState,
  FileUploadState,
  PortfolioValidationResponse,
  ValidationError,
  ApiResponse,
  DEFAULT_USER_PROFILE,
  DEFAULT_HOLDING,
  RISK_TOLERANCE_OPTIONS,
  INVESTMENT_STYLE_OPTIONS,
  TIME_HORIZON_OPTIONS
} from '../lib/types'
import PortfolioApiClient from '../lib/api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Plus, Trash2, Upload, FileText, AlertTriangle, CheckCircle, Info, X } from 'lucide-react'

// Sub-components
import FileUploadSection from './FileUploadSection'
import PortfolioPreview from './PortfolioPreview'
import UserProfileForm from './UserProfileForm'

interface InputFormProps {
  onSubmit: (portfolio: PortfolioInput, userProfile: UserProfile, analysisId?: string) => Promise<void>
  onSaveDraft?: (portfolio: PortfolioInput, userProfile: UserProfile) => void
  initialData?: {
    portfolio?: PortfolioInput
    userProfile?: UserProfile
  }
}

const InputForm: React.FC<InputFormProps> = ({ 
  onSubmit, 
  onSaveDraft, 
  initialData 
}) => {
  // =============================================================================
  // STATE MANAGEMENT
  // =============================================================================
  
  // User Profile State
  const [userProfile, setUserProfile] = useState<UserProfile>({
    ...DEFAULT_USER_PROFILE,
    ...initialData?.userProfile
  } as UserProfile)
  
  // Portfolio Holdings State
  const [holdings, setHoldings] = useState<PortfolioHolding[]>(
    initialData?.portfolio?.holdings || []
  )
  
  // Current holding being entered
  const [currentHolding, setCurrentHolding] = useState<Partial<PortfolioHolding>>(
    { ...DEFAULT_HOLDING }
  )
  
  // Form State
  const [formState, setFormState] = useState<InputFormState>({
    currentHolding: { ...DEFAULT_HOLDING },
    holdings: [],
    validationErrors: [],
    validationWarnings: [],
    isValidating: false,
    isDirty: false,
    isSubmitting: false,
    submitErrors: []
  })
  
  // File Upload State
  const [fileUploadState, setFileUploadState] = useState<FileUploadState>({
    file: null,
    isUploading: false,
    isProcessing: false,
    uploadProgress: 0,
    portfolio: null,
    errors: [],
    warnings: []
  })
  
  // UI State - Enhanced to default to 'upload' tab
  const [activeTab, setActiveTab] = useState<'manual' | 'upload'>('upload')
  const [editingIndex, setEditingIndex] = useState<number | null>(null)
  const [showPreview, setShowPreview] = useState(false)
  const [validationResponse, setValidationResponse] = useState<PortfolioValidationResponse | null>(null)
  
  // =============================================================================
  // EFFECTS
  // =============================================================================
  
  // Mark form as dirty when data changes
  useEffect(() => {
    const isDirty = holdings.length > 0 || 
                   Object.values(currentHolding).some(val => val !== '' && val !== 0) ||
                   JSON.stringify(userProfile) !== JSON.stringify(DEFAULT_USER_PROFILE)
    
    setFormState(prev => ({ ...prev, isDirty }))
  }, [holdings, currentHolding, userProfile])
  
  // Auto-save draft
  useEffect(() => {
    if (formState.isDirty && onSaveDraft) {
      const timer = setTimeout(() => {
        if (holdings.length > 0) {
          onSaveDraft({ holdings }, userProfile)
        }
      }, 2000) // Auto-save after 2 seconds of inactivity
      
      return () => clearTimeout(timer)
    }
  }, [holdings, userProfile, formState.isDirty, onSaveDraft])
  
  // =============================================================================
  // VALIDATION FUNCTIONS
  // =============================================================================
  
  const validateHolding = useCallback((holding: Partial<PortfolioHolding>): ValidationError[] => {
    const errors: ValidationError[] = []
    
    if (!holding.ticker || holding.ticker.trim() === '') {
      errors.push({ field: 'ticker', message: 'Ticker symbol is required' })
    } else if (!/^[A-Z0-9\.\-&]+$/i.test(holding.ticker)) {
      errors.push({ field: 'ticker', message: 'Invalid ticker format' })
    }
    
    if (!holding.quantity || holding.quantity <= 0) {
      errors.push({ field: 'quantity', message: 'Quantity must be a positive number' })
    }
    
    if (!holding.avgBuyPrice || holding.avgBuyPrice <= 0) {
      errors.push({ field: 'avgBuyPrice', message: 'Average buy price must be positive' })
    }
    
    // Check for duplicate ticker
    const existingIndex = holdings.findIndex(h => h.ticker.toUpperCase() === holding.ticker?.toUpperCase())
    if (existingIndex !== -1 && editingIndex !== existingIndex) {
      errors.push({ field: 'ticker', message: 'This ticker already exists in your portfolio' })
    }
    
    return errors
  }, [holdings, editingIndex])
  
  const validatePortfolio = useCallback(async () => {
    if (holdings.length === 0) return
    
    setFormState(prev => ({ ...prev, isValidating: true }))
    
    try {
      // Call backend validation API using our API client
      const portfolio: PortfolioInput = { holdings }
      const validationResult = await PortfolioApiClient.validatePortfolio(portfolio, userProfile)
      
      setValidationResponse(validationResult)
      setFormState(prev => ({
        ...prev,
        validationErrors: validationResult.errors || [],
        validationWarnings: validationResult.warnings || []
      }))
    } catch (error) {
      console.error('Validation error:', error)
    } finally {
      setFormState(prev => ({ ...prev, isValidating: false }))
    }
  }, [holdings, userProfile])
  
  // =============================================================================
  // HANDLERS
  // =============================================================================
  
  const handleAddHolding = useCallback(() => {
    const errors = validateHolding(currentHolding)
    
    if (errors.length > 0) {
      setFormState(prev => ({ ...prev, validationErrors: errors }))
      return
    }
    
    const newHolding: PortfolioHolding = {
      ticker: currentHolding.ticker!.toUpperCase().trim(),
      quantity: currentHolding.quantity!,
      avgBuyPrice: currentHolding.avgBuyPrice!,
      buyDate: currentHolding.buyDate,
      currentPrice: currentHolding.currentPrice
    }
    
    if (editingIndex !== null) {
      // Update existing holding
      const updatedHoldings = [...holdings]
      updatedHoldings[editingIndex] = newHolding
      setHoldings(updatedHoldings)
      setEditingIndex(null)
    } else {
      // Add new holding
      setHoldings(prev => [...prev, newHolding])
    }
    
    // Clear current holding
    setCurrentHolding({ ...DEFAULT_HOLDING })
    setFormState(prev => ({ ...prev, validationErrors: [] }))
    
    // Trigger validation
    setTimeout(() => validatePortfolio(), 100)
  }, [currentHolding, holdings, editingIndex, validateHolding, validatePortfolio])
  
  const handleEditHolding = useCallback((index: number) => {
    const holding = holdings[index]
    setCurrentHolding(holding)
    setEditingIndex(index)
    setActiveTab('manual')
  }, [holdings])
  
  const handleRemoveHolding = useCallback((index: number) => {
    setHoldings(prev => prev.filter((_, i) => i !== index))
    if (editingIndex === index) {
      setCurrentHolding({ ...DEFAULT_HOLDING })
      setEditingIndex(null)
    }
    setTimeout(() => validatePortfolio(), 100)
  }, [editingIndex, validatePortfolio])
  
  const handleClearForm = useCallback(() => {
    setHoldings([])
    setCurrentHolding({ ...DEFAULT_HOLDING })
    setEditingIndex(null)
    setValidationResponse(null)
    setFormState({
      currentHolding: { ...DEFAULT_HOLDING },
      holdings: [],
      validationErrors: [],
      validationWarnings: [],
      isValidating: false,
      isDirty: false,
      isSubmitting: false,
      submitErrors: []
    })
  }, [])
  
  const handleFileUploadSuccess = useCallback((portfolio: PortfolioInput) => {
    setHoldings(portfolio.holdings)
    setActiveTab('manual') // Switch to manual tab to show imported data
    setTimeout(() => validatePortfolio(), 100)
  }, [validatePortfolio])
  
  const handleSubmit = useCallback(async () => {
    if (holdings.length === 0) {
      setFormState(prev => ({
        ...prev,
        submitErrors: ['Portfolio cannot be empty. Please add at least one holding.']
      }))
      return
    }
    
    setFormState(prev => ({ ...prev, isSubmitting: true, submitErrors: [] }))
    
    try {
      const portfolio: PortfolioInput = { holdings }
      // First validate the portfolio
      const validationResult = await PortfolioApiClient.validatePortfolio(portfolio, userProfile)
      
      if (!validationResult.isValid) {
        setFormState(prev => ({
          ...prev,
          submitErrors: ['Portfolio validation failed. Please fix the errors and try again.'],
          validationErrors: validationResult.errors
        }))
        setValidationResponse(validationResult)
        return
      }
      
      // If validation passes, start analysis
      const analysisResult = await PortfolioApiClient.analyzePortfolio(portfolio, userProfile)
      
      // Call the parent component's onSubmit with analysis ID
      await onSubmit(portfolio, userProfile, analysisResult.analysisId)
    } catch (error: any) {
      setFormState(prev => ({
        ...prev,
        submitErrors: [error.message || 'Failed to submit portfolio']
      }))
    } finally {
      setFormState(prev => ({ ...prev, isSubmitting: false }))
    }
  }, [holdings, userProfile, onSubmit])
  
  // =============================================================================
  // COMPUTED VALUES
  // =============================================================================
  
  const totalValue = holdings.reduce((sum, holding) => 
    sum + (holding.quantity * holding.avgBuyPrice), 0
  )
  
  const currentHoldingErrors = validateHolding(currentHolding)
  const canAddHolding = Object.values(currentHolding).some(val => val !== '' && val !== 0) && 
                       currentHoldingErrors.length === 0
  
  const canSubmit = holdings.length > 0 && 
                   validationResponse?.isValid !== false && 
                   !formState.isSubmitting
  
  // =============================================================================
  // RENDER
  // =============================================================================
  
  return (
    <div className="max-w-4xl mx-auto p-6 space-y-8">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Portfolio Analysis Setup
        </h1>
        <p className="text-gray-600">
          Enter your portfolio data and preferences to get personalized investment analysis
        </p>
      </div>
      
      {/* User Profile Section */}
      <UserProfileForm 
        userProfile={userProfile}
        onChange={setUserProfile}
      />
      
      {/* Portfolio Input Tabs */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileText className="h-5 w-5" />
            Portfolio Holdings
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as 'manual' | 'upload')}>
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="manual">Manual Entry</TabsTrigger>
              <TabsTrigger value="upload">File Upload</TabsTrigger>
            </TabsList>
            
            {/* Manual Entry Tab */}
            <TabsContent value="manual" className="space-y-6">
              {/* Current Holding Form */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 p-4 border rounded-lg bg-gray-50">
                <div>
                  <Label htmlFor="ticker">Stock Symbol *</Label>
                  <Input
                    id="ticker"
                    value={currentHolding.ticker || ''}
                    onChange={(e) => setCurrentHolding(prev => ({ 
                      ...prev, 
                      ticker: e.target.value.toUpperCase() 
                    }))}
                    placeholder="e.g., RELIANCE"
                    className={currentHoldingErrors.find(e => e.field === 'ticker') ? 'border-red-500' : ''}
                  />
                </div>
                
                <div>
                  <Label htmlFor="quantity">Quantity *</Label>
                  <Input
                    id="quantity"
                    type="number"
                    value={currentHolding.quantity || ''}
                    onChange={(e) => setCurrentHolding(prev => ({ 
                      ...prev, 
                      quantity: parseInt(e.target.value) || 0 
                    }))}
                    placeholder="Number of shares"
                    className={currentHoldingErrors.find(e => e.field === 'quantity') ? 'border-red-500' : ''}
                  />
                </div>
                
                <div>
                  <Label htmlFor="avgBuyPrice">Avg Buy Price (₹) *</Label>
                  <Input
                    id="avgBuyPrice"
                    type="number"
                    step="0.01"
                    value={currentHolding.avgBuyPrice || ''}
                    onChange={(e) => setCurrentHolding(prev => ({ 
                      ...prev, 
                      avgBuyPrice: parseFloat(e.target.value) || 0 
                    }))}
                    placeholder="Price per share"
                    className={currentHoldingErrors.find(e => e.field === 'avgBuyPrice') ? 'border-red-500' : ''}
                  />
                </div>
                
                <div className="flex items-end">
                  <Button 
                    onClick={handleAddHolding}
                    disabled={!canAddHolding}
                    className="w-full"
                  >
                    <Plus className="h-4 w-4 mr-2" />
                    {editingIndex !== null ? 'Update' : 'Add'}
                  </Button>
                </div>
              </div>
              
              {/* Current Holding Errors */}
              {currentHoldingErrors.length > 0 && (
                <Alert variant="destructive">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertTitle>Please fix the following errors:</AlertTitle>
                  <AlertDescription>
                    <ul className="list-disc list-inside">
                      {currentHoldingErrors.map((error, index) => (
                        <li key={index}>{error.message}</li>
                      ))}
                    </ul>
                  </AlertDescription>
                </Alert>
              )}
              
              {/* Holdings List */}
              {holdings.length > 0 && (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold">Your Holdings ({holdings.length})</h3>
                    <div className="text-sm text-gray-600">
                      Total Value: ₹{totalValue.toLocaleString('en-IN', { maximumFractionDigits: 2 })}
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    {holdings.map((holding, index) => (
                      <div key={index} className="flex items-center justify-between p-3 border rounded-lg bg-white">
                        <div className="flex-1 grid grid-cols-4 gap-4">
                          <div>
                            <span className="font-medium">{holding.ticker}</span>
                          </div>
                          <div>
                            <span className="text-gray-600">{holding.quantity.toLocaleString()} shares</span>
                          </div>
                          <div>
                            <span className="text-gray-600">₹{holding.avgBuyPrice.toFixed(2)}</span>
                          </div>
                          <div>
                            <span className="font-medium">
                              ₹{(holding.quantity * holding.avgBuyPrice).toLocaleString('en-IN', { maximumFractionDigits: 2 })}
                            </span>
                          </div>
                        </div>
                        
                        <div className="flex items-center gap-2">
                          <Button 
                            variant="outline" 
                            size="sm"
                            onClick={() => handleEditHolding(index)}
                          >
                            Edit
                          </Button>
                          <Button 
                            variant="outline" 
                            size="sm"
                            onClick={() => handleRemoveHolding(index)}
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </TabsContent>
            
            {/* File Upload Tab */}
            <TabsContent value="upload">
              <FileUploadSection 
                onSuccess={handleFileUploadSuccess}
                state={fileUploadState}
                onStateChange={setFileUploadState}
              />
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
      
      {/* Validation Results */}
      {validationResponse && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              {validationResponse.isValid ? (
                <CheckCircle className="h-5 w-5 text-green-600" />
              ) : (
                <AlertTriangle className="h-5 w-5 text-red-600" />
              )}
              Portfolio Validation
            </CardTitle>
          </CardHeader>
          <CardContent>
            {validationResponse.errors && validationResponse.errors.length > 0 && (
              <Alert variant="destructive" className="mb-4">
                <AlertTriangle className="h-4 w-4" />
                <AlertTitle>Validation Errors</AlertTitle>
                <AlertDescription>
                  <ul className="list-disc list-inside">
                    {validationResponse.errors.map((error, index) => (
                      <li key={index}>{error.message}</li>
                    ))}
                  </ul>
                </AlertDescription>
              </Alert>
            )}
            
            {validationResponse.warnings && validationResponse.warnings.length > 0 && (
              <Alert className="mb-4">
                <Info className="h-4 w-4" />
                <AlertTitle>Recommendations</AlertTitle>
                <AlertDescription>
                  <ul className="list-disc list-inside">
                    {validationResponse.warnings.map((warning, index) => (
                      <li key={index}>{warning}</li>
                    ))}
                  </ul>
                </AlertDescription>
              </Alert>
            )}
            
            {validationResponse.isValid && (
              <Alert>
                <CheckCircle className="h-4 w-4" />
                <AlertTitle>Portfolio looks good!</AlertTitle>
                <AlertDescription>
                  Your portfolio has been validated successfully. 
                  Total Value: ₹{validationResponse.totalValue?.toLocaleString('en-IN') || 0} • 
                  Holdings: {validationResponse.holdingsCount || 0}
                </AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>
      )}
      
      {/* Submit Errors */}
      {formState.submitErrors.length > 0 && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Submission Error</AlertTitle>
          <AlertDescription>
            <ul className="list-disc list-inside">
              {formState.submitErrors.map((error, index) => (
                <li key={index}>{error}</li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      )}
      
      {/* Action Buttons */}
      <div className="flex items-center justify-between pt-6 border-t">
        <div className="flex items-center gap-4">
          <Button 
            variant="outline" 
            onClick={handleClearForm}
            disabled={formState.isSubmitting}
          >
            Clear All
          </Button>
          
          {formState.isDirty && (
            <Badge variant="secondary">Unsaved changes</Badge>
          )}
        </div>
        
        <div className="flex items-center gap-4">
          <Button 
            variant="outline"
            onClick={() => setShowPreview(true)}
            disabled={holdings.length === 0}
          >
            Preview
          </Button>
          
          <Button 
            onClick={handleSubmit}
            disabled={!canSubmit}
            className="min-w-[120px]"
          >
            {formState.isSubmitting ? (
              <div className="flex items-center gap-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                Processing...
              </div>
            ) : (
              'Analyze Portfolio'
            )}
          </Button>
        </div>
      </div>
      
      {/* Portfolio Preview Modal */}
      {showPreview && holdings.length > 0 && (
        <PortfolioPreview 
          portfolio={{ holdings }}
          validationResponse={validationResponse}
          onClose={() => setShowPreview(false)}
          onConfirm={handleSubmit}
          isLoading={formState.isSubmitting}
        />
      )}
    </div>
  )
}

export default InputForm