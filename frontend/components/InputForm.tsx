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
  
  // UI State
  const [activeTab, setActiveTab] = useState<'manual' | 'upload'>('manual')
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
    } finally {\n      setFormState(prev => ({ ...prev, isValidating: false }))\n    }\n  }, [holdings, userProfile])\n  \n  // =============================================================================\n  // HANDLERS\n  // =============================================================================\n  \n  const handleAddHolding = useCallback(() => {\n    const errors = validateHolding(currentHolding)\n    \n    if (errors.length > 0) {\n      setFormState(prev => ({ ...prev, validationErrors: errors }))\n      return\n    }\n    \n    const newHolding: PortfolioHolding = {\n      ticker: currentHolding.ticker!.toUpperCase().trim(),\n      quantity: currentHolding.quantity!,\n      avgBuyPrice: currentHolding.avgBuyPrice!,\n      buyDate: currentHolding.buyDate,\n      currentPrice: currentHolding.currentPrice\n    }\n    \n    if (editingIndex !== null) {\n      // Update existing holding\n      const updatedHoldings = [...holdings]\n      updatedHoldings[editingIndex] = newHolding\n      setHoldings(updatedHoldings)\n      setEditingIndex(null)\n    } else {\n      // Add new holding\n      setHoldings(prev => [...prev, newHolding])\n    }\n    \n    // Clear current holding\n    setCurrentHolding({ ...DEFAULT_HOLDING })\n    setFormState(prev => ({ ...prev, validationErrors: [] }))\n    \n    // Trigger validation\n    setTimeout(() => validatePortfolio(), 100)\n  }, [currentHolding, holdings, editingIndex, validateHolding, validatePortfolio])\n  \n  const handleEditHolding = useCallback((index: number) => {\n    const holding = holdings[index]\n    setCurrentHolding(holding)\n    setEditingIndex(index)\n    setActiveTab('manual')\n  }, [holdings])\n  \n  const handleRemoveHolding = useCallback((index: number) => {\n    setHoldings(prev => prev.filter((_, i) => i !== index))\n    if (editingIndex === index) {\n      setCurrentHolding({ ...DEFAULT_HOLDING })\n      setEditingIndex(null)\n    }\n    setTimeout(() => validatePortfolio(), 100)\n  }, [editingIndex, validatePortfolio])\n  \n  const handleClearForm = useCallback(() => {\n    setHoldings([])\n    setCurrentHolding({ ...DEFAULT_HOLDING })\n    setEditingIndex(null)\n    setValidationResponse(null)\n    setFormState({\n      currentHolding: { ...DEFAULT_HOLDING },\n      holdings: [],\n      validationErrors: [],\n      validationWarnings: [],\n      isValidating: false,\n      isDirty: false,\n      isSubmitting: false,\n      submitErrors: []\n    })\n  }, [])\n  \n  const handleFileUploadSuccess = useCallback((portfolio: PortfolioInput) => {\n    setHoldings(portfolio.holdings)\n    setActiveTab('manual') // Switch to manual tab to show imported data\n    setTimeout(() => validatePortfolio(), 100)\n  }, [validatePortfolio])\n  \n  const handleSubmit = useCallback(async () => {\n    if (holdings.length === 0) {\n      setFormState(prev => ({\n        ...prev,\n        submitErrors: ['Portfolio cannot be empty. Please add at least one holding.']\n      }))\n      return\n    }\n    \n    setFormState(prev => ({ ...prev, isSubmitting: true, submitErrors: [] }))\n    \n    try {\n      const portfolio: PortfolioInput = { holdings }\n      // First validate the portfolio
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
      await onSubmit(portfolio, userProfile, analysisResult.analysisId)\n    } catch (error: any) {\n      setFormState(prev => ({\n        ...prev,\n        submitErrors: [error.message || 'Failed to submit portfolio']\n      }))\n    } finally {\n      setFormState(prev => ({ ...prev, isSubmitting: false }))\n    }\n  }, [holdings, userProfile, onSubmit])\n  \n  // =============================================================================\n  // COMPUTED VALUES\n  // =============================================================================\n  \n  const totalValue = holdings.reduce((sum, holding) => \n    sum + (holding.quantity * holding.avgBuyPrice), 0\n  )\n  \n  const currentHoldingErrors = validateHolding(currentHolding)\n  const canAddHolding = Object.values(currentHolding).some(val => val !== '' && val !== 0) && \n                       currentHoldingErrors.length === 0\n  \n  const canSubmit = holdings.length > 0 && \n                   validationResponse?.isValid !== false && \n                   !formState.isSubmitting\n  \n  // =============================================================================\n  // RENDER\n  // =============================================================================\n  \n  return (\n    <div className=\"max-w-4xl mx-auto p-6 space-y-8\">\n      {/* Header */}\n      <div className=\"text-center\">\n        <h1 className=\"text-3xl font-bold text-gray-900 mb-2\">\n          Portfolio Analysis Setup\n        </h1>\n        <p className=\"text-gray-600\">\n          Enter your portfolio data and preferences to get personalized investment analysis\n        </p>\n      </div>\n      \n      {/* User Profile Section */}\n      <UserProfileForm \n        userProfile={userProfile}\n        onChange={setUserProfile}\n      />\n      \n      {/* Portfolio Input Tabs */}\n      <Card>\n        <CardHeader>\n          <CardTitle className=\"flex items-center gap-2\">\n            <FileText className=\"h-5 w-5\" />\n            Portfolio Holdings\n          </CardTitle>\n        </CardHeader>\n        <CardContent>\n          <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as 'manual' | 'upload')}>\n            <TabsList className=\"grid w-full grid-cols-2\">\n              <TabsTrigger value=\"manual\">Manual Entry</TabsTrigger>\n              <TabsTrigger value=\"upload\">File Upload</TabsTrigger>\n            </TabsList>\n            \n            {/* Manual Entry Tab */}\n            <TabsContent value=\"manual\" className=\"space-y-6\">\n              {/* Current Holding Form */}\n              <div className=\"grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 p-4 border rounded-lg bg-gray-50\">\n                <div>\n                  <Label htmlFor=\"ticker\">Stock Symbol *</Label>\n                  <Input\n                    id=\"ticker\"\n                    value={currentHolding.ticker || ''}\n                    onChange={(e) => setCurrentHolding(prev => ({ \n                      ...prev, \n                      ticker: e.target.value.toUpperCase() \n                    }))}\n                    placeholder=\"e.g., RELIANCE\"\n                    className={currentHoldingErrors.find(e => e.field === 'ticker') ? 'border-red-500' : ''}\n                  />\n                </div>\n                \n                <div>\n                  <Label htmlFor=\"quantity\">Quantity *</Label>\n                  <Input\n                    id=\"quantity\"\n                    type=\"number\"\n                    value={currentHolding.quantity || ''}\n                    onChange={(e) => setCurrentHolding(prev => ({ \n                      ...prev, \n                      quantity: parseInt(e.target.value) || 0 \n                    }))}\n                    placeholder=\"Number of shares\"\n                    className={currentHoldingErrors.find(e => e.field === 'quantity') ? 'border-red-500' : ''}\n                  />\n                </div>\n                \n                <div>\n                  <Label htmlFor=\"avgBuyPrice\">Avg Buy Price (₹) *</Label>\n                  <Input\n                    id=\"avgBuyPrice\"\n                    type=\"number\"\n                    step=\"0.01\"\n                    value={currentHolding.avgBuyPrice || ''}\n                    onChange={(e) => setCurrentHolding(prev => ({ \n                      ...prev, \n                      avgBuyPrice: parseFloat(e.target.value) || 0 \n                    }))}\n                    placeholder=\"Price per share\"\n                    className={currentHoldingErrors.find(e => e.field === 'avgBuyPrice') ? 'border-red-500' : ''}\n                  />\n                </div>\n                \n                <div className=\"flex items-end\">\n                  <Button \n                    onClick={handleAddHolding}\n                    disabled={!canAddHolding}\n                    className=\"w-full\"\n                  >\n                    <Plus className=\"h-4 w-4 mr-2\" />\n                    {editingIndex !== null ? 'Update' : 'Add'}\n                  </Button>\n                </div>\n              </div>\n              \n              {/* Current Holding Errors */}\n              {currentHoldingErrors.length > 0 && (\n                <Alert variant=\"destructive\">\n                  <AlertTriangle className=\"h-4 w-4\" />\n                  <AlertTitle>Please fix the following errors:</AlertTitle>\n                  <AlertDescription>\n                    <ul className=\"list-disc list-inside\">\n                      {currentHoldingErrors.map((error, index) => (\n                        <li key={index}>{error.message}</li>\n                      ))}\n                    </ul>\n                  </AlertDescription>\n                </Alert>\n              )}\n              \n              {/* Holdings List */}\n              {holdings.length > 0 && (\n                <div className=\"space-y-4\">\n                  <div className=\"flex items-center justify-between\">\n                    <h3 className=\"text-lg font-semibold\">Your Holdings ({holdings.length})</h3>\n                    <div className=\"text-sm text-gray-600\">\n                      Total Value: ₹{totalValue.toLocaleString('en-IN', { maximumFractionDigits: 2 })}\n                    </div>\n                  </div>\n                  \n                  <div className=\"space-y-2\">\n                    {holdings.map((holding, index) => (\n                      <div key={index} className=\"flex items-center justify-between p-3 border rounded-lg bg-white\">\n                        <div className=\"flex-1 grid grid-cols-4 gap-4\">\n                          <div>\n                            <span className=\"font-medium\">{holding.ticker}</span>\n                          </div>\n                          <div>\n                            <span className=\"text-gray-600\">{holding.quantity.toLocaleString()} shares</span>\n                          </div>\n                          <div>\n                            <span className=\"text-gray-600\">₹{holding.avgBuyPrice.toFixed(2)}</span>\n                          </div>\n                          <div>\n                            <span className=\"font-medium\">\n                              ₹{(holding.quantity * holding.avgBuyPrice).toLocaleString('en-IN', { maximumFractionDigits: 2 })}\n                            </span>\n                          </div>\n                        </div>\n                        \n                        <div className=\"flex items-center gap-2\">\n                          <Button \n                            variant=\"outline\" \n                            size=\"sm\"\n                            onClick={() => handleEditHolding(index)}\n                          >\n                            Edit\n                          </Button>\n                          <Button \n                            variant=\"outline\" \n                            size=\"sm\"\n                            onClick={() => handleRemoveHolding(index)}\n                          >\n                            <Trash2 className=\"h-4 w-4\" />\n                          </Button>\n                        </div>\n                      </div>\n                    ))}\n                  </div>\n                </div>\n              )}\n            </TabsContent>\n            \n            {/* File Upload Tab */}\n            <TabsContent value=\"upload\">\n              <FileUploadSection \n                onSuccess={handleFileUploadSuccess}\n                state={fileUploadState}\n                onStateChange={setFileUploadState}\n              />\n            </TabsContent>\n          </Tabs>\n        </CardContent>\n      </Card>\n      \n      {/* Validation Results */}\n      {validationResponse && (\n        <Card>\n          <CardHeader>\n            <CardTitle className=\"flex items-center gap-2\">\n              {validationResponse.isValid ? (\n                <CheckCircle className=\"h-5 w-5 text-green-600\" />\n              ) : (\n                <AlertTriangle className=\"h-5 w-5 text-red-600\" />\n              )}\n              Portfolio Validation\n            </CardTitle>\n          </CardHeader>\n          <CardContent>\n            {validationResponse.errors && validationResponse.errors.length > 0 && (\n              <Alert variant=\"destructive\" className=\"mb-4\">\n                <AlertTriangle className=\"h-4 w-4\" />\n                <AlertTitle>Validation Errors</AlertTitle>\n                <AlertDescription>\n                  <ul className=\"list-disc list-inside\">\n                    {validationResponse.errors.map((error, index) => (\n                      <li key={index}>{error.message}</li>\n                    ))}\n                  </ul>\n                </AlertDescription>\n              </Alert>\n            )}\n            \n            {validationResponse.warnings && validationResponse.warnings.length > 0 && (\n              <Alert className=\"mb-4\">\n                <Info className=\"h-4 w-4\" />\n                <AlertTitle>Recommendations</AlertTitle>\n                <AlertDescription>\n                  <ul className=\"list-disc list-inside\">\n                    {validationResponse.warnings.map((warning, index) => (\n                      <li key={index}>{warning}</li>\n                    ))}\n                  </ul>\n                </AlertDescription>\n              </Alert>\n            )}\n            \n            {validationResponse.isValid && (\n              <Alert>\n                <CheckCircle className=\"h-4 w-4\" />\n                <AlertTitle>Portfolio looks good!</AlertTitle>\n                <AlertDescription>\n                  Your portfolio has been validated successfully. \n                  Total Value: ₹{validationResponse.totalValue?.toLocaleString('en-IN') || 0} • \n                  Holdings: {validationResponse.holdingsCount || 0}\n                </AlertDescription>\n              </Alert>\n            )}\n          </CardContent>\n        </Card>\n      )}\n      \n      {/* Submit Errors */}\n      {formState.submitErrors.length > 0 && (\n        <Alert variant=\"destructive\">\n          <AlertTriangle className=\"h-4 w-4\" />\n          <AlertTitle>Submission Error</AlertTitle>\n          <AlertDescription>\n            <ul className=\"list-disc list-inside\">\n              {formState.submitErrors.map((error, index) => (\n                <li key={index}>{error}</li>\n              ))}\n            </ul>\n          </AlertDescription>\n        </Alert>\n      )}\n      \n      {/* Action Buttons */}\n      <div className=\"flex items-center justify-between pt-6 border-t\">\n        <div className=\"flex items-center gap-4\">\n          <Button \n            variant=\"outline\" \n            onClick={handleClearForm}\n            disabled={formState.isSubmitting}\n          >\n            Clear All\n          </Button>\n          \n          {formState.isDirty && (\n            <Badge variant=\"secondary\">Unsaved changes</Badge>\n          )}\n        </div>\n        \n        <div className=\"flex items-center gap-4\">\n          <Button \n            variant=\"outline\"\n            onClick={() => setShowPreview(true)}\n            disabled={holdings.length === 0}\n          >\n            Preview\n          </Button>\n          \n          <Button \n            onClick={handleSubmit}\n            disabled={!canSubmit}\n            className=\"min-w-[120px]\"\n          >\n            {formState.isSubmitting ? (\n              <div className=\"flex items-center gap-2\">\n                <div className=\"animate-spin rounded-full h-4 w-4 border-b-2 border-white\"></div>\n                Processing...\n              </div>\n            ) : (\n              'Analyze Portfolio'\n            )}\n          </Button>\n        </div>\n      </div>\n      \n      {/* Portfolio Preview Modal */}\n      {showPreview && holdings.length > 0 && (\n        <PortfolioPreview \n          portfolio={{ holdings }}\n          validationResponse={validationResponse}\n          onClose={() => setShowPreview(false)}\n          onConfirm={handleSubmit}\n          isLoading={formState.isSubmitting}\n        />\n      )}\n    </div>\n  )\n}\n\nexport default InputForm"}