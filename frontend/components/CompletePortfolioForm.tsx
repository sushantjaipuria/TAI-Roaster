/**
 * Complete Portfolio Form Component
 * 
 * Single-screen form that combines:
 * 1. User Investment Profile (risk, amount, style, horizon, goals)
 * 2. Portfolio Input (manual entry + file upload)
 * 3. Review & Submit
 * 
 * Top-down flow design for simplicity and better UX
 */

'use client'

import React, { useState, useCallback } from 'react'
import { useRouter } from 'next/navigation'
import { 
  UserProfile, 
  PortfolioHolding, 
  PortfolioInput,
  FileUploadState,
  DEFAULT_USER_PROFILE,
  DEFAULT_HOLDING,
  RISK_TOLERANCE_OPTIONS,
  INVESTMENT_STYLE_OPTIONS,
  TIME_HORIZON_OPTIONS
} from '../lib/types'
import { PortfolioApiClient } from '../lib/api'
import FileUploadSection from './FileUploadSection'

// Simple styled components (no external UI library dependencies)
const Card = ({ children, className = '' }: { children: React.ReactNode; className?: string }) => (
  <div className={`bg-white rounded-lg shadow-sm border border-gray-200 ${className}`}>
    {children}
  </div>
)

const CardHeader = ({ children }: { children: React.ReactNode }) => (
  <div className="px-6 py-4 border-b border-gray-200">
    {children}
  </div>
)

const CardTitle = ({ children, className = '' }: { children: React.ReactNode; className?: string }) => (
  <h3 className={`text-lg font-semibold text-gray-900 ${className}`}>
    {children}
  </h3>
)

const CardContent = ({ children, className = '' }: { children: React.ReactNode; className?: string }) => (
  <div className={`px-6 py-4 ${className}`}>
    {children}
  </div>
)

const Button = ({ children, onClick, disabled = false, className = '', size = 'md', variant = 'primary' }: { 
  children: React.ReactNode; 
  onClick?: () => void; 
  disabled?: boolean; 
  className?: string; 
  size?: 'sm' | 'md' | 'lg';
  variant?: 'primary' | 'secondary' | 'outline';
}) => {
  const baseClasses = 'inline-flex items-center justify-center rounded-md font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2'
  const sizeClasses = {
    sm: 'px-3 py-2 text-sm',
    md: 'px-4 py-2 text-sm',
    lg: 'px-6 py-3 text-base'
  }
  const variantClasses = {
    primary: 'bg-blue-600 text-white hover:bg-blue-700 focus:ring-blue-500 disabled:bg-gray-300',
    secondary: 'bg-gray-100 text-gray-900 hover:bg-gray-200 focus:ring-gray-500',
    outline: 'border border-gray-300 bg-white text-gray-700 hover:bg-gray-50 focus:ring-blue-500'
  }
  
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`${baseClasses} ${sizeClasses[size]} ${variantClasses[variant]} ${disabled ? 'cursor-not-allowed' : ''} ${className}`}
    >
      {children}
    </button>
  )
}

const Progress = ({ value, className = '' }: { value: number; className?: string }) => (
  <div className={`w-full bg-gray-200 rounded-full h-2 ${className}`}>
    <div
      className="bg-blue-600 h-2 rounded-full transition-all duration-300"
      style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
    />
  </div>
)

const Alert = ({ children, className = '' }: { children: React.ReactNode; className?: string }) => (
  <div className={`rounded-md border border-yellow-200 bg-yellow-50 p-4 ${className}`}>
    {children}
  </div>
)

const AlertDescription = ({ children, className = '' }: { children: React.ReactNode; className?: string }) => (
  <div className={`text-sm text-yellow-700 ${className}`}>
    {children}
  </div>
)

interface CompletePortfolioFormProps {
  className?: string
}

const CompletePortfolioForm: React.FC<CompletePortfolioFormProps> = ({ className = '' }) => {
  const router = useRouter()
  
  // =============================================================================
  // STATE MANAGEMENT
  // =============================================================================
  
  // User Profile State
  const [userProfile, setUserProfile] = useState<UserProfile>({
    riskTolerance: 'moderate',
    investmentAmount: 0,
    investmentStyle: 'balanced',
    timeHorizon: 'medium',
    goals: []
  })
  
  // Portfolio State
  const [holdings, setHoldings] = useState<PortfolioHolding[]>([])
  const [currentHolding, setCurrentHolding] = useState<Partial<PortfolioHolding>>({
    ticker: '',
    quantity: 0,
    avgBuyPrice: 0
  })
  
  // Form State
  const [activeTab, setActiveTab] = useState<'manual' | 'upload'>('upload')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState<string | null>(null)
  const [validationErrors, setValidationErrors] = useState<{[key: string]: string}>({})
  
  // Progress tracking
  const [completedSections, setCompletedSections] = useState<{
    profile: boolean
    portfolio: boolean
  }>({
    profile: false,
    portfolio: false
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
  
  // =============================================================================
  // VALIDATION & HELPERS
  // =============================================================================
  
  const validateProfile = useCallback((): boolean => {
    const isValid = !!(
      userProfile.riskTolerance &&
      userProfile.investmentAmount > 0 &&
      userProfile.investmentStyle &&
      userProfile.timeHorizon
    )
    
    setCompletedSections(prev => ({ ...prev, profile: isValid }))
    return isValid
  }, [userProfile])
  
  const validatePortfolio = useCallback((): boolean => {
    const isValid = holdings.length > 0
    setCompletedSections(prev => ({ ...prev, portfolio: isValid }))
    return isValid
  }, [holdings])
  
  const calculateProgress = useCallback((): number => {
    let progress = 0
    if (completedSections.profile) progress += 50
    if (completedSections.portfolio) progress += 50
    return progress
  }, [completedSections])
  
  // =============================================================================
  // HANDLERS
  // =============================================================================
  
  const handleProfileFieldChange = (field: keyof UserProfile, value: any) => {
    setUserProfile(prev => ({
      ...prev,
      [field]: value
    }))
    setValidationErrors(prev => {
      const newErrors = { ...prev }
      delete newErrors.profile
      return newErrors
    })
  }
  
  const handleAddHolding = useCallback(() => {
    if (currentHolding.ticker && currentHolding.quantity && currentHolding.avgBuyPrice) {
      const newHolding: PortfolioHolding = {
        ticker: currentHolding.ticker.toUpperCase(),
        quantity: currentHolding.quantity,
        avgBuyPrice: currentHolding.avgBuyPrice,
        buyDate: currentHolding.buyDate
      }
      
      // Check for duplicates
      const existingIndex = holdings.findIndex(h => h.ticker === newHolding.ticker)
      if (existingIndex !== -1) {
        setValidationErrors(prev => ({
          ...prev,
          holding: `${newHolding.ticker} already exists in your portfolio`
        }))
        return
      }
      
      setHoldings(prev => [...prev, newHolding])
      setCurrentHolding({
        ticker: '',
        quantity: 0,
        avgBuyPrice: 0
      })
      setValidationErrors(prev => {
        const newErrors = { ...prev }
        delete newErrors.holding
        return newErrors
      })
    }
  }, [currentHolding, holdings])
  
  const handleRemoveHolding = useCallback((index: number) => {
    setHoldings(prev => prev.filter((_, i) => i !== index))
  }, [])
  
  const handleSubmit = useCallback(async () => {
    // Validate all sections
    const isProfileValid = validateProfile()
    const isPortfolioValid = validatePortfolio()
    
    if (!isProfileValid) {
      setValidationErrors(prev => ({
        ...prev,
        profile: 'Please complete your investment profile'
      }))
    }
    
    if (!isPortfolioValid) {
      setValidationErrors(prev => ({
        ...prev,
        portfolio: 'Please add at least one stock to your portfolio'
      }))
    }
    
    if (!isProfileValid || !isPortfolioValid) {
      return
    }
    
    setIsSubmitting(true)
    setSubmitError(null)
    
    try {
      const portfolio: PortfolioInput = { holdings }
      
      // Submit to API (reusing existing functionality)
      const response = await PortfolioApiClient.analyzePortfolio(
        portfolio,
        userProfile,
        'comprehensive'
      )
      
      if (response && response.analysisId) {
        // Navigate to results page with analysis ID
        router.push(`/results?id=${response.analysisId}`)
      } else {
        setSubmitError('Failed to start portfolio analysis')
      }
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : 'An error occurred')
    } finally {
      setIsSubmitting(false)
    }
  }, [userProfile, holdings, validateProfile, validatePortfolio, router])
  
  // =============================================================================
  // EFFECTS
  // =============================================================================
  
  React.useEffect(() => {
    validateProfile()
  }, [validateProfile])
  
  React.useEffect(() => {
    validatePortfolio()
  }, [validatePortfolio])
  
  // =============================================================================
  // RENDER HELPERS
  // =============================================================================
  
  const renderHeader = () => (
    <div className="text-center mb-8">
      <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
        TAI Roaster Portfolio Setup
      </h1>
      <p className="text-lg text-gray-600 max-w-2xl mx-auto">
        Set up your investment profile and portfolio to get personalized AI-powered analysis and recommendations.
      </p>
    </div>
  )
  
  const renderProgress = () => (
    <Card className="mb-8">
      <CardContent className="pt-6">
        <div className="flex items-center justify-between mb-4">
          <span className="text-sm font-medium text-gray-700">
            Overall Progress
          </span>
          <span className="text-sm font-medium text-gray-700">
            {Math.round(calculateProgress())}% Complete
          </span>
        </div>
        <Progress value={calculateProgress()} className="mb-4" />
        
        <div className="flex items-center justify-between text-sm">
          <div className="flex items-center gap-2">
            {completedSections.profile ? (
              <div className="h-4 w-4 bg-green-600 rounded-full flex items-center justify-center">
                <span className="text-white text-xs">✓</span>
              </div>
            ) : (
              <div className="h-4 w-4 border-2 border-gray-300 rounded-full" />
            )}
            <span className={completedSections.profile ? 'text-green-600' : 'text-gray-500'}>
              Investment Profile
            </span>
          </div>
          
          <div className="flex items-center gap-2">
            {completedSections.portfolio ? (
              <div className="h-4 w-4 bg-green-600 rounded-full flex items-center justify-center">
                <span className="text-white text-xs">✓</span>
              </div>
            ) : (
              <div className="h-4 w-4 border-2 border-gray-300 rounded-full" />
            )}
            <span className={completedSections.portfolio ? 'text-green-600' : 'text-gray-500'}>
              Portfolio Input
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
  
  const renderProfileSection = () => (
    <Card className="mb-8">
      <CardHeader>
        <CardTitle>📊 Investment Profile</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Investment Amount */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Investment Amount (₹) *
          </label>
          <input
            type="number"
            value={userProfile.investmentAmount || ''}
            onChange={(e) => handleProfileFieldChange('investmentAmount', parseFloat(e.target.value) || 0)}
            placeholder="Total amount to invest"
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <p className="text-xs text-gray-500 mt-1">
            Enter the total amount you plan to invest across all holdings
          </p>
        </div>
        
        {/* Risk Tolerance */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Risk Tolerance *
          </label>
          <div className="grid grid-cols-1 gap-3">
            {RISK_TOLERANCE_OPTIONS.map(option => (
              <div
                key={option.value}
                className={`p-4 border rounded-lg cursor-pointer transition-all ${
                  userProfile.riskTolerance === option.value 
                    ? 'border-blue-500 bg-blue-50 ring-2 ring-blue-200' 
                    : 'border-gray-200 hover:border-gray-300'
                }`}
                onClick={() => handleProfileFieldChange('riskTolerance', option.value)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="font-medium">{option.label}</span>
                      {userProfile.riskTolerance === option.value && (
                        <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">Selected</span>
                      )}
                    </div>
                    <p className="text-sm text-gray-600 mt-1">{option.description}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
        
        {/* Investment Style */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Investment Style *
          </label>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {INVESTMENT_STYLE_OPTIONS.map(option => (
              <div
                key={option.value}
                className={`p-3 border rounded-lg cursor-pointer transition-all ${
                  userProfile.investmentStyle === option.value 
                    ? 'border-blue-500 bg-blue-50 ring-1 ring-blue-200' 
                    : 'border-gray-200 hover:border-gray-300'
                }`}
                onClick={() => handleProfileFieldChange('investmentStyle', option.value)}
              >
                <div className="flex items-center gap-2 mb-1">
                  <span className="font-medium">{option.label}</span>
                  {userProfile.investmentStyle === option.value && (
                    <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">Selected</span>
                  )}
                </div>
                <p className="text-xs text-gray-600">{option.description}</p>
              </div>
            ))}
          </div>
        </div>
        
        {/* Time Horizon */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Investment Time Horizon *
          </label>
          <div className="grid grid-cols-1 gap-2">
            {TIME_HORIZON_OPTIONS.map(option => (
              <label
                key={option.value}
                className={`flex items-center p-3 border rounded-lg cursor-pointer transition-all ${
                  userProfile.timeHorizon === option.value 
                    ? 'border-blue-500 bg-blue-50' 
                    : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                <input
                  type="radio"
                  name="timeHorizon"
                  value={option.value}
                  checked={userProfile.timeHorizon === option.value}
                  onChange={(e) => handleProfileFieldChange('timeHorizon', e.target.value)}
                  className="mr-3"
                />
                <div className="flex-1">
                  <span className="font-medium">{option.label}</span>
                  <p className="text-sm text-gray-600">{option.description}</p>
                </div>
              </label>
            ))}
          </div>
        </div>
      </CardContent>
      
      {validationErrors.profile && (
        <div className="mx-6 mb-6">
          <Alert>
            <AlertDescription>⚠️ {validationErrors.profile}</AlertDescription>
          </Alert>
        </div>
      )}
    </Card>
  )
  
  const renderManualEntry = () => (
    <div className="space-y-6">
      {/* Current Holdings List */}
      {holdings.length > 0 && (
        <div>
          <h3 className="text-lg font-medium mb-4">Your Portfolio ({holdings.length} holdings)</h3>
          <div className="space-y-3">
            {holdings.map((holding, index) => (
              <Card key={index}>
                <CardContent className="pt-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="font-semibold">{holding.ticker}</div>
                      <div className="text-sm text-gray-600">
                        {holding.quantity} shares @ ₹{holding.avgBuyPrice}
                        {holding.buyDate && ` (${new Date(holding.buyDate).toLocaleDateString()})`}
                      </div>
                      <div className="text-sm font-medium text-gray-900">
                        Total: ₹{(holding.quantity * holding.avgBuyPrice).toLocaleString('en-IN')}
                      </div>
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleRemoveHolding(index)}
                    >
                      Remove
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}
      
      {/* Add New Holding Form */}
      <Card>
        <CardHeader>
          <CardTitle>Add Stock</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1">Ticker Symbol *</label>
              <input
                type="text"
                value={currentHolding.ticker || ''}
                onChange={(e) => setCurrentHolding(prev => ({ 
                  ...prev, 
                  ticker: e.target.value.toUpperCase() 
                }))}
                placeholder="e.g., RELIANCE"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-1">Quantity *</label>
              <input
                type="number"
                value={currentHolding.quantity || ''}
                onChange={(e) => setCurrentHolding(prev => ({ 
                  ...prev, 
                  quantity: parseInt(e.target.value) || 0 
                }))}
                placeholder="Number of shares"
                min="1"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-1">Avg. Buy Price (₹) *</label>
              <input
                type="number"
                value={currentHolding.avgBuyPrice || ''}
                onChange={(e) => setCurrentHolding(prev => ({ 
                  ...prev, 
                  avgBuyPrice: parseFloat(e.target.value) || 0 
                }))}
                placeholder="Price per share"
                min="0"
                step="0.01"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-1">Buy Date (Optional)</label>
              <input
                type="date"
                value={currentHolding.buyDate || ''}
                onChange={(e) => setCurrentHolding(prev => ({ 
                  ...prev, 
                  buyDate: e.target.value 
                }))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
          
          {validationErrors.holding && (
            <Alert>
              <AlertDescription>⚠️ {validationErrors.holding}</AlertDescription>
            </Alert>
          )}
          
          <Button 
            onClick={handleAddHolding}
            disabled={!currentHolding.ticker || !currentHolding.quantity || !currentHolding.avgBuyPrice}
            className="w-full md:w-auto"
          >
            Add to Portfolio
          </Button>
        </CardContent>
      </Card>
    </div>
  )
  
  const renderPortfolioSection = () => (
    <Card className="mb-8">
      <CardHeader>
        <CardTitle>🥧 Portfolio Input</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="mb-6">
          <div className="flex space-x-1 rounded-lg bg-gray-100 p-1">
            <button
              onClick={() => setActiveTab('manual')}
              className={`flex-1 py-2 px-3 text-sm font-medium rounded-md transition-colors ${
                activeTab === 'manual' 
                  ? 'bg-white shadow-sm' 
                  : 'hover:bg-white hover:shadow-sm'
              }`}
            >
              Manual Entry
            </button>
            <button
              onClick={() => setActiveTab('upload')}
              className={`flex-1 py-2 px-3 text-sm font-medium rounded-md transition-colors ${
                activeTab === 'upload' 
                  ? 'bg-white shadow-sm' 
                  : 'hover:bg-white hover:shadow-sm'
              }`}
            >
              File Upload
            </button>
          </div>
        </div>
        
        {activeTab === 'manual' && renderManualEntry()}
        
        {activeTab === 'upload' && (
          <FileUploadSection 
            onSuccess={(portfolio) => {
              setHoldings(portfolio.holdings)
              setActiveTab('manual') // Switch to manual tab to show imported data
            }}
            state={fileUploadState}
            onStateChange={setFileUploadState}
          />
        )}
        
        {validationErrors.portfolio && (
          <Alert className="mt-4">
            <AlertDescription>⚠️ {validationErrors.portfolio}</AlertDescription>
          </Alert>
        )}
      </CardContent>
    </Card>
  )
  
  const renderReviewSection = () => (
    <Card className="mb-8">
      <CardHeader>
        <CardTitle>📤 Review & Submit</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Profile Summary */}
          <div className="p-4 bg-gray-50 rounded-lg">
            <h4 className="font-medium mb-2">Investment Profile</h4>
            <div className="text-sm text-gray-600 space-y-1">
              <p><strong>Risk Tolerance:</strong> {userProfile.riskTolerance || 'Not set'}</p>
              <p><strong>Investment Style:</strong> {userProfile.investmentStyle || 'Not set'}</p>
              <p><strong>Time Horizon:</strong> {userProfile.timeHorizon || 'Not set'}</p>
              <p><strong>Investment Amount:</strong> {userProfile.investmentAmount ? `₹${userProfile.investmentAmount.toLocaleString('en-IN')}` : 'Not set'}</p>
            </div>
          </div>
          
          {/* Portfolio Summary */}
          <div className="p-4 bg-gray-50 rounded-lg">
            <h4 className="font-medium mb-2">Portfolio Summary</h4>
            <div className="text-sm text-gray-600">
              <p><strong>Holdings:</strong> {holdings.length} stocks</p>
              {holdings.length > 0 && (
                <p><strong>Total Value:</strong> ₹{holdings.reduce((sum, h) => sum + (h.quantity * h.avgBuyPrice), 0).toLocaleString('en-IN')}</p>
              )}
            </div>
          </div>
          
          {submitError && (
            <Alert className="border-red-200 bg-red-50">
              <AlertDescription className="text-red-700">⚠️ {submitError}</AlertDescription>
            </Alert>
          )}
          
          <Button 
            onClick={handleSubmit}
            disabled={isSubmitting || !completedSections.profile || !completedSections.portfolio}
            className="w-full"
            size="lg"
          >
            {isSubmitting ? 'Analyzing Portfolio...' : 'Start Portfolio Analysis'}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
  
  // =============================================================================
  // MAIN RENDER
  // =============================================================================
  
  return (
    <div className={`min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 ${className}`}>
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          {renderHeader()}
          {renderProgress()}
          {renderProfileSection()}
          {renderPortfolioSection()}
          {renderReviewSection()}
        </div>
      </div>
    </div>
  )
}

export default CompletePortfolioForm 