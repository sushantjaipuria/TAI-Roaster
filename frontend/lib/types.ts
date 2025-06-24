// =============================================================================
// ENHANCED TYPESCRIPT TYPES - Matches Backend Schemas
// =============================================================================

// Enums matching backend
export type RiskTolerance = 'conservative' | 'moderate' | 'aggressive'
export type InvestmentStyle = 'growth' | 'value' | 'income' | 'balanced'
export type TimeHorizon = 'short' | 'medium' | 'long'
export type AnalysisType = 'comprehensive' | 'quick' | 'risk_only' | 'diversification_only'

// =============================================================================
// USER PROFILE TYPES
// =============================================================================

export interface UserProfile {
  riskTolerance: RiskTolerance
  investmentAmount: number
  investmentStyle: InvestmentStyle
  timeHorizon: TimeHorizon
  goals: string[]
  age?: number
  annualIncome?: number
}

// =============================================================================
// PORTFOLIO INPUT TYPES (New Schema-based)
// =============================================================================

export interface PortfolioHolding {
  ticker: string
  quantity: number
  avg_buy_price: number
  buy_date?: string  // ISO date string
  current_price?: number
}

export interface PortfolioInput {
  holdings: PortfolioHolding[]
  name?: string
}

// =============================================================================
// FILE UPLOAD TYPES
// =============================================================================

export interface FileUploadRequest {
  filename: string
  fileSize: number
  contentType?: string
}

export interface FileUploadState {
  file: File | null
  isUploading: boolean
  isProcessing: boolean
  uploadProgress: number
  portfolio: PortfolioInput | null
  errors: string[]
  warnings: string[]
}

export interface FileParseResponse {
  success: boolean
  portfolio?: PortfolioInput
  errors: string[]
  warnings: string[]
  rowsProcessed?: number
  rowsSkipped?: number
}

// =============================================================================
// VALIDATION TYPES
// =============================================================================

export interface ValidationError {
  field: string
  message: string
  value?: string | number
}

export interface PortfolioValidationResponse {
  isValid: boolean
  errors: ValidationError[]
  warnings: string[]
  totalValue?: number
  holdingsCount?: number
}

export interface ValidationIssue {
  severity: 'error' | 'warning' | 'info'
  field: string
  message: string
  value?: any
  suggestion?: string
}

// =============================================================================
// API REQUEST/RESPONSE TYPES
// =============================================================================

export interface AnalysisRequest {
  portfolio: PortfolioInput
  userProfile: UserProfile
  analysisType?: AnalysisType
  includeRecommendations?: boolean
}

export interface BulkPortfolioUpload {
  fileRequest: FileUploadRequest
  userProfile: UserProfile
  overrideValidation?: boolean
}

export interface PortfolioUpdateRequest {
  portfolioId?: string
  holdingsToAdd: PortfolioHolding[]
  holdingsToRemove: string[]
  holdingsToUpdate: PortfolioHolding[]
}

// Generic API Response wrapper
export interface ApiResponse<T> {
  success: boolean
  data?: T
  error?: string
  message?: string
}

// =============================================================================
// ONBOARDING API TYPES
// =============================================================================

// Backend request format (snake_case)
export interface UserProfileApiRequest {
  risk_tolerance: RiskTolerance
  investment_amount: number
  investment_style: InvestmentStyle
  time_horizon: TimeHorizon
  goals: string[]
}

// Backend response format
export interface OnboardingApiResponse {
  success: boolean
  session_id: string
  message: string
  data?: UserProfileApiRequest
}

// =============================================================================
// LEGACY PORTFOLIO TYPES (For backward compatibility)
// =============================================================================

export interface PortfolioItem {
  ticker: string
  quantity: number
  avgPrice: number
  currentPrice?: number
  totalValue?: number
  allocation?: number
}

export interface Portfolio {
  items: PortfolioItem[]
  totalValue: number
  lastUpdated: Date
}

// =============================================================================
// ANALYSIS RESULT TYPES
// =============================================================================

export interface StockAnalysis {
  ticker: string
  score: number
  recommendation: 'buy' | 'hold' | 'sell'
  reasoning: string
  riskLevel: 'low' | 'medium' | 'high'
  targetPrice?: number
  priceChange?: number
}

export interface AllocationBreakdown {
  sectors: { [sector: string]: number }
  assetTypes: { [type: string]: number }
  riskLevels: { [level: string]: number }
}

export interface PortfolioAnalysis {
  overallScore: number
  riskLevel: 'low' | 'medium' | 'high'
  diversificationScore: number
  summary: string
  recommendations: string[]
  redFlags: string[]
  allocation: {
    current: AllocationBreakdown
    recommended: AllocationBreakdown
  }
  stocks: StockAnalysis[]
  insights: {
    [ticker: string]: string
  }
}

// =============================================================================
// UI STATE TYPES
// =============================================================================

export interface OnboardingState {
  currentStep: number
  totalSteps: number
  userProfile: Partial<UserProfile>
  isComplete: boolean
}

export interface InputFormState {
  // Manual entry state
  currentHolding: Partial<PortfolioHolding>
  holdings: PortfolioHolding[]
  
  // Validation state
  validationErrors: ValidationError[]
  validationWarnings: string[]
  isValidating: boolean
  
  // Form state
  isDirty: boolean
  isSubmitting: boolean
  submitErrors: string[]
}

export interface AnalysisState {
  requestId: string | null
  isLoading: boolean
  progress: number
  analysis: PortfolioAnalysis | null
  error: string | null
}

// =============================================================================
// FORM COMPONENT TYPES
// =============================================================================

export interface FormFieldProps {
  label: string
  error?: string
  required?: boolean
  helpText?: string
  children: React.ReactNode
}

export interface StockEntryProps {
  holding: Partial<PortfolioHolding>
  onChange: (holding: Partial<PortfolioHolding>) => void
  onAdd: () => void
  onClear: () => void
  errors?: ValidationError[]
  isValidating?: boolean
}

export interface HoldingsListProps {
  holdings: PortfolioHolding[]
  onEdit: (index: number, holding: PortfolioHolding) => void
  onRemove: (index: number) => void
  onReorder?: (fromIndex: number, toIndex: number) => void
  validationResponse?: PortfolioValidationResponse
}

export interface FileUploadProps {
  onFileSelect: (file: File) => void
  onFileRemove: () => void
  onUpload: () => Promise<void>
  state: FileUploadState
  acceptedFormats: string[]
  maxFileSize: number
}

export interface PortfolioPreviewProps {
  portfolio: PortfolioInput
  validationResponse?: PortfolioValidationResponse
  onEdit: () => void
  onConfirm: () => void
  isLoading?: boolean
}

// =============================================================================
// UTILITY TYPES
// =============================================================================

export interface SampleFormat {
  csvExample: {
    headers: string[]
    sampleRows: string[][]
  }
  supportedFormats: string[]
  columnVariations: { [key: string]: string }
  dataFormats: { [key: string]: string }
  notes: string[]
}

export interface ToastMessage {
  id: string
  type: 'success' | 'error' | 'warning' | 'info'
  title: string
  message?: string
  duration?: number
  actions?: Array<{
    label: string
    onClick: () => void
  }>
}

// =============================================================================
// FORM VALIDATION TYPES
// =============================================================================

export interface FieldValidation {
  isValid: boolean
  error?: string
  warning?: string
}

export interface FormValidation {
  [fieldName: string]: FieldValidation
}

export interface FormState<T> {
  values: T
  errors: { [K in keyof T]?: string }
  warnings: { [K in keyof T]?: string }
  touched: { [K in keyof T]?: boolean }
  isSubmitting: boolean
  isValid: boolean
  isDirty: boolean
}

// =============================================================================
// COMPONENT STATE TYPES
// =============================================================================

export interface ManualEntryState {
  holdings: PortfolioHolding[]
  currentHolding: Partial<PortfolioHolding>
  editingIndex: number | null
  showAdvanced: boolean
}

export interface FileUploadComponentState {
  dragActive: boolean
  uploadState: FileUploadState
  showPreview: boolean
  selectedFile: File | null
}

export interface ValidationSummary {
  totalErrors: number
  totalWarnings: number
  isValid: boolean
  criticalIssues: string[]
  suggestions: string[]
}

// =============================================================================
// HOOK RETURN TYPES
// =============================================================================

export interface UsePortfolioInput {
  // State
  holdings: PortfolioHolding[]
  currentHolding: Partial<PortfolioHolding>
  validationResponse: PortfolioValidationResponse | null
  isValidating: boolean
  
  // Actions
  addHolding: (holding: PortfolioHolding) => void
  removeHolding: (index: number) => void
  updateHolding: (index: number, holding: PortfolioHolding) => void
  setCurrentHolding: (holding: Partial<PortfolioHolding>) => void
  validatePortfolio: () => Promise<void>
  clearPortfolio: () => void
  
  // Computed
  totalValue: number
  holdingsCount: number
  isValid: boolean
}

export interface UseFileUpload {
  // State
  uploadState: FileUploadState
  
  // Actions
  selectFile: (file: File) => void
  removeFile: () => void
  uploadFile: () => Promise<void>
  clearUpload: () => void
  
  // Utils
  validateFile: (file: File) => { isValid: boolean; errors: string[] }
}

export interface UseValidation {
  // State
  validationResponse: PortfolioValidationResponse | null
  isValidating: boolean
  
  // Actions
  validatePortfolio: (portfolio: PortfolioInput, userProfile?: UserProfile) => Promise<void>
  clearValidation: () => void
  
  // Computed
  validationSummary: ValidationSummary
}

// =============================================================================
// CONSTANTS AND DEFAULTS
// =============================================================================

export const RISK_TOLERANCE_OPTIONS: Array<{ value: RiskTolerance; label: string; description: string }> = [
  { 
    value: 'conservative', 
    label: 'Conservative', 
    description: 'Prefer stable returns with minimal risk' 
  },
  { 
    value: 'moderate', 
    label: 'Moderate', 
    description: 'Balance between growth and stability' 
  },
  { 
    value: 'aggressive', 
    label: 'Aggressive', 
    description: 'Seek high returns, comfortable with higher risk' 
  }
]

export const INVESTMENT_STYLE_OPTIONS: Array<{ value: InvestmentStyle; label: string; description: string }> = [
  { 
    value: 'growth', 
    label: 'Growth', 
    description: 'Focus on companies with high growth potential' 
  },
  { 
    value: 'value', 
    label: 'Value', 
    description: 'Look for undervalued companies with strong fundamentals' 
  },
  { 
    value: 'income', 
    label: 'Income', 
    description: 'Prioritize dividend-paying stocks for regular income' 
  },
  { 
    value: 'balanced', 
    label: 'Balanced', 
    description: 'Mix of growth, value, and income strategies' 
  }
]

export const TIME_HORIZON_OPTIONS: Array<{ value: TimeHorizon; label: string; description: string }> = [
  { 
    value: 'short', 
    label: 'Short Term (< 3 years)', 
    description: 'Planning to invest for less than 3 years' 
  },
  { 
    value: 'medium', 
    label: 'Medium Term (3-10 years)', 
    description: 'Investment horizon between 3 to 10 years' 
  },
  { 
    value: 'long', 
    label: 'Long Term (> 10 years)', 
    description: 'Long-term investment for more than 10 years' 
  }
]

export const DEFAULT_USER_PROFILE: Partial<UserProfile> = {
  riskTolerance: 'moderate',
  investmentStyle: 'balanced',
  timeHorizon: 'medium',
  goals: []
}

export const DEFAULT_HOLDING: Partial<PortfolioHolding> = {
  ticker: '',
  quantity: 0,
  avg_buy_price: 0
}

export const FILE_UPLOAD_CONFIG = {
  maxFileSize: 10 * 1024 * 1024, // 10MB
  acceptedFormats: ['.csv', '.xlsx', '.xls', '.tsv'],
  acceptedMimeTypes: [
    'text/csv',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/vnd.ms-excel',
    'text/tab-separated-values'
  ]
}

// =============================================================================
// TYPE GUARDS AND UTILITIES
// =============================================================================

export const isValidRiskTolerance = (value: string): value is RiskTolerance => {
  return ['conservative', 'moderate', 'aggressive'].includes(value)
}

export const isValidInvestmentStyle = (value: string): value is InvestmentStyle => {
  return ['growth', 'value', 'income', 'balanced'].includes(value)
}

export const isValidTimeHorizon = (value: string): value is TimeHorizon => {
  return ['short', 'medium', 'long'].includes(value)
}

export const isPortfolioHolding = (obj: any): obj is PortfolioHolding => {
  return obj && 
    typeof obj.ticker === 'string' && 
    typeof obj.quantity === 'number' && 
    typeof obj.avg_buy_price === 'number'
}

export const isValidationError = (obj: any): obj is ValidationError => {
  return obj && 
    typeof obj.field === 'string' && 
    typeof obj.message === 'string'
}
