/**
 * Portfolio Preview Component
 * 
 * Shows parsed portfolio data for user review before final submission:
 * - Portfolio summary and statistics
 * - Detailed holdings breakdown
 * - Validation results display
 * - Sector allocation visualization
 * - Action buttons for edit/confirm
 * 
 * Features:
 * - Comprehensive portfolio overview
 * - Validation status with errors/warnings
 * - Interactive holdings table
 * - Visual allocation charts
 * - Modal/overlay interface
 * - Responsive design
 */

'use client'

import React, { useState, useMemo } from 'react'
import { 
  PortfolioInput, 
  PortfolioValidationResponse,
  PortfolioHolding 
} from '../lib/types'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Progress } from '@/components/ui/progress'
import { 
  X, 
  CheckCircle, 
  AlertTriangle, 
  Info, 
  PieChart,
  TrendingUp,
  DollarSign,
  BarChart3,
  Edit,
  Send
} from 'lucide-react'

interface PortfolioPreviewProps {
  portfolio: PortfolioInput
  validationResponse?: PortfolioValidationResponse | null
  onClose: () => void
  onConfirm: () => void
  isLoading?: boolean
}

const PortfolioPreview: React.FC<PortfolioPreviewProps> = ({
  portfolio,
  validationResponse,
  onClose,
  onConfirm,
  isLoading = false
}) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'holdings' | 'validation'>('overview')
  
  const portfolioStats = useMemo(() => {
    const holdings = portfolio.holdings
    const totalValue = holdings.reduce((sum, h) => sum + (h.quantity * h.avg_buy_price), 0)
    const totalShares = holdings.reduce((sum, h) => sum + h.quantity, 0)
    const avgPricePerShare = totalShares > 0 ? totalValue / totalShares : 0
    
    return {
      totalValue,
      totalShares,
      avgPricePerShare,
      holdingsCount: holdings.length
    }
  }, [portfolio.holdings])

  return (
    <div className="fixed inset-0 z-50 bg-black bg-opacity-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-4xl max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b">
          <h2 className="text-2xl font-bold text-gray-900">Portfolio Preview</h2>
          <Button variant="ghost" size="sm" onClick={onClose}>
            <X className="h-5 w-5" />
          </Button>
        </div>
        
        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          <div className="space-y-6">
            {/* Portfolio Summary */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-gray-600">Total Value</p>
                      <p className="text-2xl font-bold text-green-600">
                        ₹{portfolioStats.totalValue.toLocaleString('en-IN')}
                      </p>
                    </div>
                    <DollarSign className="h-8 w-8 text-green-600" />
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-gray-600">Holdings</p>
                      <p className="text-2xl font-bold text-blue-600">
                        {portfolioStats.holdingsCount}
                      </p>
                    </div>
                    <BarChart3 className="h-8 w-8 text-blue-600" />
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-gray-600">Avg Price</p>
                      <p className="text-2xl font-bold text-purple-600">
                        ₹{portfolioStats.avgPricePerShare.toFixed(0)}
                      </p>
                    </div>
                    <TrendingUp className="h-8 w-8 text-purple-600" />
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Holdings Table */}
            <Card>
              <CardHeader>
                <CardTitle>Holdings Breakdown</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">Ticker</th>
                        <th className="px-4 py-3 text-right text-sm font-medium text-gray-600">Quantity</th>
                        <th className="px-4 py-3 text-right text-sm font-medium text-gray-600">Avg Price</th>
                        <th className="px-4 py-3 text-right text-sm font-medium text-gray-600">Total Value</th>
                        <th className="px-4 py-3 text-right text-sm font-medium text-gray-600">Allocation</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200">
                      {portfolio.holdings.map((holding, index) => {
                        const totalValue = holding.quantity * holding.avg_buy_price
                        const allocation = (totalValue / portfolioStats.totalValue) * 100
                        
                        return (
                          <tr key={index} className="hover:bg-gray-50">
                            <td className="px-4 py-3 font-medium text-gray-900">{holding.ticker}</td>
                            <td className="px-4 py-3 text-right text-sm text-gray-900">
                              {holding.quantity.toLocaleString()}
                            </td>
                            <td className="px-4 py-3 text-right text-sm text-gray-900">
                              ₹{holding.avg_buy_price.toFixed(2)}
                            </td>
                            <td className="px-4 py-3 text-right font-medium text-gray-900">
                              ₹{totalValue.toLocaleString('en-IN')}
                            </td>
                            <td className="px-4 py-3 text-right text-sm font-medium">
                              {allocation.toFixed(1)}%
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
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
                    Validation Results
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {validationResponse.errors && validationResponse.errors.length > 0 && (
                    <Alert variant="destructive" className="mb-4">
                      <AlertTriangle className="h-4 w-4" />
                      <AlertTitle>Validation Errors</AlertTitle>
                      <AlertDescription>
                        <ul className="list-disc list-inside space-y-1">
                          {validationResponse.errors.map((error, index) => (
                            <li key={index}>
                              <span className="font-medium">{error.field}:</span> {error.message}
                            </li>
                          ))}
                        </ul>
                      </AlertDescription>
                    </Alert>
                  )}
                  
                  {validationResponse.warnings && validationResponse.warnings.length > 0 && (
                    <Alert>
                      <Info className="h-4 w-4" />
                      <AlertTitle>Recommendations</AlertTitle>
                      <AlertDescription>
                        <ul className="list-disc list-inside space-y-1">
                          {validationResponse.warnings.map((warning, index) => (
                            <li key={index}>{warning}</li>
                          ))}
                        </ul>
                      </AlertDescription>
                    </Alert>
                  )}
                </CardContent>
              </Card>
            )}
          </div>
        </div>
        
        {/* Footer */}
        <div className="flex items-center justify-between p-6 border-t bg-gray-50">
          <div className="text-sm text-gray-600">
            {portfolio.holdings.length} holdings • ₹{portfolioStats.totalValue.toLocaleString('en-IN')} total value
          </div>
          
          <div className="flex items-center gap-4">
            <Button variant="outline" onClick={onClose}>
              <Edit className="h-4 w-4 mr-2" />
              Edit Portfolio
            </Button>
            
            <Button 
              onClick={onConfirm}
                             disabled={isLoading || (validationResponse ? !validationResponse.isValid : false)}
              className="min-w-[140px]"
            >
              {isLoading ? (
                <div className="flex items-center gap-2">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                  Processing...
                </div>
              ) : (
                <>
                  <Send className="h-4 w-4 mr-2" />
                  Confirm & Analyze
                </>
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default PortfolioPreview