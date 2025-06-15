'use client'

import React, { useState, useEffect } from 'react'
import { DollarSign } from 'lucide-react'
import { formatCurrency } from '@/lib/utils'

interface InvestmentSizeCardProps {
  value?: number
  onChange: (value: number) => void
  error?: string
}

const quickAmounts = [
  { label: '$10K', value: 10000 },
  { label: '$25K', value: 25000 },
  { label: '$50K', value: 50000 },
  { label: '$100K', value: 100000 },
  { label: '$250K', value: 250000 },
  { label: '$500K', value: 500000 },
]

export function InvestmentSizeCard({ value, onChange, error }: InvestmentSizeCardProps) {
  const [inputValue, setInputValue] = useState('')
  const [isFocused, setIsFocused] = useState(false)

  useEffect(() => {
    if (value && !isFocused) {
      setInputValue(value.toString())
    }
  }, [value, isFocused])

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const rawValue = e.target.value.replace(/[^0-9.]/g, '')
    setInputValue(rawValue)
    
    const numValue = parseFloat(rawValue)
    if (!isNaN(numValue) && numValue > 0) {
      onChange(numValue)
    }
  }

  const handleQuickAmount = (amount: number) => {
    setInputValue(amount.toString())
    onChange(amount)
  }

  const formatInputValue = () => {
    if (isFocused) return inputValue
    if (value) return formatCurrency(value).replace('$', '')
    return inputValue
  }

  return (
    <div className="max-w-2xl mx-auto">
      <div className="card">
        <div className="card-header text-center">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-primary-100 rounded-full mb-4">
            <DollarSign size={32} className="text-primary-600" />
          </div>
          <h3 className="card-title">Investment Amount</h3>
          <p className="text-gray-600">How much are you planning to invest initially?</p>
        </div>

        <div className="card-content">
          {/* Custom Amount Input */}
          <div className="mb-6">
            <label htmlFor="investment-amount" className="block text-sm font-medium text-gray-700 mb-2">
              Investment Amount
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <span className="text-gray-500 text-lg">$</span>
              </div>
              <input
                id="investment-amount"
                type="text"
                value={formatInputValue()}
                onChange={handleInputChange}
                onFocus={() => setIsFocused(true)}
                onBlur={() => setIsFocused(false)}
                placeholder="Enter amount"
                className={`
                  input pl-8 text-lg
                  ${error ? 'border-danger-300 focus:border-danger-500 focus:ring-danger-500' : ''}
                `}
              />
            </div>
            {error && (
              <p className="mt-2 text-sm text-danger-600">{error}</p>
            )}
          </div>

          {/* Quick Amount Buttons */}
          <div>
            <p className="text-sm font-medium text-gray-700 mb-3">Or choose a quick amount:</p>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
              {quickAmounts.map((amount) => (
                <button
                  key={amount.value}
                  onClick={() => handleQuickAmount(amount.value)}
                  className={`
                    px-4 py-3 rounded-lg border text-center transition-colors
                    ${value === amount.value
                      ? 'bg-primary-600 text-white border-primary-600'
                      : 'bg-white text-gray-700 border-gray-300 hover:bg-primary-50 hover:border-primary-300'
                    }
                  `}
                >
                  <div className="font-medium">{amount.label}</div>
                  <div className="text-xs opacity-75">
                    {formatCurrency(amount.value)}
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Investment Range Guidance */}
          <div className="mt-6 p-4 bg-blue-50 rounded-lg">
            <h4 className="font-medium text-blue-900 mb-2">Investment Guidance</h4>
            <div className="text-sm text-blue-800 space-y-1">
              <p>• Minimum recommended: $1,000 for proper diversification</p>
              <p>• Most users invest: $10,000 - $100,000 initially</p>
              <p>• You can always add more funds later</p>
            </div>
          </div>

          {value && (
            <div className="mt-4 p-4 bg-gray-50 rounded-lg">
              <h4 className="font-medium text-gray-900 mb-2">Your Investment</h4>
              <p className="text-gray-600 text-sm">
                You're planning to invest <strong className="text-gray-900">{formatCurrency(value)}</strong>.
                This will allow for good diversification across multiple asset classes and investment opportunities.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
} 