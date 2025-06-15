'use client'

import React from 'react'
import { TrendingUp, Shield, DollarSign, Scale } from 'lucide-react'

interface InvestmentStyleCardProps {
  value?: 'growth' | 'value' | 'income' | 'balanced'
  onChange: (value: 'growth' | 'value' | 'income' | 'balanced') => void
  error?: string
}

const styleOptions = [
  {
    value: 'growth' as const,
    title: 'Growth',
    description: 'Focus on companies with high growth potential',
    icon: TrendingUp,
    color: 'bg-green-50 border-green-200',
    selectedColor: 'bg-green-100 border-green-400',
    characteristics: ['High growth potential', 'Technology & innovation', 'Long-term appreciation'],
    example: 'Tech stocks, emerging markets'
  },
  {
    value: 'value' as const,
    title: 'Value',
    description: 'Invest in undervalued companies with strong fundamentals',
    icon: Shield,
    color: 'bg-blue-50 border-blue-200',
    selectedColor: 'bg-blue-100 border-blue-400',
    characteristics: ['Undervalued stocks', 'Strong fundamentals', 'Lower risk profile'],
    example: 'Blue-chip stocks, dividend stocks'
  },
  {
    value: 'income' as const,
    title: 'Income',
    description: 'Generate regular income through dividends and interest',
    icon: DollarSign,
    color: 'bg-purple-50 border-purple-200',
    selectedColor: 'bg-purple-100 border-purple-400',
    characteristics: ['Regular dividends', 'Stable income', 'Lower volatility'],
    example: 'REITs, utility stocks, bonds'
  },
  {
    value: 'balanced' as const,
    title: 'Balanced',
    description: 'Mix of growth, value, and income investments',
    icon: Scale,
    color: 'bg-orange-50 border-orange-200',
    selectedColor: 'bg-orange-100 border-orange-400',
    characteristics: ['Diversified approach', 'Moderate risk', 'Steady growth'],
    example: 'Index funds, balanced ETFs'
  }
]

export function InvestmentStyleCard({ value, onChange, error }: InvestmentStyleCardProps) {
  return (
    <div className="max-w-5xl mx-auto">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {styleOptions.map((option) => {
          const Icon = option.icon
          const isSelected = value === option.value
          
          return (
            <button
              key={option.value}
              onClick={() => onChange(option.value)}
              className={`
                relative p-6 rounded-lg border-2 text-left transition-all duration-200 hover:shadow-md
                ${isSelected 
                  ? `${option.selectedColor} border-opacity-100` 
                  : `${option.color} border-opacity-50 hover:border-opacity-75`
                }
              `}
            >
              <div className="flex flex-col items-center text-center mb-4">
                <div className={`
                  p-3 rounded-full mb-3
                  ${isSelected ? 'bg-white bg-opacity-50' : 'bg-white bg-opacity-30'}
                `}>
                  <Icon size={24} className="text-gray-700" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900">{option.title}</h3>
              </div>
              
              <p className="text-sm text-gray-600 mb-4">
                {option.description}
              </p>
              
              <div className="space-y-2 mb-4">
                {option.characteristics.map((characteristic, index) => (
                  <div key={index} className="flex items-center text-xs text-gray-600">
                    <div className="w-1 h-1 rounded-full bg-gray-400 mr-2" />
                    {characteristic}
                  </div>
                ))}
              </div>

              <div className="text-xs text-gray-500 italic">
                Examples: {option.example}
              </div>
              
              {isSelected && (
                <div className="absolute top-4 right-4">
                  <div className="w-6 h-6 bg-gray-700 rounded-full flex items-center justify-center">
                    <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                  </div>
                </div>
              )}
            </button>
          )
        })}
      </div>
      
      {error && (
        <div className="mt-4 text-center">
          <p className="text-danger-600 text-sm font-medium">{error}</p>
        </div>
      )}
      
      {value && (
        <div className="mt-6 max-w-3xl mx-auto">
          <div className="p-4 bg-gray-50 rounded-lg">
            <h4 className="font-medium text-gray-900 mb-2">Your Investment Style: {value.charAt(0).toUpperCase() + value.slice(1)}</h4>
            <p className="text-gray-600 text-sm">
              {styleOptions.find(opt => opt.value === value)?.description}
              {' '}This approach will guide our portfolio recommendations to match your investment preferences.
            </p>
          </div>
        </div>
      )}
    </div>
  )
} 