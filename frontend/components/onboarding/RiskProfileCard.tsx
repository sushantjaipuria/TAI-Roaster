'use client'

import React from 'react'
import { Shield, TrendingUp, Zap } from 'lucide-react'

interface RiskProfileCardProps {
  value?: 'conservative' | 'moderate' | 'aggressive'
  onChange: (value: 'conservative' | 'moderate' | 'aggressive') => void
  error?: string
}

const riskOptions = [
  {
    value: 'conservative' as const,
    title: 'Conservative',
    description: 'Prefer stable, low-risk investments with steady returns',
    icon: Shield,
    color: 'bg-success-50 border-success-200 text-success-700',
    selectedColor: 'bg-success-100 border-success-400',
    characteristics: ['Lower volatility', 'Steady income', 'Capital preservation']
  },
  {
    value: 'moderate' as const,
    title: 'Moderate',
    description: 'Balanced approach with moderate risk for potential growth',
    icon: TrendingUp,
    color: 'bg-warning-50 border-warning-200 text-warning-700',
    selectedColor: 'bg-warning-100 border-warning-400',
    characteristics: ['Balanced risk/reward', 'Diversified portfolio', 'Moderate growth']
  },
  {
    value: 'aggressive' as const,
    title: 'Aggressive',
    description: 'Higher risk tolerance for potentially higher returns',
    icon: Zap,
    color: 'bg-danger-50 border-danger-200 text-danger-700',
    selectedColor: 'bg-danger-100 border-danger-400',
    characteristics: ['Higher volatility', 'Growth potential', 'Long-term focus']
  }
]

export function RiskProfileCard({ value, onChange, error }: RiskProfileCardProps) {
  return (
    <div className="max-w-4xl mx-auto">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {riskOptions.map((option) => {
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
              <div className="flex items-center mb-4">
                <div className={`
                  p-3 rounded-full mr-4
                  ${isSelected ? 'bg-white bg-opacity-50' : 'bg-white bg-opacity-30'}
                `}>
                  <Icon size={24} />
                </div>
                <h3 className="text-xl font-semibold">{option.title}</h3>
              </div>
              
              <p className="text-sm mb-4 opacity-80">
                {option.description}
              </p>
              
              <div className="space-y-2">
                {option.characteristics.map((characteristic, index) => (
                  <div key={index} className="flex items-center text-sm opacity-70">
                    <div className="w-1.5 h-1.5 rounded-full bg-current mr-3" />
                    {characteristic}
                  </div>
                ))}
              </div>
              
              {isSelected && (
                <div className="absolute top-4 right-4">
                  <div className="w-6 h-6 bg-current rounded-full flex items-center justify-center">
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
        <div className="mt-6 p-4 bg-gray-50 rounded-lg">
          <h4 className="font-medium text-gray-900 mb-2">Your Selection:</h4>
          <p className="text-gray-600 text-sm">
            You've selected a <strong className="text-gray-900">{value}</strong> risk profile. 
            This means your portfolio will be tailored to match your risk tolerance and investment timeline.
          </p>
        </div>
      )}
    </div>
  )
} 