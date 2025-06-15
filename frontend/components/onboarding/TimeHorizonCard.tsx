'use client'

import React from 'react'
import { Clock, Calendar, CalendarRange } from 'lucide-react'

interface TimeHorizonCardProps {
  value?: 'short' | 'medium' | 'long'
  onChange: (value: 'short' | 'medium' | 'long') => void
  error?: string
}

const timeHorizonOptions = [
  {
    value: 'short' as const,
    title: 'Short-term',
    subtitle: '1-3 years',
    description: 'Planning to use the money relatively soon',
    icon: Clock,
    color: 'bg-red-50 border-red-200',
    selectedColor: 'bg-red-100 border-red-400',
    characteristics: ['Higher liquidity', 'Lower risk tolerance', 'Conservative approach'],
    considerations: 'Good for emergency funds or near-term goals'
  },
  {
    value: 'medium' as const,
    title: 'Medium-term',
    subtitle: '3-10 years',
    description: 'Balanced timeframe for moderate growth',
    icon: Calendar,
    color: 'bg-yellow-50 border-yellow-200',
    selectedColor: 'bg-yellow-100 border-yellow-400',
    characteristics: ['Balanced approach', 'Moderate risk', 'Steady growth'],
    considerations: 'Great for retirement planning or major purchases'
  },
  {
    value: 'long' as const,
    title: 'Long-term',
    subtitle: '10+ years',
    description: 'Long-term wealth building and growth',
    icon: CalendarRange,
    color: 'bg-green-50 border-green-200',
    selectedColor: 'bg-green-100 border-green-400',
    characteristics: ['Higher growth potential', 'Can weather volatility', 'Compound growth'],
    considerations: 'Ideal for retirement and long-term wealth building'
  }
]

export function TimeHorizonCard({ value, onChange, error }: TimeHorizonCardProps) {
  return (
    <div className="max-w-4xl mx-auto">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {timeHorizonOptions.map((option) => {
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
              <div className="text-center mb-4">
                <div className={`
                  inline-flex items-center justify-center w-16 h-16 rounded-full mb-3
                  ${isSelected ? 'bg-white bg-opacity-50' : 'bg-white bg-opacity-30'}
                `}>
                  <Icon size={28} className="text-gray-700" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900">{option.title}</h3>
                <p className="text-lg font-medium text-gray-600 mb-2">{option.subtitle}</p>
              </div>
              
              <p className="text-sm text-gray-600 mb-4 text-center">
                {option.description}
              </p>
              
              <div className="space-y-2 mb-4">
                {option.characteristics.map((characteristic, index) => (
                  <div key={index} className="flex items-center text-sm text-gray-600">
                    <div className="w-1.5 h-1.5 rounded-full bg-gray-400 mr-3" />
                    {characteristic}
                  </div>
                ))}
              </div>

              <div className="text-xs text-gray-500 italic border-t border-gray-200 pt-3">
                {option.considerations}
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
            <h4 className="font-medium text-gray-900 mb-2">
              Your Time Horizon: {timeHorizonOptions.find(opt => opt.value === value)?.title}
            </h4>
            <p className="text-gray-600 text-sm">
              With a {value}-term investment horizon, we'll design a portfolio that aligns with your timeline and helps you achieve your financial goals within your preferred timeframe.
            </p>
          </div>
        </div>
      )}
    </div>
  )
} 