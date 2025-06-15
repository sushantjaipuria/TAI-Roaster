'use client'

import React from 'react'
import { Target, Home, GraduationCap, Heart, Car, Plane, PiggyBank, TrendingUp } from 'lucide-react'

interface GoalsCardProps {
  value: string[]
  onChange: (value: string[]) => void
  error?: string
}

const goalOptions = [
  {
    id: 'retirement',
    title: 'Retirement',
    description: 'Long-term retirement planning',
    icon: PiggyBank,
    color: 'text-blue-600 bg-blue-50 border-blue-200'
  },
  {
    id: 'wealth-building',
    title: 'Wealth Building',
    description: 'Long-term wealth accumulation',
    icon: TrendingUp,
    color: 'text-green-600 bg-green-50 border-green-200'
  },
  {
    id: 'home-purchase',
    title: 'Home Purchase',
    description: 'Saving for a house or real estate',
    icon: Home,
    color: 'text-purple-600 bg-purple-50 border-purple-200'
  },
  {
    id: 'education',
    title: 'Education',
    description: 'College or education funding',
    icon: GraduationCap,
    color: 'text-indigo-600 bg-indigo-50 border-indigo-200'
  },
  {
    id: 'emergency-fund',
    title: 'Emergency Fund',
    description: 'Financial safety net',
    icon: Heart,
    color: 'text-red-600 bg-red-50 border-red-200'
  },
  {
    id: 'major-purchase',
    title: 'Major Purchase',
    description: 'Car, boat, or other big expenses',
    icon: Car,
    color: 'text-orange-600 bg-orange-50 border-orange-200'
  },
  {
    id: 'travel',
    title: 'Travel & Leisure',
    description: 'Vacation and travel funds',
    icon: Plane,
    color: 'text-cyan-600 bg-cyan-50 border-cyan-200'
  },
  {
    id: 'general-investing',
    title: 'General Investing',
    description: 'Growing money without specific goals',
    icon: Target,
    color: 'text-gray-600 bg-gray-50 border-gray-200'
  }
]

export function GoalsCard({ value, onChange, error }: GoalsCardProps) {
  const toggleGoal = (goalId: string) => {
    if (value.includes(goalId)) {
      onChange(value.filter(id => id !== goalId))
    } else {
      onChange([...value, goalId])
    }
  }

  return (
    <div className="max-w-5xl mx-auto">
      <div className="mb-6 text-center">
        <p className="text-gray-600">Select all that apply. You can choose multiple goals.</p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {goalOptions.map((goal) => {
          const Icon = goal.icon
          const isSelected = value.includes(goal.id)
          
          return (
            <button
              key={goal.id}
              onClick={() => toggleGoal(goal.id)}
              className={`
                relative p-4 rounded-lg border-2 text-left transition-all duration-200 hover:shadow-md
                ${isSelected 
                  ? `${goal.color} border-opacity-100 shadow-sm` 
                  : 'bg-white border-gray-200 hover:border-gray-300'
                }
              `}
            >
              <div className="flex flex-col items-center text-center">
                <div className={`
                  p-3 rounded-full mb-3 transition-colors
                  ${isSelected 
                    ? 'bg-white bg-opacity-50' 
                    : 'bg-gray-100'
                  }
                `}>
                  <Icon 
                    size={24} 
                    className={isSelected ? goal.color.split(' ')[0] : 'text-gray-600'} 
                  />
                </div>
                
                <h3 className={`
                  text-sm font-semibold mb-1
                  ${isSelected ? goal.color.split(' ')[0] : 'text-gray-900'}
                `}>
                  {goal.title}
                </h3>
                
                <p className={`
                  text-xs leading-tight
                  ${isSelected ? 'text-gray-700' : 'text-gray-500'}
                `}>
                  {goal.description}
                </p>
              </div>
              
              {isSelected && (
                <div className="absolute top-2 right-2">
                  <div className={`
                    w-5 h-5 rounded-full flex items-center justify-center
                    ${goal.color.split(' ')[0]} bg-current
                  `}>
                    <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
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
      
      {value.length > 0 && (
        <div className="mt-6 max-w-3xl mx-auto">
          <div className="p-4 bg-gray-50 rounded-lg">
            <h4 className="font-medium text-gray-900 mb-2">
              Your Selected Goals ({value.length})
            </h4>
            <div className="flex flex-wrap gap-2">
              {value.map(goalId => {
                const goal = goalOptions.find(g => g.id === goalId)
                return goal ? (
                  <span 
                    key={goalId}
                    className="inline-flex items-center px-3 py-1 rounded-full text-sm bg-primary-100 text-primary-800"
                  >
                    {goal.title}
                  </span>
                ) : null
              })}
            </div>
            <p className="text-gray-600 text-sm mt-2">
              These goals will help us tailor your portfolio to match your investment objectives and timeline.
            </p>
          </div>
        </div>
      )}
    </div>
  )
} 