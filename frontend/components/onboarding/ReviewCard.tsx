'use client'

import React from 'react'
import { Edit3, Shield, DollarSign, TrendingUp, Clock, Target } from 'lucide-react'
import { UserProfile } from '@/lib/types'
import { formatCurrency } from '@/lib/utils'

interface ReviewCardProps {
  userProfile: UserProfile
  onEdit: (step: number) => void
}

export function ReviewCard({ userProfile, onEdit }: ReviewCardProps) {
  const getRiskIcon = (risk: string) => {
    switch (risk) {
      case 'conservative': return Shield
      case 'moderate': return TrendingUp
      case 'aggressive': return TrendingUp
      default: return Shield
    }
  }

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'conservative': return 'text-green-600 bg-green-50'
      case 'moderate': return 'text-yellow-600 bg-yellow-50'
      case 'aggressive': return 'text-red-600 bg-red-50'
      default: return 'text-gray-600 bg-gray-50'
    }
  }

  const getTimeHorizonLabel = (horizon: string) => {
    switch (horizon) {
      case 'short': return 'Short-term (1-3 years)'
      case 'medium': return 'Medium-term (3-10 years)'
      case 'long': return 'Long-term (10+ years)'
      default: return horizon
    }
  }

  const goalLabels: Record<string, string> = {
    'retirement': 'Retirement',
    'wealth-building': 'Wealth Building',
    'home-purchase': 'Home Purchase',
    'education': 'Education',
    'emergency-fund': 'Emergency Fund',
    'major-purchase': 'Major Purchase',
    'travel': 'Travel & Leisure',
    'general-investing': 'General Investing'
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="space-y-6">
        {/* Risk Tolerance */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center">
              <div className={`p-2 rounded-lg mr-3 ${getRiskColor(userProfile.riskTolerance)}`}>
                {React.createElement(getRiskIcon(userProfile.riskTolerance), { size: 20 })}
              </div>
              <div>
                <h3 className="font-semibold text-gray-900">Risk Tolerance</h3>
                <p className="text-sm text-gray-600">Your comfort level with investment risk</p>
              </div>
            </div>
            <button
              onClick={() => onEdit(0)}
              className="text-primary-600 hover:text-primary-700 flex items-center text-sm"
            >
              <Edit3 size={16} className="mr-1" />
              Edit
            </button>
          </div>
          <div className={`inline-flex items-center px-3 py-2 rounded-full text-sm font-medium ${getRiskColor(userProfile.riskTolerance)}`}>
            {userProfile.riskTolerance.charAt(0).toUpperCase() + userProfile.riskTolerance.slice(1)}
          </div>
        </div>

        {/* Investment Amount */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center">
              <div className="p-2 rounded-lg mr-3 bg-primary-50 text-primary-600">
                <DollarSign size={20} />
              </div>
              <div>
                <h3 className="font-semibold text-gray-900">Investment Amount</h3>
                <p className="text-sm text-gray-600">Your initial investment</p>
              </div>
            </div>
            <button
              onClick={() => onEdit(1)}
              className="text-primary-600 hover:text-primary-700 flex items-center text-sm"
            >
              <Edit3 size={16} className="mr-1" />
              Edit
            </button>
          </div>
          <div className="text-2xl font-bold text-gray-900">
            {formatCurrency(userProfile.investmentAmount)}
          </div>
        </div>

        {/* Investment Style */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center">
              <div className="p-2 rounded-lg mr-3 bg-purple-50 text-purple-600">
                <TrendingUp size={20} />
              </div>
              <div>
                <h3 className="font-semibold text-gray-900">Investment Style</h3>
                <p className="text-sm text-gray-600">Your preferred investment approach</p>
              </div>
            </div>
            <button
              onClick={() => onEdit(2)}
              className="text-primary-600 hover:text-primary-700 flex items-center text-sm"
            >
              <Edit3 size={16} className="mr-1" />
              Edit
            </button>
          </div>
          <div className="inline-flex items-center px-3 py-2 rounded-full text-sm font-medium bg-purple-50 text-purple-700">
            {userProfile.investmentStyle.charAt(0).toUpperCase() + userProfile.investmentStyle.slice(1)}
          </div>
        </div>

        {/* Time Horizon */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center">
              <div className="p-2 rounded-lg mr-3 bg-blue-50 text-blue-600">
                <Clock size={20} />
              </div>
              <div>
                <h3 className="font-semibold text-gray-900">Time Horizon</h3>
                <p className="text-sm text-gray-600">Your investment timeline</p>
              </div>
            </div>
            <button
              onClick={() => onEdit(3)}
              className="text-primary-600 hover:text-primary-700 flex items-center text-sm"
            >
              <Edit3 size={16} className="mr-1" />
              Edit
            </button>
          </div>
          <div className="inline-flex items-center px-3 py-2 rounded-full text-sm font-medium bg-blue-50 text-blue-700">
            {getTimeHorizonLabel(userProfile.timeHorizon)}
          </div>
        </div>

        {/* Investment Goals */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center">
              <div className="p-2 rounded-lg mr-3 bg-orange-50 text-orange-600">
                <Target size={20} />
              </div>
              <div>
                <h3 className="font-semibold text-gray-900">Investment Goals</h3>
                <p className="text-sm text-gray-600">What you're investing for</p>
              </div>
            </div>
            <button
              onClick={() => onEdit(4)}
              className="text-primary-600 hover:text-primary-700 flex items-center text-sm"
            >
              <Edit3 size={16} className="mr-1" />
              Edit
            </button>
          </div>
          <div className="flex flex-wrap gap-2">
            {userProfile.goals.map((goalId, index) => (
              <span 
                key={index}
                className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-orange-50 text-orange-700"
              >
                {goalLabels[goalId] || goalId}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Summary */}
      <div className="mt-8 p-6 bg-gradient-to-r from-primary-50 to-primary-100 rounded-lg">
        <h3 className="text-lg font-semibold text-gray-900 mb-3">Ready to Continue?</h3>
        <p className="text-gray-700 mb-4">
          Based on your preferences, we'll create a personalized portfolio recommendation. 
          You can upload your current portfolio next to get detailed analysis and suggestions.
        </p>
        <div className="text-sm text-gray-600">
          <p>• <strong>Risk Level:</strong> {userProfile.riskTolerance}</p>
          <p>• <strong>Investment:</strong> {formatCurrency(userProfile.investmentAmount)}</p>
          <p>• <strong>Style:</strong> {userProfile.investmentStyle}</p>
          <p>• <strong>Timeline:</strong> {getTimeHorizonLabel(userProfile.timeHorizon)}</p>
          <p>• <strong>Goals:</strong> {userProfile.goals.length} selected</p>
        </div>
      </div>
    </div>
  )
} 