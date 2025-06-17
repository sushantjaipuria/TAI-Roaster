/**
 * User Profile Form Component
 * 
 * Handles user profile and investment preferences
 */

'use client'

import React, { useState } from 'react'
import { UserProfile } from '../lib/types'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { User, DollarSign } from 'lucide-react'

interface UserProfileFormProps {
  userProfile: UserProfile
  onChange: (profile: UserProfile) => void
  showAdvanced?: boolean
}

const UserProfileForm: React.FC<UserProfileFormProps> = ({
  userProfile,
  onChange,
  showAdvanced = false
}) => {
  const [errors, setErrors] = useState<{ [key: string]: string }>({})
  
  const handleFieldChange = (field: keyof UserProfile, value: any) => {
    onChange({
      ...userProfile,
      [field]: value
    })
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <User className="h-5 w-5" />
          Investment Profile
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Investment Amount */}
        <div className="space-y-2">
          <Label htmlFor="investmentAmount" className="flex items-center gap-2">
            <DollarSign className="h-4 w-4" />
            Investment Amount (₹) *
          </Label>
          <Input
            id="investmentAmount"
            type="number"
            value={userProfile.investmentAmount || ''}
            onChange={(e) => handleFieldChange('investmentAmount', parseFloat(e.target.value) || 0)}
            placeholder="Total amount to invest"
          />
          <p className="text-xs text-gray-500">
            Enter the total amount you plan to invest across all holdings
          </p>
        </div>

        {/* Risk Tolerance */}
        <div className="space-y-3">
          <Label className="text-base font-medium">Risk Tolerance *</Label>
          <div className="grid grid-cols-1 gap-3">
            {['conservative', 'moderate', 'aggressive'].map(option => (
              <div
                key={option}
                className={`
                  p-4 border rounded-lg cursor-pointer transition-all
                  ${userProfile.riskTolerance === option 
                    ? 'border-blue-500 bg-blue-50 ring-2 ring-blue-200' 
                    : 'border-gray-200 hover:border-gray-300'
                  }
                `}
                onClick={() => handleFieldChange('riskTolerance', option)}
              >
                <span className="font-medium capitalize">{option}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Investment Style */}
        <div className="space-y-3">
          <Label className="text-base font-medium">Investment Style *</Label>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {['growth', 'value', 'income', 'balanced'].map(option => (
              <div
                key={option}
                className={`
                  p-3 border rounded-lg cursor-pointer transition-all
                  ${userProfile.investmentStyle === option 
                    ? 'border-blue-500 bg-blue-50 ring-1 ring-blue-200' 
                    : 'border-gray-200 hover:border-gray-300'
                  }
                `}
                onClick={() => handleFieldChange('investmentStyle', option)}
              >
                <span className="font-medium capitalize">{option}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Time Horizon */}
        <div className="space-y-3">
          <Label className="text-base font-medium">Investment Time Horizon *</Label>
          <div className="grid grid-cols-1 gap-2">
            {['short', 'medium', 'long'].map(option => (
              <label
                key={option}
                className={`
                  flex items-center p-3 border rounded-lg cursor-pointer transition-all
                  ${userProfile.timeHorizon === option 
                    ? 'border-blue-500 bg-blue-50' 
                    : 'border-gray-200 hover:border-gray-300'
                  }
                `}
              >
                <input
                  type="radio"
                  name="timeHorizon"
                  value={option}
                  checked={userProfile.timeHorizon === option}
                  onChange={(e) => handleFieldChange('timeHorizon', e.target.value)}
                  className="mr-3"
                />
                <span className="font-medium capitalize">{option} Term</span>
              </label>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export default UserProfileForm 