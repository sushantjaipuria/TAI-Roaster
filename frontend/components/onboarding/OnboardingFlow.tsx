'use client'

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronLeft, ChevronRight, Check } from 'lucide-react'
import { UserProfile } from '@/lib/types'
import { RiskProfileCard } from './RiskProfileCard'
import { InvestmentSizeCard } from './InvestmentSizeCard'
import { InvestmentStyleCard } from './InvestmentStyleCard'
import { TimeHorizonCard } from './TimeHorizonCard'
import { GoalsCard } from './GoalsCard'
import { ReviewCard } from './ReviewCard'

interface OnboardingFlowProps {
  onComplete: (userProfile: UserProfile) => void
  isSubmitting?: boolean
}

const steps = [
  { id: 'risk', title: 'Risk Tolerance', description: 'How comfortable are you with investment risk?' },
  { id: 'amount', title: 'Investment Amount', description: 'How much are you planning to invest?' },
  { id: 'style', title: 'Investment Style', description: 'What\'s your preferred investment approach?' },
  { id: 'timeHorizon', title: 'Time Horizon', description: 'How long do you plan to invest?' },
  { id: 'goals', title: 'Investment Goals', description: 'What are you hoping to achieve?' },
  { id: 'review', title: 'Review', description: 'Review and confirm your preferences' },
]

export function OnboardingFlow({ onComplete, isSubmitting = false }: OnboardingFlowProps) {
  const [currentStep, setCurrentStep] = useState(0)
  const [userProfile, setUserProfile] = useState<Partial<UserProfile>>({})
  const [errors, setErrors] = useState<Record<string, string>>({})

  const updateProfile = (updates: Partial<UserProfile>) => {
    setUserProfile(prev => ({ ...prev, ...updates }))
    // Clear related errors
    Object.keys(updates).forEach(key => {
      if (errors[key]) {
        setErrors(prev => {
          const newErrors = { ...prev }
          delete newErrors[key]
          return newErrors
        })
      }
    })
  }

  const validateStep = (stepIndex: number): boolean => {
    const step = steps[stepIndex]
    const newErrors: Record<string, string> = {}

    switch (step.id) {
      case 'risk':
        if (!userProfile.riskTolerance) {
          newErrors.riskTolerance = 'Please select your risk tolerance'
        }
        break
      case 'amount':
        if (!userProfile.investmentAmount || userProfile.investmentAmount <= 0) {
          newErrors.investmentAmount = 'Please enter a valid investment amount'
        }
        break
      case 'style':
        if (!userProfile.investmentStyle) {
          newErrors.investmentStyle = 'Please select your investment style'
        }
        break
      case 'timeHorizon':
        if (!userProfile.timeHorizon) {
          newErrors.timeHorizon = 'Please select your time horizon'
        }
        break
      case 'goals':
        if (!userProfile.goals || userProfile.goals.length === 0) {
          newErrors.goals = 'Please select at least one investment goal'
        }
        break
    }

    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const handleNext = () => {
    if (validateStep(currentStep)) {
      if (currentStep < steps.length - 1) {
        setCurrentStep(currentStep + 1)
      } else {
        // Final step - submit
        handleSubmit()
      }
    }
  }

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1)
    }
  }

  const handleSubmit = () => {
    if (validateStep(currentStep - 1) && userProfile as UserProfile) {
      onComplete(userProfile as UserProfile)
    }
  }

  const renderStepContent = () => {
    const step = steps[currentStep]
    
    switch (step.id) {
      case 'risk':
        return (
          <RiskProfileCard
            value={userProfile.riskTolerance}
            onChange={(value) => updateProfile({ riskTolerance: value })}
            error={errors.riskTolerance}
          />
        )
      case 'amount':
        return (
          <InvestmentSizeCard
            value={userProfile.investmentAmount}
            onChange={(value) => updateProfile({ investmentAmount: value })}
            error={errors.investmentAmount}
          />
        )
      case 'style':
        return (
          <InvestmentStyleCard
            value={userProfile.investmentStyle}
            onChange={(value) => updateProfile({ investmentStyle: value })}
            error={errors.investmentStyle}
          />
        )
      case 'timeHorizon':
        return (
          <TimeHorizonCard
            value={userProfile.timeHorizon}
            onChange={(value) => updateProfile({ timeHorizon: value })}
            error={errors.timeHorizon}
          />
        )
      case 'goals':
        return (
          <GoalsCard
            value={userProfile.goals || []}
            onChange={(value) => updateProfile({ goals: value })}
            error={errors.goals}
          />
        )
      case 'review':
        return (
          <ReviewCard
            userProfile={userProfile as UserProfile}
            onEdit={(step) => setCurrentStep(step)}
          />
        )
      default:
        return null
    }
  }

  const progress = ((currentStep + 1) / steps.length) * 100

  return (
    <div className="max-w-4xl mx-auto">
      {/* Progress Bar */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-700">
            Step {currentStep + 1} of {steps.length}
          </span>
          <span className="text-sm font-medium text-gray-700">
            {Math.round(progress)}% Complete
          </span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className="bg-primary-600 h-2 rounded-full transition-all duration-300 ease-out"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* Step Indicators */}
      <div className="flex items-center justify-center mb-8 space-x-4">
        {steps.map((step, index) => (
          <div key={step.id} className="flex items-center">
            <div
              className={`
                w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium transition-colors
                ${index < currentStep 
                  ? 'bg-primary-600 text-white' 
                  : index === currentStep 
                    ? 'bg-primary-100 text-primary-600 border-2 border-primary-600'
                    : 'bg-gray-200 text-gray-500'
                }
              `}
            >
              {index < currentStep ? <Check size={16} /> : index + 1}
            </div>
            {index < steps.length - 1 && (
              <div 
                className={`w-8 h-0.5 mx-2 ${
                  index < currentStep ? 'bg-primary-600' : 'bg-gray-200'
                }`} 
              />
            )}
          </div>
        ))}
      </div>

      {/* Step Content */}
      <div className="mb-8">
        <div className="text-center mb-6">
          <h2 className="text-2xl font-bold text-gray-900 mb-2">
            {steps[currentStep].title}
          </h2>
          <p className="text-gray-600">
            {steps[currentStep].description}
          </p>
        </div>

        <AnimatePresence mode="wait">
          <motion.div
            key={currentStep}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.3 }}
          >
            {renderStepContent()}
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Navigation */}
      <div className="flex items-center justify-between">
        <button
          onClick={handlePrevious}
          disabled={currentStep === 0}
          className="btn-secondary flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <ChevronLeft size={16} />
          <span>Previous</span>
        </button>

        <button
          onClick={handleNext}
          disabled={isSubmitting}
          className="btn-primary flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <span>
            {isSubmitting 
              ? 'Submitting...' 
              : currentStep === steps.length - 1 
                ? 'Complete Setup' 
                : 'Next'
            }
          </span>
          {!isSubmitting && <ChevronRight size={16} />}
        </button>
      </div>
    </div>
  )
} 