'use client'

import React, { useState } from 'react'
import { useRouter } from 'next/navigation'
import { OnboardingFlow } from '@/components/onboarding/OnboardingFlow'
import { UserProfile } from '@/lib/types'
import { ApiClient } from '@/lib/api'
import { storage } from '@/lib/utils'

export default function OnboardingPage() {
  const router = useRouter()
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleComplete = async (userProfile: UserProfile) => {
    setIsSubmitting(true)
    setError(null)

    try {
      // Submit onboarding data to backend
      const response = await ApiClient.submitOnboarding(userProfile)
      
      if (response.success) {
        // Store session ID locally
        storage.set('sessionId', response.session_id)
        storage.set('userProfile', userProfile)
        
        // Redirect to portfolio upload
        router.push('/upload')
      } else {
        setError(response.message || 'Failed to save profile')
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 to-primary-100">
      <div className="container mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
            Let's Get to Know You
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Help us understand your investment preferences so we can provide personalized portfolio recommendations.
          </p>
        </div>

        {error && (
          <div className="max-w-2xl mx-auto mb-6">
            <div className="bg-danger-50 border border-danger-200 text-danger-700 px-4 py-3 rounded-md">
              <p className="font-medium">Error</p>
              <p className="text-sm">{error}</p>
            </div>
          </div>
        )}

        <OnboardingFlow 
          onComplete={handleComplete}
          isSubmitting={isSubmitting}
        />
      </div>
    </div>
  )
} 