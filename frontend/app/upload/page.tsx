'use client'

import React, { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { UserProfile } from '@/lib/types'
import { storage } from '@/lib/utils'

export default function UploadPage() {
  const router = useRouter()
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  // Load user profile from localStorage on mount
  useEffect(() => {
    const savedProfile = storage.get('userProfile')
    const sessionId = storage.get('sessionId')
    
    if (!savedProfile || !sessionId) {
      // If no profile found, redirect back to onboarding
      router.push('/onboarding')
      return
    }
    
    setUserProfile(savedProfile)
    setIsLoading(false)
  }, [router])

  // Show loading while checking for user profile
  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-primary-50 to-primary-100 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading...</p>
        </div>
      </div>
    )
  }

  // Show error if no user profile found
  if (!userProfile) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-primary-50 to-primary-100 flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-600">Redirecting to onboarding...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 to-primary-100">
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto p-6 space-y-8">
          <div className="text-center">
            <h1 className="text-3xl font-bold text-gray-900 mb-2">
              Portfolio Input
            </h1>
            <p className="text-gray-600">
              Welcome {userProfile.riskTolerance} investor! The portfolio input form will be available here.
            </p>
          </div>
          
          <div className="bg-white rounded-lg p-6 shadow-sm">
            <h2 className="text-xl font-semibold mb-4">Your Profile</h2>
            <div className="space-y-2">
              <p><strong>Risk Tolerance:</strong> {userProfile.riskTolerance}</p>
              <p><strong>Investment Style:</strong> {userProfile.investmentStyle}</p>
              <p><strong>Time Horizon:</strong> {userProfile.timeHorizon}</p>
              <p><strong>Investment Amount:</strong> ₹{userProfile.investmentAmount?.toLocaleString('en-IN')}</p>
              {userProfile.goals && userProfile.goals.length > 0 && (
                <p><strong>Goals:</strong> {userProfile.goals.join(', ')}</p>
              )}
            </div>
          </div>
          
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-blue-900 mb-2">
              🎉 Success! Upload Page is Working
            </h3>
            <p className="text-blue-800">
              The missing /upload route has been created successfully. Your onboarding data is being loaded correctly.
            </p>
            <p className="text-blue-700 text-sm mt-2">
              The full InputForm component will be integrated next once import issues are resolved.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
} 