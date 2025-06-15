import React from 'react'
import Link from 'next/link'

export default function HomePage() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-primary-50 to-primary-100">
      <div className="max-w-4xl mx-auto px-4 text-center">
        <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
          TAI Roaster
        </h1>
        <p className="text-xl md:text-2xl text-gray-600 mb-8 max-w-2xl mx-auto">
          Intelligent portfolio analysis and personalized investment recommendations
        </p>
        <p className="text-lg text-gray-500 mb-12 max-w-xl mx-auto">
          Upload your portfolio, get AI-powered insights, and discover how to optimize your investments for better returns.
        </p>
        
        <div className="space-y-4 sm:space-y-0 sm:space-x-4 sm:flex sm:justify-center">
          <Link 
            href="/onboarding" 
            className="btn-primary text-lg px-8 py-3 w-full sm:w-auto"
          >
            Start Analysis
          </Link>
          <Link 
            href="/learn-more" 
            className="btn-secondary text-lg px-8 py-3 w-full sm:w-auto"
          >
            Learn More
          </Link>
        </div>
        
        <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">Risk Assessment</h3>
            </div>
            <div className="card-content">
              Analyze your portfolio's risk profile and get personalized recommendations
            </div>
          </div>
          
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">Portfolio Optimization</h3>
            </div>
            <div className="card-content">
              Discover allocation improvements and investment opportunities
            </div>
          </div>
          
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">AI Insights</h3>
            </div>
            <div className="card-content">
              Get detailed explanations for every recommendation and insight
            </div>
          </div>
        </div>
      </div>
    </div>
  )
} 