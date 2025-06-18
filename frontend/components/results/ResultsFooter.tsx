import React from 'react'
import { format } from 'date-fns'

export default function ResultsFooter() {
  return (
    <footer className="bg-gray-50 border-t border-gray-200 mt-16">
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Disclaimer */}
          <div>
            <h3 className="text-sm font-semibold text-gray-900 mb-3">Important Disclaimer</h3>
            <div className="text-xs text-gray-600 space-y-2">
              <p>
                This analysis is for informational purposes only and should not be considered as investment advice. 
                Past performance does not guarantee future results.
              </p>
              <p>
                All investments carry risks including the potential loss of principal. Please consult with a qualified 
                financial advisor before making investment decisions.
              </p>
              <p>
                Market data and analysis are based on available information and may not reflect real-time market conditions.
              </p>
            </div>
          </div>

          {/* Branding & Info */}
          <div className="text-right">
            <div className="text-sm font-semibold text-gray-900 mb-3">
              TAI Roaster Portfolio Analysis
            </div>
            <div className="text-xs text-gray-600 space-y-1">
              <p>Powered by Advanced AI & Machine Learning</p>
              <p>Generated on {format(new Date(), 'MMMM d, yyyy \'at\' h:mm a')}</p>
              <p className="mt-4 text-gray-500">
                © 2024 TAI Roaster. All rights reserved.
              </p>
            </div>
          </div>
        </div>
        
        {/* Bottom Bar */}
        <div className="mt-8 pt-4 border-t border-gray-200">
          <div className="flex flex-col sm:flex-row justify-between items-center gap-4 text-xs text-gray-500">
            <div>
              Analysis validity: 30 days from generation date
            </div>
            <div className="flex items-center gap-4">
              <span>Version 1.0</span>
              <span>•</span>
              <span>Real-time market data</span>
              <span>•</span>
              <span>AI-powered insights</span>
            </div>
          </div>
        </div>
      </div>
    </footer>
  )
} 