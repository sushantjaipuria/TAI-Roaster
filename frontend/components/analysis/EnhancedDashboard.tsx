import React, { useState, useEffect } from 'react'
import { 
  Activity, 
  TrendingUp, 
  BarChart3, 
  Target, 
  Brain, 
  Zap,
  Eye,
  AlertTriangle,
  CheckCircle,
  Star,
  ArrowRight,
  Sparkles,
  LineChart
} from 'lucide-react'
import EnhancedStockInsightsPanel from './EnhancedStockInsightsPanel'
import PortfolioHealthDashboard from '../results/PortfolioHealthDashboard'
import PerformanceMetrics from '../results/PerformanceMetrics'
import AllocationDashboard from '../results/AllocationDashboard'
import ActionPlan from '../results/ActionPlan'
import RecommendationsPanel from '../results/RecommendationsPanel'
import EnhancedStockInsights from '../results/EnhancedStockInsights'
import StockInsightCards from '../results/StockInsightCards'
import PortfolioSummaryCard from '../results/PortfolioSummaryCard'
import FundamentalInsights from '../results/FundamentalInsights'
import InsightsAndRisksSection from '../results/InsightsAndRisksSection'
import { RiskAnalysisPanel } from './RiskAnalysisPanel'
import { MLPredictionsChart } from './MLPredictionsChart'
import PortfolioGraphs from './PortfolioGraphs'

interface EnhancedDashboardProps {
  data: any
}

// Enhanced Status Banner Component
const EnhancedStatusBanner = ({ data }: { data: any }) => {
  const enhancedStocks = data.enhanced_stocks || []
  const totalStocks = data.stocks?.length || 0
  const enhancedCount = enhancedStocks.filter((stock: any) => stock.enhanced_analysis_available).length
  const coveragePercentage = totalStocks > 0 ? (enhancedCount / totalStocks) * 100 : 0
  
  const hasRealTimeData = enhancedStocks.some((stock: any) => 
    stock.enhanced_analysis_available && stock.current_price && stock.current_price > 0
  )

  const hasAIInsights = enhancedStocks.some((stock: any) => 
    stock.business_story || stock.investment_thesis || (stock.key_strengths && stock.key_strengths.length > 0)
  )

  const hasMLPredictions = data.ml_predictions && data.ml_predictions.length > 0

  return (
    <div className="bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600 text-white mb-6">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 bg-white/20 rounded-full flex items-center justify-center">
              <Sparkles className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold">Enhanced Portfolio Analysis</h1>
              <p className="text-blue-100">AI-powered insights with real-time market data</p>
            </div>
          </div>
          <div className="text-right">
            <div className="text-sm text-blue-100">Analysis Coverage</div>
            <div className="text-2xl font-bold">{coveragePercentage.toFixed(0)}%</div>
            <div className="text-xs text-blue-200">{enhancedCount}/{totalStocks} stocks enhanced</div>
          </div>
        </div>

        {/* Enhanced Features Status Grid */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className={`p-4 rounded-lg ${enhancedCount > 0 ? 'bg-white/20' : 'bg-white/10'} border border-white/20`}>
            <div className="flex items-center gap-2 mb-2">
              <div className={`w-2 h-2 rounded-full ${enhancedCount > 0 ? 'bg-green-400' : 'bg-gray-400'}`}></div>
              <span className="text-sm font-medium">Multi-Factor Analysis</span>
            </div>
            <div className="text-lg font-bold">{enhancedCount} stocks</div>
            <div className="text-xs text-blue-200">6-factor scoring system</div>
          </div>

          <div className={`p-4 rounded-lg ${hasRealTimeData ? 'bg-white/20' : 'bg-white/10'} border border-white/20`}>
            <div className="flex items-center gap-2 mb-2">
              <div className={`w-2 h-2 rounded-full ${hasRealTimeData ? 'bg-green-400 animate-pulse' : 'bg-gray-400'}`}></div>
              <span className="text-sm font-medium">Real-time Data</span>
            </div>
            <div className="text-lg font-bold">{hasRealTimeData ? 'Live' : 'Static'}</div>
            <div className="text-xs text-blue-200">Current market prices</div>
          </div>

          <div className={`p-4 rounded-lg ${hasAIInsights ? 'bg-white/20' : 'bg-white/10'} border border-white/20`}>
            <div className="flex items-center gap-2 mb-2">
              <div className={`w-2 h-2 rounded-full ${hasAIInsights ? 'bg-green-400' : 'bg-gray-400'}`}></div>
              <span className="text-sm font-medium">AI Insights</span>
            </div>
            <div className="text-lg font-bold">{hasAIInsights ? 'Available' : 'Limited'}</div>
            <div className="text-xs text-blue-200">Investment thesis & catalysts</div>
          </div>

          <div className={`p-4 rounded-lg ${hasMLPredictions ? 'bg-white/20' : 'bg-white/10'} border border-white/20`}>
            <div className="flex items-center gap-2 mb-2">
              <div className={`w-2 h-2 rounded-full ${hasMLPredictions ? 'bg-green-400' : 'bg-gray-400'}`}></div>
              <span className="text-sm font-medium">ML Predictions</span>
            </div>
            <div className="text-lg font-bold">{hasMLPredictions ? 'Active' : 'Pending'}</div>
            <div className="text-xs text-blue-200">Ensemble forecasting</div>
          </div>
        </div>
      </div>
    </div>
  )
}

// Enhanced Quick Insights Component
const EnhancedQuickInsights = ({ data }: { data: any }) => {
  const enhancedStocks = data.enhanced_stocks || []
  const portfolioMetrics = data.portfolio_enhanced_metrics || {}
  
  // Calculate metrics from actual enhanced stock data
  const enhancedStocksWithScores = enhancedStocks.filter((stock: any) => 
    stock.enhanced_analysis_available && stock.overall_score > 0
  )
  
  const topPerformer = enhancedStocksWithScores
    .sort((a: any, b: any) => b.overall_score - a.overall_score)[0]

  const totalUpside = portfolioMetrics.total_upside_potential || 
    (enhancedStocksWithScores.length > 0 ? 
      enhancedStocksWithScores.reduce((sum: number, stock: any) => 
        sum + ((stock.upside_potential || 0) * (stock.weight || 0) / 100), 0
      ) : 0)
      
  const highConvictionCount = portfolioMetrics.high_conviction_stocks || 
    enhancedStocksWithScores.filter((stock: any) => 
      stock.confidence_level === 'HIGH' || stock.overall_score >= 80
    ).length
    
  const weightedScore = portfolioMetrics.weighted_overall_score || 
    (enhancedStocksWithScores.length > 0 ? 
      enhancedStocksWithScores.reduce((sum: number, stock: any) => 
        sum + stock.overall_score * (stock.weight || 0) / 100, 0
      ) : (data.rating?.taiScore || 0))

  return (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
      {/* Portfolio Score */}
      <div className="bg-white rounded-xl p-6 border-2 border-gray-200 hover:border-blue-300 transition-colors">
        <div className="flex items-center gap-3 mb-3">
          <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
            <BarChart3 className="w-5 h-5 text-blue-600" />
          </div>
          <div>
            <div className="text-sm text-gray-600">Enhanced Score</div>
            <div className="text-2xl font-bold text-gray-900">{weightedScore.toFixed(0)}/100</div>
          </div>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div 
            className={`h-2 rounded-full ${
              weightedScore >= 70 ? 'bg-green-500' : 
              weightedScore >= 50 ? 'bg-blue-500' : 'bg-red-500'
            }`}
            style={{ width: `${Math.min(weightedScore, 100)}%` }}
          ></div>
        </div>
      </div>

      {/* Top Performer */}
      <div className="bg-white rounded-xl p-6 border-2 border-gray-200 hover:border-green-300 transition-colors">
        <div className="flex items-center gap-3 mb-3">
          <div className="w-10 h-10 bg-green-100 rounded-full flex items-center justify-center">
            <Star className="w-5 h-5 text-green-600" />
          </div>
          <div>
            <div className="text-sm text-gray-600">Top Performer</div>
            <div className="text-lg font-bold text-gray-900">{topPerformer?.ticker || 'Analyzing...'}</div>
          </div>
        </div>
        <div className="text-sm text-gray-600">
          Score: {topPerformer?.overall_score?.toFixed(0) || 'N/A'}/100
        </div>
      </div>

      {/* Upside Potential */}
      <div className="bg-white rounded-xl p-6 border-2 border-gray-200 hover:border-purple-300 transition-colors">
        <div className="flex items-center gap-3 mb-3">
          <div className="w-10 h-10 bg-purple-100 rounded-full flex items-center justify-center">
            <Target className="w-5 h-5 text-purple-600" />
          </div>
          <div>
            <div className="text-sm text-gray-600">Total Upside</div>
            <div className="text-2xl font-bold text-gray-900">{totalUpside.toFixed(1)}%</div>
          </div>
        </div>
        <div className="text-sm text-gray-600">
          Weighted portfolio potential
        </div>
      </div>

      {/* High Conviction */}
      <div className="bg-white rounded-xl p-6 border-2 border-gray-200 hover:border-orange-300 transition-colors">
        <div className="flex items-center gap-3 mb-3">
          <div className="w-10 h-10 bg-orange-100 rounded-full flex items-center justify-center">
            <CheckCircle className="w-5 h-5 text-orange-600" />
          </div>
          <div>
            <div className="text-sm text-gray-600">High Conviction</div>
            <div className="text-2xl font-bold text-gray-900">{highConvictionCount}</div>
          </div>
        </div>
        <div className="text-sm text-gray-600">
          Strong buy signals
        </div>
      </div>
    </div>
  )
}

export default function EnhancedDashboard({ data }: EnhancedDashboardProps) {
  const [activeTab, setActiveTab] = useState('overview')
  const [realTimeUpdate, setRealTimeUpdate] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Handle data loading errors
  useEffect(() => {
    if (!data) {
      setError('Failed to load analysis data')
      return
    }

    if (data.error) {
      setError(data.error)
      return
    }

    setError(null)
  }, [data])

  // Simulate real-time updates (for demo purposes)
  useEffect(() => {
    const interval = setInterval(() => {
      setRealTimeUpdate(prev => !prev)
    }, 30000) // Every 30 seconds

    return () => clearInterval(interval)
  }, [])

  const tabs = [
    {
      id: 'overview',
      name: 'Portfolio Overview',
      icon: Activity
    },
    {
      id: 'enhanced',
      name: 'Enhanced Analysis',
      icon: Brain
    },
    {
      id: 'performance',
      name: 'Performance & ML',
      icon: TrendingUp
    },
    {
      id: 'graphs',
      name: 'Portfolio Graphs',
      icon: BarChart3
    }
  ]

  // Check if this is likely demo data
  const isDemoData = () => {
    if (!data) return true
    
    // Check for demo data indicators
    const indicators = [
      // Check if stocks are demo stocks
      data.stocks?.some((stock: any) => 
        ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK'].includes(stock.ticker)
      ),
      
      // Check if recommendations are generic or empty
      !data.recommendations || data.recommendations.length === 0 || 
      data.recommendations.every((rec: string) => 
        rec.includes('diversify') || rec.includes('consider') || rec.includes('review')
      ),
      
      // Check if overall score is default (0 or 50)
      data.overallScore === 0 || data.overallScore === 50,
      
      // Check if summary is generic
      !data.summary || data.summary.length < 50,
      
      // Check if red flags are empty (demo data often has no red flags)
      !data.redFlags || data.redFlags.length === 0
    ]
    
    return indicators.some(indicator => indicator === true)
  }

  const demoDataWarning = isDemoData()

  const renderTabContent = () => {
    switch (activeTab) {
      case 'overview':
        return (
          <div className="space-y-8">
            <EnhancedQuickInsights data={data} />
            <PortfolioHealthDashboard data={data} />
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <PerformanceMetrics data={data} />
              <AllocationDashboard data={data} />
            </div>
          </div>
        )

      case 'enhanced':
        return (
          <div className="space-y-8">
            <div className="bg-white rounded-xl border border-gray-200 p-6">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center">
                  <Brain className="w-6 h-6 text-purple-600" />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-gray-900">Enhanced Individual Stock Analysis</h2>
                  <p className="text-gray-600">Comprehensive technical and fundamental analysis with multi-factor scoring</p>
                </div>
                {realTimeUpdate && (
                  <div className="ml-auto flex items-center gap-2 text-green-600 text-sm">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                    Live Data
                  </div>
                )}
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-6">
                <div className="text-center">
                  <div className="text-3xl font-bold text-blue-600 mb-1">6</div>
                  <div className="text-sm font-medium text-gray-700 mb-1">Analysis Factors</div>
                  <div className="text-xs text-gray-500">Technical • Fundamental • Momentum • Value • Quality • Sentiment</div>
                </div>
                <div className="text-center">
                  <div className="text-3xl font-bold text-green-600 mb-1">Real-time</div>
                  <div className="text-sm font-medium text-gray-700 mb-1">Market Data</div>
                  <div className="text-xs text-gray-500">Live prices • Financial metrics • Risk indicators</div>
                </div>
                <div className="text-center">
                  <div className="text-3xl font-bold text-purple-600 mb-1">AI-Powered</div>
                  <div className="text-sm font-medium text-gray-700 mb-1">Insights</div>
                  <div className="text-xs text-gray-500">Investment thesis • Catalysts • Risk assessment</div>
                </div>
              </div>
            </div>
            
            <EnhancedStockInsightsPanel 
              enhanced_stocks={data.enhanced_stocks || []}
              portfolio_enhanced_metrics={data.portfolio_enhanced_metrics}
            />
            
            <EnhancedStockInsights data={data} />
          </div>
        )

      case 'performance':
        return (
          <div className="space-y-8">
            <div className="bg-white rounded-xl border border-gray-200 p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
                  <TrendingUp className="w-5 h-5 text-blue-600" />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-gray-900">Performance Analytics & ML Predictions</h2>
                  <p className="text-gray-600">Advanced performance metrics with machine learning forecasts</p>
                </div>
              </div>
            </div>
            <PerformanceMetrics data={data} />
            {data.ml_predictions && data.ml_predictions.length > 0 && (
              <MLPredictionsChart predictions={data.ml_predictions} />
            )}
          </div>
        )

      case 'graphs':
        return <PortfolioGraphs data={data} />

      default:
        return null
    }
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center max-w-md mx-auto">
          <div className="bg-red-100 rounded-full w-16 h-16 mx-auto mb-4 flex items-center justify-center">
            <svg className="w-8 h-8 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.5 0L4.268 16.5c-.77.833.192 2.5 1.732 2.5z" />
            </svg>
          </div>
          <h1 className="text-2xl font-bold text-red-600 mb-4">Analysis Error</h1>
          <p className="text-gray-600">{error}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex justify-between items-start">
            <div className="space-y-2">
              <div className="flex items-center gap-3">
                <h1 className="text-2xl font-bold text-gray-900">
                  Portfolio Analysis Dashboard
                </h1>
                {demoDataWarning && (
                  <div className="flex items-center gap-1 px-2 py-1 bg-yellow-100 rounded-lg">
                    <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
                    <span className="text-xs font-medium text-yellow-800">Demo Data</span>
                  </div>
                )}
              </div>
              <p className="text-gray-600">
                AI-powered insights and real-time portfolio analysis
              </p>
            </div>
            
            {/* Real-time indicator */}
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-2 px-3 py-1 bg-green-100 rounded-lg">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-sm font-medium text-green-800">Live</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Demo Data Notice */}
      {demoDataWarning && (
        <div className="bg-yellow-50 border-b border-yellow-200">
          <div className="max-w-7xl mx-auto px-4 py-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <svg className="w-5 h-5 text-yellow-600 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.5 0L4.268 16.5c-.77.833.192 2.5 1.732 2.5z" />
                </svg>
                <div>
                  <h3 className="text-sm font-medium text-yellow-800">
                    Demo Data Mode
                  </h3>
                  <p className="text-sm text-yellow-700">
                    This analysis is using sample data. For real portfolio analysis, please upload your actual portfolio data.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Tab Navigation */}
        <div className="mb-8">
          <nav className="flex space-x-8">
            {tabs.map((tab) => {
              const Icon = tab.icon
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center space-x-2 py-2 px-1 border-b-2 font-medium text-sm transition-colors ${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{tab.name}</span>
                </button>
              )
            })}
          </nav>
        </div>

        {/* Tab Content */}
        <div className="space-y-8">
          {activeTab === 'overview' && (
            <div className="space-y-8">
              <PortfolioSummaryCard data={data} />
              <AllocationDashboard data={data} />
              <StockInsightCards data={data} />
            </div>
          )}
          
          {activeTab === 'enhanced' && (
            <div className="space-y-8">
              <EnhancedStockInsights data={data} />
              <FundamentalInsights data={data} />
              <InsightsAndRisksSection data={data} />
            </div>
          )}
          
          {activeTab === 'performance' && (
            <div className="space-y-8">
              <PerformanceMetrics data={data} />
              <MLPredictionsChart predictions={data.ml_predictions || []} />
            </div>
          )}
          
          {activeTab === 'graphs' && (
            <div className="space-y-8">
              <PortfolioGraphs data={data} />
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <div className="bg-white border-t border-gray-200 mt-16">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex justify-between items-center">
            <div className="text-sm text-gray-500">
              Analysis generated on {new Date().toLocaleDateString()}
            </div>
            <div className="flex items-center space-x-4">
              <button className="text-sm text-gray-500 hover:text-gray-700">
                Export PDF
              </button>
              <button className="text-sm text-gray-500 hover:text-gray-700">
                Share Analysis
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
} 