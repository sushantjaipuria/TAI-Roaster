import React, { useState } from 'react'
import { 
  Star, 
  AlertTriangle, 
  TrendingUp, 
  Shield, 
  PieChart, 
  Target,
  CheckCircle,
  XCircle,
  Eye,
  Activity
} from 'lucide-react'
import { ResultsComponentProps } from '../../lib/types-results'

// Insight Card Component
const InsightCard = ({ 
  type, 
  title, 
  description, 
  impact, 
  severity = 'medium',
  actionable = false
}: {
  type: 'strength' | 'risk' | 'opportunity' | 'hygiene'
  title: string
  description: string
  impact?: string
  severity?: 'low' | 'medium' | 'high'
  actionable?: boolean
}) => {
  const typeConfig = {
    strength: {
      icon: <CheckCircle className="w-5 h-5" />,
      bgColor: 'bg-green-50',
      borderColor: 'border-green-200',
      textColor: 'text-green-800',
      iconColor: 'text-green-600'
    },
    risk: {
      icon: <AlertTriangle className="w-5 h-5" />,
      bgColor: 'bg-red-50',
      borderColor: 'border-red-200',
      textColor: 'text-red-800',
      iconColor: 'text-red-600'
    },
    opportunity: {
      icon: <TrendingUp className="w-5 h-5" />,
      bgColor: 'bg-blue-50',
      borderColor: 'border-blue-200',
      textColor: 'text-blue-800',
      iconColor: 'text-blue-600'
    },
    hygiene: {
      icon: <Shield className="w-5 h-5" />,
      bgColor: 'bg-purple-50',
      borderColor: 'border-purple-200',
      textColor: 'text-purple-800',
      iconColor: 'text-purple-600'
    }
  }

  const config = typeConfig[type]
  
  const severityIndicator = severity === 'high' ? '🔴' : severity === 'medium' ? '🟡' : '🟢'

  return (
    <div className={`p-4 rounded-lg border-2 ${config.bgColor} ${config.borderColor} transition-all duration-200 hover:shadow-md`}>
      <div className="flex items-start gap-3">
        <div className={config.iconColor}>
          {config.icon}
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-2">
            <h4 className={`font-semibold ${config.textColor}`}>{title}</h4>
            {type === 'risk' && <span>{severityIndicator}</span>}
            {actionable && <span className="text-xs bg-white px-2 py-1 rounded-full">Actionable</span>}
          </div>
          <p className={`text-sm ${config.textColor} opacity-90 mb-2`}>{description}</p>
          {impact && (
            <p className={`text-xs ${config.textColor} opacity-75 italic`}>Impact: {impact}</p>
          )}
        </div>
      </div>
    </div>
  )
}

// Portfolio Strengths Section
const PortfolioStrengthsCard = ({ data }: { data: any }) => {
  const strengths = []

  // Generate insights based on data
  if (data.rating.taiScore >= 75) {
    strengths.push({
      title: "High Portfolio Score",
      description: `Your TAI Score of ${data.rating.taiScore} indicates excellent portfolio management.`,
      impact: "Strong foundation for long-term wealth creation"
    })
  }

  if (data.diversificationScore >= 80) {
    strengths.push({
      title: "Excellent Diversification",
      description: `Diversification score of ${Math.round(data.diversificationScore)} shows optimal risk spreading.`,
      impact: "Reduced portfolio volatility and better risk management"
    })
  }

  if (data.rating.liquidityScore >= 75) {
    strengths.push({
      title: "High Liquidity",
      description: `Liquidity score of ${data.rating.liquidityScore} ensures easy access to funds.`,
      impact: "Flexibility for rebalancing and emergency needs"
    })
  }

  // Check latest performance
  const latestPerformance = data.performanceMetrics[data.performanceMetrics.length - 1]
  if (latestPerformance.outperformance > 0) {
    strengths.push({
      title: "Benchmark Outperformance",
      description: `Your portfolio is outperforming ${data.benchmarkName} by ${latestPerformance.outperformance.toFixed(1)}%.`,
      impact: "Superior risk-adjusted returns"
    })
  }

  if (latestPerformance.metrics.sharpeRatio >= 1) {
    strengths.push({
      title: "Strong Risk-Adjusted Returns",
      description: `Sharpe ratio of ${latestPerformance.metrics.sharpeRatio.toFixed(2)} indicates efficient risk utilization.`,
      impact: "Optimal balance between risk and return"
    })
  }

  return (
    <div className="bg-white rounded-xl border-2 border-gray-200 p-6">
      <div className="flex items-center gap-3 mb-6">
        <Star className="w-6 h-6 text-green-600" />
        <h3 className="text-xl font-semibold text-gray-900">Portfolio Strengths</h3>
        <span className="bg-green-100 text-green-800 text-sm px-2 py-1 rounded-full">{strengths.length}</span>
      </div>

      {strengths.length > 0 ? (
        <div className="space-y-3">
          {strengths.map((strength, index) => (
            <InsightCard
              key={index}
              type="strength"
              title={strength.title}
              description={strength.description}
              impact={strength.impact}
            />
          ))}
        </div>
      ) : (
        <div className="text-center py-8 text-gray-500">
          <Target className="w-12 h-12 mx-auto mb-3 opacity-50" />
          <p>Continue building your portfolio to unlock more strengths</p>
        </div>
      )}
    </div>
  )
}

// Portfolio Risks Section
const PortfolioRisksCard = ({ data }: { data: any }) => {
  const risks = []

  // Concentration risk
  if (data.allocation.concentration.topHoldingsPct > 50) {
    risks.push({
      title: "High Concentration Risk",
      description: `${data.allocation.concentration.topHoldingsPct.toFixed(0)}% of your portfolio is in top holdings.`,
      impact: "Reduced diversification increases volatility",
      severity: 'high' as const,
      actionable: true
    })
  }

  // Risk management score
  if (data.rating.riskManagement < 50) {
    risks.push({
      title: "Risk Management Concerns",
      description: `Risk management score of ${data.rating.riskManagement} suggests review needed.`,
      impact: "Potential for higher than expected losses",
      severity: 'high' as const,
      actionable: true
    })
  }

  // Sector concentration
  data.allocation.concentration.sectorConcentration.forEach((sector: any) => {
    if (sector.isOverweight && sector.allocation > 30) {
      risks.push({
        title: `${sector.sector} Overexposure`,
        description: `${sector.allocation.toFixed(1)}% allocation vs ${sector.benchmark}% benchmark.`,
        impact: "Sector-specific risks could impact performance",
        severity: 'medium' as const,
        actionable: true
      })
    }
  })

  // Volatility risk
  const latestPerformance = data.performanceMetrics[data.performanceMetrics.length - 1]
  if (latestPerformance.metrics.volatility > 25) {
    risks.push({
      title: "High Volatility",
      description: `Annual volatility of ${latestPerformance.metrics.volatility.toFixed(1)}% indicates high price swings.`,
      impact: "Increased short-term uncertainty in returns",
      severity: 'medium' as const
    })
  }

  // Drawdown risk
  if (Math.abs(latestPerformance.metrics.maxDrawdown) > 15) {
    risks.push({
      title: "Significant Drawdown Risk",
      description: `Maximum drawdown of ${latestPerformance.metrics.maxDrawdown.toFixed(1)}% shows potential for large losses.`,
      impact: "Higher probability of substantial temporary losses",
      severity: 'medium' as const
    })
  }

  // Add concentration flags
  data.allocation.concentration.riskFlags.forEach((flag: any) => {
    risks.push({
      title: flag.type.replace('_', ' ').toLowerCase().replace(/\b\w/g, (l: string) => l.toUpperCase()),
      description: flag.message,
      impact: "Increased portfolio risk exposure",
      severity: flag.severity.toLowerCase() as 'low' | 'medium' | 'high',
      actionable: true
    })
  })

  return (
    <div className="bg-white rounded-xl border-2 border-gray-200 p-6">
      <div className="flex items-center gap-3 mb-6">
        <AlertTriangle className="w-6 h-6 text-red-600" />
        <h3 className="text-xl font-semibold text-gray-900">Risk Assessment</h3>
        <span className="bg-red-100 text-red-800 text-sm px-2 py-1 rounded-full">{risks.length}</span>
      </div>

      {risks.length > 0 ? (
        <div className="space-y-3">
          {risks.map((risk, index) => (
            <InsightCard
              key={index}
              type="risk"
              title={risk.title}
              description={risk.description}
              impact={risk.impact}
              severity={risk.severity}
              actionable={risk.actionable}
            />
          ))}
        </div>
      ) : (
        <div className="text-center py-8 text-green-600">
          <Shield className="w-12 h-12 mx-auto mb-3" />
          <p className="font-medium">No significant risks detected</p>
          <p className="text-sm text-gray-600 mt-1">Your portfolio shows good risk management</p>
        </div>
      )}
    </div>
  )
}

// Portfolio Hygiene Section
const PortfolioHygieneCard = ({ data }: { data: any }) => {
  const hygieneItems = []

  // Check hygiene factors
  if (data.hygiene.pennyStocks.count > 0) {
    hygieneItems.push({
      title: "Penny Stocks Detected",
      description: `${data.hygiene.pennyStocks.count} penny stocks found: ${data.hygiene.pennyStocks.tickers.join(', ')}.`,
      impact: data.hygiene.pennyStocks.impact
    })
  } else {
    hygieneItems.push({
      title: "No Penny Stocks",
      description: "Portfolio is free from high-risk penny stocks.",
      impact: "Reduced speculative risk exposure"
    })
  }

  if (data.hygiene.excessiveCash.isExcessive) {
    hygieneItems.push({
      title: "Excessive Cash Holdings",
      description: `${data.hygiene.excessiveCash.percentage.toFixed(1)}% cash allocation may be too high.`,
      impact: "Potential opportunity cost from uninvested cash"
    })
  } else {
    hygieneItems.push({
      title: "Optimal Cash Allocation",
      description: data.hygiene.excessiveCash.suggestion,
      impact: "Efficient capital deployment"
    })
  }

  if (data.hygiene.smallCapOverexposure.isExcessive) {
    hygieneItems.push({
      title: "Small Cap Overexposure",
      description: `${data.hygiene.smallCapOverexposure.percentage}% allocation exceeds ${data.hygiene.smallCapOverexposure.threshold}% threshold.`,
      impact: "Increased volatility and liquidity risk"
    })
  }

  if (data.hygiene.lowLiquidityStocks.count > 0) {
    hygieneItems.push({
      title: "Low Liquidity Stocks",
      description: `${data.hygiene.lowLiquidityStocks.count} stocks with low liquidity detected.`,
      impact: data.hygiene.lowLiquidityStocks.impact
    })
  } else {
    hygieneItems.push({
      title: "Good Liquidity Profile",
      description: data.hygiene.lowLiquidityStocks.impact,
      impact: "Easy entry and exit from positions"
    })
  }

  return (
    <div className="bg-white rounded-xl border-2 border-gray-200 p-6">
      <div className="flex items-center gap-3 mb-6">
        <Activity className="w-6 h-6 text-purple-600" />
        <h3 className="text-xl font-semibold text-gray-900">Portfolio Hygiene</h3>
      </div>

      <div className="space-y-3">
        {hygieneItems.map((item, index) => (
          <InsightCard
            key={index}
            type="hygiene"
            title={item.title}
            description={item.description}
            impact={item.impact}
          />
        ))}
      </div>
    </div>
  )
}

// Opportunities Section  
const OpportunitiesCard = ({ data }: { data: any }) => {
  const opportunities = []

  // Generate opportunities based on analysis
  if (data.rating.returnQuality < 75) {
    opportunities.push({
      title: "Enhance Return Quality",
      description: `Return quality score of ${data.rating.returnQuality} has room for improvement.`,
      impact: "Potential for better risk-adjusted returns"
    })
  }

  if (data.rating.costEfficiency < 80) {
    opportunities.push({
      title: "Optimize Costs",
      description: `Cost efficiency score of ${data.rating.costEfficiency} suggests expense reduction opportunities.`,
      impact: "Higher net returns through lower fees"
    })
  }

  // Sector underweight opportunities
  data.allocation.concentration.sectorConcentration.forEach((sector: any) => {
    if (!sector.isOverweight && sector.allocation < sector.benchmark * 0.5) {
      opportunities.push({
        title: `Consider ${sector.sector} Exposure`,
        description: `Currently ${sector.allocation.toFixed(1)}% vs ${sector.benchmark}% benchmark allocation.`,
        impact: "Potential diversification benefits"
      })
    }
  })

  // Generic opportunities if we have high-priority improvements
  data.actionPlan.improvements
    .filter((imp: any) => imp.priority === 'HIGH')
    .slice(0, 2)
    .forEach((improvement: any) => {
      opportunities.push({
        title: improvement.area,
        description: improvement.suggestion,
        impact: improvement.expectedBenefit
      })
    })

  return (
    <div className="bg-white rounded-xl border-2 border-gray-200 p-6">
      <div className="flex items-center gap-3 mb-6">
        <Eye className="w-6 h-6 text-blue-600" />
        <h3 className="text-xl font-semibold text-gray-900">Growth Opportunities</h3>
        <span className="bg-blue-100 text-blue-800 text-sm px-2 py-1 rounded-full">{opportunities.length}</span>
      </div>

      {opportunities.length > 0 ? (
        <div className="space-y-3">
          {opportunities.map((opportunity, index) => (
            <InsightCard
              key={index}
              type="opportunity"
              title={opportunity.title}
              description={opportunity.description}
              impact={opportunity.impact}
              actionable={true}
            />
          ))}
        </div>
      ) : (
        <div className="text-center py-8 text-gray-500">
          <CheckCircle className="w-12 h-12 mx-auto mb-3 text-green-500" />
          <p className="font-medium text-green-600">Portfolio Well Optimized</p>
          <p className="text-sm text-gray-600 mt-1">No immediate optimization opportunities identified</p>
        </div>
      )}
    </div>
  )
}

export default function InsightsAndRisksSection({ data }: ResultsComponentProps) {
  const [activeTab, setActiveTab] = useState<'strengths' | 'risks' | 'hygiene' | 'opportunities'>('strengths')

  const tabs = [
    { id: 'strengths', label: 'Strengths', icon: <Star className="w-4 h-4" /> },
    { id: 'risks', label: 'Risks', icon: <AlertTriangle className="w-4 h-4" /> },
    { id: 'hygiene', label: 'Hygiene', icon: <Activity className="w-4 h-4" /> },
    { id: 'opportunities', label: 'Opportunities', icon: <Eye className="w-4 h-4" /> }
  ]

  return (
    <div className="bg-white py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="text-center mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Portfolio Insights & Analysis</h2>
          <p className="text-gray-600">Comprehensive assessment of your portfolio's health and opportunities</p>
        </div>

        {/* Tab Navigation */}
        <div className="flex flex-wrap justify-center gap-2 mb-8">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-all duration-200 ${
                activeTab === tab.id
                  ? 'bg-blue-600 text-white shadow-lg'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        <div className="max-w-4xl mx-auto">
          {activeTab === 'strengths' && <PortfolioStrengthsCard data={data} />}
          {activeTab === 'risks' && <PortfolioRisksCard data={data} />}
          {activeTab === 'hygiene' && <PortfolioHygieneCard data={data} />}
          {activeTab === 'opportunities' && <OpportunitiesCard data={data} />}
        </div>
      </div>
    </div>
  )
} 